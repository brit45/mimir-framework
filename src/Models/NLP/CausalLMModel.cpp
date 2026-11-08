#include "CausalLMModel.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace
{
    size_t checkedMul(size_t a, size_t b)
    {
        if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
        {
            throw std::overflow_error("CausalLMModel: parameter count overflow");
        }
        return a * b;
    }

    int defaultCausalLatentDim(int model_dim)
    {
        if (model_dim < 2)
            return 0;
        return std::max(1, std::min(128, model_dim / 2));
    }

    const std::vector<std::string> &causalControlTokens()
    {
        static const std::vector<std::string> tokens = {
            "<SYSTEM>", "<USER>", "<CONTEXT>", "</CONTEXT>",
            "<REFLECT>", "<ASSISTANT>", "<FINAL>"};
        return tokens;
    }

    bool hasCausalControlTokens(const Tokenizer &tokenizer)
    {
        return std::all_of(causalControlTokens().begin(), causalControlTokens().end(),
                           [&](const std::string &token)
                           { return tokenizer.hasToken(token) && tokenizer.isSpecialToken(token); });
    }

    void replaceAll(std::string &text, const std::string &from, const std::string &to)
    {
        if (from.empty())
            return;
        size_t position = 0;
        while ((position = text.find(from, position)) != std::string::npos)
        {
            text.replace(position, from.size(), to);
            position += to.size();
        }
    }

    void markQuotedPassage(std::string &text)
    {
        const std::vector<std::string> openings = {"Passage : «", "Passage: «", "Passage «"};
        for (const auto &opening : openings)
        {
            const size_t position = text.find(opening);
            if (position == std::string::npos)
                continue;
            text.replace(position, opening.size(), "<CONTEXT>");
            const size_t closing = text.find("»", position + std::string("<CONTEXT>").size());
            if (closing != std::string::npos)
                text.replace(closing, std::string("»").size(), "</CONTEXT>");
            return;
        }
    }

    std::string formatCausalPrompt(std::string text, bool use_control_tokens)
    {
        if (!use_control_tokens)
            return text;

        for (const auto &labels : {
                 std::pair<std::string, std::string>{"Système:", "<SYSTEM>"},
                 {"Système :", "<SYSTEM>"}, {"System:", "<SYSTEM>"},
                 {"Utilisateur:", "<USER>"}, {"Utilisateur :", "<USER>"},
                 {"User:", "<USER>"}, {"Assistante:", "<ASSISTANT>"},
                 {"Assistante :", "<ASSISTANT>"}, {"Assistant:", "<ASSISTANT>"}})
            replaceAll(text, labels.first, labels.second);
        markQuotedPassage(text);

        const std::string reflection_guidance =
            "Lis les balises de rôle et le contexte. Examine l'intention, les contraintes et les ambiguïtés; "
            "vérifie la cohérence et distingue les faits fournis des incertitudes. La réponse finale suit <FINAL>.\n";
        const size_t system_position = text.find("<SYSTEM>");
        if (system_position == std::string::npos)
            text = "<SYSTEM>" + reflection_guidance + text;
        else
            text.insert(system_position + std::string("<SYSTEM>").size(), reflection_guidance);

        const size_t assistant_position = text.rfind("<ASSISTANT>");
        if (assistant_position != std::string::npos)
        {
            const size_t preceding = text.rfind("<REFLECT>", assistant_position);
            if (preceding == std::string::npos)
                text.insert(assistant_position, "<REFLECT>\n");
        }
        return text;
    }

    std::vector<int> removeResponseMarkers(std::vector<int> ids, const Tokenizer &tokenizer)
    {
        if (!ids.empty() && tokenizer.getTokenById(ids.front()) == "<FINAL>")
            ids.erase(ids.begin());
        const int eos_id = tokenizer.getEosId();
        while (!ids.empty() && ids.back() == eos_id)
            ids.pop_back();
        return ids;
    }

    void configureLinear(Model &model, const std::string &name, const std::string &input,
                         const std::string &output, int in_features, int out_features,
                         int seq_len)
    {
        model.push(name, "Linear", checkedMul(static_cast<size_t>(in_features), static_cast<size_t>(out_features)));
        auto *layer = model.getLayerByName(name);
        layer->inputs = {input};
        layer->output = output;
        layer->in_features = in_features;
        layer->out_features = out_features;
        layer->seq_len = seq_len;
        layer->use_bias = false;
    }

    std::string trimText(std::string text)
    {
        const auto not_space = [](unsigned char value)
        { return !std::isspace(value); };
        text.erase(text.begin(), std::find_if(text.begin(), text.end(), not_space));
        text.erase(std::find_if(text.rbegin(), text.rend(), not_space).base(), text.end());
        return text;
    }

    std::string joinUnique(const std::vector<std::string> &values, size_t limit)
    {
        std::vector<std::string> unique;
        std::unordered_set<std::string> seen;
        for (const auto &value : values)
        {
            if (value.empty() || !seen.insert(value).second)
                continue;
            unique.push_back(value);
            if (unique.size() >= limit)
                break;
        }
        std::ostringstream out;
        for (size_t index = 0; index < unique.size(); ++index)
        {
            if (index)
                out << ", ";
            out << unique[index];
        }
        return unique.empty() ? std::string("aucun") : out.str();
    }

    std::string findAssistantMarker(const std::string &text, size_t *position = nullptr)
    {
        std::string lowered = text;
        std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char value)
                       { return static_cast<char>(std::tolower(value)); });
        const std::vector<std::string> markers = {"assistante:", "assistant:", "<assistant>"};
        size_t best = std::string::npos;
        std::string found;
        for (const auto &marker : markers)
        {
            const size_t candidate = lowered.rfind(marker);
            if (candidate != std::string::npos && (best == std::string::npos || candidate > best))
            {
                best = candidate;
                found = text.substr(candidate, marker.size());
            }
        }
        if (position)
            *position = best;
        return found;
    }

    bool isWordLikeToken(const std::string &token)
    {
        for (unsigned char value : token)
        {
            if (std::isalnum(value) || value >= 128)
                return true;
        }
        return false;
    }

    void appendTokenSequence(std::vector<int> &tokens, std::vector<uint8_t> &target_mask,
                             const std::vector<int> &sequence, bool target)
    {
        for (int token_id : sequence)
        {
            tokens.push_back(token_id);
            target_mask.push_back(target ? 1u : 0u);
        }
    }

    std::vector<int> removeBoundaryTokens(std::vector<int> ids, int bos_id, int eos_id)
    {
        if (!ids.empty() && ids.front() == bos_id)
            ids.erase(ids.begin());
        if (!ids.empty() && ids.back() == eos_id)
            ids.pop_back();
        return ids;
    }
} // namespace

CausalLMModel::CausalLMModel()
{
    setModelName("CausalLMModel");
    setHasEncoder(false);
}

void CausalLMModel::buildFromConfig(const Config &cfg)
{
    cfg_ = cfg;
    if (cfg_.latent_dim < 0)
        cfg_.latent_dim = defaultCausalLatentDim(cfg_.d_model);
    buildInto(*this, cfg_);
}

std::string CausalLMModel::decorateContext(const std::string &prompt) const
{
    const Tokenizer::TextUnderstanding analysis = getTokenizer().analyzeText(prompt);
    const auto keywords = getTokenizer().extractKeywords(prompt, 8);
    const auto frequencies = getTokenizer().getTokenFrequencies(prompt);

    std::vector<std::pair<std::string, int>> frequent(frequencies.begin(), frequencies.end());
    std::sort(frequent.begin(), frequent.end(), [](const auto &left, const auto &right)
              {
        if (left.second != right.second) return left.second > right.second;
        return left.first < right.first; });

    std::vector<std::pair<std::string, float>> sentiments(
        analysis.sentiment.begin(), analysis.sentiment.end());
    std::sort(sentiments.begin(), sentiments.end(), [](const auto &left, const auto &right)
              { return left.first < right.first; });

    std::ostringstream features;
    features << "Repères du tokenizer: sujet="
             << (analysis.mainSubject.empty() ? "aucun" : analysis.mainSubject)
             << "; entités=" << joinUnique(analysis.entities, 8)
             << "; actions=" << joinUnique(analysis.actions, 8)
             << "; modificateurs=" << joinUnique(analysis.modifiers, 8)
             << "; contexte=" << (analysis.context.empty() ? "général" : analysis.context)
             << "; complexité=" << analysis.complexity << "; mots-clés="
             << joinUnique(keywords, 8) << "; fréquences=";
    if (frequent.empty())
    {
        features << "aucune";
    }
    else
    {
        const size_t count = std::min<size_t>(6, frequent.size());
        for (size_t index = 0; index < count; ++index)
        {
            if (index)
                features << ", ";
            features << frequent[index].first << "=" << frequent[index].second;
        }
    }
    features << "; sentiment=";
    if (sentiments.empty())
    {
        features << "neutre";
    }
    else
    {
        const size_t count = std::min<size_t>(8, sentiments.size());
        for (size_t index = 0; index < count; ++index)
        {
            if (index)
                features << ", ";
            features << sentiments[index].first << "=" << std::fixed << std::setprecision(2)
                     << sentiments[index].second;
        }
    }
    features << "; relations=";
    if (analysis.relations.empty())
    {
        features << "aucune";
    }
    else
    {
        const size_t count = std::min<size_t>(8, analysis.relations.size());
        for (size_t index = 0; index < count; ++index)
        {
            if (index)
                features << ", ";
            features << analysis.relations[index].first << "→" << analysis.relations[index].second;
        }
    }

    size_t marker_position = std::string::npos;
    const std::string marker = findAssistantMarker(prompt, &marker_position);
    if (marker_position == std::string::npos)
        return prompt + "\n" + features.str();
    return prompt.substr(0, marker_position) + "\n" + features.str() + "\n" +
           prompt.substr(marker_position);
}

std::vector<int> CausalLMModel::tokenizeContent(const std::string &text) const
{
    const Tokenizer &tokenizer = getTokenizer();
    std::vector<int> ids = removeBoundaryTokens(tokenizer.tokenizeBPE(text),
                                                tokenizer.getBosId(), tokenizer.getEosId());
    const int unk_id = tokenizer.getUnkId();
    if (unk_id >= 0 && std::find(ids.begin(), ids.end(), unk_id) != ids.end())
    {
        throw std::runtime_error(
            "CausalLMModel: tokenizer produced <UNK>; the corpus/BPE vocabulary does not cover this text");
    }
    return ids;
}

void CausalLMModel::prepareTrainingCorpus(const std::string &corpus,
                                          size_t vocab_capacity,
                                          int bpe_merges,
                                          const std::string &tokenizer_path,
                                          int seed,
                                          bool persist_tokenizer)
{
    if (vocab_capacity < 16)
    {
        throw std::invalid_argument("CausalLMModel: tokenizer capacity must be at least 16");
    }
    if (bpe_merges <= 0)
    {
        throw std::invalid_argument("CausalLMModel: BPE merge count must be positive");
    }
    if (!layer_weight_blocks.empty())
    {
        throw std::logic_error("CausalLMModel: prepare the corpus before allocating parameters");
    }
    if (corpus.empty())
        throw std::invalid_argument("CausalLMModel: training corpus is empty");

    const bool strict_resume = modelConfig.value("resume", false);

    const bool validation_requested = modelConfig.value("validate_every_steps", 100) > 0 &&
        modelConfig.value("validate_items", 8) > 0;
    const bool validation_holdout = validation_requested &&
        modelConfig.value("validate_holdout", true);
    const int configured_holdout_items = std::max(0, modelConfig.value("validate_holdout_items", 0));
    const double configured_holdout_fraction = modelConfig.value("validate_holdout_frac", 0.1);
    if (!std::isfinite(configured_holdout_fraction) ||
        configured_holdout_fraction < 0.0 || configured_holdout_fraction >= 1.0)
    {
        throw std::invalid_argument(
            "CausalLMModel: validate_holdout_frac must be finite and in [0, 1)");
    }

    Tokenizer prepared_tokenizer(vocab_capacity);
    bool loaded_tokenizer = false;
    if (!tokenizer_path.empty())
    {
        std::error_code path_error;
        const bool tokenizer_exists = std::filesystem::exists(tokenizer_path, path_error);
        if (path_error)
        {
            throw std::runtime_error("CausalLMModel: cannot inspect tokenizer " +
                                     tokenizer_path + ": " + path_error.message());
        }
        if (tokenizer_exists)
        {
            std::ifstream input(tokenizer_path, std::ios::binary);
            if (!input)
            {
                throw std::runtime_error("CausalLMModel: cannot read tokenizer: " + tokenizer_path);
            }
            json serialized;
            try
            {
                input >> serialized;
            }
            catch (const std::exception &error)
            {
                throw std::runtime_error("CausalLMModel: invalid tokenizer JSON " +
                                         tokenizer_path + ": " + error.what());
            }
            if (!serialized.is_object() ||
                (!serialized.contains("vocab") && !serialized.contains("id2token")))
            {
                throw std::runtime_error("CausalLMModel: tokenizer JSON has no vocabulary: " +
                                         tokenizer_path);
            }
            prepared_tokenizer.from_json(serialized);
            if (prepared_tokenizer.getVocabSize() < 7)
            {
                throw std::runtime_error("CausalLMModel: tokenizer has no usable vocabulary: " +
                                         tokenizer_path);
            }
            if (prepared_tokenizer.getVocabSize() > vocab_capacity)
            {
                if (strict_resume)
                {
                    throw std::runtime_error(
                        "CausalLMModel: checkpoint tokenizer exceeds the configured vocabulary capacity");
                }
                std::cerr << "[causal_lm] tokenizer existant ignoré: vocab="
                          << prepared_tokenizer.getVocabSize() << " > capacité="
                          << vocab_capacity << "; recomposition depuis le corpus.\n";
            }
            else
            {
                const int special_ids[] = {
                    prepared_tokenizer.getPadId(), prepared_tokenizer.getUnkId(),
                    prepared_tokenizer.getSeqId(), prepared_tokenizer.getModId(),
                    prepared_tokenizer.getMagId(), prepared_tokenizer.getBosId(),
                    prepared_tokenizer.getEosId()};
                for (const int id : special_ids)
                {
                    if (id < 0 || static_cast<size_t>(id) >= prepared_tokenizer.getVocabSize())
                    {
                        throw std::runtime_error("CausalLMModel: tokenizer has an invalid special-token id: " +
                                                 tokenizer_path);
                    }
                }
                if (prepared_tokenizer.getPadId() != cfg_.padding_idx)
                {
                    throw std::runtime_error("CausalLMModel: tokenizer PAD id differs from model padding_idx");
                }
                loaded_tokenizer = true;
            }
        }
    }
    if (strict_resume && !loaded_tokenizer)
    {
        throw std::runtime_error(
            "CausalLMModel: resume requires the tokenizer saved with the checkpoint");
    }
    if (!loaded_tokenizer)
        prepared_tokenizer = Tokenizer(vocab_capacity);
    prepared_tokenizer.setMaxVocab(vocab_capacity);
    prepared_tokenizer.setMaxSequenceLength(cfg_.seq_len);
    const size_t loaded_vocab_size = prepared_tokenizer.getVocabSize();
    setTokenizer(prepared_tokenizer);
    Tokenizer &tokenizer = getMutableTokenizer();
    const size_t before_control_tokens = tokenizer.getVocabSize();
    for (const auto &token : causalControlTokens())
        tokenizer.addSpecialToken(token);
    if (strict_resume && tokenizer.getVocabSize() != before_control_tokens)
    {
        throw std::runtime_error(
            "CausalLMModel: checkpoint tokenizer is missing required causal control tokens");
    }
    if (!hasCausalControlTokens(tokenizer))
    {
        throw std::runtime_error(
            "CausalLMModel: vocabulary capacity cannot hold the required dialogue control tokens");
    }

    struct SampleText
    {
        std::string prompt;
        std::string answer;
    };
    std::vector<SampleText> samples;
    std::string paragraph;
    auto consume_paragraph = [&](const std::string &text)
    {
        const std::string cleaned = trimText(text);
        if (cleaned.empty())
            return;
        size_t marker_position = std::string::npos;
        const std::string marker = findAssistantMarker(cleaned, &marker_position);
        if (marker_position == std::string::npos)
            return;
        const std::string prefix = trimText(cleaned.substr(0, marker_position));
        const std::string answer = trimText(cleaned.substr(marker_position + marker.size()));
        if (prefix.empty() || answer.empty())
            return;
        samples.push_back({prefix + "\n" + marker, answer});
    };
    std::istringstream corpus_lines(corpus);
    std::string line;
    while (std::getline(corpus_lines, line))
    {
        if (trimText(line).empty())
        {
            consume_paragraph(paragraph);
            paragraph.clear();
            continue;
        }
        if (!paragraph.empty())
            paragraph.push_back('\n');
        paragraph += line;
    }
    consume_paragraph(paragraph);

    std::vector<std::string> bpe_corpus;
    bpe_corpus.push_back(corpus);
    std::vector<std::string> decorated_prompts;
    decorated_prompts.reserve(samples.size());
    for (const auto &sample : samples)
    {
        decorated_prompts.push_back(
            formatCausalPrompt(decorateContext(sample.prompt), true));
        bpe_corpus.push_back(decorated_prompts.back());
    }

    // Preserve punctuation and control symbols, while letting BPE own ordinary
    // words and subwords. This keeps the output vocabulary trainable end-to-end.
    for (const auto &text : bpe_corpus)
    {
        if (strict_resume) break;
        for (const auto &raw_token : tokenizer.splitTokens(text))
        {
            const std::string token = tokenizer.normalizeToken(raw_token);
            if (!token.empty() && !isWordLikeToken(token))
                tokenizer.addToken(token);
        }
    }
    bool needs_bpe_update = !loaded_tokenizer;
    if (loaded_tokenizer)
    {
        const int unk_id = tokenizer.getUnkId();
        for (const auto &text : bpe_corpus)
        {
            const auto ids = tokenizer.tokenizeBPE(text);
            if (std::find(ids.begin(), ids.end(), unk_id) != ids.end())
            {
                needs_bpe_update = true;
                break;
            }
        }
    }
    if (strict_resume && needs_bpe_update)
    {
        throw std::runtime_error(
            "CausalLMModel: checkpoint tokenizer cannot encode this corpus without <UNK>; "
            "resume requires the original corpus and frozen BPE vocabulary");
    }
    if (needs_bpe_update)
        tokenizer.learnBPEFromCorpus(bpe_corpus, bpe_merges);
    if (strict_resume && tokenizer.getVocabSize() != loaded_vocab_size)
    {
        throw std::runtime_error(
            "CausalLMModel: tokenizer vocabulary changed while preparing a resumed run");
    }
    tokenizer.setMaxSequenceLength(cfg_.seq_len);

    training_tokens_.clear();
    training_target_mask_.clear();
    training_target_positions_.clear();
    validation_sequences_.clear();
    const int bos_id = tokenizer.getBosId();
    const int eos_id = tokenizer.getEosId();
    const int unk_id = tokenizer.getUnkId();
    auto encode_content = [&](const std::string &text)
    {
        std::vector<int> ids = tokenizeContent(text);
        for (int id : ids)
        {
            if (id == unk_id)
            {
                throw std::runtime_error(
                    "CausalLMModel: tokenizer BPE left <UNK> in the prepared corpus; "
                    "increase vocab_size or add the missing character coverage");
            }
        }
        return ids;
    };
    auto append_training_segment = [&](const std::vector<int> &ids, bool target)
    {
        appendTokenSequence(training_tokens_, training_target_mask_, ids, target);
    };
    size_t training_sample_count = 0;
    size_t heldout_sample_count = 0;
    bool token_split_used = false;

    if (samples.empty())
    {
        // A plain-text corpus has no dialogue/sample boundaries. Reserve its
        // final token span, retaining the preceding text as causal context.
        const std::vector<int> content = encode_content(
            "<CONTEXT>" + corpus + "</CONTEXT><FINAL>");
        const bool split_tokens = validation_holdout && content.size() >= 2;
        size_t train_content_end = content.size();
        size_t validation_target_begin = 1;
        if (split_tokens)
        {
            size_t heldout = static_cast<size_t>(std::ceil(
                static_cast<double>(content.size()) * configured_holdout_fraction));
            heldout = std::clamp<size_t>(heldout, 1, content.size() - 1);
            train_content_end = content.size() - heldout;
            validation_target_begin = 1 + train_content_end;
        }

        std::vector<int> train_sequence;
        train_sequence.reserve(train_content_end + 2);
        train_sequence.push_back(bos_id);
        train_sequence.insert(train_sequence.end(), content.begin(),
                              content.begin() + static_cast<std::ptrdiff_t>(train_content_end));
        train_sequence.push_back(eos_id);
        training_tokens_.insert(training_tokens_.end(), train_sequence.begin(), train_sequence.end());
        training_target_mask_.insert(training_target_mask_.end(), train_sequence.size(), 0u);
        for (size_t position = 1; position <= train_content_end; ++position)
            training_target_mask_[training_target_mask_.size() - train_sequence.size() + position] = 1u;
        // When a token holdout is present, do not teach EOS immediately after
        // the training prefix: the real continuation belongs to validation.
        if (!split_tokens)
            training_target_mask_.back() = 1u;

        ValidationSequence validation;
        validation.tokens.reserve(content.size() + 2);
        validation.tokens.push_back(bos_id);
        validation.tokens.insert(validation.tokens.end(), content.begin(), content.end());
        validation.tokens.push_back(eos_id);
        validation.target_begin = validation_holdout ? validation_target_begin : 1;
        validation_sequences_.push_back(std::move(validation));
        training_sample_count = 1;
        token_split_used = split_tokens;
    }
    else
    {
        std::vector<uint8_t> is_validation(samples.size(), 0u);
        if (validation_holdout && samples.size() >= 2)
        {
            std::vector<size_t> indices(samples.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::mt19937 split_rng(static_cast<uint32_t>(seed));
            std::shuffle(indices.begin(), indices.end(), split_rng);
            heldout_sample_count = configured_holdout_items > 0
                ? static_cast<size_t>(configured_holdout_items)
                : static_cast<size_t>(std::ceil(
                    static_cast<double>(samples.size()) * configured_holdout_fraction));
            heldout_sample_count = std::clamp<size_t>(heldout_sample_count, 1, samples.size() - 1);
            for (size_t index = 0; index < heldout_sample_count; ++index)
                is_validation[indices[index]] = 1u;
        }

        for (size_t index = 0; index < samples.size(); ++index)
        {
            const std::vector<int> prompt_ids = encode_content(decorated_prompts[index]);
            const std::vector<int> completion_ids = encode_content(
                "<FINAL> " + samples[index].answer);
            std::vector<int> full_sequence;
            full_sequence.reserve(1 + prompt_ids.size() + completion_ids.size() + 1);
            full_sequence.push_back(bos_id);
            full_sequence.insert(full_sequence.end(), prompt_ids.begin(), prompt_ids.end());
            const size_t completion_begin = full_sequence.size();
            full_sequence.insert(full_sequence.end(), completion_ids.begin(), completion_ids.end());
            full_sequence.push_back(eos_id);

            bool partial_holdout = false;
            size_t train_completion_end = completion_ids.size();
            size_t validation_target_begin = completion_begin;
            if (validation_holdout && samples.size() == 1 && completion_ids.size() >= 2)
            {
                size_t heldout = static_cast<size_t>(std::ceil(
                    static_cast<double>(completion_ids.size()) * configured_holdout_fraction));
                heldout = std::clamp<size_t>(heldout, 1, completion_ids.size() - 1);
                train_completion_end = completion_ids.size() - heldout;
                validation_target_begin = completion_begin + train_completion_end;
                partial_holdout = true;
                token_split_used = true;
            }

            const bool include_validation = !validation_holdout || is_validation[index] ||
                samples.size() == 1;
            if (include_validation)
            {
                ValidationSequence validation;
                validation.tokens = full_sequence;
                validation.target_begin = validation_holdout
                    ? validation_target_begin : completion_begin;
                validation_sequences_.push_back(std::move(validation));
            }

            if (is_validation[index])
                continue;

            const size_t sequence_start = training_tokens_.size();
            training_tokens_.push_back(bos_id);
            training_target_mask_.push_back(0u);
            append_training_segment(prompt_ids, false);
            if (partial_holdout)
            {
                std::vector<int> training_completion(
                    completion_ids.begin(), completion_ids.begin() +
                        static_cast<std::ptrdiff_t>(train_completion_end));
                append_training_segment(training_completion, true);
                training_tokens_.push_back(eos_id);
                training_target_mask_.push_back(0u);
            }
            else
            {
                append_training_segment(completion_ids, true);
                training_tokens_.push_back(eos_id);
                training_target_mask_.push_back(1u);
            }
            if (training_tokens_.size() <= sequence_start)
                throw std::logic_error("CausalLMModel: empty encoded training sample");
        }
        training_sample_count = samples.size() - heldout_sample_count;

        if (!validation_holdout)
        {
            // This explicit opt-out measures fit on the training dialogues.
            // Mark the source in checkpoint/UI metadata so it is not mistaken
            // for an independent holdout score.
            heldout_sample_count = 0;
        }
        else if (samples.size() == 1 && !token_split_used)
        {
            // A one-token answer cannot be split without leaving no supervised
            // continuation in training; retain it and report train-set validation.
            validation_sequences_.front().target_begin =
                1 + encode_content(decorated_prompts.front()).size();
        }
        if (samples.size() == 1)
            training_sample_count = 1;
    }
    if (training_tokens_.size() <= static_cast<size_t>(cfg_.seq_len))
    {
        throw std::invalid_argument(
            "CausalLMModel: corpus must contain more tokens than seq_len after BPE");
    }
    for (size_t index = 1; index < training_target_mask_.size(); ++index)
    {
        if (training_target_mask_[index])
            training_target_positions_.push_back(index);
    }
    if (training_target_positions_.empty())
    {
        throw std::invalid_argument("CausalLMModel: corpus produced no target tokens");
    }

    const int effective_vocab = static_cast<int>(tokenizer.getVocabSize());
    if (effective_vocab < 2 || static_cast<size_t>(effective_vocab) > vocab_capacity)
    {
        throw std::runtime_error("CausalLMModel: prepared tokenizer vocabulary is invalid");
    }
    const json previous_config = modelConfig;
    cfg_.vocab_size = effective_vocab;
    buildInto(*this, cfg_);
    for (auto it = previous_config.begin(); it != previous_config.end(); ++it)
    {
        if (!modelConfig.contains(it.key()))
            modelConfig[it.key()] = it.value();
    }
    modelConfig["vocab_size"] = effective_vocab;
    modelConfig["output_dim"] = cfg_.seq_len * effective_vocab;
    modelConfig["tokenizer_capacity"] = vocab_capacity;
    modelConfig["tokenizer_bpe_merges"] = tokenizer.getBPEMergeCount();
    modelConfig["tokenizer_reused"] = loaded_tokenizer && !needs_bpe_update &&
                                      tokenizer.getVocabSize() == loaded_vocab_size;
    modelConfig["tokenizer_bpe_updated"] = needs_bpe_update;
    modelConfig["tokenizer_updated"] = !loaded_tokenizer || needs_bpe_update ||
                                       tokenizer.getVocabSize() != loaded_vocab_size;
    const size_t validation_target_tokens = std::accumulate(
        validation_sequences_.begin(), validation_sequences_.end(), size_t{0},
        [](size_t total, const ValidationSequence &sequence) {
            return total + (sequence.target_begin < sequence.tokens.size()
                ? sequence.tokens.size() - sequence.target_begin : 0);
        });
    modelConfig["dataset_samples"] = samples.empty() ? 1 : samples.size();
    modelConfig["training_samples"] = training_sample_count;
    modelConfig["training_tokens"] = training_tokens_.size();
    modelConfig["validation_samples"] = validation_sequences_.size();
    modelConfig["validation_tokens"] = validation_target_tokens;
    const bool has_independent_holdout = validation_holdout &&
        (samples.empty() ? token_split_used : (samples.size() >= 2 || token_split_used));
    modelConfig["validation_holdout"] = has_independent_holdout;
    modelConfig["validation_source"] = has_independent_holdout
        ? (samples.size() == 1 ? "token_holdout" : "holdout") : "training";
    modelConfig["validation_holdout_fraction"] = configured_holdout_fraction;
    modelConfig["validation_holdout_items"] = configured_holdout_items;
    if (!tokenizer_path.empty())
        modelConfig["tokenizer_path"] = tokenizer_path;
    setHasTokenizer(true);
    training_rng_.seed(static_cast<uint32_t>(seed));
    training_corpus_prepared_ = true;

    if (persist_tokenizer && modelConfig.value("tokenizer_updated", false))
        savePreparedTokenizer(tokenizer_path);
}

void CausalLMModel::savePreparedTokenizer(const std::string &tokenizer_path)
{
    if (tokenizer_path.empty())
        return;
    const std::filesystem::path output_path(tokenizer_path);
    std::error_code error;
    if (!output_path.parent_path().empty())
    {
        std::filesystem::create_directories(output_path.parent_path(), error);
        if (error)
        {
            throw std::runtime_error("CausalLMModel: cannot create tokenizer directory: " +
                                     error.message());
        }
    }
    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("CausalLMModel: cannot write tokenizer: " + tokenizer_path);
    output << getTokenizer().to_json().dump(2) << '\n';
    if (!output)
        throw std::runtime_error("CausalLMModel: tokenizer write failed: " + tokenizer_path);
    modelConfig["tokenizer_path"] = tokenizer_path;
}

std::optional<Model::TrainStepResult> CausalLMModel::trainStep(
    const TrainStepRequest &request)
{
    const auto input_it = request.int_inputs.find("__input__");
    const auto target_it = request.int_inputs.find("target_ids");
    const auto mask_it = request.int_inputs.find("target_mask");
    if (input_it == request.int_inputs.end() || !input_it->second ||
        target_it == request.int_inputs.end() || !target_it->second ||
        mask_it == request.int_inputs.end() || !mask_it->second || !request.optimizer)
    {
        return std::nullopt;
    }
    const auto &input = *input_it->second;
    const auto &targets = *target_it->second;
    const auto &mask = *mask_it->second;
    const size_t seq_len = static_cast<size_t>(cfg_.seq_len);
    const size_t vocab_size = static_cast<size_t>(cfg_.vocab_size);
    if (input.size() != seq_len || targets.size() != seq_len || mask.size() != seq_len)
    {
        throw std::invalid_argument("CausalLMModel::trainStep: input, target and mask must match seq_len");
    }
    size_t active = 0;
    for (size_t index = 0; index < seq_len; ++index)
    {
        if (mask[index])
            ++active;
        if (input[index] < 0 || static_cast<size_t>(input[index]) >= vocab_size ||
            targets[index] < 0 || static_cast<size_t>(targets[index]) >= vocab_size)
        {
            throw std::invalid_argument("CausalLMModel::trainStep: token id outside model vocabulary");
        }
    }
    if (active == 0)
        throw std::invalid_argument("CausalLMModel::trainStep: sample has no target tokens");

    if (request.mode == TrainStepMode::Optimize)
        zeroGradients();
    // Keep supervision metadata in the pending named inputs so the Viz tap can
    // report predictions only for the positions that contribute to this loss.
    // The graph itself consumes only __input__; the extra tensors are telemetry.
    const std::unordered_map<std::string, std::vector<int>> int_inputs = {
        {"__input__", input}, {"target_ids", targets}, {"target_mask", mask}};
    const auto &logits = forwardPassNamedView({}, int_inputs, true);
    if (logits.size() != seq_len * vocab_size)
    {
        throw std::runtime_error("CausalLMModel::trainStep: output logits have an invalid shape");
    }

    std::vector<float> gradient(logits.size(), 0.0f);
    double loss_sum = 0.0;
    double top_two_margin_sum = 0.0;
    size_t correct_next_tokens = 0;
    for (size_t position = 0; position < seq_len; ++position)
    {
        if (!mask[position])
            continue;
        const size_t offset = position * vocab_size;
        float maximum = -std::numeric_limits<float>::infinity();
        float second_maximum = -std::numeric_limits<float>::infinity();
        size_t top_token = 0;
        for (size_t token = 0; token < vocab_size; ++token)
        {
            const float value = logits[offset + token];
            if (value > maximum)
            {
                second_maximum = maximum;
                maximum = value;
                top_token = token;
            }
            else if (value > second_maximum)
            {
                second_maximum = value;
            }
        }
        if (top_token == static_cast<size_t>(targets[position]))
            ++correct_next_tokens;
        top_two_margin_sum += static_cast<double>(maximum) - second_maximum;
        double denominator = 0.0;
        for (size_t token = 0; token < vocab_size; ++token)
        {
            denominator += std::exp(static_cast<double>(logits[offset + token] - maximum));
        }
        const size_t gold = static_cast<size_t>(targets[position]);
        loss_sum += std::log(denominator) + maximum - logits[offset + gold];
        for (size_t token = 0; token < vocab_size; ++token)
        {
            const double probability = std::exp(
                                           static_cast<double>(logits[offset + token] - maximum)) /
                                       denominator;
            gradient[offset + token] = static_cast<float>(
                (probability - (token == gold ? 1.0 : 0.0)) / static_cast<double>(active));
        }
    }
    if (request.grad_scale != 1.0f)
    {
        for (float &value : gradient)
            value *= request.grad_scale;
    }
    backwardPass(gradient);

    TrainStepResult result;
    result.loss = static_cast<float>(loss_sum / static_cast<double>(active));
    result.metrics["cross_entropy"] = result.loss;
    result.metrics["perplexity"] = static_cast<float>(std::exp(std::min(result.loss, 20.0f)));
    result.metrics["active_tokens"] = static_cast<float>(active);
    result.metrics["correct_next_tokens"] = static_cast<float>(correct_next_tokens);
    result.metrics["next_token_accuracy_pct"] = static_cast<float>(
        100.0 * static_cast<double>(correct_next_tokens) / static_cast<double>(active));
    result.metrics["mean_top1_top2_logit_margin"] = static_cast<float>(
        top_two_margin_sum / static_cast<double>(active));
    double norm_squared = 0.0;
    for (const auto &layer : getLayers())
    {
        for (float value : layer.grad_weights)
        {
            if (!std::isfinite(value))
                throw std::runtime_error("CausalLMModel: non-finite gradient");
            norm_squared += static_cast<double>(value) * value;
            result.grad_max_abs = std::max(result.grad_max_abs, std::abs(value));
        }
        for (float value : layer.grad_bias)
        {
            if (!std::isfinite(value))
                throw std::runtime_error("CausalLMModel: non-finite gradient");
            norm_squared += static_cast<double>(value) * value;
            result.grad_max_abs = std::max(result.grad_max_abs, std::abs(value));
        }
    }
    result.grad_norm = static_cast<float>(std::sqrt(norm_squared));
    if (request.mode == TrainStepMode::Optimize)
    {
        if (!std::isfinite(request.learning_rate) || request.learning_rate <= 0.0f)
        {
            throw std::invalid_argument("CausalLMModel::trainStep: learning rate must be positive and finite");
        }
        optimizerStep(*request.optimizer, request.learning_rate);
    }
    return result;
}

CausalLMModel::ValidationResult CausalLMModel::evaluateValidation(size_t max_items)
{
    if (validation_sequences_.empty())
        throw std::logic_error("CausalLMModel: validation set is empty");
    if (cfg_.seq_len < 2 || cfg_.vocab_size < 2)
        throw std::logic_error("CausalLMModel: invalid dimensions for validation");

    ValidationResult result;
    const size_t sequence_length = static_cast<size_t>(cfg_.seq_len);
    const size_t vocab_size = static_cast<size_t>(cfg_.vocab_size);
    const size_t selected_items = max_items == 0
        ? validation_sequences_.size()
        : std::min(max_items, validation_sequences_.size());
    const size_t stride = std::max<size_t>(1, sequence_length / 2);
    const size_t context_tokens = sequence_length - stride;
    const int padding_id = getTokenizer().getPadId();
    double loss_sum = 0.0;
    size_t correct_tokens = 0;

    for (size_t item = 0; item < selected_items; ++item)
    {
        const ValidationSequence &sequence = validation_sequences_[item];
        if (sequence.tokens.size() < 2 || sequence.target_begin == 0 ||
            sequence.target_begin >= sequence.tokens.size())
            continue;

        ++result.items;
        size_t chunk_begin = sequence.target_begin;
        while (chunk_begin < sequence.tokens.size())
        {
            const size_t chunk_end = std::min(sequence.tokens.size(), chunk_begin + stride);
            const size_t first_input = chunk_begin > context_tokens + 1
                ? chunk_begin - context_tokens - 1 : 0;
            std::vector<int> input(sequence_length, padding_id);
            std::vector<int> targets(sequence_length, padding_id);
            std::vector<int> mask(sequence_length, 0);
            for (size_t position = 0; position < sequence_length; ++position)
            {
                const size_t input_position = first_input + position;
                if (input_position >= sequence.tokens.size())
                    continue;
                input[position] = sequence.tokens[input_position];
                const size_t target_position = input_position + 1;
                if (target_position >= sequence.tokens.size())
                    continue;
                targets[position] = sequence.tokens[target_position];
                if (target_position >= chunk_begin && target_position < chunk_end)
                    mask[position] = 1;
            }

            const std::unordered_map<std::string, std::vector<int>> inputs = {
                {"__input__", input}};
            const std::vector<float> &logits = forwardPassNamedView({}, inputs, false);
            if (logits.size() != sequence_length * vocab_size)
                throw std::runtime_error(
                    "CausalLMModel validation: output logits have an invalid shape");

            for (size_t position = 0; position < sequence_length; ++position)
            {
                if (!mask[position])
                    continue;
                const int target_id = targets[position];
                if (target_id < 0 || static_cast<size_t>(target_id) >= vocab_size)
                    throw std::runtime_error(
                        "CausalLMModel validation: target id outside model vocabulary");
                const size_t offset = position * vocab_size;
                float maximum = -std::numeric_limits<float>::infinity();
                size_t top_token = 0;
                for (size_t token = 0; token < vocab_size; ++token)
                {
                    const float value = logits[offset + token];
                    if (!std::isfinite(value))
                        throw std::runtime_error(
                            "CausalLMModel validation: non-finite logit");
                    if (value > maximum)
                    {
                        maximum = value;
                        top_token = token;
                    }
                }
                double denominator = 0.0;
                for (size_t token = 0; token < vocab_size; ++token)
                    denominator += std::exp(static_cast<double>(logits[offset + token] - maximum));
                if (!std::isfinite(denominator) || denominator <= 0.0)
                    throw std::runtime_error(
                        "CausalLMModel validation: invalid softmax denominator");
                loss_sum += std::log(denominator) + maximum -
                    logits[offset + static_cast<size_t>(target_id)];
                if (top_token == static_cast<size_t>(target_id))
                    ++correct_tokens;
                ++result.active_tokens;
            }
            chunk_begin = chunk_end;
        }
    }

    if (result.active_tokens == 0 || result.items == 0)
        throw std::runtime_error("CausalLMModel: validation set contains no target tokens");
    result.cross_entropy = loss_sum / static_cast<double>(result.active_tokens);
    result.perplexity = std::exp(std::min(result.cross_entropy, 20.0));
    result.next_token_accuracy_pct = 100.0 * static_cast<double>(correct_tokens) /
        static_cast<double>(result.active_tokens);
    return result;
}

int CausalLMModel::trainCorpus(int steps, Optimizer &optimizer, float learning_rate,
                               int seed, const TrainingCallback &callback,
                               const TrainingControl &continue_training)
{
    if (!training_corpus_prepared_ || training_target_positions_.empty())
    {
        throw std::logic_error("CausalLMModel: prepareTrainingCorpus must run before training");
    }
    if (layer_weight_blocks.empty())
    {
        throw std::logic_error("CausalLMModel: allocate and initialize parameters before training");
    }
    if (steps < 1 || !std::isfinite(learning_rate) || learning_rate <= 0.0f)
    {
        throw std::invalid_argument("CausalLMModel: training steps and learning rate must be positive");
    }
    training_rng_.seed(static_cast<uint32_t>(seed));
    std::uniform_int_distribution<size_t> choose_target(0, training_target_positions_.size() - 1);
    // Rebuild the per-rank sampling stream from its original seed. Checkpoints
    // store optimizer.step, so advancing the same distribution reproduces the
    // training-window sequence without serializing implementation-specific RNG
    // internals into the portable checkpoint format.
    for (size_t skipped = 0; skipped < optimizer.step; ++skipped)
        (void)choose_target(training_rng_);
    const size_t seq_len = static_cast<size_t>(cfg_.seq_len);
    const size_t last_first = training_tokens_.size() - seq_len - 1;
    const int validate_every_steps = std::max(0,
        modelConfig.value("validate_every_steps", 100));
    const int validate_items = std::max(0, modelConfig.value("validate_items", 8));
    const bool validation_configured = validate_every_steps > 0 && validate_items > 0 &&
        !validation_sequences_.empty();
    std::optional<ValidationResult> last_validation;
    size_t last_validation_step = 0;

    int completed_steps = 0;
    double loss_sum = 0.0;
    for (int step = 1; step <= steps; ++step)
    {
        if (continue_training && !continue_training(step))
            break;
        const size_t target_position = training_target_positions_[choose_target(training_rng_)];
        const size_t first = target_position > seq_len
                                 ? std::min(target_position - seq_len, last_first)
                                 : 0;
        std::vector<int> input(seq_len), targets(seq_len), mask(seq_len, 0);
        for (size_t index = 0; index < seq_len; ++index)
        {
            input[index] = training_tokens_[first + index];
            targets[index] = training_tokens_[first + index + 1];
            mask[index] = training_target_mask_[first + index + 1] ? 1 : 0;
        }
        TrainStepRequest request;
        request.int_inputs["__input__"] = &input;
        request.int_inputs["target_ids"] = &targets;
        request.int_inputs["target_mask"] = &mask;
        request.optimizer = &optimizer;
        request.learning_rate = learning_rate;
        float effective_learning_rate = learning_rate;
        if (optimizer.warmup_steps > 0 || optimizer.decay_strategy != LRDecayStrategy::NONE)
        {
            const float base = std::max(1e-12f, optimizer.initial_lr);
            effective_learning_rate = optimizer.getCurrentLR() * (learning_rate / base);
        }
        auto result = trainStep(request);
        if (!result)
            throw std::runtime_error("CausalLMModel: native trainStep rejected its own sample");
        loss_sum += result->loss;
        result->metrics["learning_rate"] = effective_learning_rate;
        result->metrics["average_loss"] = static_cast<float>(loss_sum / step);
        result->metrics["optimizer_step"] = static_cast<float>(optimizer.step);
        const bool validation_enabled = modelConfig.value("validation_enabled", true);
        const bool validation_performed = validation_configured && validation_enabled &&
            (optimizer.step % static_cast<size_t>(validate_every_steps) == 0 || step == steps);
        if (validation_performed)
        {
            last_validation = evaluateValidation(static_cast<size_t>(validate_items));
            last_validation_step = optimizer.step;
        }
        result->metrics["validation_performed"] = validation_performed ? 1.0f : 0.0f;
        if (last_validation)
        {
            result->metrics["last_validation_cross_entropy"] =
                static_cast<float>(last_validation->cross_entropy);
            result->metrics["last_validation_perplexity"] =
                static_cast<float>(last_validation->perplexity);
            result->metrics["last_validation_next_token_accuracy_pct"] =
                static_cast<float>(last_validation->next_token_accuracy_pct);
            result->metrics["last_validation_active_tokens"] =
                static_cast<float>(last_validation->active_tokens);
            result->metrics["last_validation_items"] =
                static_cast<float>(last_validation->items);
            result->metrics["last_validation_step"] =
                static_cast<float>(last_validation_step);
            if (validation_performed)
            {
                result->metrics["validation_cross_entropy"] =
                    static_cast<float>(last_validation->cross_entropy);
                result->metrics["validation_perplexity"] =
                    static_cast<float>(last_validation->perplexity);
                result->metrics["validation_next_token_accuracy_pct"] =
                    static_cast<float>(last_validation->next_token_accuracy_pct);
                result->metrics["validation_active_tokens"] =
                    static_cast<float>(last_validation->active_tokens);
                result->metrics["validation_items"] =
                    static_cast<float>(last_validation->items);
                result->metrics["validation_step"] =
                    static_cast<float>(last_validation_step);
            }
        }
        for (const char *name : {"training_samples", "training_tokens", "vocab_size",
                                 "dataset_samples", "validation_samples", "validation_tokens",
                                 "tokenizer_capacity", "tokenizer_bpe_merges", "seq_len",
                                 "latent_dim", "local_conv_kernel", "local_conv_dilation",
                                 "causal_arch_version", "validate_every_steps", "validate_items",
                                 "validation_holdout_fraction"})
        {
            const auto value = modelConfig.find(name);
            if (value != modelConfig.end() && value->is_number())
                result->metrics[name] = static_cast<float>(value->get<double>());
        }
        result->metrics["fused_swiglu"] = cfg_.fused_swiglu ? 1.0f : 0.0f;
        if (modelConfig["training_state"].is_null() || !modelConfig["training_state"].is_object())
            modelConfig["training_state"] = json::object();
        modelConfig["training_state"]["global_step"] = optimizer.step;
        modelConfig["training_state"]["training_step"] = optimizer.step;
        modelConfig["training_state"]["loss"] = result->loss;
        modelConfig["training_state"]["perplexity"] = result->metrics.at("perplexity");
        modelConfig["training_state"]["learning_rate"] = effective_learning_rate;
        if (validation_performed && last_validation)
        {
            modelConfig["training_state"]["validation_step"] = last_validation_step;
            modelConfig["training_state"]["validation_cross_entropy"] =
                last_validation->cross_entropy;
            modelConfig["training_state"]["validation_perplexity"] =
                last_validation->perplexity;
            modelConfig["training_state"]["validation_next_token_accuracy_pct"] =
                last_validation->next_token_accuracy_pct;
            modelConfig["training_state"]["validation_active_tokens"] =
                last_validation->active_tokens;
            modelConfig["training_state"]["validation_items"] = last_validation->items;
        }
        completed_steps = step;
        if (callback)
            callback(step, *result, optimizer);
    }
    setSerializedOptimizer(optimizer);
    modelConfig["training_state"]["completed_steps"] = optimizer.step;
    return completed_steps;
}

mimir::viz::LatentSpaceRawSnapshot CausalLMModel::captureLatentSpaceSnapshot(
    size_t optimizer_step, size_t max_token_vectors,
    size_t max_sequence_positions, size_t max_vectors_per_matrix) const
{
    auto snapshot = Model::captureLatentSpaceSnapshot(optimizer_step, max_token_vectors,
        max_sequence_positions, max_vectors_per_matrix);
    // The shared capture owns all sampling/layout logic. Retain only actual
    // language-model metadata here; other architectures use the same capture.
    snapshot.d_model = cfg_.d_model;
    snapshot.latent_dim = modelConfig.value("latent_dim", cfg_.latent_dim);
    snapshot.position_encoding = cfg_.rope_theta > 0.f ? "RoPE (non appris)" : "";
    if (hasTensorInt("__input__"))
        snapshot.sequence_length = static_cast<int>(std::min<size_t>(
            getTensorInt("__input__").size(), std::numeric_limits<int>::max()));
    return snapshot;
}

void CausalLMModel::buildInto(Model &model, const Config &cfg)
{
    if (cfg.vocab_size < 2 || cfg.seq_len < 2 || cfg.d_model < 2 ||
        cfg.num_layers < 1 || cfg.num_heads < 1 || cfg.num_kv_heads < 1 ||
        cfg.mlp_hidden < 1)
    {
        throw std::invalid_argument("CausalLMModel: invalid non-positive dimensions");
    }
    if (cfg.causal_arch_version < 1 || cfg.causal_arch_version > 2 ||
        cfg.local_conv_kernel < 0 || cfg.local_conv_dilation < 1)
        throw std::invalid_argument("CausalLMModel: invalid architecture version or local convolution geometry");
    const int latent_dim = cfg.latent_dim < 0
                               ? defaultCausalLatentDim(cfg.d_model)
                               : cfg.latent_dim;
    if (latent_dim < 0 || latent_dim >= cfg.d_model)
    {
        throw std::invalid_argument(
            "CausalLMModel: latent_dim must be zero or smaller than d_model; use -1 for the automatic default");
    }
    if (cfg.d_model % cfg.num_heads != 0 || cfg.num_heads % cfg.num_kv_heads != 0)
    {
        throw std::invalid_argument("CausalLMModel: d_model must be divisible by num_heads");
    }

    model.getMutableLayers().clear();
    model.setModelName("CausalLMModel");
    model.setHasEncoder(false);
    model.modelConfig = {
        {"type", "causal_lm"},
        {"task", "causal_language_modeling"},
        {"input_dim", cfg.seq_len},
        {"output_dim", cfg.seq_len * cfg.vocab_size},
        {"vocab_size", cfg.vocab_size},
        {"seq_len", cfg.seq_len},
        {"d_model", cfg.d_model},
        {"num_layers", cfg.num_layers},
        {"num_heads", cfg.num_heads},
        {"num_kv_heads", cfg.num_kv_heads},
        {"mlp_hidden", cfg.mlp_hidden},
        {"causal_arch_version", cfg.causal_arch_version},
        {"local_conv_kernel", cfg.local_conv_kernel},
        {"local_conv_dilation", cfg.local_conv_dilation},
        {"fused_swiglu", cfg.fused_swiglu},
        {"latent_dim", latent_dim},
        {"padding_idx", cfg.padding_idx},
        {"norm_eps", cfg.norm_eps},
        {"rope_theta", cfg.rope_theta},
        {"causal", true},
    };

    model.push("causal_lm/token_embedding", "Embedding",
               checkedMul(cfg.vocab_size, cfg.d_model));
    if (auto *layer = model.getLayerByName("causal_lm/token_embedding"))
    {
        layer->inputs = {"__input__"};
        layer->output = "causal_lm/token_vectors";
        layer->vocab_size = cfg.vocab_size;
        layer->embed_dim = cfg.d_model;
        layer->padding_idx = cfg.padding_idx;
        layer->seq_len = cfg.seq_len;
    }

    std::string x = "causal_lm/token_vectors";
    for (int index = 0; index < cfg.num_layers; ++index)
    {
        const std::string prefix = "causal_lm/block" + std::to_string(index + 1);

        model.push(prefix + "/attn_norm", "RMSNorm", cfg.d_model);
        if (auto *layer = model.getLayerByName(prefix + "/attn_norm"))
        {
            layer->inputs = {x};
            layer->output = prefix + "/attn_norm_out";
            layer->in_features = cfg.d_model;
            layer->eps = cfg.norm_eps;
            layer->affine = true;
            layer->use_bias = false;
        }

        const int head_dim = cfg.d_model / cfg.num_heads;
        const int kv_dim = cfg.num_kv_heads * head_dim;
        const size_t attention_params = checkedMul(
            cfg.d_model, static_cast<size_t>(2 * cfg.d_model + 2 * kv_dim));
        model.push(prefix + "/attention", "MultiHeadAttention", attention_params);
        if (auto *layer = model.getLayerByName(prefix + "/attention"))
        {
            layer->inputs = {prefix + "/attn_norm_out"};
            layer->output = prefix + "/attention_out";
            layer->seq_len = cfg.seq_len;
            layer->embed_dim = cfg.d_model;
            layer->num_heads = cfg.num_heads;
            layer->num_kv_heads = cfg.num_kv_heads;
            layer->causal = true;
            layer->rope_theta = cfg.rope_theta;
            layer->use_bias = false;
        }

        model.push(prefix + "/attn_residual", "Add", 0);
        if (auto *layer = model.getLayerByName(prefix + "/attn_residual"))
        {
            layer->inputs = {x, prefix + "/attention_out"};
            layer->output = prefix + "/post_attention";
        }

        std::string ffn_input = prefix + "/post_attention";
        if (cfg.local_conv_kernel > 0)
        {
            model.push(prefix + "/local_norm", "RMSNorm", cfg.d_model);
            auto *norm = model.getLayerByName(prefix + "/local_norm");
            norm->inputs = {ffn_input}; norm->output = prefix + "/local_norm_out";
            norm->in_features = cfg.d_model; norm->eps = cfg.norm_eps;
            norm->affine = true; norm->use_bias = false;

            model.push(prefix + "/local_conv", "CausalDepthwiseConv1d",
                checkedMul(static_cast<size_t>(cfg.d_model), static_cast<size_t>(cfg.local_conv_kernel)));
            auto *conv = model.getLayerByName(prefix + "/local_conv");
            conv->inputs = {prefix + "/local_norm_out"}; conv->output = prefix + "/local_conv_out";
            conv->in_features = conv->out_features = conv->embed_dim = cfg.d_model;
            conv->kernel_size = cfg.local_conv_kernel; conv->dilation = cfg.local_conv_dilation;
            conv->seq_len = cfg.seq_len; conv->use_bias = false; conv->causal = true;

            model.push(prefix + "/local_activation", "SiLU", 0);
            auto *activation = model.getLayerByName(prefix + "/local_activation");
            activation->inputs = {prefix + "/local_conv_out"}; activation->output = prefix + "/local_activated";
            configureLinear(model, prefix + "/local_out", prefix + "/local_activated",
                prefix + "/local_update", cfg.d_model, cfg.d_model, cfg.seq_len);
            model.push(prefix + "/local_residual", "Add", 0);
            auto *residual = model.getLayerByName(prefix + "/local_residual");
            residual->inputs = {ffn_input, prefix + "/local_update"}; residual->output = prefix + "/local_context";
            ffn_input = prefix + "/local_context";
        }
        if (latent_dim > 0)
        {
            model.push(prefix + "/latent_norm", "RMSNorm", cfg.d_model);
            if (auto *layer = model.getLayerByName(prefix + "/latent_norm"))
            {
                layer->inputs = {ffn_input};
                layer->output = prefix + "/latent_norm_out";
                layer->in_features = cfg.d_model;
                layer->eps = cfg.norm_eps;
                layer->affine = true;
                layer->use_bias = false;
            }

            configureLinear(model, prefix + "/latent_down",
                            prefix + "/latent_norm_out", prefix + "/latent_code",
                            cfg.d_model, latent_dim, cfg.seq_len);
            model.push(prefix + "/latent_activation", "SiLU", 0);
            if (auto *layer = model.getLayerByName(prefix + "/latent_activation"))
            {
                layer->inputs = {prefix + "/latent_code"};
                layer->output = prefix + "/latent_activated";
            }
            configureLinear(model, prefix + "/latent_up",
                            prefix + "/latent_activated", prefix + "/latent_update",
                            latent_dim, cfg.d_model, cfg.seq_len);

            model.push(prefix + "/latent_residual", "Add", 0);
            if (auto *layer = model.getLayerByName(prefix + "/latent_residual"))
            {
                layer->inputs = {ffn_input, prefix + "/latent_update"};
                layer->output = prefix + "/latent_context";
            }
            ffn_input = prefix + "/latent_context";
        }

        model.push(prefix + "/ffn_norm", "RMSNorm", cfg.d_model);
        if (auto *layer = model.getLayerByName(prefix + "/ffn_norm"))
        {
            layer->inputs = {ffn_input};
            layer->output = prefix + "/ffn_norm_out";
            layer->in_features = cfg.d_model;
            layer->eps = cfg.norm_eps;
            layer->affine = true;
            layer->use_bias = false;
        }

        configureLinear(model, prefix + "/gate", prefix + "/ffn_norm_out",
                        prefix + "/gate_pre", cfg.d_model, cfg.mlp_hidden, cfg.seq_len);
        if (!cfg.fused_swiglu)
        {
            model.push(prefix + "/gate_silu", "SiLU", 0);
            auto *layer = model.getLayerByName(prefix + "/gate_silu");
            layer->inputs = {prefix + "/gate_pre"};
            layer->output = prefix + "/gate_out";
        }
        configureLinear(model, prefix + "/up", prefix + "/ffn_norm_out",
                        prefix + "/up_out", cfg.d_model, cfg.mlp_hidden, cfg.seq_len);
        model.push(prefix + "/swiglu", cfg.fused_swiglu ? "SwiGLU" : "Multiply", 0);
        if (auto *layer = model.getLayerByName(prefix + "/swiglu"))
        {
            layer->inputs = {prefix + (cfg.fused_swiglu ? "/gate_pre" : "/gate_out"), prefix + "/up_out"};
            layer->output = prefix + "/swiglu_out";
            layer->in_features = layer->out_features = layer->embed_dim = cfg.mlp_hidden;
            layer->seq_len = cfg.seq_len;
        }
        configureLinear(model, prefix + "/down", prefix + "/swiglu_out",
                        prefix + "/ffn_out", cfg.mlp_hidden, cfg.d_model, cfg.seq_len);

        model.push(prefix + "/ffn_residual", "Add", 0);
        if (auto *layer = model.getLayerByName(prefix + "/ffn_residual"))
        {
            layer->inputs = {ffn_input, prefix + "/ffn_out"};
            layer->output = prefix + "/output";
        }
        x = prefix + "/output";
    }

    model.push("causal_lm/final_norm", "RMSNorm", cfg.d_model);
    if (auto *layer = model.getLayerByName("causal_lm/final_norm"))
    {
        layer->inputs = {x};
        layer->output = "causal_lm/final_hidden";
        layer->in_features = cfg.d_model;
        layer->eps = cfg.norm_eps;
        layer->affine = true;
        layer->use_bias = false;
    }
    model.push("causal_lm/lm_head", "Linear", 0);
    if (auto *layer = model.getLayerByName("causal_lm/lm_head"))
    {
        layer->inputs = {"causal_lm/final_hidden"};
        layer->output = "x";
        layer->in_features = cfg.d_model;
        layer->out_features = cfg.vocab_size;
        layer->seq_len = cfg.seq_len;
        layer->use_bias = false;
        layer->shared_weights_from = "causal_lm/token_embedding";
    }
}

bool CausalLMModel::InitVizTips()
{
    clearVizTipsRegistry();
    registerVizTip("causal_lm/token_embedding", "Texte / embeddings des tokens");
    const int num_layers = modelConfig.value("num_layers", 0);
    const int latent_dim = modelConfig.value("latent_dim", 0);
    for (int index = 1; index <= num_layers; ++index)
    {
        const std::string prefix = "causal_lm/block" + std::to_string(index);
        registerVizTip(prefix + "/attention", "Contexte causal / attention " + std::to_string(index));
        registerVizTip(prefix + "/attn_residual", "Contexte causal / état résiduel " + std::to_string(index));
        if (modelConfig.value("local_conv_kernel", 0) > 0)
        {
            registerVizTip(prefix + "/local_conv", "Contexte local / convolution causale " + std::to_string(index));
            registerVizTip(prefix + "/local_residual", "Contexte local / état réinjecté " + std::to_string(index));
        }
        if (latent_dim > 0)
        {
            registerVizTip(prefix + "/latent_down", "Latent / code de contexte " + std::to_string(index));
            registerVizTip(prefix + "/latent_residual", "Latent / contexte réinjecté " + std::to_string(index));
        }
        registerVizTip(prefix + "/swiglu", "Langage / activation SwiGLU " + std::to_string(index));
        registerVizTip(prefix + "/ffn_residual", "Contexte / sortie du bloc " + std::to_string(index));
    }
    registerVizTip("causal_lm/final_norm", "Contexte / représentation finale");
    registerVizTip("causal_lm/lm_head", "Texte / logits du vocabulaire");
    return true;
}

bool CausalLMModel::UpdateVizTips(const Layer &layer, VizFrame &frame)
{
    const bool registered = Model::UpdateVizTips(layer, frame);
    if (layer.name.rfind("causal_lm/", 0) != 0)
        return registered;

    if (!registered)
        frame.label += " | Causal LM";
    frame.tensor_info += " | seq_len=" + std::to_string(modelConfig.value("seq_len", 0)) +
                         " d_model=" + std::to_string(modelConfig.value("d_model", 0)) +
                         " vocab=" + std::to_string(modelConfig.value("vocab_size", 0));
    if (layer.name.find("/attention") != std::string::npos)
        frame.tensor_info += " | accès au token courant et au passé uniquement";
    if (layer.name.find("/latent_down") != std::string::npos)
        frame.tensor_info += " | code compact par position, latent_dim=" +
                             std::to_string(modelConfig.value("latent_dim", 0));
    if (layer.name.find("/latent_residual") != std::string::npos)
        frame.tensor_info += " | mise à jour latente réinjectée par connexion résiduelle";
    if (layer.type_enum == LayerType::CausalDepthwiseConv1d)
        frame.tensor_info += " | convolution par canal, noyau=" + std::to_string(layer.kernel_size) +
            " dilatation=" + std::to_string(layer.dilation) + "; token courant et passé uniquement";
    if (layer.name == "causal_lm/lm_head")
        frame.tensor_info += " | chaque ligne de logits propose le token suivant";
    return true;
}

const std::vector<float> &CausalLMModel::forwardPassNamedView(
    const std::unordered_map<std::string, std::vector<float>> &float_inputs,
    const std::unordered_map<std::string, std::vector<int>> &int_inputs,
    bool training)
{
    const auto &output = Model::forwardPassNamedView(float_inputs, int_inputs, training);
    if (!isVizTapsEnabled() || modelConfig.value("viz_taps_suppressed", false))
        return output;

    std::vector<int> input_ids;
    for (const char *key : {"__input__", "seq", "text_ids", "input_ids", "x"})
    {
        auto int_it = int_inputs.find(key);
        if (int_it != int_inputs.end())
        {
            input_ids = int_it->second;
            break;
        }
        auto float_it = float_inputs.find(key);
        if (float_it != float_inputs.end())
        {
            input_ids.reserve(float_it->second.size());
            for (float value : float_it->second)
            {
                if (!std::isfinite(value))
                {
                    input_ids.clear();
                    break;
                }
                input_ids.push_back(static_cast<int>(std::lround(value)));
            }
            if (!input_ids.empty())
                break;
        }
    }

    const int vocab = modelConfig.value("vocab_size", 0);
    const int seq_len = modelConfig.value("seq_len", 0);
    if (input_ids.empty() || vocab < 2 || seq_len < 1 ||
        output.size() < static_cast<size_t>(vocab) * static_cast<size_t>(seq_len))
    {
        return output;
    }

    // Keep the complete model sequence in the text statistics. Viz's max-side
    // setting is for visual previews and must not truncate the reported tokens.
    const size_t count = std::min(input_ids.size(), static_cast<size_t>(seq_len));
    if (count == 0)
        return output;

    const auto target_ids_it = int_inputs.find("target_ids");
    const auto target_mask_it = int_inputs.find("target_mask");
    const bool has_supervision_mask = training &&
        target_ids_it != int_inputs.end() && target_ids_it->second.size() >= count &&
        target_mask_it != int_inputs.end() && target_mask_it->second.size() >= count;

    const int padding_idx = modelConfig.value("padding_idx", 0);
    VizFrame frame;
    frame.label = "Texte / contexte causal et prédictions";
    frame.w = static_cast<int>(count);
    frame.h = 1;
    frame.channels = 3;
    frame.pixels.assign(count * 3, 0);
    frame.tensor_sample_channels = 1;

    std::vector<int> visible_context;
    std::vector<int> predicted_next;
    std::vector<int> supervised_targets;
    visible_context.reserve(count);
    predicted_next.reserve(count);
    supervised_targets.reserve(count);
    size_t exact = 0;
    size_t comparable = 0;
    double margin_sum = 0.0;
    size_t margin_count = 0;

    for (size_t position = 0; position < count; ++position)
    {
        const size_t offset = position * static_cast<size_t>(vocab);
        float best_logit = -std::numeric_limits<float>::infinity();
        float second_logit = -std::numeric_limits<float>::infinity();
        int best_id = 0;
        for (int token_id = 0; token_id < vocab; ++token_id)
        {
            const float value = output[offset + static_cast<size_t>(token_id)];
            if (!std::isfinite(value))
                continue;
            if (value > best_logit)
            {
                second_logit = best_logit;
                best_logit = value;
                best_id = token_id;
            }
            else if (value > second_logit)
            {
                second_logit = value;
            }
        }
        if (!std::isfinite(best_logit))
            continue;

        visible_context.push_back(input_ids[position]);
        const bool score_position = !has_supervision_mask ||
                                    target_mask_it->second[position] != 0;
        if (score_position)
            predicted_next.push_back(best_id);
        const double raw_margin = std::isfinite(second_logit)
                                      ? std::max(0.0, static_cast<double>(best_logit) - static_cast<double>(second_logit))
                                      : 0.0;
        const double margin = std::min(raw_margin,
                                       static_cast<double>(std::numeric_limits<float>::max()));
        if (score_position)
        {
            margin_sum += margin;
            ++margin_count;
        }

        const double strength = margin / (1.0 + margin);
        const size_t pixel = position * 3;
        frame.pixels[pixel] = static_cast<uint8_t>(std::lround(220.0 * (1.0 - strength)));
        frame.pixels[pixel + 1] = static_cast<uint8_t>(std::lround(200.0 * strength));
        frame.pixels[pixel + 2] = 45;
        frame.tensor_values.push_back(static_cast<float>(margin));
        frame.tensor_indices.push_back(offset + static_cast<size_t>(best_id));

        if (has_supervision_mask && score_position)
        {
            const int target_id = target_ids_it->second[position];
            supervised_targets.push_back(target_id);
            ++comparable;
            exact += best_id == target_id ? 1u : 0u;
        }
        else if (!has_supervision_mask && position + 1 < count &&
                 input_ids[position + 1] != padding_idx)
        {
            ++comparable;
            exact += best_id == input_ids[position + 1] ? 1u : 0u;
        }
    }

    const double mean_margin = margin_count ? margin_sum / static_cast<double>(margin_count) : 0.0;
    const std::string prediction_label = has_supervision_mask
        ? "Top-1 aux positions supervisées (argmax, pas une génération libre): "
        : "Top-1 à chaque position visible (argmax, pas une génération libre): ";
    frame.tensor_info = std::string(training ? "Mode: entraînement" : "Mode: inférence") +
                        "\nContexte fourni: " + getTokenizer().decode(visible_context) +
                        (has_supervision_mask
                             ? "\nCibles supervisées: " + getTokenizer().decode(supervised_targets)
                             : std::string()) +
                        "\n" + prediction_label + getTokenizer().decode(predicted_next) +
                        (has_supervision_mask
                             ? "\nAccord top-1 avec les cibles supervisées: "
                             : "\nAccord avec le token suivant visible: ") +
                        std::to_string(exact) + "/" +
                        std::to_string(comparable) +
                        "\nMarge moyenne top-1/top-2" +
                        (has_supervision_mask ? " sur positions supervisées: " : " par position: ") +
                        std::to_string(mean_margin) +
                        "\nAperçu: " + std::to_string(count) + " positions" +
                        "\nLa couleur indique la marge top-1/top-2 par position (rouge faible, vert forte); ce n'est pas une probabilité.";
    addVizTapFrame(std::move(frame));
    return output;
}

namespace
{
    int selectLanguageToken(const std::vector<float> &logits,
                            const CausalLMModel::GenerationOptions &options,
                            std::mt19937 &rng)
    {
        if (logits.empty())
            throw std::invalid_argument("CausalLMModel: empty next-token logits");
        if (!std::isfinite(options.temperature) || options.temperature < 0.0f ||
            !std::isfinite(options.top_p) || options.top_p <= 0.0f || options.top_p > 1.0f ||
            options.top_k < 0)
        {
            throw std::invalid_argument("CausalLMModel: invalid sampling options");
        }
        std::vector<std::pair<int, float>> candidates;
        candidates.reserve(logits.size());
        for (size_t index = 0; index < logits.size(); ++index)
        {
            if (std::isfinite(logits[index]))
                candidates.emplace_back(static_cast<int>(index), logits[index]);
        }
        if (candidates.empty())
            throw std::runtime_error("CausalLMModel: all next-token logits are non-finite");
        std::sort(candidates.begin(), candidates.end(), [](const auto &left, const auto &right)
                  {
        if (left.second != right.second) return left.second > right.second;
        return left.first < right.first; });
        if (options.temperature == 0.0f)
            return candidates.front().first;
        if (options.top_k > 0 && candidates.size() > static_cast<size_t>(options.top_k))
            candidates.resize(static_cast<size_t>(options.top_k));

        const double maximum = static_cast<double>(candidates.front().second) / options.temperature;
        std::vector<double> weights;
        weights.reserve(candidates.size());
        double total = 0.0;
        for (const auto &candidate : candidates)
        {
            const double weight = std::exp(static_cast<double>(candidate.second) /
                                               options.temperature -
                                           maximum);
            weights.push_back(weight);
            total += weight;
        }
        if (options.top_p < 1.0f)
        {
            size_t retained = 0;
            double cumulative = 0.0;
            for (; retained < weights.size(); ++retained)
            {
                cumulative += weights[retained] / total;
                if (cumulative >= options.top_p)
                {
                    ++retained;
                    break;
                }
            }
            retained = std::max<size_t>(1, std::min(retained, weights.size()));
            candidates.resize(retained);
            weights.resize(retained);
        }
        std::discrete_distribution<size_t> sample(weights.begin(), weights.end());
        return candidates[sample(rng)].first;
    }
} // namespace

std::vector<int> CausalLMModel::generateFromTokenContext(
    std::vector<int> context, const GenerationOptions &options)
{
    if (options.max_new_tokens < 0)
        throw std::invalid_argument("CausalLMModel: max_new_tokens must be non-negative");
    const size_t vocab_size = static_cast<size_t>(cfg_.vocab_size);
    const int pad_id = getTokenizer().getPadId();
    const int bos_id = getTokenizer().getBosId();
    const int eos_id = getTokenizer().getEosId();
    const int stop_id = options.stop_id >= 0 ? options.stop_id : eos_id;
    if (bos_id < 0 || eos_id < 0 || pad_id < 0)
        throw std::runtime_error("CausalLMModel: tokenizer is missing BOS, EOS or PAD");
    if (context.empty())
        context.push_back(bos_id);
    if (context.front() != bos_id)
        context.insert(context.begin(), bos_id);
    while (context.size() > 1 && context.back() == eos_id)
        context.pop_back();

    if (static_cast<size_t>(options.max_new_tokens) >
        std::numeric_limits<size_t>::max() - context.size())
        throw std::length_error("CausalLMModel: prompt plus generation length overflows size_t");
    const size_t runtime_seq_len = context.size() + static_cast<size_t>(options.max_new_tokens);
    if (runtime_seq_len > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        vocab_size == 0 || runtime_seq_len >
            static_cast<size_t>(std::numeric_limits<int>::max()) / vocab_size)
        throw std::length_error("CausalLMModel: prompt is too large for the native tensor dimensions");

    std::vector<int> previous_layer_lengths;
    previous_layer_lengths.reserve(getLayers().size());
    for (const auto &layer : getLayers())
        previous_layer_lengths.push_back(layer.seq_len);
    const int previous_config_seq_len = modelConfig.value("seq_len", cfg_.seq_len);
    const int previous_input_dim = modelConfig.value("input_dim", cfg_.seq_len);
    const int previous_output_dim = modelConfig.value("output_dim", cfg_.seq_len * cfg_.vocab_size);
    auto apply_sequence_shape = [&](int sequence_length)
    {
        for (auto &layer : getMutableLayers())
        {
            if (layer.seq_len > 0)
                layer.seq_len = sequence_length;
        }
        modelConfig["seq_len"] = sequence_length;
        modelConfig["input_dim"] = sequence_length;
        modelConfig["output_dim"] = sequence_length * cfg_.vocab_size;
        static_plan_.built = false;
        static_plan_.runtime_scan_dumped = false;
    };
    auto restore_sequence_shape = [&]()
    {
        auto &layers = getMutableLayers();
        for (size_t index = 0; index < layers.size() && index < previous_layer_lengths.size(); ++index)
            layers[index].seq_len = previous_layer_lengths[index];
        modelConfig["seq_len"] = previous_config_seq_len;
        modelConfig["input_dim"] = previous_input_dim;
        modelConfig["output_dim"] = previous_output_dim;
        static_plan_.built = false;
        static_plan_.runtime_scan_dumped = false;
    };

    std::mt19937 rng(static_cast<uint32_t>(options.seed));
    std::vector<int> generated;
    generated.reserve(static_cast<size_t>(options.max_new_tokens) + 1);
    apply_sequence_shape(static_cast<int>(runtime_seq_len));
    clearKVCache();
    try
    {
        for (int step = 0; step < options.max_new_tokens; ++step)
        {
            std::vector<int> input = context;
            input.resize(runtime_seq_len, pad_id);
            const std::unordered_map<std::string, std::vector<int>> int_inputs = {{"__input__", input}};
            const auto &logits = forwardPassNamedView({}, int_inputs, false);
            const size_t expected = runtime_seq_len * vocab_size;
            if (logits.size() != expected)
                throw std::runtime_error("CausalLMModel: generation logits have an invalid dynamic shape");
            const size_t offset = (context.size() - 1) * vocab_size;
            std::vector<float> next_logits(vocab_size);
            std::copy_n(logits.begin() + static_cast<std::ptrdiff_t>(offset), vocab_size,
                        next_logits.begin());
            const int token_id = selectLanguageToken(next_logits, options, rng);
            if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_size)
                throw std::runtime_error("CausalLMModel: sampled token outside vocabulary");
            if (token_id == eos_id)
            {
                generated.push_back(eos_id);
                break;
            }
            if (token_id == stop_id)
                break;
            generated.push_back(token_id);
            context.push_back(token_id);
        }
    }
    catch (...)
    {
        restore_sequence_shape();
        throw;
    }
    restore_sequence_shape();
    if (generated.empty() || generated.back() != eos_id)
        generated.push_back(eos_id);
    return generated;
}

std::vector<int> CausalLMModel::generateTokenIds(
    const std::string &prompt, const GenerationOptions &options)
{
    const Tokenizer &tokenizer = getTokenizer();
    const std::string system_prompt = options.system_prompt.empty()
                                          ? modelConfig.value("system_prompt", std::string())
                                          : options.system_prompt;
    const bool use_control_tokens = hasCausalControlTokens(tokenizer);
    std::string conditioned_prompt;
    if (use_control_tokens)
    {
        conditioned_prompt = "<SYSTEM>" + system_prompt + "\n<USER>" + prompt + "\n<ASSISTANT>";
    }
    else
    {
        conditioned_prompt = "Système: " + system_prompt + "\nUtilisateur: " + prompt + "\nAssistante:";
    }
    conditioned_prompt = formatCausalPrompt(decorateContext(conditioned_prompt), use_control_tokens);
    std::vector<int> prompt_ids = tokenizeContent(conditioned_prompt);
    std::vector<int> context;
    context.reserve(prompt_ids.size() + 1);
    context.push_back(tokenizer.getBosId());
    context.insert(context.end(), prompt_ids.begin(), prompt_ids.end());
    return generateFromTokenContext(std::move(context), options);
}

std::string CausalLMModel::generateText(
    const std::string &prompt, const GenerationOptions &options)
{
    const std::vector<int> tokens = removeResponseMarkers(generateTokenIds(prompt, options), getTokenizer());
    return getTokenizer().decode(tokens);
}

void CausalLMModel::loadConversationMemory(const std::string &memory_path)
{
    if (conversation_memory_path_ == memory_path)
        return;
    conversation_memory_path_ = memory_path;
    conversation_memory_.clear();
    if (memory_path.empty())
        return;
    std::ifstream input(memory_path, std::ios::binary);
    if (!input)
        return;
    try
    {
        json saved;
        input >> saved;
        if (!saved.is_object() || !saved.contains("turns") || !saved["turns"].is_array())
            return;
        for (const auto &turn : saved["turns"])
        {
            if (!turn.is_object() || !turn.contains("user") || !turn.contains("assistant") ||
                !turn["user"].is_string() || !turn["assistant"].is_string())
                continue;
            conversation_memory_.push_back({turn["user"].get<std::string>(),
                                            turn["assistant"].get<std::string>()});
        }
    }
    catch (...)
    {
        conversation_memory_.clear();
    }
}

void CausalLMModel::saveConversationMemory() const
{
    if (conversation_memory_path_.empty())
        return;
    const std::filesystem::path path(conversation_memory_path_);
    std::error_code error;
    if (!path.parent_path().empty())
    {
        std::filesystem::create_directories(path.parent_path(), error);
        if (error)
            throw std::runtime_error("CausalLMModel: cannot create memory directory: " + error.message());
    }
    json saved = json::object();
    saved["version"] = 1;
    saved["turns"] = json::array();
    for (const auto &turn : conversation_memory_)
    {
        saved["turns"].push_back({{"user", turn.user}, {"assistant", turn.assistant}});
    }
    const auto temporary = path.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output)
            throw std::runtime_error("CausalLMModel: cannot write conversation memory");
        output << saved.dump(2) << '\n';
        if (!output)
            throw std::runtime_error("CausalLMModel: memory write failed");
    }
    std::filesystem::rename(temporary, path, error);
    if (error)
    {
        std::filesystem::remove(path, error);
        error.clear();
        std::filesystem::rename(temporary, path, error);
    }
    if (error)
        throw std::runtime_error("CausalLMModel: cannot replace conversation memory: " + error.message());
}

std::string CausalLMModel::chat(const std::string &user_message,
                                const std::string &memory_path,
                                const GenerationOptions &options)
{
    std::string resolved_memory_path = memory_path;
    if (options.use_memory && resolved_memory_path.empty())
    {
        const std::string directory = modelConfig.value("checkpoint_dir", std::string("checkpoints/causal_lm"));
        resolved_memory_path = (std::filesystem::path(directory) / "conversation_memory.json").string();
    }
    if (options.use_memory)
        loadConversationMemory(resolved_memory_path);
    else
        conversation_memory_.clear();

    const std::string system_prompt = options.system_prompt.empty()
                                          ? modelConfig.value("system_prompt", std::string())
                                          : options.system_prompt;
    const bool use_control_tokens = hasCausalControlTokens(getTokenizer());
    std::ostringstream prompt;
    if (use_control_tokens)
    {
        prompt << "<SYSTEM>" << system_prompt << '\n';
        for (const auto &turn : conversation_memory_)
        {
            prompt << "<USER>" << turn.user << "\n<ASSISTANT><FINAL>"
                   << turn.assistant << "<EOS>\n";
        }
        prompt << "<USER>" << user_message << "\n<ASSISTANT>";
    }
    else
    {
        if (!system_prompt.empty())
            prompt << "Système: " << system_prompt << '\n';
        for (const auto &turn : conversation_memory_)
        {
            prompt << "Utilisateur: " << turn.user << '\n'
                   << "Assistante: " << turn.assistant << '\n';
        }
        prompt << "Utilisateur: " << user_message << "\nAssistante:";
    }
    std::string serialized_prompt = decorateContext(prompt.str());
    serialized_prompt = formatCausalPrompt(std::move(serialized_prompt), use_control_tokens);
    const std::vector<int> encoded_context = tokenizeContent(serialized_prompt);
    std::vector<int> context;
    context.reserve(encoded_context.size() + 1);
    context.push_back(getTokenizer().getBosId());
    context.insert(context.end(), encoded_context.begin(), encoded_context.end());
    const std::vector<int> response_ids = generateFromTokenContext(std::move(context), options);
    const std::string response = getTokenizer().decode(
        removeResponseMarkers(response_ids, getTokenizer()));

    if (options.use_memory)
    {
        conversation_memory_.push_back({user_message, response});
        saveConversationMemory();
    }
    return response;
}

void CausalLMModel::resetConversationMemory(const std::string &memory_path)
{
    std::string resolved_memory_path = memory_path;
    if (resolved_memory_path.empty())
    {
        const std::string directory = modelConfig.value("checkpoint_dir", std::string("checkpoints/causal_lm"));
        resolved_memory_path = (std::filesystem::path(directory) / "conversation_memory.json").string();
    }
    conversation_memory_path_ = resolved_memory_path;
    conversation_memory_.clear();
    saveConversationMemory();
}

size_t CausalLMModel::conversationMemorySize(const std::string &memory_path)
{
    std::string resolved_memory_path = memory_path;
    if (resolved_memory_path.empty())
    {
        const std::string directory = modelConfig.value("checkpoint_dir", std::string("checkpoints/causal_lm"));
        resolved_memory_path = (std::filesystem::path(directory) / "conversation_memory.json").string();
    }
    loadConversationMemory(resolved_memory_path);
    return conversation_memory_.size();
}
