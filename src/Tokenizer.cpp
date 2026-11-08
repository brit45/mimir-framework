#include "Tokenizer.hpp"
#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <sstream>
#include <iostream>

namespace {
const std::string kBpeWordStart = "▁";
const std::string kBpeContinuation = "##";

std::vector<std::string> splitUtf8Codepoints(const std::string& text) {
    std::vector<std::string> out;
    for (size_t offset = 0; offset < text.size();) {
        const unsigned char lead = static_cast<unsigned char>(text[offset]);
        size_t length = lead < 0x80 ? 1 :
            ((lead & 0xe0) == 0xc0 ? 2 :
             ((lead & 0xf0) == 0xe0 ? 3 :
              ((lead & 0xf8) == 0xf0 ? 4 : 1)));
        if (offset + length > text.size()) length = 1;
        for (size_t i = 1; i < length; ++i) {
            if ((static_cast<unsigned char>(text[offset + i]) & 0xc0) != 0x80) {
                length = 1;
                break;
            }
        }
        out.push_back(text.substr(offset, length));
        offset += length;
    }
    return out;
}

std::string bpeSymbolContent(const std::string& symbol) {
    if (symbol.rfind(kBpeWordStart, 0) == 0) return symbol.substr(kBpeWordStart.size());
    if (symbol.rfind(kBpeContinuation, 0) == 0) return symbol.substr(kBpeContinuation.size());
    return symbol;
}

std::string mergeBpeSymbols(const std::string& left, const std::string& right) {
    const std::string prefix = left.rfind(kBpeWordStart, 0) == 0
        ? kBpeWordStart : kBpeContinuation;
    return prefix + bpeSymbolContent(left) + bpeSymbolContent(right);
}

bool isBpeWord(const std::string& token) {
    for (unsigned char c : token) {
        if (std::isalnum(c) || c >= 128) return true;
    }
    return false;
}

std::vector<std::string> initialBpeSymbols(const std::string& word) {
    auto codepoints = splitUtf8Codepoints(word);
    std::vector<std::string> symbols;
    symbols.reserve(codepoints.size());
    for (size_t i = 0; i < codepoints.size(); ++i) {
        symbols.push_back((i == 0 ? kBpeWordStart : kBpeContinuation) + codepoints[i]);
    }
    return symbols;
}

void applyBpeMerge(std::vector<std::string>& symbols,
                   const std::pair<std::string, std::string>& pair,
                   const std::string& merged) {
    if (symbols.size() < 2) return;
    std::vector<std::string> result;
    result.reserve(symbols.size());
    for (size_t i = 0; i < symbols.size();) {
        if (i + 1 < symbols.size() && symbols[i] == pair.first &&
            symbols[i + 1] == pair.second) {
            result.push_back(merged);
            i += 2;
        } else {
            result.push_back(symbols[i++]);
        }
    }
    symbols = std::move(result);
}
} // namespace

Tokenizer::Tokenizer(size_t max_vocab_)
    : maxVocab(max_vocab_)
{
    vocab.reserve(std::min<size_t>(maxVocab, 4096));
    reverse_vocab.reserve(std::min<size_t>(maxVocab, 4096));
    ensureSpecialTokens();
    initializeUnderstandingDictionaries();
}

size_t Tokenizer::getMaxVocab() const {
    return maxVocab;
}

void Tokenizer::setMaxVocab(size_t new_max) {
    const size_t cur = vocab.size();
    maxVocab = std::max(new_max, cur);
    // Best-effort: keep hash tables reasonably sized.
    try {
        vocab.reserve(std::min<size_t>(maxVocab, maxVocab));
        reverse_vocab.reserve(std::min<size_t>(maxVocab, maxVocab));
    } catch (...) {
    }
}

int Tokenizer::addToken(const std::string &tok)
{
    auto it = vocab.find(tok);
    if (it != vocab.end()) return it->second;

    if (vocab.size() >= maxVocab) {
        auto itunk = vocab.find("<UNK>");
        if (itunk != vocab.end()) return itunk->second;
        return 1;  // ID de <UNK>
    }

    int id = static_cast<int>(vocab.size());
    vocab.emplace(tok, id);
    reverse_vocab.emplace(id, tok);
    vocabSize = vocab.size();
    return id;
}

int Tokenizer::addSpecialToken(const std::string &tok)
{
    if (tok.empty())
        return getUnkId() >= 0 ? getUnkId() : 0;
    const int id = addToken(tok);
    const auto found = vocab.find(tok);
    if (found == vocab.end())
        return id;
    if (std::find(specialTokens_.begin(), specialTokens_.end(), tok) == specialTokens_.end())
    {
        specialTokens_.push_back(tok);
        std::stable_sort(specialTokens_.begin(), specialTokens_.end(),
                         [](const std::string &left, const std::string &right)
                         {
                             return left.size() > right.size();
                         });
    }
    return found->second;
}

bool Tokenizer::hasToken(const std::string &tok) const
{
    return vocab.find(tok) != vocab.end();
}

bool Tokenizer::isSpecialToken(const std::string &tok) const
{
    return std::find(specialTokens_.begin(), specialTokens_.end(), tok) != specialTokens_.end();
}

void Tokenizer::ensureVocabFromText(const std::string &text)
{
    // add all words from text to vocab
    auto toks = splitTokens(text);
    for (const auto &w : toks) {
        std::string n = normalizeToken(w);
        if (!n.empty()) addToken(n);
    }
}

int Tokenizer::getPadId() const { auto it = vocab.find("<PAD>"); return it != vocab.end() ? it->second : -1; }
int Tokenizer::getUnkId() const { auto it = vocab.find("<UNK>"); return it != vocab.end() ? it->second : -1; }
int Tokenizer::getSeqId() const { auto it = vocab.find("<SEQ>"); return it != vocab.end() ? it->second : -1; }
int Tokenizer::getModId() const { auto it = vocab.find("<MOD>"); return it != vocab.end() ? it->second : -1; }
int Tokenizer::getMagId() const { auto it = vocab.find("<MAG>"); return it != vocab.end() ? it->second : -1; }
int Tokenizer::getBosId() const { auto it = vocab.find("<BOS>"); return it != vocab.end() ? it->second : -1; }
int Tokenizer::getEosId() const { auto it = vocab.find("<EOS>"); return it != vocab.end() ? it->second : -1; }

size_t Tokenizer::getVocabSize() const { return vocab.size(); }

std::string Tokenizer::getTokenById(int id) const {
    auto it = reverse_vocab.find(id);
    if (it != reverse_vocab.end()) return it->second;
    return "<UNK>";
}

void Tokenizer::ensureSpecialTokens()
{
    // Tokens de contrôle
    const std::vector<std::string> specials = {
        "<PAD>", "<UNK>", "<SEQ>", "<MOD>", "<MAG>", "<BOS>", "<EOS>",
        // Ponctuation courante
        ".", ",", "!", "?", ";", ":", "-", "'", "\"", 
        "(", ")", "[", "]", "{", "}", 
        // Symboles courants
        "+", "-", "*", "/", "=", "<", ">", "&", "|", "~", "@", "#", "$", "%", "^",
        // Espaces spéciaux
        "\n", "\t",
        // Apostrophes françaises courantes
        "l'", "d'", "qu'", "c'", "j'", "m'", "t'", "s'", "n'"
    };
    for (const auto &s : specials) addSpecialToken(s);
}

std::vector<std::string> Tokenizer::splitTokens(const std::string &text) const
{
    std::vector<std::string> out;
    std::string cur;
    
    auto isAccentedChar = [](unsigned char c) -> bool {
        return c >= 128;  // UTF-8 multi-byte characters (accents, unicode)
    };
    
    for (size_t i = 0; i < text.size(); ++i) {
        bool matched_special = false;
        for (const auto &special : specialTokens_)
        {
            if (special.empty() || i + special.size() > text.size() ||
                text.compare(i, special.size(), special) != 0)
                continue;
            if (!cur.empty())
            {
                out.push_back(std::move(cur));
                cur.clear();
            }
            out.push_back(special);
            i += special.size() - 1;
            matched_special = true;
            break;
        }
        if (matched_special)
            continue;

        unsigned char c = static_cast<unsigned char>(text[i]);
        
        // Gérer les caractères UTF-8 multi-byte (accents)
        if (isAccentedChar(c)) {
            cur.push_back(text[i]);
            // Continuer à lire les bytes UTF-8 suivants
            while (i + 1 < text.size() && (static_cast<unsigned char>(text[i+1]) & 0xC0) == 0x80) {
                cur.push_back(text[++i]);
            }
        }
        // Alphanumeric
        else if (std::isalnum(c)) {
            cur.push_back(text[i]);
        }
        // Apostrophe contextuelle (l', d', qu', c', j', m', t', s', n')
        else if (c == '\'' && !cur.empty() && i + 1 < text.size() && std::isalpha(text[i+1])) {
            // Vérifier si c'est une contraction française
            std::string lower_cur = cur;
            std::transform(lower_cur.begin(), lower_cur.end(), lower_cur.begin(), ::tolower);
            
            if (lower_cur == "l" || lower_cur == "d" || lower_cur == "qu" || 
                lower_cur == "c" || lower_cur == "j" || lower_cur == "m" || 
                lower_cur == "t" || lower_cur == "s" || lower_cur == "n") {
                // Émettre la contraction comme token séparé
                out.push_back(cur + "'");
                cur.clear();
            } else {
                // Apostrophe interne (ex: "can't", "it's")
                cur.push_back(text[i]);
            }
        }
        // Trait d'union interne (ex: "arc-en-ciel", "peut-être")
        else if (c == '-' && !cur.empty() && i + 1 < text.size() && std::isalpha(text[i+1])) {
            cur.push_back(text[i]);
        }
        // Ponctuation ou whitespace
        else {
            // Émettre le mot courant
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
            
            // Émettre la ponctuation comme token distinct (sauf espaces)
            if (!std::isspace(c)) {
                std::string punct(1, static_cast<char>(c));
                out.push_back(punct);
            }
        }
    }
    
    if (!cur.empty()) out.push_back(cur);
    return out;
}

std::string Tokenizer::normalizeToken(std::string s) const
{
    if (s.empty()) return s;
    if (isSpecialToken(s)) return s;
    
    // Détection de ponctuation pure (garder telle quelle)
    bool isPurePunct = true;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c >= 128) {  // alnum ou UTF-8
            isPurePunct = false;
            break;
        }
    }
    if (isPurePunct) return s;  // Ponctuation garde sa casse
    
    // Normalisation pour les mots
    std::string result;
    result.reserve(s.size());
    
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        
        // Caractères UTF-8 (accents) : préserver
        if (c >= 128) {
            result.push_back(s[i]);
            // Copier les bytes UTF-8 suivants
            while (i + 1 < s.size() && (static_cast<unsigned char>(s[i+1]) & 0xC0) == 0x80) {
                result.push_back(s[++i]);
            }
        }
        // ASCII lowercase
        else if (std::isupper(c)) {
            result.push_back(std::tolower(c));
        }
        else {
            result.push_back(s[i]);
        }
    }
    
    // Trim ponctuation aux extrémités (mais garder apostrophes/traits d'union internes)
    auto isEdgePunct = [](unsigned char c) -> bool {
        return !std::isalnum(c) && c < 128 && c != '\'' && c != '-';
    };
    
    while (!result.empty() && isEdgePunct(static_cast<unsigned char>(result.front()))) {
        result.erase(result.begin());
    }
    while (!result.empty() && isEdgePunct(static_cast<unsigned char>(result.back()))) {
        result.pop_back();
    }
    
    return result;
}

std::string Tokenizer::removeAccents(const std::string &text) const
{
    // Table de conversion UTF-8 : accents → ASCII
    static const std::unordered_map<std::string, char> accentMap = {
        // Minuscules
        {"à", 'a'}, {"á", 'a'}, {"â", 'a'}, {"ã", 'a'}, {"ä", 'a'}, {"å", 'a'},
        {"è", 'e'}, {"é", 'e'}, {"ê", 'e'}, {"ë", 'e'},
        {"ì", 'i'}, {"í", 'i'}, {"î", 'i'}, {"ï", 'i'},
        {"ò", 'o'}, {"ó", 'o'}, {"ô", 'o'}, {"õ", 'o'}, {"ö", 'o'},
        {"ù", 'u'}, {"ú", 'u'}, {"û", 'u'}, {"ü", 'u'},
        {"ç", 'c'}, {"ñ", 'n'}, {"ý", 'y'}, {"ÿ", 'y'},
        {"æ", 'a'}, {"œ", 'o'},
        // Majuscules
        {"À", 'A'}, {"Á", 'A'}, {"Â", 'A'}, {"Ã", 'A'}, {"Ä", 'A'}, {"Å", 'A'},
        {"È", 'E'}, {"É", 'E'}, {"Ê", 'E'}, {"Ë", 'E'},
        {"Ì", 'I'}, {"Í", 'I'}, {"Î", 'I'}, {"Ï", 'I'},
        {"Ò", 'O'}, {"Ó", 'O'}, {"Ô", 'O'}, {"Õ", 'O'}, {"Ö", 'O'},
        {"Ù", 'U'}, {"Ú", 'U'}, {"Û", 'U'}, {"Ü", 'U'},
        {"Ç", 'C'}, {"Ñ", 'N'}, {"Ý", 'Y'},
        {"Æ", 'A'}, {"Œ", 'O'}
    };
    
    std::string result;
    result.reserve(text.size());
    
    for (size_t i = 0; i < text.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        
        // Caractère UTF-8 (potentiellement accent)
        if (c >= 128) {
            // Extraire le caractère UTF-8 complet (2-4 bytes)
            std::string utf8_char;
            utf8_char.push_back(text[i]);
            while (i + 1 < text.size() && (static_cast<unsigned char>(text[i+1]) & 0xC0) == 0x80) {
                utf8_char.push_back(text[++i]);
            }
            
            // Chercher conversion
            auto it = accentMap.find(utf8_char);
            if (it != accentMap.end()) {
                result.push_back(it->second);
            } else {
                result += utf8_char;  // Garder si non trouvé
            }
        } else {
            result.push_back(text[i]);
        }
    }
    
    return result;
}

bool Tokenizer::containsAccents(const std::string &text) const
{
    for (unsigned char c : text) {
        if (c >= 128) return true;  // UTF-8 détecté
    }
    return false;
}

std::string Tokenizer::escapePunctuation(const std::string &text) const
{
    // Échapper les caractères spéciaux pour regex
    static const std::string special_chars = ".^$*+?()[{\\|";
    std::string result;
    result.reserve(text.size() * 2);
    
    for (char c : text) {
        if (special_chars.find(c) != std::string::npos) {
            result.push_back('\\');
        }
        result.push_back(c);
    }
    
    return result;
}

std::vector<int> Tokenizer::tokenize(const std::string &text) const
{
    std::vector<int> out;
    auto toks = splitTokens(text);
    int unk = getUnkId();
    if (unk < 0) unk = 0;

    const int bos = getBosId();
    const int eos = getEosId();
    const bool use_bos_eos = (bos >= 0 && eos >= 0);

    out.reserve(toks.size() + (use_bos_eos ? 2 : 0));
    if (use_bos_eos) out.push_back(bos);

    for (const auto &w : toks) {
        std::string n = normalizeToken(w);
        if (n.empty()) continue;
        auto it = vocab.find(n);
        if (it != vocab.end()) out.push_back(it->second);
        else out.push_back(unk);
    }

    if (use_bos_eos) out.push_back(eos);
    return out;
}

std::vector<int> Tokenizer::tokenizeEnsure(const std::string &text)
{
    std::vector<int> out;
    auto toks = splitTokens(text);

    const int bos = getBosId();
    const int eos = getEosId();
    const bool use_bos_eos = (bos >= 0 && eos >= 0);

    out.reserve(toks.size() + (use_bos_eos ? 2 : 0));
    if (use_bos_eos) out.push_back(bos);

    for (const auto &w : toks) {
        std::string n = normalizeToken(w);
        if (n.empty()) continue;
        int id = addToken(n);
        out.push_back(id);
    }

    if (use_bos_eos) out.push_back(eos);
    return out;
}

std::string Tokenizer::decode(const std::vector<int> &ids) const
{
    std::ostringstream ss;
    
    auto isPunctuation = [](const std::string& s) -> bool {
        if (s.empty()) return false;
        if (s.size() > 1) return false;  // Multi-char n'est pas ponctuation simple
        unsigned char c = s[0];
        return (c >= 33 && c <= 47) || (c >= 58 && c <= 64) || 
               (c >= 91 && c <= 96) || (c >= 123 && c <= 126);
    };
    
    auto needsSpaceBefore = [](const std::string& s) -> bool {
        return s != "." && s != "," && s != "!" && s != "?" && 
               s != ";" && s != ":" && s != ")" && s != "]" && 
               s != "}" && s != "'" && s != "\"" && 
               !s.empty() && s[0] != '\'';
    };
    
    auto needsSpaceAfter = [](const std::string& s) -> bool {
        return s != "(" && s != "[" && s != "{" && s != "'" && 
               s != "\"" && !s.empty() && s.back() != '\'';
    };
    
    std::string prev;
    for (int id : ids) {
        std::string raw_token = getTokenById(id);
        const bool continuation = raw_token.rfind(kBpeContinuation, 0) == 0;
        std::string token = raw_token;
        if (token.rfind(kBpeWordStart, 0) == 0) token.erase(0, kBpeWordStart.size());
        else if (continuation) token.erase(0, kBpeContinuation.size());
        
        // Ajouter espace intelligent
        if (!prev.empty()) {
            bool curr_is_punct = isPunctuation(token);
            bool prev_is_punct = isPunctuation(prev);
            
            // Pas d'espace avant ponctuation fermante ou après ouvrante
            if (!continuation && !(curr_is_punct && !needsSpaceBefore(token)) &&
                !(prev_is_punct && !needsSpaceAfter(prev))) {
                ss << ' ';
            }
        }
        
        ss << token;
        prev = token;
    }
    
    return ss.str();
}

json Tokenizer::to_json() const
{
    json j;
    j["maxVocab"] = maxVocab;
    j["vocabSize"] = vocabSize;
    
    try {
        // Sauvegarder vocab directement (token -> id)
        json vocab_obj = json::object();
        for (const auto& [token, id] : vocab) {
            // IMPORTANT: keep tokens verbatim. JSON dumping will escape control chars
            // (e.g. "\n", "\t") in a reversible way. Replacing them by spaces
            // would introduce key collisions and lose vocabulary entries.
            vocab_obj[token] = id;
        }
        j["vocab"] = std::move(vocab_obj);
    } catch (const std::exception &e) {
        j["vocab"] = json::object();
        j["vocab_error"] = std::string("to_json exception: ") + e.what();
    } catch (...) {
        j["vocab"] = json::object();
        j["vocab_error"] = "to_json unknown exception";
    }

    j["bpe_enabled"] = bpeReady;
    j["special_tokens"] = specialTokens_;
    j["bpe_merges"] = json::array();
    std::vector<std::pair<std::pair<std::string, std::string>, int>> ordered_merges;
    ordered_merges.reserve(bpeMerges.size());
    for (const auto& merge : bpeMerges) ordered_merges.push_back(merge);
    std::sort(ordered_merges.begin(), ordered_merges.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second < b.second;
        return a.first < b.first;
    });
    for (const auto& merge : ordered_merges) {
        j["bpe_merges"].push_back({
            {"left", merge.first.first},
            {"right", merge.first.second},
            {"rank", merge.second}
        });
    }
    return j;
}

void Tokenizer::from_json(const json &j)
{
    bpeMerges.clear();
    bpeReady = false;
    specialTokens_.clear();
    if (j.contains("maxVocab")) maxVocab = j["maxVocab"].get<size_t>();
    
    // Support ancien format id2token (rétrocompatibilité)
    if (j.contains("id2token")) {
        try {
            auto id2token_arr = j["id2token"].get<std::vector<std::string>>();
            vocab.clear();
            reverse_vocab.clear();
            vocab.reserve(id2token_arr.size());
            reverse_vocab.reserve(id2token_arr.size());
            for (size_t i = 0; i < id2token_arr.size(); ++i) {
                vocab.emplace(id2token_arr[i], static_cast<int>(i));
                reverse_vocab.emplace(static_cast<int>(i), id2token_arr[i]);
            }
            vocabSize = vocab.size();
        } catch (...) {
            // ignore malformed json
        }
    }
    
    // Nouveau format vocab (token -> id)
    if (j.contains("vocab")) {
        try {
            auto vocab_obj = j["vocab"].get<std::unordered_map<std::string, int>>();
            vocab = std::move(vocab_obj);
            reverse_vocab.clear();
            reverse_vocab.reserve(vocab.size());
            for (const auto& [token, id] : vocab) {
                reverse_vocab.emplace(id, token);
            }
            vocabSize = vocab.size();
        } catch (...) {
            // ignore malformed json
        }
    }

    if (j.contains("special_tokens") && j["special_tokens"].is_array())
    {
        try
        {
            for (const auto &item : j["special_tokens"])
            {
                if (!item.is_string())
                    continue;
                const std::string token = item.get<std::string>();
                if (!token.empty() &&
                    std::find(specialTokens_.begin(), specialTokens_.end(), token) == specialTokens_.end())
                    specialTokens_.push_back(token);
            }
        }
        catch (...)
        {
            specialTokens_.clear();
        }
    }
    // Old tokenizer files do not record this list. Keep the original boundary
    // tokens atomic without mutating their vocabulary or ids while loading.
    const std::vector<std::string> boundary_tokens = {
        "<PAD>", "<UNK>", "<SEQ>", "<MOD>", "<MAG>", "<BOS>", "<EOS>"};
    for (const auto &token : boundary_tokens)
    {
        if (std::find(specialTokens_.begin(), specialTokens_.end(), token) == specialTokens_.end())
            specialTokens_.push_back(token);
    }
    std::stable_sort(specialTokens_.begin(), specialTokens_.end(),
                     [](const std::string &left, const std::string &right)
                     {
                         return left.size() > right.size();
                     });

    if (j.contains("bpe_merges") && j["bpe_merges"].is_array()) {
        try {
            for (const auto& item : j["bpe_merges"]) {
                if (!item.is_object() || !item.contains("left") ||
                    !item.contains("right") || !item.contains("rank")) continue;
                const std::string left = item["left"].get<std::string>();
                const std::string right = item["right"].get<std::string>();
                const int rank = item["rank"].get<int>();
                if (rank < 0) continue;
                const std::string merged = mergeBpeSymbols(left, right);
                if (vocab.count(left) && vocab.count(right) && vocab.count(merged))
                    bpeMerges[{left, right}] = rank;
            }
        } catch (...) {
            bpeMerges.clear();
        }
    }
    bpeReady = j.value("bpe_enabled", !bpeMerges.empty()) || !bpeMerges.empty();
}

void Tokenizer::setMaxSequenceLength(int max_len) {
    maxSequenceLength = max_len;
}

std::vector<int> Tokenizer::padSequence(const std::vector<int>& tokens, int target_len) const {
    int len = (target_len < 0) ? maxSequenceLength : target_len;

    if (len <= 0) return {};

    const int pad_id = getPadId();
    const int bos = getBosId();
    const int eos = getEosId();
    const bool use_bos_eos = (len >= 2 && bos >= 0 && eos >= 0);

    // Travail sur un contenu sans BOS/EOS, puis on reconstruit.
    size_t start = 0;
    size_t end = tokens.size();
    if (use_bos_eos) {
        if (start < end && tokens[start] == bos) ++start;
        if (end > start && tokens[end - 1] == eos) --end;
    }

    std::vector<int> content;
    if (end > start) {
        content.assign(tokens.begin() + static_cast<std::ptrdiff_t>(start), tokens.begin() + static_cast<std::ptrdiff_t>(end));
    }

    std::vector<int> padded;
    padded.reserve(static_cast<size_t>(len));

    if (use_bos_eos) {
        padded.push_back(bos);
        const int content_cap = std::max(0, len - 2);
        if (static_cast<int>(content.size()) > content_cap) {
            content.resize(static_cast<size_t>(content_cap));
        }
        padded.insert(padded.end(), content.begin(), content.end());
        while (static_cast<int>(padded.size()) < (len - 1)) {
            padded.push_back(pad_id);
        }
        padded.push_back(eos);
        return padded;
    }

    // Fallback: padding/troncature classique.
    padded = tokens;
    
    // Tronquer si trop long
    if (padded.size() > static_cast<size_t>(len)) {
        padded.resize(len);
    }
    
    // Padding si trop court
    while (padded.size() < static_cast<size_t>(len)) {
        padded.push_back(pad_id);
    }
    
    return padded;
}

std::vector<std::vector<int>> Tokenizer::batchTokenize(
    const std::vector<std::string>& texts, 
    int max_len
) {
    std::vector<std::vector<int>> batch;
    batch.reserve(texts.size());
    
    for (const auto& text : texts) {
        auto tokens = tokenizeEnsure(text);
        auto padded = padSequence(tokens, max_len);
        batch.push_back(padded);
    }
    
    return batch;
}

void Tokenizer::printVocabStats() const {
    std::cerr << "\n📊 Statistiques du Vocabulaire" << std::endl;
    std::cerr << "================================" << std::endl;
    std::cerr << "  Taille actuelle : " << vocabSize << " / " << maxVocab << std::endl;
    std::cerr << "  Tokens spéciaux : 7 (<PAD>, <UNK>, <SEQ>, <MOD>, <MAG>, <BOS>, <EOS>)" << std::endl;
    std::cerr << "  Tokens utilisateur : " << (vocabSize > 7 ? (vocabSize - 7) : 0) << std::endl;
    std::cerr << "  Max séquence : " << maxSequenceLength << std::endl;
    
    // Afficher les 10 premiers tokens
    std::cerr << "\n  Premiers tokens :" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(10, vocab.size()); ++i) {
        std::cerr << "    [" << i << "] " << getTokenById(static_cast<int>(i)) << std::endl;
    }
}

std::unordered_map<std::string, int> Tokenizer::getTokenFrequencies(
    const std::string& text
) const {
    std::unordered_map<std::string, int> freqs;
    auto tokens = splitTokens(text);
    
    for (const auto& tok : tokens) {
        std::string normalized = normalizeToken(tok);
        if (!normalized.empty()) {
            freqs[normalized]++;
        }
    }
    
    return freqs;
}

void Tokenizer::learnBPEFromCorpus(
    const std::vector<std::string>& corpus, 
    int num_merges
) {
    std::cerr << "[BPE] Étude démarrée: " << corpus.size()
              << " texte(s), jusqu'à " << std::max(0, num_merges) << " merges." << std::endl;

    struct WordEntry {
        std::vector<std::string> pieces;
        size_t frequency = 0;
    };

    std::vector<WordEntry> words;
    std::unordered_map<std::string, size_t> word_indices;
    std::set<std::string> base_symbols;
    for (const auto& text : corpus) {
        for (const auto& raw : splitTokens(text)) {
            if (isSpecialToken(raw)) continue;
            const std::string word = normalizeToken(raw);
            if (word.empty() || !isBpeWord(word)) continue;
            auto found = word_indices.find(word);
            if (found == word_indices.end()) {
                WordEntry entry;
                entry.pieces = initialBpeSymbols(word);
                entry.frequency = 1;
                const size_t index = words.size();
                words.push_back(std::move(entry));
                word_indices.emplace(word, index);
                for (const auto& piece : words.back().pieces) base_symbols.insert(piece);
            } else {
                ++words[found->second].frequency;
            }
        }
    }
    std::cerr << "[BPE] " << words.size() << " mots distincts préparés; "
              << base_symbols.size() << " symboles de base." << std::endl;

    size_t base_symbols_added = 0;
    for (const auto& symbol : base_symbols) {
        if (vocab.find(symbol) != vocab.end()) continue;
        const size_t before = vocab.size();
        addToken(symbol);
        if (vocab.find(symbol) != vocab.end() && vocab.size() > before)
            ++base_symbols_added;
    }

    // Re-apply previously saved merges so a later corpus can extend, rather
    // than forget, the tokenizer's learned subword inventory.
    std::vector<std::pair<std::pair<std::string, std::string>, int>> ordered_merges;
    ordered_merges.reserve(bpeMerges.size());
    for (const auto& merge : bpeMerges) ordered_merges.push_back(merge);
    std::sort(ordered_merges.begin(), ordered_merges.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second < b.second;
        return a.first < b.first;
    });

    for (auto& word : words) {
        const bool has_all_base_symbols = std::all_of(
            word.pieces.begin(), word.pieces.end(),
            [&](const std::string& piece) { return vocab.find(piece) != vocab.end(); });
        if (!has_all_base_symbols) {
            word.pieces.clear();
            continue;
        }
        for (const auto& merge : ordered_merges) {
            const std::string merged = mergeBpeSymbols(merge.first.first, merge.first.second);
            if (vocab.find(merged) != vocab.end())
                applyBpeMerge(word.pieces, merge.first, merged);
        }
    }

    int next_rank = 0;
    for (const auto& merge : bpeMerges) next_rank = std::max(next_rank, merge.second + 1);
    size_t learned_merges = 0;
    const int merge_limit = std::max(0, num_merges);
    for (int merge_index = 0; merge_index < merge_limit; ++merge_index) {
        std::map<std::pair<std::string, std::string>, size_t> pair_counts;
        for (const auto& word : words) {
            if (word.pieces.empty()) continue;
            for (size_t i = 0; i + 1 < word.pieces.size(); ++i)
                pair_counts[{word.pieces[i], word.pieces[i + 1]}] += word.frequency;
        }

        auto best = pair_counts.end();
        for (auto it = pair_counts.begin(); it != pair_counts.end(); ++it) {
            if (bpeMerges.find(it->first) != bpeMerges.end()) continue;
            if (best == pair_counts.end() || it->second > best->second)
                best = it;
        }
        if (best == pair_counts.end() || best->second < 2) break;

        const auto pair = best->first;
        const std::string merged = mergeBpeSymbols(pair.first, pair.second);
        if (vocab.find(merged) == vocab.end()) {
            const size_t before = vocab.size();
            addToken(merged);
            if (vocab.find(merged) == vocab.end() || vocab.size() == before) {
                // Existing tokens are reusable; a full vocabulary with no such
                // token means this merge cannot be represented by the model.
                if (vocab.find(merged) == vocab.end()) break;
            }
        }

        bpeMerges[pair] = next_rank++;
        ++learned_merges;
        for (auto& word : words) {
            if (!word.pieces.empty()) applyBpeMerge(word.pieces, pair, merged);
        }
        if (learned_merges % 100 == 0 || learned_merges == static_cast<size_t>(merge_limit)) {
            std::cerr << "[BPE] progression " << learned_merges << "/" << merge_limit
                      << " merges; vocab=" << vocab.size() << "/" << maxVocab << std::endl;
        }
    }

    bpeReady = true;
    std::cerr << "🔤 BPE corpus étudié : " << words.size() << " mots distincts, "
              << base_symbols_added << " symboles de base ajoutés, "
              << learned_merges << " merges appris (" << bpeMerges.size()
              << " au total)." << std::endl;
}

std::vector<int> Tokenizer::tokenizeBPE(const std::string& text) const {
    std::vector<int> ids;
    const int bos = getBosId();
    const int eos = getEosId();
    const int unk = getUnkId() >= 0 ? getUnkId() : 0;
    const bool use_bos_eos = bos >= 0 && eos >= 0;
    auto tokens = splitTokens(text);
    ids.reserve(tokens.size() + (use_bos_eos ? 2 : 0));
    if (use_bos_eos) ids.push_back(bos);

    std::vector<std::pair<std::pair<std::string, std::string>, int>> ordered_merges;
    ordered_merges.reserve(bpeMerges.size());
    for (const auto& merge : bpeMerges) ordered_merges.push_back(merge);
    std::sort(ordered_merges.begin(), ordered_merges.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second < b.second;
        return a.first < b.first;
    });

    for (const auto& raw : tokens) {
        const std::string word = normalizeToken(raw);
        if (word.empty()) continue;
        const auto exact = vocab.find(word);
        if (exact != vocab.end()) {
            ids.push_back(exact->second);
            continue;
        }
        if (!bpeReady || !isBpeWord(word)) {
            ids.push_back(unk);
            continue;
        }

        auto pieces = initialBpeSymbols(word);
        for (const auto& merge : ordered_merges) {
            const std::string merged = mergeBpeSymbols(merge.first.first, merge.first.second);
            if (vocab.find(merged) != vocab.end())
                applyBpeMerge(pieces, merge.first, merged);
        }
        for (const auto& piece : pieces) {
            const auto found = vocab.find(piece);
            ids.push_back(found == vocab.end() ? unk : found->second);
        }
    }

    if (use_bos_eos) ids.push_back(eos);
    return ids;
}

size_t Tokenizer::getBPEMergeCount() const {
    return bpeMerges.size();
}

void Tokenizer::initializeUnderstandingDictionaries() {
    // Stop words (mots vides courants)
    stopWords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                 "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
                 "been", "being", "have", "has", "had", "do", "does", "did", "will",
                 "would", "should", "could", "may", "might", "must", "can", "this",
                 "that", "these", "those", "i", "you", "he", "she", "it", "we", "they"};
    
    // Verbes d'action courants
    actionVerbs = {"create", "make", "draw", "paint", "generate", "show", "display",
                   "render", "visualize", "depict", "illustrate", "portray", "design",
                   "compose", "produce", "build", "craft", "form", "shape", "construct"};
    
    // Types d'entités courantes
    entityTypes["red"] = "color"; entityTypes["blue"] = "color"; entityTypes["green"] = "color";
    entityTypes["yellow"] = "color"; entityTypes["orange"] = "color"; entityTypes["purple"] = "color";
    entityTypes["pink"] = "color"; entityTypes["white"] = "color"; entityTypes["black"] = "color";
    entityTypes["brown"] = "color"; entityTypes["gray"] = "color"; entityTypes["grey"] = "color";
    
    entityTypes["flower"] = "object"; entityTypes["rose"] = "object"; entityTypes["tree"] = "object";
    entityTypes["leaf"] = "object"; entityTypes["petal"] = "object"; entityTypes["stem"] = "object";
    entityTypes["garden"] = "place"; entityTypes["forest"] = "place"; entityTypes["mountain"] = "place";
    entityTypes["ocean"] = "place"; entityTypes["sky"] = "place"; entityTypes["sun"] = "object";
    entityTypes["moon"] = "object"; entityTypes["star"] = "object"; entityTypes["cloud"] = "object";
    
    entityTypes["beautiful"] = "modifier"; entityTypes["pretty"] = "modifier";
    entityTypes["stunning"] = "modifier"; entityTypes["gorgeous"] = "modifier";
    entityTypes["bright"] = "modifier"; entityTypes["dark"] = "modifier";
    entityTypes["large"] = "modifier"; entityTypes["small"] = "modifier";
    entityTypes["big"] = "modifier"; entityTypes["tiny"] = "modifier";
    
    // Scores de sentiment (-1.0 = négatif, 0.0 = neutre, 1.0 = positif)
    sentimentScores["beautiful"] = 0.9f; sentimentScores["ugly"] = -0.9f;
    sentimentScores["good"] = 0.7f; sentimentScores["bad"] = -0.7f;
    sentimentScores["great"] = 0.8f; sentimentScores["terrible"] = -0.8f;
    sentimentScores["amazing"] = 0.9f; sentimentScores["awful"] = -0.9f;
    sentimentScores["wonderful"] = 0.9f; sentimentScores["horrible"] = -0.9f;
    sentimentScores["excellent"] = 0.95f; sentimentScores["poor"] = -0.6f;
    sentimentScores["stunning"] = 0.9f; sentimentScores["disgusting"] = -0.9f;
    sentimentScores["bright"] = 0.5f; sentimentScores["dark"] = -0.3f;
    sentimentScores["happy"] = 0.8f; sentimentScores["sad"] = -0.7f;
    sentimentScores["fresh"] = 0.6f; sentimentScores["rotten"] = -0.8f;
}

Tokenizer::TextUnderstanding Tokenizer::analyzeText(const std::string& text) const {
    TextUnderstanding understanding;
    
    auto tokens = splitTokens(text);
    std::vector<std::string> normalizedTokens;
    
    for (const auto& tok : tokens) {
        std::string normalized = normalizeToken(tok);
        if (!normalized.empty() && stopWords.find(normalized) == stopWords.end()) {
            normalizedTokens.push_back(normalized);
        }
    }
    
    // Détection des entités et classificiation
    for (const auto& tok : normalizedTokens) {
        auto it = entityTypes.find(tok);
        if (it != entityTypes.end()) {
            if (it->second == "object" || it->second == "place") {
                understanding.entities.push_back(tok);
            } else if (it->second == "modifier") {
                understanding.modifiers.push_back(tok);
            }
        }
        
        if (actionVerbs.find(tok) != actionVerbs.end()) {
            understanding.actions.push_back(tok);
        }
    }
    
    // Sujet principal (première entité significative ou mot le plus fréquent)
    if (!understanding.entities.empty()) {
        understanding.mainSubject = understanding.entities[0];
    } else if (!normalizedTokens.empty()) {
        // Trouver le mot le plus long (souvent le plus significatif)
        understanding.mainSubject = *std::max_element(
            normalizedTokens.begin(), normalizedTokens.end(),
            [](const std::string& a, const std::string& b) { return a.length() < b.length(); }
        );
    }
    
    // Analyse de sentiment
    understanding.sentiment = analyzeSentiment(text);
    
    // Contexte
    understanding.context = inferContext(text);
    
    // Complexité
    understanding.complexity = static_cast<int>(computeTextComplexity(text));
    
    // Relations (mots adjacents significatifs)
    for (size_t i = 0; i + 1 < normalizedTokens.size(); ++i) {
        const auto& t1 = normalizedTokens[i];
        const auto& t2 = normalizedTokens[i + 1];
        
        // Créer une relation si les deux mots sont des entités ou modificateur + entité
        bool t1_entity = entityTypes.find(t1) != entityTypes.end();
        bool t2_entity = entityTypes.find(t2) != entityTypes.end();
        
        if (t1_entity && t2_entity) {
            understanding.relations.push_back({t1, t2});
        }
    }
    
    return understanding;
}

std::vector<std::string> Tokenizer::extractKeywords(const std::string& text, int topN) const {
    auto freqs = getTokenFrequencies(text);
    
    // Filtrer les stop words et trier par fréquence
    std::vector<std::pair<std::string, int>> filtered;
    for (const auto& [word, freq] : freqs) {
        if (stopWords.find(word) == stopWords.end()) {
            filtered.push_back({word, freq});
        }
    }
    
    std::sort(filtered.begin(), filtered.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    std::vector<std::string> keywords;
    for (int i = 0; i < topN && i < static_cast<int>(filtered.size()); ++i) {
        keywords.push_back(filtered[i].first);
    }
    
    return keywords;
}

std::vector<std::string> Tokenizer::detectEntities(const std::string& text) const {
    std::vector<std::string> entities;
    auto tokens = splitTokens(text);
    
    for (const auto& tok : tokens) {
        std::string normalized = normalizeToken(tok);
        auto it = entityTypes.find(normalized);
        
        if (it != entityTypes.end() && 
            (it->second == "object" || it->second == "place" || it->second == "color")) {
            entities.push_back(normalized);
        }
    }
    
    return entities;
}

std::string Tokenizer::inferContext(const std::string& text) const {
    auto entities = detectEntities(text);
    
    // Compter les catégories d'entités
    std::unordered_map<std::string, int> categoryCount;
    for (const auto& entity : entities) {
        auto it = entityTypes.find(entity);
        if (it != entityTypes.end()) {
            categoryCount[it->second]++;
        }
    }
    
    // Déterminer le contexte dominant
    if (categoryCount["color"] > 0 && categoryCount["object"] > 0) {
        return "visual_description";
    } else if (categoryCount["place"] > 0) {
        return "location";
    } else if (categoryCount["object"] > 0) {
        return "object_focused";
    } else if (!entities.empty()) {
        return "descriptive";
    }
    
    return "general";
}

float Tokenizer::computeTextComplexity(const std::string& text) const {
    auto tokens = splitTokens(text);
    if (tokens.empty()) return 0.0f;
    
    // Facteurs de complexité:
    // 1. Longueur moyenne des mots
    float avgLength = 0.0f;
    for (const auto& tok : tokens) {
        avgLength += tok.length();
    }
    avgLength /= tokens.size();
    
    // 2. Diversité du vocabulaire (ratio tokens uniques / total)
    std::unordered_set<std::string> unique;
    for (const auto& tok : tokens) {
        unique.insert(normalizeToken(tok));
    }
    float diversity = static_cast<float>(unique.size()) / tokens.size();
    
    // 3. Nombre de tokens
    float lengthFactor = std::min(10.0f, static_cast<float>(tokens.size()) / 5.0f);
    
    // Combiner les facteurs (0-10)
    float complexity = (avgLength * 0.5f) + (diversity * 5.0f) + (lengthFactor * 0.3f);
    
    return std::min(10.0f, std::max(0.0f, complexity));
}

std::unordered_map<std::string, float> Tokenizer::analyzeSentiment(const std::string& text) const {
    std::unordered_map<std::string, float> sentiment;
    auto tokens = splitTokens(text);
    
    for (const auto& tok : tokens) {
        std::string normalized = normalizeToken(tok);
        auto it = sentimentScores.find(normalized);
        
        if (it != sentimentScores.end()) {
            sentiment[normalized] = it->second;
        }
    }
    
    return sentiment;
}

