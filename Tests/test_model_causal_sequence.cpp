#include "test_utils.hpp"
#include "Models/NLP/CausalLMModel.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "Serialization/LayerMetadata.hpp"
#include "Serialization/Serialization.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(bool value, const std::string& message) { if (!value) throw std::runtime_error(message); }
void compare(const std::vector<float>& actual, const std::vector<float>& expected,
             const std::string& label, float tolerance = 8e-5f) {
    check(actual.size() == expected.size(), label + " size");
    for (size_t i = 0; i < actual.size(); ++i)
        check(std::isfinite(actual[i]) && std::isfinite(expected[i]) &&
              std::abs(actual[i] - expected[i]) <= tolerance * std::max(1.f, std::abs(expected[i])),
              label + " index=" + std::to_string(i));
}
const Layer* find(const Model& model, const std::string& name) {
    const auto& layers = model.getLayers();
    const auto it = std::find_if(layers.begin(), layers.end(), [&](const Layer& layer) { return layer.name == name; });
    return it == layers.end() ? nullptr : &*it;
}
json fixture() {
    return {{"vocab_size", 512}, {"seq_len", 6}, {"d_model", 8}, {"num_layers", 2},
            {"num_heads", 2}, {"num_kv_heads", 1}, {"mlp_hidden", 12}, {"latent_dim", 4},
            {"padding_idx", 0}, {"dtype", "float32"}, {"local_conv_dilation", 2}};
}
void copyWeights(const Model& source, Model& destination) {
    for (auto& layer : destination.getMutableLayers()) {
        if (!layer.getWeightsSize() || !layer.shared_weights_from.empty()) continue;
        const auto* original = find(source, layer.name);
        check(original && original->getWeightsSize() == layer.getWeightsSize(), "weight mapping " + layer.name);
        std::copy_n(original->getWeights(), layer.getWeightsSize(), layer.getWeights());
    }
}
void checkGraph(const Model& model, bool local, bool fused) {
    size_t convs = 0, swiglus = 0;
    for (const auto& layer : model.getLayers()) {
        if (layer.type == "CausalDepthwiseConv1d") ++convs;
        if (layer.type == "SwiGLU") ++swiglus;
    }
    const int blocks = model.modelConfig.at("num_layers").get<int>();
    check(convs == (local ? static_cast<size_t>(blocks) : 0), "local convolution count");
    check(swiglus == (fused ? static_cast<size_t>(blocks) : 0), "fused SwiGLU count");
    for (int block = 1; block <= blocks; ++block) {
        const std::string prefix = "causal_lm/block" + std::to_string(block);
        if (local) {
            const auto* conv = find(model, prefix + "/local_conv");
            check(conv && conv->in_features == 8 && conv->out_features == 8 && conv->kernel_size == 3 &&
                  conv->dilation == model.modelConfig.at("local_conv_dilation").get<int>() &&
                  conv->params_count == 24 && !conv->use_bias && conv->causal && conv->seq_len == 6,
                  "convolution metadata");
            const auto* norm = find(model, prefix + "/local_norm");
            const auto* activation = find(model, prefix + "/local_activation");
            const auto* projection = find(model, prefix + "/local_out");
            const auto* residual = find(model, prefix + "/local_residual");
            check(norm && activation && projection && residual, "local branch missing layer");
            check(norm->inputs == std::vector<std::string>{prefix + "/post_attention"}, "local norm wrong source");
            check(conv->inputs == std::vector<std::string>{norm->output} &&
                  activation->inputs == std::vector<std::string>{conv->output} &&
                  projection->inputs == std::vector<std::string>{activation->output} &&
                  residual->inputs == std::vector<std::string>{prefix + "/post_attention", projection->output},
                  "local branch is disconnected");
        }
        const auto* gate = find(model, prefix + "/swiglu");
        check(gate && gate->type == (fused ? "SwiGLU" : "Multiply"), "FFN gate type");
        check(bool(find(model, prefix + "/gate_silu")) == !fused, "legacy SiLU gate topology");
    }
}

std::vector<float> logits(Model& model, const std::vector<int>& tokens, bool training = false) {
    return model.forwardPassNamedView({}, {{"__input__", tokens}}, training);
}
void causalPrefix(Model& model) {
    const std::vector<int> tokens{1,2,3,4,5,6}, future{1,2,3,19,20,21};
    const auto original = logits(model, tokens), changed = logits(model, future);
    check(original.size() == 6 * 512, "model logits shape");
    for (size_t i = 0; i < 3 * 512; ++i)
        check(original[i] == changed[i], "future token changed prefix logits");
}
void localDependence() {
    auto cfg = fixture(); cfg["num_layers"] = 1; cfg["latent_dim"] = 0; cfg["local_conv_dilation"] = 1;
    auto model = ModelArchitectures::create("causal_lm", cfg);
    model->allocateParams(); model->initializeWeights("xavier", 123u);
    auto* attention = model->getLayerByName("causal_lm/block1/attention");
    std::fill_n(attention->getWeights(), attention->getWeightsSize(), 0.f);
    auto* embedding = model->getLayerByName("causal_lm/token_embedding");
    std::fill_n(embedding->getWeights() + 8, 16, 0.f);
    embedding->getWeights()[8] = 1.f; embedding->getWeights()[17] = 1.f;
    auto* conv = model->getLayerByName("causal_lm/block1/local_conv");
    std::fill_n(conv->getWeights(), conv->getWeightsSize(), 0.f);
    for (int channel = 0; channel < 8; ++channel) conv->getWeights()[channel * 3 + 1] = 1.f;
    logits(*model, {1,1,1,1,1,1}); const auto before = model->getTensor("causal_lm/block1/local_conv_out");
    logits(*model, {1,2,1,1,1,1}); const auto after = model->getTensor("causal_lm/block1/local_conv_out");
    check(before.size() == 6 * 8 && after.size() == before.size(), "local tensor shape");
    double changed = 0.;
    for (size_t channel = 0; channel < 8; ++channel) changed += std::abs(before[2 * 8 + channel] - after[2 * 8 + channel]);
    check(changed > .1, "past token has no effect through the convolution");
    for (size_t i = 0; i < 2 * 8; ++i) check(before[i] == after[i], "one-lag convolution used the current/future token");
}

void trainAndCompareComposite(Model& model, const json& cfg) {
    auto explicit_cfg = cfg; explicit_cfg["fused_swiglu"] = false;
    auto composite = ModelArchitectures::create("causal_lm", explicit_cfg);
    composite->allocateParams(); copyWeights(model, *composite);
    checkGraph(*composite, true, false);
    const std::vector<int> tokens{1,2,3,4,5,6};
    model.zeroGradients(); composite->zeroGradients();
    const auto fused = logits(model, tokens, true), explicit_output = logits(*composite, tokens, true);
    compare(fused, explicit_output, "model fused/composite output");
    std::vector<float> gradient(fused.size());
    for (size_t i = 0; i < gradient.size(); ++i) gradient[i] = .01f * std::sin(float(i) * .173f);
    model.backwardPass(gradient); composite->backwardPass(gradient);
    bool nonzero = false;
    for (const auto& layer : model.getLayers()) {
        if (layer.grad_weights.empty()) continue;
        const auto* original = find(*composite, layer.name);
        check(original, "missing composite gradient layer");
        compare(layer.grad_weights, original->grad_weights, "model fused/composite gradient " + layer.name, 3e-4f);
        for (float value : layer.grad_weights) check(std::isfinite(value), "nonfinite model gradient");
        if (layer.type == "CausalDepthwiseConv1d")
            for (float value : layer.grad_weights) nonzero = nonzero || std::abs(value) > 1e-9f;
    }
    check(nonzero, "local convolution has no parameter gradient");
    auto* conv = model.getLayerByName("causal_lm/block1/local_conv");
    const std::vector<float> old(conv->getWeights(), conv->getWeights() + conv->getWeightsSize());
    Optimizer optimizer; optimizer.type = OptimizerType::SGD; optimizer.decay_strategy = LRDecayStrategy::NONE;
    model.optimizerStep(optimizer, .01f);
    bool updated = false;
    for (size_t i = 0; i < old.size(); ++i) {
        check(std::isfinite(conv->getWeights()[i]), "nonfinite updated convolution");
        updated = updated || old[i] != conv->getWeights()[i];
    }
    check(updated, "optimizer did not update convolution weights");
}

Tokenizer alphabetTokenizer() {
    Tokenizer tokenizer(512);
    for (const auto* token : {"<SYSTEM>","<USER>","<CONTEXT>","</CONTEXT>","<REFLECT>","<ASSISTANT>","<FINAL>"})
        tokenizer.addSpecialToken(token);
    std::string alphabet;
    for (char c = 'a'; c <= 'z'; ++c) { alphabet += c; alphabet += ' '; alphabet += 'a'; alphabet += c; alphabet += ' '; }
    for (char c = '0'; c <= '9'; ++c) { alphabet += c; alphabet += ' '; alphabet += 'a'; alphabet += c; alphabet += ' '; }
    alphabet += "é aé è aè ê aê à aà ç aç ù aù ô aô ï aï â aâ î aî ü aü û aû œ aœ a-b";
    tokenizer.learnBPEFromCorpus({alphabet}, 0);
    tokenizer.ensureVocabFromText(". , ; : = - ' / ( ) [ ] ! ? + _ %");
    return tokenizer;
}
void dynamicGeneration(CausalLMModel& model) {
    model.setTokenizer(alphabetTokenizer());
    const auto before_config = model.modelConfig;
    std::vector<int> sequence_lengths;
    for (const auto& layer : model.getLayers()) sequence_lengths.push_back(layer.seq_len);
    CausalLMModel::GenerationOptions options; options.max_new_tokens = 2; options.top_k = 1;
    options.temperature = 0.f; options.use_memory = false;
    const std::string prompt = "alpha beta alpha beta alpha beta alpha beta alpha beta";
    const auto generated = model.generateTokenIds(prompt, options);
    check(!generated.empty() && generated.back() == model.getTokenizer().getEosId(), "generation did not terminate with EOS");
    check(model.getTensor("causal_lm/block1/local_conv_out").size() > 6 * 8, "long prompt was cropped to training window");
    check(model.modelConfig.at("seq_len") == before_config.at("seq_len") &&
          model.modelConfig.at("input_dim") == before_config.at("input_dim") &&
          model.modelConfig.at("output_dim") == before_config.at("output_dim"), "generation did not restore model shape");
    for (size_t i = 0; i < sequence_lengths.size(); ++i)
        check(model.getLayers()[i].seq_len == sequence_lengths[i], "generation did not restore layer sequence length");
    check(generated == model.generateTokenIds(prompt, options), "generation retained stale convolution state");
    std::vector<int> tokens{1,2,3,4,5,6}, targets{2,3,4,5,6,7}, mask(6, 1);
    Optimizer optimizer; optimizer.type = OptimizerType::SGD; optimizer.decay_strategy = LRDecayStrategy::NONE;
    Model::TrainStepRequest request;
    request.int_inputs = {{"__input__", &tokens}, {"target_ids", &targets}, {"target_mask", &mask}};
    request.optimizer = &optimizer; request.learning_rate = .001f;
    const auto result = model.trainStep(request);
    check(result && std::isfinite(result->loss) && result->loss > 0.f, "native trainStep failed after long generation");
}

struct TemporaryDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("mimir_causal_sequence_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ~TemporaryDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
};
void roundtrip(Model& source, bool legacy) {
    using namespace Mimir::Serialization;
    TemporaryDirectory directory;
    const std::vector<int> tokens{1,2,3,4,5,6};
    const auto expected = logits(source, tokens);
    for (const auto format : {CheckpointFormat::RawFolder, CheckpointFormat::SafeTensors}) {
        const auto path = directory.path / (format == CheckpointFormat::RawFolder ? "raw" : "model.safetensors");
        SaveOptions save; save.format = format; save.save_encoder = save.save_tokenizer = false;
        save.include_git_info = false;
        std::string error;
        check(save_checkpoint(source, path.string(), save, &error), "save: " + error);
        auto restored = ModelArchitectures::create("causal_lm", source.modelConfig);
        checkGraph(*restored, !legacy, !legacy);
        restored->allocateParams();
        LoadOptions load; load.format = format; load.load_encoder = load.load_tokenizer = false; load.strict_mode = true;
        check(load_checkpoint(*restored, path.string(), load, &error), "strict load: " + error);
        compare(logits(*restored, tokens), expected, "checkpoint logits", 0.f);
        check(restored->getLayers().size() == source.getLayers().size(), "checkpoint topology changed");
        for (size_t i = 0; i < source.getLayers().size(); ++i) {
            const auto& before = source.getLayers()[i]; const auto& after = restored->getLayers()[i];
            check(before.type == after.type && before.name == after.name && before.inputs == after.inputs &&
                  before.output == after.output && before.shared_weights_from == after.shared_weights_from,
                  "checkpoint graph routing changed");
            if (before.type == "CausalDepthwiseConv1d")
                check(before.in_features == after.in_features && before.kernel_size == after.kernel_size &&
                      before.dilation == after.dilation && before.seq_len == after.seq_len, "checkpoint convolution geometry");
        }
    }
}
void legacyCompatibility() {
    auto old_cfg = fixture();
    old_cfg["type"] = "causal_lm";
    old_cfg.erase("local_conv_dilation");
    auto legacy = ModelArchitectures::create("causal_lm", old_cfg);
    check(legacy->modelConfig.at("local_conv_kernel") == 0 && legacy->modelConfig.at("fused_swiglu") == false,
          "old serialized config activated new operations");
    checkGraph(*legacy, false, false);
    auto explicit_cfg = old_cfg;
    explicit_cfg["causal_arch_version"] = 1; explicit_cfg["local_conv_kernel"] = 0;
    explicit_cfg["fused_swiglu"] = false; explicit_cfg["local_conv_dilation"] = 1;
    auto explicit_legacy = ModelArchitectures::create("causal_lm", explicit_cfg);
    legacy->allocateParams(); legacy->initializeWeights("xavier", 731u);
    explicit_legacy->allocateParams(); copyWeights(*legacy, *explicit_legacy);
    compare(logits(*legacy, {1,2,3,4,5,6}), logits(*explicit_legacy, {1,2,3,4,5,6}), "legacy graph parity", 0.f);
    for (const auto* key : {"causal_arch_version", "local_conv_kernel", "local_conv_dilation", "fused_swiglu"})
        legacy->modelConfig.erase(key); // Reproduce the on-disk pre-v2 configuration.
    roundtrip(*legacy, true);
}
} // namespace

int main() {
    try {
        for (const auto* backend : {"MIMIR_DISABLE_VULKAN", "MIMIR_DISABLE_OPENCL", "MIMIR_DISABLE_CUDA", "MIMIR_DISABLE_ROCM"})
            setTestEnvironment(backend, "1");
        setTestEnvironment("MIMIR_RUNTIME_AUTOTUNE", "0"); setTestEnvironment("MIMIR_PLANNER_FUSION", "0");
        setTestEnvironment("MIMIR_PLANNER_BUFFER_REUSE", "0");
        const auto defaults = ModelArchitectures::defaultConfig("causal_lm");
        check(defaults.at("causal_arch_version") == 2 && defaults.at("local_conv_kernel") == 3 &&
              defaults.at("local_conv_dilation") == 1 && defaults.at("fused_swiglu") == true, "new registry defaults");
        auto cfg = fixture();
        auto model = ModelArchitectures::create("causal_lm", cfg);
        checkGraph(*model, true, true);
        model->allocateParams(); model->initializeWeights("xavier", 419u);
        causalPrefix(*model); localDependence(); trainAndCompareComposite(*model, cfg);
        auto* causal = dynamic_cast<CausalLMModel*>(model.get()); check(causal, "native CausalLMModel missing");
        dynamicGeneration(*causal); roundtrip(*model, false); legacyCompatibility();
        std::cout << "CAUSAL_SEQUENCE_MODEL_OK\n"; return 0;
    } catch (const std::exception& error) { std::cerr << "CAUSAL_SEQUENCE_MODEL_FAILED: " << error.what() << '\n'; return 1; }
}
