#include "test_utils.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "Models/Diffusion/LumenLatentDiffusionModel.hpp"
#include "Serialization/Serialization.hpp"
#include <filesystem>
#include <cmath>
#include <chrono>

int main()
{
    auto cfg = ModelArchitectures::defaultConfig("t5");
    cfg.update({{"vocab_size", 16}, {"seq_len", 4}, {"d_model", 8}, {"num_layers", 1}, {"num_heads", 2}, {"mlp_hidden", 16}});
    auto model = ModelArchitectures::create("t5", cfg);
    model->allocateParams();
    model->initializeWeights("xavier", 42);
    const std::unordered_map<std::string, std::vector<int>> input{{"text_ids", {2, 3, 4, 0}}};
    model->zeroGradients();
    auto output = model->forwardPassNamedView({}, input, true);
    TASSERT_TRUE(output.size() == 64);
    for (float v : output)
        TASSERT_TRUE(std::isfinite(v));
    std::vector<float> gradient(64, 0);
    gradient[18] = 1;
    model->backwardPass(gradient);
    auto *bias = model->getLayerByName("t5/relative_bias");
    float norm = 0;
    size_t index = 0;
    for (size_t i = 0; i < bias->grad_weights.size(); ++i)
    {
        norm += std::abs(bias->grad_weights[i]);
        if (std::abs(bias->grad_weights[i]) > std::abs(bias->grad_weights[index]))
            index = i;
    }
    TASSERT_TRUE(norm > 1e-8f);
    const float analytic = bias->grad_weights[index], initial = bias->getWeights()[index], eps = 1e-3f;
    bias->getWeights()[index] = initial + eps;
    float plus = model->forwardPassNamedView({}, input, false)[18];
    bias->getWeights()[index] = initial - eps;
    float minus = model->forwardPassNamedView({}, input, false)[18];
    bias->getWeights()[index] = initial;
    TASSERT_NEAR(analytic, (plus - minus) / (2 * eps), 2e-3f);
    // Relative positions must distinguish permutations of the same bag of tokens.
    auto original = model->forwardPassNamedView({}, input, false);
    auto shifted = model->forwardPassNamedView({}, {{"text_ids", {0, 2, 3, 4}}}, false);
    for (size_t t = 0; t < 3; ++t)
        for (size_t v = 0; v < 16; ++v)
            TASSERT_NEAR(original[t * 16 + v], shifted[(t + 1) * 16 + v], 1e-5f);
    auto permuted = model->forwardPassNamedView({}, {{"text_ids", {4, 3, 2, 0}}}, false);
    TASSERT_TRUE(std::abs(original[18] - permuted[18]) > 1e-8f);
    const auto root = std::filesystem::temp_directory_path() /
                      ("mimir-t5-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Mimir::Serialization::SaveOptions save;
    save.format = Mimir::Serialization::CheckpointFormat::RawFolder;
    save.save_optimizer = false;
    save.save_tokenizer = false;
    save.save_encoder = false;
    std::string error;
    TASSERT_TRUE(Mimir::Serialization::save_checkpoint(*model, root.string(), save, &error));
    auto lc = ModelArchitectures::defaultConfig("lumen_diffusion");
    lc.update({{"image_w", 8}, {"image_h", 8}, {"latent_w", 4}, {"latent_h", 4}, {"latent_c", 4}, {"patch_size", 2}, {"hidden_size", 8}, {"depth", 1}, {"mlp_ratio", 2.0}, {"num_heads", 2}, {"vocab_size", 16}, {"text_seq_len", 4}, {"text_vae_checkpoint", ""}, {"text_t5_checkpoint", root.string()}});
    auto lumen = ModelArchitectures::create("lumen_diffusion", lc);
    lumen->allocateParams();
    lumen->initializeWeights("xavier", 42);
    const auto *imported = lumen->getLayerByName("lumen/t5/t5/shared");
    const auto *source = model->getLayerByName("t5/shared");
    TASSERT_TRUE(imported && source);
    for (size_t i = 0; i < source->getWeightsSize(); ++i)
        TASSERT_NEAR(imported->getWeights()[i], source->getWeights()[i], 0);
    TASSERT_TRUE(!lumen->modelConfig.at("dynamic_text_seq_len").get<bool>());
    lumen->zeroGradients();
    auto result = lumen->forwardPassNamedView({{"__input__", std::vector<float>(64, 0.1f)},
                                               {"timestep", std::vector<float>(8, 0.2f)}},
                                              {{"text_ids", {2, 3, 4, 0}}}, true);
    TASSERT_TRUE(!result.empty());
    for (float v : result)
        TASSERT_TRUE(std::isfinite(v));
    // Frozen imported encoder survives diffusion optimizer updates.
    lumen->backwardPass(std::vector<float>(result.size(), 0.01f));
    Optimizer opt;
    lumen->optimizerStep(opt, 0.001f);
    for (size_t i = 0; i < source->getWeightsSize(); ++i)
        TASSERT_NEAR(imported->getWeights()[i], source->getWeights()[i], 0);
    std::filesystem::remove_all(root);
}
