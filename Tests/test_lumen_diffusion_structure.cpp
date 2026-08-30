#include "test_utils.hpp"

#include "Models/Diffusion/LumenLatentDiffusionModel.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "Models/Vision/VAEConvModel.hpp"
#include "include/json.hpp"

#include <cmath>
#include <string>
#include <unordered_set>

using json = nlohmann::json;

class TestableLumenModel : public LumenLatentDiffusionModel {
public:
    void addTestTips(const std::vector<float>& generated_image,
                     const std::vector<unsigned char>& original_image) {
        addDiffusionVizTips(generated_image, original_image);
    }

    void addTestComparisonTips(const std::vector<float>& oracle_image,
                               const std::vector<float>& noisy_baseline_image,
                               const std::vector<float>& predicted_image,
                               int timestep) {
        addDiffusionComparisonVizTips(
            oracle_image, noisy_baseline_image, predicted_image, timestep);
    }

    std::vector<float> encodeTestImage(const std::vector<float>& image) {
        return encodeImage(image);
    }

    std::vector<float> decodeTestImage(const std::vector<float>& rgb_chw) {
        return decodeImage(rgb_chw);
    }
};

int main() {
    json cfg = ModelArchitectures::defaultConfig("lumen_diffusion");
    TASSERT_TRUE(cfg["latent_w"].get<int>() == 64);
    TASSERT_TRUE(cfg["latent_h"].get<int>() == 64);
    TASSERT_TRUE(cfg["latent_c"].get<int>() == 4);
    TASSERT_TRUE(cfg["vae_base_channels"].get<int>() == 8);
    TASSERT_TRUE(cfg["vae_use_resnet"].get<bool>());
    TASSERT_TRUE(!cfg["vae_use_attn"].get<bool>());
    TASSERT_TRUE(!cfg["vae_use_skip_connections"].get<bool>());
    TASSERT_TRUE(!cfg["vae_use_encoder_prior"].get<bool>());
    TASSERT_TRUE(cfg["vae_resnet_max_tokens"].get<int>() == 4096);
    TASSERT_TRUE(cfg["vae_decoder_upsample"].get<std::string>() == "nearest_conv");
    TASSERT_TRUE(cfg["patch_size"].get<int>() == 4);
    cfg["image_w"] = 16;
    cfg["image_h"] = 16;
    cfg["latent_w"] = 16;
    cfg["latent_h"] = 16;
    cfg["latent_c"] = 3;
    cfg["patch_size"] = 4;
    cfg["hidden_size"] = 8;
    cfg["depth"] = 1;
    cfg["mlp_ratio"] = 2.0f;
    cfg["vocab_size"] = 32;
    cfg["text_seq_len"] = 4;
    cfg["text_layers"] = 1;
    cfg["num_heads"] = 2;
    cfg["kl_beta"] = 0.5f;
    cfg["kl_warmup_steps"] = 10;
    cfg["vae_decoder_upsample"] = "nearest_conv";

    auto model = ModelArchitectures::create("lumen_diffusion", cfg);
    TASSERT_TRUE(static_cast<bool>(model));
    auto* lumen = dynamic_cast<LumenLatentDiffusionModel*>(model.get());
    TASSERT_TRUE(lumen != nullptr);
    TASSERT_TRUE(lumen->getConfig().image_w * lumen->getConfig().image_h *
                     lumen->getConfig().image_c == 768);
    TASSERT_TRUE(model->modelConfig["type"].get<std::string>() == "lumen_diffusion");
    TASSERT_TRUE(model->modelConfig["patch_size"].get<int>() == 4);
    TASSERT_TRUE(model->modelConfig["kl_beta"].get<float>() == 0.0f);
    TASSERT_TRUE(model->modelConfig["kl_warmup_steps"].get<int>() == 0);
    TASSERT_TRUE(model->modelConfig["vae_decoder_upsample"].get<std::string>() ==
                 "nearest_conv");
    TASSERT_TRUE(model->getLayerByName("lumen/text/token_embedding") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/time/input") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/dit/patch_embed") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/dit/block1/self_attention") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/dit/block1/cross_attention") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/dit/unpatchify") != nullptr);
    TASSERT_TRUE(lumen->InitVizTips());

    Model skip_decoder;
    VAEConvModel::Config skip_cfg;
    skip_cfg.image_w = 16;
    skip_cfg.image_h = 16;
    skip_cfg.image_c = 3;
    skip_cfg.latent_w = 4;
    skip_cfg.latent_h = 4;
    skip_cfg.latent_c = 8;
    skip_cfg.base_channels = 8;
    skip_cfg.use_skip_connections = true;
    VAEConvModel::buildDecoderInto(skip_decoder, skip_cfg);
    const auto* skip_up2 = skip_decoder.getLayerByName("vae_conv/dec/up2/skip_cat");
    const auto* skip_up1 = skip_decoder.getLayerByName("vae_conv/dec/up1/skip_cat");
    TASSERT_TRUE(skip_up2 != nullptr && skip_up2->inputs.size() == 2);
    TASSERT_TRUE(skip_up1 != nullptr && skip_up1->inputs.size() == 2);
    TASSERT_TRUE(skip_up2->inputs[1] == "vae_conv/encoder_skip_1");
    TASSERT_TRUE(skip_up1->inputs[1] == "vae_conv/encoder_skip_0");
    TASSERT_TRUE(skip_decoder.getLayerByName(
        "vae_conv/dec/up2/zero_enc_skip") == nullptr);

    TestableLumenModel tips_model;
    LumenLatentDiffusionModel::Config tips_cfg;
    tips_cfg.image_w = 16;
    tips_cfg.image_h = 16;
    tips_cfg.latent_w = 16;
    tips_cfg.latent_h = 16;
    tips_cfg.latent_c = 3;
    tips_cfg.patch_size = 4;
    tips_cfg.hidden_size = 8;
    tips_cfg.depth = 1;
    tips_cfg.mlp_ratio = 2.0f;
    tips_cfg.vocab_size = 32;
    tips_cfg.text_seq_len = 4;
    tips_cfg.text_layers = 1;
    tips_cfg.num_heads = 2;
    tips_model.buildFromConfig(tips_cfg);
    tips_model.setVizTapsEnabled(true);
    tips_model.setVizTapsLimits(16, 8);
    std::vector<float> generated_image(16 * 16 * 3, -1.0f);
    std::vector<unsigned char> original_image(16 * 16 * 3, 0);
    generated_image[1] = 0.0f;
    generated_image[2] = 1.0f;
    original_image[0] = 255;
    original_image[1] = 128;
    tips_model.addTestTips(generated_image, original_image);
    std::unordered_set<std::string> tip_labels;
    for (const auto& frame : tips_model.consumeVizTaps()) {
        tip_labels.insert(frame.label);
        TASSERT_TRUE(frame.w == tips_cfg.image_w);
        TASSERT_TRUE(frame.h == tips_cfg.image_h);
        TASSERT_TRUE(frame.channels == 3);
        TASSERT_TRUE(frame.pixels.size() ==
                     static_cast<size_t>(tips_cfg.image_w * tips_cfg.image_h * 3));
        if (frame.label == "diffusion_out") {
            TASSERT_TRUE(frame.pixels[0] == 0);
            TASSERT_TRUE(frame.pixels[1] == 128);
            TASSERT_TRUE(frame.pixels[2] == 255);
        } else if (frame.label == "resdiff_abs") {
            TASSERT_TRUE(frame.pixels[0] == 255);
            TASSERT_TRUE(frame.pixels[1] == 0);
            TASSERT_TRUE(frame.pixels[2] == 255);
        } else if (frame.label == "resdiff_norm") {
            TASSERT_TRUE(frame.pixels[0] == 0);
            TASSERT_TRUE(frame.pixels[1] == 128);
            TASSERT_TRUE(frame.pixels[2] == 255);
        } else if (frame.label == "resdiff_max") {
            TASSERT_TRUE(frame.pixels[0] == 0);
            TASSERT_TRUE(frame.pixels[1] == 0);
            TASSERT_TRUE(frame.pixels[2] == 255);
        } else if (frame.label == "resdiff_min") {
            TASSERT_TRUE(frame.pixels[0] == 255);
            TASSERT_TRUE(frame.pixels[1] == 0);
            TASSERT_TRUE(frame.pixels[2] == 0);
        }
    }
    TASSERT_TRUE(tip_labels.count("resdiff_abs") == 1);
    TASSERT_TRUE(tip_labels.count("resdiff_norm") == 1);
    TASSERT_TRUE(tip_labels.count("resdiff_max") == 1);
    TASSERT_TRUE(tip_labels.count("resdiff_min") == 1);
    TASSERT_TRUE(tip_labels.count("diffusion_out") == 1);

    const std::vector<float> oracle_preview(16 * 16 * 3, -1.0f);
    const std::vector<float> baseline_preview(16 * 16 * 3, 0.0f);
    const std::vector<float> predicted_preview(16 * 16 * 3, 1.0f);
    tips_model.addTestComparisonTips(
        oracle_preview, baseline_preview, predicted_preview, 50);
    const auto comparison_frames = tips_model.consumeVizTaps();
    std::vector<Model::VizFrame> comparison_triptych;
    for (const auto& frame : comparison_frames) {
        if (frame.label.find("diffusion/compare/") == 0) {
            comparison_triptych.push_back(frame);
        }
    }
    TASSERT_TRUE(comparison_triptych.size() == 3);
    TASSERT_TRUE(comparison_triptych[0].label ==
                 "diffusion/compare/A_oracle_decode_z0 | timestep=50");
    TASSERT_TRUE(comparison_triptych[1].label ==
                 "diffusion/compare/B_noisy_baseline | timestep=50");
    TASSERT_TRUE(comparison_triptych[2].label ==
                 "diffusion/compare/C_model_decode_z0_pred | timestep=50");
    TASSERT_TRUE(comparison_triptych[0].pixels[0] == 0);
    TASSERT_TRUE(comparison_triptych[1].pixels[0] == 128);
    TASSERT_TRUE(comparison_triptych[2].pixels[0] == 255);

    TestableLumenModel direct_model;
    LumenLatentDiffusionModel::Config direct_cfg = tips_cfg;
    direct_cfg.image_w = 8;
    direct_cfg.image_h = 4;
    direct_cfg.image_c = 3;
    direct_cfg.latent_w = 8;
    direct_cfg.latent_h = 4;
    direct_cfg.latent_c = 3;
    direct_model.modelConfig["external_graphs"] = {{"stale", true}};
    direct_model.modelConfig["graph_bindings"] = {{"stale", true}};
    direct_model.buildFromConfig(direct_cfg);
    TASSERT_TRUE(!direct_model.modelConfig.contains("external_graphs"));
    TASSERT_TRUE(!direct_model.modelConfig.contains("graph_bindings"));
    TASSERT_TRUE(direct_model.modelConfig["architecture"].get<std::string>() ==
                 "dit_latent_vae_conv");
    TASSERT_TRUE(direct_model.modelConfig["input_dim"].get<int>() == 8 * 4 * 3);
    TASSERT_TRUE(direct_model.modelConfig["output_dim"].get<int>() == 8 * 4 * 3);
    std::vector<float> direct_image(8 * 4 * 3);
    for (size_t index = 0; index < direct_image.size(); ++index) {
        direct_image[index] = static_cast<float>(index) / direct_image.size();
    }
    const auto direct_chw = direct_model.encodeTestImage(direct_image);
    const auto direct_round_trip = direct_model.decodeTestImage(direct_chw);
    TASSERT_TRUE(direct_round_trip == direct_image);

    direct_model.allocateParams();
    direct_model.initializeWeights("xavier", 123U);
    Optimizer train_optimizer;
    train_optimizer.type = OptimizerType::SGD;
    train_optimizer.decay_strategy = LRDecayStrategy::NONE;
    std::vector<unsigned char> train_image(8 * 4 * 3, 127);
    const auto train_stats = direct_model.trainDiffusionStep(
        train_image, "test", 123U, train_optimizer, 1e-4f);
    TASSERT_TRUE(std::isfinite(train_stats.loss));
    TASSERT_TRUE(std::isfinite(train_stats.grad_norm));
    TASSERT_TRUE(train_stats.kl_beta_effective == 0.0f);

    const std::vector<float> diagnostic_values{-1.0f, -0.25f, 0.5f, 1.0f};
    const auto diagnostics = Model::computeStepDiagnostics(
        diagnostic_values, diagnostic_values);
    TASSERT_TRUE(std::abs(diagnostics.wasserstein) < 1e-6f);
    TASSERT_TRUE(std::abs(diagnostics.entropy_diff) < 1e-6f);
    TASSERT_TRUE(std::abs(diagnostics.moment_mismatch) < 1e-6f);
    TASSERT_TRUE(std::abs(diagnostics.spatial_coherence) < 1e-6f);
    TASSERT_TRUE(std::abs(diagnostics.temporal_consistency - 1.0f) < 1e-6f);

    bool rejected_invalid_image = false;
    try {
        Optimizer optimizer;
        lumen->trainDiffusionStep({}, "test", 123U, optimizer, 1e-4f);
    } catch (const std::runtime_error&) {
        rejected_invalid_image = true;
    }
    TASSERT_TRUE(rejected_invalid_image);

    bool rejected_invalid_patch_shape = false;
    try {
        LumenLatentDiffusionModel invalid_model;
        auto invalid_cfg = tips_cfg;
        invalid_cfg.latent_w = 10;
        invalid_model.buildFromConfig(invalid_cfg);
    } catch (const std::runtime_error&) {
        rejected_invalid_patch_shape = true;
    }
    TASSERT_TRUE(rejected_invalid_patch_shape);

    return 0;
}
