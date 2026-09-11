#include "test_utils.hpp"

#include "Models/Diffusion/LumenLatentDiffusionModel.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "Models/Vision/VAEConvModel.hpp"
#include "Serialization/Serialization.hpp"
#include "include/json.hpp"

#include <cmath>
#include <filesystem>
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
    TASSERT_TRUE(cfg["vae_resnet"].get<bool>());
    TASSERT_TRUE(!cfg["vae_attention"].get<bool>());
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
    TASSERT_TRUE(model->modelConfig.contains("patch_size"));
    TASSERT_TRUE(model->modelConfig["patch_size"].is_number_integer());
    TASSERT_TRUE(model->modelConfig["patch_size"].get<int>() == 4);
    TASSERT_TRUE(model->modelConfig.contains("architecture_version"));
    TASSERT_TRUE(model->modelConfig["architecture_version"].is_number_integer());
    TASSERT_TRUE(model->modelConfig["architecture_version"].get<int>() == 2);
    TASSERT_TRUE(model->modelConfig.contains("kl_beta"));
    TASSERT_TRUE(model->modelConfig["kl_beta"].is_number());
    TASSERT_TRUE(model->modelConfig["kl_beta"].get<float>() == 0.0f);
    TASSERT_TRUE(model->modelConfig.contains("kl_warmup_steps"));
    TASSERT_TRUE(model->modelConfig["kl_warmup_steps"].is_number_integer());
    TASSERT_TRUE(model->modelConfig["kl_warmup_steps"].get<int>() == 0);
    TASSERT_TRUE(model->modelConfig["vae_decoder_upsample"].get<std::string>() ==
                 "nearest_conv");
    TASSERT_TRUE(model->getLayerByName("lumen/text/token_embedding") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/text/block1/norm1") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/text/block1/mlp_fc1") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/text/block1/mlp_gelu") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/text/block1/mlp_fc2") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/text/block1/add_mlp") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/text/final_norm") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/time/input") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/dit/patch_embed") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/dit/block1/self_attention") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/dit/block1/cross_attention") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/dit/unpatchify") != nullptr);
    TASSERT_TRUE(lumen->InitVizTips());

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

    const auto vae_checkpoint = std::filesystem::temp_directory_path() /
        "mimir_lumen_adaptive_vae.safetensors";
    std::filesystem::remove(vae_checkpoint);
    VAEConvModel adaptive_vae;
    VAEConvModel::Config adaptive_vae_cfg;
    adaptive_vae_cfg.image_w = 8;
    adaptive_vae_cfg.image_h = 8;
    adaptive_vae_cfg.image_c = 3;
    adaptive_vae_cfg.latent_w = 2;
    adaptive_vae_cfg.latent_h = 2;
    adaptive_vae_cfg.latent_c = 2;
    adaptive_vae_cfg.base_channels = 8;
    adaptive_vae_cfg.stochastic_latent = false;
    adaptive_vae_cfg.resnet = false;
    adaptive_vae_cfg.attention = false;
    adaptive_vae_cfg.use_skip_connections = false;
    adaptive_vae_cfg.enc_norm = "none";
    adaptive_vae_cfg.dec_norm = "none";
    adaptive_vae_cfg.decoder_upsample = "nearest_conv";
    adaptive_vae.buildFromConfig(adaptive_vae_cfg);
    adaptive_vae.allocateParams();
    adaptive_vae.initializeWeights("xavier", 321U);

    Mimir::Serialization::SaveOptions save_options;
    save_options.format = Mimir::Serialization::CheckpointFormat::SafeTensors;
    save_options.save_tokenizer = false;
    save_options.save_encoder = false;
    save_options.save_optimizer = false;
    std::string save_error;
    TASSERT_TRUE(Mimir::Serialization::save_checkpoint(
        adaptive_vae, vae_checkpoint.string(), save_options, &save_error));

    LumenLatentDiffusionModel adaptive_lumen;
    auto adaptive_lumen_cfg = tips_cfg;
    adaptive_lumen_cfg.image_w = 16;
    adaptive_lumen_cfg.image_h = 16;
    adaptive_lumen_cfg.latent_w = 4;
    adaptive_lumen_cfg.latent_h = 4;
    adaptive_lumen_cfg.latent_c = 3;
    adaptive_lumen_cfg.patch_size = 4;
    adaptive_lumen_cfg.vae_checkpoint = vae_checkpoint.string();
    adaptive_lumen.buildFromConfig(adaptive_lumen_cfg);

    const auto& adapted = adaptive_lumen.getConfig();
    TASSERT_TRUE(adapted.image_w == adaptive_vae_cfg.image_w);
    TASSERT_TRUE(adapted.image_h == adaptive_vae_cfg.image_h);
    TASSERT_TRUE(adapted.image_c == adaptive_vae_cfg.image_c);
    TASSERT_TRUE(adapted.latent_w == adaptive_vae_cfg.latent_w);
    TASSERT_TRUE(adapted.latent_h == adaptive_vae_cfg.latent_h);
    TASSERT_TRUE(adapted.latent_c == adaptive_vae_cfg.latent_c);
    TASSERT_TRUE(adapted.vae_stochastic_latent ==
                 adaptive_vae_cfg.stochastic_latent);
    TASSERT_TRUE(adapted.vae_decoder_upsample ==
                 adaptive_vae_cfg.decoder_upsample);
    TASSERT_TRUE(adapted.patch_size == 2);
    TASSERT_TRUE(adaptive_lumen.modelConfig["input_dim"].get<int>() == 2 * 2 * 2);
    TASSERT_TRUE(adaptive_lumen.modelConfig["image_w"].get<int>() == 8);
    TASSERT_TRUE(adaptive_lumen.modelConfig["latent_c"].get<int>() == 2);
    std::filesystem::remove(vae_checkpoint);

    const auto raw_vae_checkpoint = std::filesystem::temp_directory_path() /
        "mimir_lumen_adaptive_vae_raw";
    std::filesystem::remove_all(raw_vae_checkpoint);
    save_options.format = Mimir::Serialization::CheckpointFormat::RawFolder;
    TASSERT_TRUE(Mimir::Serialization::save_checkpoint(
        adaptive_vae, raw_vae_checkpoint.string(), save_options, &save_error));
    LumenLatentDiffusionModel raw_adaptive_lumen;
    adaptive_lumen_cfg.vae_checkpoint = raw_vae_checkpoint.string();
    raw_adaptive_lumen.buildFromConfig(adaptive_lumen_cfg);
    TASSERT_TRUE(raw_adaptive_lumen.getConfig().image_w == 8);
    TASSERT_TRUE(raw_adaptive_lumen.getConfig().latent_w == 2);
    TASSERT_TRUE(raw_adaptive_lumen.getConfig().latent_c == 2);
    TASSERT_TRUE(raw_adaptive_lumen.getConfig().patch_size == 2);
    TASSERT_TRUE(raw_adaptive_lumen.modelConfig["input_dim"].get<int>() == 2 * 2 * 2);
    std::filesystem::remove_all(raw_vae_checkpoint);

    return 0;
}
