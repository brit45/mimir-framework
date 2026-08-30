#include "LumenLatentDiffusionModel.hpp"
#include "Models/Vision/VAEConvModel.hpp"
#include "Serialization/Serialization.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>

namespace {

size_t parameterCount(int out_channels, int in_channels, int kernel_size) {
    return static_cast<size_t>(out_channels) * static_cast<size_t>(in_channels) *
           static_cast<size_t>(kernel_size) * static_cast<size_t>(kernel_size);
}

} // namespace

LumenLatentDiffusionModel::LumenLatentDiffusionModel() {
    setModelName("LumenLatentDiffusionModel");
    setHasEncoder(false);
}

static bool graphProduces(const Model& model, const std::string& tensor_name) {
    return std::any_of(
        model.getLayers().begin(), model.getLayers().end(),
        [&](const Layer& layer) { return layer.output == tensor_name; });
}

static std::string resolveVaeEncoderOutput(const Model& model) {
    for (const char* tensor_name : {"vae_conv/z_biased", "vae_conv/z"}) {
        if (graphProduces(model, tensor_name)) return tensor_name;
    }
    throw std::runtime_error(
        "LumenLatentDiffusionModel: VAE encoder graph has no usable latent output");
}

static nlohmann::json external_execution_graph(
    const Model& model,
    const nlohmann::json& bindings
) {
    nlohmann::json nodes = nlohmann::json::array();
    for (const Layer& layer : model.getLayers()) {
        nodes.push_back({
            {"name", layer.name},
            {"type", layer.type},
            {"inputs", layer.inputs},
            {"output", layer.output},
        });
    }
    return {
        {"model_name", model.getModelName()},
        {"bindings", bindings},
        {"nodes", std::move(nodes)},
    };
}

bool LumenLatentDiffusionModel::InitVizTips() {
    clearVizTipsRegistry();
    clearVizTaps();
    registerVizTip("lumen/text/token_embedding", "Conditioning/tokens");
    registerVizTip("lumen/text/add_position", "Conditioning/position");
    registerVizTip("lumen/time/projection", "Diffusion/timestep");
    registerVizTip("lumen/dit/patch_embed", "DiT/patches");
    registerVizTip("lumen/dit/block1/self_attention", "DiT/self_attention");
    registerVizTip("lumen/dit/block1/cross_attention", "DiT/cross_attention");
    registerVizTip("lumen/dit/unpatchify", "DiT/epsilon_rgb");
    return true;
}

bool LumenLatentDiffusionModel::UpdateVizTips(const Layer& layer, VizFrame& frame) {
    if (Model::UpdateVizTips(layer, frame)) return true;
    if (layer.name.empty()) return false;

    auto attach = [&](const std::string& label) {
        if (frame.label.empty()) frame.label = label;
        else frame.label += " | " + label;
        return true;
    };
    const std::string& name = layer.name;
    if (name.find("lumen/text/") == 0) return attach("Conditioning/text");
    if (name.find("lumen/time/") == 0) return attach("Diffusion/time");
    if (name.find("lumen/dit/") == 0) return attach("Diffusion Transformer");
    return false;
}

void LumenLatentDiffusionModel::addDiffusionVizTips(
    const std::vector<float>& generated_image,
    const std::vector<unsigned char>& original_image
) {
    if (!isVizTapsEnabled() || cfg_.image_c != 3) return;
    const size_t image_size = static_cast<size_t>(cfg_.image_w) * cfg_.image_h * 3;
    if (generated_image.size() != image_size || original_image.size() != image_size) return;

    std::vector<uint8_t> generated(image_size);
    std::vector<uint8_t> absolute(image_size);
    std::vector<uint8_t> normalized(image_size);
    std::vector<uint8_t> positive(image_size);
    std::vector<uint8_t> negative(image_size);
    for (size_t index = 0; index < image_size; ++index) {
        const float value = std::clamp(generated_image[index] * 0.5f + 0.5f, 0.0f, 1.0f);
        generated[index] = static_cast<uint8_t>(std::lround(value * 255.0f));
        const int delta = static_cast<int>(generated[index]) -
                          static_cast<int>(original_image[index]);
        absolute[index] = static_cast<uint8_t>(std::abs(delta));
        normalized[index] = static_cast<uint8_t>(std::lround(127.5f + 0.5f * delta));
        positive[index] = static_cast<uint8_t>(std::max(0, delta));
        negative[index] = static_cast<uint8_t>(std::max(0, -delta));
    }

    auto add_frame = [&](const char* label, const std::vector<uint8_t>& pixels) {
        VizFrame frame;
        frame.pixels = pixels;
        frame.w = cfg_.image_w;
        frame.h = cfg_.image_h;
        frame.channels = 3;
        frame.label = label;
        addVizTapFrame(std::move(frame));
    };

    add_frame("resdiff_abs", absolute);
    add_frame("resdiff_norm", normalized);
    add_frame("resdiff_max", positive);
    add_frame("resdiff_min", negative);
    add_frame("diffusion_out", generated);
}

void LumenLatentDiffusionModel::addDiffusionComparisonVizTips(
    const std::vector<float>& oracle_image,
    const std::vector<float>& noisy_baseline_image,
    const std::vector<float>& predicted_image,
    int timestep
) {
    if (!isVizTapsEnabled() || cfg_.image_c != 3) return;
    const size_t image_size = static_cast<size_t>(cfg_.image_w) * cfg_.image_h * 3;
    if (oracle_image.size() != image_size ||
        noisy_baseline_image.size() != image_size ||
        predicted_image.size() != image_size) {
        return;
    }

    auto add_frame = [&](const char* name, const std::vector<float>& image) {
        VizFrame frame;
        frame.pixels.resize(image_size);
        for (size_t index = 0; index < image_size; ++index) {
            const float normalized = std::clamp(image[index] * 0.5f + 0.5f, 0.0f, 1.0f);
            frame.pixels[index] = static_cast<uint8_t>(std::lround(normalized * 255.0f));
        }
        frame.w = cfg_.image_w;
        frame.h = cfg_.image_h;
        frame.channels = 3;
        frame.label = std::string("diffusion/compare/") + name +
                      " | timestep=" + std::to_string(timestep);
        addVizTapFrame(std::move(frame));
    };

    add_frame("A_oracle_decode_z0", oracle_image);
    add_frame("B_noisy_baseline", noisy_baseline_image);
    add_frame("C_model_decode_z0_pred", predicted_image);
}

void LumenLatentDiffusionModel::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    if (cfg_.image_w <= 0 || cfg_.image_h <= 0 || cfg_.image_c <= 0) {
        throw std::runtime_error("LumenLatentDiffusionModel: invalid image dimensions");
    }
    if (cfg_.latent_w <= 0 || cfg_.latent_h <= 0 || cfg_.latent_c <= 0) {
        throw std::runtime_error("LumenLatentDiffusionModel: invalid latent dimensions");
    }
    if (cfg_.patch_size <= 0 || cfg_.latent_w % cfg_.patch_size != 0 ||
        cfg_.latent_h % cfg_.patch_size != 0) {
        throw std::runtime_error(
            "LumenLatentDiffusionModel: latent dimensions must be divisible by patch_size");
    }
    if (modelConfig.is_object()) {
        modelConfig.erase("external_graphs");
        modelConfig.erase("graph_bindings");
    }
    vae_encoder_.reset();
    vae_decoder_.reset();
    vae_encoder_output_.clear();
    vae_skip_bindings_.clear();
    vae_encoder_skips_.clear();
    if (!cfg_.vae_checkpoint.empty()) {
        VAEConvModel::Config vae_cfg;
        vae_cfg.image_w = cfg_.image_w;
        vae_cfg.image_h = cfg_.image_h;
        vae_cfg.image_c = cfg_.image_c;
        vae_cfg.latent_w = cfg_.latent_w;
        vae_cfg.latent_h = cfg_.latent_h;
        vae_cfg.latent_c = cfg_.latent_c;
        vae_cfg.base_channels = cfg_.vae_base_channels;
        vae_cfg.stochastic_latent = cfg_.vae_stochastic_latent;
        vae_cfg.use_attention = cfg_.vae_use_resnet;
        vae_cfg.use_attn = cfg_.vae_use_attn;
        vae_cfg.use_skip_connections = cfg_.vae_use_skip_connections;
        vae_cfg.use_encoder_prior = cfg_.vae_use_encoder_prior;
        vae_cfg.enc_norm = cfg_.vae_enc_norm;
        vae_cfg.dec_norm = cfg_.vae_dec_norm;
        vae_cfg.decoder_upsample = cfg_.vae_decoder_upsample;
        vae_cfg.enc_gn_groups = cfg_.vae_enc_gn_groups;
        vae_cfg.dec_gn_groups = cfg_.vae_dec_gn_groups;
        vae_cfg.attn_heads = cfg_.vae_attn_heads;
        vae_cfg.attn_max_tokens = cfg_.vae_attn_max_tokens;
        vae_cfg.resnet_max_tokens = cfg_.vae_resnet_max_tokens;
        vae_cfg.d_model = 1024;
        vae_cfg.text_cond = false;

        vae_encoder_ = std::make_unique<VAEConvModel>();
        static_cast<VAEConvModel&>(*vae_encoder_).buildFromConfig(vae_cfg);
        vae_encoder_->allocateParams();

        vae_decoder_ = std::make_unique<Model>();
        VAEConvModel::buildDecoderInto(*vae_decoder_, vae_cfg);
        vae_decoder_->allocateParams();

        Mimir::Serialization::LoadOptions load_options;
        load_options.format = std::filesystem::is_directory(cfg_.vae_checkpoint)
            ? Mimir::Serialization::CheckpointFormat::RawFolder
            : Mimir::Serialization::CheckpointFormat::SafeTensors;
        load_options.load_optimizer = false;
        load_options.load_tokenizer = false;
        load_options.load_encoder = false;
        load_options.validate_checksums = true;
        load_options.apply_model_name = false;
        load_options.apply_model_config = false;

        std::string load_error;
        // A full VAE checkpoint also contains decoder and optimizer tensors that
        // do not belong to this encoder-only graph. Known tensors still undergo
        // checksum, dtype and shape validation in non-strict mode.
        load_options.strict_mode = false;
        if (!Mimir::Serialization::load_checkpoint(
                *vae_encoder_, cfg_.vae_checkpoint, load_options, &load_error)) {
            throw std::runtime_error(
                "LumenLatentDiffusionModel: cannot load VAE encoder: " + load_error);
        }
        load_options.strict_mode = false;
        if (!Mimir::Serialization::load_checkpoint(
                *vae_decoder_, cfg_.vae_checkpoint, load_options, &load_error)) {
            throw std::runtime_error(
                "LumenLatentDiffusionModel: cannot load VAE decoder: " + load_error);
        }
        vae_encoder_->freezeParameters();
        vae_decoder_->freezeParameters();
        vae_encoder_output_ = resolveVaeEncoderOutput(*vae_encoder_);
        if (cfg_.vae_use_skip_connections &&
            vae_encoder_->modelConfig.contains("encoder_skip_outputs")) {
            for (const auto& skip :
                 vae_encoder_->modelConfig["encoder_skip_outputs"]) {
                VaeSkipBinding binding;
                binding.decoder_input = skip.value("decoder_input", "");
                binding.encoder_output = skip.value("encoder_output", "");
                binding.values = static_cast<size_t>(skip.value("channels", 0)) *
                    static_cast<size_t>(skip.value("height", 0)) *
                    static_cast<size_t>(skip.value("width", 0));
                if (binding.decoder_input.empty() ||
                    binding.encoder_output.empty() || binding.values == 0) {
                    throw std::runtime_error(
                        "LumenLatentDiffusionModel: invalid VAE skip binding");
                }
                vae_skip_bindings_.push_back(std::move(binding));
            }
        }
        if (!graphProduces(*vae_decoder_, "x")) {
            throw std::runtime_error(
                "LumenLatentDiffusionModel: VAE decoder graph has no image output 'x'");
        }
    }

    setHasEncoder(false);
    buildInto(*this, cfg_);

    if (vae_encoder_ && vae_decoder_) {
        nlohmann::json decoder_graph_inputs = {{"latent", "__input__"}};
        nlohmann::json decoder_skip_inputs = nlohmann::json::array();
        for (const auto& binding : vae_skip_bindings_) {
            decoder_graph_inputs[binding.decoder_input] = binding.decoder_input;
            decoder_skip_inputs.push_back(binding.decoder_input);
        }
        modelConfig["external_graphs"]["vae_encoder"] =
            external_execution_graph(*vae_encoder_, {
                {"inputs", {{"image", "__input__"}}},
                {"outputs", {{"latent", vae_encoder_output_}}},
                {"feeds", {{"model", "lumen"}, {"input", "__input__"}}},
            });
        modelConfig["external_graphs"]["vae_decoder"] =
            external_execution_graph(*vae_decoder_, {
                {"inputs", decoder_graph_inputs},
                {"outputs", {{"image", "x"}}},
                {"fed_by", {{"model", "lumen"}, {"output", "x"}}},
            });
        modelConfig["vae_skip_bindings"] = nlohmann::json::array();
        for (const auto& binding : vae_skip_bindings_) {
            modelConfig["vae_skip_bindings"].push_back({
                {"encoder_output", binding.encoder_output},
                {"decoder_input", binding.decoder_input},
                {"values", binding.values},
            });
        }
        modelConfig["graph_bindings"] = {
            {"image_encoder", {
                {"model", "vae_encoder"},
                {"input", "__input__"},
                {"output", vae_encoder_output_},
            }},
            {"denoiser", {
                {"model", "lumen"},
                {"input", "__input__"},
                {"output", "x"},
            }},
            {"image_decoder", {
                {"model", "vae_decoder"},
                {"input", "__input__"},
                {"skip_inputs", decoder_skip_inputs},
                {"output", "x"},
            }},
        };
    }
}

LumenLatentDiffusionModel::TrainStats LumenLatentDiffusionModel::trainDiffusionStep(
    const std::vector<unsigned char>& rgb_image,
    const std::string& prompt,
    unsigned int seed,
    Optimizer& optimizer,
    float learning_rate
) {
    const size_t image_values = static_cast<size_t>(cfg_.image_w) *
                                static_cast<size_t>(cfg_.image_h) *
                                static_cast<size_t>(cfg_.image_c);
    if (rgb_image.size() != image_values) {
        throw std::runtime_error(
            "LumenLatentDiffusionModel: training image size mismatch; expected " +
            std::to_string(image_values) + " RGB values");
    }
    if (!std::isfinite(learning_rate) || learning_rate <= 0.0f) {
        throw std::runtime_error("LumenLatentDiffusionModel: learning rate must be positive");
    }
    std::vector<float> image(image_values);
    for (size_t index = 0; index < image_values; ++index) {
        image[index] = static_cast<float>(rgb_image[index]) / 127.5f - 1.0f;
    }
    std::vector<float> clean_latent = encodeImage(image);
    const size_t latent_values = static_cast<size_t>(cfg_.latent_w) *
                                 cfg_.latent_h * cfg_.latent_c;
    if (clean_latent.size() != latent_values) {
        throw std::runtime_error("LumenLatentDiffusionModel: training latent tensor size mismatch");
    }
    std::mt19937 random(seed);
    std::uniform_int_distribution<int> timestep_distribution(0, std::max(1, cfg_.diffusion_steps) - 1);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const int timestep = timestep_distribution(random);

    float alpha_bar = 1.0f;
    const int total_steps = std::max(2, cfg_.diffusion_steps);
    const float beta_start = std::clamp(cfg_.beta_start, 1e-7f, 0.999f);
    const float beta_end = std::clamp(cfg_.beta_end, beta_start, 0.999f);
    for (int index = 0; index <= timestep; ++index) {
        const float ratio = static_cast<float>(index) / static_cast<float>(total_steps - 1);
        alpha_bar *= 1.0f - (beta_start + ratio * (beta_end - beta_start));
    }

    std::vector<float> noise(latent_values);
    std::vector<float> noisy_latent(latent_values);
    const float signal_scale = std::sqrt(std::max(alpha_bar, 1e-8f));
    const float noise_scale = std::sqrt(std::max(0.0f, 1.0f - alpha_bar));
    for (size_t index = 0; index < latent_values; ++index) {
        noise[index] = normal(random);
        noisy_latent[index] = signal_scale * clean_latent[index] + noise_scale * noise[index];
    }

    const int context_dim = std::max(8, cfg_.hidden_size);
    const auto step_stats = trainStepNamed(
        {{"__input__", noisy_latent},
         {"timestep", timestepEmbedding(timestep, context_dim)}},
        {{"text_ids", promptIds(prompt)}},
        noise,
        optimizer,
        learning_rate,
        0.0f,
        0);
    if (isVizTapsEnabled()) {
        const int preview_timestep = std::clamp(
            modelConfig.value("preview_timestep", cfg_.preview_timestep),
            0,
            total_steps - 1);
        float preview_alpha_bar = 1.0f;
        for (int index = 0; index <= preview_timestep; ++index) {
            const float ratio = static_cast<float>(index) /
                                static_cast<float>(total_steps - 1);
            preview_alpha_bar *=
                1.0f - (beta_start + ratio * (beta_end - beta_start));
        }
        const float preview_signal_scale =
            std::sqrt(std::max(preview_alpha_bar, 1e-8f));
        const float preview_noise_scale =
            std::sqrt(std::max(0.0f, 1.0f - preview_alpha_bar));
        std::vector<float> preview_noisy_latent(latent_values);
        for (size_t index = 0; index < latent_values; ++index) {
            preview_noisy_latent[index] =
                preview_signal_scale * clean_latent[index] +
                preview_noise_scale * noise[index];
        }
        const std::vector<float> preview_prediction = forwardPassNamed(
            {{"__input__", preview_noisy_latent},
             {"timestep", timestepEmbedding(preview_timestep, context_dim)}},
            {{"text_ids", promptIds(prompt)}},
            true);
        if (preview_prediction.size() == preview_noisy_latent.size()) {
            std::vector<float> predicted_clean(preview_prediction.size());
            std::vector<float> noisy_baseline(preview_prediction.size());
            for (size_t index = 0; index < preview_prediction.size(); ++index) {
                predicted_clean[index] =
                    (preview_noisy_latent[index] -
                     preview_noise_scale * preview_prediction[index]) /
                    preview_signal_scale;
                noisy_baseline[index] =
                    preview_noisy_latent[index] / preview_signal_scale;
            }
            const std::vector<float> oracle_image = decodeImage(clean_latent, true);
            const std::vector<float> noisy_baseline_image = decodeImage(noisy_baseline, true);
            const std::vector<float> predicted_image = decodeImage(predicted_clean, true);
            addDiffusionComparisonVizTips(
                oracle_image,
                noisy_baseline_image,
                predicted_image,
                preview_timestep);
            addDiffusionVizTips(predicted_image, rgb_image);
        }
    }

    TrainStats stats;
    stats.loss = step_stats.loss;
    stats.mse = step_stats.mse;
    stats.kl = step_stats.kl_divergence;
    stats.kl_beta_effective = step_stats.kl_beta_effective;
    stats.grad_norm = step_stats.grad_norm;
    stats.grad_max_abs = step_stats.grad_max_abs;
    stats.wasserstein = step_stats.wasserstein;
    stats.entropy_diff = step_stats.entropy_diff;
    stats.moment_mismatch = step_stats.moment_mismatch;
    stats.spatial_coherence = step_stats.spatial_coherence;
    stats.temporal_consistency = step_stats.temporal_consistency;
    stats.timestep = timestep;
    return stats;
}

LumenLatentDiffusionModel::TrainStats LumenLatentDiffusionModel::validateDiffusionStep(
    const std::vector<unsigned char>& rgb_image,
    const std::string& prompt,
    unsigned int seed
) {
    const size_t image_values = static_cast<size_t>(cfg_.image_w) *
                                static_cast<size_t>(cfg_.image_h) *
                                static_cast<size_t>(cfg_.image_c);
    if (rgb_image.size() != image_values) {
        throw std::runtime_error(
            "LumenLatentDiffusionModel: validation image size mismatch; expected " +
            std::to_string(image_values) + " values");
    }
    std::vector<float> image(image_values);
    for (size_t index = 0; index < image_values; ++index) {
        image[index] = static_cast<float>(rgb_image[index]) / 127.5f - 1.0f;
    }
    std::vector<float> clean_latent = encodeImage(image);
    const size_t latent_values = static_cast<size_t>(cfg_.latent_w) *
                                 cfg_.latent_h * cfg_.latent_c;
    if (clean_latent.size() != latent_values) {
        throw std::runtime_error("LumenLatentDiffusionModel: validation latent tensor size mismatch");
    }
    std::mt19937 random(seed);
    std::uniform_int_distribution<int> timestep_distribution(
        0, std::max(1, cfg_.diffusion_steps) - 1);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const int timestep = timestep_distribution(random);
    const int total_steps = std::max(2, cfg_.diffusion_steps);
    const float beta_start = std::clamp(cfg_.beta_start, 1e-7f, 0.999f);
    const float beta_end = std::clamp(cfg_.beta_end, beta_start, 0.999f);
    float alpha_bar = 1.0f;
    for (int index = 0; index <= timestep; ++index) {
        const float ratio = static_cast<float>(index) / static_cast<float>(total_steps - 1);
        alpha_bar *= 1.0f - (beta_start + ratio * (beta_end - beta_start));
    }

    std::vector<float> noise(latent_values);
    std::vector<float> noisy_latent(latent_values);
    const float signal_scale = std::sqrt(std::max(alpha_bar, 1e-8f));
    const float noise_scale = std::sqrt(std::max(0.0f, 1.0f - alpha_bar));
    for (size_t index = 0; index < latent_values; ++index) {
        noise[index] = normal(random);
        noisy_latent[index] = signal_scale * clean_latent[index] + noise_scale * noise[index];
    }

    const int context_dim = std::max(8, cfg_.hidden_size);
    const std::vector<float> prediction = forwardPassNamed(
        {{"__input__", noisy_latent},
         {"timestep", timestepEmbedding(timestep, context_dim)}},
        {{"text_ids", promptIds(prompt)}}, false);
    if (prediction.size() != noise.size()) {
        throw std::runtime_error("LumenLatentDiffusionModel: validation output size mismatch");
    }
    std::vector<float> predicted_clean(prediction.size());
    std::vector<float> noisy_baseline(prediction.size());
    for (size_t index = 0; index < prediction.size(); ++index) {
        predicted_clean[index] =
            (noisy_latent[index] - noise_scale * prediction[index]) / signal_scale;
        noisy_baseline[index] = noisy_latent[index] / signal_scale;
    }
    const std::vector<float> reconstructed = decodeImage(predicted_clean, true);
    if (isVizTapsEnabled()) {
        const std::vector<float> oracle_image = decodeImage(clean_latent, true);
        const std::vector<float> noisy_baseline_image = decodeImage(noisy_baseline, true);
        addDiffusionComparisonVizTips(
            oracle_image,
            noisy_baseline_image,
            reconstructed,
            timestep);
        addDiffusionVizTips(reconstructed, rgb_image);
    }

    double squared_error = 0.0;
    for (size_t index = 0; index < noise.size(); ++index) {
        const double difference = static_cast<double>(prediction[index]) - noise[index];
        squared_error += difference * difference;
    }
    double reconstruction_abs_error = 0.0;
    double reconstruction_squared_error = 0.0;
    for (size_t index = 0; index < reconstructed.size(); ++index) {
        const double generated = std::clamp(
            static_cast<double>(reconstructed[index]) * 0.5 + 0.5, 0.0, 1.0);
        const double original = static_cast<double>(rgb_image[index]) / 255.0;
        const double difference = generated - original;
        reconstruction_abs_error += std::abs(difference);
        reconstruction_squared_error += difference * difference;
    }
    TrainStats stats;
    stats.loss = static_cast<float>(squared_error / static_cast<double>(noise.size()));
    stats.mse = stats.loss;
    const StepStats diagnostics = computeStepDiagnostics(prediction, noise);
    stats.kl = diagnostics.kl_divergence;
    stats.wasserstein = diagnostics.wasserstein;
    stats.entropy_diff = diagnostics.entropy_diff;
    stats.moment_mismatch = diagnostics.moment_mismatch;
    stats.spatial_coherence = diagnostics.spatial_coherence;
    stats.temporal_consistency = diagnostics.temporal_consistency;
    stats.reconstruction_mae = static_cast<float>(
        reconstruction_abs_error / static_cast<double>(reconstructed.size()));
    stats.reconstruction_mse = static_cast<float>(
        reconstruction_squared_error / static_cast<double>(reconstructed.size()));
    stats.timestep = timestep;
    return stats;
}

void LumenLatentDiffusionModel::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("LumenLatentDiffusionModel");
    model.modelConfig["type"] = "lumen_diffusion";
    model.modelConfig["task"] = "text_to_image_rgb_diffusion_transformer";

    const int latent_w = std::max(1, cfg.latent_w);
    const int latent_h = std::max(1, cfg.latent_h);
    const int latent_c = std::max(1, cfg.latent_c);
    const int patch_size = std::max(1, cfg.patch_size);
    const int patch_w = latent_w / patch_size;
    const int patch_h = latent_h / patch_size;
    const int patch_tokens = patch_w * patch_h;
    const int patch_dim = latent_c * patch_size * patch_size;
    const int depth = std::max(1, cfg.depth);
    const int context_dim = std::max(8, cfg.hidden_size);
    const int mlp_hidden = std::max(context_dim,
        static_cast<int>(std::lround(context_dim * std::max(1.0f, cfg.mlp_ratio))));
    const int text_len = std::max(1, cfg.text_seq_len);
    const int vocab = std::max(8, cfg.vocab_size);
    int heads = std::max(1, std::min(cfg.num_heads, context_dim));
    while (heads > 1 && context_dim % heads != 0) --heads;

    model.modelConfig["image_w"] = cfg.image_w;
    model.modelConfig["image_h"] = cfg.image_h;
    model.modelConfig["image_c"] = cfg.image_c;
    model.modelConfig["latent_w"] = latent_w;
    model.modelConfig["latent_h"] = latent_h;
    model.modelConfig["latent_c"] = latent_c;
    model.modelConfig["architecture"] = "dit_latent_vae_conv";
    model.modelConfig["patch_size"] = patch_size;
    model.modelConfig["patch_tokens"] = patch_tokens;
    model.modelConfig["hidden_size"] = context_dim;
    model.modelConfig["mlp_ratio"] = cfg.mlp_ratio;
    model.modelConfig["depth"] = depth;
    model.modelConfig["context_dim"] = context_dim;
    model.modelConfig["vocab_size"] = vocab;
    model.modelConfig["text_seq_len"] = text_len;
    model.modelConfig["text_layers"] = std::max(1, cfg.text_layers);
    model.modelConfig["num_heads"] = heads;
    model.modelConfig["diffusion_steps"] = std::max(2, cfg.diffusion_steps);
    model.modelConfig["beta_start"] = cfg.beta_start;
    model.modelConfig["beta_end"] = cfg.beta_end;
    model.modelConfig["preview_timestep"] = std::clamp(
        cfg.preview_timestep, 0, std::max(1, cfg.diffusion_steps) - 1);
    model.modelConfig["kl_beta"] = 0.0f;
    model.modelConfig["kl_warmup_steps"] = 0;
    model.modelConfig["vae_scale"] = cfg.vae_scale;
    model.modelConfig["vae_shift"] = cfg.vae_shift;
    model.modelConfig["vae_decoder_upsample"] = cfg.vae_decoder_upsample;
    model.modelConfig["vae_calibrated"] =
        std::isfinite(cfg.vae_scale) && cfg.vae_scale > 0.0f;
    model.modelConfig["input_dim"] = latent_w * latent_h * latent_c;
    model.modelConfig["output_dim"] = latent_w * latent_h * latent_c;

    auto addConv = [&](const std::string& name,
                       const std::string& input,
                       const std::string& output,
                       int in_channels,
                       int out_channels,
                       int height,
                       int width,
                       int kernel,
                       int stride,
                       int padding) {
        model.push(name, "Conv2d", parameterCount(out_channels, in_channels, kernel));
        if (auto* layer = model.getLayerByName(name)) {
            layer->inputs = {input};
            layer->output = output;
            layer->in_channels = in_channels;
            layer->out_channels = out_channels;
            layer->input_height = height;
            layer->input_width = width;
            layer->kernel_size = kernel;
            layer->stride = stride;
            layer->padding = padding;
            layer->use_bias = false;
        }
    };

    auto addActivation = [&](const std::string& name,
                             const std::string& input,
                             const std::string& output) {
        model.push(name, "SiLU", 0);
        if (auto* layer = model.getLayerByName(name)) {
            layer->inputs = {input};
            layer->output = output;
        }
    };

    model.push("lumen/text/token_embedding", "Embedding",
               static_cast<size_t>(vocab) * static_cast<size_t>(context_dim));
    if (auto* layer = model.getLayerByName("lumen/text/token_embedding")) {
        layer->inputs = {"text_ids"};
        layer->output = "lumen/text/tokens";
        layer->vocab_size = vocab;
        layer->embed_dim = context_dim;
        layer->seq_len = text_len;
        layer->padding_idx = 0;
    }

    model.push("lumen/text/position", "Constant",
               static_cast<size_t>(text_len) * static_cast<size_t>(context_dim));
    if (auto* layer = model.getLayerByName("lumen/text/position")) {
        layer->output = "lumen/text/position_values";
        layer->seq_len = text_len;
        layer->embed_dim = context_dim;
    }
    model.push("lumen/text/add_position", "Add", 0);
    if (auto* layer = model.getLayerByName("lumen/text/add_position")) {
        layer->inputs = {"lumen/text/tokens", "lumen/text/position_values"};
        layer->output = "lumen/text/context0";
    }

    std::string context = "lumen/text/context0";
    for (int index = 0; index < std::max(1, cfg.text_layers); ++index) {
        const std::string prefix = "lumen/text/block" + std::to_string(index + 1);
        model.push(prefix + "/norm", "LayerNorm", static_cast<size_t>(2 * context_dim));
        if (auto* layer = model.getLayerByName(prefix + "/norm")) {
            layer->inputs = {context};
            layer->output = prefix + "/norm_out";
            layer->in_features = context_dim;
            layer->affine = true;
            layer->use_bias = true;
        }
        model.push(prefix + "/attention", "SelfAttention",
                   static_cast<size_t>(4) * static_cast<size_t>(context_dim) *
                       static_cast<size_t>(context_dim));
        if (auto* layer = model.getLayerByName(prefix + "/attention")) {
            layer->inputs = {prefix + "/norm_out"};
            layer->output = prefix + "/attention_out";
            layer->seq_len = text_len;
            layer->embed_dim = context_dim;
            layer->num_heads = heads;
            layer->causal = false;
        }
        model.push(prefix + "/add", "Add", 0);
        if (auto* layer = model.getLayerByName(prefix + "/add")) {
            layer->inputs = {context, prefix + "/attention_out"};
            layer->output = prefix + "/out";
        }
        context = prefix + "/out";
    }

    model.push("lumen/time/input", "Identity", 0);
    if (auto* layer = model.getLayerByName("lumen/time/input")) {
        layer->inputs = {"timestep"};
        layer->output = "lumen/time/embedding";
    }
    model.push("lumen/time/projection", "Linear",
               static_cast<size_t>(context_dim) * static_cast<size_t>(context_dim) +
                   static_cast<size_t>(context_dim));
    if (auto* layer = model.getLayerByName("lumen/time/projection")) {
        layer->inputs = {"lumen/time/embedding"};
        layer->output = "lumen/time/projected";
        layer->in_features = context_dim;
        layer->out_features = context_dim;
        layer->use_bias = true;
    }
    addActivation("lumen/time/activation", "lumen/time/projected", "lumen/time/condition");

    model.push("lumen/dit/input", "Identity", 0);
    if (auto* layer = model.getLayerByName("lumen/dit/input")) {
        layer->inputs = {"__input__"};
        layer->output = "lumen/dit/latent_chw";
    }
        addConv("lumen/dit/patch_embed", "lumen/dit/latent_chw", "lumen/dit/patch_chw",
            latent_c, context_dim, latent_h, latent_w, patch_size, patch_size, 0);
    model.push("lumen/dit/to_tokens", "Permute", 0);
    if (auto* layer = model.getLayerByName("lumen/dit/to_tokens")) {
        layer->inputs = {"lumen/dit/patch_chw"};
        layer->output = "lumen/dit/patch_tokens";
        layer->shape = {context_dim, patch_h, patch_w};
        layer->permute_dims = {1, 2, 0};
    }
    model.push("lumen/dit/position", "Constant",
               static_cast<size_t>(patch_tokens) * static_cast<size_t>(context_dim));
    if (auto* layer = model.getLayerByName("lumen/dit/position")) {
        layer->output = "lumen/dit/position_values";
        layer->seq_len = patch_tokens;
        layer->embed_dim = context_dim;
    }
    model.push("lumen/dit/add_position", "Add", 0);
    if (auto* layer = model.getLayerByName("lumen/dit/add_position")) {
        layer->inputs = {"lumen/dit/patch_tokens", "lumen/dit/position_values"};
        layer->output = "lumen/dit/positioned";
    }
    model.push("lumen/dit/add_time", "Add", 0);
    if (auto* layer = model.getLayerByName("lumen/dit/add_time")) {
        layer->inputs = {"lumen/dit/positioned", "lumen/time/condition"};
        layer->output = "lumen/dit/conditioned0";
    }

    auto addNorm = [&](const std::string& name,
                       const std::string& input,
                       const std::string& output) {
        model.push(name, "LayerNorm", static_cast<size_t>(2 * context_dim));
        if (auto* layer = model.getLayerByName(name)) {
            layer->inputs = {input};
            layer->output = output;
            layer->in_features = context_dim;
            layer->affine = true;
            layer->use_bias = true;
            layer->eps = 1e-5f;
        }
    };

    std::string tokens = "lumen/dit/conditioned0";
    const size_t attention_params = static_cast<size_t>(4) *
        static_cast<size_t>(context_dim) * static_cast<size_t>(context_dim);
    for (int index = 0; index < depth; ++index) {
        const std::string prefix = "lumen/dit/block" + std::to_string(index + 1);
        addNorm(prefix + "/norm1", tokens, prefix + "/norm1_out");
        model.push(prefix + "/self_attention", "SelfAttention", attention_params);
        if (auto* layer = model.getLayerByName(prefix + "/self_attention")) {
            layer->inputs = {prefix + "/norm1_out"};
            layer->output = prefix + "/self_out";
            layer->seq_len = patch_tokens;
            layer->embed_dim = context_dim;
            layer->num_heads = heads;
            layer->causal = false;
        }
        model.push(prefix + "/add_self", "Add", 0);
        if (auto* layer = model.getLayerByName(prefix + "/add_self")) {
            layer->inputs = {tokens, prefix + "/self_out"};
            layer->output = prefix + "/res1";
        }

        addNorm(prefix + "/norm2", prefix + "/res1", prefix + "/norm2_out");
        model.push(prefix + "/cross_attention", "CrossAttention", attention_params);
        if (auto* layer = model.getLayerByName(prefix + "/cross_attention")) {
            layer->inputs = {prefix + "/norm2_out", context};
            layer->output = prefix + "/cross_out";
            layer->embed_dim = context_dim;
            layer->in_features = context_dim;
            layer->num_heads = heads;
            layer->causal = false;
        }
        model.push(prefix + "/add_cross", "Add", 0);
        if (auto* layer = model.getLayerByName(prefix + "/add_cross")) {
            layer->inputs = {prefix + "/res1", prefix + "/cross_out"};
            layer->output = prefix + "/res2";
        }

        addNorm(prefix + "/norm3", prefix + "/res2", prefix + "/norm3_out");
        model.push(prefix + "/mlp_fc1", "Linear",
                   static_cast<size_t>(context_dim) * static_cast<size_t>(mlp_hidden) +
                       static_cast<size_t>(mlp_hidden));
        if (auto* layer = model.getLayerByName(prefix + "/mlp_fc1")) {
            layer->inputs = {prefix + "/norm3_out"};
            layer->output = prefix + "/mlp_hidden";
            layer->seq_len = patch_tokens;
            layer->in_features = context_dim;
            layer->out_features = mlp_hidden;
            layer->use_bias = true;
        }
        model.push(prefix + "/mlp_gelu", "GELU", 0);
        if (auto* layer = model.getLayerByName(prefix + "/mlp_gelu")) {
            layer->inputs = {prefix + "/mlp_hidden"};
            layer->output = prefix + "/mlp_activated";
        }
        model.push(prefix + "/mlp_fc2", "Linear",
                   static_cast<size_t>(mlp_hidden) * static_cast<size_t>(context_dim) +
                       static_cast<size_t>(context_dim));
        if (auto* layer = model.getLayerByName(prefix + "/mlp_fc2")) {
            layer->inputs = {prefix + "/mlp_activated"};
            layer->output = prefix + "/mlp_out";
            layer->seq_len = patch_tokens;
            layer->in_features = mlp_hidden;
            layer->out_features = context_dim;
            layer->use_bias = true;
        }
        model.push(prefix + "/add_mlp", "Add", 0);
        if (auto* layer = model.getLayerByName(prefix + "/add_mlp")) {
            layer->inputs = {prefix + "/res2", prefix + "/mlp_out"};
            layer->output = prefix + "/out";
        }
        tokens = prefix + "/out";
    }

    addNorm("lumen/dit/final_norm", tokens, "lumen/dit/final_norm_out");
    model.push("lumen/dit/patch_output", "Linear",
               static_cast<size_t>(context_dim) * static_cast<size_t>(patch_dim) +
                   static_cast<size_t>(patch_dim));
    if (auto* layer = model.getLayerByName("lumen/dit/patch_output")) {
        layer->inputs = {"lumen/dit/final_norm_out"};
        layer->output = "lumen/dit/patch_noise";
        layer->seq_len = patch_tokens;
        layer->in_features = context_dim;
        layer->out_features = patch_dim;
        layer->use_bias = true;
    }
    model.push("lumen/dit/to_patch_chw", "Permute", 0);
    if (auto* layer = model.getLayerByName("lumen/dit/to_patch_chw")) {
        layer->inputs = {"lumen/dit/patch_noise"};
        layer->output = "lumen/dit/patch_noise_chw";
        layer->shape = {patch_h, patch_w, patch_dim};
        layer->permute_dims = {2, 0, 1};
    }
    model.push("lumen/dit/unpatchify", "PixelShuffle", 0);
    if (auto* layer = model.getLayerByName("lumen/dit/unpatchify")) {
        layer->inputs = {"lumen/dit/patch_noise_chw"};
        layer->output = "x";
        layer->scale_h = static_cast<float>(patch_size);
        layer->in_channels = patch_dim;
        layer->input_height = patch_h;
        layer->input_width = patch_w;
    }
}

std::vector<int> LumenLatentDiffusionModel::promptIds(const std::string& prompt) const {
    std::vector<int> ids = getTokenizer().tokenize(prompt);
    const int pad = std::max(0, getTokenizer().getPadId());
    const int unknown = std::max(0, getTokenizer().getUnkId());
    ids.resize(static_cast<size_t>(std::max(1, cfg_.text_seq_len)), pad);
    if (ids.size() > static_cast<size_t>(cfg_.text_seq_len)) {
        ids.resize(static_cast<size_t>(cfg_.text_seq_len));
    }
    for (int& id : ids) {
        if (id < 0 || id >= cfg_.vocab_size) id = unknown;
    }
    return ids;
}

std::vector<float> LumenLatentDiffusionModel::timestepEmbedding(int timestep, int dim) const {
    std::vector<float> embedding(static_cast<size_t>(dim), 0.0f);
    const int half = std::max(1, dim / 2);
    for (int index = 0; index < half; ++index) {
        const float exponent = -std::log(10000.0f) * static_cast<float>(index) /
                               static_cast<float>(std::max(1, half - 1));
        const float angle = static_cast<float>(timestep) * std::exp(exponent);
        embedding[static_cast<size_t>(index)] = std::sin(angle);
        if (index + half < dim) {
            embedding[static_cast<size_t>(index + half)] = std::cos(angle);
        }
    }
    return embedding;
}

std::vector<float> LumenLatentDiffusionModel::encodeImageUncalibrated(
    const std::vector<float>& image
) {
    const size_t expected = static_cast<size_t>(cfg_.image_w) * cfg_.image_h * cfg_.image_c;
    if (image.size() != expected) {
        throw std::runtime_error("LumenLatentDiffusionModel: input image size mismatch");
    }
    if (vae_encoder_) {
        vae_encoder_skips_.clear();
        (void)vae_encoder_->forwardPassNamed(
            {{"__input__", image}}, {}, false);
        if (!vae_encoder_->hasTensor(vae_encoder_output_)) {
            throw std::runtime_error(
                "LumenLatentDiffusionModel: VAE encoder did not produce " +
                vae_encoder_output_);
        }
        const auto& latent = vae_encoder_->getTensor(vae_encoder_output_);
        const size_t latent_values = static_cast<size_t>(cfg_.latent_w) *
            cfg_.latent_h * cfg_.latent_c;
        if (latent.size() != latent_values) {
            throw std::runtime_error("LumenLatentDiffusionModel: VAE latent size mismatch");
        }
        for (const auto& binding : vae_skip_bindings_) {
            if (!vae_encoder_->hasTensor(binding.encoder_output)) {
                throw std::runtime_error(
                    "LumenLatentDiffusionModel: VAE encoder did not produce skip " +
                    binding.encoder_output);
            }
            const auto& skip = vae_encoder_->getTensor(binding.encoder_output);
            if (skip.size() != binding.values) {
                throw std::runtime_error(
                    "LumenLatentDiffusionModel: VAE encoder skip size mismatch for " +
                    binding.encoder_output);
            }
            vae_encoder_skips_[binding.decoder_input] =
                std::vector<float>(skip.begin(), skip.end());
        }
        return std::vector<float>(latent.begin(), latent.end());
    }
    const size_t latent_values = static_cast<size_t>(cfg_.latent_w) *
        cfg_.latent_h * cfg_.latent_c;
    if (latent_values != expected) {
        throw std::runtime_error(
            "LumenLatentDiffusionModel: vae_checkpoint is required to encode images");
    }
    std::vector<float> rgb_chw(expected);
    const size_t spatial = static_cast<size_t>(cfg_.image_w) * cfg_.image_h;
    for (int y = 0; y < cfg_.image_h; ++y) {
        for (int x = 0; x < cfg_.image_w; ++x) {
            for (int channel = 0; channel < cfg_.image_c; ++channel) {
                const size_t hwc = (static_cast<size_t>(y) * cfg_.image_w + x) *
                                   cfg_.image_c + channel;
                const size_t chw = static_cast<size_t>(channel) * spatial +
                                   static_cast<size_t>(y) * cfg_.image_w + x;
                rgb_chw[chw] = image[hwc];
            }
        }
    }
    return rgb_chw;
}

std::vector<float> LumenLatentDiffusionModel::encodeImage(const std::vector<float>& image) {
    std::vector<float> latent = encodeImageUncalibrated(image);
    if (!vae_encoder_) return latent;

    const float scale = modelConfig.value("vae_scale", cfg_.vae_scale);
    const float shift = modelConfig.value("vae_shift", cfg_.vae_shift);
    if (!std::isfinite(scale) || scale <= 0.0f || !std::isfinite(shift)) {
        throw std::runtime_error(
            "LumenLatentDiffusionModel: VAE latent calibration is unavailable; "
            "calibrate the VAE before training");
    }
    for (float& value : latent) value = (value - shift) * scale;
    return latent;
}

void LumenLatentDiffusionModel::beginVaeCalibration() {
    if (!vae_encoder_) {
        throw std::runtime_error(
            "LumenLatentDiffusionModel: vae_checkpoint is required for calibration");
    }
    vae_calibration_items_ = 0;
    vae_calibration_values_ = 0;
    vae_calibration_mean_ = 0.0;
    vae_calibration_m2_ = 0.0;
    modelConfig["vae_calibrated"] = false;
}

LumenLatentDiffusionModel::VaeCalibrationStats
LumenLatentDiffusionModel::addVaeCalibrationImage(
    const std::vector<unsigned char>& rgb_image
) {
    const size_t expected = static_cast<size_t>(cfg_.image_w) * cfg_.image_h * cfg_.image_c;
    if (rgb_image.size() != expected) {
        throw std::runtime_error(
            "LumenLatentDiffusionModel: calibration image size mismatch; expected " +
            std::to_string(expected) + " RGB values");
    }
    std::vector<float> image(expected);
    for (size_t index = 0; index < expected; ++index) {
        image[index] = static_cast<float>(rgb_image[index]) / 127.5f - 1.0f;
    }
    const std::vector<float> latent = encodeImageUncalibrated(image);
    for (float value : latent) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "LumenLatentDiffusionModel: non-finite VAE latent during calibration");
        }
        ++vae_calibration_values_;
        const double delta = static_cast<double>(value) - vae_calibration_mean_;
        vae_calibration_mean_ += delta / static_cast<double>(vae_calibration_values_);
        const double delta2 = static_cast<double>(value) - vae_calibration_mean_;
        vae_calibration_m2_ += delta * delta2;
    }
    ++vae_calibration_items_;
    return {vae_calibration_items_, vae_calibration_values_,
            static_cast<float>(vae_calibration_mean_), 0.0f};
}

LumenLatentDiffusionModel::VaeCalibrationStats
LumenLatentDiffusionModel::finishVaeCalibration() {
    if (vae_calibration_items_ == 0 || vae_calibration_values_ == 0) {
        throw std::runtime_error("LumenLatentDiffusionModel: VAE calibration has no samples");
    }
    const double variance = vae_calibration_m2_ /
        static_cast<double>(vae_calibration_values_);
    const float shift = static_cast<float>(vae_calibration_mean_);
    const float stddev = static_cast<float>(std::sqrt(std::max(0.0, variance)));
    const float scale = std::clamp(1.0f / std::max(stddev, 1e-6f), 1e-3f, 1e3f);
    modelConfig["vae_scale"] = scale;
    modelConfig["vae_shift"] = shift;
    modelConfig["vae_calibrated"] = true;
    modelConfig["vae_calibration_items"] = vae_calibration_items_;
    modelConfig["vae_calibration_values"] = vae_calibration_values_;
    return {vae_calibration_items_, vae_calibration_values_, shift, scale};
}

std::vector<float> LumenLatentDiffusionModel::decodeImage(
    const std::vector<float>& latent_chw,
    bool use_encoder_skips
) {
    const size_t latent_values = static_cast<size_t>(cfg_.latent_w) *
        cfg_.latent_h * cfg_.latent_c;
    if (latent_chw.size() != latent_values) {
        throw std::runtime_error("LumenLatentDiffusionModel: latent diffusion tensor size mismatch");
    }
    if (vae_decoder_) {
        const float scale = modelConfig.value("vae_scale", cfg_.vae_scale);
        const float shift = modelConfig.value("vae_shift", cfg_.vae_shift);
        if (!std::isfinite(scale) || scale <= 0.0f || !std::isfinite(shift)) {
            throw std::runtime_error(
                "LumenLatentDiffusionModel: VAE latent calibration is unavailable");
        }
        std::vector<float> decoder_latent(latent_chw.size());
        for (size_t index = 0; index < latent_chw.size(); ++index) {
            decoder_latent[index] = latent_chw[index] / scale + shift;
        }
        std::unordered_map<std::string, std::vector<float>> decoder_inputs = {
            {"__input__", std::move(decoder_latent)},
        };
        for (const auto& binding : vae_skip_bindings_) {
            const auto cached = vae_encoder_skips_.find(binding.decoder_input);
            if (use_encoder_skips && cached != vae_encoder_skips_.end()) {
                decoder_inputs.emplace(binding.decoder_input, cached->second);
            } else {
                decoder_inputs.emplace(
                    binding.decoder_input, std::vector<float>(binding.values, 0.0f));
            }
        }
        const std::vector<float> decoded = vae_decoder_->forwardPassNamed(
            decoder_inputs, {}, false);
        const size_t image_values = static_cast<size_t>(cfg_.image_w) *
            cfg_.image_h * cfg_.image_c;
        if (decoded.size() != image_values) {
            throw std::runtime_error("LumenLatentDiffusionModel: VAE decoded image size mismatch");
        }
        return decoded;
    }
    const size_t expected = static_cast<size_t>(cfg_.image_w) * cfg_.image_h * cfg_.image_c;
    if (latent_values != expected) {
        throw std::runtime_error(
            "LumenLatentDiffusionModel: vae_checkpoint is required to decode latents");
    }
    std::vector<float> image(expected);
    const size_t spatial = static_cast<size_t>(cfg_.image_w) * cfg_.image_h;
    for (int y = 0; y < cfg_.image_h; ++y) {
        for (int x = 0; x < cfg_.image_w; ++x) {
            for (int channel = 0; channel < cfg_.image_c; ++channel) {
                const size_t chw = static_cast<size_t>(channel) * spatial +
                                   static_cast<size_t>(y) * cfg_.image_w + x;
                const size_t hwc = (static_cast<size_t>(y) * cfg_.image_w + x) *
                                   cfg_.image_c + channel;
                image[hwc] = latent_chw[chw];
            }
        }
    }
    return image;
}

LumenLatentDiffusionModel::GeneratedImage LumenLatentDiffusionModel::generate(
    const std::string& prompt,
    int seed,
    int sample_steps,
    float guidance_scale,
    const std::function<void(int, int)>& progress
) {
    if (getMutableLayers().empty()) {
        throw std::runtime_error("LumenLatentDiffusionModel: model is not built");
    }
    const int image_dim = cfg_.latent_w * cfg_.latent_h * cfg_.latent_c;
    const int context_dim = std::max(8, cfg_.hidden_size);
    const int total_steps = std::max(2, cfg_.diffusion_steps);
    const int steps = std::clamp(sample_steps, 1, total_steps);
    const float beta_start = std::clamp(cfg_.beta_start, 1e-7f, 0.999f);
    const float beta_end = std::clamp(cfg_.beta_end, beta_start, 0.999f);

    std::vector<float> alpha_bar(static_cast<size_t>(total_steps));
    float cumulative = 1.0f;
    for (int index = 0; index < total_steps; ++index) {
        const float ratio = static_cast<float>(index) / static_cast<float>(total_steps - 1);
        const float beta = beta_start + ratio * (beta_end - beta_start);
        cumulative *= 1.0f - beta;
        alpha_bar[static_cast<size_t>(index)] = cumulative;
    }

    std::mt19937 random(static_cast<unsigned int>(seed));
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> sample(static_cast<size_t>(image_dim));
    for (float& value : sample) value = normal(random);

    const std::vector<int> conditioned_ids = promptIds(prompt);
    const std::vector<int> unconditioned_ids = promptIds("");
    for (int step_index = steps - 1; step_index >= 0; --step_index) {
        const int timestep = static_cast<int>(std::lround(
            static_cast<double>(step_index) * static_cast<double>(total_steps - 1) /
            static_cast<double>(std::max(1, steps - 1))));
        const int previous_timestep = step_index > 0
            ? static_cast<int>(std::lround(
                  static_cast<double>(step_index - 1) * static_cast<double>(total_steps - 1) /
                  static_cast<double>(std::max(1, steps - 1))))
            : -1;

        std::unordered_map<std::string, std::vector<float>> float_inputs{
            {"__input__", sample},
            {"timestep", timestepEmbedding(timestep, context_dim)},
        };
        const std::vector<float> unconditioned = forwardPassNamed(
            float_inputs, {{"text_ids", unconditioned_ids}}, false);
        const std::vector<float> conditioned = forwardPassNamed(
            float_inputs, {{"text_ids", conditioned_ids}}, false);
        if (conditioned.size() != sample.size() || unconditioned.size() != sample.size()) {
            throw std::runtime_error("LumenLatentDiffusionModel: denoiser output size mismatch");
        }

        std::vector<float> diffusion_output(sample.size());
        for (size_t index = 0; index < sample.size(); ++index) {
            diffusion_output[index] = unconditioned[index] + guidance_scale *
                (conditioned[index] - unconditioned[index]);
        }
        const float current_alpha = std::max(alpha_bar[static_cast<size_t>(timestep)], 1e-8f);
        const float previous_alpha = previous_timestep >= 0
            ? std::max(alpha_bar[static_cast<size_t>(previous_timestep)], 1e-8f)
            : 1.0f;
        const float sqrt_current = std::sqrt(current_alpha);
        const float sqrt_current_noise = std::sqrt(std::max(0.0f, 1.0f - current_alpha));
        const float sqrt_previous = std::sqrt(previous_alpha);
        const float sqrt_previous_noise = std::sqrt(std::max(0.0f, 1.0f - previous_alpha));

        for (size_t index = 0; index < sample.size(); ++index) {
            const float epsilon = diffusion_output[index];
            const float clean = (sample[index] - sqrt_current_noise * epsilon) / sqrt_current;
            sample[index] = sqrt_previous * clean + sqrt_previous_noise * epsilon;
        }
        if (progress) progress(steps - step_index, steps);
    }

    const std::vector<float> decoded = decodeImage(sample);

    GeneratedImage image;
    image.w = cfg_.image_w;
    image.h = cfg_.image_h;
    image.channels = cfg_.image_c;
    image.pixels.resize(decoded.size());
    for (size_t index = 0; index < decoded.size(); ++index) {
        const float normalized = std::clamp(decoded[index] * 0.5f + 0.5f, 0.0f, 1.0f);
        image.pixels[index] = static_cast<unsigned char>(std::lround(normalized * 255.0f));
    }
    return image;
}
