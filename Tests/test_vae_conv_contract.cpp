#include "test_utils.hpp"

#include "Model.hpp"
#include "Models/Vision/VAEConvModel.hpp"

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

int main() {
    // Rectangular inputs are valid when both axes use the same power-of-two
    // downsampling ratio (8x4 -> 4x2 here).
    VAEConvModel::Config rectangular_cfg;
    rectangular_cfg.image_w = 8;
    rectangular_cfg.image_h = 4;
    rectangular_cfg.image_c = 1;
    rectangular_cfg.latent_w = 4;
    rectangular_cfg.latent_h = 2;
    rectangular_cfg.latent_c = 1;
    rectangular_cfg.base_channels = 8;
    rectangular_cfg.use_attention = false;
    rectangular_cfg.use_attn = false;
    VAEConvModel rectangular_vae;
    rectangular_vae.buildFromConfig(rectangular_cfg);
    TASSERT_TRUE(!rectangular_vae.getLayers().empty());

    // Decoder normalization can be disabled independently from the encoder.
    // This protects the CLI/config contract `enc_norm=groupnorm, dec_norm=none`.
    VAEConvModel::Config decoder_no_norm_cfg = rectangular_cfg;
    decoder_no_norm_cfg.use_attention = true;
    decoder_no_norm_cfg.resnet_max_tokens = 0;
    decoder_no_norm_cfg.enc_norm = "groupnorm";
    decoder_no_norm_cfg.dec_norm = "none";
    VAEConvModel decoder_no_norm_vae;
    decoder_no_norm_vae.buildFromConfig(decoder_no_norm_cfg);
    bool saw_encoder_norm = false;
    bool saw_decoder_norm = false;
    for (const auto& layer : decoder_no_norm_vae.getLayers()) {
        const bool is_norm = layer.type == "GroupNorm" || layer.type == "LayerNorm";
        if (!is_norm) continue;
        saw_encoder_norm = saw_encoder_norm || layer.name.rfind("vae_conv/enc/", 0) == 0;
        saw_decoder_norm = saw_decoder_norm || layer.name.rfind("vae_conv/dec/", 0) == 0;
    }
    TASSERT_TRUE(saw_encoder_norm);
    TASSERT_TRUE(!saw_decoder_norm);

    // A fixed Constant must remain fixed, while an explicitly trainable one
    // receives the exact upstream gradient and is updated by the optimizer.
    Model parameter_model;
    parameter_model.push("learned", "Constant", 3);
    Layer* learned = parameter_model.getLayerByName("learned");
    TASSERT_TRUE(learned != nullptr);
    learned->inputs = {};
    learned->output = "x";
    learned->trainable_parameter = true;

    parameter_model.allocateParams();
    float* learned_weights = learned->getWeights();
    TASSERT_TRUE(learned_weights != nullptr);
    std::fill(learned_weights, learned_weights + 3, 0.0f);

    const auto parameter_out = parameter_model.forwardPass(std::vector<float>{}, true);
    TASSERT_TRUE(parameter_out.size() == 3);
    parameter_model.backwardPass({1.0f, -2.0f, 0.5f});
    TASSERT_TRUE(learned->grad_weights.size() == 3);
    TASSERT_NEAR(learned->grad_weights[0], 1.0f, 1e-6f);
    TASSERT_NEAR(learned->grad_weights[1], -2.0f, 1e-6f);
    TASSERT_NEAR(learned->grad_weights[2], 0.5f, 1e-6f);

    Optimizer opt;
    opt.type = OptimizerType::SGD;
    opt.decay_strategy = LRDecayStrategy::NONE;
    parameter_model.optimizerStep(opt, 0.1f);
    TASSERT_NEAR(learned_weights[0], -0.1f, 1e-6f);
    TASSERT_NEAR(learned_weights[1], 0.2f, 1e-6f);
    TASSERT_NEAR(learned_weights[2], -0.05f, 1e-6f);

    // The packed VAE output must expose mu even when the decoder consumes a
    // stochastic, prior-biased z.
    VAEConvModel::Config cfg;
    cfg.image_w = 4;
    cfg.image_h = 4;
    cfg.image_c = 1;
    cfg.latent_w = 2;
    cfg.latent_h = 2;
    cfg.latent_c = 2;
    cfg.base_channels = 8;
    cfg.stochastic_latent = true;
    cfg.use_attention = false;
    cfg.use_attn = false;
    cfg.enc_norm = "none";
    cfg.dec_norm = "none";
    cfg.use_encoder_prior = true;

    VAEConvModel vae;
    vae.buildFromConfig(cfg);
    Layer* prior = vae.getLayerByName("vae_conv/z_prior_bias");
    TASSERT_TRUE(prior != nullptr);
    TASSERT_TRUE(prior->trainable_parameter);

    vae.allocateParams();
    vae.initializeWeights("xavier", 123u);
    std::vector<float> input(16, 0.0f);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i) / 15.0f - 0.5f;
    }

    const std::vector<float> packed = vae.forwardPass(input, true);
    const std::vector<float> mu = vae.getTensor("vae_conv/mu");
    const std::vector<float> z = vae.getTensor("vae_conv/z");
    const std::vector<float> prior_values = vae.getTensor("vae_conv/prior_bias_out");
    const std::vector<float> z_biased = vae.getTensor("vae_conv/z_biased");
    TASSERT_TRUE(mu.size() == 8);
    TASSERT_TRUE(packed.size() == 16 + 2 * mu.size());
    for (size_t i = 0; i < mu.size(); ++i) {
        TASSERT_NEAR(packed[16 + i], mu[i], 1e-6f);
    }
    TASSERT_TRUE(z_biased.size() == mu.size());
    TASSERT_TRUE(z.size() == z_biased.size());
    TASSERT_TRUE(prior_values.size() == z_biased.size());
    for (size_t i = 0; i < z_biased.size(); ++i) {
        TASSERT_NEAR(z_biased[i], z[i] + prior_values[i], 1e-6f);
    }

    // VIZ contract: a deliberately tiny historical limit must not hide graph
    // layers. Every node gets one canonical <model>/blocks/... label.
    vae.setVizTapsEnabled(true);
    vae.setVizTapsLimits(1, 32);
    (void)vae.forwardPass(input, false);
    const auto vae_taps = vae.consumeVizTaps();
    std::unordered_set<std::string> tapped_layers;
    for (const auto& frame : vae_taps) {
        const size_t bar = frame.label.find(" | ");
        const std::string base_label = frame.label.substr(0, bar);
        const size_t blocks = base_label.find("/blocks/");
        if (blocks == std::string::npos) continue; // custom recon/error frames
        size_t type_sep = base_label.find_last_of('/');
        if (base_label.compare(type_sep + 1, std::string::npos, "vec") == 0) {
            type_sep = base_label.find_last_of('/', type_sep - 1);
        }
        TASSERT_TRUE(type_sep != std::string::npos && type_sep > blocks + 8);
        std::string layer_name = base_label.substr(blocks + 8, type_sep - (blocks + 8));
        tapped_layers.insert(std::move(layer_name));
    }
    for (const auto& layer : vae.getLayers()) {
        TASSERT_TRUE(tapped_layers.find(layer.name) != tapped_layers.end());
    }

    // Composition contract: a child model executed before the root taps are
    // consumed automatically contributes its own canonical thumbnails.
    Model root;
    root.setModelName("root_model");
    root.modelConfig["type"] = "root_model";
    root.push("root/layer", "Identity", 0);
    root.getLayerByName("root/layer")->inputs = {"__input__"};
    root.getLayerByName("root/layer")->output = "x";
    root.allocateParams();
    root.setVizTapsEnabled(true);
    root.setVizTapsLimits(1, 16);
    (void)root.forwardPass(std::vector<float>{1.0f, 2.0f, 3.0f}, false);

    Model child;
    child.setModelName("child_model");
    child.modelConfig["type"] = "child_model";
    child.push("child/layer", "Identity", 0);
    child.getLayerByName("child/layer")->inputs = {"__input__"};
    child.getLayerByName("child/layer")->output = "x";
    child.allocateParams();
    (void)child.forwardPass(std::vector<float>{4.0f, 5.0f, 6.0f}, false);

    const auto composed_taps = root.consumeVizTaps();
    bool saw_root = false;
    bool saw_child = false;
    for (const auto& frame : composed_taps) {
        saw_root = saw_root || frame.label.find("root_model/blocks/root/layer/") == 0;
        saw_child = saw_child || frame.label.find("child_model/blocks/child/layer/") == 0;
    }
    TASSERT_TRUE(saw_root);
    TASSERT_TRUE(saw_child);

    // A reconstruction-only loss must train the prior through the decoder
    // feature path (prior -> Add -> decoder convolutions -> reconstruction).
    std::vector<float> reconstruction_grad(packed.size(), 0.0f);
    std::fill(reconstruction_grad.begin(), reconstruction_grad.begin() + 16, 1.0f / 16.0f);
    vae.backwardPass(reconstruction_grad);
    TASSERT_TRUE(prior->grad_weights.size() == mu.size());
    float prior_grad_l1 = 0.0f;
    for (float g : prior->grad_weights) {
        prior_grad_l1 += std::fabs(g);
    }
    TASSERT_TRUE(prior_grad_l1 > 1e-8f);

    return 0;
}
