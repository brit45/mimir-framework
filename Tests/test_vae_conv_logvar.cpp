#include "test_utils.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "Serialization/Serialization.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
double norm(const std::vector<float>& values) {
    double sum = 0;
    for (float value : values) {
        require(std::isfinite(value), "nonfinite tensor/gradient");
        sum += double(value) * value;
    }
    return std::sqrt(sum);
}
double difference(const std::vector<float>& a, const std::vector<float>& b) {
    require(a.size() == b.size(), "tensor sizes differ");
    double sum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite comparison");
        const double delta = double(a[i]) - b[i];
        sum += delta * delta;
    }
    return std::sqrt(sum);
}
std::vector<float> weights(const Layer& layer) {
    return {layer.getWeights(), layer.getWeights() + layer.getWeightsSize()};
}
json fixture(const std::string& rank, const std::string& normalization, bool stochastic) {
    return {{"dtype", "float32"}, {"image_w", 8}, {"image_h", 8}, {"image_c", 3},
        {"latent_w", 2}, {"latent_h", 2}, {"latent_c", 2}, {"base_channels", 8},
        {"conv_mode", rank}, {"resnet", false}, {"attention", false},
        {"enc_norm", normalization}, {"dec_norm", normalization},
        {"enc_gn_groups", 2}, {"dec_gn_groups", 2},
        {"latent_spatial_context", true}, {"latent_detail_path", true},
        {"latent_context_channels", 8}, {"decoder_upsample", "nearest_conv"},
        {"stochastic_latent", stochastic}, {"use_skip_connections", false},
        {"use_encoder_prior", false}, {"text_cond", false},
        {"kl_beta", 0}, {"kl_warmup_steps", 0}, {"recon_loss", "mse"}, {"weight_decay", 0},
        {"logvar_head_bias", false}};
}
std::vector<float> image() {
    std::vector<float> result(8 * 8 * 3);
    for (size_t i = 0; i < result.size(); ++i)
        result[i] = .39f * std::sin(float(i) * .47f) + .11f * std::cos(float(i) * .19f);
    return result;
}

void runtimeGradient() {
    CpuRuntime runtime;
    RuntimeConfig config;
    require(runtime.initialize(config), "CPU initialization failed");
    Layer layer("posterior", "Reparameterize", 0);
    const std::vector<float> mu{.25f, -.5f, .75f, 0.f};
    const std::vector<float> logvar(mu.size(), 0.f);
    const std::vector<float> upstream{.2f, -.4f, .6f, -.8f};
    RuntimeForwardContext stochastic;
    stochastic.stochastic_latent = true;
    std::vector<std::vector<float>> output;
    require(runtime.forwardLayerWithContext({&mu, &logvar}, output, layer, true, stochastic),
        "stochastic posterior forward failed");
    require(output.size() == 1 && difference(output[0], mu) > 1e-4,
        "logvar=0 was interpreted as zero variance instead of sigma=1");
    RuntimeBackwardContext backward;
    backward.output = &output[0];
    std::vector<std::vector<float>> gradients;
    require(runtime.backwardLayerWithContext({&mu, &logvar}, {&upstream}, gradients, layer, true, backward),
        "posterior backward failed");
    require(gradients.size() == 2 && gradients[0] == upstream && norm(gradients[1]) > 1e-5,
        "reconstruction gradient did not reach logvar=0");
    for (size_t i = 0; i < mu.size(); ++i) {
        const float expected = .5f * upstream[i] * (output[0][i] - mu[i]);
        require(std::abs(gradients[1][i] - expected) < 1e-6f,
            "posterior pathwise derivative disagrees with sampled z");
    }
    RuntimeForwardContext deterministic;
    deterministic.stochastic_latent = false;
    require(runtime.forwardLayerWithContext({&mu, &logvar}, output, layer, true, deterministic)
            && output[0] == mu, "deterministic posterior changed mu");
    backward.output = &output[0];
    require(runtime.backwardLayerWithContext({&mu, &logvar}, {&upstream}, gradients, layer, true, backward)
            && norm(gradients[1]) == 0, "deterministic posterior invented a variance gradient");
}

void reconstructionGradientAndTraining(const std::string& rank, const std::string& normalization,
                                       bool stochastic) {
    const auto cfg = fixture(rank, normalization, stochastic);
    auto model = ModelArchitectures::create("vae_conv", cfg);
    model->allocateParams();
    model->initializeWeights("xavier", 691u);
    const auto input = image();
    auto* head = model->getLayerByName("vae_conv/enc/logvar");
    auto* features = model->getLayerByName("vae_conv/enc/logvar_pre_conv");
    require(head && features, "native variance feature branch missing");
    const auto initial_head = weights(*head), initial_features = weights(*features);
    Optimizer optimizer;
    optimizer.type = OptimizerType::ADAM;
    optimizer.initial_lr = .001f;
    optimizer.weight_decay = 0;
    optimizer.decay_strategy = LRDecayStrategy::NONE;
    Model::TrainStepRequest request;
    request.float_inputs["__input__"] = &input;
    request.target = &input;
    request.optimizer = &optimizer;
    request.learning_rate = .001f;
    double head_gradient = 0, feature_gradient = 0, sampling = 0;
    float first_loss = 0, last_loss = 0;
    constexpr int steps = 12;
    for (int step = 0; step < steps; ++step) {
        const auto result = model->trainStep(request);
        require(result && std::isfinite(result->loss), "native VAE training failed");
        require(result->metrics.at("kl_beta_effective") == 0,
            "reconstruction-only fixture accidentally enabled KL");
        require(result->metrics.at("stochastic_latent_active") == (stochastic ? 1.f : 0.f),
            "posterior diagnostics do not report the active sampling policy");
        require(result->metrics.at("logvar_nonfinite") == 0 &&
            result->metrics.at("posterior_std_nonfinite") == 0 &&
            result->metrics.at("posterior_std_mean") > 0 &&
            result->metrics.at("logvar_min") <= result->metrics.at("logvar_mean") &&
            result->metrics.at("logvar_mean") <= result->metrics.at("logvar_max"),
            "posterior diagnostics are absent or inconsistent");
        require(stochastic ? result->metrics.at("logvar_head_grad_norm") > 0
                           : result->metrics.at("logvar_head_grad_norm") == 0,
            "posterior gradient diagnostics do not distinguish sampled and deterministic training");
        require(stochastic ? result->metrics.at("latent_sampling_rms") > 0
                           : result->metrics.at("latent_sampling_rms") == 0,
            "posterior sampling diagnostics do not describe the emitted latent");
        if (step == 0) first_loss = result->loss;
        last_loss = result->loss;
        head_gradient += norm(head->grad_weights);
        feature_gradient += norm(features->grad_weights);
        sampling += difference(model->getTensor("vae_conv/z"), model->getTensor("vae_conv/mu"));
    }
    const double head_change = difference(weights(*head), initial_head);
    const double feature_change = difference(weights(*features), initial_features);
    std::cout << "LOGVAR_RECON " << rank << ' ' << normalization << " stochastic=" << stochastic
              << " loss=" << first_loss << "->" << last_loss << " head_grad=" << head_gradient
              << " feature_grad=" << feature_gradient << " sample=" << sampling
              << " head_change=" << head_change << " feature_change=" << feature_change << '\n';
    if (stochastic) {
        require(head_gradient > 1e-9 && feature_gradient > 1e-9 && sampling > 1e-4,
            "reconstruction-only training disconnected variance branch: " + rank + " " + normalization);
        require(head_change > 1e-7 && feature_change > 1e-7,
            "optimizer did not update the reconstruction-trained variance branch");
    } else {
        require(head_gradient == 0 && feature_gradient == 0 && sampling == 0,
            "deterministic VAE invented a reconstruction variance signal");
        require(head_change == 0 && feature_change == 0,
            "deterministic variance weights changed with KL and weight decay disabled");
    }
}

void logvarOnlyIntervention(const std::string& rank) {
    const auto cfg = fixture(rank, "groupnorm", true);
    auto model = ModelArchitectures::create("vae_conv", cfg);
    model->allocateParams(); model->initializeWeights("xavier", 719u);
    const auto input = image();
    const auto mean_reconstruction = model->forwardPass(input, false);
    const auto mu = model->getTensor("vae_conv/mu");
    const auto before_logvar = model->getTensor("vae_conv/logvar");
    auto* head = model->getLayerByName("vae_conv/enc/logvar");
    require(head, "missing variance head");
    for (size_t i = 0; i < head->getWeightsSize(); ++i) head->getWeights()[i] *= 1.75f;
    const auto after_mean_reconstruction = model->forwardPass(input, false);
    const auto after_mu = model->getTensor("vae_conv/mu");
    const auto after_logvar = model->getTensor("vae_conv/logvar");
    require(mu == after_mu && mean_reconstruction == after_mean_reconstruction,
        "variance-only intervention changed deterministic posterior mean reconstruction");
    require(difference(before_logvar, after_logvar) > 1e-5,
        "variance head intervention did not affect logvar");

    auto decoder = ModelArchitectures::create("vae_conv_decode", cfg);
    decoder->allocateParams();
    for (auto& layer : decoder->getMutableLayers()) {
        if (!layer.getWeightsSize()) continue;
        const auto* source = model->getLayerByName(layer.name);
        require(source && source->getWeightsSize() == layer.getWeightsSize(), "decoder weight mismatch");
        std::copy_n(source->getWeights(), layer.getWeightsSize(), layer.getWeights());
    }
    std::vector<float> before_z(mu.size()), after_z(mu.size());
    for (size_t i = 0; i < mu.size(); ++i) {
        // Hold epsilon fixed so random resampling cannot masquerade as an
        // effect of changing logvar. This is the same native decoder as above.
        const float epsilon = .31f * std::sin(float(i) * 1.13f) + .27f;
        before_z[i] = mu[i] + std::exp(.5f * std::clamp(before_logvar[i], -20.f, 20.f)) * epsilon;
        after_z[i] = mu[i] + std::exp(.5f * std::clamp(after_logvar[i], -20.f, 20.f)) * epsilon;
    }
    require(difference(before_z, after_z) > 1e-5, "logvar did not affect a fixed-noise posterior sample");
    const auto before = decoder->forwardPass(before_z, false), after = decoder->forwardPass(after_z, false);
    require(difference(before, after) > 1e-5, "decoder ignored variance-induced changes in z");
}

void affineVarianceAndReload(const std::string& rank) {
    auto cfg = fixture(rank, "groupnorm", true);
    cfg["enc_gn_groups"] = 8;
    auto legacy = ModelArchitectures::create("vae_conv", cfg);
    legacy->allocateParams(); legacy->initializeWeights("xavier", 751u);
    // With zero normalization offsets and one group per feature channel, a
    // bias-free 1x1 head has no independent spatially constant variance shift.
    // GroupNorm beta can itself learn later; this is an initial capacity test,
    // not a claim that its affine offsets remain zero throughout training.
    auto* normalized = legacy->getLayerByName("vae_conv/enc/logvar_pre_norm/gn");
    auto* legacy_head = legacy->getLayerByName("vae_conv/enc/logvar");
    require(normalized && legacy_head && !legacy_head->use_bias, "legacy variance head changed");
    require(normalized->num_groups == normalized->in_channels, "centering fixture did not use per-channel groups");
    for (int channel = 0; channel < normalized->in_channels; ++channel) {
        normalized->getWeights()[channel] = 1.f;
        normalized->getWeights()[normalized->in_channels + channel] = 0.f;
    }
    const auto input = image();
    legacy->forwardPass(input, false);
    const auto old_logvar = legacy->getTensor("vae_conv/logvar");
    constexpr size_t plane = 4;
    for (size_t channel = 0; channel < old_logvar.size() / plane; ++channel) {
        double mean = 0;
        for (size_t position = 0; position < plane; ++position) mean += old_logvar[channel * plane + position];
        require(std::abs(mean / plane) < 2e-5, "zero-offset normalization did not center legacy logvar");
    }
    auto* legacy_features = legacy->getLayerByName("vae_conv/enc/logvar_pre_conv");
    require(legacy_features, "legacy variance features missing");
    std::fill_n(legacy_features->getWeights(), legacy_features->getWeightsSize(), 0.f);
    std::fill_n(legacy_head->getWeights(), legacy_head->getWeightsSize(), 0.f);
    Optimizer legacy_optimizer;
    legacy_optimizer.weight_decay = 0;
    Model::TrainStepRequest legacy_request;
    legacy_request.float_inputs["__input__"] = &input; legacy_request.target = &input;
    legacy_request.optimizer = &legacy_optimizer; legacy_request.mode = Model::TrainStepMode::Accumulate;
    legacy->zeroGradients();
    const auto legacy_result = legacy->trainStep(legacy_request);
    require(legacy_result && std::isfinite(legacy_result->loss), "legacy stationary fixture failed");
    require(norm(legacy_head->grad_weights) == 0 && norm(legacy_features->grad_weights) == 0,
        "zero features and zero bias-free head unexpectedly escaped the stationary state");

    cfg["logvar_head_bias"] = true;
    auto model = ModelArchitectures::create("vae_conv", cfg);
    model->allocateParams(); model->initializeWeights("xavier", 751u);
    auto* head = model->getLayerByName("vae_conv/enc/logvar");
    require(head && head->use_bias, "new affine variance head missing its bias");
    const size_t bias_offset = static_cast<size_t>(head->in_channels) * head->out_channels;
    require(head->getWeightsSize() == bias_offset + head->out_channels, "variance bias packing changed");
    std::fill_n(head->getWeights(), head->getWeightsSize(), 0.f);
    auto* features = model->getLayerByName("vae_conv/enc/logvar_pre_conv");
    auto* normalization = model->getLayerByName("vae_conv/enc/logvar_pre_norm/gn");
    require(features && normalization, "affine variance features missing");
    std::fill_n(features->getWeights(), features->getWeightsSize(), 0.f);
    for (int channel = 0; channel < normalization->in_channels; ++channel) {
        normalization->getWeights()[channel] = 1.f;
        normalization->getWeights()[normalization->in_channels + channel] = 0.f;
    }
    for (int channel = 0; channel < head->out_channels; ++channel) head->getWeights()[bias_offset + channel] = -1.f;
    model->forwardPass(input, false);
    for (float value : model->getTensor("vae_conv/logvar"))
        require(std::abs(value + 1.f) < 1e-6f, "variance head cannot express a constant negative logvar");

    Optimizer optimizer;
    optimizer.type = OptimizerType::ADAM; optimizer.initial_lr = .001f;
    optimizer.weight_decay = 0; optimizer.decay_strategy = LRDecayStrategy::NONE;
    Model::TrainStepRequest request;
    request.float_inputs["__input__"] = &input; request.target = &input;
    request.optimizer = &optimizer; request.learning_rate = .001f;
    const auto result = model->trainStep(request);
    require(result && std::isfinite(result->loss), "affine variance training failed");
    double bias_gradient = 0, bias_change = 0;
    for (int channel = 0; channel < head->out_channels; ++channel) {
        bias_gradient += std::abs(head->grad_weights[bias_offset + channel]);
        bias_change += std::abs(head->getWeights()[bias_offset + channel] + 1.f);
    }
    require(bias_gradient > 1e-9 && bias_change > 1e-7,
        "reconstruction did not train the independent variance offsets");
    require(std::all_of(head->grad_weights.begin(), head->grad_weights.begin() + bias_offset,
        [](float value) { return value == 0.f; }), "zero-feature variance test received coefficient gradients");

    using namespace Mimir::Serialization;
    const auto root = std::filesystem::temp_directory_path() /
        ("mimir_logvar_" + rank + "_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup { std::filesystem::path root; ~Cleanup() { std::filesystem::remove_all(root); } } cleanup{root};
    for (bool biased : {false, true}) {
        auto& source = biased ? model : legacy;
        if (!biased) source->modelConfig.erase("logvar_head_bias"); // Historical configurations omit the option.
        const auto checkpoint = root / (biased ? "affine" : "legacy");
        SaveOptions save; save.format = CheckpointFormat::RawFolder;
        save.save_optimizer = save.save_encoder = save.save_tokenizer = false;
        std::string error;
        require(save_checkpoint(*source, checkpoint.string(), save, &error), "posterior save failed: " + error);
        auto restored = ModelArchitectures::create("vae_conv", source->modelConfig);
        restored->allocateParams();
        LoadOptions load; load.format = CheckpointFormat::RawFolder;
        load.load_optimizer = load.load_encoder = load.load_tokenizer = false;
        require(load_checkpoint(*restored, checkpoint.string(), load, &error), "posterior reload failed: " + error);
        require(restored->getLayerByName("vae_conv/enc/logvar")->use_bias == biased,
            "reload changed the old/new variance head topology");
        const auto expected = source->forwardPass(input, false), actual = restored->forwardPass(input, false);
        require(difference(actual, expected) < 1e-6, "reload changed deterministic reconstruction");
        require(difference(restored->getTensor("vae_conv/logvar"), source->getTensor("vae_conv/logvar")) < 1e-6,
            "reload changed learned variance parameters");
    }
    std::cout << "LOGVAR_AFFINE " << rank << " bias_grad=" << bias_gradient << " bias_change=" << bias_change << '\n';
}
} // namespace

int main() {
    try {
        runtimeGradient();
        for (const auto& rank : {"2d", "3d", "4d"}) {
            for (const auto& normalization : {"none", "groupnorm"})
                for (bool stochastic : {true, false})
                    reconstructionGradientAndTraining(rank, normalization, stochastic);
            logvarOnlyIntervention(rank);
            affineVarianceAndReload(rank);
        }
        std::cout << "VAE_CONV_LOGVAR_RECONSTRUCTION_OK\n";
    } catch (const std::exception& error) {
        std::cerr << "VAE logvar test failed: " << error.what() << '\n';
        return 1;
    }
}
