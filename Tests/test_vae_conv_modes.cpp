#include "test_utils.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "Models/Vision/VAEConvModel.hpp"
#include "Serialization/LayerMetadata.hpp"
#include "Serialization/Serialization.hpp"
#include "runtimes/ConvolutionND.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void sameJson(const json& actual, const json& expected, const std::string& label) {
    if (actual != expected)
        throw std::runtime_error(label + " (expected -> actual):\n" + json::diff(expected, actual).dump(2));
}

void close(const std::vector<float>& actual, const std::vector<float>& expected,
           const std::string& label, float tolerance = 8e-5f) {
    require(actual.size() == expected.size(), label + ": size changed");
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]) && std::isfinite(expected[i]), label + ": nonfinite value");
        if (std::abs(actual[i] - expected[i]) > tolerance * std::max(1.f, std::abs(expected[i])))
            throw std::runtime_error(label + " at " + std::to_string(i) + ": " +
                                     std::to_string(actual[i]) + " vs " + std::to_string(expected[i]));
    }
}

void identical(const std::vector<float>& actual, const std::vector<float>& expected,
               const std::string& label) {
    close(actual, expected, label, 0.f);
    require(actual.empty() || std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0,
            label + ": legacy 2D bits changed");
}

json fixture(bool rich = false) {
    return {{"dtype", "float32"}, {"image_w", 8}, {"image_h", 4}, {"image_c", rich ? 3 : 1},
            {"latent_w", 4}, {"latent_h", 2}, {"latent_c", 4}, {"base_channels", 4},
            {"resnet", rich}, {"attention", false}, {"enc_norm", rich ? "groupnorm" : "none"},
            {"dec_norm", rich ? "groupnorm" : "none"}, {"enc_gn_groups", 2}, {"dec_gn_groups", 2},
            {"stochastic_latent", false}, {"latent_spatial_context", rich},
            {"latent_context_channels", 4}, {"latent_detail_path", rich}, {"use_encoder_prior", rich}};
}

std::vector<float> image(const json& cfg) {
    std::vector<float> result(cfg.at("image_w").get<size_t>() * cfg.at("image_h").get<size_t>() *
                              cfg.at("image_c").get<size_t>());
    for (size_t i = 0; i < result.size(); ++i)
        result[i] = .31f * std::sin(float(i) * .43f) + .07f * std::cos(float(i) * .17f);
    return result;
}

const Layer* findLayer(const Model& model, const std::string& name) {
    const auto& layers = model.getLayers();
    const auto found = std::find_if(layers.begin(), layers.end(), [&](const Layer& layer) { return layer.name == name; });
    return found == layers.end() ? nullptr : &*found;
}

void copyParameters(const Model& source, Model& destination) {
    for (auto& layer : destination.getMutableLayers()) {
        if (!layer.getWeightsSize()) continue;
        const auto* original = findLayer(source, layer.name);
        require(original, "missing source layer " + layer.name);
        require(original->getWeightsSize() == layer.getWeightsSize(), "parameter size " + layer.name);
        std::copy_n(original->getWeights(), layer.getWeightsSize(), layer.getWeights());
    }
}

void checkTopology(const Model& model, int rank) {
    require(model.modelConfig.at("conv_mode") == std::to_string(rank) + "d", "resolved conv_mode missing");
    size_t convolutions = 0;
    for (const auto& layer : model.getLayers()) {
        if (rank != 2)
            require(layer.type_enum != LayerType::Conv2d && layer.type_enum != LayerType::ConvTranspose2d,
                    "hidden 2D convolution " + layer.name);
        if (layer.type_enum != (rank == 2 ? LayerType::Conv2d : rank == 3 ? LayerType::Conv3d : LayerType::Conv4d))
            continue;
        ++convolutions;
        if (rank == 2) continue;
        const auto geometry = RuntimeConvolutionND::checkedGeometry(layer);
        require(geometry.rank == rank && geometry.batch == 1, "wrong ND rank/batch " + layer.name);
        for (const auto* shape : {&layer.input_spatial_shape, &layer.kernel_shape, &layer.stride_shape,
                                  &layer.padding_shape, &layer.dilation_shape})
            require(shape->size() == static_cast<size_t>(rank), "incomplete ND metadata " + layer.name);
        for (int axis = 0; axis < rank - 2; ++axis)
            require(geometry.input[axis] == 1 && geometry.output[axis] == 1 && geometry.kernel[axis] == 1 &&
                    geometry.stride[axis] == 1 && geometry.padding[axis] == 0 && geometry.dilation[axis] == 1,
                    "non-singleton extra image axis " + layer.name);
        require(geometry.parameter_size == layer.params_count, "ND allocation contract " + layer.name);
    }
    require(convolutions > 0, "empty convolution graph");
}

struct BackwardResult {
    std::vector<float> output;
    std::vector<float> upstream;
    std::vector<float> input_gradient;
};

BackwardResult forwardBackward(Model& model, const std::vector<float>& input) {
    model.zeroGradients();
    BackwardResult result;
    result.output = model.forwardPass(input, true);
    const size_t latent = model.modelConfig.at("latent_dim").get<size_t>();
    require(result.output.size() == input.size() + 2 * latent, "training recon/mu/logvar packing changed");
    close(result.output, result.output, "training output");
    result.upstream.resize(result.output.size());
    for (size_t i = 0; i < result.upstream.size(); ++i)
        result.upstream[i] = .1f * std::cos(float(i) * .37f) + .025f;
    model.backwardPass(result.upstream);
    require(model.hasLastInputGradient(), "missing external HWC input gradient");
    result.input_gradient = model.getLastInputGradient();
    require(result.input_gradient.size() == input.size(), "input gradient shape changed");
    close(result.input_gradient, result.input_gradient, "input gradient");
    double encoder = 0., decoder = 0.;
    for (const auto& layer : model.getLayers()) {
        if (!layer.getWeightsSize()) continue;
        require(layer.grad_weights.size() == layer.getWeightsSize(), "gradient size " + layer.name);
        for (float value : layer.grad_weights) {
            require(std::isfinite(value), "nonfinite parameter gradient " + layer.name);
            if (layer.name.find("vae_conv/enc/") == 0) encoder += std::abs(value);
            if (layer.name.find("vae_conv/dec/") == 0) decoder += std::abs(value);
        }
    }
    require(encoder > 1e-7 && decoder > 1e-7, "reconstruction graph has no encoder/decoder gradient");
    return result;
}

void compareGradients(const Model& actual, const Model& expected, bool exact = false) {
    require(actual.getLayers().size() == expected.getLayers().size(), "layer count changed");
    for (const auto& layer : actual.getLayers()) {
        const auto* original = findLayer(expected, layer.name);
        require(original, "missing gradient source " + layer.name);
        if (exact) identical(layer.grad_weights, original->grad_weights, "2D gradient " + layer.name);
        else close(layer.grad_weights, original->grad_weights, "ND gradient " + layer.name, 3e-4f);
        close(layer.grad_bias, original->grad_bias, "bias gradient " + layer.name, exact ? 0.f : 3e-4f);
    }
}

void legacyDefaults() {
    VAEConvModel::Config native_default;
    require(native_default.conv_mode == "2d", "native default changed");
    auto cfg = fixture();
    auto implicit = ModelArchitectures::create("vae_conv", cfg);
    cfg["conv_mode"] = "2d";
    auto explicit2d = ModelArchitectures::create("vae_conv", cfg);
    require(implicit->modelConfig == explicit2d->modelConfig, "omitted mode config differs from 2D");
    checkTopology(*implicit, 2);
    const auto* transpose = implicit->getLayerByName("vae_conv/dec/up1/up");
    require(transpose && transpose->type_enum == LayerType::ConvTranspose2d && transpose->kernel_size == 4 &&
            transpose->stride == 2 && transpose->padding == 1, "legacy transposed decoder changed");
    require(!implicit->getLayerByName("vae_conv/dec/up1/conv"), "legacy decoder acquired an extra convolution");
    require(implicit->getLayers().size() == explicit2d->getLayers().size(), "2D topology changed");
    for (size_t i = 0; i < implicit->getLayers().size(); ++i) {
        json left, right;
        Mimir::Serialization::LayerMetadata::writeCommon(left, implicit->getLayers()[i]);
        Mimir::Serialization::LayerMetadata::writeCommon(right, explicit2d->getLayers()[i]);
        require(left == right, "implicit 2D metadata differs from explicit 2D");
    }
    implicit->allocateParams(); explicit2d->allocateParams();
    implicit->initializeWeights("xavier", 413u);
    copyParameters(*implicit, *explicit2d);
    const auto input = image(cfg);
    const auto a = forwardBackward(*implicit, input);
    const auto b = forwardBackward(*explicit2d, input);
    identical(a.output, b.output, "implicit/explicit 2D forward");
    identical(a.input_gradient, b.input_gradient, "implicit/explicit 2D backward");
    compareGradients(*implicit, *explicit2d, true);
    identical(implicit->forwardPass(input, false), explicit2d->forwardPass(input, false), "2D inference");
}

void finiteDifference(Model& model, const std::vector<float>& input, const std::vector<float>& upstream) {
    for (const char* name : {"vae_conv/enc/conv_in", "vae_conv/dec/out"}) {
        auto* layer = model.getLayerByName(name);
        require(layer && !layer->grad_weights.empty(), "missing numerical derivative layer");
        size_t index = 0;
        for (size_t i = 1; i < layer->grad_weights.size(); ++i)
            if (std::abs(layer->grad_weights[i]) > std::abs(layer->grad_weights[index])) index = i;
        const float analytic = layer->grad_weights[index];
        require(std::abs(analytic) > 1e-8f, "zero derivative fixture " + std::string(name));
        const float old = layer->getWeights()[index], epsilon = .001f;
        auto loss = [&]() {
            const auto output = model.forwardPass(input, true);
            double value = 0.;
            for (size_t i = 0; i < output.size(); ++i) value += double(output[i]) * upstream[i];
            return value;
        };
        layer->getWeights()[index] = old + epsilon;
        const double plus = loss();
        layer->getWeights()[index] = old - epsilon;
        const double minus = loss();
        layer->getWeights()[index] = old;
        const double numeric = (plus - minus) / (2 * epsilon);
        require(std::abs(analytic - numeric) < .002 + .025 * std::abs(numeric),
                "ND finite difference failed " + std::string(name));
    }
}

void decoderParity(Model& full, const json& cfg, int rank, const std::vector<float>& input) {
    const auto expected = full.forwardPass(input, false);
    require(expected.size() == input.size(), "inference must expose HWC reconstruction only");
    const auto latent = full.getTensor(cfg.value("use_encoder_prior", false) ? "vae_conv/z_biased" : "vae_conv/z");
    auto decoder = ModelArchitectures::create("vae_conv_decode", cfg);
    checkTopology(*decoder, rank);
    decoder->allocateParams();
    copyParameters(full, *decoder);
    close(decoder->forwardPass(latent, false), expected, "decoder-only parity", 2e-5f);
}

struct TemporaryDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("mimir_vae_conv_modes_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ~TemporaryDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
};

void checkpointRoundtrip(Model& source, const std::vector<float>& input, int rank) {
    using namespace Mimir::Serialization;
    TemporaryDirectory directory;
    const auto expected = source.forwardPass(input, false);
    for (const auto format : {CheckpointFormat::RawFolder, CheckpointFormat::SafeTensors}) {
        const auto path = directory.path / (format == CheckpointFormat::RawFolder ? "raw" : "model.safetensors");
        SaveOptions save; save.format = format; save.save_encoder = save.save_tokenizer = false;
        save.include_git_info = false;
        std::string error;
        require(save_checkpoint(source, path.string(), save, &error), "save failed: " + error);
        auto restored = ModelArchitectures::create("vae_conv", source.modelConfig);
        restored->allocateParams();
        LoadOptions load; load.format = format; load.load_encoder = load.load_tokenizer = false;
        load.strict_mode = true;
        require(load_checkpoint(*restored, path.string(), load, &error), "strict load failed: " + error);
        sameJson(restored->modelConfig, source.modelConfig, "checkpoint mode/config changed");
        checkTopology(*restored, rank);
        require(restored->getLayers().size() == source.getLayers().size(), "checkpoint lost graph nodes");
        for (size_t i = 0; i < source.getLayers().size(); ++i) {
            const auto& original = source.getLayers()[i];
            const auto& loaded = restored->getLayers()[i];
            json before, after;
            LayerMetadata::writeCommon(before, original); LayerMetadata::writeCommon(after, loaded);
            require(before == after, "checkpoint structural metadata " + original.name);
            require(original.getWeightsSize() == loaded.getWeightsSize(), "checkpoint weight shape");
            for (size_t j = 0; j < original.getWeightsSize(); ++j)
                require(original.getWeights()[j] == loaded.getWeights()[j], "checkpoint weight changed");
        }
        close(restored->forwardPass(input, false), expected, "checkpoint prediction", 1e-6f);

        // Import a full checkpoint into an intentionally smaller decoder graph.
        // Its external input/output routing belongs to the prebuilt decoder.
        auto encoder_config = source.modelConfig;
        encoder_config["inference_output_tensor"] = "vae_conv/mu";
        auto encoder = ModelArchitectures::create("vae_conv", encoder_config);
        encoder->allocateParams();
        load.apply_model_config = false;
        require(load_checkpoint(*encoder, path.string(), load, &error), "encoder checkpoint load failed: " + error);
        const auto encoded = encoder->forwardPass(input, false);
        require(encoded.size() == source.modelConfig.at("latent_dim").get<size_t>(), "encoder output is not latent-only");
        close(encoded, source.getTensor("vae_conv/mu"), "checkpoint encoder mu", 1e-6f);

        auto decoder_config_input = source.modelConfig;
        decoder_config_input["inference_output_tensor"] = "x";
        auto decoder = ModelArchitectures::create("vae_conv_decode", decoder_config_input);
        decoder->allocateParams();
        const auto decoder_config = decoder->modelConfig;
        std::vector<json> decoder_metadata;
        for (const auto& layer : decoder->getLayers()) {
            json metadata; LayerMetadata::writeCommon(metadata, layer);
            decoder_metadata.push_back(std::move(metadata));
        }
        load.strict_mode = false;
        load.apply_model_config = false;
        load.load_layer_metadata = false;
        require(load_checkpoint(*decoder, path.string(), load, &error), "decoder subset load failed: " + error);
        require(decoder->modelConfig == decoder_config, "full checkpoint replaced decoder config");
        require(decoder->getLayers().back().output == "x", "decoder reconstruction no longer exposes x");
        for (size_t i = 0; i < decoder->getLayers().size(); ++i) {
            json metadata; LayerMetadata::writeCommon(metadata, decoder->getLayers()[i]);
            require(metadata == decoder_metadata[i], "full checkpoint replaced decoder routing/metadata");
        }
        const auto latent = source.getTensor(source.modelConfig.value("use_encoder_prior", false)
                                                ? "vae_conv/z_biased" : "vae_conv/z");
        close(decoder->forwardPass(latent, false), expected, "checkpoint full-to-decoder prediction", 1e-6f);
        auto operational_latent = encoded;
        if (source.modelConfig.value("use_encoder_prior", false)) {
            const auto& prior = source.getTensor("vae_conv/prior_bias_out");
            require(prior.size() == operational_latent.size(), "learned prior shape changed");
            for (size_t i = 0; i < prior.size(); ++i) operational_latent[i] += prior[i];
        }
        close(decoder->forwardPass(operational_latent, false), expected, "checkpoint encode/decode prediction", 1e-6f);
    }
}

void singletonParity(int rank, bool rich) {
    auto cfg = fixture(rich);
    cfg["decoder_upsample"] = "nearest_conv";
    cfg["conv_mode"] = "2d";
    auto source = ModelArchitectures::create("vae_conv", cfg);
    source->allocateParams(); source->initializeWeights("xavier", rich ? 197u : 179u);
    cfg["conv_mode"] = std::to_string(rank) + "d";
    auto nd = ModelArchitectures::create("vae_conv", cfg);
    checkTopology(*nd, rank);
    nd->allocateParams(); copyParameters(*source, *nd);
    const auto input = image(cfg);
    const auto expected = forwardBackward(*source, input);
    const auto actual = forwardBackward(*nd, input);
    close(actual.output, expected.output, "singleton ND forward");
    close(actual.input_gradient, expected.input_gradient, "singleton ND input gradient", 3e-4f);
    compareGradients(*nd, *source);
    if (!rich) finiteDifference(*nd, input, actual.upstream);
    decoderParity(*nd, cfg, rank, input);
    if (rich) checkpointRoundtrip(*nd, input, rank);
    std::cout << "VAE_CONV_SINGLETON_PARITY rank=" << rank << " rich=" << rich << '\n';
}

void modesAndDefaults() {
    for (const auto& alias : std::vector<std::pair<std::string, int>>{
            {"2d", 2}, {"Conv2d", 2}, {"2", 2}, {"3D", 3}, {"Conv3d", 3}, {"3", 3},
            {"4d", 4}, {"Conv4d", 4}, {"4", 4}}) {
        auto cfg = fixture(); cfg["conv_mode"] = alias.first;
        for (const auto* architecture : {"vae_conv", "vae_conv_decode"}) {
            auto model = ModelArchitectures::create(architecture, cfg);
            checkTopology(*model, alias.second);
            const auto* up = model->getLayerByName("vae_conv/dec/up1/up");
            require(up, "default decoder lost upsample");
            require(up->type_enum == (alias.second == 2 ? LayerType::ConvTranspose2d : LayerType::UpsampleNearest),
                    "default ND decoder did not resolve to native resize+convolution");
            require(model->modelConfig.at("decoder_upsample") ==
                        (alias.second == 2 ? "conv_transpose" : "nearest_conv"), "effective decoder mode missing");
        }
    }
    for (const auto* invalid : {"", "1d", "5d", "volume"}) {
        for (const auto* architecture : {"vae_conv", "vae_conv_decode"}) {
            auto cfg = fixture(); cfg["conv_mode"] = invalid;
            bool rejected = false;
            try { (void)ModelArchitectures::create(architecture, cfg); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "invalid convolution mode accepted");
        }
    }
    for (int rank : {3, 4}) {
        auto cfg = fixture(); cfg["conv_mode"] = std::to_string(rank) + "d";
        auto model = ModelArchitectures::create("vae_conv", cfg);
        model->allocateParams(); model->initializeWeights("xavier", 521u);
        forwardBackward(*model, image(cfg));
        decoderParity(*model, cfg, rank, image(cfg));
        // Other image upsamplers also remain native ND convolution graphs.
        for (const auto* upsample : {"bilinear_conv", "pixel_shuffle"}) {
            cfg["decoder_upsample"] = upsample;
            auto alternate = ModelArchitectures::create("vae_conv", cfg);
            checkTopology(*alternate, rank);
            alternate->allocateParams(); alternate->initializeWeights("xavier", 523u);
            forwardBackward(*alternate, image(cfg));
            decoderParity(*alternate, cfg, rank, image(cfg));
        }
    }
}
} // namespace

int main() {
    try {
        for (const auto* backend : {"MIMIR_DISABLE_VULKAN", "MIMIR_DISABLE_OPENCL", "MIMIR_DISABLE_CUDA", "MIMIR_DISABLE_ROCM"})
            setTestEnvironment(backend, "1");
        setTestEnvironment("MIMIR_RUNTIME_AUTOTUNE", "0");
        setTestEnvironment("MIMIR_PLANNER_FUSION", "0");
        setTestEnvironment("MIMIR_PLANNER_BUFFER_REUSE", "0");
        legacyDefaults();
        modesAndDefaults();
        for (int rank : {3, 4}) for (bool rich : {false, true}) singletonParity(rank, rich);
        std::cout << "VAE_CONV_MODES_OK\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "VAE_CONV_MODES_FAILED: " << error.what() << '\n';
        return 1;
    }
}
