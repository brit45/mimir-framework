#include "Model.hpp"
#include "Planning/Planner.hpp"
#include "Serialization/LayerMetadata.hpp"
#include "Serialization/Serialization.hpp"
#include "runtimes/ConvolutionND.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"
#include "runtimes/cpu/ConvolutionNDKernels.hpp"
#include "runtimes/cpu/SimdDispatch.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using RuntimeConvolutionND::Geometry;

static void require(bool condition, const char* expression, int line) {
    if (!condition) throw std::runtime_error(std::string("ConvND check at line ") +
        std::to_string(line) + ": " + expression);
}
#define REQUIRE(value) require(bool(value), #value, __LINE__)

static void near(double actual, double expected, double tolerance = 4e-5) {
    REQUIRE(std::isfinite(actual));
    REQUIRE(std::abs(actual - expected) <= tolerance * std::max(1.0, std::abs(expected)));
}

static void configure(Layer& layer, int rank, bool strided, bool bias) {
    layer.in_channels = 4; layer.out_channels = 6; layer.groups = 2; layer.use_bias = bias;
    layer.input_spatial_shape = rank == 3 ? std::vector<int>{3,4,19} : std::vector<int>{2,3,4,9};
    layer.kernel_shape = rank == 3 ? std::vector<int>{2,2,3} : std::vector<int>{2,1,2,2};
    layer.stride_shape = rank == 3 ? std::vector<int>{1,2,strided ? 2 : 1} : std::vector<int>{1,1,2,strided ? 2 : 1};
    layer.padding_shape = rank == 3 ? std::vector<int>{1,0,1} : std::vector<int>{1,0,1,0};
    layer.dilation_shape = rank == 3 ? std::vector<int>{1,1,2} : std::vector<int>{1,2,1,2};
}

struct Reference {
    std::vector<double> output, input_gradient, parameter_gradient;
};

// Independent scalar double reference: one output at a time, explicit complete
// multidimensional coordinates. This does not share the CPU row-slice mapping.
static Reference reference(const Layer& layer, const Geometry& g,
                           const std::vector<float>& input, const std::vector<float>& upstream) {
    Reference result{std::vector<double>(g.output_size), std::vector<double>(g.input_size),
                     std::vector<double>(g.parameter_size)};
    for (int axis = 0; axis < g.rank; ++axis) {
        const int expected = (g.input[axis] + 2*g.padding[axis] -
                              g.dilation[axis]*(g.kernel[axis]-1) - 1)/g.stride[axis] + 1;
        REQUIRE(g.output[axis] == expected);
    }
    for (int n = 0; n < g.batch; ++n) for (int oc = 0; oc < g.out_channels; ++oc)
        for (size_t output = 0; output < g.output_spatial; ++output) {
            const size_t output_index = (static_cast<size_t>(n)*g.out_channels + oc)*g.output_spatial + output;
            double sum = 0;
            for (int channel = 0; channel < g.in_per_group; ++channel)
                for (size_t kernel = 0; kernel < g.kernel_spatial; ++kernel) {
                    size_t input_offset = 0;
                    bool valid = true;
                    for (int axis = 0; axis < g.rank; ++axis) {
                        const int64_t position = static_cast<int64_t>(output/g.output_strides[axis]%g.output[axis])*g.stride[axis] -
                            g.padding[axis] + static_cast<int64_t>(kernel/g.kernel_strides[axis]%g.kernel[axis])*g.dilation[axis];
                        if (position < 0 || position >= g.input[axis]) { valid = false; break; }
                        input_offset += static_cast<size_t>(position)*g.input_strides[axis];
                    }
                    if (!valid) continue;
                    const int ic = (oc/g.out_per_group)*g.in_per_group + channel;
                    const size_t input_index = (static_cast<size_t>(n)*g.in_channels + ic)*g.input_spatial + input_offset;
                    const size_t weight_index = (static_cast<size_t>(oc)*g.in_per_group + channel)*g.kernel_spatial + kernel;
                    sum += double(input[input_index])*layer.getWeights()[weight_index];
                    result.input_gradient[input_index] += double(upstream[output_index])*layer.getWeights()[weight_index];
                    result.parameter_gradient[weight_index] += double(input[input_index])*upstream[output_index];
                }
            if (layer.use_bias) {
                sum += layer.getWeights()[g.weight_size + oc];
                result.parameter_gradient[g.weight_size + oc] += upstream[output_index];
            }
            result.output[output_index] = sum;
        }
    return result;
}

static std::vector<float> forward(RuntimeRouter& router, CpuRuntime& cpu, Layer& layer,
                                  const std::vector<float>& input) {
    std::vector<std::vector<float>> output;
    AbstractRuntime* selected = nullptr;
    REQUIRE(router.dispatchForwardLayer({&input}, output, layer, true, &selected));
    REQUIRE(selected == &cpu);
    REQUIRE(output.size() == 1);
    return output[0];
}

static std::vector<float> backward(RuntimeRouter& router, CpuRuntime& cpu, Layer& layer,
                                   const std::vector<float>& input, const std::vector<float>& upstream) {
    std::vector<std::vector<float>> input_gradient;
    AbstractRuntime* selected = nullptr;
    REQUIRE(router.dispatchBackwardLayer({&input}, {&upstream}, input_gradient, layer, true, &selected));
    REQUIRE(selected == &cpu);
    REQUIRE(input_gradient.size() == 1);
    return input_gradient[0];
}

static void analyticalCase(RuntimeRouter& router, CpuRuntime& cpu, int rank) {
    Layer layer("grouped_pointwise", rank == 3 ? "Conv3d" : "Conv4d", 4);
    layer.in_channels = layer.out_channels = layer.groups = 2;
    layer.input_time = 2; layer.input_depth = 2; layer.input_height = 2; layer.input_width = 3;
    layer.kernel_size = 1;
    layer.weights = {2.f, -3.f, 7.f, -4.f};
    auto g = RuntimeConvolutionND::checkedGeometry(layer);
    g = RuntimeConvolutionND::checkedGeometry(layer, 2*g.input_size);
    std::vector<float> input(g.input_size), upstream(g.output_size, 1.f);
    for (size_t i = 0; i < input.size(); ++i) input[i] = float(i)*.125f - 2.f;
    const auto output = forward(router, cpu, layer, input);
    const auto dx = backward(router, cpu, layer, input, upstream);
    for (int batch = 0; batch < 2; ++batch) for (int channel = 0; channel < 2; ++channel) {
        const float scale = channel == 0 ? 2.f : -3.f;
        const float bias = channel == 0 ? 7.f : -4.f;
        for (size_t spatial = 0; spatial < g.input_spatial; ++spatial) {
            const size_t index = (static_cast<size_t>(batch)*2 + channel)*g.input_spatial + spatial;
            near(output[index], input[index]*scale+bias, 0);
            near(dx[index], scale, 0);
        }
    }
    for (int channel = 0; channel < 2; ++channel)
        near(layer.grad_bias[channel], 2*g.output_spatial, 0);
}

static void runtimeCase(RuntimeRouter& router, CpuRuntime& cpu, int rank, bool strided, bool bias) {
    Layer layer("nd", rank == 3 ? "Conv3d" : "Conv4d", 0);
    configure(layer, rank, strided, bias);
    // Runtime may infer N from flattened storage. The static planner must leave
    // that batch unknown instead of presenting an arbitrary N=1 estimate.
    REQUIRE(Mimir::Planning::planned_output_shape(layer).empty());
    REQUIRE(Mimir::Planning::planned_tensor_bytes(layer) == 0);
    auto g = RuntimeConvolutionND::checkedGeometry(layer);
    g = RuntimeConvolutionND::checkedGeometry(layer, 2*g.input_size);
    REQUIRE(g.batch == 2);
    layer.weights.resize(g.parameter_size);
    std::vector<float> input(g.input_size), upstream(g.output_size);
    for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(float(i)*.317f)*.1f;
    for (size_t i = 0; i < upstream.size(); ++i) upstream[i] = std::cos(float(i)*.217f)*.1f;
    for (size_t i = 0; i < layer.weights.size(); ++i) layer.weights[i] = std::sin(float(i)*.137f)*.1f;
    const auto expected = reference(layer, g, input, upstream);
    REQUIRE(runtimeCapabilityIsNative(cpu.queryForwardCapability(layer.type_enum)));
    REQUIRE(runtimeCapabilityIsNative(cpu.queryBackwardCapability(layer.type_enum)));

    REQUIRE(RuntimeSimd::requestIsa(RuntimeSimd::Isa::Scalar));
    const auto scalar = forward(router, cpu, layer, input);
    RuntimeSimd::clearIsaRequest();
    const auto automatic = forward(router, cpu, layer, input);
    REQUIRE(automatic.size() == g.output_size);
    REQUIRE(std::memcmp(scalar.data(), automatic.data(), automatic.size()*sizeof(float)) == 0);
    for (size_t i = 0; i < automatic.size(); ++i) near(automatic[i], expected.output[i]);

    layer.grad_weights.assign(g.parameter_size, 0.f);
    layer.grad_bias.assign(bias ? g.out_channels : 0, 0.f);
    REQUIRE(RuntimeSimd::requestIsa(RuntimeSimd::Isa::Scalar));
    const auto scalar_dx = backward(router, cpu, layer, input, upstream);
    const auto scalar_dw = layer.grad_weights;
    const auto scalar_db = layer.grad_bias;
    RuntimeSimd::clearIsaRequest();
    layer.grad_weights.assign(g.parameter_size, .125f);
    layer.grad_bias.assign(bias ? g.out_channels : 0, .25f);
    for (int repeat = 1; repeat <= 2; ++repeat) {
        const auto dx = backward(router, cpu, layer, input, upstream);
        REQUIRE(dx.size() == g.input_size);
        REQUIRE(std::memcmp(scalar_dx.data(), dx.data(), dx.size()*sizeof(float)) == 0);
        for (size_t i = 0; i < dx.size(); ++i) near(dx[i], expected.input_gradient[i]);
        for (size_t i = 0; i < g.parameter_size; ++i)
            near(layer.grad_weights[i], .125 + repeat*expected.parameter_gradient[i]);
        for (size_t i = 0; i < layer.grad_bias.size(); ++i)
            near(layer.grad_bias[i], .25 + repeat*expected.parameter_gradient[g.weight_size+i]);
    }
    for (size_t i = 0; i < scalar_dw.size(); ++i) near(scalar_dw[i], expected.parameter_gradient[i]);
    for (size_t i = 0; i < scalar_db.size(); ++i) near(scalar_db[i], expected.parameter_gradient[g.weight_size+i]);

    auto objective = [&]() {
        const auto output = forward(router, cpu, layer, input);
        double value = 0;
        for (size_t i = 0; i < output.size(); ++i) value += double(output[i])*upstream[i];
        return value;
    };
    constexpr float epsilon = .001f;
    std::vector<size_t> parameters{0, g.weight_size/2, g.weight_size-1};
    if (bias) { parameters.push_back(g.weight_size); parameters.push_back(g.parameter_size-1); }
    for (size_t index : parameters) {
        const float saved = layer.weights[index];
        layer.weights[index] = saved + epsilon; const double plus = objective();
        layer.weights[index] = saved - epsilon; const double minus = objective();
        layer.weights[index] = saved;
        near((plus-minus)/(2*epsilon), expected.parameter_gradient[index], 2e-4);
    }
    for (size_t index : {size_t(0), g.input_size/2, g.input_size-1}) {
        const float saved = input[index];
        input[index] = saved + epsilon; const double plus = objective();
        input[index] = saved - epsilon; const double minus = objective();
        input[index] = saved;
        near((plus-minus)/(2*epsilon), expected.input_gradient[index], 2e-4);
    }

    // Invalid requests fail before mutating persistent gradient accumulators.
    const auto before = layer.grad_weights;
    std::vector<std::vector<float>> rejected;
    upstream.pop_back();
    REQUIRE(!cpu.backwardLayer({&input}, {&upstream}, rejected, layer, true));
    REQUIRE(layer.grad_weights == before);
}

static void metadataAndModelCase(int rank) {
    Model model;
    model.modelConfig["type"] = "conv_nd_test";
    model.push("nd", rank == 3 ? "Conv3d" : "Conv4d", 0);
    auto& layer = model.getMutableLayers()[0];
    configure(layer, rank, false, true);
    // Native Model training exposes and seeds its loss gradient at "x".
    layer.inputs = {"__input__"}; layer.output = "x";
    layer.shape = {2, layer.in_channels};
    layer.shape.insert(layer.shape.end(), layer.input_spatial_shape.begin(), layer.input_spatial_shape.end());
    const auto g = RuntimeConvolutionND::checkedGeometry(layer);
    std::vector<int> planned_shape{g.batch, g.out_channels};
    for (int axis = 0; axis < rank; ++axis) planned_shape.push_back(g.output[axis]);
    REQUIRE(Mimir::Planning::planned_output_shape(layer) == planned_shape);
    REQUIRE(Mimir::Planning::planned_tensor_bytes(layer) == g.output_size*sizeof(float));
    model.allocateParams();
    REQUIRE(layer.params_count == g.parameter_size && layer.getWeightsSize() == g.parameter_size);
    model.initializeWeights("xavier", 1234);
    for (size_t i = g.weight_size; i < g.parameter_size; ++i) near(layer.getWeights()[i], 0, 0);
    for (size_t i = 0; i < g.parameter_size; ++i) layer.getWeights()[i] = std::sin(float(i)*.17f)*.05f;
    std::vector<float> input(g.input_size), upstream(g.output_size);
    for (size_t i = 0; i < input.size(); ++i) input[i] = std::cos(float(i)*.31f)*.1f;
    for (size_t i = 0; i < upstream.size(); ++i) upstream[i] = std::sin(float(i)*.27f)*.1f;
    const auto expected = reference(layer, g, input, upstream);
    model.zeroGradients();
    const auto output = model.forwardPass(input, true);
    REQUIRE(output.size() == g.output_size);
    for (size_t i = 0; i < output.size(); ++i) near(output[i], expected.output[i]);
    model.backwardPass(upstream);
    REQUIRE(model.hasLastInputGradient());
    const auto& dx = model.getLastInputGradient();
    REQUIRE(dx.size() == g.input_size);
    for (size_t i = 0; i < dx.size(); ++i) near(dx[i], expected.input_gradient[i]);
    for (size_t i = 0; i < g.parameter_size; ++i) near(layer.grad_weights[i], expected.parameter_gradient[i]);

    using namespace Mimir::Serialization;
    const auto root = std::filesystem::temp_directory_path()/
        ("mimir_conv_nd_" + std::to_string(rank) + "_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    for (auto format : {CheckpointFormat::RawFolder, CheckpointFormat::SafeTensors}) {
        const auto path = root/(format == CheckpointFormat::RawFolder ? "raw" : "model.safetensors");
        SaveOptions save; save.format = format; save.save_encoder = save.save_tokenizer = false;
        save.include_git_info = false;
        std::string error;
        if (!save_checkpoint(model, path.string(), save, &error)) throw std::runtime_error(error);
        Model restored;
        LoadOptions load; load.format = format; load.load_encoder = load.load_tokenizer = false;
        if (!load_checkpoint(restored, path.string(), load, &error)) throw std::runtime_error(error);
        REQUIRE(restored.getLayers().size() == 1);
        const auto& loaded = restored.getLayers()[0];
        REQUIRE(loaded.type_enum == layer.type_enum && loaded.params_count == layer.params_count);
        REQUIRE(loaded.inputs == layer.inputs && loaded.output == layer.output);
        for (const auto& [name, member] : LayerMetadata::spatialIntegers()) {
            (void)name; REQUIRE(loaded.*member == layer.*member);
        }
        for (const auto& [name, member] : LayerMetadata::spatialVectors()) {
            (void)name; REQUIRE(loaded.*member == layer.*member);
        }
        REQUIRE(loaded.use_bias == layer.use_bias);
        const auto resumed = restored.forwardPass(input, false);
        REQUIRE(resumed.size() == output.size());
        for (size_t i = 0; i < output.size(); ++i) near(resumed[i], output[i], 1e-6);
    }
    std::filesystem::remove_all(root);
}

static void buildMixedGraph(Model& model, int rank, bool conflicting_metadata = false) {
    model.modelConfig["type"] = "conv_nd_mixed_test";
    model.modelConfig["fixture"] = {{"rank", rank}, {"purpose", "structural checkpoint parity"}};
    model.setDefaultDType("float32");
    model.push("mixed/nd", rank == 3 ? "Conv3d" : "Conv4d", 0);
    auto& convolution = model.getMutableLayers().back();
    convolution.in_channels = 2; convolution.out_channels = 3;
    convolution.input_spatial_shape = rank == 3 ? std::vector<int>{2,3,5} : std::vector<int>{2,2,3,5};
    convolution.kernel_shape = {1};
    convolution.shape = {2,2};
    convolution.shape.insert(convolution.shape.end(), convolution.input_spatial_shape.begin(), convolution.input_spatial_shape.end());
    convolution.inputs = {"__input__"}; convolution.output = "mixed/convolution";
    const auto geometry = RuntimeConvolutionND::checkedGeometry(convolution);
    std::vector<int> output_shape{geometry.batch, geometry.out_channels};
    for (int axis = 0; axis < rank; ++axis) output_shape.push_back(geometry.output[axis]);

    model.push("mixed/permute", "Permute", 0);
    auto& permute = model.getMutableLayers().back();
    permute.inputs = {"mixed/convolution"}; permute.output = "mixed/permuted";
    permute.shape = output_shape;
    for (size_t axis = 0; axis < output_shape.size(); ++axis) permute.permute_dims.push_back(static_cast<int>(axis));
    if (!conflicting_metadata) std::swap(permute.permute_dims[output_shape.size()-1], permute.permute_dims[output_shape.size()-2]);

    // The permutation changes which neighbors are normalized. A missing shape,
    // permutation, normalized width, affine flag, bias flag or epsilon therefore
    // changes the function or makes the restored graph fail validation.
    const int normalized = output_shape[output_shape.size()-2];
    model.push("mixed/norm", "LayerNorm", static_cast<size_t>(normalized));
    auto& norm = model.getMutableLayers().back();
    norm.inputs = {"mixed/permuted"}; norm.output = "x";
    norm.in_features = normalized;
    norm.affine = !conflicting_metadata;
    norm.use_bias = false;
    norm.eps = conflicting_metadata ? .7f : .023f;
    norm.shape = output_shape;
    std::swap(norm.shape[norm.shape.size()-1], norm.shape[norm.shape.size()-2]);
    model.allocateParams();
}

static void mixedGraphRoundTrip(int rank) {
    Model model;
    buildMixedGraph(model, rank);
    for (auto& layer : model.getMutableLayers()) for (size_t i = 0; i < layer.getWeightsSize(); ++i)
        layer.getWeights()[i] = layer.type_enum == LayerType::LayerNorm ? .7f + .15f*float(i) :
            std::sin(float(i)*.41f)*.2f;
    const auto geometry = RuntimeConvolutionND::checkedGeometry(model.getLayers()[0]);
    std::vector<float> input(geometry.input_size);
    for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(float(i)*.37f)*.3f;
    const auto expected = model.forwardPass(input, false);
    REQUIRE(expected.size() == geometry.output_size);

    using namespace Mimir::Serialization;
    const auto root = std::filesystem::temp_directory_path()/
        ("mimir_conv_nd_mixed_" + std::to_string(rank) + "_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    for (auto format : {CheckpointFormat::RawFolder, CheckpointFormat::SafeTensors}) {
        const auto path = root/(format == CheckpointFormat::RawFolder ? "raw" : "model.safetensors");
        SaveOptions save; save.format = format; save.save_encoder = save.save_tokenizer = false;
        save.include_git_info = false;
        std::string error;
        if (!save_checkpoint(model, path.string(), save, &error)) throw std::runtime_error(error);
        for (bool prebuilt : {false, true}) {
            Model restored;
            if (prebuilt) buildMixedGraph(restored, rank, true);
            LoadOptions load; load.format = format; load.load_encoder = load.load_tokenizer = false;
            if (!load_checkpoint(restored, path.string(), load, &error)) throw std::runtime_error(error);
            REQUIRE(restored.modelConfig == model.modelConfig);
            REQUIRE(restored.getLayers().size() == model.getLayers().size());
            for (size_t i = 0; i < model.getLayers().size(); ++i) {
                const auto& original = model.getLayers()[i]; const auto& loaded = restored.getLayers()[i];
                REQUIRE(loaded.name == original.name && loaded.type_enum == original.type_enum);
                REQUIRE(loaded.inputs == original.inputs && loaded.output == original.output);
                REQUIRE(loaded.shape == original.shape);
                REQUIRE(loaded.permute_dims == original.permute_dims);
                REQUIRE(loaded.in_features == original.in_features);
                REQUIRE(loaded.affine == original.affine && loaded.use_bias == original.use_bias);
                near(loaded.eps, original.eps, 0);
            }
            const auto actual = restored.forwardPass(input, false);
            REQUIRE(actual.size() == expected.size());
            for (size_t i = 0; i < actual.size(); ++i) near(actual[i], expected[i], 2e-5);
        }
    }
    std::filesystem::remove_all(root);
}

template<class Function> static void rejects(const Function& call) {
    bool rejected = false;
    try { call(); } catch (const std::invalid_argument&) { rejected = true; }
    REQUIRE(rejected);
}

static bool identical(const std::vector<float>& first, const std::vector<float>& second) {
    return first.size() == second.size() && (first.empty() ||
        std::memcmp(first.data(), second.data(), first.size()*sizeof(float)) == 0);
}

static void transactionalCase(CpuRuntime& cpu, int rank) {
    Layer layer("transaction", rank == 3 ? "Conv3d" : "Conv4d", 1);
    layer.in_channels = layer.out_channels = 1;
    layer.input_spatial_shape.assign(rank, 1);
    layer.kernel_shape = {1}; layer.use_bias = false;
    layer.weights = {0.f};
    layer.grad_weights = {.25f}; layer.grad_bias = {.75f};
    std::vector<float> input{1.f}, upstream{1.f}, dx{17.f,19.f}, output{23.f,29.f};
    auto rejectsBackward = [&] {
        const auto old_weights = layer.grad_weights, old_bias = layer.grad_bias, old_dx = dx;
        rejects([&] { CpuConvolutionND::backward(input, upstream, dx, layer); });
        REQUIRE(identical(layer.grad_weights, old_weights));
        REQUIRE(identical(layer.grad_bias, old_bias));
        REQUIRE(identical(dx, old_dx));
        std::vector<std::vector<float>> rejected;
        REQUIRE(!cpu.backwardLayer({&input}, {&upstream}, rejected, layer, true));
        REQUIRE(identical(layer.grad_weights, old_weights));
        REQUIRE(identical(layer.grad_bias, old_bias));
    };
    auto rejectsForward = [&] {
        const auto old_output = output;
        rejects([&] { CpuConvolutionND::forward(input, output, layer); });
        REQUIRE(identical(output, old_output));
    };

    const float maximum = std::numeric_limits<float>::max();
    input[0] = maximum; upstream[0] = 2.f;
    rejectsBackward(); // finite dX=0, overflowing dW
    input[0] = 0.f; layer.weights[0] = maximum;
    rejectsBackward(); // finite dW=0, overflowing dX
    input[0] = 1e38f; layer.weights[0] = 0.f; layer.grad_weights[0] = 3e38f;
    rejectsBackward(); // finite delta, overflowing accumulated old-gradient FMA

    // Bias accumulation is also staged together with the parameter/input grads.
    layer.use_bias = true; layer.params_count = 2; layer.weights = {0.f,0.f};
    layer.grad_weights = {.25f,3e38f}; layer.grad_bias = {3e38f};
    input[0] = 0.f; upstream[0] = 1e38f;
    rejectsBackward();

    // Invalid data and already-invalid accumulators fail before any mutation.
    layer.use_bias = false; layer.params_count = 1; layer.weights = {0.f};
    layer.grad_weights = {.25f}; layer.grad_bias = {.75f};
    input[0] = std::numeric_limits<float>::infinity(); upstream[0] = 1.f;
    rejectsBackward(); rejectsForward();
    input[0] = 1.f; upstream[0] = std::numeric_limits<float>::quiet_NaN();
    rejectsBackward();
    upstream[0] = 1.f; layer.weights[0] = std::numeric_limits<float>::infinity();
    rejectsBackward(); rejectsForward();
    layer.weights[0] = 0.f; layer.grad_weights[0] = std::numeric_limits<float>::quiet_NaN();
    rejectsBackward();
    layer.grad_weights[0] = .25f; layer.grad_bias[0] = std::numeric_limits<float>::infinity();
    rejectsBackward();

    // A forward overflow must leave an existing output buffer untouched.
    layer.grad_bias[0] = .75f; input[0] = maximum; layer.weights[0] = 2.f;
    rejectsForward();

    // Retaining the original old-gradient FMA order matters: rounding a
    // separate x*dy delta first would produce zero rather than -2^-46 here.
    layer.weights[0] = 0.f;
    input[0] = 1.f + std::ldexp(1.f,-23);
    upstream[0] = 1.f - std::ldexp(1.f,-23);
    layer.grad_weights[0] = -1.f;
    const float expected = std::fma(input[0], upstream[0], -1.f);
    REQUIRE(expected != 0.f);
    CpuConvolutionND::backward(input, upstream, dx, layer);
    REQUIRE(layer.grad_weights[0] == expected);
    REQUIRE(dx == std::vector<float>({0.f}));
    REQUIRE(layer.grad_bias == std::vector<float>({.75f}));
    CpuConvolutionND::forward(input, output, layer);
    REQUIRE(output == std::vector<float>({0.f}));
}

int main() {
    CpuRuntime cpu; RuntimeConfig config;
    REQUIRE(cpu.initialize(config));
    auto& router = RuntimeRouter::instance();
    router.setActivators({}, {}, {}, {}, {});
    router.setRuntimes(nullptr, nullptr, nullptr, nullptr, &cpu);
    for (int rank : {3,4}) transactionalCase(cpu, rank);
    for (int rank : {3,4}) analyticalCase(router, cpu, rank);
    for (int rank : {3,4}) for (bool strided : {false,true}) for (bool bias : {false,true})
        runtimeCase(router, cpu, rank, strided, bias);
    for (int rank : {3,4}) metadataAndModelCase(rank);
    for (int rank : {3,4}) mixedGraphRoundTrip(rank);
    Layer invalid("invalid", "Conv3d", 0); configure(invalid, 3, false, true);
    invalid.groups = 3; rejects([&] { RuntimeConvolutionND::checkedGeometry(invalid); });
    invalid.groups = 2; invalid.stride_shape = {0};
    rejects([&] { RuntimeConvolutionND::checkedGeometry(invalid); });
    invalid.stride_shape = {1}; invalid.kernel_shape = {-1};
    rejects([&] { RuntimeConvolutionND::checkedGeometry(invalid); });
    invalid.kernel_shape = {1}; invalid.input_spatial_shape = {std::numeric_limits<int>::max(),std::numeric_limits<int>::max(),2};
    rejects([&] { RuntimeConvolutionND::checkedGeometry(invalid); });
    invalid.input_spatial_shape = {2,3,4}; invalid.shape = {2,4,2,3,4};
    rejects([&] { RuntimeConvolutionND::checkedGeometry(invalid, 4*2*3*4); });
    invalid.shape = {2,0,2,3,4};
    rejects([&] { RuntimeConvolutionND::checkedGeometry(invalid); });
    REQUIRE(!runtimeCapabilityIsNative(cpu.queryForwardCapability(static_cast<LayerType>(-1))));
    REQUIRE(!runtimeCapabilityIsNative(cpu.queryBackwardCapability(static_cast<LayerType>(9999))));
    router.setRuntimes(nullptr, nullptr, nullptr, nullptr, nullptr);
    std::cout << "CONV_ND_RUNTIME_MODEL_ROUNDTRIP_OK\n";
    return 0;
}
