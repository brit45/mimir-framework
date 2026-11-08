#include "test_utils.hpp"
#include "Layers.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"
#include "runtimes/vulkan/VulkanRuntime.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void near(double actual, double expected, const std::string& message, double tolerance = 5e-5) {
    check(std::isfinite(actual) && std::isfinite(expected) &&
          std::abs(actual - expected) <= tolerance * std::max(1., std::abs(expected)),
          message + ": " + std::to_string(actual) + " vs " + std::to_string(expected));
}
template<class T> void compare(const std::vector<float>& actual, const std::vector<T>& expected,
                               const std::string& message, double tolerance = 5e-5) {
    check(actual.size() == expected.size(), message + " shape");
    for (size_t i = 0; i < actual.size(); ++i) near(actual[i], expected[i], message + "[" + std::to_string(i) + "]", tolerance);
}
bool sameBits(const std::vector<float>& left, const std::vector<float>& right) {
    return left.size() == right.size() && (left.empty() ||
        std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) == 0);
}

std::vector<float> forward(RuntimeRouter& router, AbstractRuntime& expected, Layer& layer,
                          const std::vector<const std::vector<float>*>& inputs) {
    check(runtimeCapabilityIsNative(expected.queryForwardCapability(layer.type_enum)), "forward capability must be native");
    std::vector<std::vector<float>> output;
    AbstractRuntime* selected = nullptr;
    check(router.dispatchForwardLayer(inputs, output, layer, true, &selected), "routed forward " + layer.type);
    check(selected == &expected, "forward silently fell back from " + std::string(expected.name()));
    check(output.size() == 1, "forward output count");
    return output[0];
}
std::vector<std::vector<float>> backward(RuntimeRouter& router, AbstractRuntime& expected, Layer& layer,
                                        const std::vector<const std::vector<float>*>& inputs,
                                        const std::vector<float>& upstream) {
    check(runtimeCapabilityIsNative(expected.queryBackwardCapability(layer.type_enum)), "backward capability must be native");
    std::vector<std::vector<float>> result;
    AbstractRuntime* selected = nullptr;
    check(router.dispatchBackwardLayer(inputs, {&upstream}, result, layer, true, &selected), "routed backward " + layer.type);
    check(selected == &expected, "backward silently fell back from " + std::string(expected.name()));
    check(result.size() == inputs.size(), "backward input count");
    return result;
}

struct ConvOracle {
    std::vector<double> output, input_gradient, weight_gradient, bias_gradient;
};
ConvOracle convolutionOracle(const Layer& layer, const std::vector<float>& input,
                              const std::vector<float>& upstream) {
    const int channels = layer.in_features, kernel = layer.kernel_size, dilation = layer.dilation;
    const size_t tokens = input.size() / static_cast<size_t>(channels);
    ConvOracle result{std::vector<double>(input.size()), std::vector<double>(input.size()),
                      std::vector<double>(layer.getWeightsSize()), std::vector<double>(layer.use_bias ? channels : 0)};
    const size_t matrix = static_cast<size_t>(channels) * kernel;
    for (size_t token = 0; token < tokens; ++token) for (int channel = 0; channel < channels; ++channel) {
        const size_t destination = token * channels + channel;
        for (int tap = 0; tap < kernel; ++tap) {
            const size_t lag = static_cast<size_t>(tap) * dilation;
            if (lag > token) continue;
            const size_t source = (token - lag) * channels + channel;
            const size_t parameter = static_cast<size_t>(channel) * kernel + tap;
            result.output[destination] += double(input[source]) * layer.getWeights()[parameter];
            result.input_gradient[source] += double(upstream[destination]) * layer.getWeights()[parameter];
            result.weight_gradient[parameter] += double(upstream[destination]) * input[source];
        }
        if (layer.use_bias) {
            result.output[destination] += layer.getWeights()[matrix + channel];
            result.weight_gradient[matrix + channel] += upstream[destination];
            result.bias_gradient[channel] += upstream[destination];
        }
    }
    return result;
}

Layer convolution(int channels, int kernel, int dilation, bool bias) {
    Layer layer("causal_depthwise", "CausalDepthwiseConv1d", 0);
    layer.in_features = channels; layer.out_features = channels;
    layer.kernel_size = kernel; layer.dilation = dilation; layer.use_bias = bias;
    layer.seq_len = 2; // A stale planning hint must not limit the runtime sequence.
    layer.params_count = static_cast<size_t>(channels) * (kernel + (bias ? 1 : 0));
    layer.weights.resize(layer.params_count);
    for (size_t i = 0; i < layer.weights.size(); ++i) layer.weights[i] = .2f * std::sin(float(i + 1) * .43f);
    return layer;
}

void finiteDifferences(RuntimeRouter& router, AbstractRuntime& runtime, Layer& layer,
                       std::vector<float> input, const std::vector<float>& upstream,
                       const std::vector<float>& input_gradient, const std::vector<float>& weight_gradient) {
    auto loss = [&]() {
        const auto output = forward(router, runtime, layer, {&input});
        double result = 0;
        for (size_t i = 0; i < output.size(); ++i) result += double(output[i]) * upstream[i];
        return result;
    };
    const float epsilon = .001f;
    for (size_t index : {size_t(0), layer.getWeightsSize() / 2, layer.getWeightsSize() - 1}) {
        const float old = layer.getWeights()[index];
        layer.getWeights()[index] = old + epsilon; const double plus = loss();
        layer.getWeights()[index] = old - epsilon; const double minus = loss();
        layer.getWeights()[index] = old;
        near((plus - minus) / (2 * epsilon), weight_gradient[index], "finite difference dW", 4e-4);
    }
    for (size_t index : {size_t(0), input.size() / 2, input.size() - 1}) {
        const float old = input[index];
        input[index] = old + epsilon; const double plus = loss();
        input[index] = old - epsilon; const double minus = loss();
        input[index] = old;
        near((plus - minus) / (2 * epsilon), input_gradient[index], "finite difference dX", 4e-4);
    }
}

void convolutionCase(RuntimeRouter& router, AbstractRuntime& runtime, int tokens, int channels,
                     int kernel, int dilation, bool bias, bool numerical) {
    auto layer = convolution(channels, kernel, dilation, bias);
    std::vector<float> input(static_cast<size_t>(tokens) * channels), upstream(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = .3f * std::sin(float(i + 1) * .37f);
        upstream[i] = .2f * std::cos(float(i + 1) * .23f);
    }
    const auto oracle = convolutionOracle(layer, input, upstream);
    const auto output = forward(router, runtime, layer, {&input});
    compare(output, oracle.output, "conv forward");
    const auto first = backward(router, runtime, layer, {&input}, upstream);
    compare(first[0], oracle.input_gradient, "conv dX");
    compare(layer.grad_weights, oracle.weight_gradient, "conv dW");
    compare(layer.grad_bias, oracle.bias_gradient, "conv db");
    const auto gradients = layer.grad_weights;
    backward(router, runtime, layer, {&input}, upstream);
    for (size_t i = 0; i < gradients.size(); ++i) near(layer.grad_weights[i], 2 * oracle.weight_gradient[i], "gradient accumulation");
    for (size_t i = 0; i < layer.grad_bias.size(); ++i) near(layer.grad_bias[i], 2 * oracle.bias_gradient[i], "bias accumulation");
    if (numerical) finiteDifferences(router, runtime, layer, input, upstream, first[0], gradients);

    // Equal prefix, arbitrary changed future: every preceding output is bitwise stable.
    if (tokens > 1) {
        const size_t cut = static_cast<size_t>(tokens / 2);
        auto changed = input;
        for (size_t i = (cut + 1) * channels; i < changed.size(); ++i) changed[i] += 3.25f;
        const auto future = forward(router, runtime, layer, {&changed});
        for (size_t i = 0; i < (cut + 1) * channels; ++i)
            check(output[i] == future[i], "convolution leaked future input");
        auto prefix_gradient = upstream;
        std::fill(prefix_gradient.begin() + (cut + 1) * channels, prefix_gradient.end(), 0.f);
        const auto dx = backward(router, runtime, layer, {&input}, prefix_gradient);
        for (size_t i = (cut + 1) * channels; i < dx[0].size(); ++i)
            check(dx[0][i] == 0.f, "convolution leaked prefix gradient into future");
    }
    // out_features=0 means the same channel count, not an absent output.
    layer.out_features = 0;
    compare(forward(router, runtime, layer, {&input}), oracle.output, "inferred output channels");
}

void convolutionImpulse(RuntimeRouter& router, AbstractRuntime& runtime) {
    auto layer = convolution(3, 3, 2, false);
    layer.weights = {1,2,3, 4,5,6, 7,8,9};
    std::vector<float> input(7 * 3, 0.f);
    input[1] = 2.f;
    const auto output = forward(router, runtime, layer, {&input});
    for (size_t token = 0; token < 7; ++token) for (size_t channel = 0; channel < 3; ++channel) {
        const float expected = channel == 1 && token % 2 == 0 && token <= 4 ? 2.f * layer.weights[3 + token / 2] : 0.f;
        near(output[token * 3 + channel], expected, "past impulse/channel independence", 0.);
    }
}

void rejectionAtomicity(AbstractRuntime& runtime) {
    auto layer = convolution(3, 3, 2, true);
    std::vector<float> input(21, .5f), upstream(21, .3f);
    layer.grad_weights.assign(layer.getWeightsSize(), .125f); layer.grad_bias.assign(3, .25f);
    const auto original_weights = layer.weights, original_dw = layer.grad_weights, original_db = layer.grad_bias;
    const std::vector<std::vector<float>> sentinel{{17.f, -31.f}};
    auto rejectForward = [&]() {
        auto output = sentinel;
        check(!runtime.forwardLayer({&input}, output, layer, true), "invalid convolution forward accepted");
        check(output == sentinel && sameBits(layer.grad_weights, original_dw) && sameBits(layer.grad_bias, original_db), "failed forward mutated outputs/gradients");
    };
    auto rejectBackward = [&]() {
        auto output = sentinel;
        const auto before_dw = layer.grad_weights, before_db = layer.grad_bias;
        check(!runtime.backwardLayer({&input}, {&upstream}, output, layer, true), "invalid convolution backward accepted");
        check(output == sentinel && sameBits(layer.grad_weights, before_dw) && sameBits(layer.grad_bias, before_db), "failed backward mutated outputs/gradients");
    };
    input[0] = std::numeric_limits<float>::quiet_NaN(); rejectForward(); rejectBackward(); input[0] = .5f;
    layer.weights[0] = std::numeric_limits<float>::infinity(); rejectForward(); rejectBackward(); layer.weights = original_weights;
    upstream[1] = std::numeric_limits<float>::quiet_NaN(); rejectBackward(); upstream[1] = .3f;
    layer.kernel_size = 0; rejectForward(); rejectBackward(); layer.kernel_size = 3;
    layer.out_features = 4; rejectForward(); rejectBackward(); layer.out_features = 3;
    input.push_back(.5f); rejectForward(); rejectBackward(); input.pop_back();
    layer.weights[0] = std::numeric_limits<float>::max(); input[0] = 2.f; rejectForward();
    upstream[0] = 2.f; rejectBackward();
    layer.weights = original_weights; input[0] = .5f; upstream.assign(21, 1.f);
    // No delta itself overflows: adding it to an existing finite gradient does.
    layer.grad_weights[0] = std::numeric_limits<float>::max(); input[0] = std::numeric_limits<float>::max();
    rejectBackward();
}

double sigmoid(double value) {
    if (value >= 0.) return 1. / (1. + std::exp(-value));
    const double exponential = std::exp(value); return exponential / (1. + exponential);
}
void swiglu(RuntimeRouter& router, AbstractRuntime& runtime, CpuRuntime& cpu) {
    Layer fused("fused", "SwiGLU", 0), silu("silu", "SiLU", 0), multiply("product", "Multiply", 0);
    fused.use_bias = false;
    std::vector<float> gate(257), value(257), upstream(257);
    const float special[] = {-100.f,-30.f,-3.f,-1.f,-0.f,0.f,.1f,1.f,3.f,30.f,100.f};
    for (size_t i = 0; i < gate.size(); ++i) {
        gate[i] = i < sizeof(special)/sizeof(*special) ? special[i] : float(int(i % 31) - 15) * .37f;
        value[i] = float(int(i % 17) - 8) * .125f; upstream[i] = .1f * std::cos(float(i) * .29f);
    }
    const auto output = forward(router, runtime, fused, {&gate, &value});
    const auto derivatives = backward(router, runtime, fused, {&gate, &value}, upstream);
    for (size_t i = 0; i < gate.size(); ++i) {
        const double s = sigmoid(gate[i]);
        near(output[i], double(gate[i]) * s * value[i], "SwiGLU y");
        near(derivatives[0][i], double(upstream[i]) * value[i] * (s + double(gate[i]) * s * (1 - s)), "SwiGLU dG");
        near(derivatives[1][i], double(upstream[i]) * gate[i] * s, "SwiGLU dU");
    }
    std::vector<std::vector<float>> activated, explicit_output, product_gradient, gate_gradient;
    check(cpu.forwardLayer({&gate}, activated, silu, true), "explicit SiLU");
    check(cpu.forwardLayer({&activated[0], &value}, explicit_output, multiply, true), "explicit multiply");
    check(cpu.backwardLayer({&activated[0], &value}, {&upstream}, product_gradient, multiply, true), "explicit multiply backward");
    check(cpu.backwardLayer({&gate}, {&product_gradient[0]}, gate_gradient, silu, true), "explicit SiLU backward");
    compare(output, explicit_output[0], "fused/composite output");
    compare(derivatives[0], gate_gradient[0], "fused/composite dG");
    compare(derivatives[1], product_gradient[1], "fused/composite dU");
    auto loss = [&]() {
        const auto y = forward(router, runtime, fused, {&gate, &value});
        double result = 0.; for (size_t i = 0; i < y.size(); ++i) result += double(y[i]) * upstream[i];
        return result;
    };
    for (size_t index : {size_t(5), size_t(16), size_t(123)}) {
        const float old_gate = gate[index], epsilon = .001f;
        gate[index] = old_gate + epsilon; const double plus_gate = loss();
        gate[index] = old_gate - epsilon; const double minus_gate = loss(); gate[index] = old_gate;
        near((plus_gate - minus_gate) / (2 * epsilon), derivatives[0][index], "SwiGLU finite difference dG", 4e-4);
        const float old_value = value[index];
        value[index] = old_value + epsilon; const double plus_value = loss();
        value[index] = old_value - epsilon; const double minus_value = loss(); value[index] = old_value;
        near((plus_value - minus_value) / (2 * epsilon), derivatives[1][index], "SwiGLU finite difference dU", 4e-4);
    }
    const float largest = std::numeric_limits<float>::max();
    gate = {-largest, largest, -1000.f, 1000.f, 0.f};
    value = {1.f, .5f, largest, .001f, largest}; upstream.assign(gate.size(), .125f);
    const auto extreme = forward(router, runtime, fused, {&gate, &value});
    const auto extreme_gradient = backward(router, runtime, fused, {&gate, &value}, upstream);
    for (size_t i = 0; i < gate.size(); ++i) {
        const double s = sigmoid(gate[i]);
        near(extreme[i], double(gate[i]) * s * value[i], "finite extreme SwiGLU");
        near(extreme_gradient[0][i], double(upstream[i]) * value[i] * (s + double(gate[i]) * s * (1 - s)), "finite extreme dG");
        near(extreme_gradient[1][i], double(upstream[i]) * gate[i] * s, "finite extreme dU");
    }
    const std::vector<std::vector<float>> sentinel{{42.f}};
    auto reject = [&](const std::vector<const std::vector<float>*>& inputs, bool backward_call) {
        auto result = sentinel;
        const bool accepted = backward_call ? runtime.backwardLayer(inputs, {&upstream}, result, fused, true) :
                                              runtime.forwardLayer(inputs, result, fused, true);
        check(!accepted && result == sentinel && fused.grad_weights.empty(), "invalid SwiGLU mutated output");
    };
    reject({&gate}, false); reject({&gate}, true);
    auto short_value = value; short_value.pop_back(); reject({&gate, &short_value}, false); reject({&gate, &short_value}, true);
    gate[0] = std::numeric_limits<float>::quiet_NaN(); reject({&gate, &value}, false); reject({&gate, &value}, true);
    gate[0] = largest; value[0] = 2.f; reject({&gate, &value}, false);
    gate[0] = 1.f; value[0] = largest; upstream[0] = largest; reject({&gate, &value}, true);
}
} // namespace

int main(int argc, char** argv) {
    try {
        const bool gpu = argc > 1 && std::string(argv[1]) == "--vulkan";
        setTestEnvironment("MIMIR_RUNTIME_AUTOTUNE", "0");
        RuntimeConfig cfg; cfg.conv_enabled = cfg.linear_enabled = true; cfg.conv_min_ops = cfg.linear_min_ops = 0;
        CpuRuntime cpu; check(cpu.initialize(cfg), "CPU initialize");
        AbstractRuntime* runtime = &cpu;
#ifdef ENABLE_VULKAN
        VulkanRuntime vulkan;
        if (gpu) {
            if (!vulkan.initialize(cfg)) { std::cout << "SKIP Vulkan unavailable\n"; return 77; }
            runtime = &vulkan;
        }
#else
        if (gpu) { std::cout << "SKIP Vulkan not compiled\n"; return 77; }
#endif
        auto& router = RuntimeRouter::instance(); router.setActivators({}, {}, {}, {}, {});
        router.setRuntimes(nullptr, nullptr, gpu ? runtime : nullptr, nullptr, &cpu);
        convolutionImpulse(router, *runtime);
        for (int tokens : {1,3,7}) for (int channels : {1,7,13}) for (bool bias : {false,true})
            convolutionCase(router, *runtime, tokens, channels, tokens == 3 ? 5 : 3, tokens == 7 ? 1 : 2,
                            bias, tokens == 7 && channels == 7 && bias);
        convolutionCase(router, *runtime, 9, 5, 1, 1, true, false);
        convolutionCase(router, *runtime, 7, 3, 5, std::numeric_limits<int>::max(), false, false);
        rejectionAtomicity(*runtime);
        swiglu(router, *runtime, cpu);
        router.setRuntimes(nullptr, nullptr, nullptr, nullptr, nullptr);
        std::cout << "CAUSAL_SEQUENCE_RUNTIME_OK backend=" << (gpu ? "VULKAN" : "CPU") << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << "CAUSAL_SEQUENCE_RUNTIME_FAILED: " << error.what() << '\n'; return 1; }
}
