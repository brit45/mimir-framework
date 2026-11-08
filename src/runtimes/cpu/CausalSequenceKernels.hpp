#pragma once

#include "../CausalSequenceGeometry.hpp"
#include "../FiniteCheck.hpp"
#include "../GradientCommit.hpp"
#include "Parallel.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace CpuCausalSequence {
inline void requireFinite(const std::vector<float>& values) {
    if (!Mimir::RuntimeChecks::allFinite(values.data(), values.size()))
        throw std::invalid_argument("Causal sequence: non-finite tensor");
}

inline void validateWeights(const Layer& layer, const RuntimeCausalSequence::Geometry& geometry) {
    if (!layer.getWeights() || layer.getWeightsSize() != geometry.parameter_size ||
        !Mimir::RuntimeChecks::allFinite(layer.getWeights(), geometry.parameter_size))
        throw std::invalid_argument("CausalDepthwiseConv1d: invalid packed weights");
}

inline size_t workEstimate(size_t values, size_t reduction) {
    return values && reduction > std::numeric_limits<size_t>::max()/values
        ? std::numeric_limits<size_t>::max() : values*reduction;
}

inline void forward(const std::vector<float>& input, std::vector<float>& output, const Layer& layer) {
    if (input.empty()) throw std::invalid_argument("CausalDepthwiseConv1d: empty input");
    const auto geometry = RuntimeCausalSequence::checkedGeometry(layer, input.size());
    validateWeights(layer, geometry); requireFinite(input);
    const size_t channels = static_cast<size_t>(geometry.channels);
    const size_t kernel = static_cast<size_t>(geometry.kernel);
    const size_t dilation = static_cast<size_t>(geometry.dilation);
    const float* weights = layer.getWeights();
    std::vector<float> result(geometry.value_count, 0.f);
    const size_t work = workEstimate(geometry.value_count, std::min(kernel, static_cast<size_t>(geometry.steps)));
    // Each destination owns its complete lag reduction; no atomics, future
    // tokens, channel mixing or reassociated horizontal SIMD reductions.
    RuntimeParallel::forEach(result.size(), RuntimeParallel::workers(work), [&](size_t index) {
        const size_t step = index/channels, channel = index%channels;
        const size_t last = std::min(kernel-1, step/dilation);
        float sum = 0.f;
        for (size_t lag = 0; lag <= last; ++lag)
            sum = std::fma(weights[channel*kernel+lag], input[(step-lag*dilation)*channels+channel], sum);
        if (geometry.bias) sum += weights[geometry.weight_size+channel];
        result[index] = sum;
    });
    requireFinite(result);
    output = std::move(result);
}

inline void backward(const std::vector<float>& input, const std::vector<float>& grad_output,
                     std::vector<float>& grad_input, Layer& layer) {
    if (input.empty()) throw std::invalid_argument("CausalDepthwiseConv1d: empty input");
    const auto geometry = RuntimeCausalSequence::checkedGeometry(layer, input.size());
    if (grad_output.size() != geometry.value_count)
        throw std::invalid_argument("CausalDepthwiseConv1d: gradient shape mismatch");
    validateWeights(layer, geometry); requireFinite(input); requireFinite(grad_output);
    requireFinite(layer.grad_weights); requireFinite(layer.grad_bias);
    const size_t steps = static_cast<size_t>(geometry.steps), channels = static_cast<size_t>(geometry.channels);
    const size_t kernel = static_cast<size_t>(geometry.kernel), dilation = static_cast<size_t>(geometry.dilation);
    const float* weights = layer.getWeights();
    std::vector<float> result(geometry.value_count, 0.f), dw(geometry.weight_size, 0.f);
    std::vector<float> db(geometry.bias ? channels : 0, 0.f);
    const size_t work = workEstimate(geometry.value_count, std::min(kernel, steps));
    RuntimeParallel::forEach(result.size(), RuntimeParallel::workers(work), [&](size_t index) {
        const size_t step = index/channels, channel = index%channels;
        const size_t last = std::min(kernel-1, (steps-1-step)/dilation);
        float sum = 0.f;
        for (size_t lag = 0; lag <= last; ++lag)
            sum = std::fma(grad_output[(step+lag*dilation)*channels+channel], weights[channel*kernel+lag], sum);
        result[index] = sum;
    });
    RuntimeParallel::forEach(dw.size(), RuntimeParallel::workers(work), [&](size_t index) {
        const size_t channel = index/kernel, lag = index%kernel;
        // Check before multiplying: this also handles dilation/kernel extents
        // larger than the whole sequence on 32-bit hosts without overflow.
        if (lag > (steps-1)/dilation) return;
        const size_t offset = lag*dilation;
        float sum = 0.f;
        for (size_t step = offset; step < steps; ++step)
            sum = std::fma(grad_output[step*channels+channel], input[(step-offset)*channels+channel], sum);
        dw[index] = sum;
    });
    if (geometry.bias) RuntimeParallel::forEach(channels, RuntimeParallel::workers(geometry.value_count), [&](size_t channel) {
        float sum = 0.f;
        for (size_t step = 0; step < steps; ++step) sum += grad_output[step*channels+channel];
        db[channel] = sum;
    });
    requireFinite(result); requireFinite(dw); requireFinite(db);
    // Stage metadata/gradient storage as well: allocation failures and overflow
    // in old-gradient + delta must leave the live layer and dX untouched.
    Layer staged = layer;
    if (!RuntimeGradientCommit::parameters(staged, dw, db))
        throw std::invalid_argument("CausalDepthwiseConv1d: non-finite accumulated gradients");
    layer.grad_weights = std::move(staged.grad_weights);
    layer.grad_bias = std::move(staged.grad_bias);
    grad_input = std::move(result);
}

inline float sigmoid(float value) {
    if (value >= 0.f) return 1.f/(1.f+std::exp(-value));
    const float exponential = std::exp(value);
    return exponential/(1.f+exponential);
}

inline void swigluForward(const std::vector<float>& gate, const std::vector<float>& value,
                         std::vector<float>& output) {
    if (gate.empty() || gate.size() != value.size())
        throw std::invalid_argument("SwiGLU: expected two nonempty tensors with equal shapes");
    requireFinite(gate); requireFinite(value);
    std::vector<float> result(gate.size());
    RuntimeParallel::forEach(result.size(), RuntimeParallel::workers(result.size()), [&](size_t index) {
        const float silu = gate[index]*sigmoid(gate[index]);
        result[index] = silu*value[index];
    });
    requireFinite(result);
    output = std::move(result);
}

inline void swigluBackward(const std::vector<float>& gate, const std::vector<float>& value,
                          const std::vector<float>& grad_output,
                          std::vector<float>& grad_gate, std::vector<float>& grad_value) {
    if (gate.empty() || gate.size() != value.size() || gate.size() != grad_output.size())
        throw std::invalid_argument("SwiGLU: input/output gradient shape mismatch");
    requireFinite(gate); requireFinite(value); requireFinite(grad_output);
    std::vector<float> dg(gate.size()), du(gate.size());
    RuntimeParallel::forEach(gate.size(), RuntimeParallel::workers(gate.size()), [&](size_t index) {
        const float sig = sigmoid(gate[index]);
        const float silu = gate[index]*sig;
        const float derivative = sig+silu*(1.f-sig);
        dg[index] = (grad_output[index]*value[index])*derivative;
        du[index] = grad_output[index]*silu;
    });
    requireFinite(dg); requireFinite(du);
    grad_gate = std::move(dg);
    grad_value = std::move(du);
}
} // namespace CpuCausalSequence
