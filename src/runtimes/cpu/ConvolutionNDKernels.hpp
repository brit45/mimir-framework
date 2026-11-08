#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>
#include "../ConvolutionND.hpp"
#include "../FiniteCheck.hpp"
#include "Parallel.hpp"
#include "SimdDispatch.hpp"

namespace CpuConvolutionND {
using RuntimeConvolutionND::Geometry;

// A valid contiguous output row slice for one spatial kernel location. Computing
// this once per row/kernel avoids coordinate division and padding branches for
// every multiply. The last input dimension may have a non-unit stride.
struct Slice {
    size_t input_offset = 0;
    size_t output_offset = 0;
    size_t count = 0;
};

inline Slice rowSlice(const Geometry& g, size_t output_row, size_t kernel_index) {
    Slice slice;
    const int last = g.rank - 1;
    const size_t output_row_base = output_row * static_cast<size_t>(g.output[last]);
    size_t input_offset = 0;
    for (int axis = 0; axis < last; ++axis) {
        const size_t out_coordinate = output_row_base / g.output_strides[axis] % g.output[axis];
        const size_t kernel_coordinate = kernel_index / g.kernel_strides[axis] % g.kernel[axis];
        const int64_t input_coordinate = static_cast<int64_t>(out_coordinate) * g.stride[axis] -
            g.padding[axis] + static_cast<int64_t>(kernel_coordinate) * g.dilation[axis];
        if (input_coordinate < 0 || input_coordinate >= g.input[axis]) return slice;
        input_offset += static_cast<size_t>(input_coordinate) * g.input_strides[axis];
    }
    const int64_t kernel_coordinate = static_cast<int64_t>(kernel_index % g.kernel[last]);
    const int64_t base = kernel_coordinate * g.dilation[last] - g.padding[last];
    const int64_t first = base < 0 ? (-base + g.stride[last] - 1) / g.stride[last] : 0;
    if (base > g.input[last] - 1 || first >= g.output[last]) return slice;
    const int64_t end = std::min<int64_t>(g.output[last] - 1,
        (static_cast<int64_t>(g.input[last]) - 1 - base) / g.stride[last]);
    if (end < first) return slice;
    slice.input_offset = input_offset + static_cast<size_t>(base + first * g.stride[last]);
    slice.output_offset = static_cast<size_t>(first);
    slice.count = static_cast<size_t>(end - first + 1);
    return slice;
}

inline void validateWeights(const Layer& layer, const Geometry& g) {
    if (!layer.getWeights() || layer.getWeightsSize() < g.parameter_size)
        throw std::invalid_argument("ConvND: missing or undersized packed weights/bias");
}

inline void validateFinite(const float* values, size_t count, const char* name) {
    if (!Mimir::RuntimeChecks::allFinite(values, count))
        throw std::invalid_argument(std::string("ConvND: non-finite ") + name);
}

inline void validateFinite(const std::vector<float>& values, const char* name) {
    validateFinite(values.data(), values.size(), name);
}

inline void forward(const std::vector<float>& input, std::vector<float>& output,
                    const Layer& layer) {
    const Geometry g = RuntimeConvolutionND::checkedGeometry(layer, input.size());
    if (input.size() != g.input_size) throw std::invalid_argument("ConvND: empty or invalid input");
    validateWeights(layer, g);
    const float* weights = layer.getWeights();
    validateFinite(input, "input");
    validateFinite(weights, layer.getWeightsSize(), "weights");
    const float* bias = layer.use_bias ? weights + g.weight_size : nullptr;
    std::vector<float> staged_output(g.output_size, 0.0f);
    const int last = g.rank - 1;
    const size_t row_length = static_cast<size_t>(g.output[last]);
    const size_t rows = g.output_spatial / row_length;
    const size_t tasks = RuntimeConvolutionND::checkedProduct(
        RuntimeConvolutionND::checkedProduct(g.batch, g.out_channels), rows);
    const size_t reduction = RuntimeConvolutionND::checkedProduct(g.in_per_group, g.kernel_spatial);
    // Parallelism and SIMD cover independent outputs; channel/kernel reductions
    // keep a fixed order in every ISA, with no reduced precision intermediates.
    RuntimeParallel::forEach(tasks, RuntimeParallel::workers(
        RuntimeConvolutionND::checkedProduct(g.output_size, reduction)), [&](size_t task) {
        const size_t row = task % rows;
        const int oc = static_cast<int>(task / rows % g.out_channels);
        const size_t batch = task / rows / g.out_channels;
        const int first_channel = (oc / g.out_per_group) * g.in_per_group;
        float* destination = staged_output.data() + (batch * g.out_channels + oc) * g.output_spatial + row * row_length;
        for (int channel = 0; channel < g.in_per_group; ++channel) {
            const float* source = input.data() +
                (batch * g.in_channels + first_channel + channel) * g.input_spatial;
            const size_t weight_base = (static_cast<size_t>(oc) * g.in_per_group + channel) * g.kernel_spatial;
            for (size_t kernel = 0; kernel < g.kernel_spatial; ++kernel) {
                const Slice slice = rowSlice(g, row, kernel);
                if (!slice.count) continue;
                const float weight = weights[weight_base + kernel];
                if (g.stride[last] == 1) {
                    RuntimeSimd::axpy(destination + slice.output_offset, source + slice.input_offset,
                                      weight, slice.count, true);
                } else {
                    size_t source_index = slice.input_offset;
                    for (size_t i = 0; i < slice.count; ++i, source_index += g.stride[last]) {
                        const size_t target = slice.output_offset + i;
                        destination[target] = std::fma(source[source_index], weight, destination[target]);
                    }
                }
            }
        }
        if (bias) for (size_t column = 0; column < row_length; ++column) destination[column] += bias[oc];
    });
    validateFinite(staged_output, "forward result");
    output = std::move(staged_output);
}

inline void backward(const std::vector<float>& input, const std::vector<float>& grad_output,
                     std::vector<float>& grad_input, Layer& layer) {
    const Geometry g = RuntimeConvolutionND::checkedGeometry(layer, input.size());
    if (input.size() != g.input_size || grad_output.size() != g.output_size)
        throw std::invalid_argument("ConvND: invalid backward tensor size");
    validateWeights(layer, g);
    const float* weights = layer.getWeights();
    validateFinite(input, "input");
    validateFinite(grad_output, "output gradient");
    validateFinite(weights, layer.getWeightsSize(), "weights");
    validateFinite(layer.grad_weights, "existing parameter gradients");
    validateFinite(layer.grad_bias, "existing bias gradients");
    // Accumulate into copies of the existing gradients, rather than computing a
    // separate delta: this preserves each original FP32 FMA rounding operation.
    // All authoritative buffers remain untouched if a computation overflows.
    std::vector<float> staged_weights = layer.grad_weights;
    std::vector<float> staged_bias = layer.grad_bias;
    if (staged_weights.size() != layer.getWeightsSize())
        staged_weights.assign(layer.getWeightsSize(), 0.0f);
    if (layer.use_bias && staged_bias.size() != static_cast<size_t>(g.out_channels))
        staged_bias.assign(g.out_channels, 0.0f);
    std::vector<float> staged_input(g.input_size, 0.0f);
    const int last = g.rank - 1;
    const size_t row_length = static_cast<size_t>(g.output[last]);
    const size_t rows = g.output_spatial / row_length;
    const size_t work = RuntimeConvolutionND::checkedProduct(g.output_size,
        RuntimeConvolutionND::checkedProduct(g.in_per_group, g.kernel_spatial));

    // Each output channel owns its full parameter slice, so accumulated gradients
    // require neither atomics nor thread-local copies of the parameter tensor.
    RuntimeParallel::forEach(g.out_channels, RuntimeParallel::workers(work), [&](size_t oc) {
        const int first_channel = (static_cast<int>(oc) / g.out_per_group) * g.in_per_group;
        for (int channel = 0; channel < g.in_per_group; ++channel) {
            const size_t weight_base = (oc * g.in_per_group + channel) * g.kernel_spatial;
            for (size_t kernel = 0; kernel < g.kernel_spatial; ++kernel) {
                float accumulated = staged_weights[weight_base + kernel];
                for (int batch = 0; batch < g.batch; ++batch) {
                    const float* source = input.data() +
                        (static_cast<size_t>(batch) * g.in_channels + first_channel + channel) * g.input_spatial;
                    const float* gradient = grad_output.data() +
                        (static_cast<size_t>(batch) * g.out_channels + oc) * g.output_spatial;
                    for (size_t row = 0; row < rows; ++row) {
                        const Slice slice = rowSlice(g, row, kernel);
                        size_t source_index = slice.input_offset;
                        const size_t output_index = row * row_length + slice.output_offset;
                        for (size_t i = 0; i < slice.count; ++i, source_index += g.stride[last])
                            accumulated = std::fma(source[source_index], gradient[output_index + i], accumulated);
                    }
                }
                staged_weights[weight_base + kernel] = accumulated;
            }
        }
        if (layer.use_bias) {
            float gradient = 0.0f;
            for (int batch = 0; batch < g.batch; ++batch) {
                const float* source = grad_output.data() +
                    (static_cast<size_t>(batch) * g.out_channels + oc) * g.output_spatial;
                for (size_t i = 0; i < g.output_spatial; ++i) gradient += source[i];
            }
            staged_weights[g.weight_size + oc] += gradient;
            staged_bias[oc] += gradient;
        }
    });

    // Each batch/input channel owns an entire input plane. Its scatter is local,
    // preserving a deterministic reduction order without overlapping writes.
    const size_t input_tasks = RuntimeConvolutionND::checkedProduct(g.batch, g.in_channels);
    RuntimeParallel::forEach(input_tasks, RuntimeParallel::workers(work), [&](size_t task) {
        const size_t batch = task / g.in_channels;
        const int ic = static_cast<int>(task % g.in_channels);
        const int group = ic / g.in_per_group;
        const int local_channel = ic % g.in_per_group;
        float* destination = staged_input.data() + task * g.input_spatial;
        for (int oc = group * g.out_per_group; oc < (group + 1) * g.out_per_group; ++oc) {
            const float* gradient = grad_output.data() +
                (batch * g.out_channels + oc) * g.output_spatial;
            const size_t weight_base = (static_cast<size_t>(oc) * g.in_per_group + local_channel) * g.kernel_spatial;
            for (size_t row = 0; row < rows; ++row) {
                for (size_t kernel = 0; kernel < g.kernel_spatial; ++kernel) {
                    const Slice slice = rowSlice(g, row, kernel);
                    if (!slice.count) continue;
                    const float weight = weights[weight_base + kernel];
                    const float* source = gradient + row * row_length + slice.output_offset;
                    if (g.stride[last] == 1) {
                        RuntimeSimd::axpy(destination + slice.input_offset, source, weight, slice.count, true);
                    } else {
                        size_t target = slice.input_offset;
                        for (size_t i = 0; i < slice.count; ++i, target += g.stride[last])
                            destination[target] = std::fma(source[i], weight, destination[target]);
                    }
                }
            }
        }
    });
    validateFinite(staged_input, "input gradient result");
    validateFinite(staged_weights, "parameter gradient result");
    validateFinite(staged_bias, "bias gradient result");
    // std::vector moves with the default allocator are noexcept: after all
    // checks succeed, commit the complete backward result together.
    layer.grad_weights = std::move(staged_weights);
    layer.grad_bias = std::move(staged_bias);
    grad_input = std::move(staged_input);
}
} // namespace CpuConvolutionND
