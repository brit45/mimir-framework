#pragma once

#include "../Layers.hpp"
#include <cstddef>
#include <limits>
#include <stdexcept>

// Checked metadata only. Tensor arithmetic belongs to runtime kernels.
// The sequence is token-major [steps,channels], with independent causal
// channel kernels packed as [channels,kernel], followed by optional bias.
namespace RuntimeCausalSequence {
struct Geometry {
    int steps = 0;
    int channels = 0;
    int kernel = 0;
    int dilation = 0;
    size_t value_count = 0;
    size_t weight_size = 0;
    size_t parameter_size = 0;
    bool bias = false;
};

inline size_t checkedProduct(size_t first, size_t second) {
    if (first && second > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max())/first)
        throw std::invalid_argument("CausalDepthwiseConv1d: dimensions overflow storage");
    return first*second;
}

inline Geometry checkedGeometry(const Layer& layer, size_t input_size = 0) {
    if (layer.type_enum != LayerType::CausalDepthwiseConv1d)
        throw std::invalid_argument("CausalDepthwiseConv1d: unexpected layer type");
    Geometry geometry;
    geometry.channels = layer.in_features;
    geometry.kernel = layer.kernel_size;
    geometry.dilation = layer.dilation;
    geometry.bias = layer.use_bias;
    if (geometry.channels <= 0 || geometry.kernel <= 0 || geometry.dilation <= 0 ||
        layer.out_features < 0 || (layer.out_features && layer.out_features != geometry.channels))
        throw std::invalid_argument("CausalDepthwiseConv1d: invalid channels/kernel/dilation");
    if (input_size) {
        if (input_size%static_cast<size_t>(geometry.channels))
            throw std::invalid_argument("CausalDepthwiseConv1d: input is not token-major [T,C]");
        const size_t steps = input_size/static_cast<size_t>(geometry.channels);
        if (!steps || steps > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("CausalDepthwiseConv1d: sequence length exceeds index range");
        geometry.steps = static_cast<int>(steps);
    } else {
        if (layer.seq_len <= 0)
            throw std::invalid_argument("CausalDepthwiseConv1d: seq_len required without input storage");
        geometry.steps = layer.seq_len;
    }
    geometry.value_count = checkedProduct(static_cast<size_t>(geometry.steps), static_cast<size_t>(geometry.channels));
    geometry.weight_size = checkedProduct(static_cast<size_t>(geometry.channels), static_cast<size_t>(geometry.kernel));
    const size_t bias_size = geometry.bias ? static_cast<size_t>(geometry.channels) : 0;
    if (geometry.weight_size > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max())-bias_size)
        throw std::invalid_argument("CausalDepthwiseConv1d: packed parameter size overflows");
    geometry.parameter_size = geometry.weight_size+bias_size;
    (void)checkedProduct(geometry.value_count, sizeof(float));
    (void)checkedProduct(geometry.parameter_size, sizeof(float));
    return geometry;
}
} // namespace RuntimeCausalSequence
