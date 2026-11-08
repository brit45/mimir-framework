#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "../Layers.hpp"

// Shared checked metadata. Numerical convolution belongs to each runtime.
// Spatial layout is D,H,W for Conv3d and T,D,H,W for Conv4d; tensors use
// contiguous N,C,spatial order and kernels use O,I/groups,spatial order.
namespace RuntimeConvolutionND {
struct Geometry {
    int rank = 0;
    int batch = 1;
    int in_channels = 0;
    int out_channels = 0;
    int groups = 1;
    int in_per_group = 0;
    int out_per_group = 0;
    std::array<int, 4> input{{1, 1, 1, 1}};
    std::array<int, 4> output{{1, 1, 1, 1}};
    std::array<int, 4> kernel{{1, 1, 1, 1}};
    std::array<int, 4> stride{{1, 1, 1, 1}};
    std::array<int, 4> padding{{0, 0, 0, 0}};
    std::array<int, 4> dilation{{1, 1, 1, 1}};
    std::array<size_t, 4> input_strides{{1, 1, 1, 1}};
    std::array<size_t, 4> output_strides{{1, 1, 1, 1}};
    std::array<size_t, 4> kernel_strides{{1, 1, 1, 1}};
    size_t input_spatial = 0;
    size_t output_spatial = 0;
    size_t kernel_spatial = 0;
    size_t input_size = 0;
    size_t output_size = 0;
    size_t weight_size = 0;
    size_t parameter_size = 0;
};

inline size_t checkedProduct(size_t a, size_t b) {
    if (a && b > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / a)
        throw std::invalid_argument("ConvND: dimensions overflow addressable storage");
    return a * b;
}

inline size_t spatialProduct(const std::array<int, 4>& dimensions, int rank,
                             std::array<size_t, 4>& strides) {
    size_t volume = 1;
    for (int axis = rank - 1; axis >= 0; --axis) {
        if (dimensions[axis] <= 0)
            throw std::invalid_argument("ConvND: spatial dimensions must be positive");
        strides[axis] = volume;
        volume = checkedProduct(volume, static_cast<size_t>(dimensions[axis]));
    }
    return volume;
}

inline void assignAxes(std::array<int, 4>& destination, const std::vector<int>& values,
                       int rank, const char* field) {
    if (values.empty()) return;
    if (values.size() != 1 && values.size() != static_cast<size_t>(rank))
        throw std::invalid_argument(std::string("ConvND: ") + field + " must have one or rank entries");
    for (int axis = 0; axis < rank; ++axis)
        destination[axis] = values.size() == 1 ? values[0] : values[axis];
}

// input_size==0 validates configuration without an allocated tensor. At runtime,
// an unspecified batch is inferred exactly from the tensor; spatial extents are
// never guessed from a flattened length. Explicit N,C shapes are checked.
inline Geometry checkedGeometry(const Layer& layer, size_t input_size = 0) {
    Geometry g;
    if (layer.type_enum == LayerType::Conv3d) g.rank = 3;
    else if (layer.type_enum == LayerType::Conv4d) g.rank = 4;
    else throw std::invalid_argument("ConvND: expected Conv3d or Conv4d layer");
    bool explicit_batch = false;
    int shape_channels = 0;
    std::array<int, 4> shape_spatial{{0, 0, 0, 0}};
    if (!layer.shape.empty()) {
        for (int dimension : layer.shape)
            if (dimension <= 0) throw std::invalid_argument("ConvND: shape dimensions must be positive");
        size_t offset = 0;
        if (layer.shape.size() == static_cast<size_t>(g.rank + 2)) {
            g.batch = layer.shape[0];
            explicit_batch = true;
            shape_channels = layer.shape[1];
            offset = 2;
        } else if (layer.shape.size() == static_cast<size_t>(g.rank + 1)) {
            shape_channels = layer.shape[0];
            offset = 1;
        } else if (layer.shape.size() != static_cast<size_t>(g.rank)) {
            throw std::invalid_argument("ConvND: shape must be spatial, C/spatial, or N/C/spatial");
        }
        for (int axis = 0; axis < g.rank; ++axis) shape_spatial[axis] = layer.shape[offset + axis];
    }
    g.in_channels = layer.in_channels > 0 ? layer.in_channels : shape_channels;
    g.out_channels = layer.out_channels;
    g.groups = layer.groups;
    if (g.batch <= 0 || g.in_channels <= 0 || g.out_channels <= 0 || g.groups <= 0 ||
        g.in_channels % g.groups || g.out_channels % g.groups ||
        (shape_channels && shape_channels != g.in_channels))
        throw std::invalid_argument("ConvND: invalid batch, channels, or groups");
    g.in_per_group = g.in_channels / g.groups;
    g.out_per_group = g.out_channels / g.groups;
    const auto choose = [](int axis_value, int fallback, int default_value) {
        return axis_value != default_value ? axis_value : fallback;
    };
    const int kernel_default = layer.kernel_size != 0 ? layer.kernel_size : 3;
    const std::array<int, 4> spatial{{layer.input_time, layer.input_depth,
                                    layer.input_height, layer.input_width}};
    const std::array<int, 4> kernel{{layer.kernel_t, layer.kernel_d, layer.kernel_h, layer.kernel_w}};
    const std::array<int, 4> stride{{layer.stride_t, layer.stride_d, layer.stride_h, layer.stride_w}};
    const std::array<int, 4> padding{{layer.pad_t, layer.pad_d, layer.pad_h, layer.pad_w}};
    const std::array<int, 4> dilation{{layer.dilation_t, layer.dilation_d,
                                     layer.dilation_h, layer.dilation_w}};
    for (int axis = 0; axis < g.rank; ++axis) {
        const int source = 4 - g.rank + axis;
        g.input[axis] = shape_spatial[axis] ? shape_spatial[axis] : spatial[source];
        g.kernel[axis] = kernel[source] != 0 ? kernel[source] : kernel_default;
        g.stride[axis] = choose(stride[source], layer.stride, 1);
        g.padding[axis] = choose(padding[source], layer.padding, 0);
        g.dilation[axis] = choose(dilation[source], layer.dilation, 1);
    }
    if (!layer.input_spatial_shape.empty() && layer.input_spatial_shape.size() != static_cast<size_t>(g.rank))
        throw std::invalid_argument("ConvND: input_spatial_shape must have rank entries");
    assignAxes(g.input, layer.input_spatial_shape, g.rank, "input_spatial_shape");
    assignAxes(g.kernel, layer.kernel_shape, g.rank, "kernel_shape");
    assignAxes(g.stride, layer.stride_shape, g.rank, "stride_shape");
    assignAxes(g.padding, layer.padding_shape, g.rank, "padding_shape");
    assignAxes(g.dilation, layer.dilation_shape, g.rank, "dilation_shape");
    for (int axis = 0; axis < g.rank; ++axis) {
        if (g.input[axis] <= 0 || g.kernel[axis] <= 0 || g.stride[axis] <= 0 ||
            g.padding[axis] < 0 || g.dilation[axis] <= 0)
            throw std::invalid_argument("ConvND: invalid spatial/kernel/stride/padding/dilation");
        if (shape_spatial[axis] && shape_spatial[axis] != g.input[axis])
            throw std::invalid_argument("ConvND: shape and input_spatial_shape disagree");
        const int64_t extent = static_cast<int64_t>(g.dilation[axis]) * (g.kernel[axis] - 1) + 1;
        const int64_t numerator = static_cast<int64_t>(g.input[axis]) +
                                  2 * static_cast<int64_t>(g.padding[axis]) - extent;
        if (numerator < 0)
            throw std::invalid_argument("ConvND: kernel exceeds padded input");
        const int64_t output = numerator / g.stride[axis] + 1;
        if (output <= 0 || output > std::numeric_limits<int>::max())
            throw std::invalid_argument("ConvND: invalid output dimension");
        g.output[axis] = static_cast<int>(output);
    }
    g.input_spatial = spatialProduct(g.input, g.rank, g.input_strides);
    g.output_spatial = spatialProduct(g.output, g.rank, g.output_strides);
    g.kernel_spatial = spatialProduct(g.kernel, g.rank, g.kernel_strides);
    const size_t sample_size = checkedProduct(g.in_channels, g.input_spatial);
    if (input_size) {
        if (input_size % sample_size)
            throw std::invalid_argument("ConvND: input size does not match spatial/channel dimensions");
        const size_t batch = input_size / sample_size;
        if (!batch || batch > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            (explicit_batch && batch != static_cast<size_t>(g.batch)))
            throw std::invalid_argument("ConvND: input batch differs from configured shape");
        g.batch = static_cast<int>(batch);
    }
    g.input_size = checkedProduct(g.batch, sample_size);
    g.output_size = checkedProduct(checkedProduct(g.batch, g.out_channels), g.output_spatial);
    g.weight_size = checkedProduct(checkedProduct(g.out_channels, g.in_per_group), g.kernel_spatial);
    const size_t bias_size = layer.use_bias ? static_cast<size_t>(g.out_channels) : 0;
    if (bias_size > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) - g.weight_size)
        throw std::invalid_argument("ConvND: parameter dimensions overflow");
    g.parameter_size = g.weight_size + bias_size;
    // Validate byte extents as well as element indices before allocating buffers.
    (void)checkedProduct(g.input_size, sizeof(float));
    (void)checkedProduct(g.output_size, sizeof(float));
    (void)checkedProduct(g.parameter_size, sizeof(float));
    return g;
}
} // namespace RuntimeConvolutionND
