#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "Parallel.hpp"

namespace CpuPooling {

enum class Reduction { Max, Average };

// CHW pooling shared by CPU layer dispatch and the runtime operation facade.
inline void forward2d(const std::vector<float>& input, std::vector<float>& output,
                      int height, int width, int channels, int kernel_h, int kernel_w,
                      int stride_h, int stride_w, int pad_h, int pad_w, Reduction reduction) {
    if (height <= 0 || width <= 0 || channels <= 0 || kernel_h <= 0 || kernel_w <= 0 ||
        stride_h <= 0 || stride_w <= 0 || pad_h < 0 || pad_w < 0) {
        throw std::invalid_argument("CPU Pool2d: invalid dimensions");
    }
    const size_t input_plane = static_cast<size_t>(height) * width;
    if (input.size() / static_cast<size_t>(channels) < input_plane) {
        throw std::invalid_argument("CPU Pool2d: input size mismatch");
    }
    const int64_t rows = static_cast<int64_t>(height) + 2LL * pad_h - kernel_h;
    const int64_t columns = static_cast<int64_t>(width) + 2LL * pad_w - kernel_w;
    if (rows < 0 || columns < 0) {
        throw std::invalid_argument("CPU Pool2d: kernel exceeds padded input");
    }
    const size_t out_h = static_cast<size_t>(rows / stride_h + 1);
    const size_t out_w = static_cast<size_t>(columns / stride_w + 1);
    if (out_h > output.max_size() / out_w / static_cast<size_t>(channels)) {
        throw std::length_error("CPU Pool2d: output is too large");
    }
    const size_t output_plane = out_h * out_w;
    output.resize(static_cast<size_t>(channels) * output_plane);
    RuntimeParallel::forEach(output.size(), RuntimeParallel::workers(output.size()), [&](size_t index) {
        const size_t channel = index / output_plane;
        const size_t position = index % output_plane;
        const int64_t first_h = static_cast<int64_t>(position / out_w) * stride_h - pad_h;
        const int64_t first_w = static_cast<int64_t>(position % out_w) * stride_w - pad_w;
        float value = reduction == Reduction::Max ? -std::numeric_limits<float>::infinity() : 0.0f;
        size_t count = 0;
        for (int kh = 0; kh < kernel_h; ++kh) {
            const int64_t ih = first_h + kh;
            if (ih < 0 || ih >= height) continue;
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int64_t iw = first_w + kw;
                if (iw < 0 || iw >= width) continue;
                const float sample = input[channel * input_plane + static_cast<size_t>(ih) * width +
                                           static_cast<size_t>(iw)];
                value = reduction == Reduction::Max ? std::max(value, sample) : value + sample;
                ++count;
            }
        }
        output[index] = reduction == Reduction::Average && count > 0
            ? value / static_cast<float>(count) : value;
    });
}

} // namespace CpuPooling
