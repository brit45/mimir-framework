#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "Parallel.hpp"

namespace CpuNormalization {
namespace detail {

inline void validateEpsilon(float eps) {
    if (!std::isfinite(eps) || eps < 0.0f)
        throw std::invalid_argument("CPU normalization: epsilon must be finite and non-negative");
}

inline size_t normalizedGroups(size_t count, int normalized_size, float eps) {
    validateEpsilon(eps);
    if (normalized_size <= 0 || count % static_cast<size_t>(normalized_size) != 0)
        throw std::invalid_argument("CPU normalization: invalid normalized_size");
    return count / static_cast<size_t>(normalized_size);
}

} // namespace detail

// Input and output may alias. Each group is completely reduced before its
// values are written, and groups occupy disjoint spans.
inline void layerNormForward(const float* input, float* output, size_t count,
                             int normalized_size, const float* gamma,
                             const float* beta, float eps) {
    const size_t groups = detail::normalizedGroups(count, normalized_size, eps);
    if (count != 0 && (!input || !output))
        throw std::invalid_argument("CPU LayerNorm: null tensor storage");
    const size_t width = static_cast<size_t>(normalized_size);
    RuntimeParallel::forEach(groups, groups > 1 ? RuntimeParallel::workers(count) : 1,
        [&](size_t group) {
            const size_t base = group * width;
            float mean = 0.0f;
            #pragma omp simd reduction(+:mean)
            for (size_t index = 0; index < width; ++index)
                mean += input[base + index];
            mean /= static_cast<float>(width);

            float variance = 0.0f;
            #pragma omp simd reduction(+:variance)
            for (size_t index = 0; index < width; ++index) {
                const float difference = input[base + index] - mean;
                variance += difference * difference;
            }
            variance /= static_cast<float>(width);
            const float inverse_stddev = 1.0f / std::sqrt(variance + eps);

            #pragma omp simd
            for (size_t index = 0; index < width; ++index) {
                float value = (input[base + index] - mean) * inverse_stddev;
                if (gamma) value *= gamma[index];
                if (beta) value += beta[index];
                output[base + index] = value;
            }
        });
}

inline void layerNormInPlace(std::vector<float>& data,
                             const std::vector<float>& gamma,
                             const std::vector<float>& beta,
                             int normalized_size, float eps = 1e-5f) {
    detail::normalizedGroups(data.size(), normalized_size, eps);
    if (gamma.size() < static_cast<size_t>(normalized_size) ||
        beta.size() < static_cast<size_t>(normalized_size))
        throw std::invalid_argument("CPU LayerNorm: affine parameter size mismatch");
    layerNormForward(data.data(), data.data(), data.size(), normalized_size,
                     gamma.data(), beta.data(), eps);
}

inline void batchNormInPlace(std::vector<float>& data,
                             const std::vector<float>& gamma,
                             const std::vector<float>& beta,
                             const std::vector<float>& running_mean,
                             const std::vector<float>& running_var,
                             int batch_size, int channels, int spatial_size,
                             float eps = 1e-5f, bool training = false) {
    detail::validateEpsilon(eps);
    if (batch_size <= 0 || channels <= 0 || spatial_size <= 0)
        throw std::invalid_argument("CPU BatchNorm: dimensions must be positive");
    const size_t batch = static_cast<size_t>(batch_size);
    const size_t channel_count = static_cast<size_t>(channels);
    const size_t spatial = static_cast<size_t>(spatial_size);
    if (channel_count > std::numeric_limits<size_t>::max() / spatial ||
        batch > std::numeric_limits<size_t>::max() / (channel_count * spatial) ||
        data.size() != batch * channel_count * spatial)
        throw std::invalid_argument("CPU BatchNorm: tensor size does not match dimensions");
    if (gamma.size() < channel_count || beta.size() < channel_count ||
        running_mean.size() < channel_count || running_var.size() < channel_count)
        throw std::invalid_argument("CPU BatchNorm: channel parameter size mismatch");

    const size_t channel_elements = batch * spatial;
    RuntimeParallel::forEach(channel_count, RuntimeParallel::workers(data.size()),
        [&](size_t channel) {
            float mean = running_mean[channel];
            float variance = running_var[channel];
            if (training) {
                mean = 0.0f;
                #pragma omp simd reduction(+:mean)
                for (size_t index = 0; index < channel_elements; ++index) {
                    const size_t offset = (index / spatial) * channel_count * spatial +
                        channel * spatial + index % spatial;
                    mean += data[offset];
                }
                mean /= static_cast<float>(channel_elements);
                variance = 0.0f;
                #pragma omp simd reduction(+:variance)
                for (size_t index = 0; index < channel_elements; ++index) {
                    const size_t offset = (index / spatial) * channel_count * spatial +
                        channel * spatial + index % spatial;
                    const float difference = data[offset] - mean;
                    variance += difference * difference;
                }
                variance /= static_cast<float>(channel_elements);
            }

            const float inverse_stddev = 1.0f / std::sqrt(variance + eps);
            const float scale = gamma[channel];
            const float bias = beta[channel];
            for (size_t sample = 0; sample < batch; ++sample) {
                const size_t base = (sample * channel_count + channel) * spatial;
                #pragma omp simd
                for (size_t index = 0; index < spatial; ++index)
                    data[base + index] = (data[base + index] - mean) * inverse_stddev * scale + bias;
            }
        });
}

inline void rmsNormInPlace(std::vector<float>& data,
                           const std::vector<float>& gamma,
                           int normalized_size, float eps = 1e-6f) {
    const size_t groups = detail::normalizedGroups(data.size(), normalized_size, eps);
    const size_t width = static_cast<size_t>(normalized_size);
    if (gamma.size() < width)
        throw std::invalid_argument("CPU RMSNorm: affine parameter size mismatch");
    RuntimeParallel::forEach(groups, groups > 1 ? RuntimeParallel::workers(data.size()) : 1,
        [&](size_t group) {
            const size_t base = group * width;
            float squared_sum = 0.0f;
            for (size_t index = 0; index < width; ++index) {
                const float value = data[base + index];
                squared_sum += value * value;
            }
            const float rms = std::sqrt(squared_sum / static_cast<float>(width) + eps);
            for (size_t index = 0; index < width; ++index)
                data[base + index] = data[base + index] / rms * gamma[index];
        });
}

} // namespace CpuNormalization
