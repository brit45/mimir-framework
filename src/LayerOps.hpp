#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <omp.h>
#include <immintrin.h>
#include "Layers.hpp"
#include "HardwareOpt.hpp"

// ============================================================================
// LAYER OPERATIONS - Forward pass pour tous les types de layers
// ============================================================================

namespace LayerOps {

// ============================================================================
// LINEAR (Dense / Fully Connected)
// ============================================================================

inline std::vector<float> linear_forward(
    const std::vector<float>& input,
    const Layer& layer,
    bool training = true
) {
    const int in_f = layer.in_features > 0 ? layer.in_features : input.size();
    const int out_f = layer.out_features;
    
    if (out_f <= 0) {
        throw std::runtime_error("Linear: out_features not set");
    }
    
    const float* weights = layer.getWeights();
    const float* bias = layer.use_bias ? (weights + in_f * out_f) : nullptr;
    
    std::vector<float> output(out_f, 0.0f);
    
    // Matrix-vector multiplication: y = Wx + b
    // W shape: [out_features, in_features]
    #pragma omp parallel for schedule(static) if(static_cast<long long>(out_f) * in_f > 262144)
    for (int o = 0; o < out_f; ++o) {
        float sum = 0.0f;
        const float* w_row = weights + o * in_f;
        
        // Vectorized dot product
        int i = 0;
        #ifdef __AVX2__
        __m256 acc = _mm256_setzero_ps();
        for (; i + 8 <= in_f; i += 8) {
            __m256 w = _mm256_loadu_ps(w_row + i);
            __m256 x = _mm256_loadu_ps(&input[i]);
            acc = _mm256_fmadd_ps(w, x, acc);
        }
        // Horizontal sum
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 sum128 = _mm_add_ps(hi, lo);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum = _mm_cvtss_f32(sum128);
        #endif
        
        // Remaining elements
        for (; i < in_f; ++i) {
            sum += w_row[i] * input[i];
        }
        
        // Add bias
        if (bias) sum += bias[o];
        
        output[o] = sum;
    }
    
    return output;
}

// ============================================================================
// FLATTEN
// ============================================================================

inline std::vector<float> flatten_forward(
    const std::vector<float>& input,
    const Layer& layer
) {
    // Flatten just returns input as-is (already 1D vector)
    return input;
}

// ============================================================================
// RESHAPE
// ============================================================================

inline std::vector<float> reshape_forward(
    const std::vector<float>& input,
    const Layer& layer
) {
    // Reshape preserves data, just changes interpretation
    // For now, just pass through (shape tracking is external)
    return input;
}

// ============================================================================
// ADD (Element-wise)
// ============================================================================

inline std::vector<float> add_forward(
    const std::vector<float>& input1,
    const std::vector<float>& input2
) {
    if (input1.size() != input2.size()) {
        throw std::runtime_error("Add: input sizes must match");
    }
    
    std::vector<float> output(input1.size());
    
    size_t i = 0;
    #ifdef __AVX2__
    for (; i + 8 <= input1.size(); i += 8) {
        __m256 a = _mm256_loadu_ps(&input1[i]);
        __m256 b = _mm256_loadu_ps(&input2[i]);
        __m256 c = _mm256_add_ps(a, b);
        _mm256_storeu_ps(&output[i], c);
    }
    #endif
    
    for (; i < input1.size(); ++i) {
        output[i] = input1[i] + input2[i];
    }
    
    return output;
}

// ============================================================================
// MULTIPLY (Element-wise)
// ============================================================================

inline std::vector<float> multiply_forward(
    const std::vector<float>& input1,
    const std::vector<float>& input2
) {
    if (input1.size() != input2.size()) {
        throw std::runtime_error("Multiply: input sizes must match");
    }
    
    std::vector<float> output(input1.size());
    
    size_t i = 0;
    #ifdef __AVX2__
    for (; i + 8 <= input1.size(); i += 8) {
        __m256 a = _mm256_loadu_ps(&input1[i]);
        __m256 b = _mm256_loadu_ps(&input2[i]);
        __m256 c = _mm256_mul_ps(a, b);
        _mm256_storeu_ps(&output[i], c);
    }
    #endif
    
    for (; i < input1.size(); ++i) {
        output[i] = input1[i] * input2[i];
    }
    
    return output;
}

// ============================================================================
// CONCAT
// ============================================================================

inline std::vector<float> concat_forward(
    const std::vector<std::vector<float>>& inputs,
    int axis = 1  // Channel axis for 2D data
) {
    if (inputs.empty()) {
        throw std::runtime_error("Concat: no inputs provided");
    }
    
    // Simple concatenation along flattened dimension
    size_t total_size = 0;
    for (const auto& input : inputs) {
        total_size += input.size();
    }
    
    std::vector<float> output;
    output.reserve(total_size);
    
    for (const auto& input : inputs) {
        output.insert(output.end(), input.begin(), input.end());
    }
    
    return output;
}

// ============================================================================
// LAYER NORM
// ============================================================================

inline std::vector<float> layernorm_forward(
    const std::vector<float>& input,
    const Layer& layer,
    bool training = true
) {
    const int N = input.size();
    const float eps = layer.eps;
    
    // Compute mean
    float mean = 0.0f;
    #pragma omp simd reduction(+:mean)
    for (int i = 0; i < N; ++i) {
        mean += input[i];
    }
    mean /= N;
    
    // Compute variance
    float var = 0.0f;
    #pragma omp simd reduction(+:var)
    for (int i = 0; i < N; ++i) {
        float diff = input[i] - mean;
        var += diff * diff;
    }
    var /= N;
    
    // Normalize
    std::vector<float> output(N);
    float inv_std = 1.0f / std::sqrt(var + eps);
    
    const float* weights = layer.affine ? layer.getWeights() : nullptr;
    const float* bias = (layer.affine && layer.use_bias) ? 
                        (weights + N) : nullptr;

    #pragma omp simd
    for (int i = 0; i < N; ++i) {
        float normalized = (input[i] - mean) * inv_std;
        
        // Apply affine transform if enabled
        if (weights) {
            normalized = normalized * weights[i];
            if (bias) normalized += bias[i];
        }
        
        output[i] = normalized;
    }
    
    return output;
}

// ============================================================================
// SOFTMAX
// ============================================================================

inline std::vector<float> softmax_forward(
    const std::vector<float>& input,
    const Layer& layer
) {
    const int N = input.size();
    std::vector<float> output(N);
    
    // Find max for numerical stability
    float max_val = *std::max_element(input.begin(), input.end());
    
    // Compute exp(x - max)
    float sum = 0.0f;
    #pragma omp simd reduction(+:sum)
    for (int i = 0; i < N; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    
    // Normalize
    float inv_sum = 1.0f / sum;
    #pragma omp simd
    for (int i = 0; i < N; ++i) {
        output[i] *= inv_sum;
    }
    
    return output;
}

// ============================================================================
// EMBEDDING
// ============================================================================

inline std::vector<float> embedding_forward(
    const std::vector<int>& indices,
    const Layer& layer
) {
    const int vocab_size = layer.vocab_size;
    const int embed_dim = layer.embed_dim > 0 ? layer.embed_dim : layer.out_features;
    
    if (vocab_size <= 0 || embed_dim <= 0) {
        throw std::runtime_error("Embedding: vocab_size and embed_dim must be set");
    }
    
    const float* embedding_table = layer.getWeights();
    std::vector<float> output;
    output.reserve(indices.size() * embed_dim);
    
    for (int idx : indices) {
        if (idx < 0 || idx >= vocab_size) {
            if (idx == layer.padding_idx) {
                // Padding index: return zeros
                output.insert(output.end(), embed_dim, 0.0f);
            } else {
                throw std::runtime_error("Embedding: index out of bounds");
            }
        } else {
            // Copy embedding vector
            const float* embed = embedding_table + idx * embed_dim;
            output.insert(output.end(), embed, embed + embed_dim);
        }
    }
    
    return output;
}

// ============================================================================
// AVGPOOL2D
// ============================================================================

inline std::vector<float> avgpool2d_forward(
    const std::vector<float>& input,
    const Layer& layer
) {
    const int kernel_h = layer.get_kernel_h();
    const int kernel_w = layer.get_kernel_w();
    const int stride_h = layer.get_stride_h();
    const int stride_w = layer.get_stride_w();
    const int pad_h = layer.get_pad_h();
    const int pad_w = layer.get_pad_w();
    
    const int in_channels = layer.in_channels > 0 ? layer.in_channels : 64;
    const int height = layer.input_height > 0 ? layer.input_height : 32;
    const int width = layer.input_width > 0 ? layer.input_width : 32;
    
    const int out_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    const int out_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    
    const size_t output_size = in_channels * out_height * out_width;
    std::vector<float> output(output_size, 0.0f);
    
    const float kernel_area = static_cast<float>(kernel_h * kernel_w);
    
    for (int c = 0; c < in_channels; ++c) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                float sum = 0.0f;
                int count = 0;
                
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int ih = oh * stride_h + kh - pad_h;
                        int iw = ow * stride_w + kw - pad_w;
                        
                        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                            int in_idx = c * (height * width) + ih * width + iw;
                            sum += input[in_idx];
                            ++count;
                        }
                    }
                }
                
                int out_idx = c * (out_height * out_width) + oh * out_width + ow;
                output[out_idx] = count > 0 ? (sum / count) : 0.0f;
            }
        }
    }
    
    return output;
}

// ============================================================================
// GLOBAL AVERAGE POOL 2D
// ============================================================================

inline std::vector<float> global_avgpool2d_forward(
    const std::vector<float>& input,
    const Layer& layer
) {
    const int in_channels = layer.in_channels > 0 ? layer.in_channels : 64;
    const int height = layer.input_height > 0 ? layer.input_height : 32;
    const int width = layer.input_width > 0 ? layer.input_width : 32;
    
    std::vector<float> output(in_channels, 0.0f);
    const float area = static_cast<float>(height * width);
    
    for (int c = 0; c < in_channels; ++c) {
        float sum = 0.0f;
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int idx = c * (height * width) + h * width + w;
                sum += input[idx];
            }
        }
        output[c] = sum / area;
    }
    
    return output;
}

// ============================================================================
// GROUP NORM
// ============================================================================

inline std::vector<float> groupnorm_forward(
    const std::vector<float>& input,
    const Layer& layer,
    bool training = true
) {
    const int num_groups = layer.num_groups;
    const int channels = layer.in_channels > 0 ? layer.in_channels : 64;
    const int height = layer.input_height > 0 ? layer.input_height : 32;
    const int width = layer.input_width > 0 ? layer.input_width : 32;
    const float eps = layer.eps;
    
    if (channels % num_groups != 0) {
        throw std::runtime_error("GroupNorm: channels must be divisible by num_groups");
    }
    
    const int channels_per_group = channels / num_groups;
    const int group_size = channels_per_group * height * width;
    
    std::vector<float> output(input.size());
    const float* weights = layer.affine ? layer.getWeights() : nullptr;
    const float* bias = (layer.affine && layer.use_bias) ? 
                        (weights + channels) : nullptr;
    
    for (int g = 0; g < num_groups; ++g) {
        // Compute mean for this group
        float mean = 0.0f;
        #pragma omp simd reduction(+:mean)
        for (int i = 0; i < group_size; ++i) {
            int idx = g * group_size + i;
            mean += input[idx];
        }
        mean /= group_size;
        
        // Compute variance
        float var = 0.0f;
        #pragma omp simd reduction(+:var)
        for (int i = 0; i < group_size; ++i) {
            int idx = g * group_size + i;
            float diff = input[idx] - mean;
            var += diff * diff;
        }
        var /= group_size;
        
        // Normalize
        float inv_std = 1.0f / std::sqrt(var + eps);
        #pragma omp simd
        for (int i = 0; i < group_size; ++i) {
            int idx = g * group_size + i;
            float normalized = (input[idx] - mean) * inv_std;
            
            // Apply affine transform (per-channel)
            if (weights) {
                int c = (idx / (height * width)) % channels;
                normalized = normalized * weights[c];
                if (bias) normalized += bias[c];
            }
            
            output[idx] = normalized;
        }
    }
    
    return output;
}

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================

inline std::vector<float> gelu_forward(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    const float sqrt_2_over_pi = std::sqrt(2.0f / 3.14159265359f);

    #pragma omp simd
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        float cdf = 0.5f * (1.0f + std::tanh(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
        output[i] = x * cdf;
    }
    
    return output;
}

inline std::vector<float> silu_forward(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    #pragma omp simd
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        float sigmoid = 1.0f / (1.0f + std::exp(-x));
        output[i] = x * sigmoid;
    }
    
    return output;
}

inline std::vector<float> relu_forward(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    
    size_t i = 0;
    #ifdef __AVX2__
    __m256 zero = _mm256_setzero_ps();
    for (; i + 8 <= input.size(); i += 8) {
        __m256 vals = _mm256_loadu_ps(&input[i]);
        __m256 result = _mm256_max_ps(vals, zero);
        _mm256_storeu_ps(&output[i], result);
    }
    #endif
    
    for (; i < input.size(); ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
    
    return output;
}

inline std::vector<float> tanh_forward(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    #pragma omp simd
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::tanh(input[i]);
    }
    
    return output;
}

inline std::vector<float> sigmoid_forward(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    #pragma omp simd
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
    
    return output;
}

// ============================================================================
// DROPOUT
// ============================================================================

inline std::vector<float> dropout_forward(
    const std::vector<float>& input,
    const Layer& layer,
    bool training = true
) {
    if (!training) {
        return input;  // No dropout during inference
    }
    
    const float p = layer.dropout_p;
    const float scale = 1.0f / (1.0f - p);
    
    std::vector<float> output(input.size());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < input.size(); ++i) {
        if (dist(gen) > p) {
            output[i] = input[i] * scale;
        } else {
            output[i] = 0.0f;
        }
    }
    
    return output;
}

// ============================================================================
// IDENTITY (Pass-through)
// ============================================================================

inline std::vector<float> identity_forward(const std::vector<float>& input) {
    return input;
}

// ============================================================================
// TRANSPOSE (2D matrix transpose)
// ============================================================================

inline std::vector<float> transpose_forward(
    const std::vector<float>& input,
    int rows,
    int cols
) {
    if (input.size() != static_cast<size_t>(rows * cols)) {
        throw std::runtime_error("Transpose: input size mismatch");
    }
    
    std::vector<float> output(input.size());
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
    
    return output;
}

// ============================================================================
// MATMUL (Matrix Multiplication)
// ============================================================================

inline std::vector<float> matmul_forward(
    const std::vector<float>& A,
    const std::vector<float>& B,
    int M, int K, int N  // A: M×K, B: K×N, Output: M×N
) {
    if (A.size() != static_cast<size_t>(M * K) || 
        B.size() != static_cast<size_t>(K * N)) {
        throw std::runtime_error("MatMul: dimension mismatch");
    }
    
    std::vector<float> C(M * N, 0.0f);

    #ifdef __AVX2__
    // Kernel AVX2/FMA optimisé (gère correctement les tails)
    HardwareOpt::matmul_fma_saturated(C.data(), A.data(), B.data(),
                                      static_cast<size_t>(M),
                                      static_cast<size_t>(N),
                                      static_cast<size_t>(K));
    #else
    #pragma omp parallel for schedule(static) if (static_cast<long long>(M) * N * K > 262144)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
    #endif
    
    return C;
}

// ============================================================================
// SPLIT (Split tensor along axis)
// ============================================================================

inline std::vector<std::vector<float>> split_forward(
    const std::vector<float>& input,
    int num_splits,
    int axis = 0
) {
    if (num_splits <= 0) {
        throw std::runtime_error("Split: num_splits must be positive");
    }
    
    size_t split_size = input.size() / num_splits;
    if (input.size() % num_splits != 0) {
        throw std::runtime_error("Split: input size not divisible by num_splits");
    }
    
    std::vector<std::vector<float>> outputs(num_splits);
    
    for (int i = 0; i < num_splits; ++i) {
        outputs[i].resize(split_size);
        std::copy(
            input.begin() + i * split_size,
            input.begin() + (i + 1) * split_size,
            outputs[i].begin()
        );
    }
    
    return outputs;
}

// Surcharge avec tailles explicites
inline std::vector<std::vector<float>> split_forward(
    const std::vector<float>& input,
    const std::vector<int>& split_sizes,
    int axis = 0
) {
    if (split_sizes.empty()) {
        throw std::runtime_error("Split: split_sizes cannot be empty");
    }
    
    // Vérifier que la somme des tailles correspond à input.size()
    size_t total_size = 0;
    for (int sz : split_sizes) {
        if (sz <= 0) {
            throw std::runtime_error("Split: all split_sizes must be positive");
        }
        total_size += sz;
    }
    
    if (total_size != input.size()) {
        throw std::runtime_error("Split: sum of split_sizes (" + std::to_string(total_size) + 
                                 ") != input size (" + std::to_string(input.size()) + ")");
    }
    
    std::vector<std::vector<float>> outputs(split_sizes.size());
    size_t offset = 0;
    
    for (size_t i = 0; i < split_sizes.size(); ++i) {
        outputs[i].resize(split_sizes[i]);
        std::copy(
            input.begin() + offset,
            input.begin() + offset + split_sizes[i],
            outputs[i].begin()
        );
        offset += split_sizes[i];
    }
    
    return outputs;
}

// ============================================================================
// UPSAMPLE NEAREST
// ============================================================================

inline std::vector<float> upsample_nearest_forward(
    const std::vector<float>& input,
    int in_h, int in_w, int channels,
    int scale_h, int scale_w
) {
    int out_h = in_h * scale_h;
    int out_w = in_w * scale_w;
    
    std::vector<float> output(channels * out_h * out_w);
    
    for (int c = 0; c < channels; ++c) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                int ih = oh / scale_h;
                int iw = ow / scale_w;
                
                int in_idx = c * in_h * in_w + ih * in_w + iw;
                int out_idx = c * out_h * out_w + oh * out_w + ow;
                
                output[out_idx] = input[in_idx];
            }
        }
    }
    
    return output;
}

// ============================================================================
// UPSAMPLE BILINEAR
// ============================================================================

inline std::vector<float> upsample_bilinear_forward(
    const std::vector<float>& input,
    int in_h, int in_w, int channels,
    int out_h, int out_w
) {
    std::vector<float> output(channels * out_h * out_w);
    
    float scale_h = static_cast<float>(in_h) / out_h;
    float scale_w = static_cast<float>(in_w) / out_w;
    
    for (int c = 0; c < channels; ++c) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float ih_float = oh * scale_h;
                float iw_float = ow * scale_w;
                
                int ih0 = static_cast<int>(std::floor(ih_float));
                int iw0 = static_cast<int>(std::floor(iw_float));
                int ih1 = std::min(ih0 + 1, in_h - 1);
                int iw1 = std::min(iw0 + 1, in_w - 1);
                
                float dh = ih_float - ih0;
                float dw = iw_float - iw0;
                
                // Bilinear interpolation
                float v00 = input[c * in_h * in_w + ih0 * in_w + iw0];
                float v01 = input[c * in_h * in_w + ih0 * in_w + iw1];
                float v10 = input[c * in_h * in_w + ih1 * in_w + iw0];
                float v11 = input[c * in_h * in_w + ih1 * in_w + iw1];
                
                float val = v00 * (1 - dh) * (1 - dw) +
                           v01 * (1 - dh) * dw +
                           v10 * dh * (1 - dw) +
                           v11 * dh * dw;
                
                output[c * out_h * out_w + oh * out_w + ow] = val;
            }
        }
    }
    
    return output;
}

// ============================================================================
// PERMUTE
// ============================================================================

inline std::vector<float> permute_forward(
    const std::vector<float>& input,
    const std::vector<int>& dims,
    const std::vector<int>& shape
) {
    // Permute tensor dimensions
    // dims: new order of dimensions (e.g., [0,2,1] to swap last two dims)
    // shape: original shape
    
    if (dims.empty() || shape.empty()) {
        throw std::runtime_error("Permute: dims and shape cannot be empty");
    }
    
    if (dims.size() != shape.size()) {
        throw std::runtime_error("Permute: dims and shape must have same length");
    }
    
    // Compute output shape
    std::vector<int> out_shape(shape.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        out_shape[i] = shape[dims[i]];
    }
    
    // Compute strides for input
    std::vector<int> in_strides(shape.size());
    in_strides[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * shape[i + 1];
    }
    
    // Compute strides for output
    std::vector<int> out_strides(out_shape.size());
    out_strides[out_shape.size() - 1] = 1;
    for (int i = out_shape.size() - 2; i >= 0; --i) {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }
    
    std::vector<float> output(input.size());
    
    // Permute data
    for (size_t out_idx = 0; out_idx < output.size(); ++out_idx) {
        // Convert flat index to multi-dimensional indices
        std::vector<int> out_coords(out_shape.size());
        int temp = out_idx;
        for (int i = out_shape.size() - 1; i >= 0; --i) {
            out_coords[i] = temp % out_shape[i];
            temp /= out_shape[i];
        }
        
        // Map to input coordinates using dims
        std::vector<int> in_coords(shape.size());
        for (size_t i = 0; i < dims.size(); ++i) {
            in_coords[dims[i]] = out_coords[i];
        }
        
        // Convert back to flat index
        int in_idx = 0;
        for (size_t i = 0; i < in_coords.size(); ++i) {
            in_idx += in_coords[i] * in_strides[i];
        }
        
        output[out_idx] = input[in_idx];
    }
    
    return output;
}

// ============================================================================
// ATTENTION HELPERS
// ============================================================================

// Softmax sur dernière dimension
inline void softmax_inplace(std::vector<float>& x, int seq_len, int dim) {
    for (int i = 0; i < seq_len; ++i) {
        float* row = &x[i * dim];
        
        // Find max for numerical stability
        float max_val = row[0];
        for (int j = 1; j < dim; ++j) {
            max_val = std::max(max_val, row[j]);
        }
        
        // Exp and sum
        float sum = 0.0f;
        for (int j = 0; j < dim; ++j) {
            row[j] = std::exp(row[j] - max_val);
            sum += row[j];
        }
        
        // Normalize
        for (int j = 0; j < dim; ++j) {
            row[j] /= (sum + 1e-9f);
        }
    }
}

// Matmul helper (M x K) @ (K x N) = (M x N)
inline void matmul(
    const std::vector<float>& A, const std::vector<float>& B,
    std::vector<float>& C,
    int M, int K, int N
) {
    C.resize(M * N, 0.0f);

    #ifdef __AVX2__
    HardwareOpt::matmul_fma_saturated(C.data(), A.data(), B.data(),
                                      static_cast<size_t>(M),
                                      static_cast<size_t>(N),
                                      static_cast<size_t>(K));
    #else
    #pragma omp parallel for schedule(static) if(static_cast<long long>(M) * N * K > 262144)
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
    #endif
}

// Self-Attention (simplifié - single head)
inline std::vector<float> self_attention_forward(
    const std::vector<float>& input,
    const std::vector<float>& qkv_weight,  // Combined Q,K,V weights
    const std::vector<float>& out_weight,
    int seq_len,
    int embed_dim,
    int num_heads = 1,
    bool causal = false
) {
    if (embed_dim % num_heads != 0) {
        throw std::runtime_error("SelfAttention: embed_dim must be divisible by num_heads");
    }
    
    int head_dim = embed_dim / num_heads;
    int qkv_dim = embed_dim * 3;  // Q, K, V concatenated
    
    // Input shape: [seq_len, embed_dim]
    if (input.size() != static_cast<size_t>(seq_len * embed_dim)) {
        throw std::runtime_error("SelfAttention: input size mismatch");
    }
    
    // 1. Linear projection to Q, K, V: [seq_len, embed_dim] @ [embed_dim, 3*embed_dim]
    std::vector<float> qkv(seq_len * qkv_dim);
    matmul(input, qkv_weight, qkv, seq_len, embed_dim, qkv_dim);
    
    // 2. Split into Q, K, V
    std::vector<float> Q(seq_len * embed_dim);
    std::vector<float> K(seq_len * embed_dim);
    std::vector<float> V(seq_len * embed_dim);
    
    for (int m = 0; m < seq_len; ++m) {
        const int base = m * qkv_dim;
        const int out = m * embed_dim;
        for (int k = 0; k < embed_dim; ++k) {
            Q[out + k] = qkv[base + k];
            K[out + k] = qkv[base + embed_dim + k];
            V[out + k] = qkv[base + 2 * embed_dim + k];
        }
    }
    
    // 3. Compute attention scores: Q @ K^T / sqrt(head_dim)
    // Shape: [seq_len, seq_len]
    std::vector<float> scores(seq_len * seq_len, 0.0f);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            if (causal && j > i) {
                scores[i * seq_len + j] = -1e9f;  // Mask future tokens
                continue;
            }
            
            float sum = 0.0f;
            for (int k = 0; k < embed_dim; ++k) {
                sum += Q[i * embed_dim + k] * K[j * embed_dim + k];
            }
            scores[i * seq_len + j] = sum * scale;
        }
    }
    
    // 4. Apply softmax
    softmax_inplace(scores, seq_len, seq_len);
    
    // 5. Apply attention to values: scores @ V
    std::vector<float> attended(seq_len * embed_dim);
    matmul(scores, V, attended, seq_len, seq_len, embed_dim);
    
    // 6. Output projection
    std::vector<float> output(seq_len * embed_dim);
    matmul(attended, out_weight, output, seq_len, embed_dim, embed_dim);
    
    return output;
}

// Multi-Head Attention (uses multiple heads)
inline std::vector<float> multihead_attention_forward(
    const std::vector<float>& input,
    const std::vector<float>& qkv_weight,
    const std::vector<float>& out_weight,
    int seq_len,
    int embed_dim,
    int num_heads,
    bool causal = false
) {
    // Pour l'instant, on simplifie en utilisant self_attention
    // Une vraie implémentation diviserait en heads parallèles
    return self_attention_forward(input, qkv_weight, out_weight, 
                                   seq_len, embed_dim, num_heads, causal);
}

// Cross-Attention (Q from one source, K/V from another)
inline std::vector<float> cross_attention_forward(
    const std::vector<float>& query_input,
    const std::vector<float>& kv_input,
    const std::vector<float>& q_weight,
    const std::vector<float>& kv_weight,
    const std::vector<float>& out_weight,
    int query_len,
    int kv_len,
    int embed_dim,
    int num_heads = 1
) {
    // Similar to self-attention but K,V come from different source
    // Simplified implementation
    return query_input;  // TODO: implement properly
}

} // namespace LayerOps
