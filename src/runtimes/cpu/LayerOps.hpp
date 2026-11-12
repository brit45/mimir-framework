#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include "Parallel.hpp"
#include <cstring>
#include <random>
#include <stdexcept>
#include "Layers.hpp"
#include "HardwareOpt.hpp"
#include "../LayerOps.hpp"
#include "RngContext.hpp"
#include "NormalizationKernels.hpp"
#include "PoolingKernels.hpp"
#include "ReusableScratch.hpp"

// ============================================================================
// LAYER OPERATIONS - Forward pass pour tous les types de layers
// ============================================================================

namespace LayerOps {


inline std::vector<int> infer_shape_for_axis_op(const std::vector<float>& input, const Layer& layer) {
    if (!layer.shape.empty()) {
        size_t total = 1;
        for (int d : layer.shape) {
            if (d <= 0) {
                total = 0;
                break;
            }
            total *= static_cast<size_t>(d);
        }
        if (total == input.size()) {
            return layer.shape;
        }
    }

    if (layer.in_features > 0 && (input.size() % static_cast<size_t>(layer.in_features)) == 0) {
        return {
            static_cast<int>(input.size() / static_cast<size_t>(layer.in_features)),
            layer.in_features,
        };
    }

    if (layer.embed_dim > 0 && (input.size() % static_cast<size_t>(layer.embed_dim)) == 0) {
        return {
            static_cast<int>(input.size() / static_cast<size_t>(layer.embed_dim)),
            layer.embed_dim,
        };
    }

    if (layer.seq_len > 0 && (input.size() % static_cast<size_t>(layer.seq_len)) == 0) {
        return {
            layer.seq_len,
            static_cast<int>(input.size() / static_cast<size_t>(layer.seq_len)),
        };
    }

    return {static_cast<int>(input.size())};
}

inline std::vector<float> batchnorm_forward(
    const std::vector<float>& input,
    const Layer& layer,
    bool training = true
) {
    if (input.empty()) {
        return {};
    }

    int channels = layer.in_channels > 0 ? layer.in_channels : layer.out_channels;
    if (channels <= 0) {
        if (!layer.running_mean.empty()) {
            channels = static_cast<int>(layer.running_mean.size());
        } else if (layer.affine && layer.getWeights()) {
            const size_t w_sz = layer.getWeightsSize();
            if (layer.use_bias && (w_sz % 2ULL) == 0ULL) {
                channels = static_cast<int>(w_sz / 2ULL);
            } else {
                channels = static_cast<int>(w_sz);
            }
        }
    }

    if (channels <= 0) {
        throw std::runtime_error("BatchNorm: invalid channel count");
    }

    int spatial_size = 1;
    if (layer.type_enum == LayerType::BatchNorm2d && layer.input_height > 0 && layer.input_width > 0) {
        spatial_size = layer.input_height * layer.input_width;
    }

    size_t sample_stride = static_cast<size_t>(channels) * static_cast<size_t>(spatial_size);
    if (sample_stride == 0 || (input.size() % sample_stride) != 0) {
        spatial_size = 1;
        sample_stride = static_cast<size_t>(channels);
    }
    if (sample_stride == 0 || (input.size() % sample_stride) != 0) {
        throw std::runtime_error("BatchNorm: input size incompatible with channel/spatial dimensions");
    }

    const int batch_size = static_cast<int>(input.size() / sample_stride);
    if (batch_size <= 0) {
        throw std::runtime_error("BatchNorm: invalid batch size");
    }

    std::vector<float> gamma(static_cast<size_t>(channels), 1.0f);
    std::vector<float> beta(static_cast<size_t>(channels), 0.0f);
    if (layer.affine) {
        const float* weights = layer.getWeights();
        const size_t w_sz = layer.getWeightsSize();
        if (weights) {
            const size_t gamma_n = std::min(static_cast<size_t>(channels), w_sz);
            std::copy(weights, weights + gamma_n, gamma.begin());
            if (layer.use_bias && w_sz >= static_cast<size_t>(channels) * 2ULL) {
                std::copy(weights + channels, weights + channels + channels, beta.begin());
            }
        }
    }

    std::vector<float> running_mean(static_cast<size_t>(channels), 0.0f);
    std::vector<float> running_var(static_cast<size_t>(channels), 1.0f);
    const bool has_running_stats =
        layer.track_running_stats &&
        layer.running_mean.size() == static_cast<size_t>(channels) &&
        layer.running_var.size() == static_cast<size_t>(channels);
    if (has_running_stats) {
        running_mean = layer.running_mean;
        running_var = layer.running_var;
    }

    std::vector<float> output = input;
    CpuNormalization::batchNormInPlace(
        output,
        gamma,
        beta,
        running_mean,
        running_var,
        batch_size,
        channels,
        spatial_size,
        layer.eps,
        training || !has_running_stats
    );
    return output;
}

// ============================================================================
// LINEAR (Dense / Fully Connected)
// ============================================================================

inline std::vector<float> linear_forward(
    const std::vector<float>& input,const Layer& layer,bool training=true
) {
    (void)training;
    const int in_f=layer.in_features>0?layer.in_features:static_cast<int>(input.size());
    const int out_f=layer.out_features;
    if(in_f<=0 || out_f<=0)throw std::runtime_error("Linear: invalid feature dimensions");
    const size_t in=static_cast<size_t>(in_f),out=static_cast<size_t>(out_f);
    const float* weights=layer.getWeights();
    if(!weights || layer.getWeightsSize()/in<out ||
       (layer.use_bias && layer.getWeightsSize()-in*out<out))
        throw std::runtime_error("Linear: weights not initialized or too small");
    if(input.size()<in)throw std::runtime_error("Linear: input size smaller than in_features");
    const bool infer_batch=layer.seq_len<=0 && input.size()>in && input.size()%in==0;
    size_t rows=1;
    if((layer.seq_len>0 && input.size()==static_cast<size_t>(layer.seq_len)*in) || infer_batch)
        rows=input.size()/in;
    else if(input.size()!=in)throw std::runtime_error("Linear: input size mismatch");
    const float* bias=layer.use_bias?weights+in*out:nullptr;
    std::vector<float> output(rows*out);
    const int workers=HardwareOpt::matmul_workers(rows*((out+23)/24),24,in);
    if(RuntimeLayerOps::hardwareAccelerationEnabled()) {
        // Parameters are [out,in], not [in,out]. Output lanes are independent,
        // so every ISA retains the exact scalar k-order and FMA rounding.
        RuntimeSimd::matmulTranspose(output.data(),input.data(),weights,rows,out,in,true,workers);
    } else {
        RuntimeParallel::forEach(rows,workers,[&](size_t row) {
            for(size_t o=0;o<out;++o) {
                float sum=0;
                for(size_t i=0;i<in;++i)sum=std::fma(input[row*in+i],weights[o*in+i],sum);
                output[row*out+o]=sum;
            }
        });
    }
    if(bias)RuntimeParallel::forEach(rows,workers,[&](size_t row) {
        for(size_t o=0;o<out;++o)output[row*out+o]+=bias[o];
    });
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
    // Broadcast support (minimal et sûr):
    // - tailles égales: élément-wise
    // - un des deux est scalaire (size=1): broadcast
    // - l'un divise exactement l'autre: répétition (ex: (d_model) + (seq_len*d_model))
    if (input1.empty() || input2.empty()) {
        throw std::runtime_error("Add: inputs must be non-empty");
    }

    if (input1.size() == input2.size()) {
        std::vector<float> output(input1.size());
        if(RuntimeLayerOps::hardwareAccelerationEnabled()) {
            const size_t count=input1.size();
            const int workers=RuntimeParallel::workers(count);
            RuntimeParallel::forEach(size_t(workers),workers,[&](size_t task) {
                const size_t begin=(count/size_t(workers))*task+std::min(task,count%size_t(workers));
                const size_t length=count/size_t(workers)+(task<count%size_t(workers)?1:0);
                RuntimeSimd::add(output.data()+begin,input1.data()+begin,input2.data()+begin,length);
            });
        } else {
            RuntimeParallel::forEach(input1.size(),RuntimeParallel::workers(input1.size()),[&](size_t i) {
                output[i]=input1[i]+input2[i];
            });
        }
        return output;
    }

    const std::vector<float>* big = &input1;
    const std::vector<float>* small = &input2;
    if (input2.size() > input1.size()) {
        big = &input2;
        small = &input1;
    }

    const size_t big_n = big->size();
    const size_t small_n = small->size();

    if (small_n == 1) {
        const float s = (*small)[0];
        std::vector<float> output(big_n);
        RuntimeParallel::forEach(big_n, RuntimeParallel::workers(big_n), [&](size_t i) {
            output[i] = (*big)[i] + s;
        });
        return output;
    }

    if ((big_n % small_n) != 0) {
        throw std::runtime_error("Add: input sizes must match or be broadcastable");
    }

    std::vector<float> output(big_n);
    RuntimeParallel::forEach(big_n, RuntimeParallel::workers(big_n), [&](size_t i) {
        output[i] = (*big)[i] + (*small)[i % small_n];
    });
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
    
    if(RuntimeLayerOps::hardwareAccelerationEnabled()) {
        const size_t count=input1.size();
        if(count) {
            const int workers=RuntimeParallel::workers(count);
            RuntimeParallel::forEach(size_t(workers),workers,[&](size_t task) {
                const size_t begin=(count/size_t(workers))*task+std::min(task,count%size_t(workers));
                const size_t length=count/size_t(workers)+(task<count%size_t(workers)?1:0);
                RuntimeSimd::mul(output.data()+begin,input1.data()+begin,input2.data()+begin,length);
            });
        }
    } else {
        RuntimeParallel::forEach(input1.size(),RuntimeParallel::workers(input1.size()),[&](size_t i) {
            output[i]=input1[i]*input2[i];
        });
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
    (void)training;
    const int N = static_cast<int>(input.size());
    const float eps = layer.eps;

    // Nouveau mode (séquence / groupes) si in_features est configuré:
    // - normalized_size = layer.in_features
    // - applique LN indépendamment sur chaque groupe
    const int normalized = (layer.in_features > 0) ? layer.in_features : N;
    if (normalized <= 0 || (N % normalized) != 0) {
        throw std::runtime_error("LayerNorm: invalid normalized_size");
    }
    std::vector<float> output(static_cast<size_t>(N));

    const size_t required_weights = static_cast<size_t>(normalized) *
        (layer.use_bias ? 2ULL : 1ULL);
    const float* weights = layer.affine ? layer.getWeights() : nullptr;
    if (layer.affine && (!weights || layer.getWeightsSize() < required_weights)) {
        throw std::runtime_error("LayerNorm: affine weights size mismatch");
    }
    const float* bias = (layer.affine && layer.use_bias) ? (weights + normalized) : nullptr;

    CpuNormalization::layerNormForward(input.data(), output.data(), input.size(),
                                      normalized, weights, bias, eps);

    return output;
}

// ============================================================================
// SOFTMAX
// ============================================================================

inline std::vector<float> softmax_forward(
    const std::vector<float>& input,
    const Layer& layer
) {
    if (input.empty()) {
        return {};
    }

    const std::vector<int> shape = infer_shape_for_axis_op(input, layer);
    int axis = layer.axis;
    if (axis < 0) axis += static_cast<int>(shape.size());
    if (axis < 0 || axis >= static_cast<int>(shape.size())) {
        axis = static_cast<int>(shape.size()) - 1;
    }

    size_t outer = 1;
    for (int i = 0; i < axis; ++i) outer *= static_cast<size_t>(shape[static_cast<size_t>(i)]);
    size_t axis_size = static_cast<size_t>(shape[static_cast<size_t>(axis)]);
    size_t inner = 1;
    for (size_t i = static_cast<size_t>(axis) + 1; i < shape.size(); ++i) inner *= static_cast<size_t>(shape[i]);

    std::vector<float> output(input.size(), 0.0f);
    const bool is_log = layer.type_enum == LayerType::LogSoftmax;
    const size_t rows = outer * inner;
    RuntimeParallel::forEach(rows, RuntimeParallel::workers(input.size()), [&](size_t row) {
        const size_t outer_idx = row / inner;
        const size_t inner_idx = row % inner;
        const size_t base = (outer_idx * axis_size * inner) + inner_idx;
        float max_val = input[base];
        for (size_t a = 1; a < axis_size; ++a) {
            const float v = input[base + a * inner];
            if (v > max_val) max_val = v;
        }

        float sum = 0.0f;
        for (size_t a = 0; a < axis_size; ++a) {
            const float e = std::exp(input[base + a * inner] - max_val);
            output[base + a * inner] = e;
            sum += e;
        }

        if (is_log) {
            const float log_denom = std::log(sum);
            for (size_t a = 0; a < axis_size; ++a) {
                output[base + a * inner] = (input[base + a * inner] - max_val) - log_denom;
            }
        } else {
            const float inv_sum = 1.0f / sum;
            for (size_t a = 0; a < axis_size; ++a) {
                output[base + a * inner] *= inv_sum;
            }
        }
    });

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
    
    std::vector<float> output;
    CpuPooling::forward2d(input, output, height, width, in_channels,
                         kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                         CpuPooling::Reduction::Average);
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
    
    RuntimeParallel::forEach(in_channels, RuntimeParallel::workers(input.size()), [&](int c) {
        float sum = 0.0f;
        const size_t base = static_cast<size_t>(c) * static_cast<size_t>(height) * width;
        const size_t spatial = static_cast<size_t>(height) * width;
        #pragma omp simd reduction(+:sum)
        for (size_t i = 0; i < spatial; ++i) {
            sum += input[base + i];
        }
        output[c] = sum / area;
    });
    
    return output;
}

inline std::vector<float> global_avgpool2d_backward(
    const std::vector<float>& grad_out,
    const Layer& layer,
    size_t input_size_hint = 0ULL
) {
    int in_channels = layer.in_channels > 0 ? layer.in_channels : static_cast<int>(grad_out.size());
    int height = layer.input_height > 0 ? layer.input_height : 0;
    int width = layer.input_width > 0 ? layer.input_width : 0;

    if (in_channels <= 0) {
        return {};
    }
    if (static_cast<int>(grad_out.size()) != in_channels) {
        // Shape mismatch (expected one scalar per channel).
        return {};
    }

    // If dims are missing or inconsistent, try to infer H/W from input_size_hint.
    if (input_size_hint > 0ULL) {
        const size_t c = static_cast<size_t>(in_channels);
        if (c > 0 && (input_size_hint % c) == 0ULL) {
            const size_t hw = input_size_hint / c;
            const size_t cfg_hw = (height > 0 && width > 0)
                ? static_cast<size_t>(height) * static_cast<size_t>(width)
                : 0ULL;
            if (cfg_hw != hw) {
                bool fixed = false;
                if (height > 0 && (hw % static_cast<size_t>(height)) == 0ULL) {
                    width = static_cast<int>(hw / static_cast<size_t>(height));
                    fixed = true;
                } else if (width > 0 && (hw % static_cast<size_t>(width)) == 0ULL) {
                    height = static_cast<int>(hw / static_cast<size_t>(width));
                    fixed = true;
                }
                if (!fixed) {
                    const size_t s = static_cast<size_t>(std::llround(std::sqrt(static_cast<double>(hw))));
                    if (s > 0 && s * s == hw) {
                        height = static_cast<int>(s);
                        width = static_cast<int>(s);
                    }
                }
            }
        }
    }

    if (height <= 0 || width <= 0) {
        return {};
    }

    const size_t spatial = static_cast<size_t>(height) * static_cast<size_t>(width);
    const size_t expected_in = static_cast<size_t>(in_channels) * spatial;
    if (input_size_hint > 0ULL && input_size_hint != expected_in) {
        // Still inconsistent after inference.
        return {};
    }

    std::vector<float> grad_in(expected_in, 0.0f);
    const float inv = 1.0f / static_cast<float>(spatial);

    RuntimeParallel::forEach(in_channels, RuntimeParallel::workers(expected_in), [&](int c) {
        const float g = grad_out[static_cast<size_t>(c)] * inv;
        const size_t base = static_cast<size_t>(c) * spatial;
        float* dst = grad_in.data() + base;
        for (size_t i = 0; i < spatial; ++i) {
            dst[i] = g;
        }
    });

    return grad_in;
}

// ============================================================================
// GROUP NORM
// ============================================================================

inline std::vector<float> groupnorm_forward(
    const std::vector<float>& input,
    const Layer& layer,
    bool training = true
) {
    (void)training;
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
    
    const int workers=HardwareOpt::matmul_workers(num_groups,group_size,4);
    #pragma omp parallel for if(workers>1) num_threads(workers) schedule(static)
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
        // Affine parameters are constant across a channel. Hoist channel
        // indexing instead of integer division/modulo for every spatial value.
        const int spatial=height*width;
        for(int c=0;c<channels_per_group;++c){
            const int channel=g*channels_per_group+c,base=channel*spatial;
            const float gamma=weights?weights[channel]:1.f;
            const float beta=bias?bias[channel]:0.f;
            #pragma omp simd
            for(int i=0;i<spatial;++i){
                float normalized=(input[base+i]-mean)*inv_std;
                if(weights)normalized*=gamma;
                if(bias)normalized+=beta;
                output[base+i]=normalized;
            }
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

    RuntimeParallel::forEach(input.size(), RuntimeParallel::workers(input.size()), [&](size_t i) {
        float x = input[i];
        float cdf = 0.5f * (1.0f + std::tanh(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
        output[i] = x * cdf;
    });
    
    return output;
}

inline std::vector<float> geglu_forward(const std::vector<float>& input, int seq_len, int hidden_dim) {
    if (seq_len <= 0) {
        throw std::runtime_error("GEGLU: seq_len must be > 0");
    }
    if (hidden_dim <= 0) {
        throw std::runtime_error("GEGLU: hidden_dim must be > 0");
    }

    const size_t expected = static_cast<size_t>(seq_len) * static_cast<size_t>(hidden_dim) * 2ULL;
    if (input.size() != expected) {
        throw std::runtime_error("GEGLU: input size mismatch");
    }

    std::vector<float> output(static_cast<size_t>(seq_len) * static_cast<size_t>(hidden_dim), 0.0f);
    const float sqrt_2_over_pi = std::sqrt(2.0f / 3.14159265359f);
    const size_t output_size = output.size();
    RuntimeParallel::forEach(output_size, RuntimeParallel::workers(output_size), [&](size_t i) {
        const size_t token_offset = (i / static_cast<size_t>(hidden_dim)) * static_cast<size_t>(hidden_dim) * 2ULL;
        const size_t feature = i % static_cast<size_t>(hidden_dim);
        const float value = input[token_offset + feature];
        const float gate = input[token_offset + static_cast<size_t>(hidden_dim) + feature];
        const float cdf = 0.5f * (1.0f + std::tanh(sqrt_2_over_pi * (gate + 0.044715f * gate * gate * gate)));
        output[i] = value * (gate * cdf);
    });
    return output;
}

inline std::vector<float> silu_forward(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    RuntimeParallel::forEach(input.size(), RuntimeParallel::workers(input.size()), [&](size_t i) {
        float x = input[i];
        float sigmoid = 1.0f / (1.0f + std::exp(-x));
        output[i] = x * sigmoid;
    });
    
    return output;
}

inline std::vector<float> relu_forward(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    RuntimeParallel::forEach(input.size(), RuntimeParallel::workers(input.size()), [&](size_t i) {
        output[i] = std::max(0.0f, input[i]);
    });
    
    return output;
}

inline std::vector<float> tanh_forward(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    RuntimeParallel::forEach(input.size(), RuntimeParallel::workers(input.size()), [&](size_t i) {
        output[i] = std::tanh(input[i]);
    });
    
    return output;
}

inline std::vector<float> sigmoid_forward(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    RuntimeParallel::forEach(input.size(), RuntimeParallel::workers(input.size()), [&](size_t i) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    });
    
    return output;
}

// ============================================================================
// DROPOUT
// ============================================================================

inline std::vector<float> dropout_forward(
    const std::vector<float>& input,
    const Layer& layer,
    bool training = true,
    std::vector<uint8_t>* mask = nullptr
) {
    if (mask) mask->assign(input.size(),1);
    if (!training) {
        return input;  // No dropout during inference
    }
    
    const float p = std::clamp(layer.dropout_p,0.0f,1.0f);
    const float scale = p < 1.0f ? 1.0f / (1.0f - p) : 0.0f;
    
    std::vector<float> output(input.size());
    auto& gen = MimirRng::generator();
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < input.size(); ++i) {
        if (dist(gen) > p) {
            output[i] = input[i] * scale;
        } else {
            output[i] = 0.0f;
            if (mask) (*mask)[i]=0;
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
    
    RuntimeParallel::forEach(rows * cols, RuntimeParallel::workers(input.size()), [&](int idx) {
        const int i = idx / cols;
        const int j = idx % cols;
        output[j * rows + i] = input[idx];
    });
    
    return output;
}

// ============================================================================
// MATMUL (Matrix Multiplication)
// ============================================================================

template<class A, class B, class C>
inline void matmul_into(const A& a,const B& b, C& c,int m,int k,int n) {
    if(m<0 || k<0 || n<0 || a.size()!=size_t(m)*size_t(k) || b.size()!=size_t(k)*size_t(n))
        throw std::runtime_error("MatMul: dimension mismatch");
    c.resize(size_t(m)*size_t(n));
    if(RuntimeLayerOps::hardwareAccelerationEnabled()) {
        HardwareOpt::matmul_fma_saturated(c.data(),a.data(),b.data(),size_t(m),size_t(n),size_t(k));
    } else {
        RuntimeParallel::forEach(size_t(m),HardwareOpt::matmul_workers(size_t(m),size_t(n),size_t(k)),[&](size_t row) {
            for(size_t column=0;column<size_t(n);++column) {
                float sum=0;
                for(size_t i=0;i<size_t(k);++i)sum=std::fma(a[row*size_t(k)+i],b[i*size_t(n)+column],sum);
                c[row*size_t(n)+column]=sum;
            }
        });
    }
}
inline std::vector<float> matmul_forward(const std::vector<float>& a,const std::vector<float>& b,
                                          int m,int k,int n) {
    std::vector<float> c;
    matmul_into(a,b,c,m,k,n);
    return c;
}

// ============================================================================
// SPLIT (Split tensor along axis)
// ============================================================================

inline std::vector<std::vector<float>> split_forward(
    const std::vector<float>& input,
    int num_splits,
    int axis = 0
) {
    (void)axis;
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
    (void)axis;
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
    
    const size_t output_size = output.size();
    const size_t output_plane = static_cast<size_t>(out_h) * static_cast<size_t>(out_w);
    RuntimeParallel::forEach(output_size, RuntimeParallel::workers(output_size), [&](size_t out_idx) {
        const size_t spatial_idx = out_idx % output_plane;
        const int oh = static_cast<int>(spatial_idx / static_cast<size_t>(out_w));
        const int ow = static_cast<int>(spatial_idx % static_cast<size_t>(out_w));
        const size_t in_idx = (out_idx / output_plane) * static_cast<size_t>(in_h) * in_w
            + static_cast<size_t>(oh / scale_h) * in_w + static_cast<size_t>(ow / scale_w);
        output[out_idx] = input[in_idx];
    });
    
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
    
    const size_t output_size = output.size();
    const size_t output_plane = static_cast<size_t>(out_h) * static_cast<size_t>(out_w);
    const size_t input_plane = static_cast<size_t>(in_h) * static_cast<size_t>(in_w);
    RuntimeParallel::forEach(output_size, RuntimeParallel::workers(output_size), [&](size_t out_idx) {
                const size_t spatial_idx = out_idx % output_plane;
                const int oh = static_cast<int>(spatial_idx / static_cast<size_t>(out_w));
                const int ow = static_cast<int>(spatial_idx % static_cast<size_t>(out_w));
                const size_t input_base = (out_idx / output_plane) * input_plane;
                float ih_float = oh * scale_h;
                float iw_float = ow * scale_w;
                
                int ih0 = static_cast<int>(std::floor(ih_float));
                int iw0 = static_cast<int>(std::floor(iw_float));
                int ih1 = std::min(ih0 + 1, in_h - 1);
                int iw1 = std::min(iw0 + 1, in_w - 1);
                
                float dh = ih_float - ih0;
                float dw = iw_float - iw0;
                
                // Bilinear interpolation
                float v00 = input[input_base + static_cast<size_t>(ih0) * in_w + iw0];
                float v01 = input[input_base + static_cast<size_t>(ih0) * in_w + iw1];
                float v10 = input[input_base + static_cast<size_t>(ih1) * in_w + iw0];
                float v11 = input[input_base + static_cast<size_t>(ih1) * in_w + iw1];
                
                float val = v00 * (1 - dh) * (1 - dw) +
                           v01 * (1 - dh) * dw +
                           v10 * dh * (1 - dw) +
                           v11 * dh * dw;
                
                output[out_idx] = val;
    });
    
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
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * shape[i + 1];
    }
    
    // Compute strides for output
    std::vector<int> out_strides(out_shape.size());
    out_strides[out_shape.size() - 1] = 1;
    for (int i = static_cast<int>(out_shape.size()) - 2; i >= 0; --i) {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }
    (void)out_strides;
    
    std::vector<float> output(input.size());
    
    // Reuse per-thread coordinate buffers instead of allocating them per element.
    #pragma omp parallel if(RuntimeParallel::workers(output.size())>1) num_threads(RuntimeParallel::workers(output.size()))
    {
    std::vector<int> out_coords(out_shape.size());
    std::vector<int> in_coords(shape.size());
    #pragma omp for schedule(static)
    for (size_t out_idx = 0; out_idx < output.size(); ++out_idx) {
        // Convert flat index to multi-dimensional indices
        int temp = static_cast<int>(out_idx);
        for (int i = static_cast<int>(out_shape.size()) - 1; i >= 0; --i) {
            out_coords[i] = temp % out_shape[i];
            temp /= out_shape[i];
        }
        
        // Map to input coordinates using dims
        for (size_t i = 0; i < dims.size(); ++i) {
            in_coords[dims[i]] = out_coords[i];
        }
        
        // Convert back to flat index
        int in_idx = 0;
        for (size_t i = 0; i < in_coords.size(); ++i) {
            in_idx += in_coords[i] * in_strides[i];
        }
        
        output[out_idx] = input[static_cast<size_t>(in_idx)];
    }
    }
    
    return output;
}

// ============================================================================
// ATTENTION HELPERS
// ============================================================================

// Softmax sur dernière dimension
inline void softmax_inplace(std::vector<float>& x, int seq_len, int dim) {
    RuntimeParallel::forEach(seq_len, RuntimeParallel::workers(static_cast<size_t>(seq_len) * static_cast<size_t>(dim)), [&](int i) {
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
    });
}

// Matmul helper (M x K) @ (K x N) = (M x N)
template<class A, class B, class C>
inline void matmul(const A& a,const B& b,C& c,int m,int k,int n) {
    matmul_into(a,b,c,m,k,n);
}

// Self-Attention (simplifié - single head)
inline std::vector<float> self_attention_forward(
    const std::vector<float>& input,
    RuntimeScratch::ConstFloatView qkv_weight,  // Combined Q,K,V weights [3*embed_dim, embed_dim]
    RuntimeScratch::ConstFloatView out_weight,
    RuntimeScratch::ConstFloatView qkv_bias,
    RuntimeScratch::ConstFloatView out_bias,
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
    if (!qkv_bias.empty() && qkv_bias.size() != static_cast<size_t>(qkv_dim)) {
        throw std::runtime_error("SelfAttention: qkv_bias size mismatch");
    }
    if (!out_bias.empty() && out_bias.size() != static_cast<size_t>(embed_dim)) {
        throw std::runtime_error("SelfAttention: out_bias size mismatch");
    }

    // 1. Linear projection to Q, K, V.
    // Layout attendu pour qkv_weight: [3*embed_dim, embed_dim] (Q, K, V empilés verticalement).
    constexpr int query_tile = 128;
    const int tile_rows = std::min(query_tile, seq_len);
    RuntimeScratch::FloatArena workspace({size_t(seq_len)*qkv_dim,size_t(seq_len)*embed_dim,
        size_t(seq_len)*head_dim,size_t(seq_len)*head_dim,size_t(head_dim)*seq_len,
        size_t(tile_rows)*seq_len,size_t(tile_rows)*head_dim});
    auto qkv=workspace.take(size_t(seq_len)*qkv_dim);
    if(RuntimeLayerOps::hardwareAccelerationEnabled()) {
        RuntimeSimd::matmulTransposeBiased(qkv.data(),input.data(),qkv_weight.data(),
            qkv_bias.empty()?nullptr:qkv_bias.data(),size_t(seq_len),size_t(qkv_dim),size_t(embed_dim),
            RuntimeParallel::workers(size_t(seq_len)*qkv_dim*embed_dim));
    } else {
    RuntimeParallel::forEach(seq_len, RuntimeParallel::workers(static_cast<long long>(seq_len) * qkv_dim * embed_dim), [&](int m) {
        const float* xrow = &input[static_cast<size_t>(m) * static_cast<size_t>(embed_dim)];
        float* qkv_row = &qkv[static_cast<size_t>(m) * static_cast<size_t>(qkv_dim)];
        for (int n = 0; n < qkv_dim; ++n) {
            const float* w_row = &qkv_weight[static_cast<size_t>(n) * static_cast<size_t>(embed_dim)];
            float sum = qkv_bias.empty() ? 0.0f : qkv_bias[static_cast<size_t>(n)];
            for (int k = 0; k < embed_dim; ++k) {
                sum += xrow[k] * w_row[k];
            }
            qkv_row[n] = sum;
        }
    });
    }

    // Read head slices directly from the combined projection below.
    // 3-5. Multi-head attention (GEMM-based):
    // scores = Q_h @ K_h^T, softmax(scores), context = scores @ V_h
    auto attended=workspace.take(size_t(seq_len)*embed_dim);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto Qh=workspace.take(size_t(seq_len)*head_dim);
    auto Vh=workspace.take(size_t(seq_len)*head_dim);
    auto KhT=workspace.take(size_t(head_dim)*seq_len);
    auto scores=workspace.take(size_t(tile_rows)*seq_len);
    auto context=workspace.take(size_t(tile_rows)*head_dim);

    for (int h = 0; h < num_heads; ++h) {
        const int head_off = h * head_dim;

        // Pack Q_h and V_h to contiguous [seq_len x head_dim]
        for (int i = 0; i < seq_len; ++i) {
            const float* qsrc = &qkv[static_cast<size_t>(i) * static_cast<size_t>(qkv_dim) + static_cast<size_t>(head_off)];
            const float* vsrc = &qkv[static_cast<size_t>(i) * static_cast<size_t>(qkv_dim) + 2ULL * embed_dim + static_cast<size_t>(head_off)];
            float* qdst = &Qh[static_cast<size_t>(i) * static_cast<size_t>(head_dim)];
            float* vdst = &Vh[static_cast<size_t>(i) * static_cast<size_t>(head_dim)];
            std::memcpy(qdst, qsrc, static_cast<size_t>(head_dim) * sizeof(float));
            std::memcpy(vdst, vsrc, static_cast<size_t>(head_dim) * sizeof(float));
        }

        // Build K_h^T as [head_dim x seq_len] for GEMM
        for (int j = 0; j < seq_len; ++j) {
            const float* ksrc = &qkv[static_cast<size_t>(j) * static_cast<size_t>(qkv_dim) + static_cast<size_t>(embed_dim) + static_cast<size_t>(head_off)];
            for (int k = 0; k < head_dim; ++k) {
                KhT[static_cast<size_t>(k) * static_cast<size_t>(seq_len) + static_cast<size_t>(j)] = ksrc[k];
            }
        }

        // Bound score storage independently of the number of query tokens.
        for (int query_begin = 0; query_begin < seq_len; query_begin += query_tile) {
            const int rows = std::min(query_tile, seq_len - query_begin);
            // scores for the current query tile.
            HardwareOpt::matmul_fma_saturated(scores.data(), Qh.data() + static_cast<size_t>(query_begin) * head_dim, KhT.data(),
                                              static_cast<size_t>(rows),
                                              static_cast<size_t>(seq_len),
                                              static_cast<size_t>(head_dim));

            // Scale
            const size_t scores_n = static_cast<size_t>(rows) * static_cast<size_t>(seq_len);
            #pragma omp simd
            for (size_t idx = 0; idx < scores_n; ++idx) {
                scores[idx] *= scale;
            }

            // Causal mask (in-place)
            if (causal) {
                for (int i = 0; i < rows; ++i) {
                    float* row = &scores[static_cast<size_t>(i) * static_cast<size_t>(seq_len)];
                    for (int j = query_begin + i + 1; j < seq_len; ++j) {
                        row[j] = -1e9f;
                    }
                }
            }

            // Softmax per row (in-place)
            RuntimeParallel::forEach(rows, RuntimeParallel::workers(static_cast<long long>(seq_len) * seq_len), [&](int i) {
                float* row = &scores[static_cast<size_t>(i) * static_cast<size_t>(seq_len)];
                float max_val = row[0];
                for (int j = 1; j < seq_len; ++j) {
                    if (row[j] > max_val) max_val = row[j];
                }

                float sum = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    const float e = std::exp(row[j] - max_val);
                    row[j] = e;
                    sum += e;
                }
                const float inv_sum = 1.0f / (sum + 1e-9f);
                #pragma omp simd
                for (int j = 0; j < seq_len; ++j) {
                    row[j] *= inv_sum;
                }
            });

            // Context for the current query tile.
            HardwareOpt::matmul_fma_saturated(context.data(), scores.data(), Vh.data(),
                                              static_cast<size_t>(rows),
                                              static_cast<size_t>(head_dim),
                                              static_cast<size_t>(seq_len));

            // Write context into attended (head slice)
            for (int i = 0; i < rows; ++i) {
                float* dst = &attended[static_cast<size_t>(query_begin + i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(head_off)];
                const float* src = &context[static_cast<size_t>(i) * static_cast<size_t>(head_dim)];
                std::memcpy(dst, src, static_cast<size_t>(head_dim) * sizeof(float));
            }
        }
    }

    // 6. Output projection
    std::vector<float> output(seq_len * embed_dim);
    matmul(attended, out_weight, output, seq_len, embed_dim, embed_dim);
    if (!out_bias.empty()) {
        RuntimeParallel::forEach(seq_len, RuntimeParallel::workers(static_cast<long long>(seq_len) * embed_dim), [&](int i) {
            float* row = &output[static_cast<size_t>(i) * static_cast<size_t>(embed_dim)];
            #pragma omp simd
            for (int j = 0; j < embed_dim; ++j) {
                row[j] += out_bias[static_cast<size_t>(j)];
            }
        });
    }

    return output;
}

// Multi-Head Attention (uses multiple heads)
inline std::vector<float> multihead_attention_forward(
    const std::vector<float>& input,
    RuntimeScratch::ConstFloatView qkv_weight,
    RuntimeScratch::ConstFloatView out_weight,
    RuntimeScratch::ConstFloatView qkv_bias,
    RuntimeScratch::ConstFloatView out_bias,
    int seq_len,
    int embed_dim,
    int num_heads,
    bool causal = false
) {
    // Pour l'instant, on simplifie en utilisant self_attention
    // Une vraie implémentation diviserait en heads parallèles
    return self_attention_forward(input, qkv_weight, out_weight,
                                   qkv_bias, out_bias,
                                   seq_len, embed_dim, num_heads, causal);
}

// Cross-Attention (Q from one source, K/V from another)
inline std::vector<float> cross_attention_forward(
    const std::vector<float>& query_input,
    const std::vector<float>& kv_input,
    RuntimeScratch::ConstFloatView q_weight,
    RuntimeScratch::ConstFloatView kv_weight,
    RuntimeScratch::ConstFloatView out_weight,
    RuntimeScratch::ConstFloatView out_bias,
    int query_len,
    int kv_len,
    int embed_dim,
    int kv_embed_dim,
    int num_heads = 1,
    bool causal = false
) {
    if (embed_dim % num_heads != 0) {
        throw std::runtime_error("CrossAttention: embed_dim must be divisible by num_heads");
    }
    const int head_dim = embed_dim / num_heads;

    if (query_len <= 0 || kv_len <= 0 || embed_dim <= 0) {
        throw std::runtime_error("CrossAttention: invalid dimensions");
    }
    if (query_input.size() != static_cast<size_t>(query_len * embed_dim)) {
        throw std::runtime_error("CrossAttention: query_input size mismatch");
    }
    if (kv_embed_dim <= 0) {
        throw std::runtime_error("CrossAttention: invalid kv_embed_dim");
    }
    if (kv_input.size() != static_cast<size_t>(kv_len * kv_embed_dim)) {
        throw std::runtime_error("CrossAttention: kv_input size mismatch");
    }
    if (q_weight.size() != static_cast<size_t>(embed_dim * embed_dim)) {
        throw std::runtime_error("CrossAttention: q_weight size mismatch");
    }
    if (kv_weight.size() != static_cast<size_t>(kv_embed_dim * (2 * embed_dim))) {
        throw std::runtime_error("CrossAttention: kv_weight size mismatch");
    }
    if (out_weight.size() != static_cast<size_t>(embed_dim * embed_dim)) {
        throw std::runtime_error("CrossAttention: out_weight size mismatch");
    }

    // 1) Projections: Q = query @ Wq, KV = kv @ Wkv (produces K||V)
    constexpr int query_tile = 128;
    const int tile_rows = std::min(query_tile, query_len);
    RuntimeScratch::FloatArena workspace({size_t(query_len)*embed_dim,size_t(kv_len)*(2*embed_dim),
        size_t(query_len)*embed_dim,size_t(query_len)*head_dim,size_t(kv_len)*head_dim,
        size_t(head_dim)*kv_len,size_t(tile_rows)*kv_len,size_t(tile_rows)*head_dim});
    auto Q=workspace.take(size_t(query_len)*embed_dim);
    matmul(query_input, q_weight, Q, query_len, embed_dim, embed_dim);

    auto KV=workspace.take(size_t(kv_len)*(2*embed_dim));
    matmul(kv_input, kv_weight, KV, kv_len, kv_embed_dim, 2 * embed_dim);

    // 2) Multi-head attention (GEMM-based):
    // scores = Q_h @ K_h^T, softmax(scores), context = scores @ V_h
    auto attended=workspace.take(size_t(query_len)*embed_dim);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto Qh=workspace.take(size_t(query_len)*head_dim);
    auto Vh=workspace.take(size_t(kv_len)*head_dim);
    auto KhT=workspace.take(size_t(head_dim)*kv_len);
    auto scores=workspace.take(size_t(tile_rows)*kv_len);
    auto context=workspace.take(size_t(tile_rows)*head_dim);

    for (int h = 0; h < num_heads; ++h) {
        const int head_off = h * head_dim;

        // Pack Q_h [query_len x head_dim]
        for (int i = 0; i < query_len; ++i) {
            const float* qsrc = &Q[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(head_off)];
            float* qdst = &Qh[static_cast<size_t>(i) * static_cast<size_t>(head_dim)];
            std::memcpy(qdst, qsrc, static_cast<size_t>(head_dim) * sizeof(float));
        }

        // Pack V_h [kv_len x head_dim]
        for (int j = 0; j < kv_len; ++j) {
            const float* vsrc = &KV[static_cast<size_t>(j) * (2ULL * embed_dim) + static_cast<size_t>(embed_dim) + static_cast<size_t>(head_off)];
            float* vdst = &Vh[static_cast<size_t>(j) * static_cast<size_t>(head_dim)];
            std::memcpy(vdst, vsrc, static_cast<size_t>(head_dim) * sizeof(float));
        }

        // Build K_h^T as [head_dim x kv_len]
        for (int j = 0; j < kv_len; ++j) {
            const float* ksrc = &KV[static_cast<size_t>(j) * (2ULL * embed_dim) + static_cast<size_t>(head_off)];
            for (int k = 0; k < head_dim; ++k) {
                KhT[static_cast<size_t>(k) * static_cast<size_t>(kv_len) + static_cast<size_t>(j)] = ksrc[k];
            }
        }

        for (int query_begin = 0; query_begin < query_len; query_begin += query_tile) {
            const int rows = std::min(query_tile, query_len - query_begin);
            // scores for the current query tile.
            HardwareOpt::matmul_fma_saturated(scores.data(), Qh.data() + static_cast<size_t>(query_begin) * head_dim, KhT.data(),
                                              static_cast<size_t>(rows),
                                              static_cast<size_t>(kv_len),
                                              static_cast<size_t>(head_dim));

            // Scale
            const size_t scores_n = static_cast<size_t>(rows) * static_cast<size_t>(kv_len);
            #pragma omp simd
            for (size_t idx = 0; idx < scores_n; ++idx) {
                scores[idx] *= scale;
            }

            // Causal mask (only meaningful when kv is aligned with query timeline)
            if (causal) {
                const int diag = std::min(rows, kv_len - query_begin);
                for (int i = 0; i < diag; ++i) {
                    float* row = &scores[static_cast<size_t>(i) * static_cast<size_t>(kv_len)];
                    for (int j = query_begin + i + 1; j < kv_len; ++j) {
                        row[j] = -1e9f;
                    }
                }
            }

            // Softmax per row (in-place)
            RuntimeParallel::forEach(rows, RuntimeParallel::workers(static_cast<long long>(query_len) * kv_len), [&](int i) {
                float* row = &scores[static_cast<size_t>(i) * static_cast<size_t>(kv_len)];
                float max_val = row[0];
                for (int j = 1; j < kv_len; ++j) {
                    if (row[j] > max_val) max_val = row[j];
                }

                float sum = 0.0f;
                for (int j = 0; j < kv_len; ++j) {
                    const float e = std::exp(row[j] - max_val);
                    row[j] = e;
                    sum += e;
                }
                const float inv_sum = 1.0f / (sum + 1e-9f);
                #pragma omp simd
                for (int j = 0; j < kv_len; ++j) {
                    row[j] *= inv_sum;
                }
            });

            // Context for the current query tile.
            HardwareOpt::matmul_fma_saturated(context.data(), scores.data(), Vh.data(),
                                              static_cast<size_t>(rows),
                                              static_cast<size_t>(head_dim),
                                              static_cast<size_t>(kv_len));

            // Write context into attended
            for (int i = 0; i < rows; ++i) {
                float* dst = &attended[static_cast<size_t>(query_begin + i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(head_off)];
                const float* src = &context[static_cast<size_t>(i) * static_cast<size_t>(head_dim)];
                std::memcpy(dst, src, static_cast<size_t>(head_dim) * sizeof(float));
            }
        }
    }

    // 3) Output projection
    std::vector<float> output(query_len * embed_dim);
    matmul(attended, out_weight, output, query_len, embed_dim, embed_dim);
    if (!out_bias.empty()) {
        if (out_bias.size() != static_cast<size_t>(embed_dim)) {
            throw std::runtime_error("CrossAttention: out_bias size mismatch");
        }
        for (int i = 0; i < query_len; ++i) {
            float* row = &output[static_cast<size_t>(i) * static_cast<size_t>(embed_dim)];
            #pragma omp simd
            for (int j = 0; j < embed_dim; ++j) {
                row[j] += out_bias[static_cast<size_t>(j)];
            }
        }
    }
    return output;
}

} // namespace LayerOps
