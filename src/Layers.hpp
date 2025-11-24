#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>
#include <functional>

// ============================================================================
// Énumérations et structures
// ============================================================================

enum class ActivationType {
    NONE,
    RELU,
    RELU6,
    LEAKY_RELU,
    ELU,
    SELU,
    GELU,
    SWISH,
    MISH,
    TANH,
    SIGMOID,
    SOFTMAX,
    SOFTPLUS,
    SOFTSIGN
};

enum class PoolingType {
    MAX,
    AVERAGE,
    MIN,
    SUM
};

enum class PaddingMode {
    ZEROS,
    REFLECT,
    REPLICATE,
    CIRCULAR
};

enum class NormalizationType {
    BATCH_NORM,
    LAYER_NORM,
    INSTANCE_NORM,
    GROUP_NORM,
    RMS_NORM
};

// ============================================================================
// Layer de base
// ============================================================================

struct Layer {
    std::string name;
    std::string type;
    size_t params_count;
    
    // Données des paramètres (weights, bias)
    std::vector<float> weights;
    std::vector<float> bias;
    
    // Gradients (pour backprop)
    std::vector<float> grad_weights;
    std::vector<float> grad_bias;
    
    // État interne (pour BatchNorm, etc.)
    std::vector<float> running_mean;
    std::vector<float> running_var;
    
    // Configuration
    int in_features = 0;
    int out_features = 0;
    int kernel_size = 0;
    int stride = 1;
    int padding = 0;
    int dilation = 1;
    int groups = 1;
    bool use_bias = true;
    
    ActivationType activation = ActivationType::NONE;
    float activation_param = 0.0f; // Pour LeakyReLU alpha, ELU alpha, etc.
    
    Layer() = default;
    Layer(const std::string& n, const std::string& t, size_t pc)
        : name(n), type(t), params_count(pc) {}
};

// ============================================================================
// Fonctions d'activation
// ============================================================================

namespace Activation {

// ReLU: max(0, x)
inline float relu(float x) {
    return std::max(0.0f, x);
}

inline void relu2d(std::vector<float>& data, int width, int height) {
    for (auto& val : data) {
        val = relu(val);
    }
}

inline void relu3d(std::vector<float>& data, int width, int height, int depth) {
    for (auto& val : data) {
        val = relu(val);
    }
}

// ReLU6: min(max(0, x), 6)
inline float relu6(float x) {
    return std::min(std::max(0.0f, x), 6.0f);
}

// Leaky ReLU: x if x > 0 else alpha * x
inline float leaky_relu(float x, float alpha = 0.01f) {
    return x > 0.0f ? x : alpha * x;
}

// ELU: x if x > 0 else alpha * (exp(x) - 1)
inline float elu(float x, float alpha = 1.0f) {
    return x > 0.0f ? x : alpha * (std::exp(x) - 1.0f);
}

// SELU: scaled exponential linear unit
inline float selu(float x) {
    constexpr float alpha = 1.6732632423543772848170429916717f;
    constexpr float scale = 1.0507009873554804934193349852946f;
    return scale * (x > 0.0f ? x : alpha * (std::exp(x) - 1.0f));
}

// GELU: Gaussian Error Linear Unit
inline float gelu(float x) {
    constexpr float sqrt_2_pi = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + std::tanh(sqrt_2_pi * (x + 0.044715f * x * x * x)));
}

// Swish (SiLU): x * sigmoid(x)
inline float swish(float x) {
    return x / (1.0f + std::exp(-x));
}

// Mish: x * tanh(softplus(x))
inline float mish(float x) {
    return x * std::tanh(std::log(1.0f + std::exp(x)));
}

// Tanh
inline float tanh_act(float x) {
    return std::tanh(x);
}

// Sigmoid: 1 / (1 + exp(-x))
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Softplus: log(1 + exp(x))
inline float softplus(float x) {
    return std::log(1.0f + std::exp(x));
}

// Softsign: x / (1 + |x|)
inline float softsign(float x) {
    return x / (1.0f + std::abs(x));
}

// Softmax (pour un vecteur)
inline void softmax(std::vector<float>& logits) {
    if (logits.empty()) return;
    
    // Trouver le max pour stabilité numérique
    float max_val = *std::max_element(logits.begin(), logits.end());
    
    // Exp(x - max)
    float sum = 0.0f;
    for (auto& val : logits) {
        val = std::exp(val - max_val);
        sum += val;
    }
    
    // Normaliser
    for (auto& val : logits) {
        val /= sum;
    }
}

// Application générique d'activation
inline float apply(float x, ActivationType type, float param = 0.0f) {
    switch (type) {
        case ActivationType::NONE: return x;
        case ActivationType::RELU: return relu(x);
        case ActivationType::RELU6: return relu6(x);
        case ActivationType::LEAKY_RELU: return leaky_relu(x, param);
        case ActivationType::ELU: return elu(x, param);
        case ActivationType::SELU: return selu(x);
        case ActivationType::GELU: return gelu(x);
        case ActivationType::SWISH: return swish(x);
        case ActivationType::MISH: return mish(x);
        case ActivationType::TANH: return tanh_act(x);
        case ActivationType::SIGMOID: return sigmoid(x);
        case ActivationType::SOFTPLUS: return softplus(x);
        case ActivationType::SOFTSIGN: return softsign(x);
        default: return x;
    }
}

inline void apply_inplace(std::vector<float>& data, ActivationType type, float param = 0.0f) {
    if (type == ActivationType::SOFTMAX) {
        softmax(data);
    } else {
        for (auto& val : data) {
            val = apply(val, type, param);
        }
    }
}

} // namespace Activation

// ============================================================================
// Opérations de convolution
// ============================================================================

namespace Conv {

// Convolution 1D
inline void conv1d(const std::vector<float>& input, std::vector<float>& output,
                   const std::vector<float>& kernel, const std::vector<float>& bias,
                   int in_length, int in_channels, int out_channels,
                   int kernel_size, int stride = 1, int padding = 0, int dilation = 1) {
    
    int out_length = (in_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    output.resize(out_length * out_channels, 0.0f);
    
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int ol = 0; ol < out_length; ++ol) {
            float sum = 0.0f;
            
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int k = 0; k < kernel_size; ++k) {
                    int il = ol * stride - padding + k * dilation;
                    
                    if (il >= 0 && il < in_length) {
                        int in_idx = ic * in_length + il;
                        int kernel_idx = ((oc * in_channels + ic) * kernel_size) + k;
                        sum += input[in_idx] * kernel[kernel_idx];
                    }
                }
            }
            
            if (!bias.empty()) {
                sum += bias[oc];
            }
            
            output[oc * out_length + ol] = sum;
        }
    }
}

// Convolution 2D
inline void conv2d(const std::vector<float>& input, std::vector<float>& output,
                   const std::vector<float>& kernel, const std::vector<float>& bias,
                   int in_height, int in_width, int in_channels, int out_channels,
                   int kernel_size, int stride = 1, int padding = 0, int dilation = 1) {
    
    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    output.resize(out_height * out_width * out_channels, 0.0f);
    
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                float sum = 0.0f;
                
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int ih = oh * stride - padding + kh * dilation;
                            int iw = ow * stride - padding + kw * dilation;
                            
                            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                int in_idx = (ic * in_height + ih) * in_width + iw;
                                int kernel_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                                sum += input[in_idx] * kernel[kernel_idx];
                            }
                        }
                    }
                }
                
                if (!bias.empty()) {
                    sum += bias[oc];
                }
                
                output[(oc * out_height + oh) * out_width + ow] = sum;
            }
        }
    }
}

// Convolution 3D
inline void conv3d(const std::vector<float>& input, std::vector<float>& output,
                   const std::vector<float>& kernel, const std::vector<float>& bias,
                   int in_depth, int in_height, int in_width, int in_channels, int out_channels,
                   int kernel_size, int stride = 1, int padding = 0, int dilation = 1) {
    
    int out_depth = (in_depth + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    output.resize(out_depth * out_height * out_width * out_channels, 0.0f);
    
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int od = 0; od < out_depth; ++od) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float sum = 0.0f;
                    
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kd = 0; kd < kernel_size; ++kd) {
                            for (int kh = 0; kh < kernel_size; ++kh) {
                                for (int kw = 0; kw < kernel_size; ++kw) {
                                    int id = od * stride - padding + kd * dilation;
                                    int ih = oh * stride - padding + kh * dilation;
                                    int iw = ow * stride - padding + kw * dilation;
                                    
                                    if (id >= 0 && id < in_depth && 
                                        ih >= 0 && ih < in_height && 
                                        iw >= 0 && iw < in_width) {
                                        
                                        int in_idx = ((ic * in_depth + id) * in_height + ih) * in_width + iw;
                                        int kernel_idx = (((oc * in_channels + ic) * kernel_size + kd) * kernel_size + kh) * kernel_size + kw;
                                        sum += input[in_idx] * kernel[kernel_idx];
                                    }
                                }
                            }
                        }
                    }
                    
                    if (!bias.empty()) {
                        sum += bias[oc];
                    }
                    
                    output[((oc * out_depth + od) * out_height + oh) * out_width + ow] = sum;
                }
            }
        }
    }
}

// Transposed Convolution 2D (Deconvolution)
inline void conv_transpose2d(const std::vector<float>& input, std::vector<float>& output,
                             const std::vector<float>& kernel, const std::vector<float>& bias,
                             int in_height, int in_width, int in_channels, int out_channels,
                             int kernel_size, int stride = 1, int padding = 0) {
    
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size;
    
    output.assign(out_height * out_width * out_channels, 0.0f);
    
    // Ajouter les bias
    if (!bias.empty()) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int i = 0; i < out_height * out_width; ++i) {
                output[oc * out_height * out_width + i] = bias[oc];
            }
        }
    }
    
    // Convolution transposée
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int ih = 0; ih < in_height; ++ih) {
                for (int iw = 0; iw < in_width; ++iw) {
                    float in_val = input[(ic * in_height + ih) * in_width + iw];
                    
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int oh = ih * stride + kh - padding;
                            int ow = iw * stride + kw - padding;
                            
                            if (oh >= 0 && oh < out_height && ow >= 0 && ow < out_width) {
                                int kernel_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                                int out_idx = (oc * out_height + oh) * out_width + ow;
                                output[out_idx] += in_val * kernel[kernel_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace Conv

// ============================================================================
// Opérations de pooling
// ============================================================================

namespace Pooling {

// Max Pooling 2D
inline void maxpool2d(const std::vector<float>& input, std::vector<float>& output,
                      int in_height, int in_width, int channels,
                      int kernel_size, int stride = -1) {
    if (stride < 0) stride = kernel_size;
    
    int out_height = (in_height - kernel_size) / stride + 1;
    int out_width = (in_width - kernel_size) / stride + 1;
    
    output.resize(out_height * out_width * channels);
    
    for (int c = 0; c < channels; ++c) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                float max_val = -std::numeric_limits<float>::infinity();
                
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int ih = oh * stride + kh;
                        int iw = ow * stride + kw;
                        
                        int in_idx = (c * in_height + ih) * in_width + iw;
                        max_val = std::max(max_val, input[in_idx]);
                    }
                }
                
                output[(c * out_height + oh) * out_width + ow] = max_val;
            }
        }
    }
}

// Average Pooling 2D
inline void avgpool2d(const std::vector<float>& input, std::vector<float>& output,
                      int in_height, int in_width, int channels,
                      int kernel_size, int stride = -1) {
    if (stride < 0) stride = kernel_size;
    
    int out_height = (in_height - kernel_size) / stride + 1;
    int out_width = (in_width - kernel_size) / stride + 1;
    
    output.resize(out_height * out_width * channels);
    
    for (int c = 0; c < channels; ++c) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                float sum = 0.0f;
                
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int ih = oh * stride + kh;
                        int iw = ow * stride + kw;
                        
                        int in_idx = (c * in_height + ih) * in_width + iw;
                        sum += input[in_idx];
                    }
                }
                
                output[(c * out_height + oh) * out_width + ow] = sum / (kernel_size * kernel_size);
            }
        }
    }
}

// Adaptive Average Pooling 2D
inline void adaptive_avgpool2d(const std::vector<float>& input, std::vector<float>& output,
                                int in_height, int in_width, int channels,
                                int out_height, int out_width) {
    output.resize(out_height * out_width * channels);
    
    for (int c = 0; c < channels; ++c) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                int h_start = static_cast<int>(std::floor(static_cast<float>(oh * in_height) / out_height));
                int h_end = static_cast<int>(std::ceil(static_cast<float>((oh + 1) * in_height) / out_height));
                int w_start = static_cast<int>(std::floor(static_cast<float>(ow * in_width) / out_width));
                int w_end = static_cast<int>(std::ceil(static_cast<float>((ow + 1) * in_width) / out_width));
                
                float sum = 0.0f;
                int count = 0;
                
                for (int ih = h_start; ih < h_end; ++ih) {
                    for (int iw = w_start; iw < w_end; ++iw) {
                        int in_idx = (c * in_height + ih) * in_width + iw;
                        sum += input[in_idx];
                        count++;
                    }
                }
                
                output[(c * out_height + oh) * out_width + ow] = sum / count;
            }
        }
    }
}

} // namespace Pooling

// ============================================================================
// Opérations de normalisation
// ============================================================================

namespace Normalization {

// Batch Normalization
inline void batch_norm(std::vector<float>& data, const std::vector<float>& gamma,
                       const std::vector<float>& beta, const std::vector<float>& running_mean,
                       const std::vector<float>& running_var, int batch_size, int channels,
                       int spatial_size, float eps = 1e-5f, bool training = false) {
    
    for (int c = 0; c < channels; ++c) {
        float mean = running_mean[c];
        float var = running_var[c];
        
        if (training) {
            // Calculer mean et var pour ce batch
            mean = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                for (int s = 0; s < spatial_size; ++s) {
                    mean += data[b * channels * spatial_size + c * spatial_size + s];
                }
            }
            mean /= (batch_size * spatial_size);
            
            var = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                for (int s = 0; s < spatial_size; ++s) {
                    float diff = data[b * channels * spatial_size + c * spatial_size + s] - mean;
                    var += diff * diff;
                }
            }
            var /= (batch_size * spatial_size);
        }
        
        float inv_std = 1.0f / std::sqrt(var + eps);
        
        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < spatial_size; ++s) {
                int idx = b * channels * spatial_size + c * spatial_size + s;
                data[idx] = (data[idx] - mean) * inv_std * gamma[c] + beta[c];
            }
        }
    }
}

// Layer Normalization
inline void layer_norm(std::vector<float>& data, const std::vector<float>& gamma,
                       const std::vector<float>& beta, int normalized_size,
                       float eps = 1e-5f) {
    
    int num_groups = data.size() / normalized_size;
    
    for (int g = 0; g < num_groups; ++g) {
        float mean = 0.0f;
        for (int i = 0; i < normalized_size; ++i) {
            mean += data[g * normalized_size + i];
        }
        mean /= normalized_size;
        
        float var = 0.0f;
        for (int i = 0; i < normalized_size; ++i) {
            float diff = data[g * normalized_size + i] - mean;
            var += diff * diff;
        }
        var /= normalized_size;
        
        float inv_std = 1.0f / std::sqrt(var + eps);
        
        for (int i = 0; i < normalized_size; ++i) {
            int idx = g * normalized_size + i;
            data[idx] = (data[idx] - mean) * inv_std * gamma[i] + beta[i];
        }
    }
}

// RMS Normalization (Root Mean Square)
inline void rms_norm(std::vector<float>& data, const std::vector<float>& gamma,
                     int normalized_size, float eps = 1e-6f) {
    
    int num_groups = data.size() / normalized_size;
    
    for (int g = 0; g < num_groups; ++g) {
        float rms = 0.0f;
        for (int i = 0; i < normalized_size; ++i) {
            float val = data[g * normalized_size + i];
            rms += val * val;
        }
        rms = std::sqrt(rms / normalized_size + eps);
        
        for (int i = 0; i < normalized_size; ++i) {
            int idx = g * normalized_size + i;
            data[idx] = data[idx] / rms * gamma[i];
        }
    }
}

} // namespace Normalization

// ============================================================================
// Opérations d'accumulation et de comparaison
// ============================================================================

namespace Operations {

// Addition élément par élément
inline void add(std::vector<float>& a, const std::vector<float>& b) {
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        a[i] += b[i];
    }
}

// Soustraction élément par élément
inline void subtract(std::vector<float>& a, const std::vector<float>& b) {
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        a[i] -= b[i];
    }
}

// Multiplication élément par élément (Hadamard)
inline void multiply(std::vector<float>& a, const std::vector<float>& b) {
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        a[i] *= b[i];
    }
}

// Division élément par élément
inline void divide(std::vector<float>& a, const std::vector<float>& b, float eps = 1e-8f) {
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        a[i] /= (b[i] + eps);
    }
}

// Multiplication scalaire
inline void scale(std::vector<float>& a, float scalar) {
    for (auto& val : a) {
        val *= scalar;
    }
}

// Accumulation pondérée: a = alpha * a + beta * b
inline void accumulate(std::vector<float>& a, const std::vector<float>& b, 
                       float alpha = 1.0f, float beta = 1.0f) {
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        a[i] = alpha * a[i] + beta * b[i];
    }
}

// Clipping
inline void clip(std::vector<float>& a, float min_val, float max_val) {
    for (auto& val : a) {
        val = std::clamp(val, min_val, max_val);
    }
}

// Comparaison: retourne 1.0 si condition vraie, 0.0 sinon
inline void greater_than(const std::vector<float>& a, const std::vector<float>& b, 
                         std::vector<float>& result) {
    result.resize(std::min(a.size(), b.size()));
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = (a[i] > b[i]) ? 1.0f : 0.0f;
    }
}

inline void less_than(const std::vector<float>& a, const std::vector<float>& b,
                      std::vector<float>& result) {
    result.resize(std::min(a.size(), b.size()));
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = (a[i] < b[i]) ? 1.0f : 0.0f;
    }
}

inline void equal(const std::vector<float>& a, const std::vector<float>& b,
                  std::vector<float>& result, float tolerance = 1e-6f) {
    result.resize(std::min(a.size(), b.size()));
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = (std::abs(a[i] - b[i]) < tolerance) ? 1.0f : 0.0f;
    }
}

// Réduction: somme
inline float sum(const std::vector<float>& a) {
    float s = 0.0f;
    for (auto val : a) s += val;
    return s;
}

// Réduction: moyenne
inline float mean(const std::vector<float>& a) {
    return a.empty() ? 0.0f : sum(a) / a.size();
}

// Réduction: max
inline float max(const std::vector<float>& a) {
    return a.empty() ? 0.0f : *std::max_element(a.begin(), a.end());
}

// Réduction: min
inline float min(const std::vector<float>& a) {
    return a.empty() ? 0.0f : *std::min_element(a.begin(), a.end());
}

// Norme L2
inline float norm_l2(const std::vector<float>& a) {
    float s = 0.0f;
    for (auto val : a) s += val * val;
    return std::sqrt(s);
}

// Produit scalaire (dot product)
inline float dot(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0.0f;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

} // namespace Operations
