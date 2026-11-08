#include "runtimes/LayerOps.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "SIMD_Ops.hpp"
#include "runtimes/cpu/Parallel.hpp"
#include "runtimes/cpu/ConvolutionKernels.hpp"
#include "runtimes/cpu/NormalizationKernels.hpp"
#include "runtimes/cpu/PoolingKernels.hpp"

namespace {

std::atomic<bool> g_hardware_acceleration{true};

bool useHardware(const bool requested) {
    return requested && g_hardware_acceleration.load(std::memory_order_relaxed);
}

void softmaxInPlace(std::vector<float>& values) {
    if (values.empty()) return;
    const float maximum = *std::max_element(values.begin(), values.end());
    float sum = 0.0f;
    for (float& value : values) {
        value = std::exp(value - maximum);
        sum += value;
    }
    const float inverse_sum = 1.0f / sum;
    for (float& value : values) value *= inverse_sum;
}

} // namespace

namespace RuntimeLayerOps {

void setHardwareAcceleration(const bool enable) {
    g_hardware_acceleration.store(enable, std::memory_order_relaxed);
}

bool hardwareAccelerationEnabled() {
    return g_hardware_acceleration.load(std::memory_order_relaxed);
}

void computeConv2D(const std::vector<float>& input, std::vector<float>& output,
                   const LayerParams& params, const int in_h, const int in_w,
                   const int in_c, const int out_c, bool) {
    if (in_h <= 0 || in_w <= 0 || in_c <= 0 || out_c <= 0 || params.kernel_size <= 0 ||
        params.stride <= 0 || params.padding < 0 || params.dilation <= 0 ||
        input.size() / static_cast<size_t>(in_c) < static_cast<size_t>(in_h) * in_w ||
        params.weights.size() / static_cast<size_t>(out_c) / static_cast<size_t>(in_c) /
            static_cast<size_t>(params.kernel_size) < static_cast<size_t>(params.kernel_size) ||
        (!params.bias.empty() && params.bias.size() < static_cast<size_t>(out_c))) {
        throw std::invalid_argument("RuntimeLayerOps::computeConv2D: dimensions invalides");
    }
    CpuConvolution::forward(input, output, params.weights.data(),
        params.bias.empty() ? nullptr : params.bias.data(), in_h, in_w, in_c, out_c,
        params.kernel_size, params.stride, params.padding, params.dilation);
}

void computeLinear(const std::vector<float>& input, std::vector<float>& output,
                   const LayerParams& params, const bool use_hardware) {
    if (params.in_features < 0 || params.out_features < 0 ||
        input.size() < static_cast<size_t>(params.in_features) ||
        params.weights.size() < static_cast<size_t>(params.in_features) * params.out_features ||
        (!params.bias.empty() && params.bias.size() < static_cast<size_t>(params.out_features))) {
        throw std::invalid_argument("RuntimeLayerOps::computeLinear: dimensions invalides");
    }
    output.assign(static_cast<size_t>(params.out_features), 0.0f);
    if (useHardware(use_hardware)) {
        SIMD::matmul_transpose_avx2(output.data(), input.data(), params.weights.data(),
                         1, params.out_features, params.in_features);
        if (!params.bias.empty()) {
            SIMD::add_vectors_avx2(output.data(), output.data(), params.bias.data(), params.out_features);
        }
        return;
    }
    for (int output_index = 0; output_index < params.out_features; ++output_index) {
        float sum = 0.0f;
        for (int input_index = 0; input_index < params.in_features; ++input_index) {
            sum = std::fma(input[input_index], params.weights[output_index * params.in_features + input_index], sum);
        }
        if (!params.bias.empty()) sum += params.bias[output_index];
        output[output_index] = sum;
    }
}

void computeMaxPool2D(const std::vector<float>& input, std::vector<float>& output,
                      const int in_h, const int in_w, const int channels,
                      const int kernel_size, int stride, bool) {
    if (stride < 0) stride = kernel_size;
    CpuPooling::forward2d(input, output, in_h, in_w, channels, kernel_size, kernel_size,
        stride, stride, 0, 0, CpuPooling::Reduction::Max);
}

void computeAvgPool2D(const std::vector<float>& input, std::vector<float>& output,
                      const int in_h, const int in_w, const int channels,
                      const int kernel_size, int stride, bool) {
    if (stride < 0) stride = kernel_size;
    CpuPooling::forward2d(input, output, in_h, in_w, channels, kernel_size, kernel_size,
        stride, stride, 0, 0, CpuPooling::Reduction::Average);
}

void computeActivation(std::vector<float>& data, const ActivationType activation_type,
                       const float param, const bool use_hardware) {
    if (data.empty() || activation_type == ActivationType::NONE) return;
    if (activation_type == ActivationType::SOFTMAX && useHardware(use_hardware)) {
        SIMD::softmax_avx2(data.data(), data.data(), data.size());
        return;
    }
    const auto transform = [&](const auto& operation) {
        RuntimeParallel::forEach(data.size(), RuntimeParallel::workers(data.size()), [&](size_t index) {
            data[index] = operation(data[index]);
        });
    };
    int unary_op = -1;
    switch (activation_type) {
        case ActivationType::RELU: unary_op = 0; break;
        case ActivationType::LEAKY_RELU: unary_op = 1; break;
        case ActivationType::SIGMOID: unary_op = 2; break;
        case ActivationType::TANH: unary_op = 3; break;
        case ActivationType::SWISH: unary_op = 4; break;
        case ActivationType::GELU: unary_op = 5; break;
        case ActivationType::SOFTPLUS: unary_op = 6; break;
        case ActivationType::MISH: unary_op = 7; break;
        case ActivationType::SOFTMAX: softmaxInPlace(data); return;
        case ActivationType::RELU6:
            transform([](float value) { return std::clamp(value, 0.0f, 6.0f); });
            return;
        case ActivationType::ELU:
            transform([param](float value) { return value > 0.0f ? value : param * std::expm1(value); });
            return;
        case ActivationType::SELU:
            transform([](float value) {
                constexpr float alpha = 1.6732632423543773f;
                constexpr float scale = 1.0507009873554805f;
                return scale * (value > 0.0f ? value : alpha * std::expm1(value));
            });
            return;
        case ActivationType::SOFTSIGN:
            transform([](float value) { return value / (1.0f + std::abs(value)); });
            return;
        default: return;
    }
    // Reuse the host pointwise implementation, including stable Softplus/Mish.
    unaryForwardHost(data, data, unary_op, param);
}

void computeActivation(std::vector<float>& data, const std::string& activation_type,
                       const float param, const bool use_hardware) {
    ActivationType type = ActivationType::NONE;
    if (activation_type == "relu") type = ActivationType::RELU;
    else if (activation_type == "relu6") type = ActivationType::RELU6;
    else if (activation_type == "leaky_relu") type = ActivationType::LEAKY_RELU;
    else if (activation_type == "elu") type = ActivationType::ELU;
    else if (activation_type == "selu") type = ActivationType::SELU;
    else if (activation_type == "gelu") type = ActivationType::GELU;
    else if (activation_type == "silu" || activation_type == "swish") type = ActivationType::SWISH;
    else if (activation_type == "mish") type = ActivationType::MISH;
    else if (activation_type == "tanh") type = ActivationType::TANH;
    else if (activation_type == "sigmoid") type = ActivationType::SIGMOID;
    else if (activation_type == "softmax") type = ActivationType::SOFTMAX;
    else if (activation_type == "softplus") type = ActivationType::SOFTPLUS;
    else if (activation_type == "softsign") type = ActivationType::SOFTSIGN;
    computeActivation(data, type, param, use_hardware);
}

void computeBatchNorm(std::vector<float>& data, const std::vector<float>& gamma,
                      const std::vector<float>& beta, const std::vector<float>& running_mean,
                      const std::vector<float>& running_var, const int batch_size,
                      const int channels, const int spatial_size, const float eps,
                      const bool training, bool) {
    CpuNormalization::batchNormInPlace(data, gamma, beta, running_mean, running_var,
                                     batch_size, channels, spatial_size, eps, training);
}

void computeLayerNorm(std::vector<float>& data, const std::vector<float>& gamma,
                      const std::vector<float>& beta, const int normalized_size,
                      const float eps, bool) {
    CpuNormalization::layerNormInPlace(data, gamma, beta, normalized_size, eps);
}

bool prepareLayerNorm(const std::vector<float>& input, const int normalized_size,
                      const float eps, std::vector<float>& normalized) {
    if (normalized_size <= 0 || input.empty() ||
        input.size() % static_cast<size_t>(normalized_size) != 0) {
        return false;
    }

    normalized.resize(input.size());
    const int groups = static_cast<int>(input.size() / static_cast<size_t>(normalized_size));
    for (int group = 0; group < groups; ++group) {
        const size_t base = static_cast<size_t>(group) * normalized_size;
        float mean = 0.0f;
        for (int index = 0; index < normalized_size; ++index) {
            mean += input[base + static_cast<size_t>(index)];
        }
        mean /= static_cast<float>(normalized_size);

        float variance = 0.0f;
        for (int index = 0; index < normalized_size; ++index) {
            const float centered = input[base + static_cast<size_t>(index)] - mean;
            variance += centered * centered;
        }
        const float inverse_stddev =
            1.0f / std::sqrt(variance / static_cast<float>(normalized_size) + eps);
        for (int index = 0; index < normalized_size; ++index) {
            normalized[base + static_cast<size_t>(index)] =
                (input[base + static_cast<size_t>(index)] - mean) * inverse_stddev;
        }
    }
    return true;
}

void computeConvTranspose2D(const std::vector<float>& input, std::vector<float>& output,
                            const LayerParams& params, const int in_h, const int in_w,
                            const int in_c, const int out_c, bool) {
    if (in_c <= 0 || out_c <= 0 || params.kernel_size <= 0 ||
        params.weights.size() / static_cast<size_t>(out_c) / static_cast<size_t>(in_c) /
            static_cast<size_t>(params.kernel_size) < static_cast<size_t>(params.kernel_size) ||
        (!params.bias.empty() && params.bias.size() < static_cast<size_t>(out_c))) {
        throw std::invalid_argument("RuntimeLayerOps::computeConvTranspose2D: dimensions invalides");
    }
    CpuConvolution::transposeForward(input, output, params.weights.data(),
        params.bias.empty() ? nullptr : params.bias.data(), in_h, in_w, in_c, out_c,
        params.kernel_size, params.stride, params.padding);
}

void computeAttention(const std::vector<float>& query, const std::vector<float>& key,
                      const std::vector<float>& value, std::vector<float>& output,
                      const int seq_len, const int d_model, const int num_heads,
                      const bool use_hardware) {
    if (seq_len <= 0 || d_model <= 0 || num_heads <= 0 || d_model % num_heads != 0)
        throw std::invalid_argument("RuntimeLayerOps::computeAttention: dimensions invalides");
    const size_t tensor_size = static_cast<size_t>(seq_len) * d_model;
    if (query.size() < tensor_size || key.size() < tensor_size || value.size() < tensor_size)
        throw std::invalid_argument("RuntimeLayerOps::computeAttention: tenseur trop petit");
    const size_t head_dim = static_cast<size_t>(d_model / num_heads);
    const size_t positions = static_cast<size_t>(seq_len);
    output.assign(tensor_size, 0.0f);
    std::vector<float> scores(positions * positions), q(positions * head_dim), k(q.size());
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int head = 0; head < num_heads; ++head) {
        for (size_t row = 0; row < positions; ++row) {
            const size_t source = row * d_model + static_cast<size_t>(head) * head_dim;
            std::copy_n(query.data() + source, head_dim, q.data() + row * head_dim);
            std::copy_n(key.data() + source, head_dim, k.data() + row * head_dim);
        }
        if (useHardware(use_hardware)) {
            SIMD::matmul_transpose_avx2(scores.data(), q.data(), k.data(), positions, positions, head_dim);
        } else {
            for (size_t row = 0; row < positions; ++row)
                for (size_t column = 0; column < positions; ++column) {
                    float sum = 0.0f;
                    for (size_t feature = 0; feature < head_dim; ++feature)
                        sum = std::fma(q[row * head_dim + feature], k[column * head_dim + feature], sum);
                    scores[row * positions + column] = sum;
                }
        }
        for (float& score : scores) score *= scale;
        for (size_t row = 0; row < positions; ++row) {
            auto begin = scores.begin() + row * positions;
            std::vector<float> probabilities(begin, begin + positions);
            softmaxInPlace(probabilities);
            std::copy(probabilities.begin(), probabilities.end(), begin);
        }
        for (size_t row = 0; row < positions; ++row)
            for (size_t feature = 0; feature < head_dim; ++feature) {
                float sum = 0.0f;
                for (size_t column = 0; column < positions; ++column)
                    sum = std::fma(scores[row * positions + column],
                        value[column * d_model + static_cast<size_t>(head) * head_dim + feature], sum);
                output[row * d_model + static_cast<size_t>(head) * head_dim + feature] = sum;
            }
    }
}

void conv2dSame(const std::vector<float>& input, std::vector<float>& output,
                const int width, const int height, const std::vector<float>& kernel,
                const int kernel_size) {
    if (width < 0 || height < 0 || kernel_size <= 0 ||
        input.size() < static_cast<size_t>(width) * height ||
        kernel.size() < static_cast<size_t>(kernel_size) * kernel_size) {
        throw std::invalid_argument("RuntimeLayerOps::conv2dSame: dimensions invalides");
    }
    LayerParams params;
    params.weights = kernel;
    params.kernel_size = kernel_size;
    params.padding = kernel_size / 2;
    computeConv2D(input, output, params, height, width, 1, 1, true);
}

void branchMerge(const std::vector<float>& branch1, const std::vector<float>& branch2,
                 std::vector<float>& output, const MergeOperation merge_op,
                 const bool use_hardware) {
    if (merge_op == MergeOperation::CONCATENATE) {
        output = branch1;
        output.insert(output.end(), branch2.begin(), branch2.end());
        return;
    }
    if (branch1.size() != branch2.size()) {
        throw std::invalid_argument("RuntimeLayerOps::branchMerge: tailles incompatibles");
    }
    output.resize(branch1.size());
    size_t index = 0;
    if (useHardware(use_hardware) && merge_op == MergeOperation::ADD) {
        SIMD::add_vectors_avx2(output.data(), branch1.data(), branch2.data(), branch1.size());
        return;
    }
    if (useHardware(use_hardware) && merge_op == MergeOperation::MULTIPLY) {
        SIMD::mul_vectors_avx2(output.data(), branch1.data(), branch2.data(), branch1.size());
        return;
    }
    for (; index < branch1.size(); ++index) {
        switch (merge_op) {
            case MergeOperation::MULTIPLY: output[index] = branch1[index] * branch2[index]; break;
            case MergeOperation::MAX: output[index] = std::max(branch1[index], branch2[index]); break;
            case MergeOperation::AVERAGE: output[index] = (branch1[index] + branch2[index]) * 0.5f; break;
            default: output[index] = branch1[index] + branch2[index]; break;
        }
    }
}

void branchSplit(const std::vector<float>& input, std::vector<std::vector<float>>& outputs,
                 const std::vector<int>& split_sizes) {
    size_t total_size = 0;
    for (const int split_size : split_sizes) {
        if (split_size < 0 || static_cast<size_t>(split_size) > input.size() - total_size) {
            throw std::invalid_argument("RuntimeLayerOps::branchSplit: taille de split invalide");
        }
        total_size += static_cast<size_t>(split_size);
    }
    if (total_size != input.size()) {
        throw std::invalid_argument("RuntimeLayerOps::branchSplit: les splits ne couvrent pas l'entree");
    }
    outputs.clear();
    outputs.reserve(split_sizes.size());
    auto begin = input.begin();
    for (const int split_size : split_sizes) {
        const auto end = begin + split_size;
        outputs.emplace_back(begin, end);
        begin = end;
    }
}

bool resolveUnaryOp(const LayerType type, const Layer& layer, int& op_code, float& alpha) {
    alpha = 0.01f;
    switch (type) {
        case LayerType::ReLU: op_code = 0; return true;
        case LayerType::LeakyReLU:
            op_code = 1;
            alpha = layer.leaky_relu_alpha > 0.0f ? layer.leaky_relu_alpha : 0.01f;
            return true;
        case LayerType::Sigmoid: op_code = 2; return true;
        case LayerType::Tanh: op_code = 3; return true;
        case LayerType::SiLU: op_code = 4; return true;
        case LayerType::GELU: op_code = 5; return true;
        case LayerType::Softplus: op_code = 6; return true;
        case LayerType::Mish: op_code = 7; return true;
        case LayerType::HardSigmoid: op_code = 8; return true;
        case LayerType::HardSwish: op_code = 9; return true;
        default:
            return false;
    }
}

bool resolveBinaryOp(const LayerType type, int& op_code) {
    switch (type) {
        case LayerType::Add: op_code = 0; return true;
        case LayerType::Subtract: op_code = 1; return true;
        case LayerType::Multiply: op_code = 2; return true;
        case LayerType::Divide: op_code = 3; return true;
        default:
            return false;
    }
}

void unaryForwardHost(const std::vector<float>& input, std::vector<float>& output, const int op_code, const float alpha) {
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        const float x = input[i];
        switch (op_code) {
            case 0: output[i] = x > 0.0f ? x : 0.0f; break;
            case 1: output[i] = x > 0.0f ? x : alpha * x; break;
            case 2: output[i] = 1.0f / (1.0f + std::exp(-x)); break;
            case 3: output[i] = std::tanh(x); break;
            case 4: {
                const float s = 1.0f / (1.0f + std::exp(-x));
                output[i] = x * s;
                break;
            }
            case 5: {
                const float c = 0.7978845608f;
                const float x3 = x * x * x;
                output[i] = 0.5f * x * (1.0f + std::tanh(c * (x + 0.044715f * x3)));
                break;
            }
            case 6: output[i] = x > 20.0f ? x : std::log1p(std::exp(x)); break;
            case 7: {
                const float sp = x > 20.0f ? x : std::log1p(std::exp(x));
                output[i] = x * std::tanh(sp);
                break;
            }
            case 8: {
                const float hs = (x + 3.0f) / 6.0f;
                output[i] = std::min(1.0f, std::max(0.0f, hs));
                break;
            }
            case 9: {
                const float hs = std::min(1.0f, std::max(0.0f, (x + 3.0f) / 6.0f));
                output[i] = x * hs;
                break;
            }
            default:
                output[i] = x;
                break;
        }
    }
}

void binaryForwardHost(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& output, const int op_code) {
    output.resize(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        switch (op_code) {
            case 0: output[i] = a[i] + b[i]; break;
            case 1: output[i] = a[i] - b[i]; break;
            case 2: output[i] = a[i] * b[i]; break;
            case 3: {
                const float d = b[i];
                const float safe_d = std::fabs(d) < 1e-8f
                    ? (d < 0.0f ? -1e-8f : 1e-8f)
                    : d;
                output[i] = a[i] / safe_d;
                break;
            }
            default:
                output[i] = a[i];
                break;
        }
    }
}

} // namespace RuntimeLayerOps
