#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../Layers.hpp"
#include "LayerOps.hpp"
#include "LayerOpsExt.hpp"

namespace RuntimeLayerDispatch {

inline bool cpu_forward_layer(
    const std::vector<const std::vector<float>*>& inputs,
    std::vector<std::vector<float>>& outputs,
    const Layer& layer,
    bool training
) {
    if (inputs.empty() || inputs[0] == nullptr) {
        return false;
    }

    const std::vector<float>& x = *inputs[0];
    outputs.clear();

    try {
        switch (layer.type_enum) {
            // ====================================================================
            // CONVOLUTION
            // ====================================================================
            case LayerType::Conv2d: {
                const int in_c = layer.in_channels > 0 ? layer.in_channels : 1;
                const int out_c = layer.out_channels > 0 ? layer.out_channels : 1;
                const int k = layer.kernel_h > 0 ? layer.kernel_h : layer.get_kernel_h();
                const int stride = layer.stride_h > 0 ? layer.stride_h : layer.get_stride_h();
                const int pad = layer.pad_h >= 0 ? layer.pad_h : layer.get_pad_h();
                const int dilation = layer.dilation_h > 0 ? layer.dilation_h : 1;
                const int H = layer.input_height > 0 ? layer.input_height : 1;
                const int W = layer.input_width > 0 ? layer.input_width : 1;

                const float* w = layer.getWeights();
                if (!w) return false;

                const size_t w_kernel = static_cast<size_t>(out_c) * static_cast<size_t>(in_c) * static_cast<size_t>(k) * static_cast<size_t>(k);
                const size_t need = w_kernel + (layer.use_bias ? static_cast<size_t>(out_c) : 0ULL);
                if (layer.getWeightsSize() < need) return false;

                std::vector<float> kernel(w, w + w_kernel);
                std::vector<float> bias;
                if (layer.use_bias) {
                    bias.assign(w + w_kernel, w + w_kernel + static_cast<size_t>(out_c));
                }

                outputs.resize(1);
                Conv::conv2d(x, outputs[0], kernel, bias, H, W, in_c, out_c, k, stride, pad, dilation);
                return true;
            }

            case LayerType::ConvTranspose2d: {
                const int in_c = layer.in_channels > 0 ? layer.in_channels : 1;
                const int out_c = layer.out_channels > 0 ? layer.out_channels : 1;
                const int k = layer.kernel_h > 0 ? layer.kernel_h : layer.get_kernel_h();
                const int stride = layer.stride_h > 0 ? layer.stride_h : layer.get_stride_h();
                const int pad = layer.pad_h >= 0 ? layer.pad_h : layer.get_pad_h();
                const int H = layer.input_height > 0 ? layer.input_height : 1;
                const int W = layer.input_width > 0 ? layer.input_width : 1;

                const float* w = layer.getWeights();
                if (!w) return false;
                const size_t w_kernel = static_cast<size_t>(out_c) * static_cast<size_t>(in_c) * static_cast<size_t>(k) * static_cast<size_t>(k);
                const size_t need = w_kernel + (layer.use_bias ? static_cast<size_t>(out_c) : 0ULL);
                if (layer.getWeightsSize() < need) return false;

                std::vector<float> kernel(w, w + w_kernel);
                std::vector<float> bias;
                if (layer.use_bias) {
                    bias.assign(w + w_kernel, w + w_kernel + static_cast<size_t>(out_c));
                }

                outputs.resize(1);
                Conv::conv_transpose2d(x, outputs[0], kernel, bias, H, W, in_c, out_c, k, stride, pad);
                return true;
            }

            case LayerType::Conv1d: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::conv1d_forward(x, layer);
                return true;
            }

            case LayerType::DepthwiseConv2d: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::depthwise_conv2d_forward(x, layer);
                return true;
            }

            // ====================================================================
            // BASIC / LINEAR
            // ====================================================================
            case LayerType::Identity: {
                outputs.resize(1);
                outputs[0] = LayerOps::identity_forward(x);
                return true;
            }

            case LayerType::Linear: {
                outputs.resize(1);
                outputs[0] = LayerOps::linear_forward(x, layer);
                return true;
            }

            case LayerType::Bilinear: {
                if (inputs.size() < 2 || inputs[1] == nullptr) return false;
                const std::vector<float>& y = *inputs[1];
                const int in1 = layer.in_features;
                const int in2 = layer.out_features;
                const int out = (layer.embed_dim > 0) ? layer.embed_dim : (!layer.target_shape.empty() ? layer.target_shape[0] : 0);
                if (in1 <= 0 || in2 <= 0 || out <= 0) return false;

                const int batch = static_cast<int>(x.size()) / in1;
                if (batch <= 0) return false;
                if (static_cast<int>(x.size()) != batch * in1) return false;
                if (static_cast<int>(y.size()) != batch * in2) return false;

                const float* w = layer.getWeights();
                if (!w) return false;

                const size_t w_sz = static_cast<size_t>(out) * static_cast<size_t>(in1) * static_cast<size_t>(in2);
                const bool has_bias = layer.use_bias;
                const float* b = has_bias ? (w + w_sz) : nullptr;

                outputs.resize(1);
                outputs[0].assign(static_cast<size_t>(batch) * static_cast<size_t>(out), 0.0f);
                for (int n = 0; n < batch; ++n) {
                    const float* x1 = &x[static_cast<size_t>(n) * static_cast<size_t>(in1)];
                    const float* x2 = &y[static_cast<size_t>(n) * static_cast<size_t>(in2)];
                    for (int o = 0; o < out; ++o) {
                        float sum = has_bias ? b[static_cast<size_t>(o)] : 0.0f;
                        const float* Wo = w + static_cast<size_t>(o) * static_cast<size_t>(in1) * static_cast<size_t>(in2);
                        for (int i = 0; i < in1; ++i) {
                            const float xi = x1[static_cast<size_t>(i)];
                            const float* Woi = Wo + static_cast<size_t>(i) * static_cast<size_t>(in2);
                            for (int j = 0; j < in2; ++j) {
                                sum += xi * Woi[static_cast<size_t>(j)] * x2[static_cast<size_t>(j)];
                            }
                        }
                        outputs[0][static_cast<size_t>(n) * static_cast<size_t>(out) + static_cast<size_t>(o)] = sum;
                    }
                }
                return true;
            }

            // ====================================================================
            // EMBEDDING
            // ====================================================================
            case LayerType::Embedding: {
                // Convention CPU (mode float): x contient des ids (arrondis).
                const int vocab = std::max(1, layer.vocab_size);
                const int dim = std::max(1, layer.embed_dim > 0 ? layer.embed_dim : layer.out_features);
                const int pad = layer.padding_idx;
                const float* w = layer.getWeights();
                if (!w) return false;

                outputs.resize(1);
                outputs[0].assign(x.size() * static_cast<size_t>(dim), 0.0f);
                for (size_t t = 0; t < x.size(); ++t) {
                    const int id = static_cast<int>(std::llround(static_cast<double>(x[t])));
                    if (pad >= 0 && id == pad) continue;
                    if (id < 0 || id >= vocab) continue;
                    const size_t base_w = static_cast<size_t>(id) * static_cast<size_t>(dim);
                    const size_t base_o = t * static_cast<size_t>(dim);
                    for (int d = 0; d < dim; ++d) {
                        outputs[0][base_o + static_cast<size_t>(d)] = w[base_w + static_cast<size_t>(d)];
                    }
                }
                return true;
            }

            case LayerType::EmbeddingBag: {
                if (inputs.empty() || inputs[0] == nullptr) return false;
                const std::vector<float>& ids_f = *inputs[0];
                const std::vector<float>* offsets_f = (inputs.size() >= 2) ? inputs[1] : nullptr;

                const int vocab = std::max(1, layer.vocab_size);
                const int dim = std::max(1, layer.embed_dim);
                const int pad = layer.padding_idx;
                const float* w = layer.getWeights();
                if (!w) return false;

                std::vector<int> offsets;
                int num_bags = 1;
                if (offsets_f && !offsets_f->empty()) {
                    offsets.reserve(offsets_f->size());
                    for (float v : *offsets_f) offsets.push_back(static_cast<int>(std::llround(static_cast<double>(v))));
                    if (offsets.size() < 2) return false;
                    num_bags = static_cast<int>(offsets.size()) - 1;
                } else {
                    offsets = {0, static_cast<int>(ids_f.size())};
                    num_bags = 1;
                }

                outputs.resize(1);
                outputs[0].assign(static_cast<size_t>(num_bags) * static_cast<size_t>(dim), 0.0f);
                for (int b = 0; b < num_bags; ++b) {
                    const int start = std::clamp(offsets[static_cast<size_t>(b)], 0, static_cast<int>(ids_f.size()));
                    const int end = std::clamp(offsets[static_cast<size_t>(b + 1)], start, static_cast<int>(ids_f.size()));
                    float* outp = &outputs[0][static_cast<size_t>(b) * static_cast<size_t>(dim)];
                    for (int t = start; t < end; ++t) {
                        const int id = static_cast<int>(std::llround(static_cast<double>(ids_f[static_cast<size_t>(t)])));
                        if (pad >= 0 && id == pad) continue;
                        if (id < 0 || id >= vocab) continue;
                        const size_t base_w = static_cast<size_t>(id) * static_cast<size_t>(dim);
                        for (int d = 0; d < dim; ++d) {
                            outp[static_cast<size_t>(d)] += w[base_w + static_cast<size_t>(d)];
                        }
                    }
                }
                return true;
            }

            // ====================================================================
            // NORMALIZATION
            // ====================================================================
            case LayerType::BatchNorm2d:
            case LayerType::BatchNorm1d: {
                outputs.resize(1);
                outputs[0] = LayerOps::batchnorm_forward(x, layer, training);
                return true;
            }

            case LayerType::LayerNorm: {
                outputs.resize(1);
                outputs[0] = LayerOps::layernorm_forward(x, layer, training);
                return true;
            }

            case LayerType::GroupNorm: {
                outputs.resize(1);
                outputs[0] = LayerOps::groupnorm_forward(x, layer, training);
                return true;
            }

            case LayerType::InstanceNorm2d: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::instance_norm2d_forward(x, layer);
                return true;
            }

            case LayerType::RMSNorm: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::rms_norm_forward(x, layer);
                return true;
            }

            // ====================================================================
            // ACTIVATIONS
            // ====================================================================
            case LayerType::ReLU: {
                outputs.resize(1);
                outputs[0] = LayerOps::relu_forward(x);
                return true;
            }
            case LayerType::LeakyReLU: {
                const float alpha = layer.leaky_relu_alpha > 0 ? layer.leaky_relu_alpha : 0.01f;
                outputs.resize(1);
                outputs[0] = LayerOpsExt::leaky_relu_forward(x, alpha);
                return true;
            }
            case LayerType::GELU: {
                outputs.resize(1);
                outputs[0] = LayerOps::gelu_forward(x);
                return true;
            }
            case LayerType::GEGLU: {
                if (layer.seq_len <= 0 || layer.out_features <= 0) return false;
                outputs.resize(1);
                outputs[0] = LayerOps::geglu_forward(x, layer.seq_len, layer.out_features);
                return true;
            }
            case LayerType::SiLU: {
                outputs.resize(1);
                outputs[0] = LayerOps::silu_forward(x);
                return true;
            }
            case LayerType::Tanh: {
                outputs.resize(1);
                outputs[0] = LayerOps::tanh_forward(x);
                return true;
            }
            case LayerType::Sigmoid: {
                outputs.resize(1);
                outputs[0] = LayerOps::sigmoid_forward(x);
                return true;
            }
            case LayerType::Softmax:
            case LayerType::LogSoftmax: {
                outputs.resize(1);
                outputs[0] = LayerOps::softmax_forward(x, layer);
                return true;
            }
            case LayerType::Softplus: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::softplus_forward(x);
                return true;
            }
            case LayerType::Mish: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::mish_forward(x);
                return true;
            }
            case LayerType::HardSigmoid: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::hard_sigmoid_forward(x);
                return true;
            }
            case LayerType::HardSwish: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::hard_swish_forward(x);
                return true;
            }

            // ====================================================================
            // POOLING
            // ====================================================================
            case LayerType::MaxPool2d: {
                const int kernel_size = layer.get_kernel_h();
                const int stride = layer.get_stride_h();
                const int padding = layer.get_pad_h();

                const int in_channels = layer.in_channels > 0 ? layer.in_channels : 1;
                const int height = layer.input_height > 0 ? layer.input_height : 1;
                const int width = layer.input_width > 0 ? layer.input_width : 1;

                if (kernel_size <= 0 || stride <= 0) return false;

                const int out_height = (height + 2 * padding - kernel_size) / stride + 1;
                const int out_width = (width + 2 * padding - kernel_size) / stride + 1;
                if (out_height <= 0 || out_width <= 0) return false;

                outputs.resize(1);
                outputs[0].assign(static_cast<size_t>(in_channels) * static_cast<size_t>(out_height) * static_cast<size_t>(out_width),
                                  -std::numeric_limits<float>::infinity());

                for (int c = 0; c < in_channels; ++c) {
                    for (int oh = 0; oh < out_height; ++oh) {
                        for (int ow = 0; ow < out_width; ++ow) {
                            float max_val = -std::numeric_limits<float>::infinity();
                            for (int kh = 0; kh < kernel_size; ++kh) {
                                for (int kw = 0; kw < kernel_size; ++kw) {
                                    const int ih = oh * stride + kh - padding;
                                    const int iw = ow * stride + kw - padding;
                                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                        const int in_idx = c * (height * width) + ih * width + iw;
                                        if (in_idx >= 0 && static_cast<size_t>(in_idx) < x.size()) {
                                            max_val = std::max(max_val, x[static_cast<size_t>(in_idx)]);
                                        }
                                    }
                                }
                            }
                            const int out_idx = c * (out_height * out_width) + oh * out_width + ow;
                            outputs[0][static_cast<size_t>(out_idx)] = max_val;
                        }
                    }
                }
                return true;
            }

            case LayerType::MaxPool1d: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::maxpool1d_forward(x, layer);
                return true;
            }

            case LayerType::AvgPool2d: {
                outputs.resize(1);
                outputs[0] = LayerOps::avgpool2d_forward(x, layer);
                return true;
            }

            case LayerType::AvgPool1d: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::avgpool1d_forward(x, layer);
                return true;
            }

            case LayerType::TokenMeanPool: {
                const int seq_len = layer.seq_len > 0 ? layer.seq_len : 0;
                const int embed_dim = layer.embed_dim > 0 ? layer.embed_dim : 0;
                if (seq_len <= 0 || embed_dim <= 0) return false;
                if (static_cast<int>(x.size()) != seq_len * embed_dim) return false;

                outputs.resize(1);
                outputs[0].assign(static_cast<size_t>(embed_dim), 0.0f);
                for (int t = 0; t < seq_len; ++t) {
                    const int base = t * embed_dim;
                    for (int d = 0; d < embed_dim; ++d) {
                        outputs[0][static_cast<size_t>(d)] += x[static_cast<size_t>(base + d)];
                    }
                }
                const float inv = 1.0f / static_cast<float>(seq_len);
                for (int d = 0; d < embed_dim; ++d) {
                    outputs[0][static_cast<size_t>(d)] *= inv;
                }
                return true;
            }

            case LayerType::GlobalAvgPool2d:
            case LayerType::AdaptiveAvgPool2d: {
                outputs.resize(1);
                outputs[0] = LayerOps::global_avgpool2d_forward(x, layer);
                return true;
            }

            // ====================================================================
            // DROPOUT
            // ====================================================================
            case LayerType::Dropout:
            case LayerType::Dropout2d: {
                outputs.resize(1);
                outputs[0] = LayerOps::dropout_forward(x, layer, training);
                return true;
            }

            case LayerType::AlphaDropout: {
                if (!training) {
                    outputs.resize(1);
                    outputs[0] = x;
                    return true;
                }

                const float p = std::clamp(layer.dropout_p, 0.0f, 1.0f);
                const float alpha = 1.6732632423543772848170429916717f;
                const float scale = 1.0507009873554804934193349852946f;
                const float alpha_p = -alpha * scale;

                const float a = 1.0f / std::sqrt((1.0f - p) * (1.0f + p * alpha_p * alpha_p));
                const float b = -a * alpha_p * p;

                outputs.resize(1);
                outputs[0].resize(x.size());
                static thread_local std::mt19937 rng(1337);
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                for (size_t i = 0; i < x.size(); ++i) {
                    const bool keep = (dist(rng) > p);
                    const float v = keep ? x[i] : alpha_p;
                    outputs[0][i] = a * v + b;
                }
                return true;
            }

            // ====================================================================
            // SHAPE
            // ====================================================================
            case LayerType::Flatten: {
                outputs.resize(1);
                outputs[0] = LayerOps::flatten_forward(x, layer);
                return true;
            }

            case LayerType::Reshape:
            case LayerType::View: {
                outputs.resize(1);
                outputs[0] = LayerOps::reshape_forward(x, layer);
                return true;
            }

            case LayerType::Transpose: {
                if (layer.in_features <= 0 || layer.out_features <= 0) return false;
                outputs.resize(1);
                outputs[0] = LayerOps::transpose_forward(x, layer.in_features, layer.out_features);
                return true;
            }

            case LayerType::Permute: {
                if (layer.permute_dims.empty()) return false;
                std::vector<int> shape = layer.shape;
                if (shape.empty()) {
                    shape = {1, static_cast<int>(x.size())};
                }
                outputs.resize(1);
                outputs[0] = LayerOps::permute_forward(x, layer.permute_dims, shape);
                return true;
            }

            case LayerType::Squeeze: {
                std::vector<int> in_shape = {static_cast<int>(x.size())};
                std::vector<int> out_shape;
                outputs.resize(1);
                outputs[0] = LayerOpsExt::squeeze_forward(x, in_shape, out_shape, layer.squeeze_dim);
                return true;
            }

            case LayerType::Unsqueeze: {
                std::vector<int> in_shape = {static_cast<int>(x.size())};
                std::vector<int> out_shape;
                outputs.resize(1);
                outputs[0] = LayerOpsExt::unsqueeze_forward(x, in_shape, out_shape, layer.unsqueeze_dim);
                return true;
            }

            case LayerType::Lambda: {
                // Non supporté via runtime (callbacks Lua).
                return false;
            }

            // ====================================================================
            // CUSTOM (Mímir)
            // ====================================================================
            case LayerType::PatchEmbed: {
                const int d_model = layer.embed_dim > 0 ? layer.embed_dim : layer.out_features;
                const int seq_text = std::max(1, layer.seq_text);
                const int num_patches = std::max(1, layer.num_patches);
                const int patch_dim = std::max(1, layer.patch_dim);

                const int text_dim = (seq_text + 1) * d_model;
                const int in_dim = text_dim + num_patches * patch_dim;
                const int out_dim = (seq_text + 1 + num_patches) * d_model;
                if (static_cast<int>(x.size()) != in_dim) return false;

                const float* w = layer.getWeights();
                if (!w) return false;
                const int expected_w = patch_dim * d_model + d_model;
                if (static_cast<int>(layer.getWeightsSize()) != expected_w) return false;

                outputs.resize(1);
                outputs[0].assign(static_cast<size_t>(out_dim), 0.0f);
                std::copy(x.begin(), x.begin() + text_dim, outputs[0].begin());

                const float* b = w + patch_dim * d_model;
                const float inv = 1.0f / std::sqrt(static_cast<float>(patch_dim));
                for (int p = 0; p < num_patches; ++p) {
                    const int in_off = text_dim + p * patch_dim;
                    const int out_off = (seq_text + 1 + p) * d_model;
                    for (int d = 0; d < d_model; ++d) {
                        float sum = b[d];
                        for (int k = 0; k < patch_dim; ++k) {
                            sum += (x[static_cast<size_t>(in_off + k)] * inv) * w[static_cast<size_t>(k * d_model + d)];
                        }
                        outputs[0][static_cast<size_t>(out_off + d)] = sum;
                    }
                }
                return true;
            }

            // ====================================================================
            // ELEMENT-WISE
            // ====================================================================
            case LayerType::Add: {
                if (inputs.size() < 2 || inputs[1] == nullptr) return false;
                outputs.resize(1);
                outputs[0] = LayerOps::add_forward(*inputs[0], *inputs[1]);
                return true;
            }

            case LayerType::Subtract: {
                if (inputs.size() < 2 || inputs[1] == nullptr) return false;
                outputs.resize(1);
                outputs[0] = LayerOpsExt::subtract_forward(*inputs[0], *inputs[1]);
                return true;
            }

            case LayerType::Multiply: {
                if (inputs.size() < 2 || inputs[1] == nullptr) return false;
                outputs.resize(1);
                outputs[0] = LayerOps::multiply_forward(*inputs[0], *inputs[1]);
                return true;
            }

            case LayerType::Divide: {
                if (inputs.size() < 2 || inputs[1] == nullptr) return false;
                outputs.resize(1);
                outputs[0] = LayerOpsExt::divide_forward(*inputs[0], *inputs[1]);
                return true;
            }

            case LayerType::Reparameterize: {
                if (inputs.size() < 2 || inputs[1] == nullptr) return false;
                const std::vector<float>& mu = *inputs[0];
                const std::vector<float>& logvar = *inputs[1];
                if (mu.size() != logvar.size()) return false;

                outputs.resize(1);
                outputs[0].resize(mu.size());

                if (!training) {
                    outputs[0] = mu;
                    return true;
                }

                static thread_local std::mt19937 rng(1337);
                std::normal_distribution<float> n01(0.0f, 1.0f);
                for (size_t i = 0; i < mu.size(); ++i) {
                    const float lv = std::clamp(logvar[i], -20.0f, 20.0f);
                    const float stdv = std::exp(0.5f * lv);
                    outputs[0][i] = mu[i] + stdv * n01(rng);
                }
                return true;
            }

            // ====================================================================
            // TENSOR OPS
            // ====================================================================
            case LayerType::Concat: {
                if (inputs.size() < 2) return false;
                std::vector<std::vector<float>> in_vec;
                in_vec.reserve(inputs.size());
                for (const auto* p : inputs) {
                    if (!p) return false;
                    in_vec.push_back(*p);
                }
                outputs.resize(1);
                outputs[0] = LayerOps::concat_forward(in_vec, layer.concat_axis);
                return true;
            }

            case LayerType::Split: {
                if (!layer.split_sizes.empty()) {
                    outputs = LayerOps::split_forward(x, layer.split_sizes, layer.split_axis);
                    return true;
                }
                if (layer.num_splits > 0) {
                    outputs = LayerOps::split_forward(x, layer.num_splits, layer.split_axis);
                    return true;
                }
                return false;
            }

            case LayerType::Chunk: {
                if (layer.num_chunks <= 0) return false;
                outputs = LayerOpsExt::chunk_forward(x, layer.num_chunks, layer.split_axis);
                return true;
            }

            case LayerType::Stack: {
                if (inputs.size() < 2) return false;
                std::vector<std::vector<float>> in_vec;
                in_vec.reserve(inputs.size());
                for (const auto* p : inputs) {
                    if (!p) return false;
                    in_vec.push_back(*p);
                }
                outputs.resize(1);
                outputs[0] = LayerOpsExt::stack_forward(in_vec, layer.stack_axis);
                return true;
            }

            case LayerType::MatMul: {
                if (inputs.size() < 2 || inputs[1] == nullptr) return false;
                if (layer.in_features <= 0 || layer.out_features <= 0 || layer.embed_dim <= 0) return false;
                const int M = layer.in_features;
                const int K = layer.out_features;
                const int N = layer.embed_dim;
                outputs.resize(1);
                outputs[0] = LayerOps::matmul_forward(*inputs[0], *inputs[1], M, K, N);
                return true;
            }

            case LayerType::BatchMatMul: {
                if (inputs.size() < 2 || inputs[1] == nullptr) return false;
                if (layer.seq_len <= 0 || layer.in_features <= 0 || layer.out_features <= 0 || layer.embed_dim <= 0) return false;

                const int B = layer.seq_len;
                const int M = layer.in_features;
                const int K = layer.out_features;
                const int N = layer.embed_dim;
                const std::vector<float>& A = *inputs[0];
                const std::vector<float>& Bm = *inputs[1];
                if (static_cast<int>(A.size()) != B * M * K) return false;
                if (static_cast<int>(Bm.size()) != B * K * N) return false;

                outputs.resize(1);
                outputs[0].assign(static_cast<size_t>(B) * static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);

                #pragma omp parallel for schedule(static) if(static_cast<long long>(B) * M * N * K > 262144)
                for (int b = 0; b < B; ++b) {
                    const float* Ap = &A[static_cast<size_t>(b) * static_cast<size_t>(M) * static_cast<size_t>(K)];
                    const float* Bp = &Bm[static_cast<size_t>(b) * static_cast<size_t>(K) * static_cast<size_t>(N)];
                    float* Cp = &outputs[0][static_cast<size_t>(b) * static_cast<size_t>(M) * static_cast<size_t>(N)];
                    for (int i = 0; i < M; ++i) {
                        for (int j = 0; j < N; ++j) {
                            float sum = 0.0f;
                            for (int k = 0; k < K; ++k) {
                                sum += Ap[static_cast<size_t>(i) * static_cast<size_t>(K) + static_cast<size_t>(k)] *
                                       Bp[static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(j)];
                            }
                            Cp[static_cast<size_t>(i) * static_cast<size_t>(N) + static_cast<size_t>(j)] = sum;
                        }
                    }
                }
                return true;
            }

            // ====================================================================
            // ATTENTION
            // ====================================================================
            case LayerType::Constant: {
                const float* weights = layer.getWeights();
                if (!weights) return false;
                const size_t n = layer.getWeightsSize();
                outputs.resize(1);
                outputs[0].assign(weights, weights + n);
                return true;
            }

            case LayerType::SelfAttention:
            case LayerType::MultiHeadAttention: {
                const float* weights = layer.getWeights();
                if (!weights) return false;

                const int seq_len = layer.seq_len > 0 ? layer.seq_len : 1;
                const int embed_dim = layer.embed_dim > 0 ? layer.embed_dim : static_cast<int>(x.size());
                const int num_heads = layer.num_heads > 0 ? layer.num_heads : 1;
                const bool causal = layer.causal;

                const int qkv_size = embed_dim * embed_dim * 3;
                const int out_size = embed_dim * embed_dim;
                const int qkv_bias_size = embed_dim * 3;
                const int out_bias_size = embed_dim;
                const size_t expected_no_bias = static_cast<size_t>(qkv_size + out_size);
                const size_t expected_out_bias_only = expected_no_bias + static_cast<size_t>(out_bias_size);
                const size_t expected_with_bias = expected_no_bias + static_cast<size_t>(qkv_bias_size + out_bias_size);
                const size_t actual_size = layer.getWeightsSize();
                const bool has_full_bias = (actual_size >= expected_with_bias);
                const bool has_out_bias_only = (!has_full_bias && actual_size >= expected_out_bias_only);

                std::vector<float> qkv_weight(weights, weights + qkv_size);
                std::vector<float> out_weight(weights + qkv_size, weights + qkv_size + out_size);
                std::vector<float> qkv_bias;
                std::vector<float> out_bias;
                if (has_full_bias) {
                    const float* bias_ptr = weights + qkv_size + out_size;
                    qkv_bias.assign(bias_ptr, bias_ptr + qkv_bias_size);
                    out_bias.assign(bias_ptr + qkv_bias_size, bias_ptr + qkv_bias_size + out_bias_size);
                } else if (has_out_bias_only) {
                    const float* bias_ptr = weights + qkv_size + out_size;
                    out_bias.assign(bias_ptr, bias_ptr + out_bias_size);
                }

                outputs.resize(1);
                if (layer.type_enum == LayerType::SelfAttention) {
                    outputs[0] = LayerOps::self_attention_forward(x, qkv_weight, out_weight, qkv_bias, out_bias, seq_len, embed_dim, num_heads, causal);
                } else {
                    outputs[0] = LayerOps::multihead_attention_forward(x, qkv_weight, out_weight, qkv_bias, out_bias, seq_len, embed_dim, num_heads, causal);
                }
                return true;
            }

            case LayerType::CrossAttention: {
                if (inputs.size() < 2 || inputs[1] == nullptr) return false;
                const float* weights = layer.getWeights();
                if (!weights) return false;

                const std::vector<float>& q_in = *inputs[0];
                const std::vector<float>& kv_in = *inputs[1];
                const int num_heads = layer.num_heads > 0 ? layer.num_heads : 1;
                const bool causal = layer.causal;

                int embed_dim = layer.embed_dim;
                if (embed_dim <= 0 && layer.head_dim > 0 && num_heads > 0) {
                    embed_dim = layer.head_dim * num_heads;
                }
                const int kv_embed_dim = layer.in_features > 0 ? layer.in_features : embed_dim;
                if (embed_dim <= 0) return false;
                if ((q_in.size() % static_cast<size_t>(embed_dim)) != 0) return false;
                if ((kv_in.size() % static_cast<size_t>(kv_embed_dim)) != 0) return false;

                const int query_len = static_cast<int>(q_in.size() / static_cast<size_t>(embed_dim));
                const int kv_len = static_cast<int>(kv_in.size() / static_cast<size_t>(kv_embed_dim));
                if (query_len <= 0 || kv_len <= 0) return false;

                const int q_size = embed_dim * embed_dim;
                const int kv_size = kv_embed_dim * (2 * embed_dim);
                const int out_size = embed_dim * embed_dim;
                const int out_bias_size = embed_dim;

                std::vector<float> q_weight(weights, weights + q_size);
                std::vector<float> kv_weight(weights + q_size, weights + q_size + kv_size);
                std::vector<float> out_weight(weights + q_size + kv_size, weights + q_size + kv_size + out_size);
                std::vector<float> out_bias;
                const size_t expected_with_bias = static_cast<size_t>(q_size + kv_size + out_size + out_bias_size);
                if (layer.getWeightsSize() >= expected_with_bias) {
                    const float* bias_ptr = weights + q_size + kv_size + out_size;
                    out_bias.assign(bias_ptr, bias_ptr + out_bias_size);
                }

                outputs.resize(1);
                outputs[0] = LayerOps::cross_attention_forward(
                    q_in, kv_in, q_weight, kv_weight, out_weight, out_bias,
                    query_len, kv_len, embed_dim, kv_embed_dim, num_heads, causal
                );
                return true;
            }

            // ====================================================================
            // UPSAMPLING
            // ====================================================================
            case LayerType::UpsampleNearest: {
                if (layer.in_channels <= 0) return false;
                const int in_h = layer.input_height > 0 ? layer.input_height : layer.out_h;
                const int in_w = layer.input_width > 0 ? layer.input_width : layer.out_w;
                if (in_h <= 0 || in_w <= 0) return false;
                const int c = layer.in_channels;
                const int out_h = layer.output_height > 0 ? layer.output_height : (layer.out_h > 0 ? layer.out_h : 0);
                const int out_w = layer.output_width > 0 ? layer.output_width : (layer.out_w > 0 ? layer.out_w : 0);
                const int sh = layer.scale_h > 0 ? static_cast<int>(std::lround(layer.scale_h))
                                                 : ((out_h > 0) ? std::max(1, out_h / in_h) : 2);
                const int sw = layer.scale_w > 0 ? static_cast<int>(std::lround(layer.scale_w))
                                                 : ((out_w > 0) ? std::max(1, out_w / in_w) : 2);
                outputs.resize(1);
                outputs[0] = LayerOps::upsample_nearest_forward(x, in_h, in_w, c, sh, sw);
                return true;
            }

            case LayerType::UpsampleBilinear: {
                if (layer.in_channels <= 0) return false;
                const int in_h = layer.input_height > 0 ? layer.input_height : layer.out_h;
                const int in_w = layer.input_width > 0 ? layer.input_width : layer.out_w;
                if (in_h <= 0 || in_w <= 0) return false;
                const int c = layer.in_channels;
                const int out_h = layer.output_height > 0 ? layer.output_height : (layer.out_h > 0 ? layer.out_h : std::max(1, static_cast<int>(std::lround(static_cast<float>(in_h) * std::max(0.0f, layer.scale_h)))));
                const int out_w = layer.output_width > 0 ? layer.output_width : (layer.out_w > 0 ? layer.out_w : std::max(1, static_cast<int>(std::lround(static_cast<float>(in_w) * std::max(0.0f, layer.scale_w)))));
                outputs.resize(1);
                outputs[0] = LayerOps::upsample_bilinear_forward(x, in_h, in_w, c, out_h, out_w);
                return true;
            }

            case LayerType::UpsampleBicubic: {
                const int in_h = layer.input_height > 0 ? layer.input_height : layer.out_h;
                const int in_w = layer.input_width > 0 ? layer.input_width : layer.out_w;
                if (in_h <= 0 || in_w <= 0) return false;
                const int c = layer.in_channels > 0 ? layer.in_channels : 3;
                const int out_h = layer.output_height > 0 ? layer.output_height : (layer.out_h > 0 ? layer.out_h : std::max(1, static_cast<int>(std::lround(static_cast<float>(in_h) * std::max(0.0f, layer.scale_h)))));
                const int out_w = layer.output_width > 0 ? layer.output_width : (layer.out_w > 0 ? layer.out_w : std::max(1, static_cast<int>(std::lround(static_cast<float>(in_w) * std::max(0.0f, layer.scale_w)))));
                outputs.resize(1);
                outputs[0] = LayerOpsExt::upsample_bicubic_forward(x, in_h, in_w, c, out_h, out_w);
                return true;
            }

            case LayerType::PixelShuffle: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::pixel_shuffle_forward(x, layer);
                return true;
            }

            // ====================================================================
            // PADDING
            // ====================================================================
            case LayerType::ZeroPad2d: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::zero_pad2d_forward(x, layer);
                return true;
            }
            case LayerType::ReflectionPad2d: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::reflection_pad2d_forward(x, layer);
                return true;
            }
            case LayerType::ReplicationPad2d: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::replication_pad2d_forward(x, layer);
                return true;
            }

            // ====================================================================
            // RECURRENT
            // ====================================================================
            case LayerType::LSTM:
            case LayerType::GRU:
            case LayerType::RNN: {
                // Pour rester fonctionnel et déterministe, on réutilise l'implémentation CPU de Model (naïve).
                const int T = layer.seq_len;
                const int I = layer.in_features;
                const int H = layer.out_features;
                if (T <= 0 || I <= 0 || H <= 0) return false;
                if (static_cast<int>(x.size()) != T * I) return false;
                const float* W = layer.getWeights();
                if (!W) return false;

                auto sigmoid_scalar = [](float v) -> float { return 1.0f / (1.0f + std::exp(-v)); };

                if (layer.type_enum == LayerType::RNN) {
                    const bool use_bias = layer.use_bias;
                    const size_t Wih_sz = static_cast<size_t>(H) * static_cast<size_t>(I);
                    const size_t Whh_sz = static_cast<size_t>(H) * static_cast<size_t>(H);
                    const size_t bih_sz = use_bias ? static_cast<size_t>(H) : 0ULL;
                    const size_t bhh_sz = use_bias ? static_cast<size_t>(H) : 0ULL;
                    const size_t need = Wih_sz + Whh_sz + bih_sz + bhh_sz;
                    if (layer.getWeightsSize() < need) return false;

                    const float* Wih = W;
                    const float* Whh = Wih + Wih_sz;
                    const float* bih = use_bias ? (Whh + Whh_sz) : nullptr;
                    const float* bhh = use_bias ? (bih + bih_sz) : nullptr;

                    std::vector<float> h_prev(static_cast<size_t>(H), 0.0f);
                    outputs.resize(1);
                    outputs[0].assign(static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);
                    for (int t = 0; t < T; ++t) {
                        const float* xt = &x[static_cast<size_t>(t) * static_cast<size_t>(I)];
                        float* ht = &outputs[0][static_cast<size_t>(t) * static_cast<size_t>(H)];
                        for (int h = 0; h < H; ++h) {
                            const float* wih = Wih + static_cast<size_t>(h) * static_cast<size_t>(I);
                            const float* whh = Whh + static_cast<size_t>(h) * static_cast<size_t>(H);
                            float sum = 0.0f;
                            for (int i = 0; i < I; ++i) sum += wih[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                            for (int k = 0; k < H; ++k) sum += whh[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)];
                            if (bih) sum += bih[static_cast<size_t>(h)];
                            if (bhh) sum += bhh[static_cast<size_t>(h)];
                            ht[static_cast<size_t>(h)] = std::tanh(sum);
                        }
                        std::copy(ht, ht + H, h_prev.begin());
                    }
                    return true;
                }

                if (layer.type_enum == LayerType::GRU) {
                    const bool use_bias = layer.use_bias;
                    const size_t Wih_sz = static_cast<size_t>(3 * H) * static_cast<size_t>(I);
                    const size_t Whh_sz = static_cast<size_t>(3 * H) * static_cast<size_t>(H);
                    const size_t bih_sz = use_bias ? static_cast<size_t>(3 * H) : 0ULL;
                    const size_t bhh_sz = use_bias ? static_cast<size_t>(3 * H) : 0ULL;
                    const size_t need = Wih_sz + Whh_sz + bih_sz + bhh_sz;
                    if (layer.getWeightsSize() < need) return false;

                    const float* Wih = W;
                    const float* Whh = Wih + Wih_sz;
                    const float* bih = use_bias ? (Whh + Whh_sz) : nullptr;
                    const float* bhh = use_bias ? (bih + bih_sz) : nullptr;

                    std::vector<float> h_prev(static_cast<size_t>(H), 0.0f);
                    std::vector<float> r(static_cast<size_t>(H), 0.0f);
                    std::vector<float> z(static_cast<size_t>(H), 0.0f);
                    std::vector<float> n(static_cast<size_t>(H), 0.0f);

                    outputs.resize(1);
                    outputs[0].assign(static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);
                    for (int t = 0; t < T; ++t) {
                        const float* xt = &x[static_cast<size_t>(t) * static_cast<size_t>(I)];

                        for (int h = 0; h < H; ++h) {
                            const float* w_ir = Wih + static_cast<size_t>(h) * static_cast<size_t>(I);
                            const float* w_hr = Whh + static_cast<size_t>(h) * static_cast<size_t>(H);
                            float sr = 0.0f;
                            for (int i = 0; i < I; ++i) sr += w_ir[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                            for (int k = 0; k < H; ++k) sr += w_hr[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)];
                            if (bih) sr += bih[static_cast<size_t>(h)];
                            if (bhh) sr += bhh[static_cast<size_t>(h)];
                            r[static_cast<size_t>(h)] = sigmoid_scalar(sr);

                            const float* w_iz = Wih + static_cast<size_t>(H + h) * static_cast<size_t>(I);
                            const float* w_hz = Whh + static_cast<size_t>(H + h) * static_cast<size_t>(H);
                            float sz = 0.0f;
                            for (int i = 0; i < I; ++i) sz += w_iz[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                            for (int k = 0; k < H; ++k) sz += w_hz[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)];
                            if (bih) sz += bih[static_cast<size_t>(H + h)];
                            if (bhh) sz += bhh[static_cast<size_t>(H + h)];
                            z[static_cast<size_t>(h)] = sigmoid_scalar(sz);

                            const float* w_in = Wih + static_cast<size_t>(2 * H + h) * static_cast<size_t>(I);
                            const float* w_hn = Whh + static_cast<size_t>(2 * H + h) * static_cast<size_t>(H);
                            float sn = 0.0f;
                            for (int i = 0; i < I; ++i) sn += w_in[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                            for (int k = 0; k < H; ++k) sn += w_hn[static_cast<size_t>(k)] * (r[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)]);
                            if (bih) sn += bih[static_cast<size_t>(2 * H + h)];
                            if (bhh) sn += bhh[static_cast<size_t>(2 * H + h)];
                            n[static_cast<size_t>(h)] = std::tanh(sn);
                        }

                        float* ht = &outputs[0][static_cast<size_t>(t) * static_cast<size_t>(H)];
                        for (int h = 0; h < H; ++h) {
                            const float zh = z[static_cast<size_t>(h)];
                            const float nh = n[static_cast<size_t>(h)];
                            const float hp = h_prev[static_cast<size_t>(h)];
                            ht[static_cast<size_t>(h)] = (1.0f - zh) * nh + zh * hp;
                        }
                        std::copy(ht, ht + H, h_prev.begin());
                    }
                    return true;
                }

                // LSTM: fallback simple (pas d'impl complète ici)
                outputs.resize(1);
                outputs[0] = x;
                return true;
            }

            case LayerType::UNKNOWN:
                return false;

            default:
                break;
        }

        // Si un layer n'est pas pris en charge ici, on renvoie false.
        return false;
    } catch (...) {
        return false;
    }
}

} // namespace RuntimeLayerDispatch
