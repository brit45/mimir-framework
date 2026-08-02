#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Layers.hpp"

namespace Mimir::Planning {

// --------------------------
// Tensor lifetime analysis
// --------------------------

struct TensorLifetime {
    // Indices of layers in the static schedule.
    // first_def: first layer that defines the value
    // last_use: last layer that reads the value
    int first_def = -1;
    int last_use = -1;

    // Metadata best-effort.
    std::string dtype = "float32";
    std::vector<int> shape;
};

// Very lightweight lifetime analysis for the current Model execution semantics:
// - schedule is the layer order (static scheduling)
// - routing is by named tensors (layer.inputs / layer.output)
// - multiple writes to the same name are treated as overwriting the same slot
//   (so we track lifetime per tensor name, not SSA versions)
// This is enough to drive safe "reuse once dead" decisions for scratch buffers.
inline std::unordered_map<std::string, TensorLifetime>
analyze_tensor_lifetimes(const std::vector<Layer>& layers) {
    std::unordered_map<std::string, TensorLifetime> lifetimes;

    auto get_inputs = [](const Layer& l) -> const std::vector<std::string>& {
        static const std::vector<std::string> kDefaultX = {"x"};
        return l.inputs.empty() ? kDefaultX : l.inputs;
    };

    for (int i = 0; i < static_cast<int>(layers.size()); ++i) {
        const Layer& layer = layers[static_cast<size_t>(i)];

        // Record uses
        for (const auto& in : get_inputs(layer)) {
            auto& lt = lifetimes[in];
            if (lt.first_def < 0) lt.first_def = 0; // assume defined at graph input
            lt.last_use = std::max(lt.last_use, i);
        }

        // Record def
        const std::string out = layer.output.empty() ? "x" : layer.output;
        auto& lt_out = lifetimes[out];
        if (lt_out.first_def < 0) lt_out.first_def = i;
        // If overwritten, lifetime restarts; keep first_def as earliest seen for this slot.
        lt_out.last_use = std::max(lt_out.last_use, i);
    }

    return lifetimes;
}

// --------------------------
// Fusion + execution plan
// --------------------------

enum class FusionKind : uint8_t {
    NONE = 0,
    CONV2D_RELU = 1,
    GENERIC_ACTIVATION = 2,
    GENERIC_SPLIT = 3,
    GENERIC_CHUNK = 4,
    GENERIC_ACTIVATION_SPLIT = 5,
    GENERIC_ACTIVATION_CHUNK = 6,
    GENERIC_UNARY_SHAPE = 7,
};

struct PlannedOp {
    int layer_index = -1;
    FusionKind fusion = FusionKind::NONE;
};

struct ExecutionPlan {
    // Static scheduling: fixed op order
    std::vector<PlannedOp> ops;

    // Skip fused consumer layers at runtime.
    std::vector<uint8_t> skip_layer;

    // Per-layer fusion flags for cheap hot-path checks
    std::vector<uint8_t> fuse_relu_for_conv2d;

    // Generic inference-time fusions.
    std::vector<int> fuse_activation_consumer;
    std::vector<int> fuse_unary_consumer;
    std::vector<int> fuse_split_consumer;
    std::vector<uint8_t> fuse_split_kind; // 0=none, 1=Split, 2=Chunk

    // Generic chained fusion edges (producer -> fused consumer).
    // Allows fusing repeated adjacent layers (e.g. op->act->act->shape->shape).
    std::vector<int> fuse_chain_next;

    bool empty() const { return ops.empty(); }
};

inline const std::vector<std::string>& planner_inputs_for(const Layer& layer) {
    static const std::vector<std::string> kDefaultX = {"x"};
    return (layer.inputs.empty() && layer.type_enum != LayerType::Constant) ? kDefaultX : layer.inputs;
}

inline std::string planner_output_name_for(const Layer& layer) {
    return layer.output.empty() ? "x" : layer.output;
}

inline bool is_fusible_activation_layer(const Layer& layer) {
    switch (layer.type_enum) {
        case LayerType::ReLU:
        case LayerType::LeakyReLU:
        case LayerType::GELU:
        case LayerType::GEGLU:
        case LayerType::SiLU:
        case LayerType::Tanh:
        case LayerType::Sigmoid:
        case LayerType::Softmax:
        case LayerType::LogSoftmax:
        case LayerType::Softplus:
        case LayerType::Mish:
        case LayerType::HardSigmoid:
        case LayerType::HardSwish:
            return true;
        default:
            return false;
    }
}

inline bool is_fusible_split_layer(const Layer& layer) {
    return layer.type_enum == LayerType::Split || layer.type_enum == LayerType::Chunk;
}

inline bool is_fusible_unary_shape_layer(const Layer& layer) {
    switch (layer.type_enum) {
        case LayerType::Identity:
        case LayerType::Flatten:
        case LayerType::Reshape:
        case LayerType::View:
        case LayerType::Transpose:
        case LayerType::Permute:
        case LayerType::Squeeze:
        case LayerType::Unsqueeze:
            return true;
        default:
            return false;
    }
}

inline ExecutionPlan build_execution_plan_static(const std::vector<Layer>& layers, bool training, bool allow_training_fusion = false) {
    ExecutionPlan plan;
    plan.ops.reserve(layers.size());
    plan.skip_layer.assign(layers.size(), 0);
    plan.fuse_relu_for_conv2d.assign(layers.size(), 0);
    plan.fuse_activation_consumer.assign(layers.size(), -1);
    plan.fuse_unary_consumer.assign(layers.size(), -1);
    plan.fuse_split_consumer.assign(layers.size(), -1);
    plan.fuse_split_kind.assign(layers.size(), 0);
    plan.fuse_chain_next.assign(layers.size(), -1);

    std::unordered_map<std::string, int> tensor_use_count;
    for (const auto& layer : layers) {
        for (const auto& in : planner_inputs_for(layer)) {
            ++tensor_use_count[in];
        }
    }

    auto consumes_single_tensor = [](const Layer& consumer, const std::string& expected_input) -> bool {
        const auto& inputs = planner_inputs_for(consumer);
        return inputs.size() == 1 && inputs[0] == expected_input;
    };

    for (size_t i = 0; i < layers.size(); ++i) {
        const Layer& layer = layers[i];

        PlannedOp op;
        op.layer_index = static_cast<int>(i);
        op.fusion = FusionKind::NONE;

        if (plan.skip_layer[i] != 0) {
            plan.ops.push_back(op);
            continue;
        }

        const bool can_fuse_conv_relu = (!training) || allow_training_fusion;
        const bool can_fuse_generic = !training;

        // Conservative by default: in training, fusions are disabled unless explicitly enabled.
        if (can_fuse_conv_relu && layer.type_enum == LayerType::Conv2d && layer.activation == ActivationType::RELU) {
            op.fusion = FusionKind::CONV2D_RELU;
            plan.fuse_relu_for_conv2d[i] = 1;
        }

        if (can_fuse_generic) {
            const std::string producer_out = planner_output_name_for(layer);
            const bool producer_has_single_consumer = tensor_use_count[producer_out] == 1;

            auto maybe_mark_split = [&](size_t producer_idx, size_t consumer_idx, bool after_activation) {
                const Layer& split_layer = layers[consumer_idx];
                plan.fuse_split_consumer[producer_idx] = static_cast<int>(consumer_idx);
                plan.fuse_split_kind[producer_idx] = (split_layer.type_enum == LayerType::Split) ? 1 : 2;
                plan.skip_layer[consumer_idx] = 1;
                plan.fuse_chain_next[producer_idx] = static_cast<int>(consumer_idx);
                if (after_activation) {
                    op.fusion = (split_layer.type_enum == LayerType::Split)
                        ? FusionKind::GENERIC_ACTIVATION_SPLIT
                        : FusionKind::GENERIC_ACTIVATION_CHUNK;
                } else {
                    op.fusion = (split_layer.type_enum == LayerType::Split)
                        ? FusionKind::GENERIC_SPLIT
                        : FusionKind::GENERIC_CHUNK;
                }
            };

            if (producer_has_single_consumer && (i + 1) < layers.size()) {
                int chain_cursor = static_cast<int>(i);
                std::string chain_output = producer_out;
                bool first_in_chain = true;

                while (true) {
                    const size_t next_idx = static_cast<size_t>(chain_cursor + 1);
                    if (next_idx >= layers.size()) break;

                    const Layer& next = layers[next_idx];
                    if (!consumes_single_tensor(next, chain_output)) break;
                    if (tensor_use_count[chain_output] != 1) break;

                    if (is_fusible_activation_layer(next)) {
                        plan.skip_layer[next_idx] = 1;
                        plan.fuse_chain_next[static_cast<size_t>(chain_cursor)] = static_cast<int>(next_idx);
                        if (first_in_chain) {
                            plan.fuse_activation_consumer[i] = static_cast<int>(next_idx);
                            if (op.fusion == FusionKind::NONE) {
                                op.fusion = FusionKind::GENERIC_ACTIVATION;
                            }
                        }
                        chain_cursor = static_cast<int>(next_idx);
                        chain_output = planner_output_name_for(next);
                        first_in_chain = false;
                        continue;
                    }

                    if (is_fusible_unary_shape_layer(next)) {
                        plan.skip_layer[next_idx] = 1;
                        plan.fuse_chain_next[static_cast<size_t>(chain_cursor)] = static_cast<int>(next_idx);
                        if (first_in_chain) {
                            plan.fuse_unary_consumer[i] = static_cast<int>(next_idx);
                            if (op.fusion == FusionKind::NONE) {
                                op.fusion = FusionKind::GENERIC_UNARY_SHAPE;
                            }
                        }
                        chain_cursor = static_cast<int>(next_idx);
                        chain_output = planner_output_name_for(next);
                        first_in_chain = false;
                        continue;
                    }

                    if (is_fusible_split_layer(next)) {
                        const bool after_activation = (op.fusion == FusionKind::GENERIC_ACTIVATION);
                        maybe_mark_split(static_cast<size_t>(chain_cursor), next_idx, after_activation);
                        // Keep the legacy root-producer metadata coherent even
                        // when the split is reached through a fused chain.
                        plan.fuse_split_consumer[i] = static_cast<int>(next_idx);
                        plan.fuse_split_kind[i] = (next.type_enum == LayerType::Split) ? 1 : 2;
                        break;
                    }

                    break;
                }
            }
        }

        plan.ops.push_back(op);
    }

    return plan;
}

// --------------------------
// Memory planner (scratch)
// --------------------------

struct ScratchRequest {
    std::string tag;
    size_t min_bytes = 0;
};

// Best-effort scratch size planning for Conv2d fast path.
// Goal: allow callers to pre-warm a scratch pool and use shared tags.
struct Conv2dScratchPlan {
    size_t wT_bytes = 0;
    size_t xcol_bytes = 0;
    size_t c_bytes = 0;
};

inline Conv2dScratchPlan plan_conv2d_fastpath_scratch(const std::vector<Layer>& layers) {
    Conv2dScratchPlan out;

    // Mirror the fast-path heuristics (tile target ~32MB for Xcol).
    const size_t target_bytes = 32ULL * 1024ULL * 1024ULL;
    const size_t floats_budget = target_bytes / sizeof(float);

    for (const auto& layer : layers) {
        if (layer.type_enum != LayerType::Conv2d) continue;
        const int kernel_size = layer.get_kernel_h();
        const int in_channels = layer.in_channels;
        const int out_channels = layer.out_channels;
        if (kernel_size <= 0 || in_channels <= 0 || out_channels <= 0) continue;

        const int height = (layer.input_height > 0) ? layer.input_height : 64;
        const int width = (layer.input_width > 0) ? layer.input_width : 64;
        const int stride = layer.get_stride_h();
        const int padding = layer.get_pad_h();

        const int out_h = (height + 2 * padding - kernel_size) / std::max(1, stride) + 1;
        const int out_w = (width + 2 * padding - kernel_size) / std::max(1, stride) + 1;
        const int out_spatial = std::max(0, out_h) * std::max(0, out_w);
        const int K = in_channels * kernel_size * kernel_size;
        if (out_spatial <= 0 || K <= 0) continue;

        int tile_m = static_cast<int>(std::max<size_t>(256, std::min<size_t>(8192, floats_budget / static_cast<size_t>(K))));
        if (tile_m > out_spatial) tile_m = out_spatial;

        const size_t w_need = static_cast<size_t>(out_channels) * static_cast<size_t>(K);
        const size_t xcol_need = static_cast<size_t>(tile_m) * static_cast<size_t>(K);
        const size_t c_need = static_cast<size_t>(tile_m) * static_cast<size_t>(out_channels);

        out.wT_bytes = std::max(out.wT_bytes, w_need * sizeof(float));
        out.xcol_bytes = std::max(out.xcol_bytes, xcol_need * sizeof(float));
        out.c_bytes = std::max(out.c_bytes, c_need * sizeof(float));
    }

    return out;
}

} // namespace Mimir::Planning
