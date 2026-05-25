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
};

struct PlannedOp {
    int layer_index = -1;
    FusionKind fusion = FusionKind::NONE;
};

struct ExecutionPlan {
    // Static scheduling: fixed op order
    std::vector<PlannedOp> ops;

    // Per-layer fusion flags for cheap hot-path checks
    std::vector<uint8_t> fuse_relu_for_conv2d;

    bool empty() const { return ops.empty(); }
};

inline ExecutionPlan build_execution_plan_static(const std::vector<Layer>& layers, bool training) {
    ExecutionPlan plan;
    plan.ops.reserve(layers.size());
    plan.fuse_relu_for_conv2d.assign(layers.size(), 0);

    for (size_t i = 0; i < layers.size(); ++i) {
        const Layer& layer = layers[i];

        PlannedOp op;
        op.layer_index = static_cast<int>(i);
        op.fusion = FusionKind::NONE;

        // Conservative: do not fuse in training (backward/masks/precision considerations).
        if (!training && layer.type_enum == LayerType::Conv2d && layer.activation == ActivationType::RELU) {
            op.fusion = FusionKind::CONV2D_RELU;
            plan.fuse_relu_for_conv2d[i] = 1;
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
