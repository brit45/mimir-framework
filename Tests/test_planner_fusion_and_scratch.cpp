#include "test_utils.hpp"

#include "Layers.hpp"
#include "Planning/Planner.hpp"

#include <cstddef>

static Layer make_conv2d(int in_ch, int out_ch, int k, int h, int w, ActivationType act) {
    Layer l;
    l.name = "conv";
    l.type_enum = LayerType::Conv2d;
    l.in_channels = in_ch;
    l.out_channels = out_ch;
    l.kernel_size = k;
    l.input_height = h;
    l.input_width = w;
    l.stride = 1;
    l.padding = 1;
    l.activation = act;
    return l;
}

int main() {
    std::vector<Layer> layers;
    layers.push_back(make_conv2d(3, 8, 3, 32, 32, ActivationType::RELU));

    // Fusion plan: inference should fuse Conv2d+ReLU
    {
        auto plan = Mimir::Planning::build_execution_plan_static(layers, /*training=*/false);
        TASSERT_TRUE(plan.ops.size() == 1);
        TASSERT_TRUE(plan.fuse_relu_for_conv2d.size() == 1);
        TASSERT_TRUE(plan.fuse_relu_for_conv2d[0] == 1);
    }

    // Training plan: no fusion
    {
        auto plan = Mimir::Planning::build_execution_plan_static(layers, /*training=*/true);
        TASSERT_TRUE(plan.ops.size() == 1);
        TASSERT_TRUE(plan.fuse_relu_for_conv2d.size() == 1);
        TASSERT_TRUE(plan.fuse_relu_for_conv2d[0] == 0);
    }

    // Scratch planner: should request non-zero buffers
    {
        auto scratch = Mimir::Planning::plan_conv2d_fastpath_scratch(layers);
        TASSERT_TRUE(scratch.wT_bytes > 0);
        TASSERT_TRUE(scratch.xcol_bytes > 0);
        TASSERT_TRUE(scratch.c_bytes > 0);

        const int K = 3 * 3 * 3;
        const size_t min_wT = static_cast<size_t>(8) * static_cast<size_t>(K) * sizeof(float);
        TASSERT_TRUE(scratch.wT_bytes >= min_wT);
    }

    return 0;
}
