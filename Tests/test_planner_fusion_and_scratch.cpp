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

static Layer make_unary_layer(const char* name, LayerType type, const char* input, const char* output) {
    Layer l;
    l.name = name;
    l.type_enum = type;
    l.inputs = {input};
    l.output = output;
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

        // Training opt-in: only Conv2d+ReLU fusion should be enabled.
        auto train_fused = Mimir::Planning::build_execution_plan_static(
            layers,
            /*training=*/true,
            /*allow_training_fusion=*/true
        );
        TASSERT_TRUE(train_fused.ops.size() == 1);
        TASSERT_TRUE(train_fused.fuse_relu_for_conv2d.size() == 1);
        TASSERT_TRUE(train_fused.fuse_relu_for_conv2d[0] == 1);
    }

    // Generic producer -> activation -> chunk should be fused in inference.
    {
        Layer add;
        add.name = "add";
        add.type_enum = LayerType::Add;
        add.inputs = {"a", "b"};
        add.output = "sum";

        Layer relu = make_unary_layer("relu", LayerType::ReLU, "sum", "sum_relu");

        Layer chunk;
        chunk.name = "chunk";
        chunk.type_enum = LayerType::Chunk;
        chunk.inputs = {"sum_relu"};
        chunk.output = "chunked";
        chunk.num_chunks = 2;
        chunk.split_axis = 0;

        std::vector<Layer> generic_layers = {add, relu, chunk};
        auto plan = Mimir::Planning::build_execution_plan_static(generic_layers, /*training=*/false);
        TASSERT_TRUE(plan.ops.size() == 3);
        TASSERT_TRUE(plan.fuse_activation_consumer.size() == 3);
        TASSERT_TRUE(plan.fuse_split_consumer.size() == 3);
        TASSERT_TRUE(plan.fuse_activation_consumer[0] == 1);
        TASSERT_TRUE(plan.fuse_split_consumer[0] == 2);
        TASSERT_TRUE(plan.skip_layer[1] == 1);
        TASSERT_TRUE(plan.skip_layer[2] == 1);
        TASSERT_TRUE(plan.ops[0].fusion == Mimir::Planning::FusionKind::GENERIC_ACTIVATION_CHUNK);

        // Training even with opt-in must keep generic fusions disabled
        // to preserve intermediate tensors required by backward paths.
        auto train_plan = Mimir::Planning::build_execution_plan_static(
            generic_layers,
            /*training=*/true,
            /*allow_training_fusion=*/true
        );
        TASSERT_TRUE(train_plan.fuse_activation_consumer[0] == -1);
        TASSERT_TRUE(train_plan.fuse_split_consumer[0] == -1);
        TASSERT_TRUE(train_plan.skip_layer[1] == 0);
        TASSERT_TRUE(train_plan.skip_layer[2] == 0);
    }

    // Generic producer -> unary shape/no-op should also be fused in inference.
    {
        Layer linear;
        linear.name = "linear";
        linear.type_enum = LayerType::Linear;
        linear.inputs = {"__input__"};
        linear.output = "linear_out";

        Layer identity = make_unary_layer("identity", LayerType::Identity, "linear_out", "identity_out");

        std::vector<Layer> unary_layers = {linear, identity};
        auto plan = Mimir::Planning::build_execution_plan_static(unary_layers, /*training=*/false);
        TASSERT_TRUE(plan.ops.size() == 2);
        TASSERT_TRUE(plan.fuse_unary_consumer.size() == 2);
        TASSERT_TRUE(plan.fuse_unary_consumer[0] == 1);
        TASSERT_TRUE(plan.skip_layer[1] == 1);
        TASSERT_TRUE(plan.ops[0].fusion == Mimir::Planning::FusionKind::GENERIC_UNARY_SHAPE);

        auto train_plan = Mimir::Planning::build_execution_plan_static(unary_layers, /*training=*/true);
        TASSERT_TRUE(train_plan.fuse_unary_consumer[0] == -1);
        TASSERT_TRUE(train_plan.skip_layer[1] == 0);

        auto train_plan_opt_in = Mimir::Planning::build_execution_plan_static(
            unary_layers,
            /*training=*/true,
            /*allow_training_fusion=*/true
        );
        TASSERT_TRUE(train_plan_opt_in.fuse_unary_consumer[0] == -1);
        TASSERT_TRUE(train_plan_opt_in.skip_layer[1] == 0);
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
