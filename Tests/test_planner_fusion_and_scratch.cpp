#include "test_utils.hpp"
#include "Model.hpp"
#include <cstdlib>

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

        // Training also preserves split snapshots and secondary outputs.
        auto train_plan = Mimir::Planning::build_execution_plan_static(
            generic_layers,
            /*training=*/true,
            /*allow_training_fusion=*/true
        );
        TASSERT_TRUE(train_plan.fuse_activation_consumer[0] == 1);
        TASSERT_TRUE(train_plan.fuse_split_consumer[0] == 2);
        TASSERT_TRUE(train_plan.skip_layer[1] == 1);
        TASSERT_TRUE(train_plan.skip_layer[2] == 1);
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
        TASSERT_TRUE(train_plan_opt_in.fuse_unary_consumer[0] == 1);
        TASSERT_TRUE(train_plan_opt_in.skip_layer[1] == 1);
    }

    // Generic producer -> activations répétées -> unary shape répétés
    // doit construire une chaîne de fusion complète en inférence.
    {
        Layer add;
        add.name = "add_chain";
        add.type_enum = LayerType::Add;
        add.inputs = {"a", "b"};
        add.output = "sum0";

        Layer relu1 = make_unary_layer("relu1", LayerType::ReLU, "sum0", "sum1");
        Layer relu2 = make_unary_layer("relu2", LayerType::ReLU, "sum1", "sum2");
        Layer id1 = make_unary_layer("id1", LayerType::Identity, "sum2", "sum3");
        Layer id2 = make_unary_layer("id2", LayerType::Identity, "sum3", "sum4");

        std::vector<Layer> chain_layers = {add, relu1, relu2, id1, id2};
        auto plan = Mimir::Planning::build_execution_plan_static(chain_layers, /*training=*/false);

        TASSERT_TRUE(plan.ops.size() == chain_layers.size());
        TASSERT_TRUE(plan.fuse_chain_next.size() == chain_layers.size());
        TASSERT_TRUE(plan.fuse_chain_next[0] == 1);
        TASSERT_TRUE(plan.fuse_chain_next[1] == 2);
        TASSERT_TRUE(plan.fuse_chain_next[2] == 3);
        TASSERT_TRUE(plan.fuse_chain_next[3] == 4);
        TASSERT_TRUE(plan.fuse_chain_next[4] == -1);

        TASSERT_TRUE(plan.skip_layer[1] == 1);
        TASSERT_TRUE(plan.skip_layer[2] == 1);
        TASSERT_TRUE(plan.skip_layer[3] == 1);
        TASSERT_TRUE(plan.skip_layer[4] == 1);

        // Le premier maillon reste classé comme fusion d'activation générique.
        TASSERT_TRUE(plan.ops[0].fusion == Mimir::Planning::FusionKind::GENERIC_ACTIVATION);

        // En training, la fusion générique en chaîne reste désactivée.
        auto train_plan = Mimir::Planning::build_execution_plan_static(chain_layers, /*training=*/true);
        TASSERT_TRUE(train_plan.fuse_chain_next[0] == -1);
        TASSERT_TRUE(train_plan.skip_layer[1] == 0);
        TASSERT_TRUE(train_plan.skip_layer[2] == 0);
        TASSERT_TRUE(train_plan.skip_layer[3] == 0);
        TASSERT_TRUE(train_plan.skip_layer[4] == 0);
    }

    // Every fused logical edge must retain the unfused arithmetic and backward state.
    for (const char* dtype : {"float32", "float16"}) for (const char* producer : {"Linear", "Conv2d"}) {
        Model model;
        model.setDefaultDType(dtype);
        model.push("producer", producer, 6);
        for (const char* type : {"ReLU", "Tanh", "Sigmoid", "Identity"}) model.push(type, type, 0);
        auto& ls = model.getMutableLayers();
        ls[0].in_features = 2; ls[0].out_features = 2; ls[0].use_bias = true;
        if (std::string(producer)=="Conv2d") {
            ls[0].in_channels=2; ls[0].out_channels=2;
            ls[0].kernel_size=1; ls[0].stride=1; ls[0].padding=0;
            ls[0].input_width=1; ls[0].input_height=1;
            ls[0].output_width=1; ls[0].output_height=1;
            ls[0].activation=ActivationType::RELU;
        }
        for (size_t i=0; i<ls.size(); ++i) {
            ls[i].inputs = {i == 0 ? "__input__" : "edge" + std::to_string(i-1)};
            ls[i].output = i+1 == ls.size() ? "x" : "edge" + std::to_string(i);
        }
        model.allocateParams();
        const std::vector<float> initial{.37f,-.21f,.43f,.15f,.22f,.07f};
        std::copy(initial.begin(),initial.end(),ls[0].getWeights());
        auto plan = Mimir::Planning::build_execution_plan_global(ls, true);
        TASSERT_TRUE(plan.skip_layer[1] == 1 && plan.skip_layer[4] == 1);
        TASSERT_TRUE(plan.fusion_groups.size() >= 4);
        TASSERT_TRUE(!plan.fusion_groups.back().children.empty());
        TASSERT_TRUE(!plan.fusion_groups.back().parent.has_value());
        auto set_fusion = [](const char* value) {
#ifdef _WIN32
            _putenv_s("MIMIR_PLANNER_FUSION", value);
#else
            setenv("MIMIR_PLANNER_FUSION", value, 1);
#endif
        };
        std::vector<float> reference_output, reference_input_grad, reference_grad, reference_weights;
        for (const char* fusion : {"0", "1"}) {
            set_fusion(fusion);
            std::copy(initial.begin(),initial.end(),ls[0].getWeights());
            Optimizer optimizer; optimizer.type=OptimizerType::ADAM;
            for (int step=0; step<3; ++step) {
                model.zeroGradients();
                const auto output=model.forwardPass(std::vector<float>{-.31f,.83f},true);
                model.backwardPass({.4f,-.7f});
                if (step==0) {
                    if (*fusion=='0') {
                        reference_output=output;
                        reference_input_grad=model.getLastInputGradient();
                        reference_grad=ls[0].grad_weights;
                    } else {
                        TASSERT_TRUE(output==reference_output);
                        TASSERT_TRUE(model.getLastInputGradient()==reference_input_grad);
                        TASSERT_TRUE(ls[0].grad_weights==reference_grad);
                    }
                }
                model.optimizerStep(optimizer,.01f);
            }
            std::vector<float> weights(ls[0].getWeights(),ls[0].getWeights()+6);
            if (*fusion=='0') reference_weights=weights;
            else TASSERT_TRUE(weights==reference_weights);
        }
        set_fusion("1");
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
