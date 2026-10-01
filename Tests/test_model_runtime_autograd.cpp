#include "test_utils.hpp"
#include "Model.hpp"
#include <vector>

int main() {
    // Tensor names identify versions: x -> x must consume, not retain, dy.
    Model model;
    model.push("input", "Identity", 0);
    model.push("relu", "ReLU", 0);
    model.push("linear", "Linear", 6);
    auto& layers=model.getMutableLayers();
    layers[0].inputs={"__input__"}; layers[0].output="x";
    layers[1].inputs={"x"}; layers[1].output="x";
    layers[2].inputs={"x"}; layers[2].output="x";
    layers[2].in_features=2;layers[2].out_features=2;layers[2].use_bias=true;
    model.allocateParams();
    auto* weights=layers[2].getWeights();
    const std::vector<float> w={2,0,0,3,0,0};
    std::copy(w.begin(),w.end(),weights);
    auto output=model.forwardPass(std::vector<float>{-1,2},true);
    TASSERT_TRUE(output==std::vector<float>({0,6}));
    model.backwardPass({1,1});
    TASSERT_TRUE(model.hasLastInputGradient());
    TASSERT_TRUE(model.getLastInputGradient()==std::vector<float>({0,3}));
    TASSERT_TRUE(layers[2].grad_weights==std::vector<float>({0,2,0,2,1,1}));
    TASSERT_TRUE(model.getGradients().param_grads.size()==6);

    // A Split may receive gradients only through a secondary named output.
    for(int variant : {0, 1, 2}) {
        const bool chunk = variant == 1;
        Model split;
        split.push("split",chunk?"Chunk":"Split",0);
        split.push("out","Identity",0);
        auto& ls=split.getMutableLayers();
        ls[0].inputs={"__input__"};ls[0].output="parts";ls[0].split_sizes={2,2};ls[0].num_chunks=2;
        if (variant == 2) { ls[0].split_sizes.clear(); ls[0].num_splits=2; }
        ls[1].inputs={"parts_1"};ls[1].output="x";
        split.allocateParams();
        auto y=split.forwardPass(std::vector<float>{1,2,3,4},true);
        TASSERT_TRUE(y==std::vector<float>({3,4}));
        split.backwardPass({5,7});
        TASSERT_TRUE(split.getLastInputGradient()==std::vector<float>({0,0,5,7}));
    }
    return 0;
}
