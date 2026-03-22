#include "test_utils.hpp"

#include "Model.hpp"

#include <unordered_map>
#include <vector>

int main() {
    Model m;

    // Layer 0: skip path = Identity(x) -> skip
    m.push("skip", "Identity", 0);
    // Layer 1: main path = Multiply(x, scale) -> x, then residual merge: x += skip
    m.push("my_residual", "Multiply", 0);

    auto& layers = m.getMutableLayers();
    TASSERT_TRUE(layers.size() == 2);

    layers[0].inputs = {"x"};
    layers[0].output = "skip";

    layers[1].inputs = {"x", "scale"};
    layers[1].output = "x";

    // Force residual behavior with explicit source.
    layers[1].detectBranchType();
    TASSERT_TRUE(layers[1].branch_type == BranchType::RESIDUAL);
    layers[1].branch_sources = {0};
    layers[1].merge_op = MergeOperation::ADD;

    m.allocateParams();

    std::unordered_map<std::string, std::vector<float>> fin;
    std::unordered_map<std::string, std::vector<int>> iin;

    fin["x"] = {1.0f, 2.0f};
    fin["scale"] = {2.0f, 3.0f};

    // Branch merge is executed only when training=true.
    const auto out = m.forwardPassNamed(fin, iin, /*training=*/true);
    TASSERT_TRUE(out.size() == 2);

    // main = x*scale => [2, 6]
    // skip = x => [1, 2]
    // merged = main + skip => [3, 8]
    TASSERT_NEAR(out[0], 3.0f, 1e-6f);
    TASSERT_NEAR(out[1], 8.0f, 1e-6f);

    return 0;
}
