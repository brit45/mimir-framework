#include "test_utils.hpp"

#include "Model.hpp"

#include <unordered_map>
#include <vector>

int main() {
    Model m;

    // Build a tiny graph using named tensors:
    // sum = a + b
    // x   = sum * c
    m.push("sum", "Add", 0);
    m.push("x", "Multiply", 0);

    auto& layers = m.getMutableLayers();
    TASSERT_TRUE(layers.size() == 2);

    layers[0].inputs = {"a", "b"};
    layers[0].output = "sum";

    layers[1].inputs = {"sum", "c"};
    layers[1].output = "x";

    // Need weight blocks vector non-empty (even if paramless).
    m.allocateParams();

    std::unordered_map<std::string, std::vector<float>> fin;
    std::unordered_map<std::string, std::vector<int>> iin;

    fin["x"] = {0.0f, 0.0f};
    fin["a"] = {1.0f, 2.0f};
    fin["b"] = {3.0f, 4.0f};
    fin["c"] = {2.0f, 2.0f};

    const auto out = m.forwardPassNamed(fin, iin, /*training=*/false);
    TASSERT_TRUE(out.size() == 2);
    TASSERT_NEAR(out[0], 8.0f, 1e-6f);
    TASSERT_NEAR(out[1], 12.0f, 1e-6f);

    // Missing input should throw at runtime (TensorStore lookup).
    bool threw = false;
    try {
        auto fin2 = fin;
        fin2.erase("c");
        (void)m.forwardPassNamed(fin2, iin, /*training=*/false);
    } catch (const std::exception&) {
        threw = true;
    }
    TASSERT_TRUE(threw);

    // Layer branch detection is name-based.
    {
        Layer L("my_residual_block", "Add", 0);
        L.detectBranchType();
        TASSERT_TRUE(L.requiresBranchComputation());
        TASSERT_TRUE(L.branch_type == BranchType::RESIDUAL);
    }

    return 0;
}
