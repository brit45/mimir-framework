#include "test_utils.hpp"

#include "Models/Registry/ModelArchitectures.hpp"

#include <cstddef>

int main() {
    json cfg = {
        {"input_dim", 4},
        {"hidden_dim", 8},
        {"output_dim", 2},
        {"hidden_layers", 2},
        {"dropout", 0.0}
    };

    auto model = ModelArchitectures::create("basic_mlp", cfg);
    TASSERT_TRUE(model != nullptr);

    // Basic invariants before allocation.
    const auto& layers0 = model->getLayers();
    TASSERT_TRUE(!layers0.empty());
    TASSERT_TRUE(model->totalParamCount() > 0);

    model->allocateParams();

    const auto& layers = model->getLayers();
    size_t sum_params = 0;
    size_t sum_blocks = 0;

    for (const auto& L : layers) {
        sum_params += L.params_count;
        if (L.params_count > 0) {
            TASSERT_TRUE(L.weight_block != nullptr);
            TASSERT_TRUE(L.getWeights() != nullptr);
            TASSERT_TRUE(L.getWeightsSize() == L.params_count);
            sum_blocks += L.weight_block->getSize();
            TASSERT_TRUE(L.weight_block->getSize() == L.params_count);
        } else {
            // Paramless layer: weight_block may be nullptr.
            // (No strong assertion needed.)
        }
    }

    TASSERT_TRUE(sum_params == model->totalParamCount());
    TASSERT_TRUE(sum_blocks == model->totalParamCount());

    return 0;
}
