#include "test_utils.hpp"

#include "Models/Registry/ModelArchitectures.hpp"
#include "Model.hpp"

#include <unordered_map>
#include <vector>

static float first_weight_value(const Model& m) {
    for (const auto& L : m.getLayers()) {
        if (L.weight_block && L.weight_block->getSize() > 0) {
            const float* p = L.weight_block->getData();
            if (p) return p[0];
        }
    }
    return 0.0f;
}

int main() {
    json cfg = {
        {"input_dim", 4},
        {"hidden_dim", 8},
        {"output_dim", 2},
        {"hidden_layers", 1},
        {"dropout", 0.0}
    };

    auto model = ModelArchitectures::create("basic_mlp", cfg);
    TASSERT_TRUE(model != nullptr);

    model->allocateParams();
    model->initializeWeights("xavier", 123u);

    std::unordered_map<std::string, std::vector<float>> fin;
    std::unordered_map<std::string, std::vector<int>> iin;

    fin["x"] = {0.5f, -1.0f, 0.25f, 2.0f};
    const auto pred0 = model->forwardPassNamed(fin, iin, /*training=*/false);
    TASSERT_TRUE(pred0.size() == 2);

    const float w0 = first_weight_value(*model);

    Optimizer opt;
    opt.type = OptimizerType::SGD;
    opt.decay_strategy = LRDecayStrategy::NONE;

    std::vector<float> target = {0.0f, 1.0f};
    const auto stats = model->trainStepNamed(fin, iin, target, opt, /*learning_rate=*/1e-2f);
    TASSERT_TRUE(stats.loss >= 0.0f);

    const float w1 = first_weight_value(*model);
    TASSERT_TRUE(w1 != w0);

    // Freeze should block training (zeroGradients throws).
    model->freezeParameters(true);
    bool threw = false;
    try {
        (void)model->trainStepNamed(fin, iin, target, opt, /*learning_rate=*/1e-2f);
    } catch (const std::exception&) {
        threw = true;
    }
    TASSERT_TRUE(threw);

    model->freezeParameters(false);
    threw = false;
    try {
        (void)model->trainStepNamed(fin, iin, target, opt, /*learning_rate=*/1e-2f);
    } catch (...) {
        threw = true;
    }
    TASSERT_TRUE(!threw);

    return 0;
}
