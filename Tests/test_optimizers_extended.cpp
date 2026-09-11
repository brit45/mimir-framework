#include "test_utils.hpp"

#include "Model.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

struct Fixture {
    Model model;
    Layer* layer = nullptr;
    float* weights = nullptr;

    Fixture() {
        model.push("parameter", "Constant", 3);
        layer = model.getLayerByName("parameter");
        layer->inputs = {};
        layer->output = "x";
        layer->trainable_parameter = true;
        model.allocateParams();
        weights = layer->getWeights();
        weights[0] = 1.0f;
        weights[1] = -2.0f;
        weights[2] = 0.5f;
        layer->grad_weights = {0.2f, -0.4f, 0.1f};
    }

    const Optimizer::MomentBlock& moments(const Optimizer& optimizer) const {
        return optimizer.mv_by_param_ptr.at(reinterpret_cast<std::uintptr_t>(weights));
    }
};

Optimizer optimizer(OptimizerType type) {
    Optimizer result;
    result.type = type;
    result.decay_strategy = LRDecayStrategy::NONE;
    result.weight_decay = 0.0f;
    result.beta1 = 0.9f;
    result.beta2 = 0.9f;
    result.eps = 1e-8f;
    return result;
}

}  // namespace

int main() {
    constexpr float learning_rate = 0.01f;

    {
        Fixture fixture;
        auto opt = optimizer(OptimizerType::LION);
        fixture.model.optimizerStep(opt, learning_rate);
        TASSERT_NEAR(fixture.weights[0], 0.99f, 1e-6f);
        TASSERT_NEAR(fixture.weights[1], -1.99f, 1e-6f);
        TASSERT_NEAR(fixture.moments(opt).m[0], 0.02f, 1e-6f);
        TASSERT_TRUE(opt.step == 1);
    }

    {
        Fixture fixture;
        auto opt = optimizer(OptimizerType::ADAFACTOR);
        opt.adafactor_scale_parameter = false;
        opt.adafactor_clip_threshold = 10.0f;
        fixture.model.optimizerStep(opt, learning_rate);
        const float update = 1.0f;
        TASSERT_NEAR(fixture.weights[0], 1.0f - learning_rate * update, 1e-6f);
        TASSERT_NEAR(fixture.weights[1], -2.0f + learning_rate * update, 1e-6f);
        TASSERT_NEAR(fixture.moments(opt).v[0], 0.04f, 1e-6f);
    }

    {
        Fixture fixture;
        auto opt = optimizer(OptimizerType::RADAM);
        fixture.model.optimizerStep(opt, learning_rate);
        TASSERT_NEAR(fixture.weights[0], 1.0f - learning_rate * 0.2f, 1e-6f);
        TASSERT_NEAR(fixture.weights[1], -2.0f + learning_rate * 0.4f, 1e-6f);
    }

    {
        Fixture fixture;
        auto opt = optimizer(OptimizerType::NADAM);
        fixture.model.optimizerStep(opt, learning_rate);
        TASSERT_NEAR(fixture.weights[0], 1.0f - learning_rate * 1.9f, 1e-6f);
        TASSERT_NEAR(fixture.weights[1], -2.0f + learning_rate * 1.9f, 1e-6f);
    }

    {
        Fixture fixture;
        auto opt = optimizer(OptimizerType::RMSPROP);
        opt.rmsprop_alpha = 0.9f;
        fixture.model.optimizerStep(opt, learning_rate);
        const float update = 1.0f / std::sqrt(0.1f);
        TASSERT_NEAR(fixture.weights[0], 1.0f - learning_rate * update, 1e-6f);
        TASSERT_NEAR(fixture.weights[1], -2.0f + learning_rate * update, 1e-6f);
        TASSERT_NEAR(fixture.moments(opt).v[1], 0.016f, 1e-6f);
    }

    {
        Fixture fixture;
        auto opt = optimizer(OptimizerType::LAMB);
        fixture.model.optimizerStep(opt, learning_rate);
        const float trust_ratio = std::sqrt(5.25f / 3.0f);
        TASSERT_NEAR(fixture.weights[0], 1.0f - learning_rate * trust_ratio, 1e-6f);
        TASSERT_NEAR(fixture.weights[1], -2.0f + learning_rate * trust_ratio, 1e-6f);
        TASSERT_NEAR(fixture.moments(opt).m[0], 0.02f, 1e-6f);
        TASSERT_NEAR(fixture.moments(opt).v[0], 0.004f, 1e-6f);
    }

    return 0;
}