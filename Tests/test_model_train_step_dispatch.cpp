#include "test_utils.hpp"

#include "Model.hpp"
#include "Models/MLP/BasicMLPModel.hpp"

#include <vector>

int main() {
    Model base_model;
    Model::TrainStepRequest empty_request;
    TASSERT_TRUE(!base_model.trainStep(empty_request).has_value());

    BasicMLPModel model;
    BasicMLPModel::Config config;
    config.input_dim = 4;
    config.hidden_dim = 8;
    config.output_dim = 2;
    config.hidden_layers = 1;
    model.buildFromConfig(config);
    model.allocateParams();
    model.initializeWeights("xavier", 123u);

    std::vector<float> input = {0.5f, -1.0f, 0.25f, 2.0f};
    std::vector<float> target = {0.0f, 1.0f};
    Optimizer optimizer;
    optimizer.type = OptimizerType::SGD;
    optimizer.decay_strategy = LRDecayStrategy::NONE;

    Model::TrainStepRequest request;
    request.float_inputs["__input__"] = &input;
    request.target = &target;
    request.optimizer = &optimizer;
    request.learning_rate = 1e-2f;
    request.mode = Model::TrainStepMode::Accumulate;
    request.grad_scale = 0.5f;

    const auto accumulated = model.trainStep(request);
    TASSERT_TRUE(accumulated.has_value());
    TASSERT_TRUE(accumulated->loss >= 0.0f);
    TASSERT_TRUE(accumulated->metrics.count("mse") == 1);
    TASSERT_TRUE(optimizer.step == 0);

    request.mode = Model::TrainStepMode::Optimize;
    const auto optimized = model.trainStep(request);
    TASSERT_TRUE(optimized.has_value());
    TASSERT_TRUE(optimizer.step == 1);

    Model::TrainStepRequest missing_target = request;
    missing_target.target = nullptr;
    TASSERT_TRUE(!model.trainStep(missing_target).has_value());
    return 0;
}