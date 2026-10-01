#pragma once

#include "../Model.hpp"

class VAETraining {
public:
    Model::TrainStepResult trainImage(Model& model,
                                      const std::vector<float>& input,
                                      const std::vector<float>* target,
                                      Optimizer& optimizer,
                                      float learning_rate,
                                      Model::TrainStepMode mode,
                                      float grad_scale);

    Model::TrainStepResult trainText(Model& model,
                                     const std::vector<float>& input,
                                     const std::vector<int>& text_ids,
                                     const std::vector<float>* target,
                                     Optimizer& optimizer,
                                     float learning_rate,
                                     Model::TrainStepMode mode,
                                     float grad_scale);

private:
    std::shared_ptr<Model> perceptual_model_;
    std::vector<float> pending_perceptual_prior_;
};