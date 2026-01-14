#pragma once

#include "../Model.hpp"

#include <string>

// DiffusionModel (baseline epsilon-predictor, float-only API)
// Input: float[time_dim + image_dim] (t embedding concatenated with x_t)
// Output: float[image_dim] (predicted noise)

class DiffusionModel : public Model {
public:
    struct Config {
        int image_w = 64;
        int image_h = 64;
        int image_c = 3;
        int time_dim = 128;
        int hidden_dim = 2048;
        float dropout = 0.0f;
    };

    DiffusionModel();
    void buildFromConfig(const Config& cfg);
    const Config& getConfig() const { return cfg_; }

    static void buildInto(Model& model, const Config& cfg);

private:
    Config cfg_;
};
