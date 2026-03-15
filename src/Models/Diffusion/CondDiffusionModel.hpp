#pragma once

#include "../Model.hpp"

#include <string>

// CondDiffusionModel (baseline epsilon-predictor, MLP)
// Input: float[prompt_dim + time_dim + latent_dim]
// Output: float[latent_dim]

class CondDiffusionModel : public Model {
public:
    struct Config {
        int prompt_dim = 128;
        int latent_w = 32;
        int latent_h = 32;
        int latent_c = 4;
        int time_dim = 128;
        int hidden_dim = 2048;
        float dropout = 0.0f;
    };

    CondDiffusionModel();

    void buildFromConfig(const Config& cfg);
    const Config& getConfig() const { return cfg_; }

    static void buildInto(Model& model, const Config& cfg);

private:
    Config cfg_;
};
