#pragma once

#include "../Model.hpp"

// GanLatentModel (baseline, MLP)
// Input: float[prompt_dim + noise_dim]
// Output: float[latent_dim]

class GanLatentModel : public Model {
public:
    struct Config {
        int prompt_dim = 128;
        int noise_dim = 128;
        int latent_dim = 4096;
        int hidden_dim = 2048;
        int num_hidden_layers = 3;
        float dropout = 0.0f;
    };

    GanLatentModel();

    void buildFromConfig(const Config& cfg);
    const Config& getConfig() const { return cfg_; }

    static void buildInto(Model& model, const Config& cfg);

private:
    Config cfg_;
};
