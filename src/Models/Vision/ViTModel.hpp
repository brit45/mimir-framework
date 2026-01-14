#pragma once

#include "../Model.hpp"

// ViTModel (Vision Transformer, float-only API)
// NOTE: This implementation expects patch embeddings as input.
// Input: float[num_tokens * d_model] where num_tokens = 1 + num_patches (e.g. CLS + patches)
// Output: float[output_dim]

class ViTModel : public Model {
public:
    struct Config {
        int num_tokens = 197;  // 1 + num_patches
        int d_model = 128;
        int num_layers = 4;
        int num_heads = 4;
        int mlp_hidden = 256;
        int output_dim = 1000;
        bool causal = false;
    };

    ViTModel();
    void buildFromConfig(const Config& cfg);
    const Config& getConfig() const { return cfg_; }

    static void buildInto(Model& model, const Config& cfg);

private:
    Config cfg_;
};
