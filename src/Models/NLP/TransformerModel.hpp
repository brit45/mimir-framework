#pragma once

#include "../Model.hpp"

#include <string>

// TransformerModel (encoder-only, float-only API)
// Input: float[seq_len * d_model]
// Output: float[output_dim]

class TransformerModel : public Model {
public:
    struct Config {
        int seq_len = 64;
        int d_model = 128;
        int vocab_size = 4096;
        int padding_idx = 0;
        int num_layers = 4;
        int num_heads = 4;
        int mlp_hidden = 256;
        int output_dim = 256;
        bool causal = false;
    };

    TransformerModel();

    void buildFromConfig(const Config& cfg);
    const Config& getConfig() const { return cfg_; }

    static void buildInto(Model& model, const Config& cfg);

private:
    Config cfg_;
};
