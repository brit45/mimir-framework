#pragma once

#include "../../Model.hpp"

class HFCLIPTextEncoder1Model : public Model {
public:
    struct Config {
        int vocab_size = 49408;
        int padding_idx = 0;
        int seq_len = 77;
        int d_model = 768;
        int num_layers = 12;
        int num_heads = 12;
        int mlp_hidden = 3072;
        bool causal = true;
    };

    HFCLIPTextEncoder1Model();
    void buildFromConfig(const Config& cfg);
    static void buildInto(Model& model, const Config& cfg);

private:
    Config cfg_;
};