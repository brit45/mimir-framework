#pragma once

#include "../../Model.hpp"

class HFCLIPTextEncoder2Model : public Model {
public:
    struct Config {
        int vocab_size = 49408;
        int padding_idx = 0;
        int seq_len = 77;
        int d_model = 1280;
        int num_layers = 32;
        int num_heads = 20;
        int mlp_hidden = 5120;
        int proj_dim = 1280;
        bool causal = true;
        bool include_logit_scale = true;
    };

    HFCLIPTextEncoder2Model();
    void buildFromConfig(const Config& cfg);
    static void buildInto(Model& model, const Config& cfg);

private:
    Config cfg_;
};