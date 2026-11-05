#pragma once
#include "../../Model.hpp"

// Native T5 encoder with an optional tied denoising head (no autoregressive decoder).
class T5Model : public Model {
public:
    struct Config {
        int vocab_size = 4096, seq_len = 77, d_model = 128;
        int num_layers = 4, num_heads = 4, mlp_hidden = 512;
        int padding_idx = 0, relative_attention_num_buckets = 32;
        int relative_attention_max_distance = 128;
        float norm_eps = 1e-6f;
        bool encoder_only = false;
    };
    T5Model();
    void buildFromConfig(const Config& cfg);
    static void buildInto(Model& model, const Config& cfg);
};
