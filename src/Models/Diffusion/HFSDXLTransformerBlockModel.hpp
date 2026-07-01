#pragma once

#include "../../Model.hpp"

class HFSDXLTransformerBlockModel : public Model {
public:
    struct Config {
        int q_len = 64;
        int kv_len = 77;
        int d_model = 640;
        int context_dim = 2048;
        int num_heads = 10;
        int ff_hidden = 2560;
        bool self_attn_qkv_bias = false;
        bool self_attn_out_bias = true;
        bool cross_attn_out_bias = true;
    };

    HFSDXLTransformerBlockModel();

    void buildFromConfig(const Config& cfg);
    static void buildInto(Model& model, const Config& cfg);

private:
    Config cfg_;
};