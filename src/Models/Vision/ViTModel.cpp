#include "ViTModel.hpp"

#include <algorithm>

ViTModel::ViTModel() {
    setModelName("ViTModel");
    setHasEncoder(false);
}

void ViTModel::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
}

static inline size_t sat_mul(size_t a, size_t b) {
    if (a == 0 || b == 0) return 0;
    if (a > (static_cast<size_t>(-1) / b)) return static_cast<size_t>(-1);
    return a * b;
}

void ViTModel::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("ViTModel");
    model.modelConfig["type"] = "vit";

    const int tokens = std::max(1, cfg.num_tokens);
    const int d_model = std::max(1, cfg.d_model);
    const int heads = std::max(1, cfg.num_heads);
    const int layers = std::max(1, cfg.num_layers);
    const int mlp_hidden = std::max(1, cfg.mlp_hidden);
    const int out_dim = std::max(1, cfg.output_dim);

    const int in_dim = tokens * d_model;
    model.modelConfig["task"] = "vit_encoder";
    model.modelConfig["num_tokens"] = tokens;
    model.modelConfig["d_model"] = d_model;
    model.modelConfig["num_layers"] = layers;
    model.modelConfig["num_heads"] = heads;
    model.modelConfig["mlp_hidden"] = mlp_hidden;
    model.modelConfig["input_dim"] = in_dim;
    model.modelConfig["output_dim"] = out_dim;

    model.push("vit/raw_in", "Identity", 0);
    if (auto* L = model.getLayerByName("vit/raw_in")) {
        L->inputs = {"__input__"};
        L->output = "vit/in";
    }

    std::string x = "vit/in";

    for (int i = 0; i < layers; ++i) {
        const std::string p = "vit/block" + std::to_string(i + 1);

        model.push(p + "/ln1", "LayerNorm", static_cast<size_t>(2) * static_cast<size_t>(d_model));
        if (auto* L = model.getLayerByName(p + "/ln1")) {
            L->inputs = {x};
            L->output = p + "/ln1_out";
            L->in_features = d_model; // LN par token (groupes)
            L->affine = true;
            L->use_bias = true;
        }

        const size_t attn_params = sat_mul(static_cast<size_t>(4), sat_mul(static_cast<size_t>(d_model), static_cast<size_t>(d_model)));
        model.push(p + "/attn", "MultiHeadAttention", attn_params);
        if (auto* L = model.getLayerByName(p + "/attn")) {
            L->inputs = {p + "/ln1_out"};
            L->output = p + "/attn_out";
            L->seq_len = tokens;
            L->embed_dim = d_model;
            L->num_heads = heads;
            L->causal = cfg.causal;
        }

        model.push(p + "/add1", "Add", 0);
        if (auto* L = model.getLayerByName(p + "/add1")) {
            L->inputs = {x, p + "/attn_out"};
            L->output = p + "/res1";
        }

        model.push(p + "/ln2", "LayerNorm", static_cast<size_t>(2) * static_cast<size_t>(d_model));
        if (auto* L = model.getLayerByName(p + "/ln2")) {
            L->inputs = {p + "/res1"};
            L->output = p + "/ln2_out";
            L->in_features = d_model; // LN par token (groupes)
            L->affine = true;
            L->use_bias = true;
        }

        model.push(p + "/mlp_fc1", "Linear", sat_mul(static_cast<size_t>(d_model), static_cast<size_t>(mlp_hidden)) + static_cast<size_t>(mlp_hidden));
        if (auto* L = model.getLayerByName(p + "/mlp_fc1")) {
            L->inputs = {p + "/ln2_out"};
            L->output = p + "/mlp_h";
            L->seq_len = tokens;
            L->in_features = d_model;
            L->out_features = mlp_hidden;
            L->use_bias = true;
        }

        model.push(p + "/mlp_act", "GELU", 0);
        if (auto* L = model.getLayerByName(p + "/mlp_act")) {
            L->inputs = {p + "/mlp_h"};
            L->output = p + "/mlp_h_act";
        }

        model.push(p + "/mlp_fc2", "Linear", sat_mul(static_cast<size_t>(mlp_hidden), static_cast<size_t>(d_model)) + static_cast<size_t>(d_model));
        if (auto* L = model.getLayerByName(p + "/mlp_fc2")) {
            L->inputs = {p + "/mlp_h_act"};
            L->output = p + "/mlp_out";
            L->seq_len = tokens;
            L->in_features = mlp_hidden;
            L->out_features = d_model;
            L->use_bias = true;
        }

        model.push(p + "/add2", "Add", 0);
        if (auto* L = model.getLayerByName(p + "/add2")) {
            L->inputs = {p + "/res1", p + "/mlp_out"};
            L->output = p + "/res2";
        }

        x = p + "/res2";
    }

    model.push("vit/pool", "TokenMeanPool", 0);
    if (auto* L = model.getLayerByName("vit/pool")) {
        L->inputs = {x};
        L->output = "vit/pooled";
        L->seq_len = tokens;
        L->embed_dim = d_model;
    }

    model.push("vit/head", "Linear", sat_mul(static_cast<size_t>(d_model), static_cast<size_t>(out_dim)) + static_cast<size_t>(out_dim));
    if (auto* L = model.getLayerByName("vit/head")) {
        L->inputs = {"vit/pooled"};
        L->output = "x";
        L->in_features = d_model;
        L->out_features = out_dim;
        L->use_bias = true;
    }
}
