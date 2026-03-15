#include "SD35Model.hpp"

#include <algorithm>

SD35Model::SD35Model() {
    setModelName("SD35Model");
    // SD3.5 est un pipeline conditionné (texte, éventuellement image), donc encoder=true.
    setHasEncoder(true);
}

void SD35Model::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
}

void SD35Model::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("SD35Model");
    model.modelConfig["task"] = "sd3_5";

    const int q_len = std::max(1, cfg.q_len);
    const int kv_len = std::max(1, cfg.kv_len);
    const int d_model = std::max(1, cfg.d_model);
    const int heads = std::max(1, cfg.num_heads);
    const int layers = std::max(1, cfg.num_layers);
    const int mlp_hidden = std::max(1, cfg.mlp_hidden);

    const int q_dim = q_len * d_model;
    const int kv_dim = kv_len * d_model;
    const int input_dim = q_dim + kv_dim;
    const int output_dim = q_dim;

    model.modelConfig["stub_only"] = cfg.stub_only;
    model.modelConfig["q_len"] = q_len;
    model.modelConfig["kv_len"] = kv_len;
    model.modelConfig["d_model"] = d_model;
    model.modelConfig["num_heads"] = heads;
    model.modelConfig["num_layers"] = layers;
    model.modelConfig["mlp_hidden"] = mlp_hidden;
    model.modelConfig["causal"] = cfg.causal;
    model.modelConfig["input_dim"] = input_dim;
    model.modelConfig["output_dim"] = output_dim;

    // Entrée aplatie: [query_stream || kv_stream]
    model.push("sd3_5/in", "Identity", 0);
    if (auto* L = model.getLayerByName("sd3_5/in")) {
        L->inputs = {"__input__"};
        L->output = "sd3_5/packed";
    }

    // Split 1D (axis ignoré par LayerOps::split_forward) -> sd3_5/split_0 et sd3_5/split_1
    model.push("sd3_5/split", "Split", 0);
    if (auto* L = model.getLayerByName("sd3_5/split")) {
        L->inputs = {"sd3_5/packed"};
        L->output = "sd3_5/split";
        L->split_sizes = {q_dim, kv_dim};
        L->split_axis = 0;
    }

    // Par convention dans forwardPass: Split renvoie *_0 comme sortie principale
    std::string q = "sd3_5/split_0";
    const std::string kv = "sd3_5/split_1";

    if (cfg.stub_only) {
        model.push("sd3_5/out", "Identity", 0);
        if (auto* L = model.getLayerByName("sd3_5/out")) {
            L->inputs = {q};
            L->output = "x";
        }
        return;
    }

    auto sat_mul = [](size_t a, size_t b) -> size_t {
        if (a == 0 || b == 0) return 0;
        if (a > (static_cast<size_t>(-1) / b)) return static_cast<size_t>(-1);
        return a * b;
    };

    for (int i = 0; i < layers; ++i) {
        const std::string p = "sd3_5/block" + std::to_string(i + 1);

        // Pre-LN (query)
        model.push(p + "/ln1", "LayerNorm", static_cast<size_t>(2) * static_cast<size_t>(d_model));
        if (auto* L = model.getLayerByName(p + "/ln1")) {
            L->inputs = {q};
            L->output = p + "/ln1_out";
            L->affine = true;
            L->use_bias = true;
            L->eps = 1e-5f;
            // Activer LayerNorm token-wise: normaliser chaque token sur d_model.
            L->in_features = d_model;
        }

        // Self-attn sur query
        const size_t attn_params = sat_mul(static_cast<size_t>(4), sat_mul(static_cast<size_t>(d_model), static_cast<size_t>(d_model)));
        model.push(p + "/self_attn", "MultiHeadAttention", attn_params);
        if (auto* L = model.getLayerByName(p + "/self_attn")) {
            L->inputs = {p + "/ln1_out"};
            L->output = p + "/self_attn_out";
            L->seq_len = q_len;
            L->embed_dim = d_model;
            L->num_heads = heads;
            L->causal = cfg.causal;
        }

        model.push(p + "/add1", "Add", 0);
        if (auto* L = model.getLayerByName(p + "/add1")) {
            L->inputs = {q, p + "/self_attn_out"};
            L->output = p + "/res1";
        }

        // LN2
        model.push(p + "/ln2", "LayerNorm", static_cast<size_t>(2) * static_cast<size_t>(d_model));
        if (auto* L = model.getLayerByName(p + "/ln2")) {
            L->inputs = {p + "/res1"};
            L->output = p + "/ln2_out";
            L->affine = true;
            L->use_bias = true;
            L->eps = 1e-5f;
            // Activer LayerNorm token-wise: normaliser chaque token sur d_model.
            L->in_features = d_model;
        }

        // Cross-attn: query=ln2_out, kv=kv (pour l'instant kv est fixe)
        model.push(p + "/cross_attn", "CrossAttention", attn_params);
        if (auto* L = model.getLayerByName(p + "/cross_attn")) {
            L->inputs = {p + "/ln2_out", kv};
            L->output = p + "/cross_attn_out";
            // embed_dim + heads suffisent: forward déduit query_len/kv_len depuis les tailles
            L->embed_dim = d_model;
            L->num_heads = heads;
            L->causal = false;
        }

        model.push(p + "/add2", "Add", 0);
        if (auto* L = model.getLayerByName(p + "/add2")) {
            L->inputs = {p + "/res1", p + "/cross_attn_out"};
            L->output = p + "/res2";
        }

        // LN3
        model.push(p + "/ln3", "LayerNorm", static_cast<size_t>(2) * static_cast<size_t>(d_model));
        if (auto* L = model.getLayerByName(p + "/ln3")) {
            L->inputs = {p + "/res2"};
            L->output = p + "/ln3_out";
            L->affine = true;
            L->use_bias = true;
            L->eps = 1e-5f;
            // Activer LayerNorm token-wise: normaliser chaque token sur d_model.
            L->in_features = d_model;
        }

        // MLP token-wise: Linear appliqué par token via LayerOps (seq_len).
        model.push(p + "/mlp_fc1", "Linear",
                   sat_mul(static_cast<size_t>(d_model), static_cast<size_t>(mlp_hidden)) + static_cast<size_t>(mlp_hidden));
        if (auto* L = model.getLayerByName(p + "/mlp_fc1")) {
            L->inputs = {p + "/ln3_out"};
            L->output = p + "/mlp_h";
            L->seq_len = q_len;
            L->in_features = d_model;
            L->out_features = mlp_hidden;
            L->use_bias = true;
        }

        model.push(p + "/mlp_act", "GELU", 0);
        if (auto* L = model.getLayerByName(p + "/mlp_act")) {
            L->inputs = {p + "/mlp_h"};
            L->output = p + "/mlp_h_act";
        }

        model.push(p + "/mlp_fc2", "Linear",
                   sat_mul(static_cast<size_t>(mlp_hidden), static_cast<size_t>(d_model)) + static_cast<size_t>(d_model));
        if (auto* L = model.getLayerByName(p + "/mlp_fc2")) {
            L->inputs = {p + "/mlp_h_act"};
            L->output = p + "/mlp_out";
            L->seq_len = q_len;
            L->in_features = mlp_hidden;
            L->out_features = d_model;
            L->use_bias = true;
        }

        model.push(p + "/add3", "Add", 0);
        if (auto* L = model.getLayerByName(p + "/add3")) {
            L->inputs = {p + "/res2", p + "/mlp_out"};
            L->output = p + "/out";
        }

        q = p + "/out";
    }

    model.push("sd3_5/out", "Identity", 0);
    if (auto* L = model.getLayerByName("sd3_5/out")) {
        L->inputs = {q};
        L->output = "x";
    }
}
