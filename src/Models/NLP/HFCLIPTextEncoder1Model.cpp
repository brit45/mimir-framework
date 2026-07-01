#include "HFCLIPTextEncoder1Model.hpp"

#include <algorithm>

HFCLIPTextEncoder1Model::HFCLIPTextEncoder1Model() {
    setModelName("HFCLIPTextEncoder1Model");
    setHasEncoder(false);
}

void HFCLIPTextEncoder1Model::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
}

void HFCLIPTextEncoder1Model::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("HFCLIPTextEncoder1Model");
    model.modelConfig["type"] = "hf_clip_text_encoder_1";

    const int vocab = std::max(2, cfg.vocab_size);
    const int pad_id = std::max(0, cfg.padding_idx);
    const int seq_len = std::max(1, cfg.seq_len);
    const int d_model = std::max(8, cfg.d_model);
    const int layers = std::max(1, cfg.num_layers);
    const int heads = std::max(1, cfg.num_heads);
    const int mlp_hidden = std::max(16, cfg.mlp_hidden);

    model.modelConfig["task"] = "hf_clip_text_encoder_1";
    model.modelConfig["vocab_size"] = vocab;
    model.modelConfig["padding_idx"] = pad_id;
    model.modelConfig["seq_len"] = seq_len;
    model.modelConfig["d_model"] = d_model;
    model.modelConfig["num_layers"] = layers;
    model.modelConfig["num_heads"] = heads;
    model.modelConfig["mlp_hidden"] = mlp_hidden;
    model.modelConfig["causal"] = cfg.causal;
    model.modelConfig["sequence_dim"] = seq_len * d_model;
    model.modelConfig["output_dim"] = seq_len * d_model;

    model.push("sdxl/text_encoder_1/tok_emb", "Embedding", static_cast<size_t>(vocab) * static_cast<size_t>(d_model));
    if (auto* E = model.getLayerByName("sdxl/text_encoder_1/tok_emb")) {
        E->inputs = {"__input__"};
        E->output = "sdxl/text_encoder_1/tok_emb_out";
        E->vocab_size = vocab;
        E->embed_dim = d_model;
        E->padding_idx = pad_id;
        E->seq_len = seq_len;
    }

    model.push("conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight", "Constant", static_cast<size_t>(seq_len) * static_cast<size_t>(d_model));
    if (auto* C = model.getLayerByName("conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight")) {
        C->inputs = {};
        C->output = "sdxl/text_encoder_1/pos_emb";
    }

    model.push("sdxl/text_encoder_1/add_pos", "Add", 0);
    if (auto* A = model.getLayerByName("sdxl/text_encoder_1/add_pos")) {
        A->inputs = {"sdxl/text_encoder_1/tok_emb_out", "sdxl/text_encoder_1/pos_emb"};
        A->output = "sdxl/text_encoder_1/in";
    }

    std::string x = "sdxl/text_encoder_1/in";
    const size_t attn_params = static_cast<size_t>(4) * static_cast<size_t>(d_model) * static_cast<size_t>(d_model)
                             + static_cast<size_t>(4) * static_cast<size_t>(d_model);

    for (int i = 0; i < layers; ++i) {
        const std::string p = "sdxl/text_encoder_1/block" + std::to_string(i + 1);

        model.push(p + "/ln1", "LayerNorm", static_cast<size_t>(2) * static_cast<size_t>(d_model));
        if (auto* L = model.getLayerByName(p + "/ln1")) {
            L->inputs = {x};
            L->output = p + "/ln1_out";
            L->affine = true;
            L->use_bias = true;
            L->eps = 1e-5f;
            L->in_features = d_model;
        }

        model.push(p + "/self_attn", "MultiHeadAttention", attn_params);
        if (auto* L = model.getLayerByName(p + "/self_attn")) {
            L->inputs = {p + "/ln1_out"};
            L->output = p + "/self_attn_out";
            L->seq_len = seq_len;
            L->embed_dim = d_model;
            L->num_heads = heads;
            L->causal = cfg.causal;
        }

        model.push(p + "/add1", "Add", 0);
        if (auto* L = model.getLayerByName(p + "/add1")) {
            L->inputs = {x, p + "/self_attn_out"};
            L->output = p + "/res1";
        }

        model.push(p + "/ln2", "LayerNorm", static_cast<size_t>(2) * static_cast<size_t>(d_model));
        if (auto* L = model.getLayerByName(p + "/ln2")) {
            L->inputs = {p + "/res1"};
            L->output = p + "/ln2_out";
            L->affine = true;
            L->use_bias = true;
            L->eps = 1e-5f;
            L->in_features = d_model;
        }

        model.push(p + "/mlp_fc1", "Linear", static_cast<size_t>(d_model) * static_cast<size_t>(mlp_hidden) + static_cast<size_t>(mlp_hidden));
        if (auto* L = model.getLayerByName(p + "/mlp_fc1")) {
            L->inputs = {p + "/ln2_out"};
            L->output = p + "/mlp_h";
            L->seq_len = seq_len;
            L->in_features = d_model;
            L->out_features = mlp_hidden;
            L->use_bias = true;
        }

        model.push(p + "/mlp_act", "GELU", 0);
        if (auto* L = model.getLayerByName(p + "/mlp_act")) {
            L->inputs = {p + "/mlp_h"};
            L->output = p + "/mlp_h_act";
        }

        model.push(p + "/mlp_fc2", "Linear", static_cast<size_t>(mlp_hidden) * static_cast<size_t>(d_model) + static_cast<size_t>(d_model));
        if (auto* L = model.getLayerByName(p + "/mlp_fc2")) {
            L->inputs = {p + "/mlp_h_act"};
            L->output = p + "/mlp_out";
            L->seq_len = seq_len;
            L->in_features = mlp_hidden;
            L->out_features = d_model;
            L->use_bias = true;
        }

        model.push(p + "/add2", "Add", 0);
        if (auto* L = model.getLayerByName(p + "/add2")) {
            L->inputs = {p + "/res1", p + "/mlp_out"};
            L->output = p + "/out";
        }

        x = p + "/out";
    }

    model.push("sdxl/text_encoder_1/final_ln", "LayerNorm", static_cast<size_t>(2) * static_cast<size_t>(d_model));
    if (auto* L = model.getLayerByName("sdxl/text_encoder_1/final_ln")) {
        L->inputs = {x};
        L->output = "x";
        L->affine = true;
        L->use_bias = true;
        L->eps = 1e-5f;
        L->in_features = d_model;
    }
}