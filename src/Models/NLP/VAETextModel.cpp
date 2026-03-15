#include "VAETextModel.hpp"

#include <algorithm>

VAETextModel::VAETextModel() {
    setModelName("VAETextModel");
    setHasEncoder(false);
}

void VAETextModel::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
}

void VAETextModel::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("VAETextModel");
    model.modelConfig["type"] = "vae_text";

    const int vocab = std::max(2, cfg.vocab_size);
    const int pad_id = std::max(0, cfg.padding_idx);

    const int seq_len = std::max(1, cfg.seq_len);
    const int d_model = std::max(8, cfg.d_model);

    const int layers = std::max(0, cfg.num_layers);
    const int heads = std::max(1, cfg.num_heads);
    const int mlp_hidden = std::max(16, cfg.mlp_hidden);

    const int latent_tokens = std::max(1, cfg.latent_tokens);
    const int latent_dim = std::max(1, latent_tokens * d_model);

    const int proj_dim = std::max(1, cfg.proj_dim);

    model.modelConfig["task"] = "vae_text";
    model.modelConfig["vocab_size"] = vocab;
    model.modelConfig["padding_idx"] = pad_id;
    model.modelConfig["seq_len"] = seq_len;
    model.modelConfig["d_model"] = d_model;
    model.modelConfig["num_layers"] = layers;
    model.modelConfig["num_heads"] = heads;
    model.modelConfig["mlp_hidden"] = mlp_hidden;
    model.modelConfig["latent_tokens"] = latent_tokens;
    model.modelConfig["latent_dim"] = latent_dim;
    model.modelConfig["proj_dim"] = proj_dim;
    model.modelConfig["stochastic_latent"] = cfg.stochastic_latent;
    model.modelConfig["dropout"] = cfg.dropout;

    // Reconstruction as token logits (seq_len * vocab)
    const int logits_dim = seq_len * vocab;
    model.modelConfig["image_dim"] = logits_dim;
    model.modelConfig["target_tensor"] = "vae_text/target"; // kept for optional alignment heads
    model.modelConfig["output_dim"] = logits_dim + 2 * latent_dim + 2 * proj_dim;

    // ---------------------------------------------------------------------
    // text_ids -> embedding
    // ---------------------------------------------------------------------
    model.push("vae_text/tok_emb", "Embedding",
               static_cast<size_t>(vocab) * static_cast<size_t>(d_model));
    if (auto* E = model.getLayerByName("vae_text/tok_emb")) {
        E->inputs = {"text_ids"};
        E->output = "vae_text/tok_emb_out";
        E->vocab_size = vocab;
        E->embed_dim = d_model;
        E->padding_idx = pad_id;
        E->seq_len = seq_len;
    }

    // Target tap (token embeddings) for optional alignment heads
    model.push("vae_text/target_id", "Identity", 0);
    if (auto* L = model.getLayerByName("vae_text/target_id")) {
        L->inputs = {"vae_text/tok_emb_out"};
        L->output = "vae_text/target";
    }

    std::string h = "vae_text/tok_emb_out";

    const size_t attn_params = static_cast<size_t>(4) * static_cast<size_t>(d_model) * static_cast<size_t>(d_model);

    // ---------------------------------------------------------------------
    // Mini Transformer encoder on token embeddings
    // ---------------------------------------------------------------------
    for (int i = 0; i < layers; ++i) {
        const std::string p = "vae_text/enc/block" + std::to_string(i + 1);

        model.push(p + "/ln1", "LayerNorm", static_cast<size_t>(2) * static_cast<size_t>(d_model));
        if (auto* L = model.getLayerByName(p + "/ln1")) {
            L->inputs = {h};
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
            L->causal = false;
        }

        model.push(p + "/add1", "Add", 0);
        if (auto* L = model.getLayerByName(p + "/add1")) {
            L->inputs = {h, p + "/self_attn_out"};
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

        model.push(p + "/mlp_fc1", "Linear",
                   static_cast<size_t>(d_model) * static_cast<size_t>(mlp_hidden) + static_cast<size_t>(mlp_hidden));
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

        model.push(p + "/mlp_fc2", "Linear",
                   static_cast<size_t>(mlp_hidden) * static_cast<size_t>(d_model) + static_cast<size_t>(d_model));
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

        h = p + "/out";
    }

    // ---------------------------------------------------------------------
    // Pool tokens -> d_model, then mu/logvar -> z
    // ---------------------------------------------------------------------
    model.push("vae_text/enc/pool", "TokenMeanPool", 0);
    if (auto* P = model.getLayerByName("vae_text/enc/pool")) {
        P->inputs = {h};
        P->output = "vae_text/enc/pooled";
        P->seq_len = seq_len;
        P->embed_dim = d_model;
    }

    model.push("vae_text/mu", "Linear",
               static_cast<size_t>(d_model) * static_cast<size_t>(latent_dim) + static_cast<size_t>(latent_dim));
    if (auto* L = model.getLayerByName("vae_text/mu")) {
        L->inputs = {"vae_text/enc/pooled"};
        L->output = "vae_text/mu";
        L->in_features = d_model;
        L->out_features = latent_dim;
        L->use_bias = true;
    }

    model.push("vae_text/logvar", "Linear",
               static_cast<size_t>(d_model) * static_cast<size_t>(latent_dim) + static_cast<size_t>(latent_dim));
    if (auto* L = model.getLayerByName("vae_text/logvar")) {
        L->inputs = {"vae_text/enc/pooled"};
        L->output = "vae_text/logvar";
        L->in_features = d_model;
        L->out_features = latent_dim;
        L->use_bias = true;
    }

    model.push("vae_text/reparam", "Reparameterize", 0);
    if (auto* R = model.getLayerByName("vae_text/reparam")) {
        R->inputs = {"vae_text/mu", "vae_text/logvar"};
        R->output = "vae_text/z";
    }

    // ---------------------------------------------------------------------
    // Decoder: z -> token states -> transformer -> logits
    // ---------------------------------------------------------------------
    const int tok_dim = seq_len * d_model;
    model.push("vae_text/dec/init", "Linear",
               static_cast<size_t>(latent_dim) * static_cast<size_t>(tok_dim) + static_cast<size_t>(tok_dim));
    if (auto* L = model.getLayerByName("vae_text/dec/init")) {
        L->inputs = {"vae_text/z"};
        L->output = "vae_text/dec/h0";
        L->in_features = latent_dim;
        L->out_features = tok_dim;
        L->use_bias = true;
    }

    std::string dh = "vae_text/dec/h0";
    for (int i = 0; i < layers; ++i) {
        const std::string p = "vae_text/dec/block" + std::to_string(i + 1);

        model.push(p + "/ln1", "LayerNorm", static_cast<size_t>(2) * static_cast<size_t>(d_model));
        if (auto* L = model.getLayerByName(p + "/ln1")) {
            L->inputs = {dh};
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
            L->causal = false;
        }

        model.push(p + "/add1", "Add", 0);
        if (auto* L = model.getLayerByName(p + "/add1")) {
            L->inputs = {dh, p + "/self_attn_out"};
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

        model.push(p + "/mlp_fc1", "Linear",
                   static_cast<size_t>(d_model) * static_cast<size_t>(mlp_hidden) + static_cast<size_t>(mlp_hidden));
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

        model.push(p + "/mlp_fc2", "Linear",
                   static_cast<size_t>(mlp_hidden) * static_cast<size_t>(d_model) + static_cast<size_t>(d_model));
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

        dh = p + "/out";
    }

    model.push("vae_text/dec/lm_head", "Linear",
               static_cast<size_t>(d_model) * static_cast<size_t>(vocab) + static_cast<size_t>(vocab));
    if (auto* L = model.getLayerByName("vae_text/dec/lm_head")) {
        L->inputs = {dh};
        L->output = "vae_text/logits";
        L->seq_len = seq_len;
        L->in_features = d_model;
        L->out_features = vocab;
        L->use_bias = true;
    }

    // ---------------------------------------------------------------------
    // Projections for alignment (optional loss)
    // ---------------------------------------------------------------------
    model.push("vae_text/dec/pool", "TokenMeanPool", 0);
    if (auto* P = model.getLayerByName("vae_text/dec/pool")) {
        P->inputs = {dh};
        P->output = "vae_text/dec/pooled";
        P->seq_len = seq_len;
        P->embed_dim = d_model;
    }

    model.push("vae_text/img_proj", "Linear",
               static_cast<size_t>(d_model) * static_cast<size_t>(proj_dim) + static_cast<size_t>(proj_dim));
    if (auto* L = model.getLayerByName("vae_text/img_proj")) {
        L->inputs = {"vae_text/dec/pooled"};
        L->output = "vae_text/img_proj";
        L->in_features = d_model;
        L->out_features = proj_dim;
        L->use_bias = true;
    }

    model.push("vae_text/tgt_pool", "TokenMeanPool", 0);
    if (auto* P = model.getLayerByName("vae_text/tgt_pool")) {
        P->inputs = {"vae_text/target"};
        P->output = "vae_text/tgt_pooled";
        P->seq_len = seq_len;
        P->embed_dim = d_model;
    }

    model.push("vae_text/text_proj", "Linear",
               static_cast<size_t>(d_model) * static_cast<size_t>(proj_dim) + static_cast<size_t>(proj_dim));
    if (auto* L = model.getLayerByName("vae_text/text_proj")) {
        L->inputs = {"vae_text/tgt_pooled"};
        L->output = "vae_text/text_proj";
        L->in_features = d_model;
        L->out_features = proj_dim;
        L->use_bias = true;
    }

    // ---------------------------------------------------------------------
    // Pack output: logits || mu || logvar || img_proj || text_proj
    // ---------------------------------------------------------------------
    model.push("vae_text/out_pack", "Concat", 0);
    if (auto* L = model.getLayerByName("vae_text/out_pack")) {
        L->inputs = {"vae_text/logits", "vae_text/mu", "vae_text/logvar", "vae_text/img_proj", "vae_text/text_proj"};
        L->output = "x";
        L->concat_axis = 0;
    }
}

void VAETextModel::buildDecoderInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("VAETextModel");
    model.modelConfig["type"] = "vae_text_decode";

    const int vocab = std::max(2, cfg.vocab_size);
    const int seq_len = std::max(1, cfg.seq_len);
    const int d_model = std::max(8, cfg.d_model);

    const int layers = std::max(0, cfg.num_layers);
    const int heads = std::max(1, cfg.num_heads);
    const int mlp_hidden = std::max(16, cfg.mlp_hidden);

    const int latent_tokens = std::max(1, cfg.latent_tokens);
    const int latent_dim = std::max(1, latent_tokens * d_model);

    model.modelConfig["task"] = "vae_text_decoder";
    model.modelConfig["vocab_size"] = vocab;
    model.modelConfig["seq_len"] = seq_len;
    model.modelConfig["d_model"] = d_model;
    model.modelConfig["num_layers"] = layers;
    model.modelConfig["num_heads"] = heads;
    model.modelConfig["mlp_hidden"] = mlp_hidden;
    model.modelConfig["latent_tokens"] = latent_tokens;
    model.modelConfig["latent_dim"] = latent_dim;

    const int logits_dim = seq_len * vocab;
    model.modelConfig["input_dim"] = latent_dim;
    model.modelConfig["image_dim"] = logits_dim;
    model.modelConfig["output_dim"] = logits_dim;

    const size_t attn_params = static_cast<size_t>(4) * static_cast<size_t>(d_model) * static_cast<size_t>(d_model);

    // Input latent vector -> vae_text/z
    model.push("vae_text/raw_z", "Identity", 0);
    if (auto* L = model.getLayerByName("vae_text/raw_z")) {
        L->inputs = {"__input__"};
        L->output = "vae_text/z";
    }

    // Decoder: z -> token states -> transformer -> logits
    const int tok_dim = seq_len * d_model;
    model.push("vae_text/dec/init", "Linear",
               static_cast<size_t>(latent_dim) * static_cast<size_t>(tok_dim) + static_cast<size_t>(tok_dim));
    if (auto* L = model.getLayerByName("vae_text/dec/init")) {
        L->inputs = {"vae_text/z"};
        L->output = "vae_text/dec/h0";
        L->in_features = latent_dim;
        L->out_features = tok_dim;
        L->use_bias = true;
    }

    std::string dh = "vae_text/dec/h0";
    for (int i = 0; i < layers; ++i) {
        const std::string p = "vae_text/dec/block" + std::to_string(i + 1);

        model.push(p + "/ln1", "LayerNorm", static_cast<size_t>(2) * static_cast<size_t>(d_model));
        if (auto* L = model.getLayerByName(p + "/ln1")) {
            L->inputs = {dh};
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
            L->causal = false;
        }

        model.push(p + "/add1", "Add", 0);
        if (auto* L = model.getLayerByName(p + "/add1")) {
            L->inputs = {dh, p + "/self_attn_out"};
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

        model.push(p + "/mlp_fc1", "Linear",
                   static_cast<size_t>(d_model) * static_cast<size_t>(mlp_hidden) + static_cast<size_t>(mlp_hidden));
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

        model.push(p + "/mlp_fc2", "Linear",
                   static_cast<size_t>(mlp_hidden) * static_cast<size_t>(d_model) + static_cast<size_t>(d_model));
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

        dh = p + "/out";
    }

    model.push("vae_text/dec/lm_head", "Linear",
               static_cast<size_t>(d_model) * static_cast<size_t>(vocab) + static_cast<size_t>(vocab));
    if (auto* L = model.getLayerByName("vae_text/dec/lm_head")) {
        L->inputs = {dh};
        L->output = "x";
        L->seq_len = seq_len;
        L->in_features = d_model;
        L->out_features = vocab;
        L->use_bias = true;
    }
}
