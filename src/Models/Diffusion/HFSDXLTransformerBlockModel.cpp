#include "HFSDXLTransformerBlockModel.hpp"

#include <algorithm>

namespace {
static size_t sat_mul_local(size_t a, size_t b) {
    if (a == 0 || b == 0) return 0;
    if (a > (static_cast<size_t>(-1) / b)) return static_cast<size_t>(-1);
    return a * b;
}
}

HFSDXLTransformerBlockModel::HFSDXLTransformerBlockModel() {
    setModelName("HFSDXLTransformerBlockModel");
    setHasEncoder(true);
}

void HFSDXLTransformerBlockModel::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
}

void HFSDXLTransformerBlockModel::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("HFSDXLTransformerBlockModel");
    model.modelConfig["type"] = "hf_sdxl_transformer_block";
    model.modelConfig["task"] = "hf_sdxl_transformer_block";

    const int q_len = std::max(1, cfg.q_len);
    const int kv_len = std::max(1, cfg.kv_len);
    const int d_model = std::max(1, cfg.d_model);
    const int context_dim = std::max(1, cfg.context_dim);
    const int num_heads = std::max(1, cfg.num_heads);
    const int ff_hidden = std::max(1, cfg.ff_hidden);

    const int q_dim = q_len * d_model;
    const int kv_dim = kv_len * context_dim;
    const int input_dim = q_dim + kv_dim;

    model.modelConfig["q_len"] = q_len;
    model.modelConfig["kv_len"] = kv_len;
    model.modelConfig["d_model"] = d_model;
    model.modelConfig["context_dim"] = context_dim;
    model.modelConfig["num_heads"] = num_heads;
    model.modelConfig["ff_hidden"] = ff_hidden;
    model.modelConfig["input_dim"] = input_dim;
    model.modelConfig["output_dim"] = q_dim;

    model.push("sdxl/transformer_block/in", "Identity", 0);
    if (auto* L = model.getLayerByName("sdxl/transformer_block/in")) {
        L->inputs = {"__input__"};
        L->output = "sdxl/transformer_block/packed";
    }

    model.push("sdxl/transformer_block/split", "Split", 0);
    if (auto* L = model.getLayerByName("sdxl/transformer_block/split")) {
        L->inputs = {"sdxl/transformer_block/packed"};
        L->output = "sdxl/transformer_block/split";
        L->split_sizes = {q_dim, kv_dim};
        L->split_axis = 0;
    }

    const std::string q = "sdxl/transformer_block/split_0";
    const std::string kv = "sdxl/transformer_block/split_1";

    auto add_ln = [&](const std::string& name, const std::string& input, const std::string& output) {
        model.push(name, "LayerNorm", static_cast<size_t>(2) * static_cast<size_t>(d_model));
        if (auto* L = model.getLayerByName(name)) {
            L->inputs = {input};
            L->output = output;
            L->affine = true;
            L->use_bias = true;
            L->eps = 1e-5f;
            L->in_features = d_model;
        }
    };

    add_ln("sdxl/transformer_block/norm1", q, "sdxl/transformer_block/norm1_out");

    size_t self_attn_params = static_cast<size_t>(4) * sat_mul_local(static_cast<size_t>(d_model), static_cast<size_t>(d_model));
    if (cfg.self_attn_qkv_bias) self_attn_params += static_cast<size_t>(3) * static_cast<size_t>(d_model);
    if (cfg.self_attn_out_bias) self_attn_params += static_cast<size_t>(d_model);
    model.push("sdxl/transformer_block/attn1", "SelfAttention", self_attn_params);
    if (auto* L = model.getLayerByName("sdxl/transformer_block/attn1")) {
        L->inputs = {"sdxl/transformer_block/norm1_out"};
        L->output = "sdxl/transformer_block/attn1_out";
        L->seq_len = q_len;
        L->embed_dim = d_model;
        L->num_heads = num_heads;
        L->causal = false;
    }

    model.push("sdxl/transformer_block/add1", "Add", 0);
    if (auto* L = model.getLayerByName("sdxl/transformer_block/add1")) {
        L->inputs = {q, "sdxl/transformer_block/attn1_out"};
        L->output = "sdxl/transformer_block/res1";
    }

    add_ln("sdxl/transformer_block/norm2", "sdxl/transformer_block/res1", "sdxl/transformer_block/norm2_out");

    size_t cross_attn_params = sat_mul_local(static_cast<size_t>(d_model), static_cast<size_t>(d_model));
    cross_attn_params += sat_mul_local(static_cast<size_t>(context_dim), static_cast<size_t>(2 * d_model));
    cross_attn_params += sat_mul_local(static_cast<size_t>(d_model), static_cast<size_t>(d_model));
    if (cfg.cross_attn_out_bias) cross_attn_params += static_cast<size_t>(d_model);
    model.push("sdxl/transformer_block/attn2", "CrossAttention", cross_attn_params);
    if (auto* L = model.getLayerByName("sdxl/transformer_block/attn2")) {
        L->inputs = {"sdxl/transformer_block/norm2_out", kv};
        L->output = "sdxl/transformer_block/attn2_out";
        L->embed_dim = d_model;
        L->in_features = context_dim;
        L->num_heads = num_heads;
        L->causal = false;
    }

    model.push("sdxl/transformer_block/add2", "Add", 0);
    if (auto* L = model.getLayerByName("sdxl/transformer_block/add2")) {
        L->inputs = {"sdxl/transformer_block/res1", "sdxl/transformer_block/attn2_out"};
        L->output = "sdxl/transformer_block/res2";
    }

    add_ln("sdxl/transformer_block/norm3", "sdxl/transformer_block/res2", "sdxl/transformer_block/norm3_out");

    model.push("sdxl/transformer_block/ff_proj", "Linear",
               sat_mul_local(static_cast<size_t>(d_model), static_cast<size_t>(2 * ff_hidden)) + static_cast<size_t>(2 * ff_hidden));
    if (auto* L = model.getLayerByName("sdxl/transformer_block/ff_proj")) {
        L->inputs = {"sdxl/transformer_block/norm3_out"};
        L->output = "sdxl/transformer_block/ff_proj_out";
        L->seq_len = q_len;
        L->in_features = d_model;
        L->out_features = 2 * ff_hidden;
        L->use_bias = true;
    }

    model.push("sdxl/transformer_block/geglu", "GEGLU", 0);
    if (auto* L = model.getLayerByName("sdxl/transformer_block/geglu")) {
        L->inputs = {"sdxl/transformer_block/ff_proj_out"};
        L->output = "sdxl/transformer_block/geglu_out";
        L->seq_len = q_len;
        L->out_features = ff_hidden;
    }

    model.push("sdxl/transformer_block/ff_out", "Linear",
               sat_mul_local(static_cast<size_t>(ff_hidden), static_cast<size_t>(d_model)) + static_cast<size_t>(d_model));
    if (auto* L = model.getLayerByName("sdxl/transformer_block/ff_out")) {
        L->inputs = {"sdxl/transformer_block/geglu_out"};
        L->output = "sdxl/transformer_block/ff_out_out";
        L->seq_len = q_len;
        L->in_features = ff_hidden;
        L->out_features = d_model;
        L->use_bias = true;
    }

    model.push("sdxl/transformer_block/add3", "Add", 0);
    if (auto* L = model.getLayerByName("sdxl/transformer_block/add3")) {
        L->inputs = {"sdxl/transformer_block/res2", "sdxl/transformer_block/ff_out_out"};
        L->output = "sdxl/transformer_block/out";
    }

    model.push("sdxl/transformer_block/out_id", "Identity", 0);
    if (auto* L = model.getLayerByName("sdxl/transformer_block/out_id")) {
        L->inputs = {"sdxl/transformer_block/out"};
        L->output = "x";
    }
}