#include "CondDiffusionModel.hpp"

#include <algorithm>

CondDiffusionModel::CondDiffusionModel() {
    setModelName("CondDiffusionModel");
    setHasEncoder(true);
}

void CondDiffusionModel::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
}

void CondDiffusionModel::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("CondDiffusionModel");
    model.modelConfig["type"] = "cond_diffusion";

    const int prompt_dim = std::max(1, cfg.prompt_dim);
    const int lw = std::max(1, cfg.latent_w);
    const int lh = std::max(1, cfg.latent_h);
    const int lc = std::max(1, cfg.latent_c);
    const int time_dim = std::max(1, cfg.time_dim);
    const int hidden = std::max(64, cfg.hidden_dim);
    const float dropout = std::clamp(cfg.dropout, 0.0f, 0.95f);

    const int latent_dim = lw * lh * lc;
    const int in_dim = prompt_dim + time_dim + latent_dim;

    model.modelConfig["task"] = "cond_diffusion_eps_predictor";
    model.modelConfig["prompt_dim"] = prompt_dim;
    model.modelConfig["latent_w"] = lw;
    model.modelConfig["latent_h"] = lh;
    model.modelConfig["latent_c"] = lc;
    model.modelConfig["latent_dim"] = latent_dim;
    model.modelConfig["time_dim"] = time_dim;
    model.modelConfig["hidden_dim"] = hidden;
    model.modelConfig["dropout"] = dropout;
    model.modelConfig["input_dim"] = in_dim;
    model.modelConfig["output_dim"] = latent_dim;

    model.push("cdiff/raw_in", "Identity", 0);
    if (auto* L = model.getLayerByName("cdiff/raw_in")) {
        L->inputs = {"__input__"};
        L->output = "cdiff/in";
    }

    model.push("cdiff/fc1", "Linear", static_cast<size_t>(in_dim) * static_cast<size_t>(hidden) + static_cast<size_t>(hidden));
    if (auto* L = model.getLayerByName("cdiff/fc1")) {
        L->inputs = {"cdiff/in"};
        L->output = "cdiff/h1";
        L->in_features = in_dim;
        L->out_features = hidden;
        L->use_bias = true;
    }

    model.push("cdiff/act1", "GELU", 0);
    if (auto* L = model.getLayerByName("cdiff/act1")) {
        L->inputs = {"cdiff/h1"};
        L->output = "cdiff/h1_act";
    }

    if (dropout > 0.0f) {
        model.push("cdiff/drop", "Dropout", 0);
        if (auto* L = model.getLayerByName("cdiff/drop")) {
            L->inputs = {"cdiff/h1_act"};
            L->output = "cdiff/h1_drop";
            L->dropout_p = dropout;
        }
    }

    const std::string mid = (dropout > 0.0f) ? "cdiff/h1_drop" : "cdiff/h1_act";

    model.push("cdiff/out", "Linear", static_cast<size_t>(hidden) * static_cast<size_t>(latent_dim) + static_cast<size_t>(latent_dim));
    if (auto* L = model.getLayerByName("cdiff/out")) {
        L->inputs = {mid};
        L->output = "x";
        L->in_features = hidden;
        L->out_features = latent_dim;
        L->use_bias = true;
    }
}
