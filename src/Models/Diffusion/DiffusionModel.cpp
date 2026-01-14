#include "DiffusionModel.hpp"

#include <algorithm>

DiffusionModel::DiffusionModel() {
    setModelName("DiffusionModel");
    setHasEncoder(false);
}

void DiffusionModel::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
}

void DiffusionModel::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("DiffusionModel");
    model.modelConfig["type"] = "diffusion";

    const int W = std::max(1, cfg.image_w);
    const int H = std::max(1, cfg.image_h);
    const int C = std::max(1, cfg.image_c);
    const int time_dim = std::max(1, cfg.time_dim);
    const int hidden = std::max(64, cfg.hidden_dim);
    const int image_dim = W * H * C;
    const int in_dim = time_dim + image_dim;

    model.modelConfig["task"] = "diffusion_eps_predictor";
    model.modelConfig["image_w"] = W;
    model.modelConfig["image_h"] = H;
    model.modelConfig["image_c"] = C;
    model.modelConfig["image_dim"] = image_dim;
    model.modelConfig["time_dim"] = time_dim;
    model.modelConfig["hidden_dim"] = hidden;
    model.modelConfig["input_dim"] = in_dim;
    model.modelConfig["output_dim"] = image_dim;

    model.push("diff/raw_in", "Identity", 0);
    if (auto* L = model.getLayerByName("diff/raw_in")) {
        L->inputs = {"__input__"};
        L->output = "diff/in";
    }

    model.push("diff/fc1", "Linear", static_cast<size_t>(in_dim) * static_cast<size_t>(hidden) + static_cast<size_t>(hidden));
    if (auto* L = model.getLayerByName("diff/fc1")) {
        L->inputs = {"diff/in"};
        L->output = "diff/h1";
        L->in_features = in_dim;
        L->out_features = hidden;
        L->use_bias = true;
    }

    model.push("diff/act1", "GELU", 0);
    if (auto* L = model.getLayerByName("diff/act1")) {
        L->inputs = {"diff/h1"};
        L->output = "diff/h1_act";
    }

    if (cfg.dropout > 0.0f) {
        model.push("diff/drop", "Dropout", 0);
        if (auto* L = model.getLayerByName("diff/drop")) {
            L->inputs = {"diff/h1_act"};
            L->output = "diff/h1_drop";
            L->dropout_p = std::clamp(cfg.dropout, 0.0f, 0.95f);
        }
    }

    const std::string mid = (cfg.dropout > 0.0f) ? "diff/h1_drop" : "diff/h1_act";

    model.push("diff/out", "Linear", static_cast<size_t>(hidden) * static_cast<size_t>(image_dim) + static_cast<size_t>(image_dim));
    if (auto* L = model.getLayerByName("diff/out")) {
        L->inputs = {mid};
        L->output = "x";
        L->in_features = hidden;
        L->out_features = image_dim;
        L->use_bias = true;
    }
}
