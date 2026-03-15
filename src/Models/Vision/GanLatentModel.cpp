#include "GanLatentModel.hpp"

#include <algorithm>

GanLatentModel::GanLatentModel() {
    setModelName("GanLatentModel");
    setHasEncoder(true);
}

void GanLatentModel::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
}

void GanLatentModel::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("GanLatentModel");
    model.modelConfig["type"] = "gan_latent";

    const int prompt_dim = std::max(1, cfg.prompt_dim);
    const int noise_dim = std::max(1, cfg.noise_dim);
    const int latent_dim = std::max(1, cfg.latent_dim);
    const int hidden = std::max(32, cfg.hidden_dim);
    const int n_hidden = std::max(1, cfg.num_hidden_layers);
    const float dropout = std::clamp(cfg.dropout, 0.0f, 0.95f);

    const int in_dim = prompt_dim + noise_dim;

    model.modelConfig["task"] = "gan_latent_generator";
    model.modelConfig["prompt_dim"] = prompt_dim;
    model.modelConfig["noise_dim"] = noise_dim;
    model.modelConfig["latent_dim"] = latent_dim;
    model.modelConfig["hidden_dim"] = hidden;
    model.modelConfig["num_hidden_layers"] = n_hidden;
    model.modelConfig["dropout"] = dropout;
    model.modelConfig["input_dim"] = in_dim;
    model.modelConfig["output_dim"] = latent_dim;

    model.push("gan/raw_in", "Identity", 0);
    if (auto* L = model.getLayerByName("gan/raw_in")) {
        L->inputs = {"__input__"};
        L->output = "gan/in";
    }

    std::string cur = "gan/in";
    for (int i = 0; i < n_hidden; ++i) {
        const std::string fc = "gan/fc" + std::to_string(i + 1);
        const std::string act = "gan/act" + std::to_string(i + 1);
        const std::string drop = "gan/drop" + std::to_string(i + 1);

        model.push(fc, "Linear",
                   static_cast<size_t>((i == 0 ? in_dim : hidden)) * static_cast<size_t>(hidden) + static_cast<size_t>(hidden));
        if (auto* L = model.getLayerByName(fc)) {
            L->inputs = {cur};
            L->output = fc + "/out";
            L->in_features = (i == 0 ? in_dim : hidden);
            L->out_features = hidden;
            L->use_bias = true;
        }

        model.push(act, "GELU", 0);
        if (auto* L = model.getLayerByName(act)) {
            L->inputs = {fc + "/out"};
            L->output = act + "/out";
        }

        cur = act + "/out";

        if (dropout > 0.0f) {
            model.push(drop, "Dropout", 0);
            if (auto* L = model.getLayerByName(drop)) {
                L->inputs = {cur};
                L->output = drop + "/out";
                L->dropout_p = dropout;
            }
            cur = drop + "/out";
        }
    }

    model.push("gan/out", "Linear",
               static_cast<size_t>(hidden) * static_cast<size_t>(latent_dim) + static_cast<size_t>(latent_dim));
    if (auto* L = model.getLayerByName("gan/out")) {
        L->inputs = {cur};
        L->output = "x";
        L->in_features = hidden;
        L->out_features = latent_dim;
        L->use_bias = true;
    }
}
