#include "VAEModel.hpp"

#include <algorithm>

VAEModel::VAEModel() {
    setModelName("VAEModel");
    setHasEncoder(false);
}

void VAEModel::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
}

void VAEModel::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("VAEModel");
    model.modelConfig["type"] = "vae";

    const int W = std::max(1, cfg.image_w);
    const int H = std::max(1, cfg.image_h);
    const int C = std::max(1, cfg.image_c);
    const int image_dim = W * H * C;
    const int latent = std::max(1, cfg.latent_dim);
    const int hidden = std::max(16, cfg.hidden_dim);

    model.modelConfig["task"] = "vae_autoencoder";
    model.modelConfig["image_w"] = W;
    model.modelConfig["image_h"] = H;
    model.modelConfig["image_c"] = C;
    model.modelConfig["image_dim"] = image_dim;
    model.modelConfig["latent_dim"] = latent;
    model.modelConfig["hidden_dim"] = hidden;
    model.modelConfig["input_dim"] = image_dim;
    model.modelConfig["output_dim"] = image_dim + 2 * latent;

    model.push("vae/raw_in", "Identity", 0);
    if (auto* L = model.getLayerByName("vae/raw_in")) {
        L->inputs = {"__input__"};
        L->output = "vae/in";
    }

    // Encoder (MLP)
    model.push("vae/enc_fc1", "Linear", static_cast<size_t>(image_dim) * static_cast<size_t>(hidden) + static_cast<size_t>(hidden));
    if (auto* L = model.getLayerByName("vae/enc_fc1")) {
        L->inputs = {"vae/in"};
        L->output = "vae/enc_h";
        L->in_features = image_dim;
        L->out_features = hidden;
        L->use_bias = true;
    }

    model.push("vae/enc_act1", "GELU", 0);
    if (auto* L = model.getLayerByName("vae/enc_act1")) {
        L->inputs = {"vae/enc_h"};
        L->output = "vae/enc_h_act";
    }

    model.push("vae/mu", "Linear", static_cast<size_t>(hidden) * static_cast<size_t>(latent) + static_cast<size_t>(latent));
    if (auto* L = model.getLayerByName("vae/mu")) {
        L->inputs = {"vae/enc_h_act"};
        L->output = "vae/mu";
        L->in_features = hidden;
        L->out_features = latent;
        L->use_bias = true;
    }

    model.push("vae/logvar", "Linear", static_cast<size_t>(hidden) * static_cast<size_t>(latent) + static_cast<size_t>(latent));
    if (auto* L = model.getLayerByName("vae/logvar")) {
        L->inputs = {"vae/enc_h_act"};
        L->output = "vae/logvar";
        L->in_features = hidden;
        L->out_features = latent;
        L->use_bias = true;
    }

    // Deterministic latent for graph: use mu (reparameterization is handled outside)
    model.push("vae/latent_act", "Identity", 0);
    if (auto* L = model.getLayerByName("vae/latent_act")) {
        L->inputs = {"vae/mu"};
        L->output = "vae/z";
    }

    // Decoder (MLP)
    model.push("vae/dec_fc1", "Linear", static_cast<size_t>(latent) * static_cast<size_t>(hidden) + static_cast<size_t>(hidden));
    if (auto* L = model.getLayerByName("vae/dec_fc1")) {
        L->inputs = {"vae/z"};
        L->output = "vae/dec_h";
        L->in_features = latent;
        L->out_features = hidden;
        L->use_bias = true;
    }

    model.push("vae/dec_act1", "GELU", 0);
    if (auto* L = model.getLayerByName("vae/dec_act1")) {
        L->inputs = {"vae/dec_h"};
        L->output = "vae/dec_h_act";
    }

    model.push("vae/recon", "Linear", static_cast<size_t>(hidden) * static_cast<size_t>(image_dim) + static_cast<size_t>(image_dim));
    if (auto* L = model.getLayerByName("vae/recon")) {
        L->inputs = {"vae/dec_h_act"};
        L->output = "vae/recon_pre";
        L->in_features = hidden;
        L->out_features = image_dim;
        L->use_bias = true;
    }

    model.push("vae/recon_tanh", "Tanh", 0);
    if (auto* L = model.getLayerByName("vae/recon_tanh")) {
        L->inputs = {"vae/recon_pre"};
        L->output = "vae/recon";
    }

    // Pack output: recon || mu || logvar
    model.push("vae/out_concat", "Concat", 0);
    if (auto* L = model.getLayerByName("vae/out_concat")) {
        L->inputs = {"vae/recon", "vae/mu", "vae/logvar"};
        L->output = "x";
        L->concat_axis = 0;
    }
}
