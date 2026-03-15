#pragma once

#include "../Model.hpp"

#include <string>

// VAETextModel (text-only variational autoencoder)
// - Input: text_ids (int[seq_len])
// - Output packs: [logits(seq_len*vocab), mu(latent_dim), logvar(latent_dim), img_proj(proj_dim), text_proj(proj_dim)]
//   where latent_dim = latent_tokens * d_model
// - Intended training path: Model::trainStepVAEText with x empty; reconstruction loss is typically CE over tokens.

class VAETextModel : public Model {
public:
    struct Config {
        int vocab_size = 32000;
        int padding_idx = 0;

        int seq_len = 256;
        int d_model = 256;

        int num_layers = 4;
        int num_heads = 8;
        int mlp_hidden = 1024;

        int latent_tokens = 32; // latent_dim = latent_tokens * d_model
        int proj_dim = 256;

        // Si true, Reparameterize est stochastique pendant l'entraînement.
        bool stochastic_latent = true;

        float dropout = 0.0f;
    };

    VAETextModel();
    void buildFromConfig(const Config& cfg);
    const Config& getConfig() const { return cfg_; }

    static void buildInto(Model& model, const Config& cfg);
    static void buildDecoderInto(Model& model, const Config& cfg);

private:
    Config cfg_;
};
