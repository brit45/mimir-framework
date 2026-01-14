#pragma once

#include "../Model.hpp"

#include <string>

// VAEModel (VAE-style autoencoder graph, float-only API)
// Output packs: [recon(image_dim), mu(latent_dim), logvar(latent_dim)]

class VAEModel : public Model {
public:
    struct Config {
        int image_w = 64;
        int image_h = 64;
        int image_c = 3;
        int latent_dim = 128;
        int hidden_dim = 1024;
    };

    VAEModel();
    void buildFromConfig(const Config& cfg);
    const Config& getConfig() const { return cfg_; }

    static void buildInto(Model& model, const Config& cfg);

private:
    Config cfg_;
};
