#pragma once

#include "../../Model.hpp"

class HFVaeDecoderModel : public Model {
public:
    struct Config {
        int image_w = 512;
        int image_h = 512;
        int image_c = 3;
        int latent_w = 64;
        int latent_h = 64;
        int latent_c = 4;
        int num_heads = 1;
        int norm_groups = 32;
    };

    HFVaeDecoderModel();
    void buildFromConfig(const Config& cfg);
    static void buildInto(Model& model, const Config& cfg);

private:
    Config cfg_;
};