#pragma once

#include "Model.hpp"

// PatchDiscriminatorModel: small PatchGAN-like discriminator for adversarial image loss.
// Input: HWC float image vector in [-1,1]. Output: flattened patch logits.
class PatchDiscriminatorModel : public Model {
public:
    struct Config {
        int image_w = 64;
        int image_h = 64;
        int image_c = 3;
        int base_channels = 32;
        int num_down = 3;
    };

    PatchDiscriminatorModel();
    void buildFromConfig(const Config& cfg);
    static void buildInto(Model& model, const Config& cfg);

private:
    Config cfg_{};
};
