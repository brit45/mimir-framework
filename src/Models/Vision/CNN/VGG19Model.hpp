#pragma once

#include "../Model.hpp"

// VGG19Model (simplified, float-only API)
// Input: float[image_dim]
// Output: float[num_classes]

class VGG19Model : public Model {
public:
    struct Config {
        int image_w = 64;
        int image_h = 64;
        int image_c = 3;
        int base_channels = 32;
        int num_classes = 1000;
        int fc_hidden = 512;
    };

    VGG19Model();
    void buildFromConfig(const Config& cfg);
    const Config& getConfig() const { return cfg_; }

    static void buildInto(Model& model, const Config& cfg);

private:
    Config cfg_;
};
