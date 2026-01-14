#pragma once

#include "../Model.hpp"

// ResNetModel (simplified ResNet-18-like, float-only API)
// Input: float[image_dim]
// Output: float[num_classes]

class ResNetModel : public Model {
public:
    struct Config {
        int image_w = 64;
        int image_h = 64;
        int image_c = 3;
        int base_channels = 32;
        int num_classes = 1000;
        int blocks1 = 2;
        int blocks2 = 2;
        int blocks3 = 2;
        int blocks4 = 2;
    };

    ResNetModel();
    void buildFromConfig(const Config& cfg);
    const Config& getConfig() const { return cfg_; }

    static void buildInto(Model& model, const Config& cfg);

private:
    Config cfg_;
};
