#pragma once

#include "../Model.hpp"

// UNetModel (simplified, float-only API)
// Input: float[image_dim]
// Output: float[image_dim]

class UNetModel : public Model {
public:
    struct Config {
        int image_w = 64;
        int image_h = 64;
        int image_c = 3;
        int base_channels = 32;
        int depth = 3; // number of down/upsampling steps
    };

    UNetModel();
    void buildFromConfig(const Config& cfg);
    const Config& getConfig() const { return cfg_; }

    static void buildInto(Model& model, const Config& cfg);

private:
    Config cfg_;
};
