#pragma once

#include "Model.hpp"

// VGG16FeatModel: small VGG-like feature extractor for perceptual loss.
// Input: HWC float image vector in [-1,1]. Output: concatenated GAP features across blocks.
class VGG16FeatModel : public Model {
public:
    struct Config {
        int image_w = 64;
        int image_h = 64;
        int image_c = 3;
        int base_channels = 8;
    };

    VGG16FeatModel();
    void buildFromConfig(const Config& cfg);
    static void buildInto(Model& model, const Config& cfg);

private:
    Config cfg_{};
};
