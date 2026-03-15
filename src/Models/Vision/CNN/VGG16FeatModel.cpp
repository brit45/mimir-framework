#include "VGG16FeatModel.hpp"

#include <algorithm>
#include <stdexcept>

static inline int conv_out(int in, int k, int s, int p) {
    return (in + 2 * p - k) / s + 1;
}

VGG16FeatModel::VGG16FeatModel() {
    setModelName("VGG16FeatModel");
    setHasEncoder(false);
}

void VGG16FeatModel::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
}

void VGG16FeatModel::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("VGG16FeatModel");
    model.modelConfig["type"] = "vgg16_feat";

    const int W = std::max(1, cfg.image_w);
    const int H = std::max(1, cfg.image_h);
    const int C = std::max(1, cfg.image_c);
    const int base = std::max(4, cfg.base_channels);

    const int image_dim = W * H * C;
    model.modelConfig["task"] = "perceptual_features";
    model.modelConfig["image_w"] = W;
    model.modelConfig["image_h"] = H;
    model.modelConfig["image_c"] = C;
    model.modelConfig["base_channels"] = base;
    model.modelConfig["input_dim"] = image_dim;

    auto sat_mul = [](size_t a, size_t b) -> size_t {
        if (a == 0 || b == 0) return 0;
        if (a > (static_cast<size_t>(-1) / b)) return static_cast<size_t>(-1);
        return a * b;
    };

    // Input vector -> HWC -> CHW
    model.push("vgg16_feat/raw_in", "Identity", 0);
    if (auto* L = model.getLayerByName("vgg16_feat/raw_in")) {
        L->inputs = {"__input__"};
        L->output = "vgg16_feat/in_vec";
    }
    model.push("vgg16_feat/in_reshape", "Reshape", 0);
    if (auto* R = model.getLayerByName("vgg16_feat/in_reshape")) {
        R->inputs = {"vgg16_feat/in_vec"};
        R->output = "vgg16_feat/in_hwc";
        R->target_shape = {H, W, C};
    }
    model.push("vgg16_feat/in_to_chw", "Permute", 0);
    if (auto* P = model.getLayerByName("vgg16_feat/in_to_chw")) {
        P->inputs = {"vgg16_feat/in_hwc"};
        P->output = "vgg16_feat/in_chw";
        P->shape = {H, W, C};
        P->permute_dims = {2, 0, 1};
    }

    auto add_conv_act = [&](const std::string& name,
                            const std::string& in,
                            const std::string& out,
                            int in_c,
                            int out_c,
                            int in_h,
                            int in_w,
                            int k,
                            int s,
                            int p) {
        model.push(name, "Conv2d",
                   sat_mul(static_cast<size_t>(out_c), sat_mul(static_cast<size_t>(in_c), sat_mul(static_cast<size_t>(k), static_cast<size_t>(k)))));
        if (auto* L = model.getLayerByName(name)) {
            L->inputs = {in};
            L->output = out;
            L->in_channels = in_c;
            L->out_channels = out_c;
            L->input_height = in_h;
            L->input_width = in_w;
            L->kernel_size = k;
            L->stride = s;
            L->padding = p;
            L->use_bias = false;
        }
        model.push(name + "/act", "ReLU", 0);
        if (auto* A = model.getLayerByName(name + "/act")) {
            A->inputs = {out};
            A->output = out + "_act";
        }
        return out + "_act";
    };

    auto add_gap = [&](const std::string& name,
                       const std::string& in,
                       int in_c,
                       int in_h,
                       int in_w) {
        model.push(name, "GlobalAvgPool2d", 0);
        if (auto* G = model.getLayerByName(name)) {
            G->inputs = {in};
            G->output = name + "_y";
            G->in_channels = in_c;
            G->input_height = in_h;
            G->input_width = in_w;
        }
        return name + "_y";
    };

    std::string x = "vgg16_feat/in_chw";
    int cur_c = C;
    int cur_h = H;
    int cur_w = W;

    std::vector<std::string> feats;

    auto block = [&](int bi, int convs, int out_c) {
        const std::string p = "vgg16_feat/b" + std::to_string(bi);
        for (int i = 0; i < convs; ++i) {
            x = add_conv_act(p + "/c" + std::to_string(i + 1), x, p + "/y" + std::to_string(i + 1), cur_c, out_c, cur_h, cur_w, 3, 1, 1);
            cur_c = out_c;
        }
        // Downsample via stride-2 conv
        x = add_conv_act(p + "/down", x, p + "/down_y", cur_c, cur_c, cur_h, cur_w, 3, 2, 1);
        cur_h = conv_out(cur_h, 3, 2, 1);
        cur_w = conv_out(cur_w, 3, 2, 1);

        // Feature tap: GAP after down
        feats.push_back(add_gap(p + "/gap", x, cur_c, cur_h, cur_w));
    };

    block(1, 2, base);
    block(2, 2, base * 2);
    block(3, 3, base * 4);
    block(4, 3, base * 8);
    block(5, 3, base * 8);

    // Concat features to output
    model.push("vgg16_feat/out", "Concat", 0);
    if (auto* L = model.getLayerByName("vgg16_feat/out")) {
        L->inputs = feats;
        L->output = "x";
        L->concat_axis = 0;
    }

    // output_dim = sum channels of each gap
    int out_dim = base + base * 2 + base * 4 + base * 8 + base * 8;
    model.modelConfig["output_dim"] = std::max(1, out_dim);
}
