#include "VGG16Model.hpp"

#include <algorithm>

VGG16Model::VGG16Model() {
    setModelName("VGG16Model");
    setHasEncoder(false);
}

void VGG16Model::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
}

static inline int conv_out(int in, int k, int s, int p) {
    return (in + 2 * p - k) / s + 1;
}

void VGG16Model::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("VGG16Model");
    model.modelConfig["type"] = "vgg16";

    int W = std::max(1, cfg.image_w);
    int H = std::max(1, cfg.image_h);
    int C = std::max(1, cfg.image_c);
    const int base = std::max(4, cfg.base_channels);
    const int num_classes = std::max(1, cfg.num_classes);
    const int fc_hidden = std::max(16, cfg.fc_hidden);
    const int image_dim = W * H * C;

    model.modelConfig["task"] = "image_classification";
    model.modelConfig["image_w"] = W;
    model.modelConfig["image_h"] = H;
    model.modelConfig["image_c"] = C;
    model.modelConfig["base_channels"] = base;
    model.modelConfig["num_classes"] = num_classes;
    model.modelConfig["input_dim"] = image_dim;
    model.modelConfig["output_dim"] = num_classes;

    model.push("vgg16/raw_in", "Identity", 0);
    if (auto* L = model.getLayerByName("vgg16/raw_in")) {
        L->inputs = {"__input__"};
        L->output = "vgg16/in";
    }

    auto add_conv_relu = [&](const std::string& name,
                             const std::string& in,
                             const std::string& out,
                             int in_c,
                             int out_c,
                             int in_h,
                             int in_w,
                             int k,
                             int s,
                             int p) {
        model.push(name, "Conv2d", static_cast<size_t>(out_c) * static_cast<size_t>(in_c) * static_cast<size_t>(k) * static_cast<size_t>(k));
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
        model.push(name + "/relu", "ReLU", 0);
        if (auto* R = model.getLayerByName(name + "/relu")) {
            R->inputs = {out};
            R->output = out + "_act";
        }
        return out + "_act";
    };

    std::string x = "vgg16/in";
    int cur_c = C;
    int cur_h = H;
    int cur_w = W;

    // VGG16 conv counts: [2,2,3,3,3] with downsample between blocks.
    auto block = [&](int bi, int convs, int out_c) {
        const std::string p = "vgg16/b" + std::to_string(bi);
        for (int i = 0; i < convs; ++i) {
            x = add_conv_relu(p + "/c" + std::to_string(i + 1), x, p + "/y" + std::to_string(i + 1), cur_c, out_c, cur_h, cur_w, 3, 1, 1);
            cur_c = out_c;
        }
        // Downsample with stride-2 conv (instead of MaxPool for safety)
        x = add_conv_relu(p + "/down", x, p + "/down_y", cur_c, cur_c, cur_h, cur_w, 3, 2, 1);
        cur_h = conv_out(cur_h, 3, 2, 1);
        cur_w = conv_out(cur_w, 3, 2, 1);
    };

    block(1, 2, base);
    block(2, 2, base * 2);
    block(3, 3, base * 4);
    block(4, 3, base * 8);
    block(5, 3, base * 8);

    model.push("vgg16/gap", "GlobalAvgPool2d", 0);
    if (auto* L = model.getLayerByName("vgg16/gap")) {
        L->inputs = {x};
        L->output = "vgg16/gap_y";
        L->in_channels = cur_c;
        L->input_height = cur_h;
        L->input_width = cur_w;
    }

    model.push("vgg16/fc1", "Linear", static_cast<size_t>(cur_c) * static_cast<size_t>(fc_hidden) + static_cast<size_t>(fc_hidden));
    if (auto* L = model.getLayerByName("vgg16/fc1")) {
        L->inputs = {"vgg16/gap_y"};
        L->output = "vgg16/fc1_y";
        L->in_features = cur_c;
        L->out_features = fc_hidden;
        L->use_bias = true;
    }
    model.push("vgg16/fc1_act", "GELU", 0);
    if (auto* L = model.getLayerByName("vgg16/fc1_act")) {
        L->inputs = {"vgg16/fc1_y"};
        L->output = "vgg16/fc1_act";
    }
    model.push("vgg16/head", "Linear", static_cast<size_t>(fc_hidden) * static_cast<size_t>(num_classes) + static_cast<size_t>(num_classes));
    if (auto* L = model.getLayerByName("vgg16/head")) {
        L->inputs = {"vgg16/fc1_act"};
        L->output = "x";
        L->in_features = fc_hidden;
        L->out_features = num_classes;
        L->use_bias = true;
    }
}
