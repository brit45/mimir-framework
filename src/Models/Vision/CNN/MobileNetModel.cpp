#include "MobileNetModel.hpp"

#include <algorithm>

MobileNetModel::MobileNetModel() {
    setModelName("MobileNetModel");
    setHasEncoder(false);
}

void MobileNetModel::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
}

static inline int conv_out(int in, int k, int s, int p) {
    return (in + 2 * p - k) / s + 1;
}

void MobileNetModel::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("MobileNetModel");
    model.modelConfig["type"] = "mobilenet";

    int W = std::max(1, cfg.image_w);
    int H = std::max(1, cfg.image_h);
    int C = std::max(1, cfg.image_c);
    const int base = std::max(4, cfg.base_channels);
    const int num_classes = std::max(1, cfg.num_classes);
    const int image_dim = W * H * C;

    model.modelConfig["task"] = "image_classification";
    model.modelConfig["image_w"] = W;
    model.modelConfig["image_h"] = H;
    model.modelConfig["image_c"] = C;
    model.modelConfig["base_channels"] = base;
    model.modelConfig["num_classes"] = num_classes;
    model.modelConfig["input_dim"] = image_dim;
    model.modelConfig["output_dim"] = num_classes;

    model.push("mobilenet/raw_in", "Identity", 0);
    if (auto* L = model.getLayerByName("mobilenet/raw_in")) {
        L->inputs = {"__input__"};
        L->output = "mobilenet/in";
    }

    auto add_relu = [&](const std::string& name, const std::string& in, const std::string& out) {
        model.push(name, "ReLU", 0);
        if (auto* L = model.getLayerByName(name)) {
            L->inputs = {in};
            L->output = out;
        }
    };

    auto add_pw = [&](const std::string& name,
                      const std::string& in,
                      const std::string& out,
                      int in_c,
                      int out_c,
                      int in_h,
                      int in_w) {
        model.push(name, "Conv2d", static_cast<size_t>(out_c) * static_cast<size_t>(in_c) * 1ULL * 1ULL);
        if (auto* L = model.getLayerByName(name)) {
            L->inputs = {in};
            L->output = out;
            L->in_channels = in_c;
            L->out_channels = out_c;
            L->input_height = in_h;
            L->input_width = in_w;
            L->kernel_size = 1;
            L->stride = 1;
            L->padding = 0;
            L->use_bias = false;
        }
    };

    auto add_dw = [&](const std::string& name,
                      const std::string& in,
                      const std::string& out,
                      int channels,
                      int in_h,
                      int in_w,
                      int k,
                      int s,
                      int p) {
        const size_t params = static_cast<size_t>(channels) * static_cast<size_t>(k) * static_cast<size_t>(k) + static_cast<size_t>(channels);
        model.push(name, "DepthwiseConv2d", params);
        if (auto* L = model.getLayerByName(name)) {
            L->inputs = {in};
            L->output = out;
            L->in_channels = channels;
            L->out_channels = channels;
            L->input_height = in_h;
            L->input_width = in_w;
            L->kernel_size = k;
            L->stride_h = s;
            L->stride_w = s;
            L->pad_h = p;
            L->pad_w = p;
            L->use_bias = true;
        }
    };

    std::string x = "mobilenet/in";
    int cur_c = C;
    int cur_h = H;
    int cur_w = W;

    // Stem: 3x3 stride 2 pointwise-ish conv
    add_pw("mobilenet/stem", x, "mobilenet/stem_y", cur_c, base, cur_h, cur_w);
    add_relu("mobilenet/stem_relu", "mobilenet/stem_y", "mobilenet/stem_act");
    x = "mobilenet/stem_act";
    cur_c = base;

    auto dw_pw_block = [&](int idx, int out_c, int stride) {
        const std::string p = "mobilenet/b" + std::to_string(idx);
        add_dw(p + "/dw", x, p + "/dw_y", cur_c, cur_h, cur_w, 3, stride, 1);
        add_relu(p + "/dw_relu", p + "/dw_y", p + "/dw_act");
        int nh = conv_out(cur_h, 3, stride, 1);
        int nw = conv_out(cur_w, 3, stride, 1);
        add_pw(p + "/pw", p + "/dw_act", p + "/pw_y", cur_c, out_c, nh, nw);
        add_relu(p + "/pw_relu", p + "/pw_y", p + "/out");
        x = p + "/out";
        cur_c = out_c;
        cur_h = nh;
        cur_w = nw;
    };

    dw_pw_block(1, base * 2, 2);
    dw_pw_block(2, base * 2, 1);
    dw_pw_block(3, base * 4, 2);
    dw_pw_block(4, base * 4, 1);
    dw_pw_block(5, base * 8, 2);
    dw_pw_block(6, base * 8, 1);

    model.push("mobilenet/gap", "GlobalAvgPool2d", 0);
    if (auto* L = model.getLayerByName("mobilenet/gap")) {
        L->inputs = {x};
        L->output = "mobilenet/gap_y";
        L->in_channels = cur_c;
        L->input_height = cur_h;
        L->input_width = cur_w;
    }

    model.push("mobilenet/head", "Linear", static_cast<size_t>(cur_c) * static_cast<size_t>(num_classes) + static_cast<size_t>(num_classes));
    if (auto* L = model.getLayerByName("mobilenet/head")) {
        L->inputs = {"mobilenet/gap_y"};
        L->output = "x";
        L->in_features = cur_c;
        L->out_features = num_classes;
        L->use_bias = true;
    }
}
