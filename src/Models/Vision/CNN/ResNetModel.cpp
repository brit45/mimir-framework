#include "ResNetModel.hpp"

#include <algorithm>

ResNetModel::ResNetModel() {
    setModelName("ResNetModel");
    setHasEncoder(false);
}

void ResNetModel::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
}

static inline int conv_out(int in, int k, int s, int p) {
    return (in + 2 * p - k) / s + 1;
}

void ResNetModel::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("ResNetModel");
    model.modelConfig["type"] = "resnet";

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

    model.push("resnet/raw_in", "Identity", 0);
    if (auto* L = model.getLayerByName("resnet/raw_in")) {
        L->inputs = {"__input__"};
        L->output = "resnet/in";
    }

    auto add_conv = [&](const std::string& name,
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
    };

    auto add_relu = [&](const std::string& name, const std::string& in, const std::string& out) {
        model.push(name, "ReLU", 0);
        if (auto* L = model.getLayerByName(name)) {
            L->inputs = {in};
            L->output = out;
        }
    };

    auto add_resblock = [&](const std::string& prefix,
                            const std::string& in,
                            int in_c,
                            int out_c,
                            int& cur_h,
                            int& cur_w,
                            int stride) -> std::string {
        const int k = 3;
        const int p = 1;
        const int s1 = stride;
        const int h1 = conv_out(cur_h, k, s1, p);
        const int w1 = conv_out(cur_w, k, s1, p);

        add_conv(prefix + "/conv1", in, prefix + "/c1", in_c, out_c, cur_h, cur_w, k, s1, p);
        add_relu(prefix + "/relu1", prefix + "/c1", prefix + "/c1_act");

        add_conv(prefix + "/conv2", prefix + "/c1_act", prefix + "/c2", out_c, out_c, h1, w1, k, 1, p);

        std::string skip = in;
        if (stride != 1 || in_c != out_c) {
            add_conv(prefix + "/skip", in, prefix + "/skip_y", in_c, out_c, cur_h, cur_w, 1, stride, 0);
            skip = prefix + "/skip_y";
        }

        model.push(prefix + "/add", "Add", 0);
        if (auto* A = model.getLayerByName(prefix + "/add")) {
            A->inputs = {prefix + "/c2", skip};
            A->output = prefix + "/sum";
        }

        add_relu(prefix + "/relu2", prefix + "/sum", prefix + "/out");

        cur_h = h1;
        cur_w = w1;
        return prefix + "/out";
    };

    std::string x = "resnet/in";
    int cur_c = C;
    int cur_h = H;
    int cur_w = W;

    // Stem: 3x3 stride 2
    add_conv("resnet/stem", x, "resnet/stem_y", cur_c, base, cur_h, cur_w, 3, 2, 1);
    add_relu("resnet/stem_relu", "resnet/stem_y", "resnet/stem_act");
    cur_h = conv_out(cur_h, 3, 2, 1);
    cur_w = conv_out(cur_w, 3, 2, 1);
    cur_c = base;
    x = "resnet/stem_act";

    const int b1 = std::max(1, cfg.blocks1);
    const int b2 = std::max(1, cfg.blocks2);
    const int b3 = std::max(1, cfg.blocks3);
    const int b4 = std::max(1, cfg.blocks4);

    auto stage = [&](int si, int blocks, int out_c, int first_stride) {
        for (int bi = 0; bi < blocks; ++bi) {
            const int s = (bi == 0) ? first_stride : 1;
            x = add_resblock("resnet/s" + std::to_string(si) + "/b" + std::to_string(bi + 1), x, cur_c, out_c, cur_h, cur_w, s);
            cur_c = out_c;
        }
    };

    stage(1, b1, base, 1);
    stage(2, b2, base * 2, 2);
    stage(3, b3, base * 4, 2);
    stage(4, b4, base * 8, 2);

    // Global average pool -> [cur_c]
    model.push("resnet/gap", "GlobalAvgPool2d", 0);
    if (auto* L = model.getLayerByName("resnet/gap")) {
        L->inputs = {x};
        L->output = "resnet/gap_y";
        L->in_channels = cur_c;
        L->input_height = cur_h;
        L->input_width = cur_w;
    }

    model.push("resnet/head", "Linear", static_cast<size_t>(cur_c) * static_cast<size_t>(num_classes) + static_cast<size_t>(num_classes));
    if (auto* L = model.getLayerByName("resnet/head")) {
        L->inputs = {"resnet/gap_y"};
        L->output = "x";
        L->in_features = cur_c;
        L->out_features = num_classes;
        L->use_bias = true;
    }
}
