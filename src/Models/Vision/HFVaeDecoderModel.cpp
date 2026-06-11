#include "HFVaeDecoderModel.hpp"

#include <algorithm>
#include <stdexcept>

namespace {
static void check_divisible_local(int a, int b, const std::string& msg) {
    if (a <= 0 || b <= 0 || (a % b) != 0) throw std::runtime_error(msg);
}

static int gn_groups_for(int channels, int requested) {
    int groups = std::max(1, std::min(requested, channels));
    while (groups > 1 && channels % groups != 0) --groups;
    return groups;
}
}

HFVaeDecoderModel::HFVaeDecoderModel() {
    setModelName("HFVaeDecoderModel");
    setHasEncoder(false);
}

void HFVaeDecoderModel::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
}

void HFVaeDecoderModel::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("HFVaeDecoderModel");
    model.modelConfig["type"] = "hf_vae_decoder";

    const int W = std::max(1, cfg.image_w);
    const int H = std::max(1, cfg.image_h);
    const int C = std::max(1, cfg.image_c);
    const int LW = std::max(1, cfg.latent_w);
    const int LH = std::max(1, cfg.latent_h);
    const int LC = std::max(1, cfg.latent_c);
    const int heads = std::max(1, cfg.num_heads);
    const int norm_groups = std::max(1, cfg.norm_groups);

    check_divisible_local(W, LW, "HFVaeDecoderModel: image_w must be divisible by latent_w");
    check_divisible_local(H, LH, "HFVaeDecoderModel: image_h must be divisible by latent_h");

    const int scale_w = W / LW;
    const int scale_h = H / LH;
    if (scale_w != scale_h || scale_w != 8) {
        throw std::runtime_error("HFVaeDecoderModel: expected x8 latent scale");
    }

    model.modelConfig["task"] = "hf_vae_decoder";
    model.modelConfig["image_w"] = W;
    model.modelConfig["image_h"] = H;
    model.modelConfig["image_c"] = C;
    model.modelConfig["latent_w"] = LW;
    model.modelConfig["latent_h"] = LH;
    model.modelConfig["latent_c"] = LC;
    model.modelConfig["input_dim"] = LH * LW * LC;
    model.modelConfig["output_dim"] = H * W * C;

    auto conv2d = [&](const std::string& name,
                      const std::string& in,
                      const std::string& out,
                      int in_c,
                      int out_c,
                      int in_h,
                      int in_w,
                      int k,
                      int p,
                      bool use_bias) {
        model.push(name, "Conv2d", static_cast<size_t>(out_c) * static_cast<size_t>(in_c) * static_cast<size_t>(k) * static_cast<size_t>(k) + (use_bias ? static_cast<size_t>(out_c) : 0ULL));
        if (auto* L = model.getLayerByName(name)) {
            L->inputs = {in};
            L->output = out;
            L->in_channels = in_c;
            L->out_channels = out_c;
            L->input_height = in_h;
            L->input_width = in_w;
            L->kernel_size = k;
            L->stride = 1;
            L->padding = p;
            L->use_bias = use_bias;
        }
        return out;
    };

    auto groupnorm = [&](const std::string& name,
                         const std::string& in,
                         const std::string& out,
                         int channels,
                         int h,
                         int w) {
        model.push(name, "GroupNorm", static_cast<size_t>(channels) * 2ULL);
        if (auto* L = model.getLayerByName(name)) {
            L->inputs = {in};
            L->output = out;
            L->in_channels = channels;
            L->input_height = h;
            L->input_width = w;
            L->num_groups = gn_groups_for(channels, norm_groups);
            L->affine = true;
            L->use_bias = true;
            L->eps = 1e-6f;
        }
        return out;
    };

    auto resblock = [&](const std::string& prefix,
                        const std::string& in,
                        int in_c,
                        int out_c,
                        int h,
                        int w) {
        std::string y = groupnorm(prefix + "/norm1", in, prefix + "/norm1_out", in_c, h, w);
        model.push(prefix + "/act1", "SiLU", 0);
        if (auto* L = model.getLayerByName(prefix + "/act1")) {
            L->inputs = {y};
            L->output = prefix + "/act1_out";
        }
        y = conv2d(prefix + "/conv1", prefix + "/act1_out", prefix + "/conv1_out", in_c, out_c, h, w, 3, 1, true);
        y = groupnorm(prefix + "/norm2", y, prefix + "/norm2_out", out_c, h, w);
        model.push(prefix + "/act2", "SiLU", 0);
        if (auto* L = model.getLayerByName(prefix + "/act2")) {
            L->inputs = {y};
            L->output = prefix + "/act2_out";
        }
        y = conv2d(prefix + "/conv2", prefix + "/act2_out", prefix + "/conv2_out", out_c, out_c, h, w, 3, 1, true);

        std::string skip = in;
        if (in_c != out_c) {
            skip = conv2d(prefix + "/nin_shortcut", in, prefix + "/skip", in_c, out_c, h, w, 1, 0, true);
        }

        model.push(prefix + "/add", "Add", 0);
        if (auto* L = model.getLayerByName(prefix + "/add")) {
            L->inputs = {y, skip};
            L->output = prefix + "/out";
        }
        return std::make_pair(std::string(prefix + "/out"), out_c);
    };

    auto attn_block = [&](const std::string& prefix,
                          const std::string& in,
                          int ch,
                          int h,
                          int w) {
        std::string y = groupnorm(prefix + "/norm", in, prefix + "/norm_out", ch, h, w);
        model.push(prefix + "/to_hwc", "Permute", 0);
        if (auto* L = model.getLayerByName(prefix + "/to_hwc")) {
            L->inputs = {y};
            L->output = prefix + "/hwc";
            L->shape = {ch, h, w};
            L->permute_dims = {1, 2, 0};
        }
        model.push(prefix + "/attn", "SelfAttention", static_cast<size_t>(4) * static_cast<size_t>(ch) * static_cast<size_t>(ch) + static_cast<size_t>(4) * static_cast<size_t>(ch));
        if (auto* L = model.getLayerByName(prefix + "/attn")) {
            L->inputs = {prefix + "/hwc"};
            L->output = prefix + "/attn_out";
            L->seq_len = h * w;
            L->embed_dim = ch;
            L->num_heads = heads;
        }
        model.push(prefix + "/to_chw", "Permute", 0);
        if (auto* L = model.getLayerByName(prefix + "/to_chw")) {
            L->inputs = {prefix + "/attn_out"};
            L->output = prefix + "/chw";
            L->shape = {h, w, ch};
            L->permute_dims = {2, 0, 1};
        }
        model.push(prefix + "/add", "Add", 0);
        if (auto* L = model.getLayerByName(prefix + "/add")) {
            L->inputs = {prefix + "/chw", in};
            L->output = prefix + "/out";
        }
        return prefix + "/out";
    };

    auto upsample_conv = [&](const std::string& prefix,
                             const std::string& in,
                             int ch,
                             int h,
                             int w) {
        model.push(prefix + "/up", "UpsampleNearest", 0);
        if (auto* L = model.getLayerByName(prefix + "/up")) {
            L->inputs = {in};
            L->output = prefix + "/up_out";
            L->in_channels = ch;
            L->out_h = h;
            L->out_w = w;
            L->scale_h = 2.0f;
            L->scale_w = 2.0f;
        }
        return conv2d(prefix + "/conv", prefix + "/up_out", prefix + "/out", ch, ch, h * 2, w * 2, 3, 1, true);
    };

    const int stage_channels[4] = {128, 256, 512, 512};

    model.push("sdxl/vae_decoder/raw_z", "Identity", 0);
    if (auto* L = model.getLayerByName("sdxl/vae_decoder/raw_z")) {
        L->inputs = {"__input__"};
        L->output = "sdxl/vae_decoder/z";
    }

    std::string x = conv2d("sdxl/vae_decoder/post_quant_conv", "sdxl/vae_decoder/z", "sdxl/vae_decoder/post_q", LC, LC, LH, LW, 1, 0, true);
    x = conv2d("sdxl/vae_decoder/conv_in", x, "sdxl/vae_decoder/conv_in_out", LC, 512, LH, LW, 3, 1, true);

    int cur_c = 512;
    int cur_h = LH;
    int cur_w = LW;

    x = resblock("sdxl/vae_decoder/mid/block_1", x, cur_c, 512, cur_h, cur_w).first;
    x = attn_block("sdxl/vae_decoder/mid/attn_1", x, 512, cur_h, cur_w);
    x = resblock("sdxl/vae_decoder/mid/block_2", x, 512, 512, cur_h, cur_w).first;

    for (int stage = 3; stage >= 0; --stage) {
        const int out_c = stage_channels[stage];
        for (int block = 0; block < 3; ++block) {
            auto rb = resblock(
                "sdxl/vae_decoder/up." + std::to_string(stage) + "/block." + std::to_string(block),
                x,
                cur_c,
                out_c,
                cur_h,
                cur_w
            );
            x = rb.first;
            cur_c = rb.second;
        }
        if (stage > 0) {
            x = upsample_conv("sdxl/vae_decoder/up." + std::to_string(stage) + "/upsample", x, cur_c, cur_h, cur_w);
            cur_h *= 2;
            cur_w *= 2;
        }
    }

    x = groupnorm("sdxl/vae_decoder/norm_out", x, "sdxl/vae_decoder/norm_out_out", cur_c, cur_h, cur_w);
    model.push("sdxl/vae_decoder/act_out", "SiLU", 0);
    if (auto* L = model.getLayerByName("sdxl/vae_decoder/act_out")) {
        L->inputs = {x};
        L->output = "sdxl/vae_decoder/act_out_out";
    }
    x = conv2d("sdxl/vae_decoder/conv_out", "sdxl/vae_decoder/act_out_out", "sdxl/vae_decoder/conv_out_out", cur_c, C, cur_h, cur_w, 3, 1, true);

    model.push("sdxl/vae_decoder/tanh", "Tanh", 0);
    if (auto* L = model.getLayerByName("sdxl/vae_decoder/tanh")) {
        L->inputs = {x};
        L->output = "sdxl/vae_decoder/recon_chw";
    }

    model.push("sdxl/vae_decoder/recon_to_hwc", "Permute", 0);
    if (auto* L = model.getLayerByName("sdxl/vae_decoder/recon_to_hwc")) {
        L->inputs = {"sdxl/vae_decoder/recon_chw"};
        L->output = "x";
        L->shape = {C, H, W};
        L->permute_dims = {1, 2, 0};
    }
}