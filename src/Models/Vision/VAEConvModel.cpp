#include "VAEConvModel.hpp"

#include <algorithm>
#include <stdexcept>

VAEConvModel::VAEConvModel() {
    setModelName("VAEConvModel");
    setHasEncoder(false);
}

void VAEConvModel::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
}

static inline void check_divisible(int a, int b, const std::string& msg) {
    if (b <= 0 || a <= 0 || (a % b) != 0) {
        throw std::runtime_error(msg);
    }
}

void VAEConvModel::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("VAEConvModel");
    model.modelConfig["type"] = "vae_conv";

    const int W = std::max(1, cfg.image_w);
    const int H = std::max(1, cfg.image_h);
    const int C = std::max(1, cfg.image_c);

    const int LH = std::max(1, cfg.latent_h);
    const int LW = std::max(1, cfg.latent_w);
    const int LC = std::max(1, cfg.latent_c);

    const bool text_cond = cfg.text_cond;
    const bool stochastic_latent = cfg.stochastic_latent;
    const int vocab = std::max(1, cfg.vocab_size);
    const int seq_len = std::max(1, cfg.seq_len);
    const int d_model = std::max(1, cfg.text_d_model);
    const int proj_dim = std::max(1, cfg.proj_dim);

    check_divisible(H, LH, "VAEConvModel: image_h must be divisible by latent_h");
    check_divisible(W, LW, "VAEConvModel: image_w must be divisible by latent_w");

    int down_h = H;
    int down_w = W;
    int downsamples = 0;
    while (down_h > LH && down_w > LW) {
        if ((down_h % 2) != 0 || (down_w % 2) != 0) break;
        down_h /= 2;
        down_w /= 2;
        ++downsamples;
    }
    if (down_h != LH || down_w != LW) {
        throw std::runtime_error("VAEConvModel: cannot reach latent_h/latent_w with /2 downsamples");
    }

    const int image_dim = W * H * C;
    const int latent_dim = LH * LW * LC;
    const int base = std::max(8, cfg.base_channels);

    model.modelConfig["task"] = text_cond ? "vae_conv_text_autoencoder" : "vae_conv_autoencoder";
    model.modelConfig["image_w"] = W;
    model.modelConfig["image_h"] = H;
    model.modelConfig["image_c"] = C;
    model.modelConfig["image_dim"] = image_dim;
    model.modelConfig["latent_h"] = LH;
    model.modelConfig["latent_w"] = LW;
    model.modelConfig["latent_c"] = LC;
    model.modelConfig["latent_dim"] = latent_dim;
    model.modelConfig["base_channels"] = base;
    model.modelConfig["downsamples"] = downsamples;
    model.modelConfig["input_dim"] = image_dim;
    // Recon loss: avoid MSE for VAEConv (better behaved on images in [-1,1]).
    model.modelConfig["recon_loss"] = "l1";
    model.modelConfig["text_cond"] = text_cond;
    model.modelConfig["stochastic_latent"] = stochastic_latent;
    if (text_cond) {
        model.modelConfig["vocab_size"] = vocab;
        model.modelConfig["seq_len"] = seq_len;
        model.modelConfig["text_d_model"] = d_model;
        model.modelConfig["proj_dim"] = proj_dim;
        model.modelConfig["output_dim"] = image_dim + 2 * latent_dim + 2 * proj_dim;
    } else {
        model.modelConfig["output_dim"] = image_dim + 2 * latent_dim;
    }

    auto sat_mul = [](size_t a, size_t b) -> size_t {
        if (a == 0 || b == 0) return 0;
        if (a > (static_cast<size_t>(-1) / b)) return static_cast<size_t>(-1);
        return a * b;
    };

    auto conv2d = [&](const std::string& name,
                      const std::string& in,
                      const std::string& out,
                      int in_c,
                      int out_c,
                      int in_h,
                      int in_w,
                      int k,
                      int s,
                      int p,
                      bool act) {
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
        std::string y = out;
        if (act) {
            model.push(name + "/act", "SiLU", 0);
            if (auto* A = model.getLayerByName(name + "/act")) {
                A->inputs = {out};
                A->output = out + "_act";
            }
            y = out + "_act";
        }
        return y;
    };

    auto upsample2x = [&](const std::string& name,
                          const std::string& in,
                          const std::string& out,
                          int channels,
                          int in_h,
                          int in_w) {
        model.push(name, "UpsampleNearest", 0);
        if (auto* U = model.getLayerByName(name)) {
            U->inputs = {in};
            U->output = out;
            U->in_channels = channels;
            U->out_h = in_h;
            U->out_w = in_w;
            U->scale_h = 2.0f;
            U->scale_w = 2.0f;
        }
        return out;
    };

    // ===== Optional text encoder (ids -> pooled embedding) =====
    // Input ids are expected under IntTensorStore key "text_ids" when text_cond=true.
    // Outputs:
    // - vae_conv/text/pooled (d_model)
    // - vae_conv/text/cond (LC) used to bias z spatial channels
    // - vae_conv/text/proj (proj_dim) used for alignment loss
    std::string text_pooled = "";
    if (text_cond) {
        model.push("vae_conv/text/tok_embed", "Embedding", static_cast<size_t>(vocab) * static_cast<size_t>(d_model));
        if (auto* L = model.getLayerByName("vae_conv/text/tok_embed")) {
            L->inputs = {"text_ids"};
            L->output = "vae_conv/text/tok";
            L->vocab_size = vocab;
            L->embed_dim = d_model;
            L->padding_idx = 0;
        }

        model.push("vae_conv/text/pool", "TokenMeanPool", 0);
        if (auto* L = model.getLayerByName("vae_conv/text/pool")) {
            L->inputs = {"vae_conv/text/tok"};
            L->output = "vae_conv/text/pooled";
            L->seq_len = seq_len;
            L->embed_dim = d_model;
        }
        text_pooled = "vae_conv/text/pooled";

        // Text -> conditioning vector for latent channels
        model.push("vae_conv/text/cond_fc", "Linear",
                   sat_mul(static_cast<size_t>(d_model), static_cast<size_t>(LC)) + static_cast<size_t>(LC));
        if (auto* L = model.getLayerByName("vae_conv/text/cond_fc")) {
            L->inputs = {text_pooled};
            L->output = "vae_conv/text/cond_pre";
            L->in_features = d_model;
            L->out_features = LC;
            L->use_bias = true;
        }
        model.push("vae_conv/text/cond_act", "Tanh", 0);
        if (auto* L = model.getLayerByName("vae_conv/text/cond_act")) {
            L->inputs = {"vae_conv/text/cond_pre"};
            L->output = "vae_conv/text/cond";
        }

        // Text -> projection head (for align loss)
        model.push("vae_conv/text/proj", "Linear",
                   sat_mul(static_cast<size_t>(d_model), static_cast<size_t>(proj_dim)) + static_cast<size_t>(proj_dim));
        if (auto* L = model.getLayerByName("vae_conv/text/proj")) {
            L->inputs = {text_pooled};
            L->output = "vae_conv/text/proj_vec";
            L->in_features = d_model;
            L->out_features = proj_dim;
            L->use_bias = true;
        }
    }

    // Input vector -> HWC -> CHW
    model.push("vae_conv/raw_in", "Identity", 0);
    if (auto* L = model.getLayerByName("vae_conv/raw_in")) {
        L->inputs = {"__input__"};
        L->output = "vae_conv/in_vec";
    }

    model.push("vae_conv/in_reshape", "Reshape", 0);
    if (auto* R = model.getLayerByName("vae_conv/in_reshape")) {
        R->inputs = {"vae_conv/in_vec"};
        R->output = "vae_conv/in_hwc";
        R->target_shape = {H, W, C};
    }

    model.push("vae_conv/in_to_chw", "Permute", 0);
    if (auto* P = model.getLayerByName("vae_conv/in_to_chw")) {
        P->inputs = {"vae_conv/in_hwc"};
        P->output = "vae_conv/in_chw";
        P->shape = {H, W, C};
        P->permute_dims = {2, 0, 1};
    }

    // Encoder
    std::string x = "vae_conv/in_chw";
    int cur_h = H;
    int cur_w = W;
    int cur_c = C;

    int ch = base;
    x = conv2d("vae_conv/enc/conv_in", x, "vae_conv/enc/c0", cur_c, ch, cur_h, cur_w, 3, 1, 1, true);
    cur_c = ch;

    for (int i = 0; i < downsamples; ++i) {
        const std::string b = "vae_conv/enc/down" + std::to_string(i + 1);
        x = conv2d(b + "/conv", x, b + "/y", cur_c, cur_c, cur_h, cur_w, 3, 2, 1, true);
        cur_h = std::max(1, (cur_h + 2 * 1 - 3) / 2 + 1);
        cur_w = std::max(1, (cur_w + 2 * 1 - 3) / 2 + 1);
    }

    // Project to mu/logvar at latent resolution
    x = conv2d("vae_conv/enc/proj", x, "vae_conv/enc/h", cur_c, cur_c, cur_h, cur_w, 3, 1, 1, true);
    model.push("vae_conv/enc/mu", "Conv2d",
               sat_mul(static_cast<size_t>(LC), sat_mul(static_cast<size_t>(cur_c), sat_mul(static_cast<size_t>(1), static_cast<size_t>(1)))));
    if (auto* L = model.getLayerByName("vae_conv/enc/mu")) {
        L->inputs = {"vae_conv/enc/h_act"};
        L->output = "vae_conv/mu";
        L->in_channels = cur_c;
        L->out_channels = LC;
        L->input_height = LH;
        L->input_width = LW;
        L->kernel_size = 1;
        L->stride = 1;
        L->padding = 0;
        L->use_bias = false;
    }

    model.push("vae_conv/enc/logvar", "Conv2d",
               sat_mul(static_cast<size_t>(LC), sat_mul(static_cast<size_t>(cur_c), sat_mul(static_cast<size_t>(1), static_cast<size_t>(1)))));
    if (auto* L = model.getLayerByName("vae_conv/enc/logvar")) {
        L->inputs = {"vae_conv/enc/h_act"};
        L->output = "vae_conv/logvar";
        L->in_channels = cur_c;
        L->out_channels = LC;
        L->input_height = LH;
        L->input_width = LW;
        L->kernel_size = 1;
        L->stride = 1;
        L->padding = 0;
        L->use_bias = false;
    }

    // Reparameterize (spatial latent CHW)
    // NOTE: le layer Reparameterize respecte modelConfig["stochastic_latent"].
    model.push("vae_conv/reparam", "Reparameterize", 0);
    if (auto* L = model.getLayerByName("vae_conv/reparam")) {
        L->inputs = {"vae_conv/mu", "vae_conv/logvar"};
        L->output = "vae_conv/z";
    }

    // Image projection head (for align loss): mu -> proj_dim
    if (text_cond) {
        model.push("vae_conv/img/proj", "Linear",
                   sat_mul(static_cast<size_t>(latent_dim), static_cast<size_t>(proj_dim)) + static_cast<size_t>(proj_dim));
        if (auto* L = model.getLayerByName("vae_conv/img/proj")) {
            L->inputs = {"vae_conv/mu"};
            L->output = "vae_conv/img/proj_vec";
            L->in_features = latent_dim;
            L->out_features = proj_dim;
            L->use_bias = true;
        }
    }

    // Optional conditioning: bias z channels with text-derived vector (broadcast over spatial).
    std::string z_in = "vae_conv/z";
    if (text_cond) {
        model.push("vae_conv/z_add_text", "Add", 0);
        if (auto* L = model.getLayerByName("vae_conv/z_add_text")) {
            L->inputs = {"vae_conv/z", "vae_conv/text/cond"};
            L->output = "vae_conv/z_cond";
        }
        z_in = "vae_conv/z_cond";
    }

    std::string y = z_in;
    int dy_h = LH;
    int dy_w = LW;
    int dy_c = LC;

    // Bring to base channels
    y = conv2d("vae_conv/dec/conv_in", y, "vae_conv/dec/c0", dy_c, base, dy_h, dy_w, 3, 1, 1, true);
    dy_c = base;

    for (int i = downsamples - 1; i >= 0; --i) {
        const std::string b = "vae_conv/dec/up" + std::to_string(i + 1);
        const int in_h = dy_h;
        const int in_w = dy_w;
        y = upsample2x(b + "/up", y, b + "/up_y", dy_c, in_h, in_w);
        dy_h = in_h * 2;
        dy_w = in_w * 2;
        y = conv2d(b + "/conv", y, b + "/c", dy_c, dy_c, dy_h, dy_w, 3, 1, 1, true);
    }

    // Final to RGB
    y = conv2d("vae_conv/dec/out", y, "vae_conv/dec/out_pre", dy_c, C, dy_h, dy_w, 3, 1, 1, false);

    model.push("vae_conv/dec/tanh", "Tanh", 0);
    if (auto* T = model.getLayerByName("vae_conv/dec/tanh")) {
        T->inputs = {"vae_conv/dec/out_pre"};
        T->output = "vae_conv/recon_chw";
    }

    // CHW -> HWC (image vector)
    model.push("vae_conv/recon_to_hwc", "Permute", 0);
    if (auto* P = model.getLayerByName("vae_conv/recon_to_hwc")) {
        P->inputs = {"vae_conv/recon_chw"};
        P->output = "vae_conv/recon";
        P->shape = {C, H, W};
        P->permute_dims = {1, 2, 0};
    }

    // Pack output recon || mu || logvar
    model.push("vae_conv/out_concat", "Concat", 0);
    if (auto* L = model.getLayerByName("vae_conv/out_concat")) {
        if (text_cond) {
            L->inputs = {"vae_conv/recon", "vae_conv/mu", "vae_conv/logvar", "vae_conv/img/proj_vec", "vae_conv/text/proj_vec"};
        } else {
            L->inputs = {"vae_conv/recon", "vae_conv/mu", "vae_conv/logvar"};
        }
        L->output = "x";
        L->concat_axis = 0;
    }
}

void VAEConvModel::buildDecoderInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("VAEConvModel");
    model.modelConfig["type"] = "vae_conv_decode";

    const int W = std::max(1, cfg.image_w);
    const int H = std::max(1, cfg.image_h);
    const int C = std::max(1, cfg.image_c);

    const int LH = std::max(1, cfg.latent_h);
    const int LW = std::max(1, cfg.latent_w);
    const int LC = std::max(1, cfg.latent_c);

    check_divisible(H, LH, "VAEConvModel(decode): image_h must be divisible by latent_h");
    check_divisible(W, LW, "VAEConvModel(decode): image_w must be divisible by latent_w");

    int down_h = H;
    int down_w = W;
    int downsamples = 0;
    while (down_h > LH && down_w > LW) {
        if ((down_h % 2) != 0 || (down_w % 2) != 0) break;
        down_h /= 2;
        down_w /= 2;
        ++downsamples;
    }
    if (down_h != LH || down_w != LW) {
        throw std::runtime_error("VAEConvModel(decode): cannot reach latent_h/latent_w with /2 downsamples");
    }

    const int image_dim = W * H * C;
    const int latent_dim = LH * LW * LC;
    const int base = std::max(8, cfg.base_channels);

    model.modelConfig["task"] = "vae_conv_decoder";
    model.modelConfig["image_w"] = W;
    model.modelConfig["image_h"] = H;
    model.modelConfig["image_c"] = C;
    model.modelConfig["image_dim"] = image_dim;
    model.modelConfig["latent_h"] = LH;
    model.modelConfig["latent_w"] = LW;
    model.modelConfig["latent_c"] = LC;
    model.modelConfig["latent_dim"] = latent_dim;
    model.modelConfig["base_channels"] = base;
    model.modelConfig["downsamples"] = downsamples;
    model.modelConfig["input_dim"] = latent_dim;
    model.modelConfig["output_dim"] = image_dim;

    auto sat_mul = [](size_t a, size_t b) -> size_t {
        if (a == 0 || b == 0) return 0;
        if (a > (static_cast<size_t>(-1) / b)) return static_cast<size_t>(-1);
        return a * b;
    };

    auto conv2d = [&](const std::string& name,
                      const std::string& in,
                      const std::string& out,
                      int in_c,
                      int out_c,
                      int in_h,
                      int in_w,
                      int k,
                      int s,
                      int p,
                      bool act) {
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
        std::string y = out;
        if (act) {
            model.push(name + "/act", "SiLU", 0);
            if (auto* A = model.getLayerByName(name + "/act")) {
                A->inputs = {out};
                A->output = out + "_act";
            }
            y = out + "_act";
        }
        return y;
    };

    auto upsample2x = [&](const std::string& name,
                          const std::string& in,
                          const std::string& out,
                          int channels,
                          int in_h,
                          int in_w) {
        model.push(name, "UpsampleNearest", 0);
        if (auto* U = model.getLayerByName(name)) {
            U->inputs = {in};
            U->output = out;
            U->in_channels = channels;
            U->out_h = in_h;
            U->out_w = in_w;
            U->scale_h = 2.0f;
            U->scale_w = 2.0f;
        }
        return out;
    };

    // Input latent vector -> vae_conv/z
    model.push("vae_conv/raw_z", "Identity", 0);
    if (auto* L = model.getLayerByName("vae_conv/raw_z")) {
        L->inputs = {"__input__"};
        L->output = "vae_conv/z";
    }

    std::string y = "vae_conv/z";
    int dy_h = LH;
    int dy_w = LW;
    int dy_c = LC;

    y = conv2d("vae_conv/dec/conv_in", y, "vae_conv/dec/c0", dy_c, base, dy_h, dy_w, 3, 1, 1, true);
    dy_c = base;

    for (int i = downsamples - 1; i >= 0; --i) {
        const std::string b = "vae_conv/dec/up" + std::to_string(i + 1);
        const int in_h = dy_h;
        const int in_w = dy_w;
        y = upsample2x(b + "/up", y, b + "/up_y", dy_c, in_h, in_w);
        dy_h = in_h * 2;
        dy_w = in_w * 2;
        y = conv2d(b + "/conv", y, b + "/c", dy_c, dy_c, dy_h, dy_w, 3, 1, 1, true);
    }

    y = conv2d("vae_conv/dec/out", y, "vae_conv/dec/out_pre", dy_c, C, dy_h, dy_w, 3, 1, 1, false);

    model.push("vae_conv/dec/tanh", "Tanh", 0);
    if (auto* T = model.getLayerByName("vae_conv/dec/tanh")) {
        T->inputs = {"vae_conv/dec/out_pre"};
        T->output = "vae_conv/recon_chw";
    }

    model.push("vae_conv/recon_to_hwc", "Permute", 0);
    if (auto* P = model.getLayerByName("vae_conv/recon_to_hwc")) {
        P->inputs = {"vae_conv/recon_chw"};
        P->output = "x";
        P->shape = {C, H, W};
        P->permute_dims = {1, 2, 0};
    }
}
