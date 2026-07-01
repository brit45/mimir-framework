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
    // ConditioningEncoder externe : embeddings mag/mod/seq appris de dimension d_model.
    // dim=d_model << latent_dim => ensureVocabSize raisonnable (~200 Mo max).
    if (cfg_.d_model > 0) {
        setHasEncoder(true);
        getMutableEncoder().ensureDim(cfg_.d_model);
        getMutableEncoder().ensureSpecialEmbeddings();
    }
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

    const bool stochastic_latent = cfg.stochastic_latent;

    // ===== Modèle PUREMENT CONVOLUTIONNEL =====
    // Aucune SelfAttention ni couche dense/texte : l'autoencodeur est 100% conv
    // (Conv2d / ConvTranspose2d / GroupNorm|LayerNorm / SiLU / résidus Add /
    //  UpsampleNearest|ConvTranspose2d / Reparameterize). Les seuls layers non
    // "calcul" sont les adaptateurs de disposition (Reshape/Permute) requis pour
    // l'I/O en vecteur plat. Les blocs ResNet (conv3x3->SiLU->conv3x3 + skip)
    // restent disponibles via `cfg.use_attention` (nom historique).
    // Les options d'attention/texte du Config sont ignorées (conservées pour
    // compatibilité de plomberie JSON/Lua uniquement).
    const bool use_resnet       = cfg.use_attention;
    const bool use_skip_conn    = cfg.use_skip_connections;
    const bool use_enc_prior    = cfg.use_encoder_prior;
    // Modèle PUREMENT CONVOLUTIONNEL : aucune normalisation (GroupNorm/LayerNorm)
    // n'est émise dans le graphe. Les champs enc_norm/dec_norm/*_gn_groups restent
    // dans la config pour compat plomberie mais sont forcés à "none".
    const std::string enc_norm_str = "none";
    const int enc_gn_groups_val    = std::max(1, cfg.enc_gn_groups);
    const std::string dec_norm_str = "none";
    const int dec_gn_groups_val    = std::max(1, cfg.dec_gn_groups > 0 ? cfg.dec_gn_groups : cfg.enc_gn_groups);
    const std::string dec_upsample = cfg.decoder_upsample.empty() ? "conv_transpose" : cfg.decoder_upsample;
    const int resnet_max_tok       = cfg.resnet_max_tokens;  // 0 = no gate

    auto resnet_gate = [&](int h, int w) -> bool {
        if (!use_resnet) return false;
        if (resnet_max_tok <= 0) return true;
        return (h * w) <= resnet_max_tok;
    };

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

    model.modelConfig["task"] = "vae_conv_autoencoder";
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
    model.modelConfig["text_cond"] = false;
    model.modelConfig["stochastic_latent"] = stochastic_latent;
    model.modelConfig["enc_norm"] = enc_norm_str;
    model.modelConfig["enc_gn_groups"] = enc_gn_groups_val;
    model.modelConfig["dec_norm"] = dec_norm_str;
    model.modelConfig["dec_gn_groups"] = dec_gn_groups_val;
    model.modelConfig["decoder_upsample"] = dec_upsample;
    model.modelConfig["use_skip_connections"] = use_skip_conn;
    model.modelConfig["use_encoder_prior"] = use_enc_prior;
    model.modelConfig["use_attention"] = use_resnet;
    model.modelConfig["resnet_max_tokens"] = resnet_max_tok;
    if (cfg.d_model > 0) model.modelConfig["d_model"] = cfg.d_model;
    model.modelConfig["output_dim"] = image_dim + 2 * latent_dim;

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
        const int out_h = std::max(1, in_h * 2);
        const int out_w = std::max(1, in_w * 2);
        model.push(name, "UpsampleNearest", 0);
        if (auto* U = model.getLayerByName(name)) {
            U->inputs = {in};
            U->output = out;
            U->in_channels = channels;
            U->input_height = in_h;
            U->input_width = in_w;
            U->output_height = out_h;
            U->output_width = out_w;
            U->out_h = out_h;
            U->out_w = out_w;
            U->scale_h = 2.0f;
            U->scale_w = 2.0f;
        }
        return out;
    };

    auto deconv2d = [&](const std::string& name,
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
        const int out_h = std::max(1, (in_h - 1) * s - 2 * p + k);
        const int out_w = std::max(1, (in_w - 1) * s - 2 * p + k);
        model.push(name, "ConvTranspose2d",
                   sat_mul(static_cast<size_t>(out_c), sat_mul(static_cast<size_t>(in_c), sat_mul(static_cast<size_t>(k), static_cast<size_t>(k)))));
        if (auto* L = model.getLayerByName(name)) {
            L->inputs = {in};
            L->output = out;
            L->in_channels = in_c;
            L->out_channels = out_c;
            L->input_height = in_h;
            L->input_width = in_w;
            L->output_height = out_h;
            L->output_width = out_w;
            L->out_h = out_h;
            L->out_w = out_w;
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

    // ===== Bloc ResNet : conv3x3 -> SiLU -> conv3x3 + skip =====
    auto resblock = [&](const std::string& prefix, const std::string& in, int ch, int h, int w) -> std::string {
        const size_t ps = sat_mul(static_cast<size_t>(ch), sat_mul(static_cast<size_t>(ch), 9));
        model.push(prefix + "/conv1", "Conv2d", ps);
        if (auto* L = model.getLayerByName(prefix + "/conv1")) {
            L->inputs = {in};        L->output = prefix + "/c1";
            L->in_channels = ch;    L->out_channels = ch;
            L->input_height = h;    L->input_width = w;
            L->kernel_size = 3;     L->stride = 1;  L->padding = 1;
            L->use_bias = false;
        }
        model.push(prefix + "/act1", "SiLU", 0);
        if (auto* A = model.getLayerByName(prefix + "/act1")) {
            A->inputs = {prefix + "/c1"};
            A->output = prefix + "/c1a";
        }
        model.push(prefix + "/conv2", "Conv2d", ps);
        if (auto* L = model.getLayerByName(prefix + "/conv2")) {
            L->inputs = {prefix + "/c1a"};  L->output = prefix + "/c2";
            L->in_channels = ch;            L->out_channels = ch;
            L->input_height = h;            L->input_width = w;
            L->kernel_size = 3;             L->stride = 1;  L->padding = 1;
            L->use_bias = false;
        }
        model.push(prefix + "/add", "Add", 0);
        if (auto* A = model.getLayerByName(prefix + "/add")) {
            A->inputs = {prefix + "/c2", in};
            A->output = prefix + "/out";
        }
        return prefix + "/out";
    };

    // ===== Normalisation encodeur (GroupNorm ou LayerNorm spatial) =====
    auto add_enc_norm = [&](const std::string& prefix, const std::string& in, int ch, int h, int w) -> std::string {
        if (enc_norm_str == "none" || enc_norm_str.empty()) return in;
        if (enc_norm_str == "groupnorm" || enc_norm_str == "gn") {
            int groups = enc_gn_groups_val;
            while (groups > 1 && ch % groups != 0) --groups;
            model.push(prefix + "/gn", "GroupNorm", static_cast<size_t>(ch) * 2);
            if (auto* L = model.getLayerByName(prefix + "/gn")) {
                L->inputs = {in};       L->output = prefix + "/gn_out";
                L->in_channels = ch;    L->num_groups = groups;
                L->input_height = h;    L->input_width = w;
            }
            return prefix + "/gn_out";
        }
        if (enc_norm_str == "layernorm" || enc_norm_str == "ln") {
            model.push(prefix + "/ln", "LayerNorm", static_cast<size_t>(ch) * 2);
            if (auto* L = model.getLayerByName(prefix + "/ln")) {
                L->inputs = {in};       L->output = prefix + "/ln_out";
                L->in_channels = ch;    L->input_height = h;  L->input_width = w;
            }
            return prefix + "/ln_out";
        }
        return in;
    };

    auto add_dec_norm = [&](const std::string& prefix, const std::string& in, int ch, int h, int w) -> std::string {
        if (dec_norm_str == "none" || dec_norm_str.empty()) return in;
        if (dec_norm_str == "groupnorm" || dec_norm_str == "gn") {
            int groups = dec_gn_groups_val;
            while (groups > 1 && ch % groups != 0) --groups;
            model.push(prefix + "/gn", "GroupNorm", static_cast<size_t>(ch) * 2);
            if (auto* L = model.getLayerByName(prefix + "/gn")) {
                L->inputs = {in};       L->output = prefix + "/gn_out";
                L->in_channels = ch;    L->num_groups = groups;
                L->input_height = h;    L->input_width = w;
            }
            return prefix + "/gn_out";
        }
        if (dec_norm_str == "layernorm" || dec_norm_str == "ln") {
            model.push(prefix + "/ln", "LayerNorm", static_cast<size_t>(ch) * 2);
            if (auto* L = model.getLayerByName(prefix + "/ln")) {
                L->inputs = {in};       L->output = prefix + "/ln_out";
                L->in_channels = ch;    L->input_height = h;  L->input_width = w;
            }
            return prefix + "/ln_out";
        }
        return in;
    };

    // NOTE: pas de SelfAttention ni d'encodeur texte : graphe purement convolutionnel.
    // Skip connections (style U-Net) : enc_skips[i] est la feature map de l'encodeur
    // à la résolution H/2^i × W/2^i, sauvegardée avant le downsampling du niveau i.
    std::vector<std::string> enc_skips;
    std::vector<int> enc_skip_h, enc_skip_w;

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

    // ConditioningEncoder
    std::string x = "vae_conv/in_chw";
    int cur_h = H;
    int cur_w = W;
    int cur_c = C;

    int ch = base;
    x = conv2d("vae_conv/enc/conv_in", x, "vae_conv/enc/c0", cur_c, ch, cur_h, cur_w, 3, 1, 1, true);
    cur_c = ch;
    if (resnet_gate(cur_h, cur_w)) {
        x = add_enc_norm("vae_conv/enc/n0", x, cur_c, cur_h, cur_w);
        x = resblock("vae_conv/enc/res0", x, cur_c, cur_h, cur_w);
    }

    for (int i = 0; i < downsamples; ++i) {
        // Sauvegarde du skip avant downsampling (résolution H/2^i × W/2^i)
        if (use_skip_conn) {
            enc_skips.push_back(x);
            enc_skip_h.push_back(cur_h);
            enc_skip_w.push_back(cur_w);
        }
        const std::string b = "vae_conv/enc/down" + std::to_string(i + 1);
        x = conv2d(b + "/conv", x, b + "/y", cur_c, cur_c, cur_h, cur_w, 3, 2, 1, true);
        cur_h = std::max(1, (cur_h + 2 * 1 - 3) / 2 + 1);
        cur_w = std::max(1, (cur_w + 2 * 1 - 3) / 2 + 1);
        if (resnet_gate(cur_h, cur_w)) {
            x = add_enc_norm(b + "/n", x, cur_c, cur_h, cur_w);
            x = resblock(b + "/res", x, cur_c, cur_h, cur_w);
        }
    }

    // Project to mu/logvar at latent resolution
    x = conv2d("vae_conv/enc/proj", x, "vae_conv/enc/h", cur_c, cur_c, cur_h, cur_w, 3, 1, 1, true);
    // Bottleneck : ResNet convolutionnel optionnel (plus aucune SelfAttention)
    if (resnet_gate(cur_h, cur_w)) {
        x = add_enc_norm("vae_conv/enc/bot_n", x, cur_c, cur_h, cur_w);
        x = resblock("vae_conv/enc/bot_res", x, cur_c, cur_h, cur_w);
    }
    model.push("vae_conv/enc/mu", "Conv2d",
               sat_mul(static_cast<size_t>(LC), sat_mul(static_cast<size_t>(cur_c), sat_mul(static_cast<size_t>(1), static_cast<size_t>(1)))));
    if (auto* L = model.getLayerByName("vae_conv/enc/mu")) {
        L->inputs = {x};
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
        L->inputs = {x};
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

    // Prior appris dans le graphe : couche Constant de taille latent_dim
    // ajoutée additivement à z avant le décodeur.
    // Les poids de cette couche (un vecteur CHW de taille LC*LH*LW) sont
    // entraînés par backprop : ils capturent un biais global appris sur
    // l'espace latent, améliorant la représentation sans interaction
    // avec le tokenizer ni la table d'embeddings.
    std::string z_in = "vae_conv/z";
    if (use_enc_prior) {
        model.push("vae_conv/z_prior_bias", "Constant",
                   static_cast<size_t>(LC) * static_cast<size_t>(LH) * static_cast<size_t>(LW));
        if (auto* L = model.getLayerByName("vae_conv/z_prior_bias")) {
            L->inputs  = {};
            L->output  = "vae_conv/prior_bias_out";
            L->in_channels  = LC;
            L->out_channels = LC;
            L->input_height = LH;
            L->input_width  = LW;
        }
        model.push("vae_conv/z_prior_add", "Add", 0);
        if (auto* L = model.getLayerByName("vae_conv/z_prior_add")) {
            L->inputs = {"vae_conv/z", "vae_conv/prior_bias_out"};
            L->output = "vae_conv/z_biased";
        }
        z_in = "vae_conv/z_biased";
    }

    std::string y = z_in;
    int dy_h = LH;
    int dy_w = LW;
    int dy_c = LC;

    // Bring to base channels
    y = conv2d("vae_conv/dec/conv_in", y, "vae_conv/dec/c0", dy_c, base, dy_h, dy_w, 3, 1, 1, true);
    dy_c = base;
    y = add_dec_norm("vae_conv/dec/n0", y, dy_c, dy_h, dy_w);

    // Bottleneck décodeur : ResNet convolutionnel optionnel (plus aucune SelfAttention)
    if (resnet_gate(dy_h, dy_w)) {
        y = add_dec_norm("vae_conv/dec/bot_n", y, dy_c, dy_h, dy_w);
        y = resblock("vae_conv/dec/bot_res", y, dy_c, dy_h, dy_w);
    }

    for (int i = downsamples - 1; i >= 0; --i) {
        const std::string b = "vae_conv/dec/up" + std::to_string(i + 1);
        const int in_h = dy_h;
        const int in_w = dy_w;
        if (dec_upsample == "nearest_conv") {
            y = upsample2x(b + "/up", y, b + "/up_y", dy_c, in_h, in_w);
            dy_h = in_h * 2;
            dy_w = in_w * 2;
            y = conv2d(b + "/conv", y, b + "/c", dy_c, dy_c, dy_h, dy_w, 3, 1, 1, true);
        } else {
            y = deconv2d(b + "/up", y, b + "/up_y", dy_c, dy_c, in_h, in_w, 4, 2, 1, true);
            dy_h = in_h * 2;
            dy_w = in_w * 2;
        }
        // Skip connection encodeur→décodeur : concat + projection 1×1 (2*base → base)
        if (use_skip_conn && i < static_cast<int>(enc_skips.size())) {
            model.push(b + "/skip_cat", "Concat", 0);
            if (auto* L = model.getLayerByName(b + "/skip_cat")) {
                L->inputs      = {y, enc_skips[static_cast<size_t>(i)]};
                L->output      = b + "/sc";
                L->concat_axis = 0;
            }
            // Conv 1×1 linéaire : comprime 2*base canaux → base
            y = conv2d(b + "/skip_proj", b + "/sc", b + "/sp",
                       2 * dy_c, dy_c, dy_h, dy_w, 1, 1, 0, /*act=*/false);
        }
        y = add_dec_norm(b + "/n", y, dy_c, dy_h, dy_w);
        if (resnet_gate(dy_h, dy_w)) {
            y = resblock(b + "/res", y, dy_c, dy_h, dy_w);
        }
    }

    // Final to RGB
    y = add_dec_norm("vae_conv/dec/out_n", y, dy_c, dy_h, dy_w);
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

    // Pack output recon || z_biased || logvar
    // z_in = vae_conv/z_biased (si use_enc_prior) ou vae_conv/z (sinon).
    // On encode le z réellement passé au décodeur (reparam + prior) plutôt que
    // mu brut, ce qui permet au mode --encode de lire directement le vecteur
    // latent opérationnel sans recalcul côté Lua.
    model.push("vae_conv/out_concat", "Concat", 0);
    if (auto* L = model.getLayerByName("vae_conv/out_concat")) {
        L->inputs = {"vae_conv/recon", z_in, "vae_conv/logvar"};
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
    // Décodeur purement convolutionnel : pas de normalisation émise.
    const std::string dec_norm_str = "none";
    const int dec_gn_groups_val = std::max(1, cfg.dec_gn_groups > 0 ? cfg.dec_gn_groups : cfg.enc_gn_groups);
    const std::string dec_upsample = cfg.decoder_upsample.empty() ? "conv_transpose" : cfg.decoder_upsample;

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
    model.modelConfig["dec_norm"] = dec_norm_str;
    model.modelConfig["dec_gn_groups"] = dec_gn_groups_val;
    model.modelConfig["decoder_upsample"] = dec_upsample;

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
        const int out_h = std::max(1, in_h * 2);
        const int out_w = std::max(1, in_w * 2);
        model.push(name, "UpsampleNearest", 0);
        if (auto* U = model.getLayerByName(name)) {
            U->inputs = {in};
            U->output = out;
            U->in_channels = channels;
            U->input_height = in_h;
            U->input_width = in_w;
            U->output_height = out_h;
            U->output_width = out_w;
            U->out_h = out_h;
            U->out_w = out_w;
            U->scale_h = 2.0f;
            U->scale_w = 2.0f;
        }
        return out;
    };

    auto deconv2d = [&](const std::string& name,
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
        const int out_h = std::max(1, (in_h - 1) * s - 2 * p + k);
        const int out_w = std::max(1, (in_w - 1) * s - 2 * p + k);
        model.push(name, "ConvTranspose2d",
                   sat_mul(static_cast<size_t>(out_c), sat_mul(static_cast<size_t>(in_c), sat_mul(static_cast<size_t>(k), static_cast<size_t>(k)))));
        if (auto* L = model.getLayerByName(name)) {
            L->inputs = {in};
            L->output = out;
            L->in_channels = in_c;
            L->out_channels = out_c;
            L->input_height = in_h;
            L->input_width = in_w;
            L->output_height = out_h;
            L->output_width = out_w;
            L->out_h = out_h;
            L->out_w = out_w;
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

    // Options blocs optionnels (purement convolutionnel : ResNet seulement)
    const bool use_resnet_dec  = cfg.use_attention;
    const int resnet_max_dec   = cfg.resnet_max_tokens;
    // Skip connections : si le checkpoint a été entraîné avec use_skip_connections=true,
    // on reconstruit le même graphe décodeur mais l'encodeur skip est remplacé par un
    // tenseur zéro (Constant non présent dans le checkpoint → initialisé à 0).
    // Résultat : skip_proj reçoit [decoder_feats, 0] → W_dec * decoder_feats
    // C'est bien meilleur que bypasser skip_proj entièrement.
    const bool use_skip_dec = cfg.use_skip_connections;
    // Stocker dans modelConfig pour que PonyXL puisse reconstruire une architecture identique.
    model.modelConfig["use_attention"] = use_resnet_dec;
    model.modelConfig["resnet_max_tokens"] = resnet_max_dec;
    model.modelConfig["use_skip_connections"] = use_skip_dec;

    auto resnet_gate_dec = [&](int h, int w) -> bool {
        if (!use_resnet_dec) return false;
        return resnet_max_dec <= 0 || (h * w) <= resnet_max_dec;
    };

    auto resblock_dec = [&](const std::string& prefix, const std::string& in, int ch, int h, int w) -> std::string {
        const size_t ps = sat_mul(static_cast<size_t>(ch), sat_mul(static_cast<size_t>(ch), 9));
        model.push(prefix + "/conv1", "Conv2d", ps);
        if (auto* L = model.getLayerByName(prefix + "/conv1")) {
            L->inputs = {in};        L->output = prefix + "/c1";
            L->in_channels = ch;    L->out_channels = ch;
            L->input_height = h;    L->input_width = w;
            L->kernel_size = 3;     L->stride = 1;  L->padding = 1;
            L->use_bias = false;
        }
        model.push(prefix + "/act1", "SiLU", 0);
        if (auto* A = model.getLayerByName(prefix + "/act1")) {
            A->inputs = {prefix + "/c1"};  A->output = prefix + "/c1a";
        }
        model.push(prefix + "/conv2", "Conv2d", ps);
        if (auto* L = model.getLayerByName(prefix + "/conv2")) {
            L->inputs = {prefix + "/c1a"};  L->output = prefix + "/c2";
            L->in_channels = ch;            L->out_channels = ch;
            L->input_height = h;            L->input_width = w;
            L->kernel_size = 3;             L->stride = 1;  L->padding = 1;
            L->use_bias = false;
        }
        model.push(prefix + "/add", "Add", 0);
        if (auto* A = model.getLayerByName(prefix + "/add")) {
            A->inputs = {prefix + "/c2", in};  A->output = prefix + "/out";
        }
        return prefix + "/out";
    };

    auto add_dec_norm = [&](const std::string& prefix, const std::string& in, int ch, int h, int w) -> std::string {
        if (dec_norm_str == "none" || dec_norm_str.empty()) return in;
        if (dec_norm_str == "groupnorm" || dec_norm_str == "gn") {
            int groups = dec_gn_groups_val;
            while (groups > 1 && ch % groups != 0) --groups;
            model.push(prefix + "/gn", "GroupNorm", static_cast<size_t>(ch) * 2);
            if (auto* L = model.getLayerByName(prefix + "/gn")) {
                L->inputs = {in};       L->output = prefix + "/gn_out";
                L->in_channels = ch;    L->num_groups = groups;
                L->input_height = h;    L->input_width = w;
            }
            return prefix + "/gn_out";
        }
        if (dec_norm_str == "layernorm" || dec_norm_str == "ln") {
            model.push(prefix + "/ln", "LayerNorm", static_cast<size_t>(ch) * 2);
            if (auto* L = model.getLayerByName(prefix + "/ln")) {
                L->inputs = {in};       L->output = prefix + "/ln_out";
                L->in_channels = ch;    L->input_height = h;  L->input_width = w;
            }
            return prefix + "/ln_out";
        }
        return in;
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
    y = add_dec_norm("vae_conv/dec/n0", y, dy_c, dy_h, dy_w);

    // Bottleneck décodeur (purement convolutionnel : ResNet seulement)
    if (resnet_gate_dec(dy_h, dy_w)) {
        y = add_dec_norm("vae_conv/dec/bot_n", y, dy_c, dy_h, dy_w);
        y = resblock_dec("vae_conv/dec/bot_res", y, dy_c, dy_h, dy_w);
    }

    for (int i = downsamples - 1; i >= 0; --i) {
        const std::string b = "vae_conv/dec/up" + std::to_string(i + 1);
        const int in_h = dy_h;
        const int in_w = dy_w;
        if (dec_upsample == "nearest_conv") {
            y = upsample2x(b + "/up", y, b + "/up_y", dy_c, in_h, in_w);
            dy_h = in_h * 2;
            dy_w = in_w * 2;
            y = conv2d(b + "/conv", y, b + "/c", dy_c, dy_c, dy_h, dy_w, 3, 1, 1, true);
        } else {
            y = deconv2d(b + "/up", y, b + "/up_y", dy_c, dy_c, in_h, in_w, 4, 2, 1, true);
            dy_h = in_h * 2;
            dy_w = in_w * 2;
        }
        // Skip connection avec encodeur zéro : charge les poids skip_proj du checkpoint
        // et applique à [decoder_feats, 0]. Le Constant n'est PAS dans le checkpoint
        // donc initialisé à zéro par l'allocateur → skip_proj applique uniquement
        // la moitié "decoder" de ses poids, ce qui est bien plus fidèle au modèle
        // entraîné qu'ignorer skip_proj entièrement.
        if (use_skip_dec) {
            const std::string zero_name = b + "/zero_enc_skip";
            model.push(zero_name, "Constant",
                       static_cast<size_t>(dy_c) * static_cast<size_t>(dy_h) * static_cast<size_t>(dy_w));
            if (auto* L = model.getLayerByName(zero_name)) {
                L->inputs  = {};
                L->output  = zero_name + "_out";
                L->in_channels  = dy_c;
                L->out_channels = dy_c;
                L->input_height = dy_h;
                L->input_width  = dy_w;
            }
            model.push(b + "/skip_cat", "Concat", 0);
            if (auto* L = model.getLayerByName(b + "/skip_cat")) {
                L->inputs      = {y, zero_name + "_out"};
                L->output      = b + "/sc";
                L->concat_axis = 0;
            }
            y = conv2d(b + "/skip_proj", b + "/sc", b + "/sp",
                       2 * dy_c, dy_c, dy_h, dy_w, 1, 1, 0, /*act=*/false);
        }
        y = add_dec_norm(b + "/n", y, dy_c, dy_h, dy_w);
        if (resnet_gate_dec(dy_h, dy_w)) {
            y = resblock_dec(b + "/res", y, dy_c, dy_h, dy_w);
        }
    }

    y = add_dec_norm("vae_conv/dec/out_n", y, dy_c, dy_h, dy_w);
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
