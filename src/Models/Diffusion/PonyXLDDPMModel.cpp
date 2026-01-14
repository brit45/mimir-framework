#include "PonyXLDDPMModel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <random>

namespace {
static inline int blur_radius_for_level(int level_percent) {
    const int percent = std::max(0, level_percent);
    return (percent <= 0) ? 0 : std::clamp(static_cast<int>(std::round(static_cast<float>(percent) * 0.125f)), 1, 8);
}
} // namespace


PonyXLDDPMModel::PonyXLDDPMModel() {
    setModelName("PonyXLDDPMModel");
    // On utilise explicitement un encoder (embeddings texte) dans ce modèle.
    setHasEncoder(true);
}

void PonyXLDDPMModel::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
}

PonyXLDDPMModel::StepStats PonyXLDDPMModel::trainStep(const std::vector<float>& input,
                                                      const std::vector<float>& target,
                                                      Optimizer& opt,
                                                      float learning_rate) {
    if (layers.empty()) {
        throw std::runtime_error("PonyXLDDPMModel::trainStep: model not built");
    }
    if (layer_weight_blocks.empty()) {
        throw std::runtime_error("PonyXLDDPMModel::trainStep: weights not allocated (call allocateParams/initWeights)");
    }

    zeroGradients();

    std::vector<float> prediction = forwardPass(input, true);

    StepStats stats;
    stats.loss = computeLoss(prediction, target, "mse");

    std::vector<float> loss_grad = computeLossGradient(prediction, target, "mse");
    backwardPass(loss_grad);

    double sum_sq = 0.0;
    float max_abs = 0.0f;
    for (const auto& layer : layers) {
        for (float g : layer.grad_weights) {
            sum_sq += static_cast<double>(g) * static_cast<double>(g);
            const float a = std::abs(g);
            if (a > max_abs) max_abs = a;
        }
    }
    stats.grad_norm = static_cast<float>(std::sqrt(sum_sq));
    stats.grad_max_abs = max_abs;

    optimizerStep(opt, learning_rate);
    return stats;
}

static std::vector<float> box_blur_rgb_f32(
    const std::vector<float>& img,
    int w,
    int h,
    int c,
    int radius
) {
    if (img.empty() || w <= 0 || h <= 0 || c <= 0) return img;
    if (radius <= 0) return img;

    std::vector<float> out(img.size(), 0.0f);
    const int r = radius;
    const int diam = 2 * r + 1;
    const float inv = 1.0f / static_cast<float>(diam * diam);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int ch = 0; ch < c; ++ch) {
                float acc = 0.0f;
                for (int ky = -r; ky <= r; ++ky) {
                    const int yy = std::clamp(y + ky, 0, h - 1);
                    for (int kx = -r; kx <= r; ++kx) {
                        const int xx = std::clamp(x + kx, 0, w - 1);
                        const size_t idx = (static_cast<size_t>(yy) * static_cast<size_t>(w) + static_cast<size_t>(xx)) * static_cast<size_t>(c) + static_cast<size_t>(ch);
                        acc += img[idx];
                    }
                }
                out[(static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)) * static_cast<size_t>(c) + static_cast<size_t>(ch)] = acc * inv;
            }
        }
    }
    return out;
}

PonyXLDDPMModel::StepStats PonyXLDDPMModel::trainStepTripleFault(
    const std::string& prompt,
    const std::vector<uint8_t>& rgb,
    int w,
    int h,
    Optimizer& opt,
    float learning_rate
) {
    if (layers.empty()) {
        throw std::runtime_error("PonyXLDDPMModel::trainStepTripleFault: model not built");
    }
    if (layer_weight_blocks.empty()) {
        throw std::runtime_error("PonyXLDDPMModel::trainStepTripleFault: weights not allocated (call allocateParams/initWeights)");
    }

    const int W = std::max(1, w);
    const int H = std::max(1, h);
    const int C = std::max(1, cfg_.image_c);
    const int D = std::max(1, cfg_.d_model);

    // RNG local pour les différents dropouts (prompt/image)
    static thread_local std::mt19937 rng(1337);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    // CFG-like dropout: parfois on retire le prompt
    std::string used_prompt = prompt;
    if (cfg_.cfg_dropout_prob > 0.0f) {
        if (u01(rng) < cfg_.cfg_dropout_prob) used_prompt.clear();
    }

    // Tokenizer -> tokens
    std::vector<int> tokens = getMutableTokenizer().tokenizeEnsure(used_prompt);
    if (cfg_.seq_len > 0) {
        if (tokens.size() > static_cast<size_t>(cfg_.seq_len)) {
            tokens.resize(static_cast<size_t>(cfg_.seq_len));
        } else if (tokens.size() < static_cast<size_t>(cfg_.seq_len)) {
            tokens.resize(static_cast<size_t>(cfg_.seq_len), getMutableTokenizer().getPadId());
        }
    }

    // Encoder (learnable embeddings) -> text vector (dim = d_model)
    // IMPORTANT: ne pas pré-positionner vocab_size avant ensureVocabSize,
    // sinon ensureVocabSize devient un no-op et token_embeddings reste vide.
    Encoder& enc = getMutableEncoder();
    if (enc.dim != D) {
        enc.dim = D;
        enc.vocab_size = 0;
        enc.token_embeddings.clear();
        enc.setSeqEmbedding({});
        enc.setModEmbedding({});
        enc.setMagEmbedding({});
    }
    enc.ensureVocabSize(static_cast<size_t>(cfg_.max_vocab), 0xC0FFEEu);
    // Assurer l'existence des embeddings spéciaux (appris pendant l'entraînement)
    enc.ensureSpecialEmbeddings(0x51A5EEDu);
    std::vector<float> text_vec = enc.encode(tokens);
    if (text_vec.size() != static_cast<size_t>(D)) {
        text_vec.resize(static_cast<size_t>(D), 0.0f);
    }

    // Original image target (RGB float in [-1,1])
    std::vector<float> x0 = imageBytesToFloatRGB(rgb, W, H);
    if (x0.size() != static_cast<size_t>(W * H * C)) {
        x0.resize(static_cast<size_t>(W * H * C), 0.0f);
    }

    const int levels = std::max(1, cfg_.blur_levels);
    const int micro_steps = 1 + levels; // original + 1..levels

    double loss_sum = 0.0;
    double grad_norm_sum = 0.0;
    float grad_max_abs = 0.0f;

    // Entrainer: 0% blur (identity) + 1..levels
    for (int lv = 0; lv <= levels; ++lv) {
        const int percent = lv; // 0..levels
        const int radius = blur_radius_for_level(percent);

        std::vector<float> x_in;
        if (radius <= 0) {
            x_in = x0;
        } else {
            x_in = box_blur_rgb_f32(x0, W, H, C, radius);
        }

        // Dropout de l'image conditionnelle: empêche la solution triviale "ignore le texte"
        // et force le modèle à apprendre des correspondances texte->image.
        bool drop_cond_image = false;
        if (cfg_.cond_image_dropout_prob > 0.0f) {
            drop_cond_image = (u01(rng) < cfg_.cond_image_dropout_prob);
        }
        if (drop_cond_image) {
            if (cfg_.cond_image_dropout_noise_std > 0.0f) {
                std::normal_distribution<float> n01(0.0f, cfg_.cond_image_dropout_noise_std);
                for (auto& v : x_in) v = n01(rng);
            } else {
                std::fill(x_in.begin(), x_in.end(), 0.0f);
            }
        }

        std::vector<float> input;
        input.reserve(text_vec.size() + x_in.size());
        input.insert(input.end(), text_vec.begin(), text_vec.end());
        input.insert(input.end(), x_in.begin(), x_in.end());

        const float lr_eff = drop_cond_image ? (learning_rate * std::clamp(cfg_.cond_image_dropout_lr_scale, 0.0f, 1.0f)) : learning_rate;
        StepStats st = trainStep(input, x0, opt, lr_eff);
        loss_sum += st.loss;
        grad_norm_sum += st.grad_norm;
        if (st.grad_max_abs > grad_max_abs) grad_max_abs = st.grad_max_abs;

        // Mise à jour de l'Encoder à partir du gradient d'entrée (partie texte).
        // Hypothèse: enc.encode() est essentiellement une somme des embeddings (hors PAD),
        // donc le gradient sur le vecteur texte se redistribue aux tokens présents.
        if (hasLastInputGradient()) {
            const auto& gin = getLastInputGradient();
            if (gin.size() >= static_cast<size_t>(D)) {
                // gradient sur la partie texte (text_vec)
                std::vector<float> grad_text(static_cast<size_t>(D), 0.0f);
                std::copy_n(gin.data(), static_cast<size_t>(D), grad_text.data());

                // Pooling pondéré (doit matcher Encoder::encode)
                const int pad = getMutableTokenizer().getPadId();
                float wsum = 0.0f;
                int pos = 0;
                for (int id : tokens) {
                    if (id < 0 || id == pad || id >= enc.vocab_size) { pos++; continue; }
                    const float w = (pos < enc.magik_prefix_count && enc.magik_prefix_weight > 0.0f) ? enc.magik_prefix_weight : 1.0f;
                    wsum += w;
                    pos++;
                }
                const float inv_wsum = (wsum > 0.0f) ? (1.0f / wsum) : 0.0f;

                pos = 0;
                for (int id : tokens) {
                    if (id < 0 || id == pad || id >= enc.vocab_size) { pos++; continue; }
                    const float w = (pos < enc.magik_prefix_count && enc.magik_prefix_weight > 0.0f) ? enc.magik_prefix_weight : 1.0f;
                    const float scale = w * inv_wsum;
                    const size_t base = static_cast<size_t>(id) * static_cast<size_t>(D);
                    for (int d = 0; d < D; ++d) {
                        // SGD simple sur embeddings (même lr que le modèle) + pondération cohérente
                        enc.token_embeddings[base + static_cast<size_t>(d)] -= lr_eff * grad_text[static_cast<size_t>(d)] * scale;
                    }
                    pos++;
                }

                // MàJ des embeddings spéciaux (ils sont ajoutés directement à text_vec)
                enc.sgdUpdateSpecialEmbeddings(grad_text, lr_eff, true, true, true);
            }
        }
    }

    StepStats out;
    out.loss = static_cast<float>(loss_sum / static_cast<double>(micro_steps));
    out.grad_norm = static_cast<float>(grad_norm_sum / static_cast<double>(micro_steps));
    out.grad_max_abs = grad_max_abs;
    return out;
}

void PonyXLDDPMModel::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("PonyXLDDPMModel");
    model.modelConfig["type"] = "ponyxl_ddpm";

    const int d_model = std::max(1, cfg.d_model);
    const int seq_text = std::max(1, cfg.seq_len);
    const int image_w = std::max(1, cfg.image_w);
    const int image_h = std::max(1, cfg.image_h);
    const int image_c = std::max(1, cfg.image_c);
    const int image_dim = image_w * image_h * image_c;

    model.modelConfig["task"] = "prompt_to_image_autoencoder";
    model.modelConfig["image_w"] = image_w;
    model.modelConfig["image_h"] = image_h;
    model.modelConfig["image_c"] = image_c;
    model.modelConfig["image_dim"] = image_dim;
    model.modelConfig["d_model"] = d_model;
    model.modelConfig["seq_len"] = seq_text;
    model.modelConfig["latent_dim"] = std::max(1, cfg.latent_dim);
    model.modelConfig["hidden_dim"] = std::max(1, cfg.hidden_dim);
    model.modelConfig["blur_levels"] = std::max(1, cfg.blur_levels);
    model.modelConfig["max_vocab"] = std::max(1, cfg.max_vocab);

    // I/O: input = [text_vec(d_model), image_rgb(w*h*c)] -> output = image_rgb
    const int in_dim = d_model + image_dim;
    model.modelConfig["input_dim"] = in_dim;
    model.modelConfig["output_dim"] = image_dim;

    // Marquer la présence d'un Encoder (embeddings texte) dans le checkpoint
    model.setHasEncoder(true);
    {
        auto& enc = model.getMutableEncoder();
        if (enc.dim != d_model) {
            enc.dim = d_model;
            enc.vocab_size = 0;
            enc.token_embeddings.clear();
            enc.setSeqEmbedding({});
            enc.setModEmbedding({});
            enc.setMagEmbedding({});
        }
        enc.ensureVocabSize(static_cast<size_t>(std::max(1, cfg.max_vocab)), 0xC0FFEEu);
        enc.ensureSpecialEmbeddings(0x51A5EEDu);
    }
    model.getMutableTokenizer();

    // Routing input
    model.push("ponyxl_t2i/raw_in", "Identity", 0);
    if (auto* L = model.getLayerByName("ponyxl_t2i/raw_in")) {
        L->inputs = {"__input__"};
        L->output = "ponyxl_t2i/in";
    }

    const int hidden = std::max(64, cfg.hidden_dim);
    const int latent = std::max(16, cfg.latent_dim);

    // Encoder -> latent
    model.push("ponyxl_t2i/enc_fc1", "Linear", static_cast<size_t>(in_dim) * static_cast<size_t>(hidden) + static_cast<size_t>(hidden));
    if (auto* L = model.getLayerByName("ponyxl_t2i/enc_fc1")) {
        L->inputs = {"ponyxl_t2i/in"};
        L->output = "ponyxl_t2i/enc/h1";
        L->in_features = in_dim;
        L->out_features = hidden;
        L->use_bias = true;
    }
    model.push("ponyxl_t2i/enc_act1", "GELU", 0);
    if (auto* L = model.getLayerByName("ponyxl_t2i/enc_act1")) {
        L->inputs = {"ponyxl_t2i/enc/h1"};
        L->output = "ponyxl_t2i/enc/h1_act";
    }
    model.push("ponyxl_t2i/latent_fc", "Linear", static_cast<size_t>(hidden) * static_cast<size_t>(latent) + static_cast<size_t>(latent));
    if (auto* L = model.getLayerByName("ponyxl_t2i/latent_fc")) {
        L->inputs = {"ponyxl_t2i/enc/h1_act"};
        L->output = "ponyxl_t2i/latent";
        L->in_features = hidden;
        L->out_features = latent;
        L->use_bias = true;
    }
    model.push("ponyxl_t2i/latent_act", "GELU", 0);
    if (auto* L = model.getLayerByName("ponyxl_t2i/latent_act")) {
        L->inputs = {"ponyxl_t2i/latent"};
        L->output = "ponyxl_t2i/latent_act";
    }

    // Decoder -> image
    model.push("ponyxl_t2i/dec_fc1", "Linear", static_cast<size_t>(latent) * static_cast<size_t>(hidden) + static_cast<size_t>(hidden));
    if (auto* L = model.getLayerByName("ponyxl_t2i/dec_fc1")) {
        L->inputs = {"ponyxl_t2i/latent_act"};
        L->output = "ponyxl_t2i/dec/h1";
        L->in_features = latent;
        L->out_features = hidden;
        L->use_bias = true;
    }
    model.push("ponyxl_t2i/dec_act1", "GELU", 0);
    if (auto* L = model.getLayerByName("ponyxl_t2i/dec_act1")) {
        L->inputs = {"ponyxl_t2i/dec/h1"};
        L->output = "ponyxl_t2i/dec/h1_act";
    }
    model.push("ponyxl_t2i/out_fc", "Linear", static_cast<size_t>(hidden) * static_cast<size_t>(image_dim) + static_cast<size_t>(image_dim));
    if (auto* L = model.getLayerByName("ponyxl_t2i/out_fc")) {
        L->inputs = {"ponyxl_t2i/dec/h1_act"};
        L->output = "ponyxl_t2i/out_pre";
        L->in_features = hidden;
        L->out_features = image_dim;
        L->use_bias = true;
    }
    model.push("ponyxl_t2i/out_tanh", "Tanh", 0);
    if (auto* L = model.getLayerByName("ponyxl_t2i/out_tanh")) {
        L->inputs = {"ponyxl_t2i/out_pre"};
        L->output = "x";
    }
}

std::vector<float> PonyXLDDPMModel::imageBytesToFloatRGB(const std::vector<uint8_t>& rgb, int w, int h) {
    const int W = std::max(1, w);
    const int H = std::max(1, h);
    const size_t outN = static_cast<size_t>(W) * static_cast<size_t>(H) * 3ULL;
    std::vector<float> y(outN, 0.0f);

    const size_t limit = std::min(outN, rgb.size());
    for (size_t i = 0; i < limit; ++i) {
        y[i] = (static_cast<float>(rgb[i]) / 255.0f) * 2.0f - 1.0f;
    }
    return y;
}
