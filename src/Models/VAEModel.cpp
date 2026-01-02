#include "VAEModel.hpp"
#include "../LayerOps.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

VAEModel::VAEModel() {
    setModelName("VAEModel");
}

VAEModel::~VAEModel() = default;

void VAEModel::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("VAEModel");
    model.modelConfig["type"] = "vae";
    model.modelConfig["input_dim"] = cfg.input_dim;
    model.modelConfig["latent_dim"] = cfg.latent_dim;
    model.modelConfig["kl_beta"] = cfg.kl_beta;

    // Encoder MLP
    int prev = cfg.input_dim;
    for (size_t i = 0; i < cfg.encoder_hidden.size(); ++i) {
        const int h = cfg.encoder_hidden[i];
        const std::string fc = "vae/enc_fc" + std::to_string(i);
        model.push(fc, "Linear", static_cast<size_t>(prev) * static_cast<size_t>(h) + static_cast<size_t>(h));
        if (auto* L = model.getLayerByName(fc)) {
            L->in_features = prev;
            L->out_features = h;
            L->use_bias = true;
        }

        const std::string act = "vae/enc_act" + std::to_string(i);
        // On encode l'activation comme un layer explicite pour la serialization/architecture.
        // Pas de params.
        switch (cfg.activation) {
            case ActivationType::RELU: model.push(act, "ReLU", 0); break;
            case ActivationType::GELU: model.push(act, "GELU", 0); break;
            case ActivationType::TANH: model.push(act, "Tanh", 0); break;
            case ActivationType::SIGMOID: model.push(act, "Sigmoid", 0); break;
            case ActivationType::SWISH: model.push(act, "SiLU", 0); break;
            default: model.push(act, "ReLU", 0); break;
        }

        prev = h;
    }

    // Latent heads
    model.push("vae/mu", "Linear", static_cast<size_t>(prev) * static_cast<size_t>(cfg.latent_dim) + static_cast<size_t>(cfg.latent_dim));
    if (auto* L = model.getLayerByName("vae/mu")) {
        L->in_features = prev;
        L->out_features = cfg.latent_dim;
        L->use_bias = true;
    }

    model.push("vae/logvar", "Linear", static_cast<size_t>(prev) * static_cast<size_t>(cfg.latent_dim) + static_cast<size_t>(cfg.latent_dim));
    if (auto* L = model.getLayerByName("vae/logvar")) {
        L->in_features = prev;
        L->out_features = cfg.latent_dim;
        L->use_bias = true;
    }

    // Decoder MLP
    prev = cfg.latent_dim;
    for (size_t i = 0; i < cfg.decoder_hidden.size(); ++i) {
        const int h = cfg.decoder_hidden[i];
        const std::string fc = "vae/dec_fc" + std::to_string(i);
        model.push(fc, "Linear", static_cast<size_t>(prev) * static_cast<size_t>(h) + static_cast<size_t>(h));
        if (auto* L = model.getLayerByName(fc)) {
            L->in_features = prev;
            L->out_features = h;
            L->use_bias = true;
        }

        const std::string act = "vae/dec_act" + std::to_string(i);
        switch (cfg.activation) {
            case ActivationType::RELU: model.push(act, "ReLU", 0); break;
            case ActivationType::GELU: model.push(act, "GELU", 0); break;
            case ActivationType::TANH: model.push(act, "Tanh", 0); break;
            case ActivationType::SIGMOID: model.push(act, "Sigmoid", 0); break;
            case ActivationType::SWISH: model.push(act, "SiLU", 0); break;
            default: model.push(act, "ReLU", 0); break;
        }

        prev = h;
    }

    model.push("vae/out", "Linear", static_cast<size_t>(prev) * static_cast<size_t>(cfg.input_dim) + static_cast<size_t>(cfg.input_dim));
    if (auto* L = model.getLayerByName("vae/out")) {
        L->in_features = prev;
        L->out_features = cfg.input_dim;
        L->use_bias = true;
    }
}

void VAEModel::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
    refreshLayerPtrs();
}

void VAEModel::refreshLayerPtrs() {
    enc_fcs_.clear();
    dec_fcs_.clear();
    mu_fc_ = getLayerByName("vae/mu");
    logvar_fc_ = getLayerByName("vae/logvar");
    out_fc_ = getLayerByName("vae/out");

    for (size_t i = 0; i < cfg_.encoder_hidden.size(); ++i) {
        enc_fcs_.push_back(getLayerByName("vae/enc_fc" + std::to_string(i)));
    }
    for (size_t i = 0; i < cfg_.decoder_hidden.size(); ++i) {
        dec_fcs_.push_back(getLayerByName("vae/dec_fc" + std::to_string(i)));
    }

    auto require = [&](Layer* p, const std::string& name) {
        if (!p) throw std::runtime_error("VAEModel: layer missing: " + name);
    };

    for (size_t i = 0; i < enc_fcs_.size(); ++i) require(enc_fcs_[i], "vae/enc_fc" + std::to_string(i));
    for (size_t i = 0; i < dec_fcs_.size(); ++i) require(dec_fcs_[i], "vae/dec_fc" + std::to_string(i));
    require(mu_fc_, "vae/mu");
    require(logvar_fc_, "vae/logvar");
    require(out_fc_, "vae/out");
}

void VAEModel::apply_activation_inplace(std::vector<float>& v, ActivationType act) {
    switch (act) {
        case ActivationType::RELU:
            for (auto& x : v) x = std::max(0.0f, x);
            break;
        case ActivationType::GELU: {
            // Approx GELU (tanh)
            for (auto& x : v) {
                const float c = 0.044715f;
                const float s = std::sqrt(2.0f / 3.14159265359f);
                float u = s * (x + c * x * x * x);
                x = 0.5f * x * (1.0f + std::tanh(u));
            }
            break;
        }
        case ActivationType::TANH:
            for (auto& x : v) x = std::tanh(x);
            break;
        case ActivationType::SIGMOID:
            for (auto& x : v) x = 1.0f / (1.0f + std::exp(-x));
            break;
        case ActivationType::SWISH:
            for (auto& x : v) {
                float s = 1.0f / (1.0f + std::exp(-x));
                x = x * s;
            }
            break;
        default:
            for (auto& x : v) x = std::max(0.0f, x);
            break;
    }
}

void VAEModel::activation_backward_inplace(std::vector<float>& grad, const std::vector<float>& preact, ActivationType act) {
    if (grad.size() != preact.size()) return;

    switch (act) {
        case ActivationType::RELU:
            for (size_t i = 0; i < grad.size(); ++i) {
                if (preact[i] <= 0.0f) grad[i] = 0.0f;
            }
            break;
        case ActivationType::TANH:
            for (size_t i = 0; i < grad.size(); ++i) {
                float t = std::tanh(preact[i]);
                grad[i] *= (1.0f - t * t);
            }
            break;
        case ActivationType::SIGMOID:
            for (size_t i = 0; i < grad.size(); ++i) {
                float s = 1.0f / (1.0f + std::exp(-preact[i]));
                grad[i] *= s * (1.0f - s);
            }
            break;
        case ActivationType::SWISH:
            for (size_t i = 0; i < grad.size(); ++i) {
                float s = 1.0f / (1.0f + std::exp(-preact[i]));
                grad[i] *= (s + preact[i] * s * (1.0f - s));
            }
            break;
        case ActivationType::GELU:
        default:
            // Fallback: approx derivative via numeric-ish stable approximation (OK pour entrainement simple)
            for (size_t i = 0; i < grad.size(); ++i) {
                float x = preact[i];
                float c = 0.044715f;
                float s = std::sqrt(2.0f / 3.14159265359f);
                float u = s * (x + c * x * x * x);
                float t = std::tanh(u);
                float sech2 = 1.0f - t * t;
                float du_dx = s * (1.0f + 3.0f * c * x * x);
                float dgelu = 0.5f * (1.0f + t) + 0.5f * x * sech2 * du_dx;
                grad[i] *= dgelu;
            }
            break;
    }
}

void VAEModel::linear_backward_accum(const Layer& layer,
                                    const std::vector<float>& input,
                                    const std::vector<float>& grad_out,
                                    std::vector<float>& grad_in,
                                    std::vector<float>& grad_w_out) {
    const int in_f = layer.in_features > 0 ? layer.in_features : static_cast<int>(input.size());
    const int out_f = layer.out_features;
    if (out_f <= 0) throw std::runtime_error("VAEModel: Linear out_features not set");
    if (static_cast<int>(input.size()) != in_f || static_cast<int>(grad_out.size()) != out_f) {
        throw std::runtime_error("VAEModel: Linear backward shape mismatch");
    }

    const float* W = layer.getWeights();
    const bool use_bias = layer.use_bias;

    // grad_in = W^T * grad_out
    grad_in.assign(static_cast<size_t>(in_f), 0.0f);
    for (int o = 0; o < out_f; ++o) {
        const float go = grad_out[o];
        const float* wrow = W + o * in_f;
        for (int i = 0; i < in_f; ++i) {
            grad_in[i] += wrow[i] * go;
        }
    }

    // grad weights/bias into grad_w_out (layout: W then bias)
    const size_t w_count = static_cast<size_t>(in_f) * static_cast<size_t>(out_f);
    const size_t total = w_count + (use_bias ? static_cast<size_t>(out_f) : 0);
    if (grad_w_out.size() != total) grad_w_out.assign(total, 0.0f);

    // dW[o,i] += go * x[i]
    for (int o = 0; o < out_f; ++o) {
        const float go = grad_out[o];
        const size_t row_off = static_cast<size_t>(o) * static_cast<size_t>(in_f);
        for (int i = 0; i < in_f; ++i) {
            grad_w_out[row_off + static_cast<size_t>(i)] += go * input[static_cast<size_t>(i)];
        }
    }

    // db[o] += go
    if (use_bias) {
        for (int o = 0; o < out_f; ++o) {
            grad_w_out[w_count + static_cast<size_t>(o)] += grad_out[o];
        }
    }
}

std::vector<float> VAEModel::inferMu(const std::vector<float>& x) const {
    if (static_cast<int>(x.size()) != cfg_.input_dim) {
        throw std::runtime_error("VAEModel::inferMu: input_dim mismatch");
    }

    std::vector<float> h = x;

    // Encoder
    for (size_t i = 0; i < cfg_.encoder_hidden.size(); ++i) {
        const Layer* fc = const_cast<VAEModel*>(this)->getLayerByName("vae/enc_fc" + std::to_string(i));
        if (!fc) throw std::runtime_error("VAEModel::inferMu: missing encoder fc");
        h = LayerOps::linear_forward(h, *fc, false);
        apply_activation_inplace(h, cfg_.activation);
    }

    const Layer* mu_fc = const_cast<VAEModel*>(this)->getLayerByName("vae/mu");
    if (!mu_fc) throw std::runtime_error("VAEModel::inferMu: missing mu");
    return LayerOps::linear_forward(h, *mu_fc, false);
}

std::vector<float> VAEModel::inferReconstruction(const std::vector<float>& x, uint32_t seed) const {
    if (static_cast<int>(x.size()) != cfg_.input_dim) {
        throw std::runtime_error("VAEModel::inferReconstruction: input_dim mismatch");
    }

    // Encoder -> mu/logvar
    std::vector<float> h = x;
    for (size_t i = 0; i < cfg_.encoder_hidden.size(); ++i) {
        const Layer* fc = const_cast<VAEModel*>(this)->getLayerByName("vae/enc_fc" + std::to_string(i));
        if (!fc) throw std::runtime_error("VAEModel::inferReconstruction: missing encoder fc");
        h = LayerOps::linear_forward(h, *fc, false);
        apply_activation_inplace(h, cfg_.activation);
    }

    const Layer* mu_fc = const_cast<VAEModel*>(this)->getLayerByName("vae/mu");
    const Layer* lv_fc = const_cast<VAEModel*>(this)->getLayerByName("vae/logvar");
    if (!mu_fc || !lv_fc) throw std::runtime_error("VAEModel::inferReconstruction: missing latent heads");

    std::vector<float> mu = LayerOps::linear_forward(h, *mu_fc, false);
    std::vector<float> logvar = LayerOps::linear_forward(h, *lv_fc, false);

    // z
    std::vector<float> z(mu.size(), 0.0f);
    if (cfg_.use_mean_in_infer) {
        z = mu;
    } else {
        std::mt19937 rng(seed ? seed : cfg_.seed);
        std::normal_distribution<float> N(0.0f, 1.0f);
        for (size_t i = 0; i < z.size(); ++i) {
            float stdv = std::exp(0.5f * logvar[i]);
            z[i] = mu[i] + stdv * N(rng);
        }
    }

    // Decoder
    std::vector<float> d = z;
    for (size_t i = 0; i < cfg_.decoder_hidden.size(); ++i) {
        const Layer* fc = const_cast<VAEModel*>(this)->getLayerByName("vae/dec_fc" + std::to_string(i));
        if (!fc) throw std::runtime_error("VAEModel::inferReconstruction: missing decoder fc");
        d = LayerOps::linear_forward(d, *fc, false);
        apply_activation_inplace(d, cfg_.activation);
    }

    const Layer* out_fc = const_cast<VAEModel*>(this)->getLayerByName("vae/out");
    if (!out_fc) throw std::runtime_error("VAEModel::inferReconstruction: missing out");
    return LayerOps::linear_forward(d, *out_fc, false);
}

VAEModel::StepStats VAEModel::trainStep(const std::vector<float>& x, Optimizer& opt, float learning_rate, uint32_t seed) {
    if (static_cast<int>(x.size()) != cfg_.input_dim) {
        throw std::runtime_error("VAEModel::trainStep: input_dim mismatch");
    }
    if (layers.empty()) {
        throw std::runtime_error("VAEModel::trainStep: model not built");
    }
    if (layer_weight_blocks.empty()) {
        throw std::runtime_error("VAEModel::trainStep: weights not allocated (call allocateParams/initWeights)");
    }

    refreshLayerPtrs();

    // ---------------- Forward ----------------
    std::vector<LinearCache> enc_cache;
    enc_cache.reserve(cfg_.encoder_hidden.size());

    std::vector<float> h = x;
    for (size_t i = 0; i < cfg_.encoder_hidden.size(); ++i) {
        LinearCache c;
        c.input = h;
        c.preact = LayerOps::linear_forward(h, *enc_fcs_[i], true);
        c.output = c.preact;
        apply_activation_inplace(c.output, cfg_.activation);
        h = c.output;
        enc_cache.push_back(std::move(c));
    }

    std::vector<float> mu = LayerOps::linear_forward(h, *mu_fc_, true);
    std::vector<float> logvar = LayerOps::linear_forward(h, *logvar_fc_, true);

    std::mt19937 rng(seed ? seed : cfg_.seed);
    std::normal_distribution<float> N(0.0f, 1.0f);

    std::vector<float> eps(mu.size(), 0.0f);
    std::vector<float> stdv(mu.size(), 0.0f);
    std::vector<float> z(mu.size(), 0.0f);

    for (size_t i = 0; i < mu.size(); ++i) {
        eps[i] = N(rng);
        stdv[i] = std::exp(0.5f * logvar[i]);
        z[i] = mu[i] + stdv[i] * eps[i];
    }

    std::vector<LinearCache> dec_cache;
    dec_cache.reserve(cfg_.decoder_hidden.size());

    std::vector<float> d = z;
    for (size_t i = 0; i < cfg_.decoder_hidden.size(); ++i) {
        LinearCache c;
        c.input = d;
        c.preact = LayerOps::linear_forward(d, *dec_fcs_[i], true);
        c.output = c.preact;
        apply_activation_inplace(c.output, cfg_.activation);
        d = c.output;
        dec_cache.push_back(std::move(c));
    }

    std::vector<float> x_hat = LayerOps::linear_forward(d, *out_fc_, true);

    // ---------------- Loss ----------------
    StepStats stats;

    // Recon MSE
    float recon = 0.0f;
    for (size_t i = 0; i < x_hat.size(); ++i) {
        float diff = x_hat[i] - x[i];
        recon += diff * diff;
    }
    recon /= static_cast<float>(x_hat.size());

    // KL
    float kl = 0.0f;
    for (size_t i = 0; i < mu.size(); ++i) {
        // 0.5*(mu^2 + exp(logvar) - 1 - logvar)
        kl += 0.5f * (mu[i] * mu[i] + std::exp(logvar[i]) - 1.0f - logvar[i]);
    }
    kl /= static_cast<float>(mu.size());

    stats.recon_loss = recon;
    stats.kl_loss = kl;
    stats.loss = recon + cfg_.kl_beta * kl;

    // ---------------- Backward ----------------
    // Zero grads (uniquement ce qu'on utilise)
    for (auto* L : enc_fcs_) {
        if (!L) continue;
        if (L->grad_weights.size() != L->getWeightsSize()) L->grad_weights.assign(L->getWeightsSize(), 0.0f);
        else std::fill(L->grad_weights.begin(), L->grad_weights.end(), 0.0f);
    }
    for (auto* L : dec_fcs_) {
        if (!L) continue;
        if (L->grad_weights.size() != L->getWeightsSize()) L->grad_weights.assign(L->getWeightsSize(), 0.0f);
        else std::fill(L->grad_weights.begin(), L->grad_weights.end(), 0.0f);
    }
    for (auto* L : {mu_fc_, logvar_fc_, out_fc_}) {
        if (!L) continue;
        if (L->grad_weights.size() != L->getWeightsSize()) L->grad_weights.assign(L->getWeightsSize(), 0.0f);
        else std::fill(L->grad_weights.begin(), L->grad_weights.end(), 0.0f);
    }

    // dL/dx_hat
    std::vector<float> grad_xhat(x_hat.size(), 0.0f);
    const float mse_scale = 2.0f / static_cast<float>(x_hat.size());
    for (size_t i = 0; i < x_hat.size(); ++i) {
        grad_xhat[i] = mse_scale * (x_hat[i] - x[i]);
    }

    // Backprop out_fc
    std::vector<float> grad_d;
    std::vector<float> out_gradw;
    linear_backward_accum(*out_fc_, d, grad_xhat, grad_d, out_gradw);
    for (size_t i = 0; i < out_fc_->grad_weights.size() && i < out_gradw.size(); ++i) {
        out_fc_->grad_weights[i] += out_gradw[i];
    }

    // Decoder layers (reverse)
    for (int i = static_cast<int>(cfg_.decoder_hidden.size()) - 1; i >= 0; --i) {
        // activation backward
        activation_backward_inplace(grad_d, dec_cache[static_cast<size_t>(i)].preact, cfg_.activation);

        std::vector<float> grad_in;
        std::vector<float> gradw;
        linear_backward_accum(*dec_fcs_[static_cast<size_t>(i)], dec_cache[static_cast<size_t>(i)].input, grad_d, grad_in, gradw);

        // accumulate
        auto* L = dec_fcs_[static_cast<size_t>(i)];
        for (size_t k = 0; k < L->grad_weights.size() && k < gradw.size(); ++k) {
            L->grad_weights[k] += gradw[k];
        }

        grad_d = std::move(grad_in);
    }

    // grad w.r.t z
    std::vector<float> grad_z = grad_d;

    // reparam gradients
    std::vector<float> grad_mu(mu.size(), 0.0f);
    std::vector<float> grad_logvar(mu.size(), 0.0f);

    // KL grads (moyenné)
    const float kl_scale = cfg_.kl_beta / static_cast<float>(mu.size());
    for (size_t i = 0; i < mu.size(); ++i) {
        grad_mu[i] = grad_z[i] + kl_scale * mu[i];
        // d/dlogvar: 0.5*(exp(logvar) - 1)
        grad_logvar[i] = (grad_z[i] * eps[i] * (0.5f * stdv[i])) + kl_scale * 0.5f * (std::exp(logvar[i]) - 1.0f);
    }

    // Backprop mu/logvar heads into h
    std::vector<float> grad_h_mu;
    std::vector<float> mu_gradw;
    linear_backward_accum(*mu_fc_, h, grad_mu, grad_h_mu, mu_gradw);
    for (size_t k = 0; k < mu_fc_->grad_weights.size() && k < mu_gradw.size(); ++k) {
        mu_fc_->grad_weights[k] += mu_gradw[k];
    }

    std::vector<float> grad_h_lv;
    std::vector<float> lv_gradw;
    linear_backward_accum(*logvar_fc_, h, grad_logvar, grad_h_lv, lv_gradw);
    for (size_t k = 0; k < logvar_fc_->grad_weights.size() && k < lv_gradw.size(); ++k) {
        logvar_fc_->grad_weights[k] += lv_gradw[k];
    }

    // combine grads into h
    std::vector<float> grad_h(grad_h_mu.size(), 0.0f);
    for (size_t i = 0; i < grad_h.size(); ++i) {
        float a = (i < grad_h_mu.size()) ? grad_h_mu[i] : 0.0f;
        float b = (i < grad_h_lv.size()) ? grad_h_lv[i] : 0.0f;
        grad_h[i] = a + b;
    }

    // Encoder layers (reverse)
    for (int i = static_cast<int>(cfg_.encoder_hidden.size()) - 1; i >= 0; --i) {
        activation_backward_inplace(grad_h, enc_cache[static_cast<size_t>(i)].preact, cfg_.activation);

        std::vector<float> grad_in;
        std::vector<float> gradw;
        linear_backward_accum(*enc_fcs_[static_cast<size_t>(i)], enc_cache[static_cast<size_t>(i)].input, grad_h, grad_in, gradw);

        auto* L = enc_fcs_[static_cast<size_t>(i)];
        for (size_t k = 0; k < L->grad_weights.size() && k < gradw.size(); ++k) {
            L->grad_weights[k] += gradw[k];
        }

        grad_h = std::move(grad_in);
    }

    // Gradient stats + snapshot (avant optimizerStep qui reset grad_weights)
    {
        double sumsq = 0.0;
        double maxabs = 0.0;
        last_grads_by_layer_.clear();

        auto accum_and_capture = [&](const Layer* L) {
            if (!L) return;
            if (!L->grad_weights.empty()) {
                last_grads_by_layer_[L->name] = L->grad_weights;
            }
            for (float g : L->grad_weights) {
                const double gd = static_cast<double>(g);
                sumsq += gd * gd;
                const double ag = std::abs(gd);
                if (ag > maxabs) maxabs = ag;
            }
        };

        for (auto* L : enc_fcs_) accum_and_capture(L);
        accum_and_capture(mu_fc_);
        accum_and_capture(logvar_fc_);
        for (auto* L : dec_fcs_) accum_and_capture(L);
        accum_and_capture(out_fc_);

        last_grad_norm_ = static_cast<float>(std::sqrt(sumsq));
        last_grad_maxabs_ = static_cast<float>(maxabs);
    }

    // Apply optimizer step via framework
    optimizerStep(opt, learning_rate, nullptr);

    return stats;
}

bool VAEModel::trainOnDatasetItems(std::vector<DatasetItem>& items, int epochs, float learning_rate,
                                  Optimizer& opt, size_t max_items, size_t max_text_chars) {
    if (!getHasEncoder()) {
        throw std::runtime_error("VAEModel::trainOnDatasetItems: encoder requis (model.setHasEncoder(true))");
    }

    if (cfg_.input_dim != getEncoder().dim) {
        // Auto-align: par défaut on force input_dim = encoder.dim si mismatch
        cfg_.input_dim = getEncoder().dim;
    }

    if (layers.empty()) {
        // Build from current cfg_ if not built
        buildFromConfig(cfg_);
    }

    // Ensure params allocated
    if (totalParamCount() > 0 && layer_weight_blocks.empty()) {
        allocateParams();
        initializeWeights("xavier", cfg_.seed);
    }

    size_t count = 0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float loss_sum = 0.0f;
        float recon_sum = 0.0f;
        float kl_sum = 0.0f;
        size_t steps = 0;

        for (auto& it : items) {
            if (max_items > 0 && count >= max_items) break;

            if (it.text_file.empty()) continue;
            if (!it.loadText()) continue;
            if (!it.text.has_value()) continue;

            std::string txt = *it.text;
            if (txt.size() > max_text_chars) txt.resize(max_text_chars);

            auto toks = getMutableTokenizer().tokenizeEnsure(txt);
            // S'assurer que l'encoder couvre le vocab
            getMutableEncoder().ensureVocabSize(getTokenizer().getVocabSize());

            std::vector<float> x = getEncoder().encode(toks);
            if (static_cast<int>(x.size()) != cfg_.input_dim) {
                // Sécurité
                continue;
            }

            auto s = trainStep(x, opt, learning_rate, cfg_.seed + static_cast<uint32_t>(epoch * 1337 + steps));
            loss_sum += s.loss;
            recon_sum += s.recon_loss;
            kl_sum += s.kl_loss;
            steps++;
            count++;
        }

        if (steps == 0) return false;
        (void)loss_sum; (void)recon_sum; (void)kl_sum;
    }

    return true;
}
