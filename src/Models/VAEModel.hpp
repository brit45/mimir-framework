#pragma once

#include "../Model.hpp"
#include <cstdint>
#include <random>
#include <unordered_map>

// VAEModel: Variational AutoEncoder sur des features continues (ex: embeddings Encoder)
// - Hérite de Model pour réutiliser: layers, allocation poids, serialization, Optimizer
// - Implémente un vrai training step (reconstruction + KL) avec backprop pour MLP (Linear + activations)

class VAEModel : public Model {
public:
    struct Config {
        int input_dim = 256;                 // doit matcher encoder.dim pour du texte
        int latent_dim = 64;
        std::vector<int> encoder_hidden = {512, 256};
        std::vector<int> decoder_hidden = {256, 512};
        ActivationType activation = ActivationType::RELU;
        float kl_beta = 1.0f;
        bool use_mean_in_infer = true;       // z = mu (sinon sampling)
        uint32_t seed = 0xC0FFEEu;
    };

    struct StepStats {
        float loss = 0.0f;
        float recon_loss = 0.0f;
        float kl_loss = 0.0f;
    };

    VAEModel();
    ~VAEModel();

    void buildFromConfig(const Config& cfg);
    const Config& getConfig() const { return cfg_; }

    // Forward/in_toggle
    std::vector<float> inferReconstruction(const std::vector<float>& x, uint32_t seed = 0) const;
    std::vector<float> inferMu(const std::vector<float>& x) const;

    // Training
    StepStats trainStep(const std::vector<float>& x, Optimizer& opt, float learning_rate, uint32_t seed = 0);

    // Grad stats (mis à jour à chaque trainStep, avant optimizerStep qui remet les grads à zéro)
    float getLastGradNorm() const { return last_grad_norm_; }
    float getLastGradMaxAbs() const { return last_grad_maxabs_; }

    // Snapshot des gradients (par layer.name) capturé avant optimizerStep()
    const std::unordered_map<std::string, std::vector<float>>& getLastGradientsByLayer() const { return last_grads_by_layer_; }

    // Dataset helper (utilise DatasetItem/loadText + tokenizer+encoder)
    // Entrée: items déjà chargés via loadDataset/loadDatasetCached côté framework
    bool trainOnDatasetItems(std::vector<DatasetItem>& items, int epochs, float learning_rate,
                             Optimizer& opt, size_t max_items = 0, size_t max_text_chars = 8192);

    // Builder utilitaire: permet de construire la même architecture dans n'importe quel Model
    static void buildInto(Model& model, const Config& cfg);

private:
    Config cfg_;

    float last_grad_norm_ = 0.0f;
    float last_grad_maxabs_ = 0.0f;
    std::unordered_map<std::string, std::vector<float>> last_grads_by_layer_;

    struct LinearCache {
        std::vector<float> input;      // x
        std::vector<float> preact;     // y = Wx + b (avant activation)
        std::vector<float> output;     // après activation (si appliquée)
    };

    static void apply_activation_inplace(std::vector<float>& v, ActivationType act);
    static void activation_backward_inplace(std::vector<float>& grad, const std::vector<float>& preact, ActivationType act);

    static void linear_backward_accum(const Layer& layer,
                                      const std::vector<float>& input,
                                      const std::vector<float>& grad_out,
                                      std::vector<float>& grad_in,
                                      std::vector<float>& grad_w_out);

    void refreshLayerPtrs();

    // Pointeurs/indices vers layers (mis à jour après build)
    std::vector<Layer*> enc_fcs_;
    Layer* mu_fc_ = nullptr;
    Layer* logvar_fc_ = nullptr;
    std::vector<Layer*> dec_fcs_;
    Layer* out_fc_ = nullptr;
};
