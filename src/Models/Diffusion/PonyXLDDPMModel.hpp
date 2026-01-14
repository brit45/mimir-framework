#pragma once

#include "../Model.hpp"

#include <random>
#include <string>
#include <vector>

// PonyXLDDPMModel:
// - Modèle texte + image bruitée + timestep -> prédiction de bruit (epsilon)
// - Entraînement DDPM: x_t = sqrt(a_bar[t]) * x0 + sqrt(1-a_bar[t]) * eps
//   loss = MSE(eps_pred, eps)
// - Génération: boucle t=T-1..0 (DDIM-like) pour débruiter un bruit initial.

class PonyXLDDPMModel : public Model {
public:
    struct Config {
        // Texte
        int d_model = 256;     // dimension embedding texte (style T5 encoder size simplifié)
        int seq_len = 128;     // longueur max de tokens (pad/trunc)
        int max_vocab = 4096;  // taille max vocab tokenizer/encoder

        // Image (RGB)
        int image_w = 64;
        int image_h = 64;
        int image_c = 3;       // 3 canaux (PNG)

        // Encoder/Decoder (MLP simple, latent bottleneck)
        int hidden_dim = 2048;
        int latent_dim = 512;

        // Entraînement triple-fault
        int blur_levels = 64;         // génère 1..blur_levels (en %) + original
        float cfg_dropout_prob = 0.10f; // p(drop prompt) (classifier-free guidance-like)

        // Forcer la corrélation texte->image:
        // on retire parfois l'image conditionnelle (x_in) pour éviter que le modèle
        // apprenne un autoencodeur qui ignore le texte.
        float cond_image_dropout_prob = 0.05f;     // p(drop image conditionnelle)
        float cond_image_dropout_lr_scale = 0.25f; // lr multiplier quand image conditionnelle absente
        float cond_image_dropout_noise_std = 0.0f; // si >0: remplace x_in par bruit N(0,std) au lieu de zéros

        float dropout = 0.0f;
    };

    struct StepStats {
        float loss = 0.0f;
        float grad_norm = 0.0f;
        float grad_max_abs = 0.0f;
    };

    PonyXLDDPMModel();

    void buildFromConfig(const Config& cfg);
    const Config& getConfig() const { return cfg_; }

    // Entraîne sur un micro-batch (input->target) pour compatibilité.
    StepStats trainStep(const std::vector<float>& input,
                        const std::vector<float>& target,
                        Optimizer& opt,
                        float learning_rate);

    // Entraînement "triple-fault": original + versions floutées (1..N) -> reconstruction de l'original.
    // Retourne des stats agrégées (loss moyen).
    StepStats trainStepTripleFault(const std::string& prompt,
                                   const std::vector<uint8_t>& rgb,
                                   int w,
                                   int h,
                                   Optimizer& opt,
                                   float learning_rate);

    static void buildInto(Model& model, const Config& cfg);

    static std::vector<float> imageBytesToFloatRGB(const std::vector<uint8_t>& rgb, int w, int h);

private:
    Config cfg_;
};
