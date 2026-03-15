#pragma once

#include "../Model.hpp"

#include <random>
#include <string>
#include <vector>
#include <memory>

// PonyXLDDPMModel (PonyXL SDXL-like):
// - Entraînement diffusion en espace latent: image -> VAE(mu) = x0, puis
//   x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1-alpha_bar[t]) * eps
//   et on apprend eps_pred = f(x_t, text_ids, timestep) avec loss = MSE(eps_pred, eps).

class PonyXLDDPMModel : public Model {
public:
    struct Config {
        // RNG seed (pour la reproductibilité de l'échantillonnage t/eps et des dropouts)
        int seed = 1337;

        // Texte (SDXL-like)
        int d_model = 256;     // dimension embedding
        int max_vocab = 32000;  // taille max vocab tokenizer (français: recommandé >= 32k)

        // Texte/latents: entrée attendue
        // Pour des prompts très longs (ex: ~1300 mots), il faut un ctx_len élevé.
        // NOTE: 1300 mots peuvent produire >1300 tokens selon le tokenizer.
        int text_ctx_len = 1300;

        // Compression du contexte texte pour mieux gérer les prompts très longs.
        // Si activé, on transforme la séquence (text_ctx_len) en un seul token contexte (mean-pool)
        // avant la cross-attention UNet. Ça réduit énormément le coût et force un “résumé” global.
        bool text_bottleneck_meanpool = false;

        // NOTE: Le VAE texte a été extrait en modèle séparé (`vae_text`).
        // Ce modèle de diffusion ne contient plus de bottleneck variational côté texte.
        // Par défaut on vise le VAE_conv 256x256: latent 64x64x64 => seq=4096, in_dim=64.
        int latent_seq_len = 4096;
        int latent_in_dim = 64;
        int num_heads = 8;

        // Conditionnement temporel (diffusion-like): si true, le modèle attend une entrée float
        // nommée "timestep" (scalaire) et l’injecte via un petit MLP dans les tokens latents.
        bool sdxl_time_cond = true;
        // Par défaut: plus de blocs pour éviter un stub trop peu profond.
        int unet_layers = 16;
        int text_layers = 8;
        int mlp_hidden = 1024;

        // UNet 2D (multi-échelle)
        int latent_h = 0;     // 0 = auto (inféré de latent_seq_len)
        int latent_w = 0;     // 0 = auto
        int unet_depth = 3;   // nombre de niveaux down/up

        // Image (RGB) - dimensions attendues côté dataset loader
        int image_w = 64;
        int image_h = 64;
        int image_c = 3;       // 3 canaux (PNG)

        // Diffusion schedule (DDPM)
        int ddpm_steps = 1000;
        float ddpm_beta_start = 1e-4f;
        float ddpm_beta_end = 2e-2f;

        // Loss utilisée pour l'entraînement DDPM (comparaison eps_pred vs eps)
        // Valeurs supportées (voir Model::computeLoss): "mse", "mae", "huber"/"smoothl1", "charbonnier", "gaussian_nll".
        std::string recon_loss = "mse";

        // Nombre d'updates diffusion par image ("par étapes" au sens entraînement).
        // 1 = standard DDPM (un timestep aléatoire par image).
        int ddpm_steps_per_image = 1;

        // Bruit "Peltier" (thermal / capteur) approximé: bruit gaussien avec composante floutée
        // (corrélée spatialement) pour apprendre à débruiter des artefacts "soft".
        bool peltier_noise = true;
        // Mix [0..1]: 0 => blanc, 1 => uniquement flouté
        float peltier_mix = 0.65f;
        // Rayon du box blur (2 => kernel 5x5). 0 => pas de flou.
        int peltier_blur_radius = 2;

        // === VAE (Stable Diffusion-like) ===
        // En mode sdxl_like, l'image du dataset est encodée via un VAE externe (pré-entraîné)
        // pour obtenir x0 en espace latent. Aucun placeholder.
        std::string vae_arch = "vae_conv";
        std::string vae_checkpoint = "";
        float vae_scale = 1.0f;
        // Optionnel: permet d'aligner l'architecture VAE (ex: vae_conv base_channels)
        // avec un checkpoint pré-entraîné.
        int vae_base_channels = 0;

        // CFG dropout: parfois on retire le prompt
        float cfg_dropout_prob = 0.10f;

        // Prompt preprocessing (optionnel)
        int max_text_chars = 262144;
        float dropout = 0.0f;

        // === Captions structurées (dataset texte) ===
        // Supporte des sections optionnelles sous forme:
        //   --- TAGS ---
        //   ...
        //   --- CONTEXTE ---
        //   ...
        //   --- MENTALITÉ ---
        //   ...
        //   --- TEXTE (langue...) ---
        //   ...
        // Si aucune balise n'est détectée, le prompt est utilisé tel quel.
        bool caption_structured_enable = true;
        // Si true, on recompose une forme canonique "TAGS: ...\nCONTEXTE: ...".
        // Si false, on conserve le prompt original (mais on peut appliquer le dropout par section).
        bool caption_structured_canonicalize = true;

        // Dropout par section (appliqué uniquement pendant l'entraînement).
        float caption_tags_dropout_prob = 0.0f;
        float caption_contexte_dropout_prob = 0.0f;
        float caption_mentalite_dropout_prob = 0.0f;
        float caption_texte_dropout_prob = 0.0f;

        // === Viz: séquence DDPM (noising/denoising) ===
        // 0 = désactivé. Sinon: toutes les N steps d'optimizer, on génère quelques frames
        // montrant x_t (noisé) et x0_hat (reconstruit) pour plusieurs timesteps.
        int viz_ddpm_every_steps = 0;
        // Nombre de timesteps affichés (linéairement espacés de 0 à T-1).
        int viz_ddpm_num_steps = 5;
    };

    struct StepStats {
        float loss = 0.0f;
        float grad_norm = 0.0f;
        float grad_max_abs = 0.0f;

        // DDPM timestep (normalisé [0..1] quand utilisé, sinon 0)
        float timestep = 0.0f;

        float kl_divergence = 0.0f;
        float wasserstein = 0.0f;
        float entropy_diff = 0.0f;
        float moment_mismatch = 0.0f;
        float spatial_coherence = 0.0f;
        float temporal_consistency = 0.0f;
    };

    PonyXLDDPMModel();

    void buildFromConfig(const Config& cfg);
    const Config& getConfig() const { return cfg_; }

    // Permet de modifier l'échelle VAE après création du modèle (ex: calibration auto).
    void setVaeScale(float s) { cfg_.vae_scale = s; }

    // Accumule les moments (sum, sumsq, n) sur le vecteur mu du VAE pour une image.
    // mu est extrait du packed output [image || mu || logvar] du VAE.
    void accumulateVaeMuMoments(const std::vector<uint8_t>& rgb,
                                int w,
                                int h,
                                double& sum,
                                double& sumsq,
                                size_t& n);

    // Entraînement diffusion latent SDXL-like (eps predictor)
    StepStats trainStepSdxlLatentDiffusion(const std::string& prompt,
                                           const std::vector<uint8_t>& rgb,
                                           int w,
                                           int h,
                                           Optimizer& opt,
                                           float learning_rate);

    // Validation (forward-only): calcule MSE(eps_pred, eps) et MSE(x0_hat, x0) et une marge
    // d'association texte->image (comparaison prompt correct vs prompt "mauvais").
    struct ValStats {
        double eps_mse = 0.0;
        double x0_mse = 0.0;
        // Reconstruction en espace image (après VAE decode de x0_hat), MSE sur [-1,1].
        double img_mse = 0.0;
        double eps_mse_wrong = 0.0;
        double assoc_margin = 0.0; // eps_mse_wrong - eps_mse
        float t_norm = 0.0f;
    };
    ValStats validateStepSdxlLatentDiffusion(const std::string& prompt,
                                             const std::string& wrong_prompt,
                                             const std::vector<uint8_t>& rgb,
                                             int w,
                                             int h,
                                             int seed = 12345,
                                             int ddpm_step = -1);

    // Recon preview (forward-only): produit une image RGB reconstruite via VAE decode
    // à partir de x0_hat (débruitage au timestep fixe utilisé en validation).
    struct ReconPreview {
        std::vector<uint8_t> pixels;
        int w = 0;
        int h = 0;
        int channels = 3;
    };
    ReconPreview reconstructPreviewSdxlLatentDiffusion(const std::string& prompt,
                                                       const std::vector<uint8_t>& rgb,
                                                       int w,
                                                       int h,
                                                       int max_side = 256,
                                                       int seed = 12345,
                                                       int ddpm_step = -1);

    static void buildInto(Model& model, const Config& cfg);

    static std::vector<float> imageBytesToFloatRGB(const std::vector<uint8_t>& rgb, int w, int h);

private:
    Config cfg_;

    // RNG propre au modèle (seedé depuis cfg_.seed) pour éviter un seed codé en dur.
    std::mt19937 rng_;

    // VAE chargé pour encoder les images en latents
    std::shared_ptr<Model> vae_;

    // VAE decoder (pour visualizer reconstructions en espace image, lazy-load)
    std::shared_ptr<Model> vae_decode_;
};
