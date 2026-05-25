#pragma once

#include "../Model.hpp"

#include <string>

// VAEConvModel:
// - Convolutional VAE (downsample -> mu/logvar -> reparameterize -> upsample)
// - Output packs: [recon(image_dim), mu(latent_dim), logvar(latent_dim)]
// - Latent is spatial: latent_dim = latent_h * latent_w * latent_c (CHW)

class VAEConvModel : public Model {
public:
    struct Config {
        int image_w = 64;
        int image_h = 64;
        int image_c = 3;

        int latent_h = 16;
        int latent_w = 16;
        int latent_c = 256;

        int base_channels = 64;

        // Si false, le latent utilisé par le décodeur est déterministe (z = mu).
        // Si true, on active le bruit de réparameterisation (z = mu + std * eps).
        // Pour des pipelines type SD/SDXL, on veut généralement un encodeur déterministe.
        bool stochastic_latent = false;

        // ---- Blocs optionnels ----

        // Blocs ResNet (conv3x3->SiLU->conv3x3 + skip) dans l'encodeur et le décodeur.
        // Activé via --resnet / cfg.use_attention.
        bool use_attention = true;

        // SelfAttention spatiale au bottleneck (latent resolution).
        // Activé via --attn / cfg.use_attn.
        bool use_attn = true;

        // Normalisation dans l'encodeur : "none" | "groupnorm" | "layernorm"
        std::string enc_norm = "groupnorm";

        // Nombre maximum de groupes pour GroupNorm (sera clampé au meilleur diviseur ≤ cette valeur).
        int enc_gn_groups = 32;

        // Nombre de têtes pour la SelfAttention spatiale.
        int attn_heads = 4;

        // Garde-fou ResNet : bloc ResNet injecté seulement si h*w <= resnet_max_tokens.
        // 0 = pas de garde-fou (toujours injecté).
        int resnet_max_tokens = 0;

        // Garde-fou SelfAttention : attn injectée seulement si h*w <= attn_max_tokens.
        // 0 = pas de garde-fou.
        int attn_max_tokens = 0;

        // Optional: texte (conditionnement + alignement). Si false, VAEConv reste image-only.
        bool text_cond = false;
        int vocab_size = 32000;
        int seq_len = 64;
        int text_d_model = 256;
        int proj_dim = 256;   // dim de l'espace commun image/texte pour alignement
    };

    VAEConvModel();
    void buildFromConfig(const Config& cfg);
    const Config& getConfig() const { return cfg_; }

    static void buildInto(Model& model, const Config& cfg);
    
    // Decoder-only graph: input is z (latent CHW vector), output is recon (RGB vector).
    // Layer names match the decoder portion of buildInto() so checkpoints can be loaded
    // with strict_mode=false (encoder tensors will be ignored).
    static void buildDecoderInto(Model& model, const Config& cfg);

private:
    Config cfg_;
};
