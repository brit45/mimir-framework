#pragma once

#include "../Model.hpp"

#include <string>

// VAEConvModel:
// - VAE PUREMENT CONVOLUTIONNEL (downsample -> mu/logvar -> reparameterize -> upsample)
// - Architecture majoritairement convolutionnelle, avec blocs optionnels
//   SelfAttention spatiale et ResNet.
// - Couches principales: Conv2d / ConvTranspose2d / GroupNorm|LayerNorm /
//   SiLU / résidus Add / UpsampleNearest / Reparameterize (+ adaptateurs de
//   disposition Reshape/Permute pour l'I/O).
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
        // Entièrement convolutionnels. Activé via --resnet / cfg.use_attention.
        bool use_attention = true;

        // Active la SelfAttention spatiale optionnelle (H*W tokens, embed_dim=channels)
        // sur certains blocs encodeur/décodeur.
        bool use_attn = false;

        // Normalisation dans l'encodeur : "none" | "groupnorm" | "layernorm"
        std::string enc_norm = "groupnorm";

        // Nombre maximum de groupes pour GroupNorm (sera clampé au meilleur diviseur ≤ cette valeur).
        int enc_gn_groups = 32;

        // Normalisation dans le décodeur : "none" | "groupnorm" | "layernorm".
        // Si vide, reprend enc_norm.
        std::string dec_norm = "groupnorm";

        // Nombre maximum de groupes pour la GroupNorm du décodeur.
        // Si <= 0, reprend enc_gn_groups.
        int dec_gn_groups = 32;

        // Upsampling du décodeur : "conv_transpose" | "nearest_conv".
        std::string decoder_upsample = "conv_transpose";

        // Nombre de têtes SelfAttention (sera clampé à un diviseur valide de channels).
        int attn_heads = 4;

        // Garde-fou ResNet : bloc ResNet injecté seulement si h*w <= resnet_max_tokens.
        // 0 = pas de garde-fou (toujours injecté).
        int resnet_max_tokens = 0;

        // Garde-fou SelfAttention : injectée seulement si h*w <= attn_max_tokens.
        // 0 = pas de garde-fou (toujours injectée quand use_attn=true).
        int attn_max_tokens = 0;

        // Skip connections encodeur→décodeur (style U-Net).
        // À chaque niveau de downsampling, les feature maps de l'encodeur sont
        // concaténées aux feature maps du décodeur à résolution correspondante,
        // puis projetées par une Conv 1×1 (2*base → base) sans activation.
        // Améliore la reconstruction et accélère la convergence ; incompatible
        // avec les checkpoints entraînés sans cette option.
        bool use_skip_connections = false;

        // Prior appris dans le graphe (couche Constant).
        // Quand true : un vecteur biais appris de taille latent_dim est
        // ajouté additivement à z (sortie du Reparameterize) avant le décodeur.
        // Implémenté via une couche `Constant` + `Add` dans le graphe,
        // entraînée par backprop comme n'importe quelle couche.
        // Compatible avec les checkpoints via le nom de layer "vae_conv/z_prior_bias".
        bool use_encoder_prior = false;

        // ConditioningEncoder externe (classe ConditioningEncoder) : dimension des embeddings mag/mod/seq.
        // 0 = désactivé. Quand > 0, l'ConditioningEncoder est initialisé avec dim=d_model et
        // sérialisé avec le checkpoint. Les embeddings mag/mod/seq (vecteurs appris
        // de taille d_model) sont disponibles pour conditioning externe.
        // NOTE: non injecté dans le graphe (pas de couche mag/mod dans buildInto)
        // pour éviter tout conflit de dimension avec le latent spatial.
        int d_model = 0;

        // [DÉPRÉCIÉ / IGNORÉ] Conditionnement texte. Un modèle purement convolutionnel
        // ne peut pas embarquer le chemin dense texte (Embedding/Pool/Linear) ; ces
        // champs sont conservés pour compatibilité mais n'affectent plus le graphe.
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
    // Layer names match the decoder portion of buildInto().
    static void buildDecoderInto(Model& model, const Config& cfg);

private:
    Config cfg_;
};
