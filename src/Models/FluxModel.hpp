#pragma once

#include "../Model.hpp"
#include "../Layers.hpp"
#include "../Tokenizer.hpp"
#include "../Encoder.hpp"
#include <vector>
#include <string>
#include <memory>

// ============================================================================
// Flux Model - Modèle de diffusion avancé avec VAE et conditionnement texte
// ============================================================================

namespace ModelArchitectures {

struct FluxConfig {
    // Dimensions générales
    int latent_channels = 4;        // Canaux dans l'espace latent VAE
    int latent_resolution = 64;     // Résolution de l'espace latent (64x64)
    int image_resolution = 512;     // Résolution de sortie (512x512)
    int vae_downsample_factor = 8;  // Facteur de downsampling du VAE (512/8 = 64)
    
    // VAE Encoder/Decoder
    int vae_base_channels = 128;
    int vae_num_res_blocks = 2;
    std::vector<int> vae_channel_mult = {1, 2, 4, 4};  // Multiplicateurs par niveau
    
    // Text conditioning
    int text_embed_dim = 768;       // Dimension d'embedding du texte
    int text_max_length = 77;       // Longueur maximale de séquence
    int vocab_size = 49408;         // Taille du vocabulaire
    
    // Transformer blocks (diffusion model)
    int num_transformer_blocks = 12;
    int transformer_dim = 1024;
    int num_attention_heads = 16;
    int mlp_ratio = 4;              // Ratio pour le MLP dans le transformer
    
    // Timestep conditioning
    int timestep_embed_dim = 256;
    
    // Training parameters
    int num_timesteps = 1000;       // Nombre de steps de diffusion
    float beta_start = 0.0001f;
    float beta_end = 0.02f;
    
    // Options
    bool use_flash_attention = true;
    bool use_gradient_checkpointing = false;
    float dropout = 0.0f;
    ActivationType activation = ActivationType::SWISH;  // SWISH = SiLU
};

// Classe Flux héritant de Model
class FluxModel : public Model {
public:
    FluxModel();
    FluxModel(const FluxConfig& config);
    ~FluxModel() = default;
    
    // Configuration
    void setConfig(const FluxConfig& config);
    FluxConfig getConfig() const { return config_; }
    
    // Build le modèle complet
    void buildFluxArchitecture();
    
    // VAE Components
    void buildVAEEncoder();
    void buildVAEDecoder();
    
    // Text Conditioning
    void buildTextEncoder();
    void buildTextProjection();
    
    // Diffusion Transformer
    void buildDiffusionTransformer();
    void buildTimestepEmbedding();
    
    // Forward passes spécialisés
    std::vector<float> encodeImage(const std::vector<float>& image);
    std::vector<float> decodeLatent(const std::vector<float>& latent);
    std::vector<float> encodeText(const std::vector<int>& tokens);
    
    // Diffusion process
    std::vector<float> addNoise(const std::vector<float>& latent, int timestep);
    std::vector<float> predictNoise(const std::vector<float>& noisy_latent, 
                                    const std::vector<float>& text_embedding,
                                    int timestep);
    std::vector<float> denoisingStep(const std::vector<float>& noisy_latent,
                                     const std::vector<float>& text_embedding,
                                     int timestep);
    
    // Generation (sampling)
    std::vector<float> generate(const std::string& prompt, int num_steps = 50, float guidance_scale = 7.5f);
    std::vector<float> generateFromTokens(const std::vector<int>& tokens, int num_steps = 50, float guidance_scale = 7.5f);
    
    // Training
    float computeDiffusionLoss(const std::vector<float>& image, const std::vector<int>& tokens);
    
    // Mode d'exécution
    void train() { is_training_ = true; }
    void eval() { is_training_ = false; }
    bool isTraining() const { return is_training_; }
    
    // Utilitaires
    void setPromptTokenizer(std::shared_ptr<Tokenizer> tokenizer);
    std::vector<int> tokenizePrompt(const std::string& prompt);
    
    // Getters pour les dimensions
    int getLatentChannels() const { return config_.latent_channels; }
    int getLatentResolution() const { return config_.latent_resolution; }
    int getImageResolution() const { return config_.image_resolution; }
    
protected:
    FluxConfig config_;
    std::shared_ptr<Tokenizer> prompt_tokenizer_;
    bool is_training_ = false;
    
    // Indices des couches dans le modèle
    struct LayerIndices {
        // VAE Encoder
        int vae_enc_start = -1;
        int vae_enc_end = -1;
        
        // VAE Decoder
        int vae_dec_start = -1;
        int vae_dec_end = -1;
        
        // Text Encoder
        int text_enc_start = -1;
        int text_enc_end = -1;
        
        // Diffusion Transformer
        int diff_trans_start = -1;
        int diff_trans_end = -1;
        
        // Timestep embedding
        int timestep_emb_start = -1;
        int timestep_emb_end = -1;
    };
    
    LayerIndices layer_indices_;
    
    // Noise schedule (beta values)
    std::vector<float> betas_;
    std::vector<float> alphas_;
    std::vector<float> alphas_cumprod_;
    std::vector<float> sqrt_alphas_cumprod_;
    std::vector<float> sqrt_one_minus_alphas_cumprod_;
    
    void initNoiseSchedule();
    float getTimeEmbedding(int timestep, int dim);
};

// Helper function pour construire un Flux model depuis la config
inline void buildFlux(Model& model, const FluxConfig& config) {
    FluxModel* flux_model = dynamic_cast<FluxModel*>(&model);
    if (flux_model) {
        flux_model->setConfig(config);
        flux_model->buildFluxArchitecture();
    } else {
        // Si on ne peut pas caster, on crée la structure directement dans le modèle
        // (version simplifiée sans les méthodes spécialisées)
        int current_params = 0;
        
        // 1. VAE Encoder
        int ch = config.vae_base_channels;
        for (size_t i = 0; i < config.vae_channel_mult.size(); ++i) {
            int out_ch = ch * config.vae_channel_mult[i];
            for (int j = 0; j < config.vae_num_res_blocks; ++j) {
                model.push("vae_enc_L" + std::to_string(i) + "_B" + std::to_string(j) + "_conv1", 
                          "Conv2d", ch * 3 * 3 * out_ch + out_ch);
                model.push("vae_enc_L" + std::to_string(i) + "_B" + std::to_string(j) + "_norm1", 
                          "GroupNorm", out_ch * 2);
                model.push("vae_enc_L" + std::to_string(i) + "_B" + std::to_string(j) + "_conv2", 
                          "Conv2d", out_ch * 3 * 3 * out_ch + out_ch);
                model.push("vae_enc_L" + std::to_string(i) + "_B" + std::to_string(j) + "_norm2", 
                          "GroupNorm", out_ch * 2);
            }
            if (i < config.vae_channel_mult.size() - 1) {
                model.push("vae_enc_downsample_L" + std::to_string(i), "Conv2d", 
                          out_ch * 3 * 3 * out_ch + out_ch);
            }
            ch = out_ch;
        }
        
        // Latent projection (mu et logvar pour VAE)
        model.push("vae_enc_mu", "Conv2d", ch * 1 * 1 * config.latent_channels + config.latent_channels);
        model.push("vae_enc_logvar", "Conv2d", ch * 1 * 1 * config.latent_channels + config.latent_channels);
        
        // 2. Text Encoder (CLIP-like)
        model.push("text_token_embedding", "Embedding", 
                  config.vocab_size * config.text_embed_dim);
        model.push("text_pos_embedding", "Embedding", 
                  config.text_max_length * config.text_embed_dim);
        
        for (int i = 0; i < 12; ++i) {  // 12 layers de transformer pour le texte
            std::string prefix = "text_transformer_L" + std::to_string(i);
            
            // Self-attention
            model.push(prefix + "_attn_qkv", "Linear", 
                      config.text_embed_dim * (config.text_embed_dim * 3) + (config.text_embed_dim * 3));
            model.push(prefix + "_attn_out", "Linear", 
                      config.text_embed_dim * config.text_embed_dim + config.text_embed_dim);
            model.push(prefix + "_norm1", "LayerNorm", config.text_embed_dim * 2);
            
            // MLP
            int mlp_hidden = config.text_embed_dim * 4;
            model.push(prefix + "_mlp_fc1", "Linear", 
                      config.text_embed_dim * mlp_hidden + mlp_hidden);
            model.push(prefix + "_mlp_fc2", "Linear", 
                      mlp_hidden * config.text_embed_dim + config.text_embed_dim);
            model.push(prefix + "_norm2", "LayerNorm", config.text_embed_dim * 2);
        }
        
        model.push("text_final_norm", "LayerNorm", config.text_embed_dim * 2);
        model.push("text_projection", "Linear", 
                  config.text_embed_dim * config.transformer_dim + config.transformer_dim);
        
        // 3. Timestep Embedding
        model.push("timestep_mlp_0", "Linear", 
                  config.timestep_embed_dim * (config.timestep_embed_dim * 4) + (config.timestep_embed_dim * 4));
        model.push("timestep_mlp_1", "Linear", 
                  (config.timestep_embed_dim * 4) * config.transformer_dim + config.transformer_dim);
        
        // 4. Diffusion Transformer
        for (int i = 0; i < config.num_transformer_blocks; ++i) {
            std::string prefix = "diff_trans_L" + std::to_string(i);
            
            // Latent projection
            model.push(prefix + "_latent_proj", "Linear",
                      config.latent_channels * config.transformer_dim + config.transformer_dim);
            
            // Self-attention on latents
            int head_dim = config.transformer_dim / config.num_attention_heads;
            model.push(prefix + "_self_attn_qkv", "Linear",
                      config.transformer_dim * (config.transformer_dim * 3) + (config.transformer_dim * 3));
            model.push(prefix + "_self_attn_out", "Linear",
                      config.transformer_dim * config.transformer_dim + config.transformer_dim);
            model.push(prefix + "_norm1", "LayerNorm", config.transformer_dim * 2);
            
            // Cross-attention with text
            model.push(prefix + "_cross_attn_q", "Linear",
                      config.transformer_dim * config.transformer_dim + config.transformer_dim);
            model.push(prefix + "_cross_attn_kv", "Linear",
                      config.transformer_dim * (config.transformer_dim * 2) + (config.transformer_dim * 2));
            model.push(prefix + "_cross_attn_out", "Linear",
                      config.transformer_dim * config.transformer_dim + config.transformer_dim);
            model.push(prefix + "_norm2", "LayerNorm", config.transformer_dim * 2);
            
            // MLP
            int mlp_hidden = config.transformer_dim * config.mlp_ratio;
            model.push(prefix + "_mlp_fc1", "Linear",
                      config.transformer_dim * mlp_hidden + mlp_hidden);
            model.push(prefix + "_mlp_fc2", "Linear",
                      mlp_hidden * config.transformer_dim + config.transformer_dim);
            model.push(prefix + "_norm3", "LayerNorm", config.transformer_dim * 2);
        }
        
        // Output projection to latent space
        model.push("diff_output_norm", "LayerNorm", config.transformer_dim * 2);
        model.push("diff_output_proj", "Linear",
                  config.transformer_dim * config.latent_channels + config.latent_channels);
        
        // 5. VAE Decoder
        model.push("vae_dec_input", "Conv2d",
                  config.latent_channels * 3 * 3 * ch + ch);
        
        for (int i = config.vae_channel_mult.size() - 1; i >= 0; --i) {
            int out_ch = config.vae_base_channels * config.vae_channel_mult[i];
            
            for (int j = 0; j < config.vae_num_res_blocks + 1; ++j) {
                model.push("vae_dec_L" + std::to_string(i) + "_B" + std::to_string(j) + "_conv1",
                          "Conv2d", ch * 3 * 3 * out_ch + out_ch);
                model.push("vae_dec_L" + std::to_string(i) + "_B" + std::to_string(j) + "_norm1",
                          "GroupNorm", out_ch * 2);
                model.push("vae_dec_L" + std::to_string(i) + "_B" + std::to_string(j) + "_conv2",
                          "Conv2d", out_ch * 3 * 3 * out_ch + out_ch);
                model.push("vae_dec_L" + std::to_string(i) + "_B" + std::to_string(j) + "_norm2",
                          "GroupNorm", out_ch * 2);
                ch = out_ch;
            }
            
            if (i > 0) {
                model.push("vae_dec_upsample_L" + std::to_string(i), "ConvTranspose2d",
                          ch * 4 * 4 * ch + ch);
            }
        }
        
        // Final output (RGB)
        model.push("vae_dec_output_norm", "GroupNorm", ch * 2);
        model.push("vae_dec_output_conv", "Conv2d", ch * 3 * 3 * 3 + 3);
    }
}

} // namespace ModelArchitectures
