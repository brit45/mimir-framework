#include "FluxModel.hpp"
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>

namespace ModelArchitectures {

// ============================================================================
// Constructeurs
// ============================================================================

FluxModel::FluxModel() : Model() {
    // Configuration par défaut
    config_ = FluxConfig();
    initNoiseSchedule();
}

FluxModel::FluxModel(const FluxConfig& config) : Model() {
    config_ = config;
    initNoiseSchedule();
}

// ============================================================================
// Configuration
// ============================================================================

void FluxModel::setConfig(const FluxConfig& config) {
    config_ = config;
    initNoiseSchedule();
}

void FluxModel::setPromptTokenizer(std::shared_ptr<Tokenizer> tokenizer) {
    prompt_tokenizer_ = tokenizer;
}

// ============================================================================
// Noise Schedule Initialization
// ============================================================================

void FluxModel::initNoiseSchedule() {
    int T = config_.num_timesteps;
    betas_.resize(T);
    alphas_.resize(T);
    alphas_cumprod_.resize(T);
    sqrt_alphas_cumprod_.resize(T);
    sqrt_one_minus_alphas_cumprod_.resize(T);
    
    // Linear schedule
    float beta_start = config_.beta_start;
    float beta_end = config_.beta_end;
    
    for (int t = 0; t < T; ++t) {
        betas_[t] = beta_start + (beta_end - beta_start) * (static_cast<float>(t) / (T - 1));
        alphas_[t] = 1.0f - betas_[t];
    }
    
    // Compute cumulative product
    float cumprod = 1.0f;
    for (int t = 0; t < T; ++t) {
        cumprod *= alphas_[t];
        alphas_cumprod_[t] = cumprod;
        sqrt_alphas_cumprod_[t] = std::sqrt(cumprod);
        sqrt_one_minus_alphas_cumprod_[t] = std::sqrt(1.0f - cumprod);
    }
}

// ============================================================================
// Build Architecture
// ============================================================================

void FluxModel::buildFluxArchitecture() {
    layer_indices_.vae_enc_start = layers.size();
    buildVAEEncoder();
    layer_indices_.vae_enc_end = layers.size();
    
    layer_indices_.text_enc_start = layers.size();
    buildTextEncoder();
    buildTextProjection();
    layer_indices_.text_enc_end = layers.size();
    
    layer_indices_.timestep_emb_start = layers.size();
    buildTimestepEmbedding();
    layer_indices_.timestep_emb_end = layers.size();
    
    layer_indices_.diff_trans_start = layers.size();
    buildDiffusionTransformer();
    layer_indices_.diff_trans_end = layers.size();
    
    layer_indices_.vae_dec_start = layers.size();
    buildVAEDecoder();
    layer_indices_.vae_dec_end = layers.size();
}

void FluxModel::buildVAEEncoder() {
    int ch = config_.vae_base_channels;
    
    // Input convolution
    push("vae_enc_input", "Conv2d", 3 * 3 * 3 * ch + ch);
    
    // Encoder blocks avec downsampling
    for (size_t i = 0; i < config_.vae_channel_mult.size(); ++i) {
        int out_ch = ch * config_.vae_channel_mult[i];
        
        for (int j = 0; j < config_.vae_num_res_blocks; ++j) {
            std::string prefix = "vae_enc_L" + std::to_string(i) + "_B" + std::to_string(j);
            
            // ResNet block
            push(prefix + "_norm1", "GroupNorm", ch * 2);
            push(prefix + "_conv1", "Conv2d", ch * 3 * 3 * out_ch + out_ch);
            push(prefix + "_norm2", "GroupNorm", out_ch * 2);
            push(prefix + "_conv2", "Conv2d", out_ch * 3 * 3 * out_ch + out_ch);
            
            // Residual connection (si dimensions différentes)
            if (ch != out_ch) {
                push(prefix + "_residual", "Conv2d", ch * 1 * 1 * out_ch + out_ch);
            }
            
            ch = out_ch;
        }
        
        // Downsampling (sauf dernier niveau)
        if (i < config_.vae_channel_mult.size() - 1) {
            push("vae_enc_down_L" + std::to_string(i), "Conv2d", 
                 ch * 3 * 3 * ch + ch);  // stride=2 pour downsampling
        }
    }
    
    // Middle blocks avec attention
    push("vae_enc_mid_norm1", "GroupNorm", ch * 2);
    push("vae_enc_mid_conv1", "Conv2d", ch * 3 * 3 * ch + ch);
    push("vae_enc_mid_attn", "SelfAttention", ch * ch * 3);  // Q, K, V
    push("vae_enc_mid_norm2", "GroupNorm", ch * 2);
    push("vae_enc_mid_conv2", "Conv2d", ch * 3 * 3 * ch + ch);
    
    // Output: mu et logvar pour le VAE
    push("vae_enc_norm_out", "GroupNorm", ch * 2);
    push("vae_enc_mu", "Conv2d", ch * 1 * 1 * config_.latent_channels + config_.latent_channels);
    push("vae_enc_logvar", "Conv2d", ch * 1 * 1 * config_.latent_channels + config_.latent_channels);
}

void FluxModel::buildVAEDecoder() {
    int ch = config_.vae_base_channels * config_.vae_channel_mult.back();
    
    // Input projection
    push("vae_dec_input", "Conv2d", 
         config_.latent_channels * 3 * 3 * ch + ch);
    
    // Middle blocks
    push("vae_dec_mid_norm1", "GroupNorm", ch * 2);
    push("vae_dec_mid_conv1", "Conv2d", ch * 3 * 3 * ch + ch);
    push("vae_dec_mid_attn", "SelfAttention", ch * ch * 3);
    push("vae_dec_mid_norm2", "GroupNorm", ch * 2);
    push("vae_dec_mid_conv2", "Conv2d", ch * 3 * 3 * ch + ch);
    
    // Decoder blocks avec upsampling
    for (int i = config_.vae_channel_mult.size() - 1; i >= 0; --i) {
        int out_ch = config_.vae_base_channels * config_.vae_channel_mult[i];
        
        for (int j = 0; j < config_.vae_num_res_blocks + 1; ++j) {
            std::string prefix = "vae_dec_L" + std::to_string(i) + "_B" + std::to_string(j);
            
            // ResNet block
            push(prefix + "_norm1", "GroupNorm", ch * 2);
            push(prefix + "_conv1", "Conv2d", ch * 3 * 3 * out_ch + out_ch);
            push(prefix + "_norm2", "GroupNorm", out_ch * 2);
            push(prefix + "_conv2", "Conv2d", out_ch * 3 * 3 * out_ch + out_ch);
            
            if (ch != out_ch) {
                push(prefix + "_residual", "Conv2d", ch * 1 * 1 * out_ch + out_ch);
            }
            
            ch = out_ch;
        }
        
        // Upsampling (sauf premier niveau)
        if (i > 0) {
            push("vae_dec_up_L" + std::to_string(i), "ConvTranspose2d",
                 ch * 4 * 4 * ch + ch);  // stride=2, kernel=4 pour upsampling
        }
    }
    
    // Output
    push("vae_dec_norm_out", "GroupNorm", ch * 2);
    push("vae_dec_conv_out", "Conv2d", ch * 3 * 3 * 3 + 3);  // RGB output
}

void FluxModel::buildTextEncoder() {
    int dim = config_.text_embed_dim;
    
    // Token et position embeddings
    push("text_token_emb", "Embedding", config_.vocab_size * dim);
    push("text_pos_emb", "Embedding", config_.text_max_length * dim);
    
    // Transformer layers (CLIP-like)
    for (int i = 0; i < 12; ++i) {
        std::string prefix = "text_L" + std::to_string(i);
        
        // Multi-head self-attention
        push(prefix + "_attn_norm", "LayerNorm", dim * 2);
        push(prefix + "_attn_qkv", "Linear", dim * (dim * 3) + (dim * 3));
        push(prefix + "_attn_out", "Linear", dim * dim + dim);
        
        // MLP
        int mlp_dim = dim * 4;
        push(prefix + "_mlp_norm", "LayerNorm", dim * 2);
        push(prefix + "_mlp_fc1", "Linear", dim * mlp_dim + mlp_dim);
        push(prefix + "_mlp_fc2", "Linear", mlp_dim * dim + dim);
    }
    
    push("text_final_norm", "LayerNorm", dim * 2);
}

void FluxModel::buildTextProjection() {
    // Project text embeddings to transformer dimension
    push("text_proj", "Linear", 
         config_.text_embed_dim * config_.transformer_dim + config_.transformer_dim);
}

void FluxModel::buildTimestepEmbedding() {
    int dim = config_.timestep_embed_dim;
    
    // Sinusoidal embedding + MLP
    push("timestep_mlp1", "Linear", dim * (dim * 4) + (dim * 4));
    push("timestep_mlp2", "Linear", (dim * 4) * config_.transformer_dim + config_.transformer_dim);
}

void FluxModel::buildDiffusionTransformer() {
    int dim = config_.transformer_dim;
    int heads = config_.num_attention_heads;
    int head_dim = dim / heads;
    
    // Input projection from latent space
    push("diff_input_proj", "Linear",
         config_.latent_channels * dim + dim);
    
    // Transformer blocks
    for (int i = 0; i < config_.num_transformer_blocks; ++i) {
        std::string prefix = "diff_block_" + std::to_string(i);
        
        // Self-attention sur les latents
        push(prefix + "_norm1", "LayerNorm", dim * 2);
        push(prefix + "_self_attn_qkv", "Linear", dim * (dim * 3) + (dim * 3));
        push(prefix + "_self_attn_out", "Linear", dim * dim + dim);
        
        // Cross-attention avec conditioning (text + timestep)
        push(prefix + "_norm2", "LayerNorm", dim * 2);
        push(prefix + "_cross_attn_q", "Linear", dim * dim + dim);
        push(prefix + "_cross_attn_kv", "Linear", dim * (dim * 2) + (dim * 2));
        push(prefix + "_cross_attn_out", "Linear", dim * dim + dim);
        
        // MLP avec timestep modulation
        push(prefix + "_norm3", "LayerNorm", dim * 2);
        int mlp_dim = dim * config_.mlp_ratio;
        push(prefix + "_mlp_fc1", "Linear", dim * mlp_dim + mlp_dim);
        push(prefix + "_mlp_fc2", "Linear", mlp_dim * dim + dim);
        
        // Adaptive layer norm (modulation par timestep)
        push(prefix + "_adaLN_scale", "Linear", config_.transformer_dim * dim + dim);
        push(prefix + "_adaLN_shift", "Linear", config_.transformer_dim * dim + dim);
    }
    
    // Output projection to latent space
    push("diff_output_norm", "LayerNorm", dim * 2);
    push("diff_output_proj", "Linear", dim * config_.latent_channels + config_.latent_channels);
}

// ============================================================================
// Forward Passes
// ============================================================================

std::vector<float> FluxModel::encodeImage(const std::vector<float>& image) {
    // Forward pass complet à travers le VAE encoder
    std::vector<float> x = image;
    
    // Input convolution
    x = forwardPass(x, false); // On utiliserait les layers appropriés ici
    
    // Pour une implémentation complète, on devrait:
    // 1. Appliquer les convolutions d'entrée
    // 2. Passer par tous les blocks résiduel avec downsampling
    // 3. Appliquer les middle blocks avec attention
    // 4. Projeter vers mu et logvar
    // 5. Reparametrization trick: z = mu + sigma * epsilon
    
    // Implémentation simplifiée avec convolutions directes
    int h = config_.image_resolution;
    int w = config_.image_resolution;
    int c = 3; // RGB
    
    // Simuler la réduction progressive vers l'espace latent
    for (size_t i = 0; i < config_.vae_channel_mult.size(); ++i) {
        if (i < config_.vae_channel_mult.size() - 1) {
            h /= 2;  // Downsampling
            w /= 2;
        }
        c = config_.vae_base_channels * config_.vae_channel_mult[i];
    }
    
    // Taille finale de l'espace latent
    int latent_size = config_.latent_channels * config_.latent_resolution * config_.latent_resolution;
    std::vector<float> latent(latent_size);
    
    // Extraction mu et application reparametrization trick
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    // Simplification: projection linéaire depuis x vers latent avec bruit
    for (size_t i = 0; i < latent.size(); ++i) {
        size_t x_idx = (i * x.size()) / latent.size();
        if (x_idx < x.size()) {
            float mu = x[x_idx];
            float logvar = 0.0f; // Simplification
            float sigma = std::exp(0.5f * logvar);
            float epsilon = is_training_ ? dist(gen) : 0.0f;
            latent[i] = mu + sigma * epsilon;
        }
    }
    
    return latent;
}

std::vector<float> FluxModel::decodeLatent(const std::vector<float>& latent) {
    // Forward pass complet à travers le VAE decoder
    std::vector<float> x = latent;
    
    // Pour une implémentation complète:
    // 1. Projection d'entrée du latent vers les features
    // 2. Middle blocks avec attention
    // 3. Blocks résiduel avec upsampling progressif
    // 4. Convolution finale vers RGB
    
    int image_size = 3 * config_.image_resolution * config_.image_resolution;
    std::vector<float> image(image_size);
    
    // Simuler l'upsampling progressif depuis l'espace latent
    int h = config_.latent_resolution;
    int w = config_.latent_resolution;
    
    // Upsampling vers la résolution image
    int target_h = config_.image_resolution;
    int target_w = config_.image_resolution;
    
    // Interpolation bilinéaire simplifiée pour l'upsampling
    float h_ratio = static_cast<float>(h) / target_h;
    float w_ratio = static_cast<float>(w) / target_w;
    
    for (int c = 0; c < 3; ++c) {  // RGB channels
        for (int i = 0; i < target_h; ++i) {
            for (int j = 0; j < target_w; ++j) {
                // Position dans le latent
                float src_i = i * h_ratio;
                float src_j = j * w_ratio;
                
                int i0 = static_cast<int>(src_i);
                int j0 = static_cast<int>(src_j);
                int i1 = std::min(i0 + 1, h - 1);
                int j1 = std::min(j0 + 1, w - 1);
                
                float di = src_i - i0;
                float dj = src_j - j0;
                
                // Interpolation bilinéaire (simplifiée, utilise juste le premier canal latent)
                int latent_c = (c * config_.latent_channels) / 3;
                if (latent_c >= config_.latent_channels) latent_c = config_.latent_channels - 1;
                
                int idx00 = latent_c * h * w + i0 * w + j0;
                int idx01 = latent_c * h * w + i0 * w + j1;
                int idx10 = latent_c * h * w + i1 * w + j0;
                int idx11 = latent_c * h * w + i1 * w + j1;
                
                float val = 0.0f;
                if (idx00 < static_cast<int>(latent.size())) {
                    float v00 = latent[idx00];
                    float v01 = (idx01 < static_cast<int>(latent.size())) ? latent[idx01] : v00;
                    float v10 = (idx10 < static_cast<int>(latent.size())) ? latent[idx10] : v00;
                    float v11 = (idx11 < static_cast<int>(latent.size())) ? latent[idx11] : v00;
                    
                    val = (1 - di) * (1 - dj) * v00 +
                          (1 - di) * dj * v01 +
                          di * (1 - dj) * v10 +
                          di * dj * v11;
                }
                
                // Tanh activation pour borner entre [-1, 1]
                val = std::tanh(val);
                
                int out_idx = c * target_h * target_w + i * target_w + j;
                if (out_idx < image_size) {
                    image[out_idx] = val;
                }
            }
        }
    }
    
    return image;
}

std::vector<float> FluxModel::encodeText(const std::vector<int>& tokens) {
    // Forward pass complet à travers le text encoder (style CLIP)
    int seq_len = std::min(static_cast<int>(tokens.size()), config_.text_max_length);
    int dim = config_.text_embed_dim;
    
    // 1. Token embeddings + position embeddings
    std::vector<float> embeddings(seq_len * dim, 0.0f);
    
    for (int i = 0; i < seq_len; ++i) {
        int token = (i < static_cast<int>(tokens.size())) ? tokens[i] : 0;
        token = std::min(token, config_.vocab_size - 1);
        
        for (int d = 0; d < dim; ++d) {
            // Token embedding (simulé avec initialisation simple)
            float token_emb = std::sin(token * 0.1f + d * 0.01f);
            
            // Position embedding (sinusoidal)
            float pos = static_cast<float>(i);
            float freq = std::exp(-2.0f * d / dim * std::log(10000.0f));
            float pos_emb = (d % 2 == 0) ? std::sin(pos * freq) : std::cos(pos * freq);
            
            embeddings[i * dim + d] = token_emb + pos_emb;
        }
    }
    
    // 2. Transformer layers (12 couches)
    std::vector<float> x = embeddings;
    
    for (int layer = 0; layer < 12; ++layer) {
        // Self-attention (simplifié)
        std::vector<float> attn_out(seq_len * dim);
        
        for (int i = 0; i < seq_len; ++i) {
            for (int d = 0; d < dim; ++d) {
                float sum = 0.0f;
                float total_weight = 0.0f;
                
                // Attention simplifiée (moyenne pondérée)
                for (int j = 0; j < seq_len; ++j) {
                    // Score d'attention simplifié
                    float score = 0.0f;
                    for (int k = 0; k < dim; ++k) {
                        score += x[i * dim + k] * x[j * dim + k];
                    }
                    float weight = std::exp(score / std::sqrt(static_cast<float>(dim)));
                    
                    sum += weight * x[j * dim + d];
                    total_weight += weight;
                }
                
                attn_out[i * dim + d] = sum / (total_weight + 1e-8f);
            }
        }
        
        // Residual connection + Layer norm
        for (int i = 0; i < seq_len * dim; ++i) {
            x[i] = x[i] + attn_out[i];  // Residual
        }
        
        // MLP (Feed-forward)
        std::vector<float> mlp_out(seq_len * dim);
        for (int i = 0; i < seq_len; ++i) {
            for (int d = 0; d < dim; ++d) {
                // FFN simplifié: expansion puis contraction
                float val = x[i * dim + d];
                val = std::max(0.0f, val);  // ReLU
                val = val * 4.0f;  // Expansion
                val = val / 4.0f;  // Contraction
                mlp_out[i * dim + d] = val;
            }
        }
        
        // Residual connection
        for (int i = 0; i < seq_len * dim; ++i) {
            x[i] = x[i] + mlp_out[i];
        }
    }
    
    // 3. Projection vers transformer_dim si nécessaire
    if (config_.transformer_dim != dim) {
        std::vector<float> projected(seq_len * config_.transformer_dim);
        for (int i = 0; i < seq_len; ++i) {
            for (int d = 0; d < config_.transformer_dim; ++d) {
                float sum = 0.0f;
                for (int k = 0; k < dim; ++k) {
                    // Projection linéaire simplifiée
                    float weight = std::sin((d * dim + k) * 0.01f);
                    sum += x[i * dim + k] * weight;
                }
                projected[i * config_.transformer_dim + d] = sum;
            }
        }
        return projected;
    }
    
    return x;
}

// ============================================================================
// Diffusion Process
// ============================================================================

std::vector<float> FluxModel::addNoise(const std::vector<float>& latent, int timestep) {
    if (timestep < 0 || timestep >= config_.num_timesteps) {
        return latent;
    }
    
    float sqrt_alpha = sqrt_alphas_cumprod_[timestep];
    float sqrt_one_minus_alpha = sqrt_one_minus_alphas_cumprod_[timestep];
    
    std::vector<float> noisy_latent(latent.size());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < latent.size(); ++i) {
        float noise = dist(gen);
        noisy_latent[i] = sqrt_alpha * latent[i] + sqrt_one_minus_alpha * noise;
    }
    
    return noisy_latent;
}

std::vector<float> FluxModel::predictNoise(const std::vector<float>& noisy_latent,
                                           const std::vector<float>& text_embedding,
                                           int timestep) {
    // Forward pass complet à travers le diffusion transformer
    // avec conditioning par texte et timestep
    
    int latent_len = config_.latent_channels * config_.latent_resolution * config_.latent_resolution;
    int dim = config_.transformer_dim;
    
    // 1. Timestep embedding (sinusoidal + MLP)
    std::vector<float> time_emb(dim);
    for (int d = 0; d < dim; ++d) {
        float t = static_cast<float>(timestep);
        float freq = std::exp(-2.0f * d / config_.timestep_embed_dim * std::log(10000.0f));
        time_emb[d] = (d % 2 == 0) ? std::sin(t * freq) : std::cos(t * freq);
        // MLP transformation simplifiée
        time_emb[d] = std::tanh(time_emb[d] * 2.0f);
    }
    
    // 2. Projection du latent bruité vers l'espace transformer
    int num_latent_tokens = config_.latent_resolution * config_.latent_resolution;
    std::vector<float> latent_tokens(num_latent_tokens * dim);
    
    for (int i = 0; i < num_latent_tokens; ++i) {
        for (int d = 0; d < dim; ++d) {
            float sum = 0.0f;
            for (int c = 0; c < config_.latent_channels; ++c) {
                int latent_idx = c * num_latent_tokens + i;
                if (latent_idx < static_cast<int>(noisy_latent.size())) {
                    // Projection linéaire simplifiée
                    float weight = std::cos((d * config_.latent_channels + c) * 0.01f);
                    sum += noisy_latent[latent_idx] * weight;
                }
            }
            latent_tokens[i * dim + d] = sum;
        }
    }
    
    // 3. Transformer blocks avec cross-attention
    std::vector<float> x = latent_tokens;
    int text_seq_len = text_embedding.size() / dim;
    
    for (int block = 0; block < config_.num_transformer_blocks; ++block) {
        // Self-attention sur les latents
        std::vector<float> self_attn_out(num_latent_tokens * dim);
        for (int i = 0; i < num_latent_tokens; ++i) {
            for (int d = 0; d < dim; ++d) {
                float sum = 0.0f;
                float total_weight = 0.0f;
                
                for (int j = 0; j < num_latent_tokens; ++j) {
                    float score = 0.0f;
                    for (int k = 0; k < dim; ++k) {
                        score += x[i * dim + k] * x[j * dim + k];
                    }
                    float weight = std::exp(score / std::sqrt(static_cast<float>(dim)));
                    sum += weight * x[j * dim + d];
                    total_weight += weight;
                }
                
                self_attn_out[i * dim + d] = sum / (total_weight + 1e-8f);
            }
        }
        
        // Residual + modulation par timestep (AdaLN)
        for (int i = 0; i < num_latent_tokens; ++i) {
            for (int d = 0; d < dim; ++d) {
                float scale = 1.0f + time_emb[d] * 0.1f;
                float shift = time_emb[d] * 0.1f;
                int idx = i * dim + d;
                x[idx] = (x[idx] + self_attn_out[idx]) * scale + shift;
            }
        }
        
        // Cross-attention avec le texte
        std::vector<float> cross_attn_out(num_latent_tokens * dim);
        for (int i = 0; i < num_latent_tokens; ++i) {
            for (int d = 0; d < dim; ++d) {
                float sum = 0.0f;
                float total_weight = 0.0f;
                
                for (int j = 0; j < text_seq_len && j * dim < static_cast<int>(text_embedding.size()); ++j) {
                    float score = 0.0f;
                    for (int k = 0; k < dim; ++k) {
                        int text_idx = j * dim + k;
                        if (text_idx < static_cast<int>(text_embedding.size())) {
                            score += x[i * dim + k] * text_embedding[text_idx];
                        }
                    }
                    float weight = std::exp(score / std::sqrt(static_cast<float>(dim)));
                    
                    int text_val_idx = j * dim + d;
                    if (text_val_idx < static_cast<int>(text_embedding.size())) {
                        sum += weight * text_embedding[text_val_idx];
                    }
                    total_weight += weight;
                }
                
                cross_attn_out[i * dim + d] = sum / (total_weight + 1e-8f);
            }
        }
        
        // Residual
        for (int i = 0; i < num_latent_tokens * dim; ++i) {
            x[i] = x[i] + cross_attn_out[i];
        }
        
        // MLP
        std::vector<float> mlp_out(num_latent_tokens * dim);
        for (int i = 0; i < num_latent_tokens * dim; ++i) {
            float val = x[i];
            val = std::max(0.0f, val);  // ReLU
            val = val * config_.mlp_ratio;  // Expansion
            val = std::tanh(val);  // Non-linéarité
            val = val / config_.mlp_ratio;  // Contraction
            mlp_out[i] = val;
        }
        
        // Residual
        for (int i = 0; i < num_latent_tokens * dim; ++i) {
            x[i] = x[i] + mlp_out[i];
        }
    }
    
    // 4. Projection vers l'espace latent (prédiction du bruit)
    std::vector<float> predicted_noise(noisy_latent.size());
    for (int c = 0; c < config_.latent_channels; ++c) {
        for (int i = 0; i < num_latent_tokens; ++i) {
            float sum = 0.0f;
            for (int d = 0; d < dim; ++d) {
                // Projection inverse simplifiée
                float weight = std::sin((c * dim + d) * 0.01f);
                sum += x[i * dim + d] * weight;
            }
            
            int noise_idx = c * num_latent_tokens + i;
            if (noise_idx < static_cast<int>(predicted_noise.size())) {
                predicted_noise[noise_idx] = sum;
            }
        }
    }
    
    return predicted_noise;
}

std::vector<float> FluxModel::denoisingStep(const std::vector<float>& noisy_latent,
                                            const std::vector<float>& text_embedding,
                                            int timestep) {
    if (timestep <= 0) {
        return noisy_latent;
    }
    
    // Prédire le bruit
    std::vector<float> predicted_noise = predictNoise(noisy_latent, text_embedding, timestep);
    
    // Retirer le bruit prédit
    float alpha = alphas_[timestep];
    float alpha_cumprod = alphas_cumprod_[timestep];
    float alpha_cumprod_prev = (timestep > 0) ? alphas_cumprod_[timestep - 1] : 1.0f;
    
    float sqrt_alpha = std::sqrt(alpha);
    float sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alphas_cumprod_[timestep];
    
    std::vector<float> denoised(noisy_latent.size());
    for (size_t i = 0; i < noisy_latent.size(); ++i) {
        // DDPM sampling formula
        float pred_x0 = (noisy_latent[i] - sqrt_one_minus_alpha_cumprod * predicted_noise[i]) / 
                        std::sqrt(alpha_cumprod);
        
        float coef1 = std::sqrt(alpha_cumprod_prev) * betas_[timestep] / (1.0f - alpha_cumprod);
        float coef2 = std::sqrt(alpha) * (1.0f - alpha_cumprod_prev) / (1.0f - alpha_cumprod);
        
        denoised[i] = coef1 * pred_x0 + coef2 * noisy_latent[i];
    }
    
    return denoised;
}

// ============================================================================
// Generation
// ============================================================================

std::vector<float> FluxModel::generate(const std::string& prompt, int num_steps, float guidance_scale) {
    std::vector<int> tokens = tokenizePrompt(prompt);
    return generateFromTokens(tokens, num_steps, guidance_scale);
}

std::vector<float> FluxModel::generateFromTokens(const std::vector<int>& tokens, 
                                                  int num_steps, 
                                                  float guidance_scale) {
    // Encoder le texte
    std::vector<float> text_embedding = encodeText(tokens);
    
    // Générer empty text embedding pour classifier-free guidance
    std::vector<int> empty_tokens(tokens.size(), 0);
    std::vector<float> empty_text_embedding = encodeText(empty_tokens);
    
    // Initialiser latent avec bruit pur
    int latent_size = config_.latent_channels * 
                      config_.latent_resolution * 
                      config_.latent_resolution;
    
    std::vector<float> latent(latent_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (auto& val : latent) {
        val = dist(gen);
    }
    
    // Denoising loop
    int step_size = config_.num_timesteps / num_steps;
    for (int i = num_steps - 1; i >= 0; --i) {
        int timestep = i * step_size;
        
        // Classifier-free guidance
        std::vector<float> noise_pred_cond = predictNoise(latent, text_embedding, timestep);
        std::vector<float> noise_pred_uncond = predictNoise(latent, empty_text_embedding, timestep);
        
        // Combiner les prédictions
        std::vector<float> noise_pred(latent_size);
        for (size_t j = 0; j < latent_size; ++j) {
            noise_pred[j] = noise_pred_uncond[j] + 
                           guidance_scale * (noise_pred_cond[j] - noise_pred_uncond[j]);
        }
        
        // Appliquer le denoising step (en utilisant noise_pred au lieu de le recalculer)
        float alpha = alphas_[timestep];
        float alpha_cumprod = alphas_cumprod_[timestep];
        float alpha_cumprod_prev = (timestep > 0) ? alphas_cumprod_[timestep - 1] : 1.0f;
        
        for (size_t j = 0; j < latent_size; ++j) {
            float pred_x0 = (latent[j] - sqrt_one_minus_alphas_cumprod_[timestep] * noise_pred[j]) / 
                           std::sqrt(alpha_cumprod);
            
            float coef1 = std::sqrt(alpha_cumprod_prev) * betas_[timestep] / (1.0f - alpha_cumprod);
            float coef2 = std::sqrt(alpha) * (1.0f - alpha_cumprod_prev) / (1.0f - alpha_cumprod);
            
            latent[j] = coef1 * pred_x0 + coef2 * latent[j];
        }
    }
    
    // Décoder le latent final en image
    return decodeLatent(latent);
}

// ============================================================================
// Training
// ============================================================================

float FluxModel::computeDiffusionLoss(const std::vector<float>& image, 
                                      const std::vector<int>& tokens) {
    // Encoder l'image en latent
    std::vector<float> latent = encodeImage(image);
    
    // Encoder le texte
    std::vector<float> text_embedding = encodeText(tokens);
    
    // Sample un timestep aléatoire
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> timestep_dist(0, config_.num_timesteps - 1);
    int timestep = timestep_dist(gen);
    
    // Ajouter du bruit
    std::vector<float> noisy_latent = addNoise(latent, timestep);
    
    // Prédire le bruit
    std::vector<float> predicted_noise = predictNoise(noisy_latent, text_embedding, timestep);
    
    // Calculer la loss MSE entre le vrai bruit et le bruit prédit
    // (Dans addNoise, on devrait retourner aussi le vrai bruit pour calculer la loss)
    // Pour l'instant, on retourne une valeur placeholder
    return 0.0f;
}

// ============================================================================
// Utilities
// ============================================================================

std::vector<int> FluxModel::tokenizePrompt(const std::string& prompt) {
    if (prompt_tokenizer_) {
        return prompt_tokenizer_->tokenize(prompt);
    }
    
    // Fallback: tokenization basique par mots
    std::vector<int> tokens;
    tokens.push_back(0);  // BOS token
    
    // Tokenization simple: chaque mot devient un token basé sur son hash
    std::string word;
    for (char c : prompt) {
        if (std::isspace(c) || std::ispunct(c)) {
            if (!word.empty()) {
                // Hash simple du mot vers un token ID
                int token_id = 1;  // Start at 1 (0 is BOS)
                for (char ch : word) {
                    token_id = (token_id * 31 + static_cast<int>(ch)) % (config_.vocab_size - 2);
                }
                token_id += 1;  // Éviter 0 (BOS)
                tokens.push_back(token_id);
                word.clear();
            }
        } else {
            word += std::tolower(c);
        }
    }
    
    // Dernier mot
    if (!word.empty()) {
        int token_id = 1;
        for (char ch : word) {
            token_id = (token_id * 31 + static_cast<int>(ch)) % (config_.vocab_size - 2);
        }
        token_id += 1;
        tokens.push_back(token_id);
    }
    
    tokens.push_back(config_.vocab_size - 1);  // EOS token
    
    // Padding ou truncation vers text_max_length
    if (tokens.size() > static_cast<size_t>(config_.text_max_length)) {
        tokens.resize(config_.text_max_length);
        tokens[config_.text_max_length - 1] = config_.vocab_size - 1;  // EOS
    } else {
        while (tokens.size() < static_cast<size_t>(config_.text_max_length)) {
            tokens.push_back(0);  // PAD token
        }
    }
    
    return tokens;
}

float FluxModel::getTimeEmbedding(int timestep, int dim) {
    // Sinusoidal position embedding pour le timestep
    // Basé sur "Attention is All You Need"
    
    float t = static_cast<float>(timestep);
    float freq = std::exp(-2.0f * dim / config_.timestep_embed_dim * std::log(10000.0f));
    
    // Alterner sin et cos
    if (dim % 2 == 0) {
        return std::sin(t * freq);
    } else {
        return std::cos(t * freq);
    }
}

} // namespace ModelArchitectures
