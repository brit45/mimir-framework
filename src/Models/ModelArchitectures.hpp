#pragma once

#include "../Model.hpp"
#include "../Layers.hpp"
#include <vector>
#include <string>
#include <memory>

// ============================================================================
// Configuration des architectures modernes
// ============================================================================

namespace ModelArchitectures {

// ============================================================================
// UNet - Architecture pour segmentation et génération d'images
// ============================================================================

struct UNetConfig {
    int input_channels = 3;
    int output_channels = 3;
    int base_channels = 64;
    int num_levels = 4;           // Nombre de downsampling/upsampling
    int blocks_per_level = 2;     // Blocs conv par niveau
    bool use_attention = true;    // Attention aux niveaux profonds
    bool use_residual = true;     // Skip connections
    ActivationType activation = ActivationType::RELU;
    bool use_batchnorm = true;
    float dropout = 0.0f;
};

inline void buildUNet(Model& model, const UNetConfig& config) {
    int channels = config.base_channels;
    
    // Encoder (downsampling)
    for (int level = 0; level < config.num_levels; ++level) {
        int in_c = (level == 0) ? config.input_channels : channels;
        channels = config.base_channels * (1 << level);
        
        for (int block = 0; block < config.blocks_per_level; ++block) {
            std::string name = "encoder_L" + std::to_string(level) + "_B" + std::to_string(block);
            model.push(name + "_conv", "Conv2d", in_c * 3 * 3 * channels + channels);
            
            if (config.use_batchnorm) {
                model.push(name + "_bn", "BatchNorm", channels * 2);
            }
            
            in_c = channels;
        }
        
        // Downsampling (sauf dernier niveau)
        if (level < config.num_levels - 1) {
            model.push("down_L" + std::to_string(level), "MaxPool2d", 0);
        }
    }
    
    // Bottleneck avec attention
    if (config.use_attention) {
        model.push("bottleneck_attention", "SelfAttention", channels * channels * 3);
    }
    
    // Decoder (upsampling)
    for (int level = config.num_levels - 1; level >= 0; --level) {
        channels = config.base_channels * (1 << level);
        
        if (level < config.num_levels - 1) {
            model.push("up_L" + std::to_string(level), "ConvTranspose2d", 
                      channels * 2 * 2 * 2 * channels + channels);
        }
        
        for (int block = 0; block < config.blocks_per_level; ++block) {
            std::string name = "decoder_L" + std::to_string(level) + "_B" + std::to_string(block);
            int in_c = (block == 0 && config.use_residual) ? channels * 2 : channels;
            model.push(name + "_conv", "Conv2d", in_c * 3 * 3 * channels + channels);
            
            if (config.use_batchnorm) {
                model.push(name + "_bn", "BatchNorm", channels * 2);
            }
        }
    }
    
    // Output projection
    model.push("output_conv", "Conv2d", channels * 1 * 1 * config.output_channels + config.output_channels);
}

// ============================================================================
// VAE - Variational Autoencoder
// ============================================================================

struct VAEConfig {
    int input_dim = 784;          // 28*28 pour MNIST
    int latent_dim = 128;
    std::vector<int> encoder_hidden = {512, 256};
    std::vector<int> decoder_hidden = {256, 512};
    ActivationType activation = ActivationType::RELU;
    bool use_batchnorm = false;
};

inline void buildVAE(Model& model, const VAEConfig& config) {
    // Encoder
    int prev_dim = config.input_dim;
    for (size_t i = 0; i < config.encoder_hidden.size(); ++i) {
        model.push("encoder_fc" + std::to_string(i), "Linear", 
                  prev_dim * config.encoder_hidden[i] + config.encoder_hidden[i]);
        if (config.use_batchnorm) {
            model.push("encoder_bn" + std::to_string(i), "BatchNorm", config.encoder_hidden[i] * 2);
        }
        prev_dim = config.encoder_hidden[i];
    }
    
    // Latent space (mu et logvar)
    model.push("latent_mu", "Linear", prev_dim * config.latent_dim + config.latent_dim);
    model.push("latent_logvar", "Linear", prev_dim * config.latent_dim + config.latent_dim);
    
    // Decoder
    prev_dim = config.latent_dim;
    for (size_t i = 0; i < config.decoder_hidden.size(); ++i) {
        model.push("decoder_fc" + std::to_string(i), "Linear",
                  prev_dim * config.decoder_hidden[i] + config.decoder_hidden[i]);
        if (config.use_batchnorm) {
            model.push("decoder_bn" + std::to_string(i), "BatchNorm", config.decoder_hidden[i] * 2);
        }
        prev_dim = config.decoder_hidden[i];
    }
    
    // Output
    model.push("decoder_output", "Linear", prev_dim * config.input_dim + config.input_dim);
}

// ============================================================================
// Vision Transformer (ViT)
// ============================================================================

struct ViTConfig {
    int image_size = 224;
    int patch_size = 16;
    int num_classes = 1000;
    int d_model = 768;
    int num_heads = 12;
    int num_layers = 12;
    int mlp_ratio = 4;            // MLP hidden = d_model * mlp_ratio
    float dropout = 0.1f;
    bool use_cls_token = true;
};

inline void buildViT(Model& model, const ViTConfig& config) {
    int num_patches = (config.image_size / config.patch_size) * (config.image_size / config.patch_size);
    int patch_dim = 3 * config.patch_size * config.patch_size;
    
    // Patch embedding
    model.push("patch_embed", "Linear", patch_dim * config.d_model + config.d_model);
    
    // Position embedding (learned)
    int seq_len = num_patches + (config.use_cls_token ? 1 : 0);
    model.push("pos_embed", "Embedding", seq_len * config.d_model);
    
    if (config.use_cls_token) {
        model.push("cls_token", "Embedding", config.d_model);
    }
    
    // Transformer blocks
    for (int i = 0; i < config.num_layers; ++i) {
        std::string prefix = "block" + std::to_string(i);
        
        // Layer norm 1
        model.push(prefix + "_ln1", "LayerNorm", config.d_model * 2);
        
        // Multi-head attention
        model.push(prefix + "_attn_qkv", "Linear", config.d_model * config.d_model * 3 + config.d_model * 3);
        model.push(prefix + "_attn_proj", "Linear", config.d_model * config.d_model + config.d_model);
        
        // Layer norm 2
        model.push(prefix + "_ln2", "LayerNorm", config.d_model * 2);
        
        // MLP
        int mlp_hidden = config.d_model * config.mlp_ratio;
        model.push(prefix + "_mlp_fc1", "Linear", config.d_model * mlp_hidden + mlp_hidden);
        model.push(prefix + "_mlp_fc2", "Linear", mlp_hidden * config.d_model + config.d_model);
    }
    
    // Classification head
    model.push("head_norm", "LayerNorm", config.d_model * 2);
    model.push("head", "Linear", config.d_model * config.num_classes + config.num_classes);
}

// ============================================================================
// GAN - Generative Adversarial Network
// ============================================================================

struct GANConfig {
    int latent_dim = 100;
    int image_size = 64;
    int image_channels = 3;
    int g_base_channels = 64;     // Generator
    int d_base_channels = 64;     // Discriminator
    bool spectral_norm = false;
    bool self_attention = true;
};

inline void buildGenerator(Model& model, const GANConfig& config) {
    // Projection initiale: latent -> 4x4x(base*8)
    int start_size = 4;
    int start_channels = config.g_base_channels * 8;
    model.push("gen_project", "Linear", 
              config.latent_dim * (start_size * start_size * start_channels) + (start_size * start_size * start_channels));
    
    // Upsampling blocks
    int current_size = start_size;
    int channels = start_channels;
    
    while (current_size < config.image_size) {
        std::string prefix = "gen_up" + std::to_string(current_size);
        int next_channels = channels / 2;
        
        model.push(prefix + "_upsample", "ConvTranspose2d",
                  channels * 4 * 4 * next_channels + next_channels);
        model.push(prefix + "_bn", "BatchNorm", next_channels * 2);
        
        if (config.self_attention && current_size == 16) {
            model.push(prefix + "_attn", "SelfAttention", next_channels * next_channels * 3);
        }
        
        current_size *= 2;
        channels = next_channels;
    }
    
    // Output
    model.push("gen_output", "Conv2d", channels * 3 * 3 * config.image_channels + config.image_channels);
}

inline void buildDiscriminator(Model& model, const GANConfig& config) {
    // Downsampling blocks
    int current_size = config.image_size;
    int channels = config.d_base_channels;
    int in_channels = config.image_channels;
    
    while (current_size > 4) {
        std::string prefix = "disc_down" + std::to_string(current_size);
        
        model.push(prefix + "_conv", "Conv2d", in_channels * 4 * 4 * channels + channels);
        
        if (config.spectral_norm) {
            model.push(prefix + "_sn", "SpectralNorm", 0);
        }
        
        if (config.self_attention && current_size == 16) {
            model.push(prefix + "_attn", "SelfAttention", channels * channels * 3);
        }
        
        current_size /= 2;
        in_channels = channels;
        channels *= 2;
    }
    
    // Output (real/fake classifier)
    model.push("disc_output", "Linear", (channels / 2) * 4 * 4 * 1 + 1);
}

// ============================================================================
// Diffusion Model (DDPM)
// ============================================================================

struct DiffusionConfig {
    int image_size = 32;
    int image_channels = 3;
    int base_channels = 128;
    int num_res_blocks = 2;
    std::vector<int> channel_multipliers = {1, 2, 2, 2};
    std::vector<int> attention_levels = {1, 2, 3};  // Niveaux avec attention
    int time_embed_dim = 512;
    bool use_scale_shift_norm = true;
};

inline void buildDiffusion(Model& model, const DiffusionConfig& config) {
    // Time embedding MLP
    model.push("time_mlp_fc1", "Linear", config.time_embed_dim * config.time_embed_dim * 4 + config.time_embed_dim * 4);
    model.push("time_mlp_fc2", "Linear", config.time_embed_dim * 4 * config.time_embed_dim * 4 + config.time_embed_dim * 4);
    
    // Input projection
    model.push("input_conv", "Conv2d", 
              config.image_channels * 3 * 3 * config.base_channels + config.base_channels);
    
    // Encoder
    int channels = config.base_channels;
    for (size_t level = 0; level < config.channel_multipliers.size(); ++level) {
        int out_channels = config.base_channels * config.channel_multipliers[level];
        
        for (int block = 0; block < config.num_res_blocks; ++block) {
            std::string prefix = "down_L" + std::to_string(level) + "_B" + std::to_string(block);
            
            // ResBlock
            model.push(prefix + "_conv1", "Conv2d", channels * 3 * 3 * out_channels + out_channels);
            model.push(prefix + "_time_proj", "Linear", config.time_embed_dim * 4 * out_channels + out_channels);
            model.push(prefix + "_conv2", "Conv2d", out_channels * 3 * 3 * out_channels + out_channels);
            
            // Attention si spécifié
            if (std::find(config.attention_levels.begin(), config.attention_levels.end(), level) != config.attention_levels.end()) {
                model.push(prefix + "_attn", "SelfAttention", out_channels * out_channels * 3);
            }
            
            channels = out_channels;
        }
        
        // Downsampling (sauf dernier niveau)
        if (level < config.channel_multipliers.size() - 1) {
            model.push("down_L" + std::to_string(level) + "_sample", "Conv2d",
                      channels * 3 * 3 * channels + channels);
        }
    }
    
    // Middle
    model.push("mid_block1_conv1", "Conv2d", channels * 3 * 3 * channels + channels);
    model.push("mid_attn", "SelfAttention", channels * channels * 3);
    model.push("mid_block2_conv1", "Conv2d", channels * 3 * 3 * channels + channels);
    
    // Decoder (symétrique)
    for (int level = config.channel_multipliers.size() - 1; level >= 0; --level) {
        int out_channels = config.base_channels * config.channel_multipliers[level];
        
        for (int block = 0; block < config.num_res_blocks + 1; ++block) {
            std::string prefix = "up_L" + std::to_string(level) + "_B" + std::to_string(block);
            
            model.push(prefix + "_conv1", "Conv2d", channels * 3 * 3 * out_channels + out_channels);
            model.push(prefix + "_time_proj", "Linear", config.time_embed_dim * 4 * out_channels + out_channels);
            model.push(prefix + "_conv2", "Conv2d", out_channels * 3 * 3 * out_channels + out_channels);
            
            if (std::find(config.attention_levels.begin(), config.attention_levels.end(), level) != config.attention_levels.end()) {
                model.push(prefix + "_attn", "SelfAttention", out_channels * out_channels * 3);
            }
            
            channels = out_channels;
        }
        
        // Upsampling
        if (level > 0) {
            model.push("up_L" + std::to_string(level) + "_sample", "ConvTranspose2d",
                      channels * 2 * 2 * channels + channels);
        }
    }
    
    // Output
    model.push("output_norm", "GroupNorm", channels * 2);
    model.push("output_conv", "Conv2d", channels * 3 * 3 * config.image_channels + config.image_channels);
}

// ============================================================================
// Transformer (GPT-style)
// ============================================================================

struct TransformerConfig {
    int vocab_size = 50000;
    int max_seq_len = 2048;
    int d_model = 768;
    int num_heads = 12;
    int num_layers = 12;
    int d_ff = 3072;              // FFN dimension
    float dropout = 0.1f;
    bool causal = true;           // Causal masking (GPT) vs bidirectional (BERT)
};

inline void buildTransformer(Model& model, const TransformerConfig& config) {
    // Token embedding
    model.push("token_embed", "Embedding", config.vocab_size * config.d_model);
    
    // Position embedding
    model.push("pos_embed", "Embedding", config.max_seq_len * config.d_model);
    
    // Transformer layers
    for (int i = 0; i < config.num_layers; ++i) {
        std::string prefix = "layer" + std::to_string(i);
        
        // Self-attention
        model.push(prefix + "_ln1", "LayerNorm", config.d_model * 2);
        model.push(prefix + "_attn_qkv", "Linear", config.d_model * config.d_model * 3 + config.d_model * 3);
        model.push(prefix + "_attn_out", "Linear", config.d_model * config.d_model + config.d_model);
        
        // Feed-forward
        model.push(prefix + "_ln2", "LayerNorm", config.d_model * 2);
        model.push(prefix + "_ffn_fc1", "Linear", config.d_model * config.d_ff + config.d_ff);
        model.push(prefix + "_ffn_fc2", "Linear", config.d_ff * config.d_model + config.d_model);
    }
    
    // Output head
    model.push("output_ln", "LayerNorm", config.d_model * 2);
    model.push("output_head", "Linear", config.d_model * config.vocab_size + config.vocab_size);
}

// ============================================================================
// ResNet
// ============================================================================

struct ResNetConfig {
    int num_classes = 1000;
    std::vector<int> layers = {3, 4, 6, 3};  // ResNet-50
    int base_channels = 64;
    bool use_bottleneck = true;
};

inline void buildResNet(Model& model, const ResNetConfig& config) {
    // Stem
    model.push("stem_conv", "Conv2d", 3 * 7 * 7 * config.base_channels + config.base_channels);
    model.push("stem_bn", "BatchNorm", config.base_channels * 2);
    model.push("stem_pool", "MaxPool2d", 0);
    
    // Residual stages
    int channels = config.base_channels;
    for (size_t stage = 0; stage < config.layers.size(); ++stage) {
        int out_channels = config.base_channels * (1 << stage);
        
        for (int block = 0; block < config.layers[stage]; ++block) {
            std::string prefix = "stage" + std::to_string(stage) + "_block" + std::to_string(block);
            
            if (config.use_bottleneck) {
                // Bottleneck: 1x1 -> 3x3 -> 1x1
                int bottleneck_channels = out_channels / 4;
                model.push(prefix + "_conv1", "Conv2d", channels * 1 * 1 * bottleneck_channels + bottleneck_channels);
                model.push(prefix + "_bn1", "BatchNorm", bottleneck_channels * 2);
                model.push(prefix + "_conv2", "Conv2d", bottleneck_channels * 3 * 3 * bottleneck_channels + bottleneck_channels);
                model.push(prefix + "_bn2", "BatchNorm", bottleneck_channels * 2);
                model.push(prefix + "_conv3", "Conv2d", bottleneck_channels * 1 * 1 * out_channels + out_channels);
                model.push(prefix + "_bn3", "BatchNorm", out_channels * 2);
            } else {
                // Basic block: 3x3 -> 3x3
                model.push(prefix + "_conv1", "Conv2d", channels * 3 * 3 * out_channels + out_channels);
                model.push(prefix + "_bn1", "BatchNorm", out_channels * 2);
                model.push(prefix + "_conv2", "Conv2d", out_channels * 3 * 3 * out_channels + out_channels);
                model.push(prefix + "_bn2", "BatchNorm", out_channels * 2);
            }
            
            // Shortcut projection si nécessaire
            if (channels != out_channels || (block == 0 && stage > 0)) {
                model.push(prefix + "_shortcut", "Conv2d", channels * 1 * 1 * out_channels + out_channels);
                model.push(prefix + "_shortcut_bn", "BatchNorm", out_channels * 2);
            }
            
            channels = out_channels;
        }
    }
    
    // Classification head
    model.push("avgpool", "AdaptiveAvgPool2d", 0);
    model.push("fc", "Linear", channels * config.num_classes + config.num_classes);
}

// ============================================================================
// MobileNet
// ============================================================================

struct MobileNetConfig {
    int num_classes = 1000;
    float width_multiplier = 1.0f;
    int resolution = 224;
};

inline void buildMobileNetV2(Model& model, const MobileNetConfig& config) {
    auto make_divisible = [](int v, int divisor = 8) {
        int new_v = std::max(divisor, int(v + divisor / 2) / divisor * divisor);
        if (new_v < 0.9 * v) new_v += divisor;
        return new_v;
    };
    
    int input_channels = make_divisible(32 * config.width_multiplier);
    
    // First layer
    model.push("conv1", "Conv2d", 3 * 3 * 3 * input_channels + input_channels);
    model.push("bn1", "BatchNorm", input_channels * 2);
    
    // Inverted residual blocks
    struct InvertedResidualConfig {
        int t;  // expansion factor
        int c;  // output channels
        int n;  // number of blocks
        int s;  // stride
    };
    
    std::vector<InvertedResidualConfig> inverted_residual_settings = {
        {1, 16, 1, 1},
        {6, 24, 2, 2},
        {6, 32, 3, 2},
        {6, 64, 4, 2},
        {6, 96, 3, 1},
        {6, 160, 3, 2},
        {6, 320, 1, 1},
    };
    
    for (size_t i = 0; i < inverted_residual_settings.size(); ++i) {
        auto& setting = inverted_residual_settings[i];
        int output_channels = make_divisible(setting.c * config.width_multiplier);
        
        for (int j = 0; j < setting.n; ++j) {
            std::string prefix = "block" + std::to_string(i) + "_" + std::to_string(j);
            int stride = (j == 0) ? setting.s : 1;
            int hidden_dim = input_channels * setting.t;
            
            // Expansion
            if (setting.t != 1) {
                model.push(prefix + "_expand", "Conv2d", input_channels * 1 * 1 * hidden_dim + hidden_dim);
                model.push(prefix + "_expand_bn", "BatchNorm", hidden_dim * 2);
            }
            
            // Depthwise
            model.push(prefix + "_depthwise", "Conv2d", hidden_dim * 3 * 3 * 1 + hidden_dim);
            model.push(prefix + "_depthwise_bn", "BatchNorm", hidden_dim * 2);
            
            // Projection
            model.push(prefix + "_project", "Conv2d", hidden_dim * 1 * 1 * output_channels + output_channels);
            model.push(prefix + "_project_bn", "BatchNorm", output_channels * 2);
            
            input_channels = output_channels;
        }
    }
    
    // Last layer
    int last_channels = make_divisible(1280 * config.width_multiplier);
    model.push("conv_last", "Conv2d", input_channels * 1 * 1 * last_channels + last_channels);
    model.push("bn_last", "BatchNorm", last_channels * 2);
    
    // Classifier
    model.push("avgpool", "AdaptiveAvgPool2d", 0);
    model.push("classifier", "Linear", last_channels * config.num_classes + config.num_classes);
}

} // namespace ModelArchitectures
