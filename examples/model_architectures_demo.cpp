#include "Models/ModelArchitectures.hpp"
#include <iostream>
#include <iomanip>

using namespace ModelArchitectures;

void printModelInfo(const Model& model, const std::string& name) {
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    std::cout << "  " << name << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    std::cout << "  Paramètres: " << model.totalParamCount() << std::endl;
    std::cout << "  Taille (FP32): " << std::fixed << std::setprecision(2) 
              << (model.totalParamCount() * 4.0 / 1024.0 / 1024.0) << " MB" << std::endl;
}

int main() {
    std::cout << "\n╔════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   Mímir Framework - Model Architectures Demo   ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════╝" << std::endl;
    
    // ========== UNet ==========
    {
        Model unet;
        unet.setName("UNet-Segmentation");
        
        UNetConfig config;
        config.input_channels = 3;
        config.output_channels = 1;
        config.base_channels = 64;
        config.num_levels = 4;
        config.use_attention = true;
        
        buildUNet(unet, config);
        unet.allocateParams();
        printModelInfo(unet, "UNet (Segmentation)");
        std::cout << "  Usage: Segmentation, image-to-image" << std::endl;
    }
    
    // ========== VAE ==========
    {
        Model vae;
        vae.setName("VAE-MNIST");
        
        VAEConfig config;
        config.input_dim = 784;
        config.latent_dim = 128;
        config.encoder_hidden = {512, 256};
        config.decoder_hidden = {256, 512};
        
        buildVAE(vae, config);
        vae.allocateParams();
        printModelInfo(vae, "VAE (Variational Autoencoder)");
        std::cout << "  Usage: Génération, compression" << std::endl;
    }
    
    // ========== Vision Transformer ==========
    {
        Model vit;
        vit.setName("ViT-Base");
        
        ViTConfig config;
        config.image_size = 224;
        config.patch_size = 16;
        config.num_classes = 1000;
        config.d_model = 768;
        config.num_heads = 12;
        config.num_layers = 12;
        
        buildViT(vit, config);
        vit.allocateParams();
        printModelInfo(vit, "Vision Transformer (ViT-Base)");
        std::cout << "  Usage: Classification d'images" << std::endl;
    }
    
    // ========== GAN ==========
    {
        Model generator;
        generator.setName("GAN-Generator");
        
        GANConfig config;
        config.latent_dim = 100;
        config.image_size = 64;
        config.image_channels = 3;
        config.self_attention = true;
        
        buildGenerator(generator, config);
        generator.allocateParams();
        printModelInfo(generator, "GAN Generator");
        std::cout << "  Usage: Génération d'images" << std::endl;
        
        Model discriminator;
        discriminator.setName("GAN-Discriminator");
        buildDiscriminator(discriminator, config);
        discriminator.allocateParams();
        printModelInfo(discriminator, "GAN Discriminator");
        std::cout << "  Usage: Classification real/fake" << std::endl;
    }
    
    // ========== Diffusion Model ==========
    {
        Model diffusion;
        diffusion.setName("DDPM-32x32");
        
        DiffusionConfig config;
        config.image_size = 32;
        config.image_channels = 3;
        config.base_channels = 128;
        config.num_res_blocks = 2;
        
        buildDiffusion(diffusion, config);
        diffusion.allocateParams();
        printModelInfo(diffusion, "Diffusion Model (DDPM)");
        std::cout << "  Usage: Génération d'images haute qualité" << std::endl;
    }
    
    // ========== Transformer ==========
    {
        Model transformer;
        transformer.setName("GPT-Small");
        
        TransformerConfig config;
        config.vocab_size = 50000;
        config.max_seq_len = 1024;
        config.d_model = 768;
        config.num_heads = 12;
        config.num_layers = 12;
        config.d_ff = 3072;
        config.causal = true;
        
        buildTransformer(transformer, config);
        transformer.allocateParams();
        printModelInfo(transformer, "Transformer (GPT-Small)");
        std::cout << "  Usage: Génération de texte" << std::endl;
    }
    
    // ========== ResNet ==========
    {
        Model resnet;
        resnet.setName("ResNet-50");
        
        ResNetConfig config;
        config.num_classes = 1000;
        config.layers = {3, 4, 6, 3};
        config.base_channels = 64;
        config.use_bottleneck = true;
        
        buildResNet(resnet, config);
        resnet.allocateParams();
        printModelInfo(resnet, "ResNet-50");
        std::cout << "  Usage: Classification ImageNet" << std::endl;
    }
    
    // ========== MobileNet ==========
    {
        Model mobilenet;
        mobilenet.setName("MobileNetV2");
        
        MobileNetConfig config;
        config.num_classes = 1000;
        config.width_multiplier = 1.0f;
        config.resolution = 224;
        
        buildMobileNetV2(mobilenet, config);
        mobilenet.allocateParams();
        printModelInfo(mobilenet, "MobileNetV2");
        std::cout << "  Usage: Classification mobile/embarqué" << std::endl;
    }
    
    // ========== Résumé ==========
    std::cout << "\n╔════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║              Architectures Disponibles         ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════╝" << std::endl << std::endl;
    std::cout << "  ✓ UNet          - Segmentation, U-shaped encoder-decoder" << std::endl;
    std::cout << "  ✓ VAE           - Variational Autoencoder, génération" << std::endl;
    std::cout << "  ✓ ViT           - Vision Transformer, classification" << std::endl;
    std::cout << "  ✓ GAN           - Generative Adversarial Network" << std::endl;
    std::cout << "  ✓ Diffusion     - DDPM, génération haute qualité" << std::endl;
    std::cout << "  ✓ Transformer   - GPT-style, génération texte" << std::endl;
    std::cout << "  ✓ ResNet        - Deep residual networks" << std::endl;
    std::cout << "  ✓ MobileNet     - Efficient mobile architecture" << std::endl;
    
    std::cout << "\n💡 Toutes les architectures sont prêtes à l'emploi!" << std::endl;
    std::cout << "   Utilisez buildXXX(model, config) pour construire." << std::endl;
    
    std::cout << "\n🚀 Optimisations hardware activées:" << std::endl;
    std::cout << "   • AVX2 vectorisation" << std::endl;
    std::cout << "   • FMA saturé (3 ops/cycle)" << std::endl;
    std::cout << "   • OpenMP parallélisation" << std::endl;
    std::cout << "   • F16C storage (ready)" << std::endl;
    
    return 0;
}
