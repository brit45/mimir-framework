#include "Models/Registry/ModelArchitectures.hpp"

#include <iomanip>
#include <iostream>
#include <string>

static void printModelInfo(const Model& model, const std::string& title) {
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    std::cout << "  Paramètres: " << model.totalParamCount() << std::endl;
    std::cout << "  Taille (FP32): " << std::fixed << std::setprecision(2)
              << (model.totalParamCount() * 4.0 / 1024.0 / 1024.0) << " MB" << std::endl;
}

int main() {
    std::cout << "╔════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   Mímir Framework - Model Architectures Demo   ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════╝" << std::endl;

    // ========== UNet ==========
    {
        auto cfg = ModelArchitectures::defaultConfig("unet");
        auto unet = ModelArchitectures::create("unet", cfg);
        unet->setName("UNet-Segmentation");
        unet->allocateParams();
        std::cout << "UNet layers: " << unet->getLayers().size() << "\n";
        printModelInfo(*unet, "UNet (Segmentation)");
        std::cout << "  Usage: Segmentation, image-to-image" << std::endl;
    }

    // ========== Vision Transformer ==========
    {
        auto cfg = ModelArchitectures::defaultConfig("vit");
        auto vit = ModelArchitectures::create("vit", cfg);
        vit->setName("ViT-Base");
        vit->allocateParams();
        std::cout << "ViT layers: " << vit->getLayers().size() << "\n";
        printModelInfo(*vit, "Vision Transformer (ViT-Base)");
        std::cout << "  Usage: Classification d'images" << std::endl;
    }

    // ========== GAN ==========
    {
        auto cfg_g = ModelArchitectures::defaultConfig("gan_generator");
        auto generator = ModelArchitectures::create("gan_generator", cfg_g);
        generator->setName("GAN-Generator");
        generator->allocateParams();
        std::cout << "GAN generator layers: " << generator->getLayers().size() << "\n";
        printModelInfo(*generator, "GAN Generator");
        std::cout << "  Usage: Génération d'images" << std::endl;

        auto cfg_d = ModelArchitectures::defaultConfig("gan_discriminator");
        auto discriminator = ModelArchitectures::create("gan_discriminator", cfg_d);
        discriminator->setName("GAN-Discriminator");
        discriminator->allocateParams();
        std::cout << "GAN discriminator layers: " << discriminator->getLayers().size() << "\n";
        printModelInfo(*discriminator, "GAN Discriminator");
        std::cout << "  Usage: Classification real/fake" << std::endl;
    }

    // ========== Diffusion Model ==========
    {
        auto cfg = ModelArchitectures::defaultConfig("diffusion");
        auto diffusion = ModelArchitectures::create("diffusion", cfg);
        diffusion->setName("DDPM-32x32");
        diffusion->allocateParams();
        std::cout << "Diffusion layers: " << diffusion->getLayers().size() << "\n";
        printModelInfo(*diffusion, "Diffusion Model (DDPM)");
        std::cout << "  Usage: Génération d'images haute qualité" << std::endl;
    }

    // ========== Transformer ==========
    {
        auto cfg = ModelArchitectures::defaultConfig("transformer");
        auto transformer = ModelArchitectures::create("transformer", cfg);
        transformer->setName("GPT-Small");
        transformer->allocateParams();
        std::cout << "Transformer layers: " << transformer->getLayers().size() << "\n";
        printModelInfo(*transformer, "Transformer (GPT-Small)");
        std::cout << "  Usage: Génération de texte" << std::endl;
    }

    // ========== ResNet ==========
    {
        auto cfg = ModelArchitectures::defaultConfig("resnet");
        auto resnet = ModelArchitectures::create("resnet", cfg);
        resnet->setName("ResNet-50");
        resnet->allocateParams();
        std::cout << "ResNet layers: " << resnet->getLayers().size() << "\n";
        printModelInfo(*resnet, "ResNet-50");
        std::cout << "  Usage: Classification ImageNet" << std::endl;
    }

    // ========== MobileNet ==========
    {
        auto cfg = ModelArchitectures::defaultConfig("mobilenet");
        auto mobilenet = ModelArchitectures::create("mobilenet", cfg);
        mobilenet->setName("MobileNetV2");
        mobilenet->allocateParams();
        std::cout << "MobileNet layers: " << mobilenet->getLayers().size() << "\n";
        printModelInfo(*mobilenet, "MobileNetV2");
        std::cout << "  Usage: Classification mobile/embarqué" << std::endl;
    }

    // ========== Résumé ==========
    std::cout << "\n╔════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║              Architectures Disponibles         ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════╝" << std::endl
              << std::endl;

    std::cout << "  ✓ UNet          - Segmentation, U-shaped encoder-decoder" << std::endl;
    std::cout << "  ✓ ViT           - Vision Transformer, classification" << std::endl;
    std::cout << "  ✓ GAN           - Generative Adversarial Network" << std::endl;
    std::cout << "  ✓ Diffusion     - DDPM, génération haute qualité" << std::endl;
    std::cout << "  ✓ Transformer   - GPT-style, génération texte" << std::endl;
    std::cout << "  ✓ ResNet        - Deep residual networks" << std::endl;
    std::cout << "  ✓ MobileNet     - Efficient mobile architecture" << std::endl;

    std::cout << "\n💡 Toutes les architectures sont prêtes à l'emploi!" << std::endl;
    std::cout << "   Utilisez ModelArchitectures::create(name, cfg) pour construire." << std::endl;

    return 0;
}
