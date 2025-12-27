#include "Model.hpp"
#include "Models/ModelArchitectures.hpp"
#include "LuaScripting.hpp"
#include "Helpers.hpp"
#include "HtopDisplay.hpp"
#include "Visualizer.hpp"
#include "MemorySafety.hpp"
#include "include/json.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <cstdlib>
#ifdef _OPENMP
#include <omp.h>
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;
using namespace ModelArchitectures;

void printUsage(const char *prog)
{
    std::cout << "Usage: " << prog << " [OPTIONS]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --demo <arch>            Démonstration d'architecture (unet, vae, vit, gan, diffusion, transformer, resnet, mobilenet)\n";
    std::cout << "  --lua <script.lua>       Exécuter un script Lua\n";
    std::cout << "  --config <config.json>   Charger et entraîner depuis config\n";
    std::cout << "  --help                   Afficher cette aide\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << prog << " --demo unet\n";
    std::cout << "  " << prog << " --demo transformer\n";
    std::cout << "  " << prog << " --lua scripts/test_lua_api.lua\n";
    std::cout << "  " << prog << " --config config.json\n";
}

void demoArchitecture(const std::string& arch_name) {
    std::cout << "\n╔════════════════════════════════════════════════╗\n";
    std::cout << "║      Démonstration: " << std::setw(24) << std::left << arch_name << " ║\n";
    std::cout << "╚════════════════════════════════════════════════╝\n\n";
    
    Model model;
    
    if (arch_name == "unet") {
        model.setName("UNet-Demo");
        UNetConfig config;
        config.input_channels = 3;
        config.output_channels = 1;
        config.base_channels = 64;
        config.num_levels = 4;
        config.use_attention = true;
        
        buildUNet(model, config);
        std::cout << "✓ UNet créé pour segmentation\n";
        std::cout << "  • Input: " << config.input_channels << " channels\n";
        std::cout << "  • Output: " << config.output_channels << " channels\n";
        std::cout << "  • Base filters: " << config.base_channels << "\n";
        std::cout << "  • Niveaux: " << config.num_levels << "\n";
        std::cout << "  • Attention: " << (config.use_attention ? "Oui" : "Non") << "\n";
        
    } else if (arch_name == "vae") {
        model.setName("VAE-Demo");
        VAEConfig config;
        config.input_dim = 784;
        config.latent_dim = 128;
        config.encoder_hidden = {512, 256};
        config.decoder_hidden = {256, 512};
        
        buildVAE(model, config);
        std::cout << "✓ VAE créé pour génération\n";
        std::cout << "  • Input dim: " << config.input_dim << "\n";
        std::cout << "  • Latent dim: " << config.latent_dim << "\n";
        std::cout << "  • Encoder: ";
        for (auto h : config.encoder_hidden) std::cout << h << " ";
        std::cout << "\n";
        
    } else if (arch_name == "vit") {
        model.setName("ViT-Demo");
        ViTConfig config;
        config.image_size = 224;
        config.patch_size = 16;
        config.num_classes = 1000;
        config.d_model = 768;
        config.num_heads = 12;
        config.num_layers = 12;
        
        buildViT(model, config);
        std::cout << "✓ Vision Transformer créé\n";
        std::cout << "  • Image: " << config.image_size << "x" << config.image_size << "\n";
        std::cout << "  • Patch size: " << config.patch_size << "\n";
        std::cout << "  • Classes: " << config.num_classes << "\n";
        std::cout << "  • Layers: " << config.num_layers << "\n";
        std::cout << "  • Heads: " << config.num_heads << "\n";
        
    } else if (arch_name == "gan") {
        Model generator, discriminator;
        generator.setName("GAN-Generator");
        discriminator.setName("GAN-Discriminator");
        
        GANConfig config;
        config.latent_dim = 100;
        config.image_size = 64;
        config.self_attention = true;
        
        buildGenerator(generator, config);
        buildDiscriminator(discriminator, config);
        
        std::cout << "✓ GAN créé (Generator + Discriminator)\n";
        std::cout << "  • Latent dim: " << config.latent_dim << "\n";
        std::cout << "  • Image size: " << config.image_size << "x" << config.image_size << "\n";
        std::cout << "  • Self-attention: " << (config.self_attention ? "Oui" : "Non") << "\n";
        
        generator.allocateParams();
        discriminator.allocateParams();
        
        std::cout << "\n📊 Generator:\n";
        std::cout << "  • Params: " << generator.totalParamCount() << "\n";
        std::cout << "  • Taille: " << std::fixed << std::setprecision(2)
                  << (generator.totalParamCount() * 4.0 / 1024.0 / 1024.0) << " MB\n";
        
        std::cout << "\n📊 Discriminator:\n";
        std::cout << "  • Params: " << discriminator.totalParamCount() << "\n";
        std::cout << "  • Taille: " << std::fixed << std::setprecision(2)
                  << (discriminator.totalParamCount() * 4.0 / 1024.0 / 1024.0) << " MB\n";
        
        return;
        
    } else if (arch_name == "diffusion") {
        model.setName("Diffusion-Demo");
        DiffusionConfig config;
        config.image_size = 32;
        config.base_channels = 128;
        config.num_res_blocks = 2;
        
        buildDiffusion(model, config);
        std::cout << "✓ Diffusion Model (DDPM) créé\n";
        std::cout << "  • Image: " << config.image_size << "x" << config.image_size << "\n";
        std::cout << "  • Base channels: " << config.base_channels << "\n";
        std::cout << "  • Res blocks: " << config.num_res_blocks << "\n";
        
    } else if (arch_name == "transformer") {
        model.setName("Transformer-Demo");
        TransformerConfig config;
        config.vocab_size = 50000;
        config.max_seq_len = 1024;
        config.d_model = 768;
        config.num_heads = 12;
        config.num_layers = 12;
        config.d_ff = 3072;
        config.causal = true;
        
        buildTransformer(model, config);
        std::cout << "✓ Transformer (GPT-style) créé\n";
        std::cout << "  • Vocab: " << config.vocab_size << "\n";
        std::cout << "  • Context: " << config.max_seq_len << " tokens\n";
        std::cout << "  • Model dim: " << config.d_model << "\n";
        std::cout << "  • Layers: " << config.num_layers << "\n";
        std::cout << "  • Heads: " << config.num_heads << "\n";
        std::cout << "  • Causal: " << (config.causal ? "Oui (GPT)" : "Non (BERT)") << "\n";
        
    } else if (arch_name == "resnet") {
        model.setName("ResNet-Demo");
        ResNetConfig config;
        config.num_classes = 1000;
        config.layers = {3, 4, 6, 3};
        config.use_bottleneck = true;
        
        buildResNet(model, config);
        std::cout << "✓ ResNet-50 créé\n";
        std::cout << "  • Classes: " << config.num_classes << "\n";
        std::cout << "  • Architecture: ";
        for (auto l : config.layers) std::cout << l << " ";
        std::cout << "\n";
        std::cout << "  • Bottleneck: " << (config.use_bottleneck ? "Oui" : "Non") << "\n";
        
    } else if (arch_name == "mobilenet") {
        model.setName("MobileNet-Demo");
        MobileNetConfig config;
        config.num_classes = 1000;
        config.width_multiplier = 1.0f;
        config.resolution = 224;
        
        buildMobileNetV2(model, config);
        std::cout << "✓ MobileNetV2 créé\n";
        std::cout << "  • Classes: " << config.num_classes << "\n";
        std::cout << "  • Width multiplier: " << config.width_multiplier << "x\n";
        std::cout << "  • Resolution: " << config.resolution << "\n";
        
    } else {
        std::cout << "❌ Architecture inconnue: " << arch_name << "\n";
        std::cout << "Disponibles: unet, vae, vit, gan, diffusion, transformer, resnet, mobilenet\n";
        return;
    }
    
    model.allocateParams();
    model.initializeWeights("he");
    
    std::cout << "\n📊 Statistiques:\n";
    std::cout << "  • Paramètres: " << model.totalParamCount() << "\n";
    std::cout << "  • Taille (FP32): " << std::fixed << std::setprecision(2)
              << (model.totalParamCount() * 4.0 / 1024.0 / 1024.0) << " MB\n";
    std::cout << "  • Taille (FP16): " << std::fixed << std::setprecision(2)
              << (model.totalParamCount() * 2.0 / 1024.0 / 1024.0) << " MB\n";
    
    std::cout << "\n✅ Modèle prêt à l'emploi!\n";
    std::cout << "🚀 Optimisations hardware activées (AVX2, FMA, OpenMP)\n";
}

int main(int argc, char **argv)
{
    std::cout << "╔════════════════════════════════════════╗\n";
    std::cout << "║       Mímir Framework v2.0             ║\n";
    std::cout << "║     Deep Learning Architectures        ║\n";
    std::cout << "╚════════════════════════════════════════╝\n\n";
    
    // 🛡️ SÉCURITÉ MÉMOIRE: Vérification au démarrage
    std::cout << "🛡️  Vérification de la sécurité mémoire...\n";
    MemorySafety::validateLegacyDisabled();
    MemorySafety::runMemoryIntegrityTest();
    std::cout << "\n";
    
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    std::cout << "🔧 OpenMP: " << num_threads << " threads disponibles\n";
    omp_set_num_threads(num_threads);
#endif
    
    std::cout << "🚀 Optimisations hardware:\n";
    std::cout << "  • AVX2: " << (Model::hasAVX2() ? "✓" : "✗") << "\n";
    std::cout << "  • FMA: " << (Model::hasFMA() ? "✓" : "✗") << "\n";
    std::cout << "  • F16C: " << (Model::hasF16C() ? "✓" : "✗") << "\n";
    std::cout << "  • BMI2: " << (Model::hasBMI2() ? "✓" : "✗") << "\n";
    std::cout << "\n";
    
    if (argc < 2) {
        std::cout << "💡 Utilisez --help pour voir les options\n";
        std::cout << "💡 Essayez: " << argv[0] << " --demo unet\n";
        std::cout << "💡 Ou:      " << argv[0] << " --lua scripts/test_lua_api.lua\n\n";
        return 0;
    }
    
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--help") {
            printUsage(argv[0]);
            return 0;
        }
        else if (std::string(argv[i]) == "--lua" && i + 1 < argc) {
            std::string lua_script = argv[++i];
            std::cout << "📜 Exécution du script Lua: " << lua_script << "\n";
            std::cout << "═══════════════════════════════════════════════\n\n";
            
            if (!fs::exists(lua_script)) {
                std::cerr << "❌ Fichier non trouvé: " << lua_script << "\n";
                return 1;
            }
            
            try {
                LuaScripting lua;
                lua.loadScript(lua_script);
                std::cout << "\n✅ Script Lua exécuté avec succès\n";
            } catch (const std::exception& e) {
                std::cerr << "❌ Erreur Lua: " << e.what() << "\n";
                return 1;
            }
            
            return 0;
        }
        else if (std::string(argv[i]) == "--demo" && i + 1 < argc) {
            demoArchitecture(argv[++i]);
            return 0;
        }
        else if (std::string(argv[i]) == "--config" && i + 1 < argc) {
            std::string config_path = argv[++i];
            std::cout << "⚙️  Chargement de la configuration: " << config_path << "\n";
            
            if (!fs::exists(config_path)) {
                std::cerr << "❌ Fichier non trouvé: " << config_path << "\n";
                return 1;
            }
            
            std::ifstream f(config_path);
            json config;
            f >> config;
            
            std::string arch_type = config.value("architecture", "unet");
            std::cout << "🏗️  Architecture: " << arch_type << "\n\n";
            
            // Construire le modèle selon la config
            Model model;
            
            if (arch_type == "unet") {
                UNetConfig unet_config;
                if (config.contains("unet")) {
                    auto& u = config["unet"];
                    unet_config.input_channels = u.value("input_channels", 3);
                    unet_config.output_channels = u.value("output_channels", 1);
                    unet_config.base_channels = u.value("base_channels", 64);
                    unet_config.num_levels = u.value("num_levels", 4);
                }
                buildUNet(model, unet_config);
            }
            else if (arch_type == "transformer") {
                TransformerConfig t_config;
                if (config.contains("transformer")) {
                    auto& t = config["transformer"];
                    t_config.vocab_size = t.value("vocab_size", 50000);
                    t_config.max_seq_len = t.value("max_seq_len", 1024);
                    t_config.d_model = t.value("d_model", 768);
                    t_config.num_heads = t.value("num_heads", 12);
                    t_config.num_layers = t.value("num_layers", 12);
                }
                buildTransformer(model, t_config);
            }
            // Ajouter d'autres architectures au besoin
            
            model.allocateParams();
            model.initializeWeights("he");
            
            std::cout << "✅ Modèle créé avec " << model.totalParamCount() << " paramètres\n";
            std::cout << "💡 Entraînement à implémenter selon vos besoins\n";
            
            return 0;
        }
    }
    
    std::cout << "❌ Option invalide. Utilisez --help\n";
    return 1;
}
