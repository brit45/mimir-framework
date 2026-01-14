#include "Model.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
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
    std::cout << "  --lua <script.lua>       Exécuter un script Lua\n";
    std::cout << "  --config <config.json>   Charger et entraîner depuis config\n";
    std::cout << "  --help                   Afficher cette aide\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << prog << " --lua scripts/test_lua_api.lua\n";
    std::cout << "  " << prog << " --config config.json\n";
}

int main(int argc, char **argv)
{
    std::cout << "╔════════════════════════════════════════╗\n";
    std::cout << "║       Mímir Framework v2.3.0           ║\n";
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
            
            std::string arch_type = config.value("architecture", "ponyxl_ddpm");
            std::cout << "🏗️  Architecture: " << arch_type << "\n\n";
            
            // Construire le modèle selon la config
            std::shared_ptr<Model> model;

            // Construire via le registre: on part de la config par défaut et on applique les overrides.
            // Convention: la section peut être `config[arch_type]` (ex: "ponyxl_ddpm": {...})
            // ou `config["model"]`.
            std::string arch_name = arch_type;

            json cfg = ModelArchitectures::defaultConfig(arch_name);
            if (config.contains("model") && config["model"].is_object()) {
                for (auto& it : config["model"].items()) {
                    cfg[it.key()] = it.value();
                }
            }
            if (config.contains(arch_type) && config[arch_type].is_object()) {
                for (auto& it : config[arch_type].items()) {
                    cfg[it.key()] = it.value();
                }
            }

            model = ModelArchitectures::create(arch_name, cfg);
            
            model->allocateParams();
            model->initializeWeights("he");
            
            std::cout << "✅ Modèle créé avec " << model->totalParamCount() << " paramètres\n";
            std::cout << "💡 Entraînement à implémenter selon vos besoins\n";
            
            return 0;
        }
    }
    
    std::cout << "❌ Option invalide. Utilisez --help\n";
    return 1;

}
