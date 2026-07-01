#include "Model.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "scriptings/Lua/luaScripting/LuaScripting.hpp"
#include "scriptings/Rust/rustScripting/RustScripting.hpp"
#include "scriptings/CSharp/csharpScripting/CSharpScripting.hpp"
#include "scriptings/JavaScript/jsScripting/JSScripting.hpp"
#include "AsyncMonitor.hpp"
#include "Helpers.hpp"
#include "HtopDisplay.hpp"
#include "Visualizer.hpp"
#include "MemorySafety.hpp"
#include "Tokenizer.hpp"
#include "Encoder.hpp"
#include "ConfigOverrides.hpp"
#include "runtimes/AbstractRuntime.hpp"
#include "Serialization/Serialization.hpp"
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
#include <cerrno>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef ENABLE_ROCM
#include <hip/hip_runtime.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;
using namespace ModelArchitectures;

// Spill disque: dossier attendu par le système en cas d'éviction/MemoryGuard.
static std::string g_mimir_spill_dir = ".mimir-spill";

static std::string cudaAccelStatus() {
#ifndef ENABLE_CUDA
    return "✗";
#else
    const RuntimeConfig cfg = RuntimeConfig::fromEnv("CUDA");
    if (cfg.disabled) {
        return "✗ (désactivé)";
    }

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        return "✗ (aucun device)";
    }

    int device_index = cfg.device_index;
    if (device_index < 0 || device_index >= device_count) {
        device_index = 0;
    }

    cudaDeviceProp prop;
    const bool have_props = (cudaGetDeviceProperties(&prop, device_index) == cudaSuccess);

    // Ici, on reflète l'intention d'usage: le backend existe, mais n'est "utilisé" que si opt-in.
    if (!cfg.linear_enabled) {
        return have_props ? ("— (device " + std::to_string(device_index) + ": " + prop.name + ")") : "— (device)";
    }

    return have_props ? ("✓ (device " + std::to_string(device_index) + ": " + prop.name + ")") : "✓";
#endif
}

static std::string rocmAccelStatus() {
#ifndef ENABLE_ROCM
    return "✗";
#else
    const RuntimeConfig cfg = RuntimeConfig::fromEnv("ROCM");
    if (cfg.disabled) {
        return "✗ (désactivé)";
    }

    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count <= 0) {
        return "✗ (aucun device)";
    }

    int device_index = cfg.device_index;
    if (device_index < 0 || device_index >= device_count) {
        device_index = 0;
    }

    hipDeviceProp_t prop;
    const bool have_props = (hipGetDeviceProperties(&prop, device_index) == hipSuccess);

    if (!cfg.linear_enabled) {
        return have_props ? ("— (device " + std::to_string(device_index) + ": " + prop.name + ")") : "— (device)";
    }

    return have_props ? ("✓ (device " + std::to_string(device_index) + ": " + prop.name + ")") : "✓";
#endif
}

static void clearSpillDirContentsNoThrow() noexcept {
    try {
        const fs::path dir(g_mimir_spill_dir);

        // Garde-fou: ne nettoyer QUE le dossier attendu (évite toute suppression accidentelle).
        if (dir.empty() || dir.filename() != ".mimir-spill") {
            return;
        }

        std::error_code ec;
        if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) {
            return;
        }

        for (const auto& entry : fs::directory_iterator(dir, ec)) {
            if (ec) break;
            std::error_code rm_ec;
            fs::remove_all(entry.path(), rm_ec);
        }
    } catch (...) {
    }
}

static bool applyOverride(json& target, const std::string& expr, std::string& err)
{
    return Mimir::ConfigOverrides::applyOverride(target, expr, err);
}

static bool readJsonFile(const std::string& path, json& out, std::string& err)
{
    std::ifstream f(path);
    if (!f) {
        err = "Impossible d'ouvrir le fichier: " + path;
        return false;
    }
    try {
        f >> out;
    } catch (const std::exception& e) {
        err = std::string("Erreur JSON: ") + e.what();
        return false;
    }
    return true;
}

void printUsage(const char *prog)
{
    std::cout << "Usage: " << prog << " [OPTIONS]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --lua <script.lua>       Exécuter un script Lua\n";
    std::cout << "  --js <script.js>         Exécuter un script JavaScript (Node.js)\n";
    std::cout << "  --csharp <script.csx>    Exécuter un script C# (dotnet-script/csi)\n";
    std::cout << "  --rust <script.rs>       Exécuter un script Rust (rust-script)\n";
    std::cout << "  --config <config.json>   Charger et entraîner depuis config\n";
    std::cout << "  --conf <config.json>     Charger une conf et exécuter lua.scripts\n";
    std::cout << "  --override <path=value>  Override (répétable) appliqué à la config du modèle\n";
    std::cout << "  --help                   Afficher cette aide\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << prog << " --lua scripts/test_lua_api.lua\n";
    std::cout << "  " << prog << " --js scripts/examples/example.js\n";
    std::cout << "  " << prog << " --csharp scripts/examples/example.csx\n";
    std::cout << "  " << prog << " --rust scripts/examples/example.rs\n";
    std::cout << "  " << prog << " --config config.json\n";
    std::cout << "  " << prog << " --conf config.json\n";
    std::cout << "  " << prog << " --config config.json --override max_vocab=64000\n";
    std::cout << "  " << prog << " --config config.json --override optimizer=\"adamw\" --override weight_decay=0.01\n";
}

int main(int argc, char **argv)
{
    {
        const std::string ver = Mimir::Serialization::get_mimir_version();
        // Largeur interne de la boîte = 40 chars affichage.
        // Préfixe "       Mímir Framework v" = 24 chars affichage (ASCII sauf "í" qui occupe 2 octets
        // mais 1 char affichage — on calcule le padding sur la taille d'affichage).
        const int trailing = std::max(0, 40 - 24 - static_cast<int>(ver.size()));
        std::cerr << "╔════════════════════════════════════════╗\n";
        std::cerr << "║       Mímir Framework v" << ver << std::string(trailing, ' ') << "║\n";
        std::cerr << "║     Deep Learning Architectures        ║\n";
        std::cerr << "╚════════════════════════════════════════╝\n\n";
    }

    // Préparer le spill dir + nettoyage fin de run.
    {
        std::error_code ec;
        fs::create_directories(g_mimir_spill_dir, ec);
        // Nettoyer aussi au démarrage (résidus d'un run précédent).
        clearSpillDirContentsNoThrow();
        std::atexit(clearSpillDirContentsNoThrow);
    }
    
    // 🛡️ SÉCURITÉ MÉMOIRE: Vérification au démarrage
    std::cerr << "🛡️  Vérification de la sécurité mémoire...\n";
    MemorySafety::validateLegacyDisabled();
    MemorySafety::runMemoryIntegrityTest();
    std::cerr << "\n";
    
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    std::cerr << "🔧 OpenMP: " << num_threads << " threads disponibles\n";
    omp_set_num_threads(num_threads);
#endif
    
    std::cerr << "🚀 Optimisations hardware:\n";
    std::cerr << "  • AVX2: " << (Model::hasAVX2() ? "✓" : "✗") << "\n";
    std::cerr << "  • FMA: " << (Model::hasFMA() ? "✓" : "✗") << "\n";
    std::cerr << "  • F16C: " << (Model::hasF16C() ? "✓" : "✗") << "\n";
    std::cerr << "  • BMI2: " << (Model::hasBMI2() ? "✓" : "✗") << "\n";
    std::cerr << "  • CUDA: " << cudaAccelStatus() << "\n";
    std::cerr << "  • ROCM: " << rocmAccelStatus() << "\n";
    std::cerr << "\n";
    
    if (argc < 2) {
        std::cerr << "💡 Utilisez --help pour voir les options\n";
        std::cerr << "💡 Ou:      " << argv[0] << " --lua scripts/test_lua_api.lua\n\n";
        return 0;
    }

    // Modes scripting directs: on passe tous les args après le script au runtime.
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--help") {
            printUsage(argv[0]);
            return 0;
        }
        const std::string opt = argv[i];
        if ((opt == "--lua" || opt == "--js" || opt == "--csharp" || opt == "--rust") && i + 1 < argc) {
            const std::string script_path = argv[++i];
            const std::string lang =
                (opt == "--lua") ? "Lua" :
                (opt == "--js") ? "JavaScript" :
                (opt == "--csharp") ? "C#" : "Rust";

            std::cerr << "📜 Exécution du script " << lang << ": " << script_path << "\n";
            std::cerr << "═══════════════════════════════════════════════\n\n";
            
            if (!fs::exists(script_path)) {
                std::cerr << "❌ Fichier non trouvé: " << script_path << "\n";
                return 1;
            }
            
            try {
                std::vector<std::string> script_args;
                for (int j = i + 1; j < argc; ++j) {
                    script_args.emplace_back(argv[j]);
                }

                if (opt == "--lua") {
                    LuaScripting lua;
                    lua.setArgs(script_path, script_args);
                    if (!lua.loadScript(script_path)) {
                        std::cerr << "❌ Échec exécution Lua: " << script_path << "\n";
                        return 1;
                    }

                    // UX: en ponyxl_ddpm, garder la Viz ouverte après la fin du script.
                    {
                        auto& ctx = LuaContext::getInstance();
                        if (ctx.modelType == "ponyxl_ddpm" && ctx.asyncMonitor && ctx.asyncMonitor->getViz() && ctx.asyncMonitor->getViz()->isOpen()) {
                            std::cerr << "\n🖼️  Viz ouverte — fermeture manuelle pour quitter...\n";
                            ctx.asyncMonitor->waitForVizClose();
                        }
                    }

                    {
                        auto& ctx = LuaContext::getInstance();
                        ctx.resetRuntimeState();
                    }
                } else if (opt == "--js") {
                    JSScripting js;
                    js.registerAPI();
                    js.setArgs(script_path, script_args);
                    if (!js.loadScript(script_path)) {
                        std::cerr << "❌ Échec exécution JS: " << script_path << "\n";
                        return 1;
                    }
                    JSContext::getInstance().resetRuntimeState();
                } else if (opt == "--csharp") {
                    CSharpScripting cs;
                    cs.registerAPI();
                    cs.setArgs(script_path, script_args);
                    if (!cs.loadScript(script_path)) {
                        std::cerr << "❌ Échec exécution C#: " << script_path << "\n";
                        return 1;
                    }
                    CSharpContext::getInstance().resetRuntimeState();
                } else if (opt == "--rust") {
                    RustScripting rs;
                    rs.registerAPI();
                    rs.setArgs(script_path, script_args);
                    if (!rs.loadScript(script_path)) {
                        std::cerr << "❌ Échec exécution Rust: " << script_path << "\n";
                        return 1;
                    }
                    RustContext::getInstance().resetRuntimeState();
                }

                std::cerr << "\n✅ Script " << lang << " exécuté avec succès\n";
            } catch (const std::exception& e) {
                std::cerr << "❌ Erreur script " << lang << ": " << e.what() << "\n";
                return 1;
            }
            
            return 0;
        }
    }

    // Mode --config
    std::string config_path;
    std::string conf_path;
    std::vector<std::string> overrides;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (a == "--conf" && i + 1 < argc) {
            conf_path = argv[++i];
        } else if (a == "--override" && i + 1 < argc) {
            overrides.emplace_back(argv[++i]);
        } else if (a == "--help") {
            // déjà géré plus haut
        } else if (a == "--lua" || a == "--js" || a == "--csharp" || a == "--rust") {
            // déjà géré plus haut
            ++i;
        } else {
            // En mode config on reste strict pour éviter les typos silencieuses.
            if (!config_path.empty()) {
                std::cerr << "❌ Option inconnue en mode --config: " << a << "\n";
                std::cerr << "💡 Utilisez --help pour la liste des options\n";
                return 1;
            }
            if (!conf_path.empty()) {
                std::cerr << "❌ Option inconnue en mode --conf: " << a << "\n";
                std::cerr << "💡 Utilisez --help pour la liste des options\n";
                return 1;
            }
        }
    }

    if (!conf_path.empty()) {
        std::cerr << "⚙️  Chargement de la conf: " << conf_path << "\n";

        if (!fs::exists(conf_path)) {
            std::cerr << "❌ Fichier non trouvé: " << conf_path << "\n";
            return 1;
        }

        json conf;
        {
            std::string err;
            if (!readJsonFile(conf_path, conf, err)) {
                std::cerr << "❌ " << err << "\n";
                return 1;
            }
        }

        if (!overrides.empty()) {
            std::cerr << "🧩 Application des overrides (--override) sur la conf:\n";
            for (const auto& o : overrides) {
                std::string err;
                if (!applyOverride(conf, o, err)) {
                    std::cerr << "❌ " << err << "\n";
                    return 1;
                }
                std::cerr << "  • " << o << "\n";
            }
            std::cerr << "\n";
        }

        const json* lua_conf = nullptr;
        if (conf.contains("lua") && conf["lua"].is_object()) {
            lua_conf = &conf["lua"];
        } else if (conf.contains("run") && conf["run"].is_object() && conf["run"].contains("lua") && conf["run"]["lua"].is_object()) {
            lua_conf = &conf["run"]["lua"];
        }

        if (!lua_conf) {
            std::cerr << "❌ --conf: aucune section lua trouvée (attendu: lua.scripts ou run.lua.scripts)\n";
            return 1;
        }

        std::vector<json> scripts;
        if (lua_conf->contains("scripts") && (*lua_conf)["scripts"].is_array()) {
            for (const auto& it : (*lua_conf)["scripts"]) scripts.push_back(it);
        } else if (lua_conf->contains("script")) {
            scripts.push_back((*lua_conf)["script"]);
        }

        if (scripts.empty()) {
            std::cerr << "❌ --conf: aucune entrée de script (attendu: lua.scripts=[...] ou lua.script)\n";
            return 1;
        }

        const std::string conf_abs = fs::absolute(conf_path).string();
        const std::string conf_dir = fs::absolute(fs::path(conf_path)).parent_path().string();

        for (size_t si = 0; si < scripts.size(); ++si) {
            const json& s = scripts[si];

            std::string script_path;
            std::vector<std::string> script_args;

            if (s.is_string()) {
                script_path = s.get<std::string>();
            } else if (s.is_object()) {
                script_path = s.value("script", "");
                if (s.contains("args") && s["args"].is_array()) {
                    for (const auto& a : s["args"]) {
                        if (a.is_string()) script_args.push_back(a.get<std::string>());
                        else script_args.push_back(a.dump());
                    }
                }
            } else {
                std::cerr << "❌ --conf: script invalide (string ou objet attendu)\n";
                return 1;
            }

            if (script_path.empty()) {
                std::cerr << "❌ --conf: script vide\n";
                return 1;
            }
            if (!fs::exists(script_path)) {
                std::cerr << "❌ Script Lua non trouvé: " << script_path << "\n";
                return 1;
            }

            std::cerr << "📜 [conf] Script Lua (" << (si + 1) << "/" << scripts.size() << "): " << script_path << "\n";

            try {
                LuaScripting lua;
                lua.setArgs(script_path, script_args);
                lua.setSystemConfig(conf, conf_abs, conf_dir);

                if (!lua.loadScript(script_path)) {
                    std::cerr << "❌ Échec exécution Lua: " << script_path << "\n";
                    return 1;
                }
            } catch (const std::exception& e) {
                std::cerr << "❌ Erreur Lua: " << e.what() << "\n";
                return 1;
            }
        }

        // UX: si un script a ouvert la Viz (ponyxl_ddpm), laisser la fenêtre active.
        {
            auto& ctx = LuaContext::getInstance();
            if (ctx.modelType == "ponyxl_ddpm" && ctx.asyncMonitor && ctx.asyncMonitor->getViz() && ctx.asyncMonitor->getViz()->isOpen()) {
                std::cerr << "\n🖼️  Viz ouverte — fermeture manuelle pour quitter...\n";
                ctx.asyncMonitor->waitForVizClose();
            }
        }

        // IMPORTANT: libérer explicitement les ressources LuaContext avant exit.
        {
            auto& ctx = LuaContext::getInstance();
            ctx.resetRuntimeState();
        }

        std::cerr << "\n✅ --conf: scripts exécutés avec succès\n";
        return 0;
    }

    if (!config_path.empty()) {
        std::cerr << "⚙️  Chargement de la configuration: " << config_path << "\n";

        if (!fs::exists(config_path)) {
            std::cerr << "❌ Fichier non trouvé: " << config_path << "\n";
            return 1;
        }

        json config;
        {
            std::string err;
            if (!readJsonFile(config_path, config, err)) {
                std::cerr << "❌ " << err << "\n";
                return 1;
            }
        }

        if (!overrides.empty()) {
            std::cerr << "🧩 Application des overrides (--override) sur la config:\n";
            for (const auto& o : overrides) {
                std::string err;
                if (!applyOverride(config, o, err)) {
                    std::cerr << "❌ " << err << "\n";
                    return 1;
                }
                std::cerr << "  • " << o << "\n";
            }
            std::cerr << "\n";
        }

        std::string arch_name;
        json cfg = ModelArchitectures::cfgFromConfig(config, &arch_name);
        std::cerr << "🏗️  Architecture: " << arch_name << "\n\n";

        // Construire le modèle selon la config résolue.
        std::shared_ptr<Model> model = ModelArchitectures::create(arch_name, cfg);

        // Brancher les composants framework (tokenizer/encoder) si présents dans la config.
        // (Ils restent aussi accessibles dans model->modelConfig via cfgFromConfig.)
        try {
            if (config.contains("tokenizer") && config["tokenizer"].is_object()) {
                const json& tj = config["tokenizer"];
                const int max_vocab = tj.value("max_vocab", 4096);
                auto tok = std::make_shared<Tokenizer>(static_cast<size_t>(std::max(1, max_vocab)));
                if (tj.contains("max_sequence_length") && tj["max_sequence_length"].is_number_integer()) {
                    tok->setMaxSequenceLength(tj["max_sequence_length"].get<int>());
                }
                model->setTokenizer(*tok);
            }

            if (config.contains("encoder") && config["encoder"].is_object()) {
                const json& ej = config["encoder"];
                const int dim = ej.value("embedding_dim", 64);
                int vocab_size = 4096;
                if (config.contains("tokenizer") && config["tokenizer"].is_object()) {
                    vocab_size = config["tokenizer"].value("max_vocab", vocab_size);
                }
                auto enc = std::make_shared<ConditioningEncoder>(dim, std::max(1, vocab_size));
                // Optionnel: seed
                uint64_t seed = 0;
                if (config.contains("inference") && config["inference"].is_object()) {
                    seed = static_cast<uint64_t>(config["inference"].value("seed", 0));
                }
                enc->initRandom(seed);
                model->setEncoder(*enc);
            }
        } catch (...) {
            // Best-effort: ne pas bloquer la création du modèle si config partielle.
        }

        model->allocateParams();
        model->initializeWeights("he");

        std::cerr << "✅ Modèle créé avec " << model->totalParamCount() << " paramètres\n";
        std::cerr << "💡 Entraînement à implémenter selon vos besoins\n";

        return 0;
    }
    
    std::cout << "❌ Option invalide. Utilisez --help\n";
    return 1;

}