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
#include <cerrno>
#ifdef _OPENMP
#include <omp.h>
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;
using namespace ModelArchitectures;

static std::vector<std::string> splitString(const std::string& s, char delim)
{
    std::vector<std::string> parts;
    std::string cur;
    cur.reserve(s.size());
    for (char ch : s) {
        if (ch == delim) {
            parts.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(ch);
        }
    }
    parts.push_back(cur);
    return parts;
}

static json parseOverrideValue(const std::string& raw)
{
    if (raw == "true") return true;
    if (raw == "false") return false;
    if (raw == "null") return nullptr;

    if (!raw.empty() && (raw.front() == '{' || raw.front() == '[' || raw.front() == '"')) {
        try {
            return json::parse(raw);
        } catch (...) {
            // Fallback: traiter comme string si le JSON est invalide.
            return raw;
        }
    }

    {
        char* end = nullptr;
        errno = 0;
        long long v = std::strtoll(raw.c_str(), &end, 10);
        if (errno == 0 && end && *end == '\0') {
            return v;
        }
    }
    {
        char* end = nullptr;
        errno = 0;
        double v = std::strtod(raw.c_str(), &end);
        if (errno == 0 && end && *end == '\0') {
            return v;
        }
    }

    return raw;
}

static bool applyOverride(json& target, const std::string& expr, std::string& err)
{
    const auto eq = expr.find('=');
    if (eq == std::string::npos || eq == 0 || eq + 1 >= expr.size()) {
        err = "override invalide (attendu: path.to.key=value): " + expr;
        return false;
    }

    const std::string path = expr.substr(0, eq);
    const std::string raw_value = expr.substr(eq + 1);
    const auto keys = splitString(path, '.');
    if (keys.empty()) {
        err = "override invalide (path vide): " + expr;
        return false;
    }

    json* cur = &target;
    for (size_t i = 0; i + 1 < keys.size(); ++i) {
        const std::string& key = keys[i];
        if (key.empty()) {
            err = "override invalide (segment vide): " + expr;
            return false;
        }
        if (!cur->contains(key) || !(*cur)[key].is_object()) {
            (*cur)[key] = json::object();
        }
        cur = &(*cur)[key];
    }

    const std::string& leaf = keys.back();
    if (leaf.empty()) {
        err = "override invalide (clé finale vide): " + expr;
        return false;
    }

    (*cur)[leaf] = parseOverrideValue(raw_value);
    return true;
}

void printUsage(const char *prog)
{
    std::cout << "Usage: " << prog << " [OPTIONS]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --lua <script.lua>       Exécuter un script Lua\n";
    std::cout << "  --config <config.json>   Charger et entraîner depuis config\n";
    std::cout << "  --override <path=value>  Override (répétable) appliqué à la config du modèle\n";
    std::cout << "  --help                   Afficher cette aide\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << prog << " --lua scripts/test_lua_api.lua\n";
    std::cout << "  " << prog << " --config config.json\n";
    std::cout << "  " << prog << " --config config.json --override max_vocab=64000\n";
    std::cout << "  " << prog << " --config config.json --override optimizer=\"adamw\" --override weight_decay=0.01\n";
}

int main(int argc, char **argv)
{
    std::cout << "╔════════════════════════════════════════╗\n";
    std::cout << "║       Mímir Framework v2.4.0           ║\n";
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

    // Mode --lua: on passe tous les args après le script à Lua (ne pas valider ici).
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--help") {
            printUsage(argv[0]);
            return 0;
        }
        if (std::string(argv[i]) == "--lua" && i + 1 < argc) {
            const std::string lua_script = argv[++i];
            std::cout << "📜 Exécution du script Lua: " << lua_script << "\n";
            std::cout << "═══════════════════════════════════════════════\n\n";
            
            if (!fs::exists(lua_script)) {
                std::cerr << "❌ Fichier non trouvé: " << lua_script << "\n";
                return 1;
            }
            
            try {
                LuaScripting lua;
                std::vector<std::string> script_args;
                for (int j = i + 1; j < argc; ++j) {
                    script_args.emplace_back(argv[j]);
                }
                lua.setArgs(lua_script, script_args);
                lua.loadScript(lua_script);
                std::cout << "\n✅ Script Lua exécuté avec succès\n";
            } catch (const std::exception& e) {
                std::cerr << "❌ Erreur Lua: " << e.what() << "\n";
                return 1;
            }
            
            return 0;
        }
    }

    // Mode --config
    std::string config_path;
    std::vector<std::string> overrides;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (a == "--override" && i + 1 < argc) {
            overrides.emplace_back(argv[++i]);
        } else if (a == "--help") {
            // déjà géré plus haut
        } else if (a == "--lua") {
            // déjà géré plus haut
            ++i;
        } else {
            // En mode config on reste strict pour éviter les typos silencieuses.
            if (!config_path.empty()) {
                std::cerr << "❌ Option inconnue en mode --config: " << a << "\n";
                std::cerr << "💡 Utilisez --help pour la liste des options\n";
                return 1;
            }
        }
    }

    if (!config_path.empty()) {
        std::cout << "⚙️  Chargement de la configuration: " << config_path << "\n";

        if (!fs::exists(config_path)) {
            std::cerr << "❌ Fichier non trouvé: " << config_path << "\n";
            return 1;
        }

        std::ifstream f(config_path);
        json config;
        f >> config;

        std::string arch_type = config.value("architecture", "t2i_autoencoder");
        std::cout << "🏗️  Architecture: " << arch_type << "\n\n";

        // Construire le modèle selon la config
        std::shared_ptr<Model> model;

        // Construire via le registre: on part de la config par défaut et on applique les overrides.
        // Convention: la section peut être `config[arch_type]` (ex: "t2i_autoencoder": {...})
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

        if (!overrides.empty()) {
            std::cout << "🧩 Application des overrides (--override):\n";
            for (const auto& o : overrides) {
                std::string err;
                if (!applyOverride(cfg, o, err)) {
                    std::cerr << "❌ " << err << "\n";
                    return 1;
                }
                std::cout << "  • " << o << "\n";
            }
            std::cout << "\n";
        }

        model = ModelArchitectures::create(arch_name, cfg);

        model->allocateParams();
        model->initializeWeights("he");

        std::cout << "✅ Modèle créé avec " << model->totalParamCount() << " paramètres\n";
        std::cout << "💡 Entraînement à implémenter selon vos besoins\n";

        return 0;
    }
    
    std::cout << "❌ Option invalide. Utilisez --help\n";
    return 1;

}