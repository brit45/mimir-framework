#include "LuaScripting.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "Serialization/Serialization.hpp"
#include "Serialization/DebugJsonDump.hpp"
#include "DType.hpp"
#include "AdvancedRAMManager.hpp"
#include "MemoryGuard.hpp"
#include "DynamicTensorAllocator.hpp"
#include "AsyncMonitor.hpp"
#include "VizTextPayload.hpp"
#include "Helpers.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <numeric>
#include <unordered_set>
#include <type_traits>
#include <utility>

namespace {

static bool _mimir_ends_with(const std::string& value, const std::string& suffix) {
    if (value.size() < suffix.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin());
}

static bool _mimir_load_mpk_full_config(lua_State* L,
                                        const std::string& mpk_path,
                                        std::string* out_err) {
    const int stack_top = lua_gettop(L);
    auto cleanup = [&]() { lua_settop(L, stack_top); };

    lua_getglobal(L, "dofile");
    if (!lua_isfunction(L, -1)) {
        cleanup();
        if (out_err) *out_err = "dofile unavailable";
        return false;
    }

    lua_pushstring(L, "scripts/modules/mpk.lua");
    if (lua_pcall(L, 1, 1, 0) != LUA_OK) {
        if (out_err) *out_err = lua_tostring(L, -1);
        lua_pop(L, 1);
        cleanup();
        return false;
    }

    if (!lua_istable(L, -1)) {
        if (out_err) *out_err = "MPK module did not return a table";
        cleanup();
        return false;
    }

    const int mpk_mod = lua_gettop(L);

    lua_getfield(L, mpk_mod, "read");
    if (!lua_isfunction(L, -1)) {
        if (out_err) *out_err = "MPK.read unavailable";
        cleanup();
        return false;
    }
    lua_pushstring(L, mpk_path.c_str());
    if (lua_pcall(L, 1, 1, 0) != LUA_OK) {
        if (out_err) *out_err = lua_tostring(L, -1);
        lua_pop(L, 1);
        cleanup();
        return false;
    }
    if (!lua_istable(L, -1)) {
        if (out_err) *out_err = "MPK.read did not return a package table";
        cleanup();
        return false;
    }
    const int pkg_idx = lua_gettop(L);

    lua_getfield(L, mpk_mod, "to_registry_full_config");
    if (!lua_isfunction(L, -1)) {
        if (out_err) *out_err = "MPK.to_registry_full_config unavailable";
        cleanup();
        return false;
    }
    lua_pushvalue(L, pkg_idx);
    if (lua_pcall(L, 1, 1, 0) != LUA_OK) {
        if (out_err) *out_err = lua_tostring(L, -1);
        lua_pop(L, 1);
        cleanup();
        return false;
    }
    if (!lua_istable(L, -1)) {
        if (out_err) *out_err = "MPK.to_registry_full_config did not return a table";
        cleanup();
        return false;
    }

    lua_replace(L, stack_top + 1);
    lua_settop(L, stack_top + 1);
    return true;
}

}  // namespace

std::size_t LuaScripting::autoRegisterMpkArchitectures(
    const std::string& project_root,
    std::vector<std::string>* warnings) {
    const std::filesystem::path arch_dir =
        std::filesystem::path(project_root) / "_archi";
    std::error_code ec;
    if (!std::filesystem::is_directory(arch_dir, ec) || ec) {
        return 0;
    }

    std::vector<std::filesystem::path> files;
    for (std::filesystem::directory_iterator it(arch_dir, ec), end;
         !ec && it != end;
         it.increment(ec)) {
        if (!it->is_regular_file(ec) || ec) continue;
        std::string filename = it->path().filename().string();
        std::transform(filename.begin(), filename.end(), filename.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (_mimir_ends_with(filename, ".mpk") ||
            _mimir_ends_with(filename, ".mpk.bin")) {
            files.push_back(it->path());
        }
    }
    if (ec) {
        if (warnings) warnings->push_back("lecture de _archi impossible: " + ec.message());
        return 0;
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) return 0;

    lua_State* loader = luaL_newstate();
    if (!loader) {
        if (warnings) warnings->push_back("initialisation Lua impossible pour charger _archi");
        return 0;
    }
    luaL_openlibs(loader);

    std::size_t loaded = 0;
    auto& registry = ModelArchitectures::Registry::instance();
    for (const auto& path : files) {
        const std::string filename = path.filename().string();
        std::string file_alias = filename;
        std::string file_alias_lower = file_alias;
        std::transform(file_alias_lower.begin(), file_alias_lower.end(), file_alias_lower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        const std::size_t suffix_size =
            _mimir_ends_with(file_alias_lower, ".mpk.bin") ? 8u : 4u;
        file_alias.resize(file_alias.size() - suffix_size);

        std::string err;
        if (!_mimir_load_mpk_full_config(loader, path.string(), &err)) {
            if (warnings) warnings->push_back(path.filename().string() + ": " + err);
            continue;
        }

        json full_cfg;
        try {
            full_cfg = LuaScripting::luaTableToJson(loader, -1);
        } catch (const std::exception& e) {
            lua_pop(loader, 1);
            if (warnings) warnings->push_back(path.filename().string() + ": " + e.what());
            continue;
        }
        lua_pop(loader, 1);

        const std::string base_arch = full_cfg.value("architecture", std::string());
        const json model_cfg = full_cfg.value("model", json::object());
        const json mpk_meta = full_cfg.value("mpk", json::object());
        std::string registry_name = mpk_meta.value("name", file_alias);
        const std::string description =
            mpk_meta.value("description", std::string("Architecture MPK: ") + path.filename().string());

        if (registry_name.empty() || base_arch.empty() || !model_cfg.is_object()) {
            if (warnings) warnings->push_back(path.filename().string() + ": métadonnées de registre incomplètes");
            continue;
        }
        if (registry.find(base_arch) == nullptr) {
            if (warnings) warnings->push_back(
                path.filename().string() + ": architecture de base inconnue: " + base_arch);
            continue;
        }

        // Ne jamais remplacer silencieusement une architecture native. Un MPK
        // portant le même nom devient accessible sous le nom de son fichier.
        if (registry.find(registry_name) != nullptr) {
            registry_name = file_alias;
        }
        if (registry_name.empty() || registry.find(registry_name) != nullptr) {
            if (warnings) warnings->push_back(
                path.filename().string() + ": nom déjà présent dans le registre");
            continue;
        }

        ModelArchitectures::Entry entry;
        entry.name = registry_name;
        entry.description = description + " [MPK: " + path.string() + "]";
        entry.default_config = model_cfg;
        entry.create = [base_arch](const json& cfg) {
            return ModelArchitectures::create(base_arch, cfg);
        };
        entry.origin = "mpk";
        entry.source_path = path.string();
        registry.registerArchitecture(std::move(entry));
        ++loaded;
    }

    lua_close(loader);
    return loaded;
}

template <typename T, typename = void>
struct _mimir_has_overrides_enabled : std::false_type {};

template <typename T>
struct _mimir_has_overrides_enabled<T, std::void_t<decltype(std::declval<T>().overrides_enabled)>>
    : std::true_type {};

template <typename T>
bool _mimir_live_params_overrides_enabled(const T& p) {
    if constexpr (_mimir_has_overrides_enabled<T>::value) {
        return static_cast<bool>(p.overrides_enabled);
    }
    return true;
}

static std::vector<float> latentChunkToEmbedding(const std::vector<float>& packed,
                                                 int image_dim,
                                                 int latent_dim,
                                                 int target_dim) {
    if (target_dim <= 0 || image_dim < 0 || latent_dim <= 0) return {};
    const size_t off = static_cast<size_t>(image_dim);
    if (packed.size() < off + static_cast<size_t>(latent_dim)) return {};

    std::vector<float> out(static_cast<size_t>(target_dim), 0.0f);
    for (int d = 0; d < target_dim; ++d) {
        const int lo = (d * latent_dim) / target_dim;
        int hi = ((d + 1) * latent_dim) / target_dim;
        if (hi <= lo) hi = lo + 1;
        const int hi_clamped = std::min(latent_dim, hi);
        float acc = 0.0f;
        int cnt = 0;
        for (int i = lo; i < hi_clamped; ++i) {
            acc += packed[off + static_cast<size_t>(i)];
            ++cnt;
        }
        out[static_cast<size_t>(d)] = (cnt > 0) ? (acc / static_cast<float>(cnt)) : 0.0f;
    }

    double n2 = 0.0;
    for (float v : out) n2 += static_cast<double>(v) * static_cast<double>(v);
    if (n2 > 1e-12) {
        const float inv = static_cast<float>(1.0 / std::sqrt(n2));
        for (float& v : out) v *= inv;
    }
    return out;
}

static void mergeLuaConfigIntoModelConfig(Model& model, const json& cfg) {
    if (!cfg.is_object()) return;

    static const std::unordered_set<std::string> kOverrideKeys = {
        "recon_loss", "kl_beta", "vae_kl_beta", "kl_warmup_steps",
        "marker_wass_scale", "marker_temp_scale", "marker_scale_max", "marker_warmup_steps",
        "logvar_clip_min", "logvar_clip_max", "ssim_weight", "ssim_mode", "ssim_k1", "ssim_k2",
        "ssim_L", "spectral_weight", "spectral_scales", "perceptual_weight", "perceptual_arch",
        "perceptual_checkpoint", "perceptual_base_channels", "perceptual_prior_momentum",
        "perceptual_prior_scale", "huber_delta", "smoothl1_delta",
        "smoothl1_beta", "charbonnier_eps", "nll_sigma", "gaussian_nll_sigma", "align_weight",
        "grad_clip_norm", "clip_norm", "frozen_layer_prefixes",
    };

    for (auto it = cfg.begin(); it != cfg.end(); ++it) {
        const std::string key = it.key();
        const json& value = it.value();
        if (kOverrideKeys.count(key) > 0) {
            model.modelConfig[key] = value;
            continue;
        }
        if (!model.modelConfig.contains(key)) {
            model.modelConfig[key] = value;
        }
    }
}

int LuaScripting::lua_createModel(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    // Argument: type de modèle (string)
    const char* model_type = luaL_checkstring(L, 1);
    const std::string model_type_str = model_type ? std::string(model_type) : std::string();

    if (_mimir_ends_with(model_type_str, ".mpk") ||
        _mimir_ends_with(model_type_str, ".mpk.bin")) {
        const std::filesystem::path mpk_path(model_type_str);
        std::error_code ec;
        if (std::filesystem::exists(mpk_path, ec) && !ec) {
            std::string err;
            if (!_mimir_load_mpk_full_config(L, model_type_str, &err)) {
                ctx.addLog("Erreur chargement MPK pour create(path): " + err);
                lua_pushboolean(L, false);
                lua_pushstring(L, err.c_str());
                return 2;
            }

            json full_cfg = LuaScripting::luaTableToJson(L, -1);
            lua_pop(L, 1);

            try {
                std::string arch;
                json cfg;
                ctx.currentModel = ModelArchitectures::createFromConfig(full_cfg, &cfg, &arch);

                if (ctx.currentModel) {
                    mergeLuaConfigIntoModelConfig(*ctx.currentModel, cfg);
                }
                if (ctx.currentTokenizer) {
                    ctx.currentModel->setTokenizer(*ctx.currentTokenizer);
                }
                if (ctx.currentEncoder) {
                    ctx.currentModel->setEncoder(*ctx.currentEncoder);
                }

                ctx.currentConfig = full_cfg;
                ctx.modelType = arch;
                ctx.modelConfig = cfg;

                ctx.addLog("Modèle créé depuis MPK: " + model_type_str);
                lua_pushboolean(L, true);
                return 1;
            } catch (const std::exception& e) {
                ctx.addLog("Erreur création modèle depuis MPK: " + std::string(e.what()));
                lua_pushboolean(L, false);
                lua_pushstring(L, e.what());
                return 2;
            }
        }
    }
    
    // Argument optionnel: config (table). Si absent: defaultConfig(model_type)
    json config;
    if (lua_istable(L, 2)) {
        config = luaTableToJson(L, 2);
    }
    
    try {
        const std::string name(model_type);
        if (!config.is_object() || config.empty()) {
            config = ModelArchitectures::defaultConfig(name);
        }

        // Création + construction via registre (le réseau est défini dans la classe du modèle).
        ctx.currentModel = ModelArchitectures::create(name, config);

        // Important: faire remonter les hyperparams de training (KL, recon_loss, clip, etc.)
        // dans `model.modelConfig` pour que Model::trainStepVAE/optimizerStep les voient.
        if (ctx.currentModel) {
            mergeLuaConfigIntoModelConfig(*ctx.currentModel, config);
        }

        // Propager les assets du contexte (tokenizer/encoder) au modèle.
        if (ctx.currentTokenizer) {
            ctx.currentModel->setTokenizer(*ctx.currentTokenizer);
        }
        if (ctx.currentEncoder) {
            ctx.currentModel->setEncoder(*ctx.currentEncoder);
        }

        ctx.modelType = name;
        ctx.modelConfig = config;

        // Si la viz est active, activer automatiquement les viz taps sur le modèle.
        // (sinon, les "blocks" ne seront jamais produits et la Viz semblera vide.)
        if (ctx.asyncMonitor && ctx.asyncMonitor->getViz() != nullptr && ctx.currentModel) {
            ctx.currentModel->setVizTapsEnabled(true);
            try {
                int max_frames = 12;
                int max_side = 64;
                if (ctx.modelConfig.contains("viz_taps_max_frames")) max_frames = ctx.modelConfig["viz_taps_max_frames"].get<int>();
                if (ctx.modelConfig.contains("viz_taps_max_side")) max_side = ctx.modelConfig["viz_taps_max_side"].get<int>();
                // Safety: too-small limits degenerate to 1x1 previews (often perceived as white squares)
                // or to a single constantly-replaced frame.
                ctx.currentModel->setVizTapsLimits(std::max(16, max_frames), std::max(16, max_side));
            } catch (...) {
            }
        }

        ctx.addLog("Modèle créé via registre: " + name);
        lua_pushboolean(L, true);
    } catch (const std::exception& e) {
        ctx.addLog("Erreur création modèle: " + std::string(e.what()));
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
    
    return 1;
}

int LuaScripting::lua_createEmptyModel(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    const char* model_type = lua_isstring(L, 1) ? lua_tostring(L, 1) : "custom_graph";

    json config = json::object();
    if (lua_istable(L, 2)) {
        config = luaTableToJson(L, 2);
        if (!config.is_object()) config = json::object();
    }

    try {
        ctx.currentModel = std::make_shared<Model>();
        if (!ctx.currentModel) {
            lua_pushboolean(L, false);
            lua_pushstring(L, "Echec creation modele vide");
            return 2;
        }

        const std::string name(model_type ? model_type : "custom_graph");
        ctx.currentModel->setModelName(name);

        ctx.modelType = name;
        ctx.modelConfig = config;
        ctx.currentModel->modelConfig = config;
        ctx.currentModel->modelConfig["type"] = name;

        if (ctx.currentTokenizer) {
            ctx.currentModel->setTokenizer(*ctx.currentTokenizer);
        }
        if (ctx.currentEncoder) {
            ctx.currentModel->setEncoder(*ctx.currentEncoder);
        }

        ctx.addLog("Modele vide cree: " + name);
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        ctx.addLog("Erreur create_empty: " + std::string(e.what()));
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_createModelFromConfig(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    // Argument: config complète (table)
    luaL_checktype(L, 1, LUA_TTABLE);
    json full = luaTableToJson(L, 1);

    try {
        std::string arch;
        json cfg;
        ctx.currentModel = ModelArchitectures::createFromConfig(full, &cfg, &arch);

        // Même principe que lua_createModel: injecter les clés de config (training) dans
        // `model.modelConfig` en respectant les overrides autorisés.
        if (ctx.currentModel) {
            mergeLuaConfigIntoModelConfig(*ctx.currentModel, cfg);
        }

        // Propager les assets déjà présents.
        if (ctx.currentTokenizer) {
            ctx.currentModel->setTokenizer(*ctx.currentTokenizer);
        }
        if (ctx.currentEncoder) {
            ctx.currentModel->setEncoder(*ctx.currentEncoder);
        }

        ctx.currentConfig = full;
        ctx.modelType = arch;
        ctx.modelConfig = cfg;

        ctx.addLog("Modèle créé via registre depuis config: " + arch);

        lua_pushboolean(L, true);
        lua_pushstring(L, arch.c_str());
        return 2;
    } catch (const std::exception& e) {
        ctx.addLog("Erreur création modèle depuis config: " + std::string(e.what()));
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_buildModel(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }

    try {
        // Framework moderne: Model.create(name, cfg) construit déjà le réseau via le registre.
        // On conserve Model.build() pour compat scripts, mais sans re-création (sinon on perd
        // les poids chargés, l'état optimiseur, etc.).
        if (ctx.currentTokenizer) {
            ctx.currentModel->setTokenizer(*ctx.currentTokenizer);
        }
        if (ctx.currentEncoder) {
            ctx.currentModel->setEncoder(*ctx.currentEncoder);
        }

        const size_t params = ctx.currentModel->totalParamCount();
        ctx.addLog("Model.build: no-op (moderne). Params=" + std::to_string(params));

        lua_pushboolean(L, true);
        lua_pushinteger(L, params);
        return 2;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_modelDType(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }

    const int nargs = lua_gettop(L);
    if (nargs <= 0) {
        lua_pushstring(L, ctx.currentModel->getDefaultDType().c_str());
        return 1;
    }

    const char* dtype = luaL_checkstring(L, 1);
    try {
        ctx.currentModel->setDefaultDType(dtype);
        // Keep Lua context config in sync (best-effort)
        try {
            ctx.modelConfig["dtype"] = std::string(dtype);
        } catch (...) {
        }
        ctx.addLog(std::string("Model.dtype = '") + dtype + "'");
        lua_pushboolean(L, true);
        lua_pushstring(L, dtype);
        return 2;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_trainModel(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    // Arguments: epochs (number), learning_rate (number)
    int epochs = luaL_checkinteger(L, 1);
    double lr = luaL_checknumber(L, 2);
    
    ctx.addLog("Entraînement: " + std::to_string(epochs) + " epochs, LR=" + std::to_string(lr));

    try {
        // Si la viz est active, activer les "viz taps" côté modèle et préparer
        // un pont vers AsyncMonitor pour afficher les blocs/layers.
        const bool viz_active = (ctx.asyncMonitor && ctx.asyncMonitor->getViz() != nullptr);
        if (viz_active && ctx.currentModel) {
            ctx.currentModel->setVizTapsEnabled(true);
            try {
                int max_frames = 12;
                int max_side = 64;
                if (ctx.modelConfig.contains("viz_taps_max_frames")) max_frames = ctx.modelConfig["viz_taps_max_frames"].get<int>();
                if (ctx.modelConfig.contains("viz_taps_max_side")) max_side = ctx.modelConfig["viz_taps_max_side"].get<int>();
                ctx.currentModel->setVizTapsLimits(max_frames, max_side);
            } catch (...) {
            }
        }

        // Instancier l'Optimizer à partir de la configuration
        Optimizer opt;
        opt.initial_lr = static_cast<float>(lr);
        
        // Type d'optimizer depuis la config (défaut: ADAMW)
        if (ctx.modelConfig.contains("optimizer")) {
            std::string opt_type = ctx.modelConfig["optimizer"];
            if (opt_type == "sgd" || opt_type == "SGD") {
                opt.type = OptimizerType::SGD;
            } else if (opt_type == "adam" || opt_type == "ADAM") {
                opt.type = OptimizerType::ADAM;
            } else if (opt_type == "adamw" || opt_type == "ADAMW") {
                opt.type = OptimizerType::ADAMW;
            }
        } else {
            opt.type = OptimizerType::ADAMW;  // Défaut
        }
        
        // Paramètres de l'optimizer depuis la config
        if (ctx.modelConfig.contains("beta1")) {
            opt.beta1 = ctx.modelConfig["beta1"];
        }
        if (ctx.modelConfig.contains("beta2")) {
            opt.beta2 = ctx.modelConfig["beta2"];
        }
        if (ctx.modelConfig.contains("epsilon")) {
            opt.eps = ctx.modelConfig["epsilon"];
        }
        if (ctx.modelConfig.contains("weight_decay")) {
            opt.weight_decay = ctx.modelConfig["weight_decay"];
        }
        
        // Paramètres de LR decay depuis la config
        if (ctx.modelConfig.contains("min_lr")) {
            opt.min_lr = ctx.modelConfig["min_lr"];
        }
        if (ctx.modelConfig.contains("decay_rate")) {
            opt.decay_rate = ctx.modelConfig["decay_rate"];
        }
        if (ctx.modelConfig.contains("decay_steps")) {
            opt.decay_steps = ctx.modelConfig["decay_steps"];
        }
        if (ctx.modelConfig.contains("warmup_steps")) {
            opt.warmup_steps = ctx.modelConfig["warmup_steps"];
        }

        // Strategy (optionnel): "none" | "cosine" | "step" | "exponential" | "linear"
        if (ctx.modelConfig.contains("decay_strategy")) {
            std::string s = ctx.modelConfig["decay_strategy"].get<std::string>();
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
            if (s == "none") opt.decay_strategy = LRDecayStrategy::NONE;
            else if (s == "cosine") opt.decay_strategy = LRDecayStrategy::COSINE;
            else if (s == "step") opt.decay_strategy = LRDecayStrategy::STEP;
            else if (s == "exponential") opt.decay_strategy = LRDecayStrategy::EXPONENTIAL;
            else if (s == "linear") opt.decay_strategy = LRDecayStrategy::LINEAR;
        }

        // Reprise éventuelle de l'état optimiseur (checkpoint)
        const Optimizer* saved_opt = ctx.currentModel->getSerializedOptimizer();
        const bool resume_optimizer_loaded = (saved_opt != nullptr);
        if (saved_opt) {
            opt = *saved_opt;
            // S'assurer que lr reflète l'argument (opt.initial_lr sert de base au scheduler)
            opt.initial_lr = static_cast<float>(lr);
        }
        
        ctx.addLog("Optimizer configuré: type=" + std::to_string(static_cast<int>(opt.type)) + 
                   ", beta1=" + std::to_string(opt.beta1) + 
                   ", beta2=" + std::to_string(opt.beta2) + 
                   ", weight_decay=" + std::to_string(opt.weight_decay));

        // -----------------------------------------------------------------
        // Entraînement moderne basé dataset, par type de modèle
        // -----------------------------------------------------------------
        std::string model_type;
        try {
            if (ctx.currentModel->modelConfig.contains("type")) {
                model_type = ctx.currentModel->modelConfig["type"].get<std::string>();
            }
        } catch (...) {
        }
        if (model_type.empty()) model_type = ctx.modelType;

        // Dataset requis pour la majorité des trains.
        if (ctx.currentDataset.empty()) {
            lua_pushboolean(L, false);
            lua_pushstring(L, "Aucun dataset chargé. Utilisez Dataset.load() d'abord.");
            return 2;
        }

        // Common knobs
        int max_items = 0;
        int log_every = 10;
        int seed = 1337;
        std::string checkpoint_dir;
        int autosave_every_epochs = 0;
        try {
            if (ctx.modelConfig.contains("max_items")) max_items = ctx.modelConfig["max_items"].get<int>();
            if (ctx.modelConfig.contains("log_every")) log_every = ctx.modelConfig["log_every"].get<int>();
            if (ctx.modelConfig.contains("seed")) seed = ctx.modelConfig["seed"].get<int>();
            if (ctx.modelConfig.contains("checkpoint_dir")) checkpoint_dir = ctx.modelConfig["checkpoint_dir"].get<std::string>();
            if (ctx.modelConfig.contains("autosave_every_epochs")) autosave_every_epochs = ctx.modelConfig["autosave_every_epochs"].get<int>();
            else if (ctx.modelConfig.contains("autosave_every_epoch")) autosave_every_epochs = ctx.modelConfig["autosave_every_epoch"].get<int>();
        } catch (...) {
        }
        max_items = std::max(0, max_items);
        log_every = std::max(1, log_every);
        autosave_every_epochs = std::max(0, autosave_every_epochs);

        // -----------------------------------------------------------------
        // CSV : csv_file / csv_path / csv_dir depuis la config du modèle.
        // Appliqué ICI (commun à tous les types) sur le HtopDisplay ET le Visualizer.
        // Une section spécifique peut encore surcharger la destination via csv_dir.
        // (pattern partN_epochM.csv) si aucun csv_file explicite n'est fourni.
        // -----------------------------------------------------------------
        {
            std::string csv_cfg_file;
            std::string csv_cfg_dir;
            try {
                if (ctx.modelConfig.contains("csv_file"))        csv_cfg_file = ctx.modelConfig["csv_file"].get<std::string>();
                else if (ctx.modelConfig.contains("csv_path"))   csv_cfg_file = ctx.modelConfig["csv_path"].get<std::string>();
                if (ctx.modelConfig.contains("csv_dir"))         csv_cfg_dir  = ctx.modelConfig["csv_dir"].get<std::string>();
            } catch (...) {}

            // Résoudre le chemin final pour le Visualizer (si csv_file vide, construire depuis csv_dir)
            std::string csv_viz_file;  // chemin Visualizer résolu
            if (!csv_cfg_file.empty()) {
                csv_viz_file = csv_cfg_file;
            } else if (!csv_cfg_dir.empty()) {
                namespace fs = std::filesystem;
                std::error_code ec;
                fs::create_directories(csv_cfg_dir, ec);
                std::string base_name;
                try {
                    if (ctx.modelConfig.contains("name"))            base_name = ctx.modelConfig["name"].get<std::string>();
                    else if (ctx.modelConfig.contains("model_name")) base_name = ctx.modelConfig["model_name"].get<std::string>();
                } catch (...) {}
                if (base_name.empty()) base_name = ctx.modelType;
                if (base_name.empty()) base_name = "model";
                for (char& c : base_name) {
                    const unsigned char uc = static_cast<unsigned char>(c);
                    if (!(std::isalnum(uc) || c == '_' || c == '-')) c = '_';
                }
                csv_viz_file = (fs::path(csv_cfg_dir) / (base_name + "_loss_history.csv")).string();
            }

            if (ctx.asyncMonitor && !csv_viz_file.empty()) {
                // ── HtopDisplay ──────────────────────────────────────────
                auto h = ctx.asyncMonitor->getHtop();
                if (h) {
                    h->setCsvLogFile(csv_viz_file);
                    h->setCsvEnabled(true);
                    ctx.addLog("CSV htop  (cfg): " + csv_viz_file);
                }
                // ── Visualizer ───────────────────────────────────────────
                // setLossLogFile() est thread-safe (pending appliqué au prochain tick).
                ctx.asyncMonitor->setLossLogFile(csv_viz_file);
                ctx.addLog("CSV viz   (cfg): " + csv_viz_file);
            }
        }

        // -----------------------------------------------------------------
        // Resume epoch offset (UX): si on reprend depuis checkpoint_dir/epoch_XXXX,
        // afficher/nommer les epochs à partir de l'epoch précédente.
        // Exemple: reprise à 3 => UI affiche 3 (pas 1) et autosave écrit epoch_0003+.
        // Active uniquement si un optimizer a été chargé via Serialization.load().
        // -----------------------------------------------------------------
        int epoch_offset = 0;          // Ajouté à epoch_1based (boucle) pour affichage + checkpoints
        if (resume_optimizer_loaded && !checkpoint_dir.empty()) {
            try {
                namespace fs = std::filesystem;
                const fs::path base(checkpoint_dir);
                int best = -1;
                if (fs::exists(base) && fs::is_directory(base)) {
                    for (auto& p : fs::directory_iterator(base)) {
                        if (!p.is_directory()) continue;
                        const std::string name = p.path().filename().string();
                        if (name.rfind("epoch_", 0) != 0) continue;
                        try {
                            size_t parsed = 0;
                            const int e = std::stoi(name.substr(std::string("epoch_").size()), &parsed);
                            if (parsed == 0) continue;
                            if (e > best) best = e;
                        } catch (...) {
                        }
                    }
                }
                if (best > 0) {
                    epoch_offset = best - 1;
                }
            } catch (...) {
                // best-effort: pas de resume epoch détectée => offset=0
            }
        }
        const int total_epochs_display = epoch_offset + epochs;

        auto do_checkpoint_save = [&](int epoch_1based, const std::string& suffix, std::string* err_out) -> bool {
            if (!ctx.currentModel) return false;
            if (checkpoint_dir.empty()) return false;

            const int epoch_abs = epoch_offset + epoch_1based;

            using namespace Mimir::Serialization;
            std::ostringstream name;
            name << "epoch_" << std::setw(4) << std::setfill('0') << epoch_abs;
            if (!suffix.empty()) name << suffix;

            const std::filesystem::path out = std::filesystem::path(checkpoint_dir) / name.str();
            std::error_code ec;
            std::filesystem::create_directories(out, ec);

            // Le tokenizer est enrichi pendant l'entraînement (tokenizeEnsure).
            // Le resynchroniser juste avant save garantit que le checkpoint contient
            // le vocabulaire effectivement composé avec le dataset.
            if (ctx.currentTokenizer) {
                ctx.currentModel->setTokenizer(*ctx.currentTokenizer);
            }

            // Important: synchroniser l'état optimiseur dans le modèle avant la sauvegarde.
            // (train utilise un Optimizer local, la sérialisation lit l'état du modèle)
            if (ctx.currentModel) {
                Optimizer opt_snapshot = opt;
                ctx.currentModel->setSerializedOptimizer(std::move(opt_snapshot));
            }

            SaveOptions so;
            so.format = CheckpointFormat::RawFolder;
            so.save_optimizer = true;
            so.save_tokenizer = true;
            so.save_encoder = true;
            so.include_git_info = true;
            so.include_checksums = true;

            std::string error;
            const bool ok = save_checkpoint(*ctx.currentModel, out.string(), so, &error);
            if (!ok) {
                if (err_out) *err_out = error;
                return false;
            }
            ctx.addLog("✓ Autosave checkpoint: " + out.string());
            return true;
        };

        std::mt19937 rng((uint32_t)seed);

        std::string recon_loss_type;
        try {
            if (ctx.modelConfig.contains("recon_loss")) recon_loss_type = ctx.modelConfig["recon_loss"].get<std::string>();
        } catch (...) {
        }

        auto recon_metric_label = [&]() -> std::string {
            std::string t = recon_loss_type;
            std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
            if (t.empty() || t == "mse") return "mse";
            if (t == "mae") return "l1";
            if (t == "cross_entropy") return "ce";
            if (t == "nll_gaussian") return "gaussian_nll";
            return t;
        };

        int global_step = 0;

        // -----------------------------------------------------------------
        // Live tuning via Viz (thread UI -> training thread)
        // -----------------------------------------------------------------
        uint64_t last_live_ver = 0;
        bool live_override_active = false;

        // Baselines (pour retour au comportement natif)
        const float baseline_lr = static_cast<float>(lr);
        const int baseline_lr_warmup_steps = std::max(0, opt.warmup_steps);
        float baseline_kl_beta = 0.0f;
        int baseline_kl_warmup_steps = 0;
        try {
            if (ctx.modelConfig.contains("kl_beta")) baseline_kl_beta = ctx.modelConfig["kl_beta"].get<float>();
            else if (ctx.modelConfig.contains("vae_kl_beta")) baseline_kl_beta = ctx.modelConfig["vae_kl_beta"].get<float>();
            if (ctx.modelConfig.contains("kl_warmup_steps")) baseline_kl_warmup_steps = ctx.modelConfig["kl_warmup_steps"].get<int>();
        } catch (...) {
        }
        baseline_kl_beta = std::max(0.0f, baseline_kl_beta);
        baseline_kl_warmup_steps = std::max(0, baseline_kl_warmup_steps);

        // ─────────────────────────────────────────────────────────────────────
        // Calibration par feedback de validation (récompense / punition sur LR)
        // Clés modelConfig: val_feedback_enabled, val_reward_factor,
        //   val_penalty_factor, val_lr_scale_min, val_lr_scale_max,
        //   val_improve_thresh, val_feedback_min_steps
        // La métrique utilisée est "inférieure = meilleure" (recon_loss, eps_mse, bce_loss…)
        // ─────────────────────────────────────────────────────────────────────
        bool  val_feedback_enabled   = true;
        float val_reward_factor      = 1.05f;   // boost LR si val s'améliore
        float val_penalty_factor     = 0.70f;   // réduction LR si val se dégrade
        float val_lr_scale_min       = 0.10f;   // plancher du facteur LR
        float val_lr_scale_max       = 1.50f;   // plafond du facteur LR
        float val_improve_thresh     = 0.001f;  // amélior. rel. min pour déclencher reward
        int   val_feedback_min_steps = 0;       // activer seulement après N steps
        float val_lr_scale           = 1.0f;    // facteur courant (multiplicateur step_lr)
        float val_best_metric        = std::numeric_limits<float>::max();
        try {
            if (ctx.modelConfig.contains("val_feedback_enabled"))   val_feedback_enabled   = ctx.modelConfig["val_feedback_enabled"].get<bool>();
            if (ctx.modelConfig.contains("val_reward_factor"))      val_reward_factor      = ctx.modelConfig["val_reward_factor"].get<float>();
            if (ctx.modelConfig.contains("val_penalty_factor"))     val_penalty_factor     = ctx.modelConfig["val_penalty_factor"].get<float>();
            if (ctx.modelConfig.contains("val_lr_scale_min"))       val_lr_scale_min       = ctx.modelConfig["val_lr_scale_min"].get<float>();
            if (ctx.modelConfig.contains("val_lr_scale_max"))       val_lr_scale_max       = ctx.modelConfig["val_lr_scale_max"].get<float>();
            if (ctx.modelConfig.contains("val_improve_thresh"))     val_improve_thresh     = ctx.modelConfig["val_improve_thresh"].get<float>();
            if (ctx.modelConfig.contains("val_feedback_min_steps")) val_feedback_min_steps = ctx.modelConfig["val_feedback_min_steps"].get<int>();
        } catch (...) {}

        auto poll_viz_live_params = [&]() {
            if (!ctx.asyncMonitor) return;
            auto viz = ctx.asyncMonitor->getViz();
            if (!viz) return;

            const uint64_t ver = viz->liveTrainParamsVersion();
            if (ver == 0 || ver == last_live_ver) return;
            last_live_ver = ver;

            const auto p = viz->liveTrainParamsSnapshot();
            if (p.version == 0) return;

            // Si on repasse en mode natif, restaurer la baseline une fois.
            if (!_mimir_live_params_overrides_enabled(p)) {
                if (live_override_active) {
                    live_override_active = false;
                    opt.initial_lr = std::max(1e-12f, baseline_lr);
                    opt.warmup_steps = baseline_lr_warmup_steps;

                    ctx.modelConfig["kl_beta"] = baseline_kl_beta;
                    ctx.modelConfig["kl_warmup_steps"] = baseline_kl_warmup_steps;
                    if (ctx.currentModel) {
                        ctx.currentModel->modelConfig["kl_beta"] = baseline_kl_beta;
                        ctx.currentModel->modelConfig["kl_warmup_steps"] = baseline_kl_warmup_steps;
                    }

                }
                return;
            }

            live_override_active = true;

            if (std::isfinite(p.lr) && p.lr > 0.0f) {
                opt.initial_lr = std::max(1e-12f, p.lr);
            }
            opt.warmup_steps = std::max(0, p.lr_warmup_steps);

            const float kl_beta = (p.kl_enabled ? std::max(0.0f, p.kl_beta) : 0.0f);
            const int kl_warmup_steps = std::max(0, p.kl_warmup_steps);

            // Propager vers la config runtime (consommée par Model::trainStepVAE, etc.)
            ctx.modelConfig["kl_beta"] = kl_beta;
            ctx.modelConfig["kl_warmup_steps"] = kl_warmup_steps;
            if (ctx.currentModel) {
                ctx.currentModel->modelConfig["kl_beta"] = kl_beta;
                ctx.currentModel->modelConfig["kl_warmup_steps"] = kl_warmup_steps;
            }

        };

        auto step_learning_rate = [&]() -> float {
            // Avant toute interaction UI, conserver le comportement historique (arg `lr`).
            const float base = live_override_active ? opt.getCurrentLR() : static_cast<float>(lr);
            return base * val_lr_scale;
        };

        // Feedback de validation: récompense ou punit le modèle en ajustant val_lr_scale.
        // metric : valeur scalaire (inférieure = meilleure).
        auto apply_val_feedback = [&](float metric, int step) {
            if (!val_feedback_enabled) return;
            if (step < val_feedback_min_steps) return;
            if (!std::isfinite(metric) || metric < 0.0f) return;

            const float prev = val_best_metric;
            const bool is_first = (prev == std::numeric_limits<float>::max());

            if (is_first) {
                val_best_metric = metric;
                ctx.addLog("[val_feedback] premier point metric=" + std::to_string(metric) +
                           " lr_scale=" + std::to_string(val_lr_scale));
                return;
            }

            const float denom = std::max(1e-12f, std::abs(prev));
            const float rel   = (prev - metric) / denom; // positif = amélioration

            if (rel > val_improve_thresh) {
                // Récompense: la validation s'améliore
                val_lr_scale    = std::min(val_lr_scale_max, val_lr_scale * val_reward_factor);
                val_best_metric = metric;
                ctx.addLog("[val_feedback] ✓ reward: prev=" + std::to_string(prev) +
                           " cur=" + std::to_string(metric) +
                           " rel_improve=" + std::to_string(rel) +
                           " -> lr_scale=" + std::to_string(val_lr_scale));
            } else if (rel < -val_improve_thresh) {
                // Punition: la validation se dégrade
                val_lr_scale = std::max(val_lr_scale_min, val_lr_scale * val_penalty_factor);
                ctx.addLog("[val_feedback] ✗ penalty: prev=" + std::to_string(prev) +
                           " cur=" + std::to_string(metric) +
                           " rel_degrade=" + std::to_string(-rel) +
                           " -> lr_scale=" + std::to_string(val_lr_scale));
            } else {
                // Plateau / variation trop faible: neutre
                ctx.addLog("[val_feedback] ~ plateau: prev=" + std::to_string(prev) +
                           " cur=" + std::to_string(metric) +
                           " lr_scale=" + std::to_string(val_lr_scale));
            }
        };

        // Perf stats pour la viz: time/mem/bps (best-effort).
        // - time: ms entre 2 updates successifs (approx temps/batch)
        // - bps: batches/sec (approx)
        // - mem: MemoryGuard current bytes en MB
        std::chrono::steady_clock::time_point last_metrics_ts;
        bool has_last_metrics_ts = false;
        auto apply_perf_stats = [&](AsyncMonitor::Metrics& m) {
            const auto now = std::chrono::steady_clock::now();
            if (has_last_metrics_ts) {
                const auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_metrics_ts).count();
                if (dt_ms > 0 && dt_ms < 60LL * 60LL * 1000LL) {
                    m.batch_time_ms = (int)dt_ms;
                    m.bps = 1000.0f / (float)m.batch_time_ms;
                }
            }
            last_metrics_ts = now;
            has_last_metrics_ts = true;

            auto& guard = MemoryGuard::instance();
            m.memory_mb = guard.getCurrentBytes() / 1024 / 1024;
        };

        auto log_step = [&](int global_step, const Model::VAEStepStats& st, const char* prefix) {
            if ((global_step % log_every) != 0) return;
            const std::string recon_label = recon_metric_label();
            ctx.addLog(std::string(prefix) +
                       " step=" + std::to_string(global_step) +
                       " loss=" + std::to_string(st.loss) +
                       " " + recon_label + "=" + std::to_string(st.mse) +
                       " kl=" + std::to_string(st.kl) +
                       " beta_eff=" + std::to_string(st.kl_beta_effective) +
                       " grad_norm=" + std::to_string(st.grad_norm));
        };

        auto monitor_step = [&](int epoch_1based, int batch_1based, int total_batches, const Model::VAEStepStats& st) {
            if (!ctx.asyncMonitor) return;

            AsyncMonitor::Metrics m;
            m.epoch = epoch_offset + epoch_1based;
            m.total_epochs = total_epochs_display;
            m.batch = batch_1based;
            m.total_batches = total_batches;
            m.loss = st.loss;
            m.avg_loss = st.loss;
            m.lr = opt.getCurrentLR();
            m.mse = st.mse;
            m.kl = st.kl;
            m.kl_beta_effective = st.kl_beta_effective;
            m.wass = st.wass;
            m.spat = st.spatial_coherence;
            m.temp = st.temp;
            m.timestep = st.timestep;
            m.grad_norm = st.grad_norm;
            m.grad_max = st.grad_max_abs;
            m.ent = st.entropy_diff;
            m.mom = st.moment_mismatch;
            m.params = ctx.currentModel ? ctx.currentModel->totalParamCount() : 0;
            m.recon_loss_type = recon_loss_type;

            // Optimizer state (pour affichage Htop/Viz)
            m.opt_type = (int)opt.type;
            m.opt_step = (int)opt.step;
            m.opt_beta1 = opt.beta1;
            m.opt_beta2 = opt.beta2;
            m.opt_eps = opt.eps;
            m.opt_weight_decay = opt.weight_decay;

            apply_perf_stats(m);

            ctx.asyncMonitor->updateMetrics(m);
        };

        // -----------------------------------------
        // VAEConv (images) optional text
        // -----------------------------------------
        if (model_type == "vae_conv") {
            int image_w = 0, image_h = 0, image_c = 3;
            bool graph_text_cond = false;
            bool requested_text_cond = false;
            bool text_cond_runtime = false;
            bool text_pipeline_enabled = false;
            bool semantic_label_study = false;
            int seq_len = 64;
            int pad_id = 0;
            float text_encoder_lr = 0.01f;
            float text_special_lr = 0.005f;
            int semantic_label_topk = 6;
            int semantic_label_min_df = 2;
            try {
                if (ctx.modelConfig.contains("image_w")) image_w = ctx.modelConfig["image_w"].get<int>();
                if (ctx.modelConfig.contains("image_h")) image_h = ctx.modelConfig["image_h"].get<int>();
                if (ctx.modelConfig.contains("image_c")) image_c = ctx.modelConfig["image_c"].get<int>();
                // text_cond: use model's own config (ground truth of what was actually built),
                // not the Lua config which may differ (e.g. vae_conv hardcodes text_cond=false).
                if (ctx.currentModel && ctx.currentModel->modelConfig.contains("text_cond"))
                    graph_text_cond = ctx.currentModel->modelConfig["text_cond"].get<bool>();
                else if (ctx.modelConfig.contains("text_cond"))
                    graph_text_cond = ctx.modelConfig["text_cond"].get<bool>();
                if (ctx.modelConfig.contains("text_cond"))
                    requested_text_cond = ctx.modelConfig["text_cond"].get<bool>();
                if (ctx.modelConfig.contains("seq_len")) seq_len = ctx.modelConfig["seq_len"].get<int>();
                if (ctx.modelConfig.contains("text_encoder_lr")) text_encoder_lr = ctx.modelConfig["text_encoder_lr"].get<float>();
                if (ctx.modelConfig.contains("text_special_lr")) text_special_lr = ctx.modelConfig["text_special_lr"].get<float>();
                if (ctx.modelConfig.contains("semantic_label_study")) semantic_label_study = ctx.modelConfig["semantic_label_study"].get<bool>();
                if (ctx.modelConfig.contains("semantic_label_topk")) semantic_label_topk = ctx.modelConfig["semantic_label_topk"].get<int>();
                if (ctx.modelConfig.contains("semantic_label_min_df")) semantic_label_min_df = ctx.modelConfig["semantic_label_min_df"].get<int>();
            } catch (...) {
            }

            text_cond_runtime = (!graph_text_cond && requested_text_cond);
            text_pipeline_enabled = (graph_text_cond || text_cond_runtime);
            if (text_pipeline_enabled && !ctx.modelConfig.contains("semantic_label_study")) {
                semantic_label_study = true;
            }

            if (image_w <= 0 || image_h <= 0) {
                // fallback: dataset config
                try {
                    if (ctx.currentConfig.contains("dataset")) {
                        if (ctx.currentConfig["dataset"].contains("target_w")) image_w = ctx.currentConfig["dataset"]["target_w"].get<int>();
                        if (ctx.currentConfig["dataset"].contains("target_h")) image_h = ctx.currentConfig["dataset"]["target_h"].get<int>();
                    }
                } catch (...) {
                }
            }
            image_w = std::max(1, image_w);
            image_h = std::max(1, image_h);
            image_c = std::max(1, image_c);
            seq_len = std::max(1, seq_len);
            text_encoder_lr = std::max(0.0f, text_encoder_lr);
            text_special_lr = std::max(0.0f, text_special_lr);
            semantic_label_topk = std::clamp(semantic_label_topk, 1, 32);
            semantic_label_min_df = std::max(1, semantic_label_min_df);

            auto extract_label_terms = [](const std::string& raw) -> std::vector<std::string> {
                std::string norm;
                norm.reserve(raw.size());
                bool last_space = true;
                for (char ch : raw) {
                    const unsigned char uc = static_cast<unsigned char>(ch);
                    bool sep = false;
                    switch (ch) {
                        case '.': case ',': case ';': case ':': case '|':
                        case '/': case '\\': case '_': case '-':
                        case '(': case ')': case '[': case ']':
                        case '{': case '}': case '"': case '\'':
                        case '\t': case '\n': case '\r':
                            sep = true;
                            break;
                        default:
                            sep = false;
                            break;
                    }
                    if (sep) {
                        if (!last_space) {
                            norm.push_back(' ');
                            last_space = true;
                        }
                        continue;
                    }
                    char outc = ch;
                    if (uc < 128) outc = static_cast<char>(std::tolower(uc));
                    if (std::isspace(static_cast<unsigned char>(outc))) {
                        if (!last_space) {
                            norm.push_back(' ');
                            last_space = true;
                        }
                    } else {
                        norm.push_back(outc);
                        last_space = false;
                    }
                }
                if (!norm.empty() && norm.back() == ' ') norm.pop_back();

                std::vector<std::string> terms;
                std::string cur;
                for (char c : norm) {
                    if (c == ' ') {
                        if (!cur.empty()) {
                            terms.push_back(cur);
                            cur.clear();
                        }
                    } else {
                        cur.push_back(c);
                    }
                }
                if (!cur.empty()) terms.push_back(cur);

                // Dedup en conservant l'ordre.
                std::unordered_set<std::string> seen;
                std::vector<std::string> uniq;
                uniq.reserve(terms.size());
                for (const auto& t : terms) {
                    if (t.empty()) continue;
                    if (seen.insert(t).second) uniq.push_back(t);
                }
                return uniq;
            };

            std::unordered_map<std::string, int> semantic_df;

            auto enrich_prompt_semantic = [&](const std::string& raw) -> std::string {
                if (!semantic_label_study) return raw;
                auto terms = extract_label_terms(raw);
                if (terms.empty()) return raw;

                struct TermSc {
                    std::string t;
                    int df;
                };
                std::vector<TermSc> scored;
                scored.reserve(terms.size());
                for (const auto& t : terms) {
                    auto it = semantic_df.find(t);
                    const int df = (it == semantic_df.end()) ? 1 : std::max(1, it->second);
                    if (df >= semantic_label_min_df) scored.push_back({t, df});
                }
                if (scored.empty()) {
                    for (const auto& t : terms) scored.push_back({t, 1});
                }

                std::stable_sort(scored.begin(), scored.end(), [](const TermSc& a, const TermSc& b) {
                    if (a.df != b.df) return a.df > b.df;
                    return a.t.size() > b.t.size();
                });

                std::string suffix;
                const int k = std::min<int>((int)scored.size(), semantic_label_topk);
                for (int i = 0; i < k; ++i) {
                    if (!suffix.empty()) suffix.push_back(' ');
                    suffix += scored[(size_t)i].t;
                }
                if (suffix.empty()) return raw;
                return raw + " ; semantic " + suffix;
            };

            if (text_pipeline_enabled) {
                if (!ctx.currentTokenizer) {
                    lua_pushboolean(L, false);
                    lua_pushstring(L, "text_cond=true mais aucun tokenizer n'est chargé");
                    return 2;
                }
                pad_id = ctx.currentTokenizer->getPadId();
            }

            if (text_cond_runtime) {
                ctx.addLog("ℹ️ vae_conv: text_cond runtime actif (graphe image-only conservé). Tokenizer+ConditioningEncoder seront entraînés pour l'alignement image↔texte.");
            }

            // Validation config (best-effort, optional)
            int validate_every_steps = 0;
            int validate_items = 0;
            float validate_holdout_frac = 0.0f;
            int validate_holdout_items = 0;
            bool validate_holdout = true;
            try {
                if (ctx.modelConfig.contains("validate_every_steps")) validate_every_steps = std::max(0, ctx.modelConfig["validate_every_steps"].get<int>());
                if (ctx.modelConfig.contains("validate_items")) validate_items = std::max(0, ctx.modelConfig["validate_items"].get<int>());
                if (ctx.modelConfig.contains("validate_holdout_frac")) validate_holdout_frac = ctx.modelConfig["validate_holdout_frac"].get<float>();
                if (ctx.modelConfig.contains("validate_holdout_items")) validate_holdout_items = std::max(0, ctx.modelConfig["validate_holdout_items"].get<int>());
                if (ctx.modelConfig.contains("validate_holdout")) validate_holdout = ctx.modelConfig["validate_holdout"].get<bool>();
            } catch (...) {
            }
            validate_holdout_frac = std::clamp(validate_holdout_frac, 0.0f, 0.95f);

            // -----------------------------------------------------------------
            // Détection automatique "VAE backbone prêt" (pour diffusion text→image)
            // Basée sur la validation: recon (MSE/L1) + KL, et un plateau sur une fenêtre.
            // Écrit un fichier JSON dans checkpoint_dir quand le critère est satisfait.
            // -----------------------------------------------------------------
            bool backbone_ready_enabled = true;
            bool backbone_ready_stop = false;
            int backbone_ready_window = 5;              // nb de validations consécutives
            float backbone_ready_plateau_rel = 0.01f;   // amélioration relative max (1% -> plateau)
            float backbone_ready_plateau_abs = 1e-4f;   // amélioration absolue max
            float backbone_ready_recon_target = 0.02f;  // 0 => ignorer seuil absolu
            float backbone_ready_kl_min = 0.01f;        // évite collapse (KL~0)
            float backbone_ready_kl_max = 5.0f;         // garde-fou
            int backbone_ready_min_steps = 0;           // ex: kl_warmup_steps
            std::string backbone_ready_file = "vae_backbone_ready.json";
            try {
                if (ctx.modelConfig.contains("backbone_ready")) backbone_ready_enabled = ctx.modelConfig["backbone_ready"].get<bool>();
                if (ctx.modelConfig.contains("backbone_ready_enabled")) backbone_ready_enabled = ctx.modelConfig["backbone_ready_enabled"].get<bool>();
                if (ctx.modelConfig.contains("backbone_ready_stop")) backbone_ready_stop = ctx.modelConfig["backbone_ready_stop"].get<bool>();
                if (ctx.modelConfig.contains("backbone_ready_window")) backbone_ready_window = std::max(2, ctx.modelConfig["backbone_ready_window"].get<int>());
                if (ctx.modelConfig.contains("backbone_ready_plateau_rel")) backbone_ready_plateau_rel = std::max(0.0f, ctx.modelConfig["backbone_ready_plateau_rel"].get<float>());
                if (ctx.modelConfig.contains("backbone_ready_plateau_abs")) backbone_ready_plateau_abs = std::max(0.0f, ctx.modelConfig["backbone_ready_plateau_abs"].get<float>());
                if (ctx.modelConfig.contains("backbone_ready_recon_target")) backbone_ready_recon_target = std::max(0.0f, ctx.modelConfig["backbone_ready_recon_target"].get<float>());
                if (ctx.modelConfig.contains("backbone_ready_kl_min")) backbone_ready_kl_min = std::max(0.0f, ctx.modelConfig["backbone_ready_kl_min"].get<float>());
                if (ctx.modelConfig.contains("backbone_ready_kl_max")) backbone_ready_kl_max = std::max(0.0f, ctx.modelConfig["backbone_ready_kl_max"].get<float>());
                if (ctx.modelConfig.contains("backbone_ready_min_steps")) backbone_ready_min_steps = std::max(0, ctx.modelConfig["backbone_ready_min_steps"].get<int>());
                if (ctx.modelConfig.contains("backbone_ready_file")) backbone_ready_file = ctx.modelConfig["backbone_ready_file"].get<std::string>();
            } catch (...) {
            }
            if (backbone_ready_min_steps <= 0) {
                try {
                    if (ctx.modelConfig.contains("kl_warmup_steps")) backbone_ready_min_steps = std::max(0, ctx.modelConfig["kl_warmup_steps"].get<int>());
                } catch (...) {
                }
            }
            if (backbone_ready_file.empty()) backbone_ready_file = "vae_backbone_ready.json";

            // On ne peut pas détecter la "readiness" sans validation.
            if (validate_every_steps <= 0 || validate_items <= 0) {
                backbone_ready_enabled = false;
            }

            bool backbone_ready_written = false;
            std::deque<float> backbone_recon_hist;
            std::deque<float> backbone_kl_hist;
            std::deque<int> backbone_step_hist;

            // Filtrer les items utilisables (au moins une image). Sans ça, `max_items` petit
            // peut tomber sur un item texte-only et faire 0 step.
            std::vector<int> indices;
            indices.reserve(ctx.currentDataset.size());
            for (size_t i = 0; i < ctx.currentDataset.size(); ++i) {
                if (!ctx.currentDataset[i].image_file.empty()) indices.push_back((int)i);
            }
            if (indices.empty()) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "Dataset: aucun item avec image_file (requis pour vae_conv)");
                return 2;
            }
            std::shuffle(indices.begin(), indices.end(), rng);
            if (max_items > 0 && (int)indices.size() > max_items) indices.resize((size_t)max_items);

            // Composer explicitement le tokenizer avec le dataset complet
            // avant l'entraînement, pour tout pipeline textuel.
            if (text_pipeline_enabled && ctx.currentTokenizer) {
                int composed_items = 0;
                for (size_t di = 0; di < ctx.currentDataset.size(); ++di) {
                    DatasetItem& ditem = ctx.currentDataset[di];
                    if (!ditem.text_file.empty() && !ditem.text.has_value()) {
                        ditem.loadText();
                    }
                    const std::string dprompt = ditem.text.has_value() ? ditem.text.value() : std::string();
                    if (!dprompt.empty()) {
                        if (semantic_label_study) {
                            const auto terms = extract_label_terms(dprompt);
                            std::unordered_set<std::string> seen_doc;
                            for (const auto& t : terms) {
                                if (seen_doc.insert(t).second) semantic_df[t] += 1;
                            }
                        }
                        (void)ctx.currentTokenizer->tokenizeEnsure(dprompt);
                        ++composed_items;
                    }
                }

                if (semantic_label_study) {
                    ctx.addLog("ℹ️  Semantic label study actif: concepts=" + std::to_string((int)semantic_df.size()) +
                               " topk=" + std::to_string(semantic_label_topk) +
                               " min_df=" + std::to_string(semantic_label_min_df));
                }

                if (ctx.currentModel) {
                    ctx.currentModel->setTokenizer(*ctx.currentTokenizer);
                }

                if (!checkpoint_dir.empty()) {
                    try {
                        namespace fs = std::filesystem;
                        const fs::path tok_path = fs::path(checkpoint_dir) / "tokenizer" / "tokenizer.json";
                        std::error_code ec_tok;
                        fs::create_directories(tok_path.parent_path(), ec_tok);

                        std::ofstream tok_out(tok_path);
                        if (tok_out.is_open()) {
                            const json jtok = ctx.currentTokenizer->to_json();
                            tok_out << jtok.dump(2);
                            tok_out.close();
                            ctx.addLog("✓ Tokenizer sauvegardé après composition complète dataset: " + tok_path.string() +
                                       " (items=" + std::to_string(composed_items) +
                                       ", vocab=" + std::to_string(ctx.currentTokenizer->getVocabSize()) + ")");
                        } else {
                            ctx.addLog("⚠️  Échec ouverture fichier tokenizer après composition complète dataset: " + tok_path.string());
                        }
                    } catch (...) {
                        ctx.addLog("⚠️  Échec sauvegarde tokenizer après composition complète dataset");
                    }
                }
            }

            // Split train/val (holdout) only if validation is enabled.
            std::vector<int> train_indices = indices;
            std::vector<int> val_indices;
            if (validate_every_steps > 0 && validate_items > 0 && (int)indices.size() >= 2) {
                if (validate_holdout) {
                    const int n = (int)indices.size();
                    int val_n = 0;
                    if (validate_holdout_items > 0) {
                        val_n = std::clamp(validate_holdout_items, 1, n - 1);
                    } else if (validate_holdout_frac > 0.0f) {
                        val_n = (int)std::floor((double)n * (double)validate_holdout_frac);
                        val_n = std::clamp(val_n, 1, n - 1);
                    }

                    if (val_n > 0) {
                        val_indices.assign(indices.end() - val_n, indices.end());
                        train_indices.assign(indices.begin(), indices.end() - val_n);
                        ctx.addLog("Validation holdout: train=" + std::to_string((int)train_indices.size()) + " val=" + std::to_string((int)val_indices.size()) +
                                   " (every=" + std::to_string(validate_every_steps) + " steps, items=" + std::to_string(validate_items) + ")");
                    } else {
                        // holdout demandé mais taille nulle => fallback: val sur train
                        val_indices = train_indices;
                        ctx.addLog("Validation (no holdout): using train set for val (every=" + std::to_string(validate_every_steps) + " steps, items=" + std::to_string(validate_items) + ")");
                    }
                } else {
                    // No holdout: validate on the training pool
                    val_indices = train_indices;
                    ctx.addLog("Validation (no holdout): using train set for val (every=" + std::to_string(validate_every_steps) + " steps, items=" + std::to_string(validate_items) + ")");
                }
            }

            const int use_n = (int)train_indices.size();
            opt.total_steps = std::max(1, epochs * std::max(1, use_n));

            const size_t expected_u8 = (size_t)image_w * (size_t)image_h * (size_t)image_c;
            std::vector<float> x;
            x.resize(expected_u8);

            std::vector<uint8_t> recon_u8;
            recon_u8.resize(expected_u8);

            // Helpers for validation rendering/metrics.
            auto pack_f32_to_u8 = [&](const std::vector<float>& src, size_t off, std::vector<uint8_t>& dst) {
                const size_t n = std::min(dst.size(), src.size() > off ? (src.size() - off) : (size_t)0);
                for (size_t i = 0; i < n; ++i) {
                    const float v = std::clamp(src[off + i], -1.0f, 1.0f);
                    const float u = (v + 1.0f) * 127.5f;
                    const int q = (int)std::lround((double)u);
                    dst[i] = (uint8_t)std::clamp(q, 0, 255);
                }
            };

            auto compute_val_recon = [&](const std::vector<float>& pred, const std::vector<float>& target, int recon_n) -> float {
                if (recon_n <= 0) return 0.0f;
                double acc = 0.0;
                if (recon_loss_type == "l1" || recon_loss_type == "mae") {
                    for (int i = 0; i < recon_n; ++i) {
                        const double d = (double)pred[(size_t)i] - (double)target[(size_t)i];
                        acc += std::abs(d);
                    }
                } else {
                    for (int i = 0; i < recon_n; ++i) {
                        const double d = (double)pred[(size_t)i] - (double)target[(size_t)i];
                        acc += d * d;
                    }
                }
                acc /= (double)std::max(1, recon_n);
                return (float)acc;
            };

            auto compute_val_kl = [&](const std::vector<float>& pred, int image_dim, int latent_dim) -> float {
                if (image_dim <= 0 || latent_dim <= 0) return 0.0f;

                float logvar_min = -10.0f;
                float logvar_max = 10.0f;
                try {
                    if (ctx.modelConfig.contains("logvar_clip_min")) logvar_min = ctx.modelConfig["logvar_clip_min"].get<float>();
                    if (ctx.modelConfig.contains("logvar_clip_max")) logvar_max = ctx.modelConfig["logvar_clip_max"].get<float>();
                } catch (...) {
                }
                if (logvar_min > logvar_max) std::swap(logvar_min, logvar_max);

                const int mu_off = image_dim;
                const int lv_off = image_dim + latent_dim;
                if ((int)pred.size() < image_dim + 2 * latent_dim) return 0.0f;

                double kl = 0.0;
                for (int i = 0; i < latent_dim; ++i) {
                    const float mu_f = pred[(size_t)(mu_off + i)];
                    const float lv_raw = pred[(size_t)(lv_off + i)];
                    const float lv = std::clamp(lv_raw, logvar_min, logvar_max);
                    const double mu = (double)mu_f;
                    const double ev = std::exp((double)lv);
                    kl += 0.5 * (mu * mu + ev - 1.0 - (double)lv);
                }
                kl /= (double)std::max(1, latent_dim);
                return (float)kl;
            };

            const std::string step_prefix = "[" + model_type + "]";
            bool stopped_by_ui = false;
            int trained_steps_total = 0;

            for (int epoch = 0; epoch < epochs; ++epoch) {
                std::shuffle(train_indices.begin(), train_indices.end(), rng);
                ctx.addLog("Epoch " + std::to_string(epoch_offset + (epoch + 1)) + "/" + std::to_string(total_epochs_display) +
                           " (" + model_type + ") items=" + std::to_string(use_n));

                bool stop_requested = false;
                int trained_steps_epoch = 0;

                for (int k = 0; k < use_n; ++k) {
                    DatasetItem& item = ctx.currentDataset[(size_t)train_indices[(size_t)k]];
                    if (item.image_file.empty()) continue;

                    item.loadImageRGB(image_w, image_h);
                    if (!item.img_loaded || item.img.size() != expected_u8) {
                        continue;
                    }

                    // Normalize u8 -> [-1, 1]
                    for (size_t i = 0; i < expected_u8; ++i) {
                        x[i] = (float)((double)item.img[i] / 127.5 - 1.0);
                    }

                    Model::VAEStepStats st;
                    std::vector<int> ids;
                    std::string prompt;
                    if (text_pipeline_enabled) {
                        if (!item.text_file.empty() && !item.text.has_value()) item.loadText();
                        prompt = item.text.has_value() ? item.text.value() : std::string();
                        const std::string prompt_for_model = enrich_prompt_semantic(prompt);
                        if (text_cond_runtime) {
                            ids = ctx.currentTokenizer->tokenizeEnsure(prompt_for_model);
                        } else {
                            ids = ctx.currentTokenizer->tokenize(prompt_for_model);
                        }
                        if ((int)ids.size() < seq_len) ids.resize((size_t)seq_len, pad_id);
                        else if ((int)ids.size() > seq_len) ids.resize((size_t)seq_len);
                    }

                    if (graph_text_cond) {
                        poll_viz_live_params();
                        st = ctx.currentModel->trainStepVAEText(x, ids, opt, step_learning_rate());
                    } else {
                        poll_viz_live_params();
                        st = ctx.currentModel->trainStepVAE(x, opt, step_learning_rate());

                        // Mode runtime optionnel: apprend la corrélation image↔texte via l'ConditioningEncoder,
                        // sans modifier le chemin de loss/convergence du graphe vae_conv existant.
                        if (text_cond_runtime && ctx.currentModel) {
                            try {
                                auto& enc = ctx.currentModel->getMutableEncoder();
                                enc.ensureVocabSize(ctx.currentTokenizer->getVocabSize());
                                enc.ensureSpecialEmbeddings();

                                const bool single_modality_sample = (item.countModalities() <= 1);

                                // Cible "image" en espace embedding:
                                // priorité au latent packé [recon | z | logvar] du VAE.
                                // Quand use_encoder_prior=true, ce chunk correspond à z_biased
                                // (z + prior), donc mag_embedding capture aussi ce signal.
                                std::vector<float> img_emb;
                                try {
                                    const std::vector<float>& packed = ctx.currentModel->forwardPassView(x, false);
                                    int latent_for_pack = std::max(0, st.latent_dim);
                                    const int image_dim_for_pack = static_cast<int>(expected_u8);
                                    const int packed_n = static_cast<int>(packed.size());
                                    if (latent_for_pack <= 0 && packed_n > image_dim_for_pack + 2 && ((packed_n - image_dim_for_pack) % 2) == 0) {
                                        latent_for_pack = std::max(1, (packed_n - image_dim_for_pack) / 2);
                                    }
                                    if (latent_for_pack > 0) {
                                        img_emb = latentChunkToEmbedding(packed, image_dim_for_pack, latent_for_pack, enc.dim);
                                    }
                                } catch (...) {
                                    // fallback ci-dessous
                                }
                                if (img_emb.empty()) {
                                    img_emb = imageToEmbedding(item.img, image_w, image_h, enc.dim);
                                }

                                // En mono-modalité image: remplir uniquement le vecteur image (mag).
                                enc.fillImageVectorSingleModality(img_emb, text_special_lr, single_modality_sample);

                                // En mono-modalité texte: remplir uniquement le vecteur texte (seq).
                                if (!ids.empty()) {
                                    enc.fillTextVectorSingleModality(ids, pad_id, text_special_lr, single_modality_sample);
                                }

                                // Mode multi-modalité: conserver l'alignement image↔texte historique.
                                if (!single_modality_sample && text_encoder_lr > 0.0f && !ids.empty()) {
                                    enc.trainOnTextTokens(ids, img_emb, text_encoder_lr);
                                }

                                if (!single_modality_sample && !ids.empty()) {
                                    // Lier mag_embedding à la modalité image du sample courant.
                                    std::string img_dir;
                                    try {
                                        img_dir = std::filesystem::path(item.image_file).parent_path().string();
                                    } catch (...) {
                                    }
                                    const MagicToken mt = makeMagicToken(MOD_IMAGE, img_dir.empty() ? item.image_file : img_dir);
                                    enc.setMagicFromToken(mt);

                                    // Gradient d'alignement sur seq/mod: encode(ids) -> img_emb.
                                    std::vector<float> txt_emb;
                                    enc.encodeInto(txt_emb, ids);
                                    if (txt_emb.size() == img_emb.size() && !txt_emb.empty() && text_special_lr > 0.0f) {
                                        std::vector<float> grad(txt_emb.size(), 0.0f);
                                        for (size_t d = 0; d < txt_emb.size(); ++d) {
                                            grad[d] = txt_emb[d] - img_emb[d];
                                        }
                                        enc.sgdUpdateSpecialEmbeddings(grad, text_special_lr, true, true, false);
                                    }

                                    // Compose mod_embedding comme pont seq <-> mag (image feature).
                                    const auto& seq = enc.getSeqEmbedding();
                                    const auto& mag = enc.getMagEmbedding();
                                    if (seq.size() == (size_t)enc.dim && mag.size() == (size_t)enc.dim) {
                                        std::vector<float> mod((size_t)enc.dim, 0.0f);
                                        for (int d = 0; d < enc.dim; ++d) {
                                            mod[(size_t)d] = 0.5f * (seq[(size_t)d] + mag[(size_t)d]);
                                        }
                                        enc.setModEmbedding(mod);
                                    }
                                }
                            } catch (...) {
                                // best-effort: ne jamais casser le train principal.
                            }
                        }
                    }

                    global_step += 1;
                    trained_steps_epoch += 1;
                    trained_steps_total += 1;
                    log_step(global_step, st, step_prefix.c_str());
                    monitor_step(epoch + 1, k + 1, use_n, st);

                    // STOP depuis la Viz (bouton dans le panneau Metrics)
                    if (ctx.asyncMonitor && ctx.asyncMonitor->consumeStopTrainingRequested()) {
                        ctx.addLog("⛔ Stop demandé via Viz. Sauvegarde et arrêt...");
                        stop_requested = true;
                        stopped_by_ui = true;
                        break;
                    }

                    // Validation: forward-only sur un petit holdout, puis push dans Generated.
                    if (validate_every_steps > 0 && validate_items > 0 && !val_indices.empty() && (global_step % validate_every_steps) == 0) {
                        const int image_dim = (int)expected_u8;
                        int latent_dim = 0;
                        try {
                            if (ctx.modelConfig.contains("latent_dim")) latent_dim = std::max(0, ctx.modelConfig["latent_dim"].get<int>());
                        } catch (...) {
                        }

                        const int total = std::min((int)val_indices.size(), std::max(1, validate_items));
                        if (ctx.asyncMonitor) ctx.asyncMonitor->updateValidation(true, global_step, 0, total, false, false, 0.0f, 0.0f, 0.0f);

                        const bool taps_prev = ctx.currentModel->isVizTapsEnabled();

                        // Sample a subset each time (shuffle copy for randomness)
                        std::vector<int> val_pick = val_indices;
                        std::shuffle(val_pick.begin(), val_pick.end(), rng);
                        if ((int)val_pick.size() > total) val_pick.resize((size_t)total);

                        double acc_recon = 0.0;
                        double acc_kl = 0.0;
                        int done = 0;
                        bool val_ok = true;

                        for (int vi = 0; vi < (int)val_pick.size(); ++vi) {
                            if (ctx.asyncMonitor && ctx.asyncMonitor->consumeStopTrainingRequested()) {
                                val_ok = false;
                                stop_requested = true;
                                stopped_by_ui = true;
                                break;
                            }

                            DatasetItem& vitem = ctx.currentDataset[(size_t)val_pick[(size_t)vi]];
                            if (vitem.image_file.empty()) continue;

                            vitem.loadImageRGB(image_w, image_h);
                            if (!vitem.img_loaded || vitem.img.size() != expected_u8) continue;

                            // Normalize u8 -> [-1, 1]
                            for (size_t i = 0; i < expected_u8; ++i) {
                                x[i] = (float)((double)vitem.img[i] / 127.5 - 1.0);
                            }

                            const std::vector<float>* ppred = nullptr;
                            std::vector<int> ids;
                            std::string vprompt;
                            if (text_pipeline_enabled) {
                                if (!vitem.text_file.empty() && !vitem.text.has_value()) vitem.loadText();
                                vprompt = vitem.text.has_value() ? vitem.text.value() : std::string();
                                const std::string vprompt_for_model = enrich_prompt_semantic(vprompt);
                                ids = ctx.currentTokenizer->tokenize(vprompt_for_model);
                                if ((int)ids.size() < seq_len) ids.resize((size_t)seq_len, pad_id);
                                else if ((int)ids.size() > seq_len) ids.resize((size_t)seq_len);

                                if (graph_text_cond) {
                                    std::unordered_map<std::string, std::vector<float>> fin;
                                    std::unordered_map<std::string, std::vector<int>> iin;
                                    fin["__input__"] = x;
                                    iin["text_ids"] = ids;
                                    ppred = &ctx.currentModel->forwardPassNamedView(fin, iin, false);
                                } else {
                                    ppred = &ctx.currentModel->forwardPassView(x, false);
                                }
                            } else {
                                ppred = &ctx.currentModel->forwardPassView(x, false);
                            }
                            if (!ppred) continue;
                            const std::vector<float>& pred = *ppred;
                            if ((int)pred.size() < image_dim + 2) continue;

                            const int recon_n = std::min(image_dim, (int)x.size());
                            const float vrecon = compute_val_recon(pred, x, recon_n);
                            float vkl = 0.0f;
                            if (latent_dim > 0) {
                                vkl = compute_val_kl(pred, image_dim, latent_dim);
                            }

                            acc_recon += (double)vrecon;
                            acc_kl += (double)vkl;
                            done += 1;

                            // Push images to Generated (target + recon)
                            if (ctx.asyncMonitor) {
                                const std::string idx = "i=" + std::to_string(val_pick[(size_t)vi]) + " step=" + std::to_string(global_step);
                                ctx.asyncMonitor->addImage(vitem.img, image_w, image_h, image_c, std::string("VAL target | ") + idx);
                                pack_f32_to_u8(pred, 0, recon_u8);
                                ctx.asyncMonitor->addImage(recon_u8, image_w, image_h, image_c, std::string("VAL recon | ") + idx);
                            }

                            if (ctx.asyncMonitor) {
                                const float avg_recon = (done > 0) ? (float)(acc_recon / (double)done) : 0.0f;
                                const float avg_kl = (done > 0) ? (float)(acc_kl / (double)done) : 0.0f;
                                ctx.asyncMonitor->updateValidation(true, global_step, done, total, true, false, avg_recon, avg_kl, 0.0f);
                            }
                        }

                        ctx.currentModel->setVizTapsEnabled(taps_prev);

                        const float final_recon = (done > 0) ? (float)(acc_recon / (double)done) : 0.0f;
                        const float final_kl = (done > 0) ? (float)(acc_kl / (double)done) : 0.0f;
                        if (ctx.asyncMonitor) ctx.asyncMonitor->updateValidation(false, global_step, done, total, true, val_ok, final_recon, final_kl, 0.0f);

                        // Calibration: récompense / punition selon l'évolution de la loss de reconstruction VAE.
                        if (val_ok && done > 0) apply_val_feedback(final_recon, global_step);

                        // Backbone readiness: heuristique basée sur validation.
                        if (backbone_ready_enabled && val_ok && done > 0 && !backbone_ready_written) {
                            backbone_recon_hist.push_back(final_recon);
                            backbone_kl_hist.push_back(final_kl);
                            backbone_step_hist.push_back(global_step);
                            while ((int)backbone_recon_hist.size() > backbone_ready_window) backbone_recon_hist.pop_front();
                            while ((int)backbone_kl_hist.size() > backbone_ready_window) backbone_kl_hist.pop_front();
                            while ((int)backbone_step_hist.size() > backbone_ready_window) backbone_step_hist.pop_front();

                            const bool have_window = ((int)backbone_recon_hist.size() >= backbone_ready_window);
                            const bool warm_enough = (global_step >= backbone_ready_min_steps);

                            if (have_window && warm_enough) {
                                const float r0 = backbone_recon_hist.front();
                                const float r1 = backbone_recon_hist.back();
                                const float k1 = backbone_kl_hist.back();
                                const float denom = std::max(1e-12f, std::abs(r0));
                                const float rel_improve = (r0 - r1) / denom;
                                const float abs_improve = std::abs(r0 - r1);

                                const bool plateau = (rel_improve >= 0.0f) && (rel_improve <= backbone_ready_plateau_rel) && (abs_improve <= backbone_ready_plateau_abs);

                                bool recon_ok = true;
                                if (backbone_ready_recon_target > 0.0f) {
                                    recon_ok = (r1 <= backbone_ready_recon_target);
                                }

                                bool kl_ok = (k1 >= backbone_ready_kl_min);
                                if (backbone_ready_kl_max > 0.0f) {
                                    kl_ok = kl_ok && (k1 <= backbone_ready_kl_max);
                                }

                                if (plateau && recon_ok && kl_ok) {
                                    // Écrire un marqueur JSON dans checkpoint_dir.
                                    try {
                                        namespace fs = std::filesystem;
                                        fs::path out_base = checkpoint_dir.empty() ? fs::path(".") : fs::path(checkpoint_dir);
                                        std::error_code ec;
                                        fs::create_directories(out_base, ec);
                                        fs::path out_path = out_base / backbone_ready_file;

                                        json j;
                                        j["type"] = "vae_backbone_ready";
                                        j["model_type"] = model_type;
                                        j["global_step"] = global_step;
                                        j["epoch"] = epoch_offset + (epoch + 1);
                                        j["validate_items"] = done;
                                        j["val_recon"] = final_recon;
                                        j["val_kl"] = final_kl;
                                        j["plateau_rel"] = backbone_ready_plateau_rel;
                                        j["plateau_abs"] = backbone_ready_plateau_abs;
                                        j["recon_target"] = backbone_ready_recon_target;
                                        j["kl_min"] = backbone_ready_kl_min;
                                        j["kl_max"] = backbone_ready_kl_max;
                                        j["window"] = backbone_ready_window;
                                        j["min_steps"] = backbone_ready_min_steps;
                                        j["recon_history"] = backbone_recon_hist;
                                        j["kl_history"] = backbone_kl_hist;
                                        j["step_history"] = backbone_step_hist;

                                        // Contexte utile pour diffusion
                                        try {
                                            if (ctx.modelConfig.contains("image_w")) j["image_w"] = ctx.modelConfig["image_w"].get<int>();
                                            if (ctx.modelConfig.contains("image_h")) j["image_h"] = ctx.modelConfig["image_h"].get<int>();
                                            if (ctx.modelConfig.contains("image_c")) j["image_c"] = ctx.modelConfig["image_c"].get<int>();
                                            if (ctx.modelConfig.contains("latent_h")) j["latent_h"] = ctx.modelConfig["latent_h"].get<int>();
                                            if (ctx.modelConfig.contains("latent_w")) j["latent_w"] = ctx.modelConfig["latent_w"].get<int>();
                                            if (ctx.modelConfig.contains("latent_c")) j["latent_c"] = ctx.modelConfig["latent_c"].get<int>();
                                            if (ctx.modelConfig.contains("base_channels")) j["base_channels"] = ctx.modelConfig["base_channels"].get<int>();
                                            if (ctx.modelConfig.contains("stochastic_latent")) j["stochastic_latent"] = ctx.modelConfig["stochastic_latent"].get<bool>();
                                            if (ctx.modelConfig.contains("kl_beta")) j["kl_beta"] = ctx.modelConfig["kl_beta"].get<float>();
                                        } catch (...) {
                                        }

                                        std::ofstream f(out_path);
                                        f << j.dump(2);
                                        f.close();

                                        backbone_ready_written = true;
                                        ctx.addLog("✅ VAE backbone prêt pour diffusion (marqueur: " + out_path.string() + ")");
                                    } catch (...) {
                                        // Ne casse pas l'entraînement si l'écriture échoue.
                                        ctx.addLog("⚠️  VAE backbone readiness: échec écriture marqueur JSON");
                                    }

                                    if (backbone_ready_stop) {
                                        ctx.addLog("⛔ backbone_ready_stop=true: arrêt demandé (plateau atteint)");
                                        stop_requested = true;
                                        stopped_by_ui = true;
                                    }
                                }
                            }
                        }

                        if (stop_requested) {
                            break;
                        }
                    }

                    // VIZ: pousser le contexte dataset + blocs uniquement tous les log_every steps
                    // (réduit le coût; la viz reste “stale” entre deux updates, volontairement)
                    if (viz_active && ctx.asyncMonitor && ctx.currentModel && ctx.asyncMonitor->getViz() != nullptr && ((global_step % log_every) == 0)) {
                        std::string label = "vae_conv/input/dataset/rgb";
                        label += "/i=" + std::to_string(train_indices[(size_t)k]);

                        const std::vector<int>* ids_ptr = (text_pipeline_enabled && !ids.empty()) ? &ids : nullptr;
                        const auto viz_payload = VizTextPayload::buildDatasetTextPayload(
                            ctx.currentModel.get(),
                            ctx.currentTokenizer.get(),
                            prompt,
                            ids_ptr,
                            text_pipeline_enabled ? pad_id : -1
                        );

                        ctx.asyncMonitor->setDatasetSample(
                            item.img,
                            image_w,
                            image_h,
                            image_c,
                            label,
                            viz_payload.prompt,
                            viz_payload.tags,
                            viz_payload.tokens,
                            viz_payload.encoding
                        );

                        // Important UX:
                        // Par défaut, on affiche les "taps" en état réel (ceux produits par
                        // les passes réelles du modèle durant l'entraînement), sans forcer
                        // un forward additionnel en mode inference/freeze.
                        //
                        // Si l'utilisateur veut explicitement un snapshot inference (ex: pour
                        // rendre la recon plus lisible quand stochastic_latent=true), il peut
                        // activer `viz_taps_force_inference=true` dans la config.
                        bool viz_force_inference = false;
                        try {
                            if (ctx.modelConfig.contains("viz_taps_force_inference")) {
                                viz_force_inference = ctx.modelConfig["viz_taps_force_inference"].get<bool>();
                            }
                        } catch (...) {
                        }

                        if (viz_force_inference) {
                            ctx.currentModel->clearVizTaps();
                            try {
                                if (graph_text_cond) {
                                    std::vector<int> ids = ctx.currentTokenizer ? ctx.currentTokenizer->tokenize(prompt) : std::vector<int>();
                                    if ((int)ids.size() < seq_len) ids.resize((size_t)seq_len, pad_id);
                                    else if ((int)ids.size() > seq_len) ids.resize((size_t)seq_len);

                                    std::unordered_map<std::string, std::vector<float>> fin;
                                    std::unordered_map<std::string, std::vector<int>> iin;
                                    fin["__input__"] = x;
                                    iin["text_ids"] = std::move(ids);
                                    (void)ctx.currentModel->forwardPassNamedView(fin, iin, false);
                                } else {
                                    (void)ctx.currentModel->forwardPassView(x, false);
                                }
                            } catch (...) {
                                // best-effort: ne jamais casser le training à cause de la viz
                            }
                        }

                        auto taps = ctx.currentModel->consumeVizTaps();
                        std::vector<Visualizer::BlockFrame> frames;
                        frames.reserve(taps.size());
                        for (auto& f : taps) {
                            Visualizer::BlockFrame bf;
                            bf.pixels = std::move(f.pixels);
                            bf.w = f.w;
                            bf.h = f.h;
                            bf.channels = f.channels;
                            bf.pixels_real = std::move(f.pixels_real);
                            bf.label = std::move(f.label);
                            frames.push_back(std::move(bf));
                        }
                        // Important UX: même si aucun tap n'est émis, vider les frames précédentes.
                        ctx.asyncMonitor->setLayerBlockImages(frames);
                    }
                }

                // Autosave à la fin de chaque epoch
                if (!stop_requested && autosave_every_epochs > 0 && ((epoch + 1) % autosave_every_epochs) == 0) {
                    std::string err_save;
                    (void)do_checkpoint_save(epoch + 1, std::string(), &err_save);
                    if (!err_save.empty()) {
                        ctx.addLog("⚠ Autosave échoué: " + err_save);
                    }
                }

                if (stop_requested) {
                    // Sauvegarde forcée même si autosave désactivé
                    std::string err_save;
                    (void)do_checkpoint_save(epoch + 1, "_stop", &err_save);
                    if (!err_save.empty()) {
                        ctx.addLog("⚠ Save(stop) échoué: " + err_save);
                    }
                    break;
                }

                ctx.addLog("Epoch stats: trained_steps=" + std::to_string(trained_steps_epoch) +
                           "/" + std::to_string(use_n) +
                           " global_step=" + std::to_string(global_step));

                if (trained_steps_epoch == 0) {
                    ctx.addLog("⚠ Aucun batch entraine sur cette epoch (0/" + std::to_string(use_n) + ") - images invalides/inaccessibles ou dataset non charge correctement.");
                    break;
                }
            }

            if (trained_steps_total == 0) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "Aucun batch n'a ete entraine (0 step). Verifiez dataset_root, acces fichiers image, et preload/caching des images.");
                return 2;
            }

            ctx.currentModel->setSerializedOptimizer(std::move(opt));

            if (stopped_by_ui) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "STOP_REQUESTED");
                return 2;
            }
            lua_pushboolean(L, true);
            lua_pushinteger(L, global_step);
            return 2;
        } else if (model_type == "vgg16_feat") {
            int image_w = 0, image_h = 0, image_c = 3;
            int grid = 8;
            int viz_taps_every_steps = 0;
            try {
                if (ctx.modelConfig.contains("image_w")) image_w = ctx.modelConfig["image_w"].get<int>();
                if (ctx.modelConfig.contains("image_h")) image_h = ctx.modelConfig["image_h"].get<int>();
                if (ctx.modelConfig.contains("image_c")) image_c = ctx.modelConfig["image_c"].get<int>();
                if (ctx.modelConfig.contains("pretrain_grid")) grid = ctx.modelConfig["pretrain_grid"].get<int>();
                if (ctx.modelConfig.contains("viz_taps_every_steps")) viz_taps_every_steps = ctx.modelConfig["viz_taps_every_steps"].get<int>();
                else if (ctx.modelConfig.contains("viz_every_steps")) viz_taps_every_steps = ctx.modelConfig["viz_every_steps"].get<int>();
            } catch (...) {
            }
            if (image_w <= 0 || image_h <= 0) {
                // fallback: dataset config
                try {
                    if (ctx.currentConfig.contains("dataset")) {
                        if (ctx.currentConfig["dataset"].contains("target_w")) image_w = ctx.currentConfig["dataset"]["target_w"].get<int>();
                        if (ctx.currentConfig["dataset"].contains("target_h")) image_h = ctx.currentConfig["dataset"]["target_h"].get<int>();
                    }
                } catch (...) {
                }
            }
            image_w = std::max(1, image_w);
            image_h = std::max(1, image_h);
            image_c = std::max(1, image_c);
            grid = std::clamp(grid, 2, 32);

            // Viz taps frequency (best-effort). Default: follow log_every.
            if (viz_active) {
                if (viz_taps_every_steps <= 0) viz_taps_every_steps = log_every;
                viz_taps_every_steps = std::max(1, viz_taps_every_steps);
            } else {
                viz_taps_every_steps = 0;
            }

            auto push_viz_taps = [&]() {
                if (!viz_active) return;
                if (!ctx.currentModel || !ctx.asyncMonitor) return;
                if (ctx.asyncMonitor->getViz() == nullptr) return;
                auto taps = ctx.currentModel->consumeVizTaps();
                std::vector<Visualizer::BlockFrame> frames;
                frames.reserve(taps.size());
                for (auto& f : taps) {
                    Visualizer::BlockFrame bf;
                    bf.pixels = std::move(f.pixels);
                    bf.w = f.w;
                    bf.h = f.h;
                    bf.channels = f.channels;
                    bf.pixels_real = std::move(f.pixels_real);
                            bf.label = std::move(f.label);
                    frames.push_back(std::move(bf));
                }
                // Important UX: même si aucun tap n'est émis, vider les frames précédentes.
                ctx.asyncMonitor->setLayerBlockImages(frames);
            };

            // Filter usable items
            std::vector<int> indices;
            indices.reserve(ctx.currentDataset.size());
            for (size_t i = 0; i < ctx.currentDataset.size(); ++i) {
                if (!ctx.currentDataset[i].image_file.empty()) indices.push_back((int)i);
            }
            if (indices.empty()) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "Dataset: aucun item avec image_file (requis pour vgg16_feat)");
                return 2;
            }
            std::shuffle(indices.begin(), indices.end(), rng);
            if (max_items > 0 && (int)indices.size() > max_items) indices.resize((size_t)max_items);

            const int use_n = (int)indices.size();
            opt.total_steps = std::max(1, epochs * std::max(1, use_n));

            const size_t expected_u8 = (size_t)image_w * (size_t)image_h * (size_t)image_c;
            std::vector<float> x;
            x.resize(expected_u8);

            auto compute_patch_means = [&](const std::vector<float>& src_hwc, int out_dim) -> std::vector<float> {
                // Features: grid*grid*image_c means (HWC input)
                const int feat_dim = grid * grid * image_c;
                std::vector<double> sum((size_t)feat_dim, 0.0);
                std::vector<int> cnt((size_t)(grid * grid), 0);

                for (int yy = 0; yy < image_h; ++yy) {
                    const int gy = (yy * grid) / image_h;
                    for (int xx = 0; xx < image_w; ++xx) {
                        const int gx = (xx * grid) / image_w;
                        const int cell = gy * grid + gx;
                        const int off = cell * image_c;
                        const size_t pix = (size_t)(yy * image_w + xx) * (size_t)image_c;
                        for (int cc = 0; cc < image_c; ++cc) {
                            sum[(size_t)off + (size_t)cc] += (double)src_hwc[pix + (size_t)cc];
                        }
                        cnt[(size_t)cell] += 1;
                    }
                }

                std::vector<float> feat((size_t)feat_dim, 0.0f);
                for (int cell = 0; cell < grid * grid; ++cell) {
                    const int n = std::max(1, cnt[(size_t)cell]);
                    const int off = cell * image_c;
                    for (int cc = 0; cc < image_c; ++cc) {
                        feat[(size_t)off + (size_t)cc] = (float)(sum[(size_t)off + (size_t)cc] / (double)n);
                    }
                }

                // Truncate/pad to out_dim
                out_dim = std::max(1, out_dim);
                std::vector<float> y((size_t)out_dim, 0.0f);
                const int m = std::min(out_dim, feat_dim);
                for (int i = 0; i < m; ++i) y[(size_t)i] = feat[(size_t)i];
                return y;
            };

            bool stopped_by_ui = false;

            // Perf stats vgg16_feat -> Viz.
            std::chrono::steady_clock::time_point last_vgg_metrics_ts;
            bool has_last_vgg_metrics_ts = false;
            auto apply_vgg_perf_stats = [&](AsyncMonitor::Metrics& m) {
                const auto now = std::chrono::steady_clock::now();
                if (has_last_vgg_metrics_ts) {
                    const auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_vgg_metrics_ts).count();
                    if (dt_ms > 0 && dt_ms < 60LL * 60LL * 1000LL) {
                        m.batch_time_ms = (int)dt_ms;
                        m.bps = 1000.0f / (float)m.batch_time_ms;
                    }
                }
                last_vgg_metrics_ts = now;
                has_last_vgg_metrics_ts = true;

                auto& guard = MemoryGuard::instance();
                m.memory_mb = guard.getCurrentBytes() / 1024 / 1024;
            };

            for (int epoch = 0; epoch < epochs; ++epoch) {
                std::shuffle(indices.begin(), indices.end(), rng);
                ctx.addLog("Epoch " + std::to_string(epoch_offset + (epoch + 1)) + "/" + std::to_string(total_epochs_display) +
                           " (vgg16_feat) items=" + std::to_string(use_n));

                bool stop_requested = false;

                for (int k = 0; k < use_n; ++k) {
                    DatasetItem& item = ctx.currentDataset[(size_t)indices[(size_t)k]];
                    if (item.image_file.empty()) continue;

                    item.loadImageRGB(image_w, image_h);
                    if (!item.img_loaded || item.img.size() != expected_u8) {
                        continue;
                    }

                    // Normalize u8 -> [-1, 1]
                    for (size_t i = 0; i < expected_u8; ++i) {
                        x[i] = (float)((double)item.img[i] / 127.5 - 1.0);
                    }

                    // Forward
                    ctx.currentModel->zeroGradients();
                    const std::vector<float>& pred = ctx.currentModel->forwardPassView(x, true);
                    const int out_dim = std::max(1, (int)pred.size());

                    // Target is a deterministic downsample summary of the same input.
                    const std::vector<float> y = compute_patch_means(x, out_dim);

                    // Loss + grad
                    double loss = 0.0;
                    std::vector<float> grad;
                    grad.resize((size_t)out_dim);
                    const double inv = 1.0 / (double)std::max(1, out_dim);
                    for (int i = 0; i < out_dim; ++i) {
                        const double d = (double)pred[(size_t)i] - (double)y[(size_t)i];
                        loss += d * d;
                        grad[(size_t)i] = (float)(2.0 * d * inv);
                    }
                    loss *= inv;

                    // Backward + step
                    ctx.currentModel->backwardPass(grad);

                    // Vraie norme L2 globale + max abs des gradients (avant clipping/step).
                    // Sans ça, les métriques vgg16_feat affichaient grad_norm=grad_max=0
                    // (valeurs codées en dur), masquant toute explosion de gradient.
                    float grad_norm_val = 0.0f;
                    float grad_max_val = 0.0f;
                    {
                        double sum_sq = 0.0;
                        float max_abs = 0.0f;
                        for (const auto& layer : ctx.currentModel->getLayers()) {
                            for (float g : layer.grad_weights) {
                                sum_sq += (double)g * (double)g;
                                const float a = std::fabs(g);
                                if (a > max_abs) max_abs = a;
                            }
                            for (float g : layer.grad_bias) {
                                sum_sq += (double)g * (double)g;
                                const float a = std::fabs(g);
                                if (a > max_abs) max_abs = a;
                            }
                        }
                        grad_norm_val = (float)std::sqrt(sum_sq);
                        grad_max_val = max_abs;
                    }

                    poll_viz_live_params();
                    ctx.currentModel->optimizerStep(opt, step_learning_rate(), nullptr);

                    // Metrics for UI/log
                    global_step += 1;

                    // Viz taps (throttled)
                    if (viz_taps_every_steps > 0 && (global_step % viz_taps_every_steps) == 0) {
                        push_viz_taps();
                    }

                    if ((global_step % log_every) == 0) {
                        ctx.addLog("step=" + std::to_string(global_step) +
                                   " loss=" + std::to_string((float)loss) +
                                   " grad_norm=" + std::to_string(grad_norm_val) +
                                   " lr=" + std::to_string(opt.getCurrentLR()) +
                                   " (vgg16_feat)");
                    }
                    if (ctx.asyncMonitor) {
                        AsyncMonitor::Metrics m;
                        m.epoch = epoch_offset + (epoch + 1);
                        m.total_epochs = total_epochs_display;
                        m.batch = k + 1;
                        m.total_batches = use_n;
                        m.loss = (float)loss;
                        m.avg_loss = (float)loss;
                        m.lr = opt.getCurrentLR();
                        m.mse = (float)loss;
                        m.grad_norm = grad_norm_val;
                        m.grad_max = grad_max_val;
                        m.params = ctx.currentModel ? ctx.currentModel->totalParamCount() : 0;
                        m.recon_loss_type = "mse";
                        m.opt_type = (int)opt.type;
                        m.opt_step = (int)opt.step;
                        m.opt_beta1 = opt.beta1;
                        m.opt_beta2 = opt.beta2;
                        m.opt_eps = opt.eps;
                        m.opt_weight_decay = opt.weight_decay;

                        apply_vgg_perf_stats(m);
                        ctx.asyncMonitor->updateMetrics(m);
                    }

                    // STOP via Viz
                    if (ctx.asyncMonitor && ctx.asyncMonitor->consumeStopTrainingRequested()) {
                        ctx.addLog("⛔ Stop demandé via Viz. Sauvegarde et arrêt...");
                        stop_requested = true;
                        stopped_by_ui = true;
                        break;
                    }
                }

                // Autosave per epoch
                if (!checkpoint_dir.empty() && autosave_every_epochs > 0) {
                    const int epoch_1based = epoch + 1;
                    if ((epoch_1based % autosave_every_epochs) == 0) {
                        std::string save_err;
                        if (!do_checkpoint_save(epoch_1based, std::string(), &save_err)) {
                            ctx.addLog("⚠️ Autosave failed: " + save_err);
                        }
                    }
                }

                if (stop_requested) break;
            }

            // Save optimizer state back into the model for later Serialization.save()
            if (ctx.currentModel) {
                Optimizer opt_snapshot = opt;
                ctx.currentModel->setSerializedOptimizer(std::move(opt_snapshot));
            }

            if (stopped_by_ui) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "STOP_REQUESTED");
                return 2;
            }

            lua_pushboolean(L, true);
            return 1;
        } else if (model_type == "vgg16" || model_type == "vgg19") {
            // Multi-label image tag classification.
            // Dataset items must have image_file + text, where text is dot-separated tags/short phrases.
            // Requires cfg.tags_vocab (array of strings) to define fixed class indices.

            int image_w = 0, image_h = 0, image_c = 3;
            int viz_taps_every_steps = 0;
            bool lowercase_tags = true;
            float bce_pos_weight = 1.0f;
            std::vector<float> class_pos_weights;

            // Validation (optional)
            int validate_every_steps = 0;
            int validate_every_epochs = 0;
            int validate_items = 0;
            float validate_threshold = 0.5f;
            bool validate_holdout = false;
            float validate_holdout_frac = 0.0f;
            int validate_holdout_items = 0;
            int validate_seed = 12345;
            try {
                if (ctx.modelConfig.contains("image_w")) image_w = ctx.modelConfig["image_w"].get<int>();
                if (ctx.modelConfig.contains("image_h")) image_h = ctx.modelConfig["image_h"].get<int>();
                if (ctx.modelConfig.contains("image_c")) image_c = ctx.modelConfig["image_c"].get<int>();
                if (ctx.modelConfig.contains("viz_taps_every_steps")) viz_taps_every_steps = ctx.modelConfig["viz_taps_every_steps"].get<int>();
                else if (ctx.modelConfig.contains("viz_every_steps")) viz_taps_every_steps = ctx.modelConfig["viz_every_steps"].get<int>();
                if (ctx.modelConfig.contains("lowercase_tags")) lowercase_tags = ctx.modelConfig["lowercase_tags"].get<bool>();

                if (ctx.modelConfig.contains("pos_weight")) bce_pos_weight = ctx.modelConfig["pos_weight"].get<float>();
                else if (ctx.modelConfig.contains("bce_pos_weight")) bce_pos_weight = ctx.modelConfig["bce_pos_weight"].get<float>();

                if (ctx.modelConfig.contains("class_pos_weights") && ctx.modelConfig["class_pos_weights"].is_array()) {
                    const auto& arr = ctx.modelConfig["class_pos_weights"];
                    class_pos_weights.reserve(arr.size());
                    for (const auto& v : arr) {
                        try {
                            class_pos_weights.push_back(std::max(0.0f, v.get<float>()));
                        } catch (...) {
                            class_pos_weights.push_back(0.0f);
                        }
                    }
                }

                if (ctx.modelConfig.contains("validate_every_steps")) validate_every_steps = std::max(0, ctx.modelConfig["validate_every_steps"].get<int>());
                else if (ctx.modelConfig.contains("validate_every")) validate_every_steps = std::max(0, ctx.modelConfig["validate_every"].get<int>());

                if (ctx.modelConfig.contains("validate_every_epochs")) validate_every_epochs = std::max(0, ctx.modelConfig["validate_every_epochs"].get<int>());
                else if (ctx.modelConfig.contains("validate_epochs")) validate_every_epochs = std::max(0, ctx.modelConfig["validate_epochs"].get<int>());

                if (ctx.modelConfig.contains("validate_items")) validate_items = std::max(0, ctx.modelConfig["validate_items"].get<int>());
                else if (ctx.modelConfig.contains("val_items")) validate_items = std::max(0, ctx.modelConfig["val_items"].get<int>());

                if (ctx.modelConfig.contains("validate_threshold")) validate_threshold = ctx.modelConfig["validate_threshold"].get<float>();
                else if (ctx.modelConfig.contains("val_threshold")) validate_threshold = ctx.modelConfig["val_threshold"].get<float>();

                if (ctx.modelConfig.contains("validate_holdout")) validate_holdout = ctx.modelConfig["validate_holdout"].get<bool>();
                else if (ctx.modelConfig.contains("val_holdout")) validate_holdout = ctx.modelConfig["val_holdout"].get<bool>();

                if (ctx.modelConfig.contains("validate_holdout_frac")) validate_holdout_frac = ctx.modelConfig["validate_holdout_frac"].get<float>();
                else if (ctx.modelConfig.contains("val_holdout_frac")) validate_holdout_frac = ctx.modelConfig["val_holdout_frac"].get<float>();

                if (ctx.modelConfig.contains("validate_holdout_items")) validate_holdout_items = std::max(0, ctx.modelConfig["validate_holdout_items"].get<int>());
                else if (ctx.modelConfig.contains("val_holdout_items")) validate_holdout_items = std::max(0, ctx.modelConfig["val_holdout_items"].get<int>());

                if (ctx.modelConfig.contains("validate_seed")) validate_seed = ctx.modelConfig["validate_seed"].get<int>();
                else if (ctx.modelConfig.contains("val_seed")) validate_seed = ctx.modelConfig["val_seed"].get<int>();
            } catch (...) {
            }

            bce_pos_weight = std::max(0.0f, bce_pos_weight);
            validate_threshold = std::clamp(validate_threshold, 0.0f, 1.0f);
            validate_holdout_frac = std::clamp(validate_holdout_frac, 0.0f, 0.95f);

            image_w = std::max(1, image_w);
            image_h = std::max(1, image_h);
            image_c = std::max(1, image_c);

            if (!ctx.modelConfig.contains("tags_vocab") || !ctx.modelConfig["tags_vocab"].is_array()) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "vgg16/vgg19 multi-label: cfg.tags_vocab manquant (liste de tags/classes)");
                return 2;
            }

            const auto& vocab_json = ctx.modelConfig["tags_vocab"];
            const int num_classes = (int)vocab_json.size();
            if (num_classes <= 0) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "vgg16/vgg19 multi-label: cfg.tags_vocab vide");
                return 2;
            }
            if (!class_pos_weights.empty() && (int)class_pos_weights.size() != num_classes) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "vgg16/vgg19 multi-label: cfg.class_pos_weights doit matcher cfg.tags_vocab");
                return 2;
            }

            auto trim = [](std::string& s) {
                auto is_ws = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
                size_t a = 0;
                while (a < s.size() && is_ws((unsigned char)s[a])) a++;
                size_t b = s.size();
                while (b > a && is_ws((unsigned char)s[b - 1])) b--;
                if (a != 0 || b != s.size()) s = s.substr(a, b - a);
            };

            auto trim_punct = [&](std::string& s) {
                // Retire ponctuation fréquente autour des tags (ex: "tag1," "(tag2)" "tag3\"")
                auto is_junk = [](unsigned char c) {
                    switch (c) {
                        case ' ': case '\t': case '\n': case '\r':
                        case ',': case ';': case ':':
                        case '!': case '?':
                        case '"': case '\'':
                        case '(': case ')':
                        case '[': case ']':
                        case '{': case '}':
                        case '<': case '>':
                        case '/': case '\\':
                        case '|':
                        case '-':
                            return true;
                        default:
                            return false;
                    }
                };
                size_t a = 0;
                while (a < s.size() && is_junk((unsigned char)s[a])) a++;
                size_t b = s.size();
                while (b > a && is_junk((unsigned char)s[b - 1])) b--;
                if (a != 0 || b != s.size()) s = s.substr(a, b - a);
                trim(s);
            };

            // Build label->class_id map.
            std::unordered_map<std::string, int> tag_to_id;
            tag_to_id.reserve((size_t)num_classes * 2ULL);
            for (int i = 0; i < num_classes; ++i) {
                try {
                    std::string t = vocab_json[(size_t)i].get<std::string>();
                    if (t.empty()) continue;
                    trim_punct(t);
                    if (t.empty()) continue;
                    if (lowercase_tags) {
                        for (char& ch : t) ch = (char)std::tolower((unsigned char)ch);
                    }
                    tag_to_id.emplace(std::move(t), i);
                } catch (...) {
                }
            }
            if (tag_to_id.empty()) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "vgg16/vgg19 multi-label: cfg.tags_vocab invalide (aucun tag utilisable)");
                return 2;
            }

            // Viz taps frequency (best-effort). Default: follow log_every.
            if (viz_active) {
                if (viz_taps_every_steps <= 0) viz_taps_every_steps = log_every;
                viz_taps_every_steps = std::max(1, viz_taps_every_steps);
            } else {
                viz_taps_every_steps = 0;
            }

            auto push_viz_taps = [&]() {
                if (!viz_active) return;
                if (!ctx.currentModel || !ctx.asyncMonitor) return;
                if (ctx.asyncMonitor->getViz() == nullptr) return;
                auto taps = ctx.currentModel->consumeVizTaps();
                std::vector<Visualizer::BlockFrame> frames;
                frames.reserve(taps.size());
                for (auto& f : taps) {
                    Visualizer::BlockFrame bf;
                    bf.pixels = std::move(f.pixels);
                    bf.w = f.w;
                    bf.h = f.h;
                    bf.channels = f.channels;
                    bf.pixels_real = std::move(f.pixels_real);
                            bf.label = std::move(f.label);
                    frames.push_back(std::move(bf));
                }
                ctx.asyncMonitor->setLayerBlockImages(frames);
            };

            // Filter usable items
            std::vector<int> indices;
            indices.reserve(ctx.currentDataset.size());
            for (size_t i = 0; i < ctx.currentDataset.size(); ++i) {
                const auto& it = ctx.currentDataset[i];
                if (!it.image_file.empty() && (!it.text_file.empty() || !it.text_inline.empty() || it.text.has_value())) {
                    indices.push_back((int)i);
                }
            }
            if (indices.empty()) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "Dataset: aucun item avec image_file + texte exploitable (text_file ou text_inline) pour vgg16/vgg19 multi-label");
                return 2;
            }
            std::shuffle(indices.begin(), indices.end(), rng);
            if (max_items > 0 && (int)indices.size() > max_items) indices.resize((size_t)max_items);

            const bool validation_enabled = (validate_items > 0) && ((validate_every_steps > 0) || (validate_every_epochs > 0));

            // Split train/val if requested; default is to validate on train indices (no holdout).
            std::vector<int> train_indices = indices;
            std::vector<int> val_indices;
            if (validation_enabled) {
                if (validate_holdout && (int)indices.size() >= 2) {
                    const int n = (int)indices.size();
                    int val_n = 0;
                    if (validate_holdout_items > 0) {
                        val_n = std::clamp(validate_holdout_items, 1, n - 1);
                    } else if (validate_holdout_frac > 0.0f) {
                        val_n = (int)std::floor((double)n * (double)validate_holdout_frac);
                        val_n = std::clamp(val_n, 1, n - 1);
                    }
                    if (val_n > 0) {
                        val_indices.assign(indices.end() - val_n, indices.end());
                        train_indices.assign(indices.begin(), indices.end() - val_n);
                    } else {
                        val_indices = train_indices;
                    }
                } else {
                    val_indices = train_indices;
                }

                ctx.addLog("Validation: every_steps=" + std::to_string(validate_every_steps) +
                           " every_epochs=" + std::to_string(validate_every_epochs) +
                           " items=" + std::to_string(validate_items) +
                           " threshold=" + std::to_string(validate_threshold) +
                           " holdout=" + std::string(validate_holdout ? "true" : "false") +
                           " train=" + std::to_string((int)train_indices.size()) +
                           " val=" + std::to_string((int)val_indices.size()));
            }

            const int use_n = (int)train_indices.size();

            opt.total_steps = std::max(1, epochs * std::max(1, use_n));
            const size_t expected_u8 = (size_t)image_w * (size_t)image_h * (size_t)image_c;
            std::vector<float> x_chw;
            x_chw.resize(expected_u8);
            std::vector<float> y;
            y.resize((size_t)num_classes);
            double running_loss_sum = 0.0;
            int running_loss_count = 0;

            auto split_tags = [&](const std::string& txt0) {
                auto out = PromptParsing::extractLabelCandidates(txt0, lowercase_tags);
                for (auto& s : out) trim_punct(s);
                out.erase(std::remove_if(out.begin(), out.end(), [](const std::string& s) {
                    return s.empty();
                }), out.end());
                return out;
            };

            auto sigmoid = [](double z) {
                if (z >= 0.0) {
                    const double ez = std::exp(-z);
                    return 1.0 / (1.0 + ez);
                } else {
                    const double ez = std::exp(z);
                    return ez / (1.0 + ez);
                }
            };

            struct ValStats {
                int items = 0;
                double loss = 0.0;
                double f1_micro = 0.0;
                double pos_true_rate = 0.0;
                double pos_pred_rate = 0.0;
                double avg_prob = 0.0;
            };

            std::mt19937 val_rng((uint32_t)validate_seed);
            auto pos_weight_for_class = [&](int class_id) -> double {
                if (class_id >= 0 && class_id < (int)class_pos_weights.size()) {
                    const double w = (double)class_pos_weights[(size_t)class_id];
                    if (w > 0.0) return w;
                }
                return (double)bce_pos_weight;
            };

            auto run_validation = [&](int step, int epoch_1based) -> ValStats {
                ValStats st;
                if (!validation_enabled) return st;
                if (!ctx.currentModel) return st;
                if (val_indices.empty()) return st;

                std::vector<int> work = val_indices;
                std::shuffle(work.begin(), work.end(), val_rng);

                const int want = std::min(validate_items, (int)work.size());
                if (want <= 0) return st;

                long long tp = 0, fp = 0, fn = 0;
                long long true_pos = 0;
                long long pred_pos = 0;
                double prob_sum = 0.0;
                double loss_sum = 0.0;

                int done = 0;
                for (int j = 0; j < (int)work.size() && done < want; ++j) {
                    DatasetItem& item = ctx.currentDataset[(size_t)work[(size_t)j]];
                    if (item.image_file.empty() || (!item.text.has_value() && item.text_file.empty() && item.text_inline.empty())) continue;
                    if (!item.loadText() || !item.text.has_value()) continue;

                    item.loadImageRGB(image_w, image_h);
                    if (!item.img_loaded || item.img.size() != expected_u8) continue;

                    // Labels
                    std::fill(y.begin(), y.end(), 0.0f);
                    const std::vector<std::string> tags = split_tags(item.text.value());
                    for (const auto& t : tags) {
                        auto it = tag_to_id.find(t);
                        if (it != tag_to_id.end()) {
                            const int id = it->second;
                            if (id >= 0 && id < num_classes) {
                                if (y[(size_t)id] < 0.5f) y[(size_t)id] = 1.0f;
                            }
                        }
                    }

                    // Image -> CHW
                    for (int yy = 0; yy < image_h; ++yy) {
                        for (int xx = 0; xx < image_w; ++xx) {
                            const size_t pix = (size_t)(yy * image_w + xx) * (size_t)image_c;
                            for (int cc = 0; cc < image_c; ++cc) {
                                const float v = (float)((double)item.img[pix + (size_t)cc] / 127.5 - 1.0);
                                const size_t idx = (size_t)cc * (size_t)(image_h * image_w) + (size_t)(yy * image_w + xx);
                                x_chw[idx] = v;
                            }
                        }
                    }

                    const std::vector<float>& logits = ctx.currentModel->forwardPassView(x_chw, false);
                    if ((int)logits.size() != num_classes) continue;

                    double loss = 0.0;
                    for (int i = 0; i < num_classes; ++i) {
                        const double z = (double)logits[(size_t)i];
                        const double yi = (double)y[(size_t)i];
                        const double sp_pos = std::max(0.0, z) + std::log1p(std::exp(-std::fabs(z)));
                        const double sp_neg = std::max(0.0, -z) + std::log1p(std::exp(-std::fabs(z)));
                        const double wpos = (yi >= 0.5) ? pos_weight_for_class(i) : 1.0;
                        loss += (1.0 - yi) * sp_pos + yi * wpos * sp_neg;

                        const double p = sigmoid(z);
                        prob_sum += p;
                        const bool y1 = (yi >= 0.5);
                        const bool p1 = (p >= (double)validate_threshold);
                        if (y1) true_pos++;
                        if (p1) pred_pos++;
                        if (p1 && y1) tp++;
                        else if (p1 && !y1) fp++;
                        else if (!p1 && y1) fn++;
                    }
                    loss *= (1.0 / (double)std::max(1, num_classes));
                    loss_sum += loss;
                    done++;
                }

                st.items = done;
                if (done <= 0) return st;

                const double denom = (double)done * (double)std::max(1, num_classes);
                st.loss = loss_sum / (double)done;
                st.pos_true_rate = denom > 0 ? (double)true_pos / denom : 0.0;
                st.pos_pred_rate = denom > 0 ? (double)pred_pos / denom : 0.0;
                st.avg_prob = denom > 0 ? prob_sum / denom : 0.0;
                const double f1_den = (2.0 * (double)tp + (double)fp + (double)fn);
                st.f1_micro = (f1_den > 0.0) ? (2.0 * (double)tp) / f1_den : 0.0;

                ctx.addLog(std::string("val") +
                           " step=" + std::to_string(step) +
                           " epoch=" + std::to_string(epoch_1based) +
                           " items=" + std::to_string(done) +
                           " loss=" + std::to_string((float)st.loss) +
                           " f1_micro=" + std::to_string((float)st.f1_micro) +
                           " pos_true=" + std::to_string((float)st.pos_true_rate) +
                           " pos_pred=" + std::to_string((float)st.pos_pred_rate) +
                           " avg_p=" + std::to_string((float)st.avg_prob));

                if (ctx.asyncMonitor) {
                    AsyncMonitor::Metrics vm;
                    vm.epoch = epoch_offset + epoch_1based;
                    vm.total_epochs = total_epochs_display;
                    vm.batch = 0;
                    vm.total_batches = use_n;
                    vm.loss = (float)st.loss;
                    vm.avg_loss = (float)st.loss;
                    vm.lr = opt.getCurrentLR();
                    vm.params = ctx.currentModel ? ctx.currentModel->totalParamCount() : 0;
                    vm.recon_loss_type = "bce_logits";

                    vm.val_has = true;
                    vm.val_ok = true;
                    vm.val_in_progress = false;
                    vm.val_step = step;
                    vm.val_items = done;
                    vm.val_done = done;
                    vm.val_total = done;
                    vm.val_recon = (float)st.loss;
                    vm.val_align = (float)st.f1_micro;

                    vm.opt_type = (int)opt.type;
                    vm.opt_step = (int)opt.step;
                    vm.opt_beta1 = opt.beta1;
                    vm.opt_beta2 = opt.beta2;
                    vm.opt_eps = opt.eps;
                    vm.opt_weight_decay = opt.weight_decay;

                    ctx.asyncMonitor->updateMetrics(vm);
                }

                // Collapse warnings (simple heuristics)
                if (st.pos_true_rate < 1e-9) {
                    ctx.addLog("⚠️  Labels: pos_true_rate=0 (aucun tag du texte ne matche tags_vocab). "
                               "Le modèle va naturellement converger vers des prédictions ~0.");
                } else if (st.pos_pred_rate < 1e-3 || st.pos_pred_rate > 1.0 - 1e-3) {
                    ctx.addLog("⚠️  Possible collapse: pos_pred_rate=" + std::to_string((float)st.pos_pred_rate) +
                               " (pos_true_rate=" + std::to_string((float)st.pos_true_rate) +
                               ", avg_p=" + std::to_string((float)st.avg_prob) + ")");
                }

                return st;
            };

            bool stopped_by_ui = false;

            // Perf stats vgg16_tags_multilabel -> Viz.
            std::chrono::steady_clock::time_point last_tags_metrics_ts;
            bool has_last_tags_metrics_ts = false;
            auto apply_tags_perf_stats = [&](AsyncMonitor::Metrics& m) {
                const auto now = std::chrono::steady_clock::now();
                if (has_last_tags_metrics_ts) {
                    const auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_tags_metrics_ts).count();
                    if (dt_ms > 0 && dt_ms < 60LL * 60LL * 1000LL) {
                        m.batch_time_ms = (int)dt_ms;
                        m.bps = 1000.0f / (float)m.batch_time_ms;
                    }
                }
                last_tags_metrics_ts = now;
                has_last_tags_metrics_ts = true;

                auto& guard = MemoryGuard::instance();
                m.memory_mb = guard.getCurrentBytes() / 1024 / 1024;
            };
            bool warned_no_vocab_match = false;

            for (int epoch = 0; epoch < epochs; ++epoch) {
                std::shuffle(train_indices.begin(), train_indices.end(), rng);
                ctx.addLog("Epoch " + std::to_string(epoch_offset + (epoch + 1)) + "/" + std::to_string(total_epochs_display) +
                           " (" + std::string(model_type) + " multi-label) items=" + std::to_string(use_n));

                bool stop_requested = false;

                for (int k = 0; k < use_n; ++k) {
                    DatasetItem& item = ctx.currentDataset[(size_t)train_indices[(size_t)k]];
                    if (item.image_file.empty() || (!item.text.has_value() && item.text_file.empty() && item.text_inline.empty())) continue;

                    // Lazy-load text on demand (uses DatasetMemoryManager).
                    if (!item.loadText() || !item.text.has_value()) {
                        continue;
                    }

                    item.loadImageRGB(image_w, image_h);
                    if (!item.img_loaded || item.img.size() != expected_u8) {
                        continue;
                    }

                    // Build labels (multi-hot)
                    std::fill(y.begin(), y.end(), 0.0f);
                    const std::vector<std::string> tags = split_tags(item.text.value());
                    int active_labels = 0;
                    for (const auto& t : tags) {
                        auto it = tag_to_id.find(t);
                        if (it != tag_to_id.end()) {
                            const int id = it->second;
                            if (id >= 0 && id < num_classes) {
                                if (y[(size_t)id] < 0.5f) {
                                    y[(size_t)id] = 1.0f;
                                    active_labels++;
                                }
                            }
                        }
                    }

                    // Normalize u8 -> [-1, 1] AND permute HWC -> CHW (Conv2d expects CHW)
                    for (int yy = 0; yy < image_h; ++yy) {
                        for (int xx = 0; xx < image_w; ++xx) {
                            const size_t pix = (size_t)(yy * image_w + xx) * (size_t)image_c;
                            for (int cc = 0; cc < image_c; ++cc) {
                                const float v = (float)((double)item.img[pix + (size_t)cc] / 127.5 - 1.0);
                                const size_t idx = (size_t)cc * (size_t)(image_h * image_w) + (size_t)(yy * image_w + xx);
                                x_chw[idx] = v;
                            }
                        }
                    }

                    // Forward
                    ctx.currentModel->zeroGradients();
                    const std::vector<float>& logits = ctx.currentModel->forwardPassView(x_chw, true);
                    const int out_dim = (int)logits.size();
                    if (out_dim != num_classes) {
                        lua_pushboolean(L, false);
                        lua_pushstring(L, "vgg16/vgg19 multi-label: output_dim != num_classes (cfg.num_classes doit matcher tags_vocab)");
                        return 2;
                    }

                    // BCEWithLogits loss + gradient (+ optional positive class reweighting)
                    double loss = 0.0;
                    std::vector<float> grad;
                    grad.resize((size_t)num_classes);
                    const double inv = 1.0 / (double)std::max(1, num_classes);

                    long long tp = 0, fp = 0, fn = 0;
                    long long true_pos = 0;
                    long long pred_pos = 0;
                    double prob_sum = 0.0;
                    for (int i = 0; i < num_classes; ++i) {
                        const double z = (double)logits[(size_t)i];
                        const double yi = (double)y[(size_t)i];
                        const double sp_pos = std::max(0.0, z) + std::log1p(std::exp(-std::fabs(z)));
                        const double sp_neg = std::max(0.0, -z) + std::log1p(std::exp(-std::fabs(z)));
                        const double wpos = (yi >= 0.5) ? pos_weight_for_class(i) : 1.0;
                        loss += (1.0 - yi) * sp_pos + yi * wpos * sp_neg;
                        const double p = sigmoid(z);
                        const double gw = (yi >= 0.5) ? pos_weight_for_class(i) : 1.0;
                        grad[(size_t)i] = (float)((p - yi) * gw * inv);

                        prob_sum += p;
                        const bool y1 = (yi >= 0.5);
                        const bool p1 = (p >= (double)validate_threshold);
                        if (y1) true_pos++;
                        if (p1) pred_pos++;
                        if (p1 && y1) tp++;
                        else if (p1 && !y1) fp++;
                        else if (!p1 && y1) fn++;
                    }
                    loss *= inv;

                    ctx.currentModel->backwardPass(grad);
                    poll_viz_live_params();
                    ctx.currentModel->optimizerStep(opt, step_learning_rate(), nullptr);
                    running_loss_sum += loss;
                    running_loss_count += 1;

                    global_step += 1;

                    if (viz_active && ctx.asyncMonitor && ctx.asyncMonitor->getViz() != nullptr &&
                        viz_taps_every_steps > 0 && ((global_step % viz_taps_every_steps) == 0)) {
                        std::string label = std::string(model_type) + "/input/dataset/rgb";
                        label += "/i=" + std::to_string(train_indices[(size_t)k]);
                        const auto viz_payload = VizTextPayload::buildDatasetTextPayload(
                            ctx.currentModel.get(),
                            ctx.currentTokenizer.get(),
                            item.text.value(),
                            nullptr,
                            -1,
                            false,
                            true,
                            true
                        );
                        ctx.asyncMonitor->setDatasetSample(
                            item.img,
                            image_w,
                            image_h,
                            image_c,
                            label,
                            viz_payload.prompt,
                            viz_payload.tags,
                            viz_payload.tokens,
                            viz_payload.encoding
                        );
                    }

                    if (validation_enabled && validate_every_steps > 0 && (global_step % validate_every_steps) == 0) {
                        const auto vs = run_validation(global_step, (epoch + 1));
                        if (vs.items > 0) apply_val_feedback(static_cast<float>(vs.loss), global_step);
                    }

                    if (viz_taps_every_steps > 0 && (global_step % viz_taps_every_steps) == 0) {
                        push_viz_taps();
                    }

                    if ((global_step % log_every) == 0) {
                        const double denom = (double)std::max(1, num_classes);
                        const double pos_true_rate = denom > 0.0 ? (double)true_pos / denom : 0.0;
                        const double pos_pred_rate = denom > 0.0 ? (double)pred_pos / denom : 0.0;
                        const double avg_prob = denom > 0.0 ? prob_sum / denom : 0.0;
                        const double f1_den = (2.0 * (double)tp + (double)fp + (double)fn);
                        const double f1_micro = (f1_den > 0.0) ? (2.0 * (double)tp) / f1_den : 0.0;
                        const double avg_loss = (running_loss_count > 0) ? (running_loss_sum / (double)running_loss_count) : loss;
                        ctx.addLog("step=" + std::to_string(global_step) +
                                   " loss=" + std::to_string((float)loss) +
                                   " avg_loss=" + std::to_string((float)avg_loss) +
                                   " lr=" + std::to_string(opt.getCurrentLR()) +
                                   " pos_true=" + std::to_string((float)pos_true_rate) +
                                   " pos_pred=" + std::to_string((float)pos_pred_rate) +
                                   " f1_micro=" + std::to_string((float)f1_micro) +
                                   " active_labels=" + std::to_string(active_labels) +
                                   " label_candidates=" + std::to_string((int)tags.size()));

                        if (!warned_no_vocab_match && ((int)tags.size() > 0) && active_labels == 0) {
                            warned_no_vocab_match = true;
                            ctx.addLog("⚠️  Aucun label ne matche tags_vocab (active_labels=0). "
                                       "Parsing supporté: JSON standard (`tags`/`keywords`/`tokens`) puis fallback séparateurs `.,;|` et nouvelles lignes.");
                        }
                    }
                    if (ctx.asyncMonitor) {
                        AsyncMonitor::Metrics m;
                        m.epoch = epoch_offset + (epoch + 1);
                        m.total_epochs = total_epochs_display;
                        m.batch = k + 1;
                        m.total_batches = use_n;
                        m.loss = (float)loss;
                        m.avg_loss = (float)((running_loss_count > 0) ? (running_loss_sum / (double)running_loss_count) : loss);
                        m.lr = opt.getCurrentLR();
                        m.mse = 0.0f;
                        m.grad_norm = 0.0f;
                        m.grad_max = 0.0f;
                        m.params = ctx.currentModel ? ctx.currentModel->totalParamCount() : 0;
                        m.recon_loss_type = "bce_logits";
                        m.opt_type = (int)opt.type;
                        m.opt_step = (int)opt.step;
                        m.opt_beta1 = opt.beta1;
                        m.opt_beta2 = opt.beta2;
                        m.opt_eps = opt.eps;
                        m.opt_weight_decay = opt.weight_decay;

                        apply_tags_perf_stats(m);
                        ctx.asyncMonitor->updateMetrics(m);
                    }

                    if (ctx.asyncMonitor && ctx.asyncMonitor->consumeStopTrainingRequested()) {
                        ctx.addLog("⛔ Stop demandé via Viz. Sauvegarde et arrêt...");
                        stop_requested = true;
                        stopped_by_ui = true;
                        break;
                    }
                }

                if (validation_enabled && validate_every_epochs > 0) {
                    const int epoch_1based = epoch + 1;
                    if ((epoch_1based % validate_every_epochs) == 0) {
                        const auto vse = run_validation(global_step, epoch_1based);
                        if (vse.items > 0) apply_val_feedback(static_cast<float>(vse.loss), global_step);
                    }
                }

                // Autosave per epoch
                if (!checkpoint_dir.empty() && autosave_every_epochs > 0) {
                    const int epoch_1based = epoch + 1;
                    if ((epoch_1based % autosave_every_epochs) == 0) {
                        std::string save_err;
                        if (!do_checkpoint_save(epoch_1based, std::string(), &save_err)) {
                            ctx.addLog("⚠️ Autosave failed: " + save_err);
                        }
                    }
                }

                if (stop_requested) break;
            }

            if (ctx.currentModel) {
                Optimizer opt_snapshot = opt;
                ctx.currentModel->setSerializedOptimizer(std::move(opt_snapshot));
            }

            if (stopped_by_ui) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "STOP_REQUESTED");
                return 2;
            }

            lua_pushboolean(L, true);
            return 1;
        }

        // -------------------------------
        // VAEText (text_ids) -> recon
        // -------------------------------
        if (model_type == "vae_text") {
            if (!ctx.currentTokenizer) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "Aucun tokenizer chargé (requis pour vae_text)");
                return 2;
            }

            std::vector<int> indices;
            indices.reserve(ctx.currentDataset.size());
            for (size_t i = 0; i < ctx.currentDataset.size(); ++i) {
                if (!ctx.currentDataset[i].text_file.empty()) indices.push_back((int)i);
            }
            if (indices.empty()) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "Dataset: aucun item avec text_file (requis pour vae_text)");
                return 2;
            }
            std::shuffle(indices.begin(), indices.end(), rng);
            if (max_items > 0 && (int)indices.size() > max_items) indices.resize((size_t)max_items);

            const int use_n = (int)indices.size();
            opt.total_steps = std::max(1, epochs * use_n);

            int seq_len = 256;
            int pad_id = ctx.currentTokenizer->getPadId();
            try {
                if (ctx.modelConfig.contains("seq_len")) seq_len = ctx.modelConfig["seq_len"].get<int>();
            } catch (...) {
            }
            seq_len = std::max(1, seq_len);

            std::vector<float> empty_x;

            bool stopped_by_ui = false;
            for (int epoch = 0; epoch < epochs; ++epoch) {
                std::shuffle(indices.begin(), indices.end(), rng);
                ctx.addLog("Epoch " + std::to_string(epoch_offset + (epoch + 1)) + "/" + std::to_string(total_epochs_display) +
                           " (vae_text) items=" + std::to_string(use_n));

                bool stop_requested = false;

                for (int k = 0; k < use_n; ++k) {
                    DatasetItem& item = ctx.currentDataset[(size_t)indices[(size_t)k]];
                    if (item.text_file.empty()) continue;
                    if (!item.text.has_value()) item.loadText();
                    if (!item.text.has_value()) continue;

                    std::vector<int> ids = ctx.currentTokenizer->tokenize(item.text.value());
                    if ((int)ids.size() < seq_len) ids.resize((size_t)seq_len, pad_id);
                    else if ((int)ids.size() > seq_len) ids.resize((size_t)seq_len);

                    poll_viz_live_params();
                    const Model::VAEStepStats st = ctx.currentModel->trainStepVAEText(empty_x, ids, opt, step_learning_rate());

                    // Entraînement texte mono-modalité: remplir uniquement le vecteur texte (seq).
                    try {
                        auto& enc = ctx.currentModel->getMutableEncoder();
                        enc.ensureVocabSize(ctx.currentTokenizer->getVocabSize());
                        enc.ensureSpecialEmbeddings();
                        enc.fillTextVectorSingleModality(ids, pad_id, 0.005f, true);
                    } catch (...) {
                        // best-effort: ne jamais casser le train principal.
                    }

                    global_step += 1;
                    log_step(global_step, st, "[vae_text]");
                    monitor_step(epoch + 1, k + 1, use_n, st);

                    if (ctx.asyncMonitor && ctx.asyncMonitor->consumeStopTrainingRequested()) {
                        ctx.addLog("⛔ Stop demandé via Viz. Sauvegarde et arrêt...");
                        stop_requested = true;
                        stopped_by_ui = true;
                        break;
                    }
                }

                if (!stop_requested && autosave_every_epochs > 0 && ((epoch + 1) % autosave_every_epochs) == 0) {
                    std::string err_save;
                    (void)do_checkpoint_save(epoch + 1, std::string(), &err_save);
                    if (!err_save.empty()) {
                        ctx.addLog("⚠ Autosave échoué: " + err_save);
                    }
                }

                if (stop_requested) {
                    std::string err_save;
                    (void)do_checkpoint_save(epoch + 1, "_stop", &err_save);
                    if (!err_save.empty()) {
                        ctx.addLog("⚠ Save(stop) échoué: " + err_save);
                    }
                    break;
                }
            }

            ctx.currentModel->setSerializedOptimizer(std::move(opt));

            if (stopped_by_ui) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "STOP_REQUESTED");
                return 2;
            }
            lua_pushboolean(L, true);
            lua_pushinteger(L, global_step);
            return 2;
        }

        lua_pushboolean(L, false);
        lua_pushstring(L, ("Model.train: type non supporté (type='" + model_type + "')").c_str());
        return 2;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_inferModel(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    // Argument: input (string ou table)
    if (lua_isstring(L, 1)) {
        const char* input = lua_tostring(L, 1);
        ctx.addLog("Inférence sur: " + std::string(input));
        
        try {
            // Tokenize input
            std::vector<int> tokens;
            if (ctx.currentTokenizer) {
                tokens = ctx.currentTokenizer->tokenize(input);
            } else {
                // Simple word tokenization si pas de tokenizer
                tokens = {1, 2, 3, 4, 5}; // Placeholder
            }
            
            // Encode
            std::vector<float> encoding;
            if (ctx.currentEncoder) {
                encoding = ctx.currentEncoder->encode(tokens);
                ctx.currentModel->setLastEncoding(encoding);
            }
            
            // Forward pass
            std::vector<uint8_t> output;
            ctx.currentModel->forward(output);
            
            // Decode result
            auto result = ctx.currentModel->eval(output);
            
            // Convert tokens back to text
            if (ctx.currentTokenizer && !result.tokens.empty()) {
                std::string decoded = ctx.currentTokenizer->decode(result.tokens);
                lua_pushstring(L, decoded.c_str());
            } else {
                lua_pushstring(L, "[output tokens]");
            }
        } catch (const std::exception& e) {
            lua_pushstring(L, ("Error: " + std::string(e.what())).c_str());
        }
    } else {
        lua_pushnil(L);
    }
    
    return 1;
}

// ============================================================================
// ModelArchitectures Registry helpers (Lua)
// ============================================================================

int LuaScripting::lua_archAvailable(lua_State* L) {
    try {
        const auto names = ModelArchitectures::available();
        lua_createtable(L, static_cast<int>(names.size()), 0);
        int i = 1;
        for (const auto& name : names) {
            lua_pushstring(L, name.c_str());
            lua_rawseti(L, -2, i++);
        }
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_archDefaultConfig(lua_State* L) {
    const char* name = luaL_checkstring(L, 1);
    try {
        json cfg = ModelArchitectures::defaultConfig(name);
        jsonToLuaTable(L, cfg);
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

// Renvoie toutes les infos présentes dans le registry pour une architecture.
// Sans argument: renvoie un tableau (liste) d'entrées complètes pour toutes
// les architectures. Avec un nom: renvoie l'entrée correspondante.
// Chaque entrée contient aussi origin="native"|"mpk" et, pour un MPK,
// source_path=<fichier chargé>.
int LuaScripting::lua_archInfo(lua_State* L) {
    auto pushEntry = [L](const ModelArchitectures::Entry& e) {
        lua_newtable(L);
        lua_pushstring(L, e.name.c_str());
        lua_setfield(L, -2, "name");
        lua_pushstring(L, e.description.c_str());
        lua_setfield(L, -2, "description");
        lua_pushstring(L, e.origin.c_str());
        lua_setfield(L, -2, "origin");
        if (!e.source_path.empty()) {
            lua_pushstring(L, e.source_path.c_str());
            lua_setfield(L, -2, "source_path");
        }
        jsonToLuaTable(L, e.default_config);
        lua_setfield(L, -2, "config");
    };

    try {
        auto& reg = ModelArchitectures::Registry::instance();

        // Cas 1: un nom est fourni -> entrée unique.
        if (lua_gettop(L) >= 1 && !lua_isnil(L, 1)) {
            const char* name = luaL_checkstring(L, 1);
            const ModelArchitectures::Entry* e = reg.find(name);
            if (e == nullptr) {
                lua_pushnil(L);
                lua_pushstring(L, (std::string("unknown architecture: ") + name).c_str());
                return 2;
            }
            pushEntry(*e);
            return 1;
        }

        // Cas 2: aucun nom -> liste complète de toutes les entrées.
        const auto names = ModelArchitectures::available();
        lua_createtable(L, static_cast<int>(names.size()), 0);
        int i = 1;
        for (const auto& name : names) {
            const ModelArchitectures::Entry* e = reg.find(name);
            if (e != nullptr) {
                pushEntry(*e);
                lua_rawseti(L, -2, i++);
            }
        }
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

// Renvoie la liste des dtypes pris en charge par le framework.
// Chaque entrée: { name=<canonique>, aliases=<csv>, bytes=<n>, kind="float|int|uint|bool" }.
int LuaScripting::lua_archDtypes(lua_State* L) {
    struct DtInfo {
        Mimir::DType dt;
        const char* aliases;
        const char* kind;
    };
    // Ordre logique d'affichage (flottants d'abord, puis entiers, puis bool).
    static const DtInfo kAll[] = {
        { Mimir::DType::F32,  "float, f32, float32",   "float" },
        { Mimir::DType::F16,  "f16, float16, fp16",    "float" },
        { Mimir::DType::BF16, "bf16, bfloat16",        "float" },
        { Mimir::DType::F64,  "double, f64, float64",  "float" },
        { Mimir::DType::I8,   "i8, int8",              "int"   },
        { Mimir::DType::I16,  "i16, int16",            "int"   },
        { Mimir::DType::I32,  "i32, int32",            "int"   },
        { Mimir::DType::I64,  "i64, int64",            "int"   },
        { Mimir::DType::U8,   "u8, uint8",             "uint"  },
        { Mimir::DType::U16,  "u16, uint16",           "uint"  },
        { Mimir::DType::U32,  "u32, uint32",           "uint"  },
        { Mimir::DType::U64,  "u64, uint64",           "uint"  },
        { Mimir::DType::BOOL, "bool, b1",              "bool"  },
    };

    const int n = static_cast<int>(sizeof(kAll) / sizeof(kAll[0]));
    lua_createtable(L, n, 0);
    for (int i = 0; i < n; ++i) {
        const auto& e = kAll[i];
        lua_newtable(L);
        lua_pushstring(L, Mimir::dtype_to_string(e.dt));
        lua_setfield(L, -2, "name");
        lua_pushstring(L, e.aliases);
        lua_setfield(L, -2, "aliases");
        lua_pushinteger(L, static_cast<lua_Integer>(Mimir::dtype_size_bytes(e.dt)));
        lua_setfield(L, -2, "bytes");
        lua_pushstring(L, e.kind);
        lua_setfield(L, -2, "kind");
        lua_rawseti(L, -2, i + 1);
    }
    return 1;
}

int LuaScripting::lua_saveModel(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    const char* path = luaL_checkstring(L, 1);
    
    try {
        fs::path save_dir(path);
        
        // Créer le dossier si nécessaire
        if (!fs::exists(save_dir)) {
            fs::create_directories(save_dir);
        }
        
        // Sauvegarder le checkpoint
        std::vector<MagicToken> magic_tokens;  // Empty for now
        
        Tokenizer tokenizer = ctx.currentTokenizer ? *ctx.currentTokenizer : Tokenizer();
        
        bool success = ctx.currentModel->saveCheckpoint(
            tokenizer,
            magic_tokens,
            save_dir,
            0  // epoch 0
        );
        
        if (success) {
            ctx.addLog("Modèle sauvegardé: " + std::string(path));
            lua_pushboolean(L, true);
        } else {
            lua_pushboolean(L, false);
            lua_pushstring(L, "Erreur lors de la sauvegarde");
            return 2;
        }
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
    
    return 1;
}

int LuaScripting::lua_loadModel(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    const char* path = luaL_checkstring(L, 1);
    
    try {
        fs::path load_dir(path);
        
        if (!fs::exists(load_dir)) {
            lua_pushboolean(L, false);
            lua_pushstring(L, "Le chemin n'existe pas");
            return 2;
        }
        
        // Chercher le fichier safetensors
        fs::path safetensor_path = load_dir / "weights.safetensors";
        if (!fs::exists(safetensor_path)) {
            safetensor_path = load_dir / "model.safetensors";
        }
        
        Tokenizer tokenizer;
        ConditioningEncoder encoder;
        std::vector<MagicToken> magic_tokens;
        
        bool success = ctx.currentModel->tryLoadExistingModel(
            load_dir,
            safetensor_path,
            tokenizer,
            encoder,
            magic_tokens
        );
        
        if (success) {
            ctx.currentTokenizer = std::make_shared<Tokenizer>(tokenizer);
            ctx.currentEncoder = std::make_shared<ConditioningEncoder>(encoder);

            // Si la viz est active, activer les taps pour permettre l'affichage des blocks.
            if (ctx.asyncMonitor && ctx.asyncMonitor->getViz() != nullptr && ctx.currentModel) {
                ctx.currentModel->setVizTapsEnabled(true);
                try {
                    int max_frames = 12;
                    int max_side = 64;
                    if (ctx.modelConfig.contains("viz_taps_max_frames")) max_frames = ctx.modelConfig["viz_taps_max_frames"].get<int>();
                    if (ctx.modelConfig.contains("viz_taps_max_side")) max_side = ctx.modelConfig["viz_taps_max_side"].get<int>();
                    ctx.currentModel->setVizTapsLimits(max_frames, max_side);
                } catch (...) {
                }
            }

            ctx.addLog("Modèle chargé: " + std::string(path));
            lua_pushboolean(L, true);
        } else {
            lua_pushboolean(L, false);
            lua_pushstring(L, "Erreur lors du chargement");
            return 2;
        }
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
    
    return 1;
}

// ============================================================================
// New Serialization API
// ============================================================================

int LuaScripting::lua_saveCheckpoint(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    // Arguments: path, format (optionnel), options (optionnel)
    const char* path = luaL_checkstring(L, 1);
    const char* format_str = lua_isstring(L, 2) ? lua_tostring(L, 2) : "safetensors";
    
    using namespace Mimir::Serialization;
    
    // Parse format
    CheckpointFormat format = CheckpointFormat::SafeTensors;
    std::string fmt(format_str);
    if (fmt == "safetensors" || fmt == "st") {
        format = CheckpointFormat::SafeTensors;
    } else if (fmt == "raw_folder" || fmt == "raw" || fmt == "folder") {
        format = CheckpointFormat::RawFolder;
    } else if (fmt == "debug_json" || fmt == "debug" || fmt == "json") {
        format = CheckpointFormat::DebugJson;
    } else {
        lua_pushboolean(L, false);
        lua_pushstring(L, ("Format inconnu: " + fmt).c_str());
        return 2;
    }
    
    // Parse options from table (if provided)
    SaveOptions options;
    options.format = format;
    
    if (lua_istable(L, 3)) {
        lua_getfield(L, 3, "save_tokenizer");
        if (lua_isboolean(L, -1)) {
            options.save_tokenizer = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        lua_getfield(L, 3, "save_encoder");
        if (lua_isboolean(L, -1)) {
            options.save_encoder = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        lua_getfield(L, 3, "save_optimizer");
        if (lua_isboolean(L, -1)) {
            options.save_optimizer = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        lua_getfield(L, 3, "debug_max_values");
        if (lua_isnumber(L, -1)) {
            options.debug_max_values = lua_tointeger(L, -1);
        }
        lua_pop(L, 1);
        
        lua_getfield(L, 3, "include_git_info");
        if (lua_isboolean(L, -1)) {
            options.include_git_info = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        // Enhanced DebugJson options (v1.1.0)
        lua_getfield(L, 3, "include_gradients");
        if (lua_isboolean(L, -1)) {
            options.include_gradients = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        lua_getfield(L, 3, "include_optimizer_state");
        if (lua_isboolean(L, -1)) {
            options.include_optimizer_state = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        lua_getfield(L, 3, "max_values_per_tensor");
        if (lua_isnumber(L, -1)) {
            options.max_values_per_tensor = lua_tointeger(L, -1);
        }
        lua_pop(L, 1);
        
        lua_getfield(L, 3, "include_activations");
        if (lua_isboolean(L, -1)) {
            options.include_activations = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        lua_getfield(L, 3, "include_checksums");
        if (lua_isboolean(L, -1)) {
            options.include_checksums = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        lua_getfield(L, 3, "include_weight_deltas");
        if (lua_isboolean(L, -1)) {
            options.include_weight_deltas = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
    }
    
    // Save
    std::string error;
    bool success = save_checkpoint(*ctx.currentModel, path, options, &error);
    
    if (success) {
        ctx.addLog("Checkpoint sauvegardé: " + std::string(path) + " (format: " + fmt + ")");
        lua_pushboolean(L, true);
        return 1;
    } else {
        lua_pushboolean(L, false);
        lua_pushstring(L, error.c_str());
        return 2;
    }
}

int LuaScripting::lua_loadCheckpoint(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }

    // Arguments: path, format (optionnel), options (optionnel)
    const char* path = luaL_checkstring(L, 1);
    const char* format_str = lua_isstring(L, 2) ? lua_tostring(L, 2) : nullptr;

    // Parse format (or auto-detect)
    Mimir::Serialization::CheckpointFormat format = Mimir::Serialization::CheckpointFormat::SafeTensors;
    if (format_str == nullptr) {
        format = Mimir::Serialization::detect_format(path);
    } else {
        std::string fmt(format_str);
        if (fmt == "auto") {
            format = Mimir::Serialization::detect_format(path);
        } else if (fmt == "safetensors" || fmt == "st") {
            format = Mimir::Serialization::CheckpointFormat::SafeTensors;
        } else if (fmt == "raw_folder" || fmt == "raw" || fmt == "folder") {
            format = Mimir::Serialization::CheckpointFormat::RawFolder;
        } else if (fmt == "debug_json" || fmt == "debug" || fmt == "json") {
            format = Mimir::Serialization::CheckpointFormat::DebugJson;
        } else {
            lua_pushboolean(L, false);
            lua_pushstring(L, ("Format inconnu: " + fmt).c_str());
            return 2;
        }
    }

    // Parse options from table (if provided)
    Mimir::Serialization::LoadOptions options;
    options.format = format;

    int options_idx = (format_str != nullptr) ? 3 : 2;
    if (lua_istable(L, options_idx)) {
        lua_getfield(L, options_idx, "load_tokenizer");
        if (lua_isboolean(L, -1)) options.load_tokenizer = lua_toboolean(L, -1);
        lua_pop(L, 1);

        lua_getfield(L, options_idx, "load_encoder");
        if (lua_isboolean(L, -1)) options.load_encoder = lua_toboolean(L, -1);
        lua_pop(L, 1);

        lua_getfield(L, options_idx, "load_optimizer");
        if (lua_isboolean(L, -1)) options.load_optimizer = lua_toboolean(L, -1);
        lua_pop(L, 1);

        lua_getfield(L, options_idx, "strict_mode");
        if (lua_isboolean(L, -1)) options.strict_mode = lua_toboolean(L, -1);
        lua_pop(L, 1);

        lua_getfield(L, options_idx, "validate_checksums");
        if (lua_isboolean(L, -1)) options.validate_checksums = lua_toboolean(L, -1);
        lua_pop(L, 1);

        lua_getfield(L, options_idx, "mapping_json");
        if (lua_isstring(L, -1)) options.mapping_json = lua_tostring(L, -1);
        lua_pop(L, 1);

        lua_getfield(L, options_idx, "tensor_mapping_json");
        if (options.mapping_json.empty() && lua_isstring(L, -1)) options.mapping_json = lua_tostring(L, -1);
        lua_pop(L, 1);
    }

    // Load
    std::string error;
    bool success = Mimir::Serialization::load_checkpoint(*ctx.currentModel, path, options, &error);

    if (success) {
        // En mode reprise, la sérialisation hydrate les assets dans le modèle.
        // On resynchronise le contexte Lua pour que l'entraînement utilise bien
        // le tokenizer/encoder du checkpoint (et non un asset précédemment chargé).
        if (options.load_tokenizer && ctx.currentModel) {
            ctx.currentTokenizer = std::make_shared<Tokenizer>(ctx.currentModel->getTokenizer());
        }
        if (options.load_encoder && ctx.currentModel) {
            ctx.currentEncoder = std::make_shared<ConditioningEncoder>(ctx.currentModel->getEncoder());
        }

        ctx.addLog("Checkpoint chargé: " + std::string(path));
        lua_pushboolean(L, true);
        return 1;
    }

    lua_pushboolean(L, false);
    lua_pushstring(L, error.c_str());
    return 2;
}

int LuaScripting::lua_detectFormat(lua_State* L) {
    const char* path = luaL_checkstring(L, 1);
    
    using namespace Mimir::Serialization;
    
    CheckpointFormat format = detect_format(path);
    
    switch (format) {
        case CheckpointFormat::SafeTensors:
            lua_pushstring(L, "SAFETENSORS");
            break;
        case CheckpointFormat::RawFolder:
            lua_pushstring(L, "RAWFOLDER");
            break;
        case CheckpointFormat::DebugJson:
            lua_pushstring(L, "DEBUGJSON");
            break;
        default:
            lua_pushnil(L);
            lua_pushstring(L, "Format inconnu");
            return 2;
    }
    
    return 1;
}

int LuaScripting::lua_saveEnhancedDebugJson(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    // Argument 1: path
    const char* path = luaL_checkstring(L, 1);
    
    // Argument 2 (optionnel): options table
    using namespace Mimir::Serialization;
    DebugJsonOptions options;
    
    if (lua_istable(L, 2)) {
        // include_gradients
        lua_getfield(L, 2, "include_gradients");
        if (lua_isboolean(L, -1)) {
            options.include_gradients = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        // include_optimizer_state
        lua_getfield(L, 2, "include_optimizer_state");
        if (lua_isboolean(L, -1)) {
            options.include_optimizer_state = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        // max_values_per_tensor
        lua_getfield(L, 2, "max_values_per_tensor");
        if (lua_isnumber(L, -1)) {
            options.max_values_per_tensor = lua_tointeger(L, -1);
        }
        lua_pop(L, 1);
        
        // include_activations
        lua_getfield(L, 2, "include_activations");
        if (lua_isboolean(L, -1)) {
            options.include_activations = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        // include_checksums
        lua_getfield(L, 2, "include_checksums");
        if (lua_isboolean(L, -1)) {
            options.include_checksums = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        // include_weight_deltas
        lua_getfield(L, 2, "include_weight_deltas");
        if (lua_isboolean(L, -1)) {
            options.include_weight_deltas = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        // include_git_info
        lua_getfield(L, 2, "include_git_info");
        if (lua_isboolean(L, -1)) {
            options.include_git_info = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        // save_tokenizer
        lua_getfield(L, 2, "save_tokenizer");
        if (lua_isboolean(L, -1)) {
            options.save_tokenizer = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        // save_encoder
        lua_getfield(L, 2, "save_encoder");
        if (lua_isboolean(L, -1)) {
            options.save_encoder = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
    }
    
    // Save using enhanced debug JSON
    DebugJsonDump dumper;
    bool success = dumper.save_enhanced(path, *ctx.currentModel, options);
    
    if (success) {
        ctx.addLog("Enhanced debug JSON sauvegardé: " + std::string(path));
        lua_pushboolean(L, true);
        return 1;
    } else {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Erreur lors de la sauvegarde");
        return 2;
    }
}
