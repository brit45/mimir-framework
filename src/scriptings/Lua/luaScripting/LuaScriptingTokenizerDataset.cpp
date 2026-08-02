#include "LuaScripting.hpp"
#include "VizTextPayload.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "Serialization/Serialization.hpp"
#include "Serialization/DebugJsonDump.hpp"
#include "DType.hpp"
#include "AdvancedRAMManager.hpp"
#include "MemoryGuard.hpp"
#include "DynamicTensorAllocator.hpp"
#include "AsyncMonitor.hpp"
#include "Models/Diffusion/PonyXLDDPMModel.hpp"
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
#include <unordered_set>
#include <type_traits>
#include <utility>
// ============================================================================
// Tokenizer
// ============================================================================

int LuaScripting::lua_createTokenizer(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    int max_vocab = luaL_checkinteger(L, 1);
    
    ctx.currentTokenizer = std::make_shared<Tokenizer>(max_vocab);
    ctx.addLog("Tokenizer créé (vocab_max=" + std::to_string(max_vocab) + ")");
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_tokenize(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    const char* text = luaL_checkstring(L, 1);
    auto tokens = ctx.currentTokenizer->tokenizeBPE(text);
    
    // Retourner une table Lua avec les tokens
    lua_newtable(L);
    for (size_t i = 0; i < tokens.size(); ++i) {
        lua_pushinteger(L, tokens[i]);
        lua_rawseti(L, -2, i + 1);  // Indices Lua commencent à 1
    }
    
    return 1;
}

int LuaScripting::lua_detokenize(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    // Lire table de tokens
    luaL_checktype(L, 1, LUA_TTABLE);
    
    std::vector<int> tokens;
    lua_pushnil(L);
    while (lua_next(L, 1) != 0) {
        tokens.push_back(lua_tointeger(L, -1));
        lua_pop(L, 1);
    }
    
    std::string text = ctx.currentTokenizer->decode(tokens);
    lua_pushstring(L, text.c_str());
    
    return 1;
}

// ============================================================================
// Dataset
// ============================================================================

// ---------------------------------------------------------------------------
// Database (dataset loader builder with caching)
// ---------------------------------------------------------------------------

int LuaScripting::lua_databaseLoad(lua_State* L) {
    // Builder object returned to Lua. It can optionally capture initial args.
    // Example:
    //   Mimir.Database.load(dir, 64, 64, 1).cache("dataset_cache.json", 10240, true)
    // Also supports:
    //   Mimir.Database.load().cache(dir)

    const int top = lua_gettop(L);
    std::string dir;
    int target_w = 64;
    int target_h = 64;
    int min_modalities = 1;

    if (top >= 1 && lua_isstring(L, 1)) dir = lua_tostring(L, 1);
    if (top >= 2 && lua_isinteger(L, 2)) target_w = (int)lua_tointeger(L, 2);
    if (top >= 3 && lua_isinteger(L, 3)) target_h = (int)lua_tointeger(L, 3);
    if (top >= 4 && lua_isinteger(L, 4)) min_modalities = (int)lua_tointeger(L, 4);

    lua_newtable(L); // loader

    if (!dir.empty()) {
        lua_pushstring(L, dir.c_str());
        lua_setfield(L, -2, "dir");
    }
    lua_pushinteger(L, target_w);
    lua_setfield(L, -2, "target_w");
    lua_pushinteger(L, target_h);
    lua_setfield(L, -2, "target_h");
    lua_pushinteger(L, min_modalities);
    lua_setfield(L, -2, "min_modalities");

    // cache() method: closure captures the loader table as upvalue.
    lua_pushvalue(L, -1);
    lua_pushcclosure(L, lua_databaseLoad_cache, 1);
    lua_setfield(L, -2, "cache");

    return 1;
}

int LuaScripting::lua_databaseLoad_cache(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    // Upvalue: loader table
    lua_pushvalue(L, lua_upvalueindex(1));
    const int loader_idx = lua_gettop(L);

    // Allow both dot-call and colon-call.
    int argi = 1;
    if (lua_gettop(L) >= 1 && lua_istable(L, 1)) {
        // colon call: self at arg1
        argi = 2;
    }

    auto get_loader_string = [&](const char* key) -> std::string {
        lua_getfield(L, loader_idx, key);
        std::string v;
        if (lua_isstring(L, -1)) v = lua_tostring(L, -1);
        lua_pop(L, 1);
        return v;
    };
    auto get_loader_int = [&](const char* key, int def) -> int {
        lua_getfield(L, loader_idx, key);
        int v = def;
        if (lua_isinteger(L, -1)) v = (int)lua_tointeger(L, -1);
        lua_pop(L, 1);
        return v;
    };

    std::string dataset_dir = get_loader_string("dir");
    int target_w = get_loader_int("target_w", 64);
    int target_h = get_loader_int("target_h", 64);
    int min_modalities = get_loader_int("min_modalities", 1);

    // Overrides from args (optional)
    // Signature (flexible):
    //   cache([dir], [target_w], [target_h], [min_modalities], [cache_path], [max_ram_mb], [lazy_loading])
    if (lua_gettop(L) >= argi && lua_isstring(L, argi)) {
        dataset_dir = lua_tostring(L, argi);
        argi++;
    }
    if (lua_gettop(L) >= argi && lua_isinteger(L, argi)) {
        target_w = (int)lua_tointeger(L, argi);
        argi++;
    }
    if (lua_gettop(L) >= argi && lua_isinteger(L, argi)) {
        target_h = (int)lua_tointeger(L, argi);
        argi++;
    }
    if (lua_gettop(L) >= argi && lua_isinteger(L, argi)) {
        min_modalities = (int)lua_tointeger(L, argi);
        argi++;
    }

    std::string cache_path = "dataset_cache.json";
    if (lua_gettop(L) >= argi && lua_isstring(L, argi)) {
        cache_path = lua_tostring(L, argi);
        argi++;
    }

    int max_ram_mb = 10240;
    if (lua_gettop(L) >= argi && lua_isinteger(L, argi)) {
        max_ram_mb = (int)lua_tointeger(L, argi);
        argi++;
    }

    bool lazy_loading = true;
    if (lua_gettop(L) >= argi && lua_isboolean(L, argi)) {
        lazy_loading = lua_toboolean(L, argi);
        argi++;
    }

    // Fallback: if no dir was provided, try ctx.currentConfig.dataset.dir
    if (dataset_dir.empty()) {
        try {
            if (ctx.currentConfig.contains("dataset") && ctx.currentConfig["dataset"].contains("dir")) {
                dataset_dir = ctx.currentConfig["dataset"]["dir"].get<std::string>();
            }
        } catch (...) {
        }
    }

    if (dataset_dir.empty()) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Database.load().cache(): dataset_dir manquant (passez un chemin ou chargez un dataset d'abord)");
        return 2;
    }

    try {
        fs::path dataset_path(dataset_dir);
        if (!fs::exists(dataset_path)) {
            lua_pushboolean(L, false);
            lua_pushstring(L, "Le dossier dataset n'existe pas");
            return 2;
        }

        ctx.addLog("Chargement dataset (cached): " + dataset_dir);

        std::vector<DatasetItem> items = loadDatasetCached(
            dataset_dir,
            target_w,
            target_h,
            min_modalities,
            cache_path,
            (size_t)std::max(0, max_ram_mb),
            lazy_loading
        );

        if (items.empty()) {
            ctx.addLog("⚠️  Attention: Dataset vide (cached)");
        } else {
            ctx.addLog("✓ " + std::to_string(items.size()) + " items chargés (cached)");
        }

        ctx.currentDataset = std::move(items);

        if (!ctx.currentConfig.contains("dataset")) {
            ctx.currentConfig["dataset"] = json::object();
        }
        ctx.currentConfig["dataset"]["dir"] = dataset_dir;
        ctx.currentConfig["dataset"]["target_w"] = target_w;
        ctx.currentConfig["dataset"]["target_h"] = target_h;
        ctx.currentConfig["dataset"]["min_modalities"] = min_modalities;
        ctx.currentConfig["dataset"]["num_items"] = ctx.currentDataset.size();
        ctx.currentConfig["dataset"]["cache_path"] = cache_path;
        ctx.currentConfig["dataset"]["max_ram_mb"] = max_ram_mb;
        ctx.currentConfig["dataset"]["lazy_loading"] = lazy_loading;
        ctx.currentConfig["dataset"]["loader"] = "cached";

        lua_pushboolean(L, true);
        lua_pushinteger(L, ctx.currentDataset.size());
        return 2;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    } catch (...) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "unknown error");
        return 2;
    }
}

int LuaScripting::lua_loadDataset(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    const char* dataset_dir = luaL_checkstring(L, 1);

    // Options:
    //  (dir, target_w, target_h, min_modalities, use_cache, cache_path, max_ram_mb, lazy_loading)
    const int top = lua_gettop(L);
    int target_w = (top >= 2 && lua_isinteger(L, 2)) ? (int)lua_tointeger(L, 2) : 64;
    int target_h = (top >= 3 && lua_isinteger(L, 3)) ? (int)lua_tointeger(L, 3) : 64;
    int min_modalities = (top >= 4 && lua_isinteger(L, 4)) ? (int)lua_tointeger(L, 4) : 1;

    const bool use_cache = (top >= 5 && lua_isboolean(L, 5)) ? (bool)lua_toboolean(L, 5) : false;
    std::string cache_path = (top >= 6 && lua_isstring(L, 6)) ? std::string(lua_tostring(L, 6)) : std::string("dataset_cache.json");
    int max_ram_mb = (top >= 7 && lua_isinteger(L, 7)) ? (int)lua_tointeger(L, 7) : 10240;
    bool lazy_loading = (top >= 8 && lua_isboolean(L, 8)) ? (bool)lua_toboolean(L, 8) : true;

    ctx.addLog(std::string("Chargement dataset") + (use_cache ? " (cached): " : ": ") + std::string(dataset_dir));
    
    try {
        fs::path dataset_path(dataset_dir);
        
        if (!fs::exists(dataset_path)) {
            lua_pushboolean(L, false);
            lua_pushstring(L, "Le dossier dataset n'existe pas");
            return 2;
        }
        
        // Charger les items du dataset
        std::vector<DatasetItem> items;
        if (use_cache) {
            items = loadDatasetCached(
                dataset_dir,
                target_w,
                target_h,
                min_modalities,
                cache_path,
                (size_t)std::max(0, max_ram_mb),
                lazy_loading
            );
        } else {
            items = loadDataset(dataset_dir, target_w, target_h, min_modalities);
        }
        
        if (items.empty()) {
            ctx.addLog(std::string("⚠️  Attention: Dataset vide") + (use_cache ? " (cached)" : ""));
        } else {
            ctx.addLog("✓ " + std::to_string(items.size()) + " items chargés" + (use_cache ? " (cached)" : ""));
        }
        
        // Stocker le dataset dans le contexte
        ctx.currentDataset = std::move(items);
        
        if (!ctx.currentConfig.contains("dataset")) {
            ctx.currentConfig["dataset"] = json::object();
        }
        ctx.currentConfig["dataset"]["dir"] = dataset_dir;
        ctx.currentConfig["dataset"]["target_w"] = target_w;
        ctx.currentConfig["dataset"]["target_h"] = target_h;
        ctx.currentConfig["dataset"]["min_modalities"] = min_modalities;
        ctx.currentConfig["dataset"]["num_items"] = ctx.currentDataset.size();

        ctx.currentConfig["dataset"]["use_cache"] = use_cache;
        if (use_cache) {
            ctx.currentConfig["dataset"]["cache_path"] = cache_path;
            ctx.currentConfig["dataset"]["max_ram_mb"] = max_ram_mb;
            ctx.currentConfig["dataset"]["lazy_loading"] = lazy_loading;
            ctx.currentConfig["dataset"]["loader"] = "cached";
        } else {
            if (ctx.currentConfig["dataset"].contains("cache_path")) ctx.currentConfig["dataset"].erase("cache_path");
            if (ctx.currentConfig["dataset"].contains("max_ram_mb")) ctx.currentConfig["dataset"].erase("max_ram_mb");
            if (ctx.currentConfig["dataset"].contains("lazy_loading")) ctx.currentConfig["dataset"].erase("lazy_loading");
            ctx.currentConfig["dataset"]["loader"] = "default";
        }
        
        lua_pushboolean(L, true);
        lua_pushinteger(L, ctx.currentDataset.size());
        return 2;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_getDataset(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    int index = luaL_checkinteger(L, 1);
    
    if (ctx.currentDataset.empty()) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun dataset chargé. Utilisez dataset.load() d'abord.");
        return 2;
    }
    
    if (index < 1 || index > (int)ctx.currentDataset.size()) {
        lua_pushnil(L);
        lua_pushstring(L, "Index hors limites");
        return 2;
    }
    
    try {
        auto& item = ctx.currentDataset[index - 1]; // Lua est 1-indexed
        
        // Créer une table Lua avec les informations de l'item
        lua_newtable(L);
        
        // Ajouter les chemins de fichiers
        if (!item.text_file.empty()) {
            lua_pushstring(L, item.text_file.c_str());
            lua_setfield(L, -2, "text_file");
        }
        if (!item.image_file.empty()) {
            lua_pushstring(L, item.image_file.c_str());
            lua_setfield(L, -2, "image_file");
        }
        if (!item.audio_file.empty()) {
            lua_pushstring(L, item.audio_file.c_str());
            lua_setfield(L, -2, "audio_file");
        }
        if (!item.video_file.empty()) {
            lua_pushstring(L, item.video_file.c_str());
            lua_setfield(L, -2, "video_file");
        }
        
        // Déduire les dimensions cible du dataset (si connues)
        int target_w = 64;
        int target_h = 64;
        try {
            if (ctx.currentConfig.contains("dataset")) {
                if (ctx.currentConfig["dataset"].contains("target_w")) target_w = ctx.currentConfig["dataset"]["target_w"].get<int>();
                if (ctx.currentConfig["dataset"].contains("target_h")) target_h = ctx.currentConfig["dataset"]["target_h"].get<int>();
            }
        } catch (...) {
        }
        
        // Charger et ajouter le contenu texte si présent
        if (!item.text_file.empty() && !item.text.has_value()) {
            item.loadText();
        }
        if (!item.text_file.empty() && item.text.has_value()) {
            lua_pushstring(L, item.text.value().c_str());
            lua_setfield(L, -2, "text");
        }

        // Charger et retourner l'image en bytes RGB u8 (utile pour diffusion).
        if (!item.image_file.empty()) {
            item.loadImageRGB(target_w, target_h);
            if (item.img_loaded) {
                lua_newtable(L);
                for (size_t i = 0; i < item.img.size(); ++i) {
                    lua_pushinteger(L, (lua_Integer)item.img[i]);
                    lua_rawseti(L, -2, (int)i + 1);
                }
                lua_setfield(L, -2, "image");
            }

            lua_pushinteger(L, item.w > 0 ? item.w : target_w);
            lua_setfield(L, -2, "width");
            lua_pushinteger(L, item.h > 0 ? item.h : target_h);
            lua_setfield(L, -2, "height");
            lua_pushinteger(L, item.img_c > 0 ? item.img_c : 3);
            lua_setfield(L, -2, "channels");
        }
        
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_readImageRGBU8(lua_State* L) {
    // Args:
    // 1 path (string)
    // 2 target_w (int, optional, default 256)
    // 3 target_h (int, optional, default 256)
    const char* path = luaL_checkstring(L, 1);
    const int top = lua_gettop(L);
    const int target_w = (top >= 2 && lua_isinteger(L, 2)) ? (int)lua_tointeger(L, 2) : 256;
    const int target_h = (top >= 3 && lua_isinteger(L, 3)) ? (int)lua_tointeger(L, 3) : 256;

    if (!path || std::string(path).empty()) {
        lua_pushnil(L);
        lua_pushstring(L, "Chemin image vide");
        return 2;
    }
    if (target_w <= 0 || target_h <= 0) {
        lua_pushnil(L);
        lua_pushstring(L, "Dimensions invalides");
        return 2;
    }

    try {
        int w_img = 0, h_img = 0, c = 0;
        unsigned char* data = stbi_load(path, &w_img, &h_img, &c, 3);
        if (!data) {
            lua_pushnil(L);
            lua_pushfstring(L, "Impossible de charger l'image: %s", path);
            return 2;
        }

        std::vector<unsigned char> src((size_t)w_img * (size_t)h_img * 3);
        std::memcpy(src.data(), data, src.size());
        stbi_image_free(data);

        std::vector<uint8_t> dst((size_t)target_w * (size_t)target_h * 3);
        resizeBicubicRGB_SRGBLinear(src.data(), w_img, h_img, dst.data(), target_w, target_h);

        lua_newtable(L);

        lua_newtable(L);
        for (size_t i = 0; i < dst.size(); ++i) {
            lua_pushinteger(L, (lua_Integer)dst[i]);
            lua_rawseti(L, -2, (int)i + 1);
        }
        lua_setfield(L, -2, "image");

        lua_pushinteger(L, target_w);
        lua_setfield(L, -2, "width");
        lua_pushinteger(L, target_h);
        lua_setfield(L, -2, "height");
        lua_pushinteger(L, 3);
        lua_setfield(L, -2, "channels");

        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    } catch (...) {
        lua_pushnil(L);
        lua_pushstring(L, "unknown error");
        return 2;
    }
}

// ============================================================================
// PonyXL helpers
// ============================================================================

int LuaScripting::lua_ponyxlDdpmTrainStep(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    static int s_ponyxl_global_step = 0;

    if (!ctx.currentModel) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }

    // Args:
    // 1 prompt (string)
    // 2 image (table of bytes u8)
    // 3 w (int)
    // 4 h (int)
    // 5 lr (number)
    // 6 optimizer (string, optional): "adamw" | "adam" | "sgd"
    // 7 meta (table, optional): {epoch,total_epochs,batch,total_batches,avg_loss,dataset_i}
    const char* prompt = luaL_checkstring(L, 1);
    luaL_checktype(L, 2, LUA_TTABLE);
    const int w = (int)luaL_checkinteger(L, 3);
    const int h = (int)luaL_checkinteger(L, 4);
    const float lr = (float)luaL_checknumber(L, 5);
    const char* opt_type = luaL_optstring(L, 6, "adamw");

    int meta_epoch = 0;
    int meta_total_epochs = 0;
    int meta_batch = 0;
    int meta_total_batches = 0;
    float meta_avg_loss = 0.0f;
    int meta_dataset_i = 0;

    const int nargs = lua_gettop(L);
    if (nargs >= 7 && lua_istable(L, 7)) {
        lua_getfield(L, 7, "epoch");
        if (lua_isnumber(L, -1)) meta_epoch = (int)lua_tointeger(L, -1);
        lua_pop(L, 1);

        lua_getfield(L, 7, "total_epochs");
        if (lua_isnumber(L, -1)) meta_total_epochs = (int)lua_tointeger(L, -1);
        lua_pop(L, 1);

        lua_getfield(L, 7, "batch");
        if (lua_isnumber(L, -1)) meta_batch = (int)lua_tointeger(L, -1);
        lua_pop(L, 1);

        lua_getfield(L, 7, "total_batches");
        if (lua_isnumber(L, -1)) meta_total_batches = (int)lua_tointeger(L, -1);
        lua_pop(L, 1);

        lua_getfield(L, 7, "avg_loss");
        if (lua_isnumber(L, -1)) meta_avg_loss = (float)lua_tonumber(L, -1);
        lua_pop(L, 1);

        lua_getfield(L, 7, "dataset_i");
        if (lua_isnumber(L, -1)) meta_dataset_i = (int)lua_tointeger(L, -1);
        lua_pop(L, 1);
    }

    const size_t n = (size_t)lua_rawlen(L, 2);
    if (n == 0) {
        lua_pushnil(L);
        lua_pushstring(L, "Image vide");
        return 2;
    }

    std::vector<uint8_t> rgb;
    rgb.resize(n);
    for (size_t i = 0; i < n; ++i) {
        lua_rawgeti(L, 2, (lua_Integer)i + 1);
        int v = 0;
        if (lua_isinteger(L, -1)) v = (int)lua_tointeger(L, -1);
        else if (lua_isnumber(L, -1)) v = (int)lua_tonumber(L, -1);
        lua_pop(L, 1);
        v = std::clamp(v, 0, 255);
        rgb[i] = (uint8_t)v;
    }

    // Validation stricte: le modèle attend RGB packed (w*h*3)
    if (w <= 0 || h <= 0) {
        lua_pushnil(L);
        lua_pushstring(L, "w/h invalides (doivent être > 0)");
        return 2;
    }
    const size_t expected = (size_t)w * (size_t)h * 3ULL;
    if (rgb.size() != expected) {
        const std::string msg = "Taille image invalide: got=" + std::to_string(rgb.size()) +
                                " expected=" + std::to_string(expected);
        lua_pushnil(L);
        lua_pushstring(L, msg.c_str());
        return 2;
    }

    try {
        auto* pony = dynamic_cast<PonyXLDDPMModel*>(ctx.currentModel.get());
        if (!pony) {
            lua_pushnil(L);
            lua_pushstring(L, "Le modèle courant n'est pas un PonyXLDDPMModel (type=ponyxl_ddpm attendu)");
            return 2;
        }

        // Optimizer persistant: stocké dans le modèle pour compat avec save/load.
        if (!ctx.currentModel->getSerializedOptimizer()) {
            Optimizer opt;
            opt.initial_lr = lr;

            std::string t = opt_type ? std::string(opt_type) : std::string("adamw");
            std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
            if (t == "sgd") opt.type = OptimizerType::SGD;
            else if (t == "adam") opt.type = OptimizerType::ADAM;
            else opt.type = OptimizerType::ADAMW;

            // Paramètres depuis modelConfig si présents
            try {
                if (ctx.modelConfig.contains("beta1")) opt.beta1 = ctx.modelConfig["beta1"].get<float>();
                if (ctx.modelConfig.contains("beta2")) opt.beta2 = ctx.modelConfig["beta2"].get<float>();
                if (ctx.modelConfig.contains("epsilon")) opt.eps = ctx.modelConfig["epsilon"].get<float>();
                if (ctx.modelConfig.contains("weight_decay")) opt.weight_decay = ctx.modelConfig["weight_decay"].get<float>();
            } catch (...) {
            }

            ctx.currentModel->setSerializedOptimizer(opt);
        } else {
            // S'assurer que lr reflète l'argument
            if (Optimizer* saved = ctx.currentModel->getMutableSerializedOptimizer()) {
                saved->initial_lr = lr;
            }
        }

        Optimizer* opt = ctx.currentModel->getMutableSerializedOptimizer();
        if (!opt) {
            throw std::runtime_error("optimizer state unavailable");
        }

        // If Viz is active, ensure taps are enabled before running the step so the model
        // can emit custom previews (noise/denoise, recon, etc.) during the step.
        const bool viz_active = (ctx.asyncMonitor && ctx.asyncMonitor->getViz() != nullptr);
        if (viz_active && ctx.currentModel) {
            if (!ctx.currentModel->isVizTapsEnabled()) {
                ctx.currentModel->setVizTapsEnabled(true);
            }
            // Optional per-model limits from config.json / modelConfig.
            try {
                // PonyXL DDPM emits both preview frames (x0/eps/metrics) and per-layer state tiles.
                // A tiny default like 12 makes the Blocks/Layers panel effectively unusable.
                int max_frames = 256;
                int max_side = 64;
                if (ctx.modelConfig.contains("viz_taps_max_frames")) max_frames = ctx.modelConfig["viz_taps_max_frames"].get<int>();
                if (ctx.modelConfig.contains("viz_taps_max_side")) max_side = ctx.modelConfig["viz_taps_max_side"].get<int>();
                ctx.currentModel->setVizTapsLimits(max_frames, max_side);
            } catch (...) {
            }
        }

        const PonyXLDDPMModel::StepStats st = pony->trainStepSdxlLatentDiffusion(prompt, rgb, w, h, *opt, lr);

        // Best-effort: si AsyncMonitor/Viz est actif, pousser l'input et les métriques.
        if (ctx.asyncMonitor) {
            s_ponyxl_global_step += 1;

            static std::chrono::steady_clock::time_point s_last_pony_metrics_ts;
            static bool s_has_last_pony_metrics_ts = false;

            AsyncMonitor::Metrics m;
            m.epoch = meta_epoch;
            m.total_epochs = meta_total_epochs;
            m.batch = (meta_batch > 0) ? meta_batch : s_ponyxl_global_step;
            m.total_batches = meta_total_batches;
            m.loss = st.loss;
            m.avg_loss = (meta_avg_loss > 0.0f) ? meta_avg_loss : st.loss;
            m.lr = lr;
            m.mse = st.loss;
            m.kl = st.kl_divergence;
            m.wass = st.wasserstein;
            m.ent = st.entropy_diff;
            m.mom = st.moment_mismatch;
            m.spat = st.spatial_coherence;
            m.temp = st.temporal_consistency;
            m.timestep = st.timestep;
            m.grad_norm = st.grad_norm;
            m.grad_max = st.grad_max_abs;
            m.kl_beta_effective = st.kl_beta_effective;
            m.params = ctx.currentModel ? ctx.currentModel->totalParamCount() : 0;
            m.opt_type = (int)opt->type;
            m.opt_step = opt->step;
            m.opt_beta1 = opt->beta1;
            m.opt_beta2 = opt->beta2;
            m.opt_eps = opt->eps;
            m.opt_weight_decay = opt->weight_decay;

            {
                auto& guard = MemoryGuard::instance();
                m.memory_mb = guard.getCurrentBytes() / 1024 / 1024;
            }

            {
                const auto now = std::chrono::steady_clock::now();
                if (s_has_last_pony_metrics_ts) {
                    const auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - s_last_pony_metrics_ts).count();
                    if (dt_ms > 0 && dt_ms < 60LL * 60LL * 1000LL) {
                        m.batch_time_ms = (int)dt_ms;
                        m.bps = 1000.0f / (float)m.batch_time_ms;
                    }
                }
                s_last_pony_metrics_ts = now;
                s_has_last_pony_metrics_ts = true;
            }
            ctx.asyncMonitor->updateMetrics(m);

            if (ctx.asyncMonitor->getViz() != nullptr) {
                const bool viz_resync = [&]() -> bool {
                    try {
                        auto viz = ctx.asyncMonitor->getViz();
                        if (!viz) return false;
                        return viz->consumeResyncRequested();
                    } catch (...) {
                        return false;
                    }
                }();

                // Afficher l'input dataset (image + prompt) dans la Viz.
                std::string label = "ponyxl_ddpm/input/dataset/rgb";
                if (meta_dataset_i > 0) {
                    label += "/i=" + std::to_string(meta_dataset_i);
                }
                const auto viz_payload = VizTextPayload::buildDatasetTextPayload(
                    ctx.currentModel.get(),
                    ctx.currentTokenizer.get(),
                    std::string(prompt ? prompt : ""),
                    nullptr,
                    -1,
                    true,
                    true,
                    true
                );

                ctx.asyncMonitor->setDatasetSample(
                    rgb,
                    w,
                    h,
                    3,
                    label,
                    viz_payload.prompt,
                    viz_payload.tags,
                    viz_payload.tokens,
                    viz_payload.encoding
                );

                // Pousser les viz taps (frames) générés par le modèle pendant la step.
                // Sans ça, les recon/noise previews ajoutés via Model::addVizTapFrame ne s'affichent jamais.
                if (ctx.currentModel) {
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
                    // Important UX: si le dataset change mais que le modèle n'a pas émis de taps,
                    // ne pas conserver les frames précédentes (sinon rendu stale/mélangé).
                    ctx.asyncMonitor->setLayerBlockImages(frames);
                }

                // Si l'utilisateur a demandé une resynchronisation (touche R), forcer une preview
                // de reconstruction (best-effort) afin de rafraîchir la vue avec l'état courant.
                if (viz_resync) {
                    try {
                        // seed/t choisis pour être stables et peu coûteux à interpréter.
                        const int seed = 12345;
                        const int ddpm_step = -1;
                        auto prev = pony->reconstructPreviewSdxlLatentDiffusion(
                            std::string(prompt ? prompt : ""),
                            rgb,
                            w,
                            h,
                            256,
                            seed,
                            ddpm_step
                        );
                        if (!prev.pixels.empty() && prev.w > 0 && prev.h > 0) {
                            std::string tag = "RESYNC recon";
                            if (meta_dataset_i > 0) tag += " i=" + std::to_string(meta_dataset_i);
                            ctx.asyncMonitor->addImage(prev.pixels, prev.w, prev.h, prev.channels, tag);
                        }
                    } catch (...) {
                        // ignore
                    }
                }
            }
        }

        lua_newtable(L);
        lua_pushnumber(L, st.loss);
        lua_setfield(L, -2, "loss");
        lua_pushnumber(L, st.grad_norm);
        lua_setfield(L, -2, "grad_norm");
        lua_pushnumber(L, st.grad_max_abs);
        lua_setfield(L, -2, "grad_max_abs");
        lua_pushnumber(L, st.timestep);
        lua_setfield(L, -2, "timestep");
        lua_pushnumber(L, st.kl_divergence);
        lua_setfield(L, -2, "kl_divergence");
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_ponyxlDdpmValidateStep(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    if (!ctx.currentModel) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }

    // Args:
    // 1 prompt (string)
    // 2 wrong_prompt (string, optional)
    // 3 image (table of bytes u8)
    // 4 w (int)
    // 5 h (int)
    // 6 seed (int, optional) : graine pour eps (validation déterministe)
    // 7 ddpm_step (int, optional) : timestep t dans [0..T-1], -1 => T/2
    const char* prompt = luaL_checkstring(L, 1);
    const char* wrong_prompt = luaL_optstring(L, 2, "");
    luaL_checktype(L, 3, LUA_TTABLE);
    const int w = (int)luaL_checkinteger(L, 4);
    const int h = (int)luaL_checkinteger(L, 5);
    const int seed = (int)luaL_optinteger(L, 6, 12345);
    const int ddpm_step = (int)luaL_optinteger(L, 7, -1);

    const size_t n = (size_t)lua_rawlen(L, 3);
    if (n == 0) {
        lua_pushnil(L);
        lua_pushstring(L, "Image vide");
        return 2;
    }

    std::vector<uint8_t> rgb;
    rgb.resize(n);
    for (size_t i = 0; i < n; ++i) {
        lua_rawgeti(L, 3, (lua_Integer)i + 1);
        int v = 0;
        if (lua_isinteger(L, -1)) v = (int)lua_tointeger(L, -1);
        else if (lua_isnumber(L, -1)) v = (int)lua_tonumber(L, -1);
        lua_pop(L, 1);
        v = std::clamp(v, 0, 255);
        rgb[i] = (uint8_t)v;
    }

    if (w <= 0 || h <= 0) {
        lua_pushnil(L);
        lua_pushstring(L, "w/h invalides (doivent être > 0)");
        return 2;
    }
    const size_t expected = (size_t)w * (size_t)h * 3ULL;
    if (rgb.size() != expected) {
        const std::string msg = "Taille image invalide: got=" + std::to_string(rgb.size()) +
                                " expected=" + std::to_string(expected);
        lua_pushnil(L);
        lua_pushstring(L, msg.c_str());
        return 2;
    }

    try {
        auto* pony = dynamic_cast<PonyXLDDPMModel*>(ctx.currentModel.get());
        if (!pony) {
            lua_pushnil(L);
            lua_pushstring(L, "Le modèle courant n'est pas un PonyXLDDPMModel (type=ponyxl_ddpm attendu)");
            return 2;
        }

        const PonyXLDDPMModel::ValStats st = pony->validateStepSdxlLatentDiffusion(
            prompt ? std::string(prompt) : std::string(),
            wrong_prompt ? std::string(wrong_prompt) : std::string(),
            rgb,
            w,
            h,
            seed,
            ddpm_step
        );

        lua_newtable(L);
        lua_pushnumber(L, (lua_Number)st.eps_mse);
        lua_setfield(L, -2, "eps_mse");
        lua_pushnumber(L, (lua_Number)st.x0_mse);
        lua_setfield(L, -2, "x0_mse");
        lua_pushnumber(L, (lua_Number)st.img_mse);
        lua_setfield(L, -2, "img_mse");
        lua_pushnumber(L, (lua_Number)st.eps_mse_wrong);
        lua_setfield(L, -2, "eps_mse_wrong");
        lua_pushnumber(L, (lua_Number)st.assoc_margin);
        lua_setfield(L, -2, "assoc_margin");
        lua_pushnumber(L, (lua_Number)st.t_norm);
        lua_setfield(L, -2, "t_norm");
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_ponyxlDdpmSetVaeScale(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    if (!ctx.currentModel) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }

    const float s = (float)luaL_checknumber(L, 1);
    try {
        auto* pony = dynamic_cast<PonyXLDDPMModel*>(ctx.currentModel.get());
        if (!pony) {
            lua_pushnil(L);
            lua_pushstring(L, "Modèle courant n'est pas ponyxl_ddpm");
            return 2;
        }
        pony->setVaeScale(s);
        lua_pushboolean(L, 1);
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_ponyxlDdpmGetVaeScale(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    if (!ctx.currentModel) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }

    try {
        auto* pony = dynamic_cast<PonyXLDDPMModel*>(ctx.currentModel.get());
        if (!pony) {
            lua_pushnil(L);
            lua_pushstring(L, "Modèle courant n'est pas ponyxl_ddpm");
            return 2;
        }
        lua_pushnumber(L, (lua_Number)pony->getConfig().vae_scale);
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_ponyxlDdpmVaeMuMoments(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    if (!ctx.currentModel) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }

    // Args:
    // 1 image (table of bytes u8)
    // 2 w (int)
    // 3 h (int)
    luaL_checktype(L, 1, LUA_TTABLE);
    const int w = (int)luaL_checkinteger(L, 2);
    const int h = (int)luaL_checkinteger(L, 3);
    const size_t nbytes = (size_t)lua_rawlen(L, 1);
    if (nbytes == 0) {
        lua_pushnil(L);
        lua_pushstring(L, "Image vide");
        return 2;
    }

    std::vector<uint8_t> rgb;
    rgb.resize(nbytes);
    for (size_t i = 0; i < nbytes; ++i) {
        lua_rawgeti(L, 1, (lua_Integer)(i + 1));
        const int v = (int)luaL_checkinteger(L, -1);
        lua_pop(L, 1);
        rgb[i] = (uint8_t)std::clamp(v, 0, 255);
    }

    try {
        auto* pony = dynamic_cast<PonyXLDDPMModel*>(ctx.currentModel.get());
        if (!pony) {
            lua_pushnil(L);
            lua_pushstring(L, "Modèle courant n'est pas ponyxl_ddpm");
            return 2;
        }

        double sum = 0.0;
        double sumsq = 0.0;
        size_t n = 0;
        pony->accumulateVaeMuMoments(rgb, w, h, sum, sumsq, n);

        lua_newtable(L);
        lua_pushnumber(L, (lua_Number)sum);
        lua_setfield(L, -2, "sum");
        lua_pushnumber(L, (lua_Number)sumsq);
        lua_setfield(L, -2, "sumsq");
        lua_pushinteger(L, (lua_Integer)n);
        lua_setfield(L, -2, "n");
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_ponyxlDdpmVizReconstructStep(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    if (!ctx.currentModel) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    if (!ctx.asyncMonitor || ctx.asyncMonitor->getViz() == nullptr) {
        lua_pushnil(L);
        lua_pushstring(L, "Viz non actif (démarrer avec --viz)");
        return 2;
    }

    // Args:
    // 1 prompt (string)
    // 2 image (table of bytes u8)
    // 3 w (int)
    // 4 h (int)
    // 5 label (string, optional)
    // 6 max_side (int, optional)
    // 7 seed (int, optional)
    // 8 ddpm_step (int, optional)
    const char* prompt = luaL_checkstring(L, 1);
    luaL_checktype(L, 2, LUA_TTABLE);
    const int w = (int)luaL_checkinteger(L, 3);
    const int h = (int)luaL_checkinteger(L, 4);
    const char* label = luaL_optstring(L, 5, "VAL recon");
    const int max_side = (int)luaL_optinteger(L, 6, 256);
    const int seed = (int)luaL_optinteger(L, 7, 12345);
    const int ddpm_step = (int)luaL_optinteger(L, 8, -1);

    const size_t n = (size_t)lua_rawlen(L, 2);
    if (n == 0) {
        lua_pushnil(L);
        lua_pushstring(L, "Image vide");
        return 2;
    }

    std::vector<uint8_t> rgb;
    rgb.resize(n);
    for (size_t i = 0; i < n; ++i) {
        lua_rawgeti(L, 2, (lua_Integer)i + 1);
        int v = 0;
        if (lua_isinteger(L, -1)) v = (int)lua_tointeger(L, -1);
        else if (lua_isnumber(L, -1)) v = (int)lua_tonumber(L, -1);
        lua_pop(L, 1);
        v = std::clamp(v, 0, 255);
        rgb[i] = (uint8_t)v;
    }

    if (w <= 0 || h <= 0) {
        lua_pushnil(L);
        lua_pushstring(L, "w/h invalides (doivent être > 0)");
        return 2;
    }
    const size_t expected = (size_t)w * (size_t)h * 3ULL;
    if (rgb.size() != expected) {
        const std::string msg = "Taille image invalide: got=" + std::to_string(rgb.size()) +
                                " expected=" + std::to_string(expected);
        lua_pushnil(L);
        lua_pushstring(L, msg.c_str());
        return 2;
    }

    try {
        auto* pony = dynamic_cast<PonyXLDDPMModel*>(ctx.currentModel.get());
        if (!pony) {
            lua_pushnil(L);
            lua_pushstring(L, "Le modèle courant n'est pas un PonyXLDDPMModel (type=ponyxl_ddpm attendu)");
            return 2;
        }

        const PonyXLDDPMModel::ReconPreview rp = pony->reconstructPreviewSdxlLatentDiffusion(
            prompt ? std::string(prompt) : std::string(),
            rgb,
            w,
            h,
            max_side,
            seed,
            ddpm_step
        );

        if (rp.pixels.empty() || rp.w <= 0 || rp.h <= 0) {
            lua_pushnil(L);
            lua_pushstring(L, "Reconstruction preview vide");
            return 2;
        }

        ctx.asyncMonitor->addImage(rp.pixels, rp.w, rp.h, rp.channels, std::string(label ? label : "VAL recon"));
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_ponyxlDdpmText2Img(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    if (!ctx.currentModel) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }

    // Args:
    // 1 prompt (string)
    // 2 seed (int, optional)
    // 3 sample_steps (int, optional)
    // 4 guidance_scale (number, optional)
    // 5 max_side (int, optional)
    const char* prompt = luaL_checkstring(L, 1);
    const int seed = (int)luaL_optinteger(L, 2, 12345);
    const int sample_steps = (int)luaL_optinteger(L, 3, 50);
    const float guidance_scale = (float)luaL_optnumber(L, 4, 1.0);
    const int max_side = (int)luaL_optinteger(L, 5, 0);

    try {
        auto* pony = dynamic_cast<PonyXLDDPMModel*>(ctx.currentModel.get());
        if (!pony) {
            lua_pushnil(L);
            lua_pushstring(L, "Le modèle courant n'est pas un PonyXLDDPMModel (type=ponyxl_ddpm attendu)");
            return 2;
        }

        const PonyXLDDPMModel::ReconPreview rp = pony->text2imgSdxlLatentDiffusion(
            prompt ? std::string(prompt) : std::string(),
            seed,
            sample_steps,
            guidance_scale,
            max_side
        );

        if (rp.pixels.empty() || rp.w <= 0 || rp.h <= 0) {
            lua_pushnil(L);
            lua_pushstring(L, "text2img a retourné une image vide");
            return 2;
        }

        lua_createtable(L, (int)rp.pixels.size(), 0);
        for (size_t i = 0; i < rp.pixels.size(); ++i) {
            lua_pushinteger(L, (lua_Integer)rp.pixels[i]);
            lua_rawseti(L, -2, (lua_Integer)i + 1);
        }
        lua_pushinteger(L, (lua_Integer)rp.w);
        lua_pushinteger(L, (lua_Integer)rp.h);
        lua_pushinteger(L, (lua_Integer)rp.channels);
        return 4;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_ponyxlDdpmText2ImgLatent(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    if (!ctx.currentModel) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }

    const char* prompt = luaL_checkstring(L, 1);
    const int seed = (int)luaL_optinteger(L, 2, 12345);
    const int sample_steps = (int)luaL_optinteger(L, 3, 50);
    const float guidance_scale = (float)luaL_optnumber(L, 4, 1.0);

    try {
        auto* pony = dynamic_cast<PonyXLDDPMModel*>(ctx.currentModel.get());
        if (!pony) {
            lua_pushnil(L);
            lua_pushstring(L, "Le modèle courant n'est pas un PonyXLDDPMModel (type=ponyxl_ddpm attendu)");
            return 2;
        }

        const PonyXLDDPMModel::ReconPreview rp = pony->text2imgSdxlLatentDiffusion(
            prompt ? std::string(prompt) : std::string(),
            seed,
            sample_steps,
            guidance_scale,
            0,
            false
        );

        if (rp.latent.empty() || rp.latent_w <= 0 || rp.latent_h <= 0 || rp.latent_c <= 0) {
            lua_pushnil(L);
            lua_pushstring(L, "text2img_latent a retourné un latent vide");
            return 2;
        }

        lua_createtable(L, (int)rp.latent.size(), 0);
        for (size_t i = 0; i < rp.latent.size(); ++i) {
            lua_pushnumber(L, (lua_Number)rp.latent[i]);
            lua_rawseti(L, -2, (lua_Integer)i + 1);
        }
        lua_pushinteger(L, (lua_Integer)rp.latent_w);
        lua_pushinteger(L, (lua_Integer)rp.latent_h);
        lua_pushinteger(L, (lua_Integer)rp.latent_c);
        return 4;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_prepareSequences(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    int seq_length = luaL_checkinteger(L, 1);
    
    ctx.addLog("Préparation séquences (longueur=" + std::to_string(seq_length) + ")");
    
    try {
        if (!ctx.currentConfig.contains("dataset") || 
            !ctx.currentConfig["dataset"].contains("dir")) {
            lua_pushboolean(L, false);
            lua_pushstring(L, "Aucun dataset chargé. Utilisez dataset.load() d'abord.");
            return 2;
        }
        
        std::string dataset_dir = ctx.currentConfig["dataset"]["dir"];
        std::vector<DatasetItem> items = loadDataset(dataset_dir);
        
        // Créer des séquences à partir des items
        ctx.currentSequences.clear();
        
        for (auto& item : items) {
            // Charger le texte si nécessaire (lazy loading)
            if (!item.text_file.empty() && !item.text.has_value()) {
                item.loadText();
            }
            
            if (item.text.has_value() && !item.text.value().empty()) {
                // Tokenize le texte
                std::vector<int> tokens;
                if (ctx.currentTokenizer) {
                    tokens = ctx.currentTokenizer->tokenize(item.text.value());
                    
                    // Padding/truncation à seq_length
                    if (tokens.size() < static_cast<size_t>(seq_length)) {
                        tokens.resize(seq_length, ctx.currentTokenizer->getPadId());
                    } else if (tokens.size() > static_cast<size_t>(seq_length)) {
                        tokens.resize(seq_length);
                    }
                    
                    ctx.currentSequences.push_back(tokens);
                }
            }
        }
        
        ctx.addLog("✓ " + std::to_string(ctx.currentSequences.size()) + " séquences préparées");
        
        lua_pushboolean(L, true);
        lua_pushinteger(L, ctx.currentSequences.size());
        return 2;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

// ============================================================================
// Utilitaires
// ============================================================================

int LuaScripting::lua_print(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    const char* msg = luaL_checkstring(L, 1);
    ctx.addLog(msg);
    
    return 0;
}

int LuaScripting::lua_readJSON(lua_State* L) {
    const char* filepath = luaL_checkstring(L, 1);
    
    try {
        std::ifstream f(filepath);
        json j;
        f >> j;
        
        jsonToLuaTable(L, j);
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_writeJSON(lua_State* L) {
    const char* filepath = luaL_checkstring(L, 1);
    luaL_checktype(L, 2, LUA_TTABLE);
    
    try {
        json j = luaTableToJson(L, 2);
        
        std::ofstream f(filepath);
        f << j.dump(2);
        
        lua_pushboolean(L, true);
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
    
    return 1;
}

// ============================================================================
// Conversion Lua <-> JSON
// ============================================================================

json LuaScripting::luaTableToJson(lua_State* L, int index) {
    // IMPORTANT: si l'appelant passe un index relatif (ex: -1), il devient instable
    // dès qu'on push/pop sur la stack. On le convertit donc en index absolu.
    index = lua_absindex(L, index);

    json result;
    
    // Vérifier si c'est un array ou un objet
    bool is_array = true;
    int max_index = 0;
    
    lua_pushnil(L);
    while (lua_next(L, index) != 0) {
        if (!lua_isnumber(L, -2)) {
            is_array = false;
        } else {
            int idx = lua_tointeger(L, -2);
            if (idx > max_index) max_index = idx;
        }
        lua_pop(L, 1);
    }
    
    if (is_array) {
        result = json::array();
        for (int i = 1; i <= max_index; ++i) {
            lua_rawgeti(L, index, i);
            
            if (lua_isnil(L, -1)) {
                result.push_back(nullptr);
            } else if (lua_isboolean(L, -1)) {
                result.push_back(lua_toboolean(L, -1) != 0);
            } else if (lua_isnumber(L, -1)) {
                result.push_back(lua_tonumber(L, -1));
            } else if (lua_isstring(L, -1)) {
                result.push_back(lua_tostring(L, -1));
            } else if (lua_istable(L, -1)) {
                result.push_back(luaTableToJson(L, lua_gettop(L)));
            }
            
            lua_pop(L, 1);
        }
    } else {
        result = json::object();
        lua_pushnil(L);
        while (lua_next(L, index) != 0) {
            std::string key;
            if (lua_isstring(L, -2)) {
                key = lua_tostring(L, -2);
            } else if (lua_isnumber(L, -2)) {
                key = std::to_string(lua_tointeger(L, -2));
            }
            
            if (lua_isnil(L, -1)) {
                result[key] = nullptr;
            } else if (lua_isboolean(L, -1)) {
                result[key] = (lua_toboolean(L, -1) != 0);
            } else if (lua_isnumber(L, -1)) {
                result[key] = lua_tonumber(L, -1);
            } else if (lua_isstring(L, -1)) {
                result[key] = lua_tostring(L, -1);
            } else if (lua_istable(L, -1)) {
                result[key] = luaTableToJson(L, lua_gettop(L));
            }
            
            lua_pop(L, 1);
        }
    }
    
    return result;
}

void LuaScripting::jsonToLuaTable(lua_State* L, const json& j) {
    if (j.is_null()) {
        lua_pushnil(L);
    } else if (j.is_boolean()) {
        lua_pushboolean(L, j.get<bool>());
    } else if (j.is_number_integer()) {
        lua_pushinteger(L, j.get<int>());
    } else if (j.is_number_float()) {
        lua_pushnumber(L, j.get<double>());
    } else if (j.is_string()) {
        lua_pushstring(L, j.get<std::string>().c_str());
    } else if (j.is_array()) {
        lua_newtable(L);
        for (size_t i = 0; i < j.size(); ++i) {
            jsonToLuaTable(L, j[i]);
            lua_rawseti(L, -2, i + 1);
        }
    } else if (j.is_object()) {
        lua_newtable(L);
        for (auto it = j.begin(); it != j.end(); ++it) {
            lua_pushstring(L, it.key().c_str());
            jsonToLuaTable(L, it.value());
            lua_settable(L, -3);
        }
    }
}

// ============================================================================
// Nouvelles implémentations - Model API étendue
// ============================================================================

