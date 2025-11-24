#include "LuaScripting.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

// ============================================================================
// Constructeur / Destructeur
// ============================================================================

LuaScripting::LuaScripting() {
    L = luaL_newstate();
    luaL_openlibs(L);  // Charger les bibliothèques standard Lua
    registerAPI();
}

LuaScripting::~LuaScripting() {
    if (L) {
        lua_close(L);
    }
}

// ============================================================================
// Chargement et exécution de scripts
// ============================================================================

bool LuaScripting::loadScript(const std::string& filepath) {
    if (luaL_dofile(L, filepath.c_str()) != LUA_OK) {
        std::cerr << "Erreur Lua: " << lua_tostring(L, -1) << std::endl;
        lua_pop(L, 1);
        return false;
    }
    return true;
}

bool LuaScripting::executeScript(const std::string& code) {
    if (luaL_dostring(L, code.c_str()) != LUA_OK) {
        std::cerr << "Erreur Lua: " << lua_tostring(L, -1) << std::endl;
        lua_pop(L, 1);
        return false;
    }
    return true;
}

bool LuaScripting::callFunction(const std::string& function_name) {
    lua_getglobal(L, function_name.c_str());
    if (!lua_isfunction(L, -1)) {
        std::cerr << "Fonction Lua introuvable: " << function_name << std::endl;
        lua_pop(L, 1);
        return false;
    }
    
    if (lua_pcall(L, 0, 0, 0) != LUA_OK) {
        std::cerr << "Erreur lors de l'appel de " << function_name << ": " 
                  << lua_tostring(L, -1) << std::endl;
        lua_pop(L, 1);
        return false;
    }
    
    return true;
}

// ============================================================================
// Getters / Setters
// ============================================================================

std::string LuaScripting::getString(const std::string& var_name) {
    lua_getglobal(L, var_name.c_str());
    std::string result = lua_isstring(L, -1) ? lua_tostring(L, -1) : "";
    lua_pop(L, 1);
    return result;
}

double LuaScripting::getNumber(const std::string& var_name) {
    lua_getglobal(L, var_name.c_str());
    double result = lua_isnumber(L, -1) ? lua_tonumber(L, -1) : 0.0;
    lua_pop(L, 1);
    return result;
}

bool LuaScripting::getBoolean(const std::string& var_name) {
    lua_getglobal(L, var_name.c_str());
    bool result = lua_isboolean(L, -1) ? lua_toboolean(L, -1) : false;
    lua_pop(L, 1);
    return result;
}

void LuaScripting::setString(const std::string& var_name, const std::string& value) {
    lua_pushstring(L, value.c_str());
    lua_setglobal(L, var_name.c_str());
}

void LuaScripting::setNumber(const std::string& var_name, double value) {
    lua_pushnumber(L, value);
    lua_setglobal(L, var_name.c_str());
}

void LuaScripting::setBoolean(const std::string& var_name, bool value) {
    lua_pushboolean(L, value);
    lua_setglobal(L, var_name.c_str());
}

// ============================================================================
// Enregistrement de l'API
// ============================================================================

void LuaScripting::registerAPI() {
    // Table "model"
    lua_newtable(L);
    
    lua_pushcfunction(L, lua_createModel);
    lua_setfield(L, -2, "create");
    
    lua_pushcfunction(L, lua_buildModel);
    lua_setfield(L, -2, "build");
    
    lua_pushcfunction(L, lua_trainModel);
    lua_setfield(L, -2, "train");
    
    lua_pushcfunction(L, lua_inferModel);
    lua_setfield(L, -2, "infer");
    
    lua_pushcfunction(L, lua_saveModel);
    lua_setfield(L, -2, "save");
    
    lua_pushcfunction(L, lua_loadModel);
    lua_setfield(L, -2, "load");
    
    lua_setglobal(L, "model");
    
    // Table "tokenizer"
    lua_newtable(L);
    
    lua_pushcfunction(L, lua_createTokenizer);
    lua_setfield(L, -2, "create");
    
    lua_pushcfunction(L, lua_tokenize);
    lua_setfield(L, -2, "tokenize");
    
    lua_pushcfunction(L, lua_detokenize);
    lua_setfield(L, -2, "detokenize");
    
    lua_setglobal(L, "tokenizer");
    
    // Table "dataset"
    lua_newtable(L);
    
    lua_pushcfunction(L, lua_loadDataset);
    lua_setfield(L, -2, "load");
    
    lua_pushcfunction(L, lua_prepareSequences);
    lua_setfield(L, -2, "prepare_sequences");
    
    lua_setglobal(L, "dataset");
    
    // Fonctions utilitaires globales
    lua_pushcfunction(L, lua_print);
    lua_setglobal(L, "log");
    
    lua_pushcfunction(L, lua_readJSON);
    lua_setglobal(L, "read_json");
    
    lua_pushcfunction(L, lua_writeJSON);
    lua_setglobal(L, "write_json");
}

// ============================================================================
// Implémentation des fonctions Lua
// ============================================================================

int LuaScripting::lua_createModel(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    // Argument: type de modèle (string)
    const char* model_type = luaL_checkstring(L, 1);
    
    // Argument optionnel: config (table)
    json config;
    if (lua_istable(L, 2)) {
        config = luaTableToJson(L, 2);
    }
    
    try {
        // Créer un modèle de base générique
        ctx.currentModel = std::make_shared<Model>();
        
        ctx.addLog("Modèle créé: " + std::string(model_type));
        lua_pushboolean(L, true);
    } catch (const std::exception& e) {
        ctx.addLog("Erreur création modèle: " + std::string(e.what()));
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
    
    return 1;
}

int LuaScripting::lua_buildModel(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    try {
        // Appeler build() du modèle
        ctx.currentModel->build();
        
        size_t params = ctx.currentModel->totalParamCount();
        ctx.addLog("Modèle construit avec " + std::to_string(params) + " paramètres");
        
        lua_pushboolean(L, true);
        lua_pushinteger(L, params);
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
    
    // Boucle d'entraînement basique
    try {
        if (ctx.currentSequences.empty()) {
            lua_pushboolean(L, false);
            lua_pushstring(L, "Aucune séquence chargée. Utilisez dataset.prepare_sequences() d'abord.");
            return 2;
        }
        
        Optimizer opt;
        opt.initial_lr = lr;
        opt.type = OptimizerType::ADAMW;
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float epoch_loss = 0.0f;
            
            // Training sur toutes les séquences
            for (const auto& seq : ctx.currentSequences) {
                // Forward pass (simplifié)
                std::vector<uint8_t> dummy_output;
                ctx.currentModel->forward(dummy_output);
                
                // Backward pass (simplifié)
                ctx.currentModel->optimizerStep(opt, lr, nullptr);
            }
            
            ctx.addLog("Epoch " + std::to_string(epoch + 1) + "/" + std::to_string(epochs) + " completed");
        }
        
        lua_pushboolean(L, true);
        return 1;
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
        Encoder encoder;
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
            ctx.currentEncoder = std::make_shared<Encoder>(encoder);
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

int LuaScripting::lua_loadDataset(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    const char* dataset_dir = luaL_checkstring(L, 1);
    
    ctx.addLog("Chargement dataset: " + std::string(dataset_dir));
    
    try {
        fs::path dataset_path(dataset_dir);
        
        if (!fs::exists(dataset_path)) {
            lua_pushboolean(L, false);
            lua_pushstring(L, "Le dossier dataset n'existe pas");
            return 2;
        }
        
        // Charger les items du dataset
        std::vector<DatasetItem> items = loadDataset(dataset_dir);
        
        if (items.empty()) {
            ctx.addLog("⚠️  Attention: Dataset vide");
        } else {
            ctx.addLog("✓ " + std::to_string(items.size()) + " items chargés");
        }
        
        // Stocker dans le contexte (on stocke juste le path pour lazy loading)
        if (!ctx.currentConfig.contains("dataset")) {
            ctx.currentConfig["dataset"] = json::object();
        }
        ctx.currentConfig["dataset"]["dir"] = dataset_dir;
        ctx.currentConfig["dataset"]["num_items"] = items.size();
        
        lua_pushboolean(L, true);
        lua_pushinteger(L, items.size());
        return 2;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
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
        
        for (const auto& item : items) {
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
                result.push_back(lua_toboolean(L, -1));
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
                result[key] = lua_toboolean(L, -1);
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
