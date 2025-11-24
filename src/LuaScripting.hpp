#pragma once

extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}
#include <string>
#include <memory>
#include "Model.hpp"
#include "Tokenizer.hpp"
#include "Encoder.hpp"
#include "include/json.hpp"

using json = nlohmann::json;

// ============================================================================
// LuaScripting - Interface Lua pour piloter le modèle
// ============================================================================

class LuaScripting {
public:
    LuaScripting();
    ~LuaScripting();

    // Charger et exécuter un script Lua
    bool loadScript(const std::string& filepath);
    bool executeScript(const std::string& code);
    
    // Exécuter une fonction Lua
    bool callFunction(const std::string& function_name);
    
    // Obtenir des valeurs depuis Lua
    std::string getString(const std::string& var_name);
    double getNumber(const std::string& var_name);
    bool getBoolean(const std::string& var_name);
    
    // Définir des valeurs dans Lua
    void setString(const std::string& var_name, const std::string& value);
    void setNumber(const std::string& var_name, double value);
    void setBoolean(const std::string& var_name, bool value);
    
    // Obtenir l'état Lua (pour usage avancé)
    lua_State* getState() { return L; }
    
    // Enregistrer les bindings
    void registerAPI();

private:
    lua_State* L;
    
    // Fonctions C++ appelables depuis Lua
    static int lua_createModel(lua_State* L);
    static int lua_buildModel(lua_State* L);
    static int lua_trainModel(lua_State* L);
    static int lua_inferModel(lua_State* L);
    static int lua_saveModel(lua_State* L);
    static int lua_loadModel(lua_State* L);
    
    // Tokenizer
    static int lua_createTokenizer(lua_State* L);
    static int lua_tokenize(lua_State* L);
    static int lua_detokenize(lua_State* L);
    
    // Dataset
    static int lua_loadDataset(lua_State* L);
    static int lua_prepareSequences(lua_State* L);
    
    // Utilitaires
    static int lua_print(lua_State* L);
    static int lua_log(lua_State* L);
    static int lua_readJSON(lua_State* L);
    static int lua_writeJSON(lua_State* L);
    
    // Helpers
    static json luaTableToJson(lua_State* L, int index);
    static void jsonToLuaTable(lua_State* L, const json& j);
};

// ============================================================================
// Singleton global pour accès depuis les callbacks Lua
// ============================================================================
class LuaContext {
public:
    static LuaContext& getInstance() {
        static LuaContext instance;
        return instance;
    }
    
    // Stockage des objets C++ accessibles depuis Lua
    std::shared_ptr<Model> currentModel;
    std::shared_ptr<Tokenizer> currentTokenizer;
    std::shared_ptr<Encoder> currentEncoder;
    std::vector<std::vector<int>> currentSequences;
    json currentConfig;
    
    // Logs
    std::vector<std::string> logs;
    void addLog(const std::string& msg) {
        logs.push_back(msg);
        std::cout << "[LUA] " << msg << std::endl;
    }
    
private:
    LuaContext() = default;
    ~LuaContext() = default;
    LuaContext(const LuaContext&) = delete;
    LuaContext& operator=(const LuaContext&) = delete;
};
