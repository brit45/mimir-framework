#include "test_utils.hpp"

#include "scriptings/Lua/luaScripting/LuaScripting.hpp"

int main() {
    LuaScripting lua;

    const std::string code = R"(
        TEST_OK = false
        TEST_ERR = ""

        if type(Mimir) ~= "table" then TEST_ERR = "Mimir missing" return end
        if type(Mimir.Architectures) ~= "table" then TEST_ERR = "Architectures missing" return end
        if type(Mimir.Architectures.available) ~= "function" then TEST_ERR = "available missing" return end
        if type(Mimir.Architectures.default_config) ~= "function" then TEST_ERR = "default_config missing" return end
        if type(Mimir.Architectures.info) ~= "function" then TEST_ERR = "info missing" return end
        if type(Mimir.Architectures.dtypes) ~= "function" then TEST_ERR = "dtypes missing" return end
        if type(Mimir.Model) ~= "table" then TEST_ERR = "Model missing" return end
        if type(Mimir.Model.create) ~= "function" then TEST_ERR = "Model.create missing" return end

        local archs = Mimir.Architectures.available()
        if type(archs) ~= "table" or #archs < 1 then TEST_ERR = "no architectures" return end

        local cfg = Mimir.Architectures.default_config("basic_mlp")
        if type(cfg) ~= "table" then TEST_ERR = "default_config not table" return end

        -- info() sans argument: liste complète
        local entries = Mimir.Architectures.info()
        if type(entries) ~= "table" or #entries < 1 then TEST_ERR = "info() not a list" return end
        if type(entries[1].name) ~= "string" then TEST_ERR = "info() entry has no name" return end
        if type(entries[1].config) ~= "table" then TEST_ERR = "info() entry has no config" return end

        -- info(name): entrée unique
        local one = Mimir.Architectures.info("basic_mlp")
        if type(one) ~= "table" or one.name ~= "basic_mlp" then TEST_ERR = "info(name) wrong" return end
        if type(one.description) ~= "string" then TEST_ERR = "info(name) has no description" return end

        -- info(unknown): (nil, err)
        local bad, berr = Mimir.Architectures.info("__no_such_arch__")
        if bad ~= nil or type(berr) ~= "string" then TEST_ERR = "info(unknown) should fail" return end

        -- dtypes(): liste de descripteurs
        local dts = Mimir.Architectures.dtypes()
        if type(dts) ~= "table" or #dts < 1 then TEST_ERR = "dtypes() not a list" return end
        if type(dts[1].name) ~= "string" or type(dts[1].bytes) ~= "number" then TEST_ERR = "dtypes() entry malformed" return end

        local ok = Mimir.Model.create("basic_mlp", cfg)
        if ok ~= true then TEST_ERR = "Model.create returned false" return end

        TEST_OK = true
    )";

    TASSERT_TRUE(lua.executeScript(code));
    TASSERT_TRUE(lua.getBoolean("TEST_OK"));

    LuaContext::getInstance().resetRuntimeState();

    return 0;
}
