#include "test_utils.hpp"

#include "LuaScripting.hpp"

int main() {
    LuaScripting lua;

    const std::string code = R"(
        TEST_OK = false
        TEST_ERR = ""

        if type(Mimir) ~= "table" then TEST_ERR = "Mimir missing" return end
        if type(Mimir.Architectures) ~= "table" then TEST_ERR = "Architectures missing" return end
        if type(Mimir.Architectures.available) ~= "function" then TEST_ERR = "available missing" return end
        if type(Mimir.Architectures.default_config) ~= "function" then TEST_ERR = "default_config missing" return end
        if type(Mimir.Model) ~= "table" then TEST_ERR = "Model missing" return end
        if type(Mimir.Model.create) ~= "function" then TEST_ERR = "Model.create missing" return end

        local archs = Mimir.Architectures.available()
        if type(archs) ~= "table" or #archs < 1 then TEST_ERR = "no architectures" return end

        local cfg = Mimir.Architectures.default_config("basic_mlp")
        if type(cfg) ~= "table" then TEST_ERR = "default_config not table" return end

        local ok = Mimir.Model.create("basic_mlp", cfg)
        if ok ~= true then TEST_ERR = "Model.create returned false" return end

        TEST_OK = true
    )";

    TASSERT_TRUE(lua.executeScript(code));
    TASSERT_TRUE(lua.getBoolean("TEST_OK"));

    return 0;
}
