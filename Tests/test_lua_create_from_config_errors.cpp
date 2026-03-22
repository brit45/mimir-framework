#include "test_utils.hpp"

#include "LuaScripting.hpp"

int main() {
    LuaScripting lua;

    const std::string code = R"(
        TEST_OK = false
        TEST_CASES = 0

        if type(Mimir) ~= "table" or type(Mimir.Model) ~= "table" then return end
        if type(Mimir.Model.create_from_config) ~= "function" then return end

        -- 1) Non-table arg must raise a Lua error, pcall should catch it.
        do
            local ok, err = pcall(Mimir.Model.create_from_config, 123)
            if ok then return end
            TEST_CASES = TEST_CASES + 1
        end

        -- 2) Unknown architecture must return (false, err)
        do
            local ok, arch_or_err = Mimir.Model.create_from_config({ architecture = "does_not_exist" })
            if ok then return end
            if type(arch_or_err) ~= "string" then return end
            TEST_CASES = TEST_CASES + 1
        end

        TEST_OK = (TEST_CASES == 2)
    )";

    TASSERT_TRUE(lua.executeScript(code));
    TASSERT_TRUE(lua.getBoolean("TEST_OK"));

    return 0;
}
