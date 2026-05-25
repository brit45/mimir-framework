#include "test_utils.hpp"

#include "LuaScripting.hpp"

int main() {
    LuaScripting lua;

    const std::string code = R"(
        TEST_OK = false
        TEST_ERR = ""

        local ok, err = Mimir.Model.create("basic_mlp", { input_dim = 4, hidden_dim = 4, output_dim = 4 })
        if not ok then TEST_ERR = "create failed: " .. tostring(err) return end

        -- Alias requested by spec
        if type(Mimir.model) ~= "table" then TEST_ERR = "Mimir.model missing" return end
        if type(Mimir.model.dtype) ~= "function" then TEST_ERR = "Mimir.model.dtype missing" return end

        -- Getter returns a string
        local cur = Mimir.model.dtype()
        if type(cur) ~= "string" then TEST_ERR = "dtype() getter not string" return end

        -- Setter returns (ok, value)
        local ok2, val = Mimir.model.dtype("float16")
        if ok2 ~= true then TEST_ERR = "dtype setter failed: " .. tostring(val) return end
        if val ~= "float16" then TEST_ERR = "dtype setter returned unexpected: " .. tostring(val) return end

        local cur2 = Mimir.model.dtype()
        if cur2 ~= "float16" then TEST_ERR = "dtype getter mismatch: " .. tostring(cur2) return end

        -- Unknown dtype should fail
        local ok3, err3 = Mimir.model.dtype("nope")
        if ok3 ~= false then TEST_ERR = "unknown dtype did not fail" return end
        if type(err3) ~= "string" or #err3 < 1 then TEST_ERR = "unknown dtype missing error" return end

        TEST_OK = true
    )";

    TASSERT_TRUE(lua.executeScript(code));
    TASSERT_TRUE(lua.getBoolean("TEST_OK"));

    return 0;
}
