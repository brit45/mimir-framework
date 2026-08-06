#include "test_utils.hpp"

#include "scriptings/Lua/luaScripting/LuaScripting.hpp"

#include <filesystem>

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

    const std::filesystem::path mpk_path = std::filesystem::temp_directory_path() / "mimir_create_path_test.mpk";
    const std::string mpk_code = R"(
        TEST_PATH_OK = false
        TEST_PATH_ERR = ""

        local MPK = dofile("scripts/modules/mpk.lua")
        local pkg = MPK.build({
            name = "path_plugin_test",
            type = "basic_mlp",
            author = "test",
            description = "path-based create test",
            base_config = { input_dim = 2, hidden_dim = 4, output_dim = 1, hidden_layers = 1, dropout = 0.0 },
            model_structure = MPK.model_structure_template("basic_mlp"),
            container = "pseudocode",
        })
        if not pkg then
            TEST_PATH_ERR = "MPK.build failed"
            return
        end

        local ok_write, err_write = MPK.write("__MPK_PATH__", pkg, { binary = false })
        if not ok_write then
            TEST_PATH_ERR = tostring(err_write)
            return
        end

        local mpk_text, err_text = MPK.read_text_file("__MPK_PATH__")
        if type(mpk_text) ~= "string"
            or not mpk_text:match("^# MPK")
            or not mpk_text:find("map mpk = %[%]")
            or not mpk_text:find('mpk.set%("payload",') then
            TEST_PATH_ERR = "MPK is not written as Visu-like pseudocode: " .. tostring(err_text)
            return
        end

        local loaded, err_loaded = MPK.read("__MPK_PATH__")
        if type(loaded) ~= "table" or loaded.container ~= "pseudocode" then
            TEST_PATH_ERR = "MPK pseudocode round-trip failed: " .. tostring(err_loaded)
            return
        end

        local rejected_json = MPK.write("__MPK_PATH__", pkg, { json = true })
        if rejected_json ~= nil then
            TEST_PATH_ERR = "MPK JSON output should be rejected"
            return
        end

        local compiled_path = "__MPK_PATH__.bin"
        local ok_compile, err_compile = MPK.compile("__MPK_PATH__", compiled_path)
        if not ok_compile then
            TEST_PATH_ERR = "MPK pseudocode compilation failed: " .. tostring(err_compile)
            return
        end

        local compiled_raw = MPK.read_text_file(compiled_path)
        if type(compiled_raw) ~= "string"
            or compiled_raw:sub(1, 4) ~= "MPKB"
            or compiled_raw:byte(5) ~= 4
            or compiled_raw:find("PSC3", 1, true)
            or compiled_raw:find("# MPK", 1, true) then
            TEST_PATH_ERR = "compiled MPK is not opaque binary-v4"
            return
        end
        if #compiled_raw >= #mpk_text then
            TEST_PATH_ERR = "compiled MPK should be smaller than pseudocode source"
            return
        end

        local compiled, err_compiled = MPK.read(compiled_path)
        if type(compiled) ~= "table" or compiled.container ~= "binary" then
            TEST_PATH_ERR = "compiled MPK round-trip failed: " .. tostring(err_compiled)
            return
        end

        local ok_create = Mimir.Model.create("__MPK_PATH__")
        if ok_create ~= true then
            TEST_PATH_ERR = "Mimir.Model.create(path) failed"
            return
        end

        local ok_create_compiled = Mimir.Model.create(compiled_path)
        if ok_create_compiled ~= true then
            TEST_PATH_ERR = "Mimir.Model.create(compiled path) failed"
            return
        end

        TEST_PATH_OK = true
    )";

    std::string mpk_code_filled = mpk_code;
    const std::string mpk_path_str = mpk_path.string();
    const std::string placeholder = "__MPK_PATH__";
    size_t pos = 0;
    while ((pos = mpk_code_filled.find(placeholder, pos)) != std::string::npos) {
        mpk_code_filled.replace(pos, placeholder.size(), mpk_path_str);
        pos += mpk_path_str.size();
    }

    TASSERT_TRUE(lua.executeScript(mpk_code_filled));
    TASSERT_TRUE(lua.getBoolean("TEST_PATH_OK"));

    LuaContext::getInstance().resetRuntimeState();

    return 0;
}
