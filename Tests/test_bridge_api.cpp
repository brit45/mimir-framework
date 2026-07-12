// Tests/test_bridge_api.cpp
// Teste la parite du bridge commun (ScriptingBridgeCommon) pour
// les runtimes JS, C# et Rust. Chaque sous-test:
//   1. Cree un script minimal inline.
//   2. L'execute via le runtime cible.
//   3. Verifie le contexte C++ (modele cree, params alloues).
//
#define MIMIR_ASSERT(cond, msg) TASSERT_TRUE((cond))
#define MIMIR_FAIL(msg)  do { std::cerr << "FAIL: " << (msg) << "\n"; return false; } while(0)
#include "test_utils.hpp"

#include "include/json.hpp"
#ifdef MIMIR_ENABLE_SCRIPTING_JS
#include "scriptings/JavaScript/jsScripting/JSScripting.hpp"
#endif
#ifdef MIMIR_ENABLE_SCRIPTING_CSHARP
#include "scriptings/CSharp/csharpScripting/CSharpScripting.hpp"
#endif
#ifdef MIMIR_ENABLE_SCRIPTING_RUST
#include "scriptings/Rust/rustScripting/RustScripting.hpp"
#endif
#include "scriptings/ScriptingBridgeCommon.hpp"
#include "Models/Registry/ModelArchitectures.hpp"

using json = nlohmann::json;

#include <cstdlib>
#include <filesystem>
#include <fstream>
#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool commandAvailable(const char* cmd) {
    return std::system((std::string("command -v ") + cmd + " >/dev/null 2>&1").c_str()) == 0;
}

static bool dotnetRuntimeAvailable() {
    if (const char* root = std::getenv("DOTNET_ROOT")) {
        const fs::path fxr = fs::path(root) / "host" / "fxr";
        if (fs::exists(fxr)) return true;
    }
    if (const char* home = std::getenv("HOME")) {
        const fs::path fxr = fs::path(home) / ".dotnet" / "host" / "fxr";
        if (fs::exists(fxr)) return true;
    }
    return false;
}

static int currentPid() {
#if defined(_WIN32)
    return static_cast<int>(_getpid());
#else
    return static_cast<int>(::getpid());
#endif
}

static fs::path writeTempScript(const std::string& ext, const std::string& content) {
    const auto tmp = fs::temp_directory_path() /
                     ("mimir_test_bridge" + std::to_string(currentPid()) + "." + ext);
    std::ofstream f(tmp);
    f << content;
    return tmp;
}

// ---------------------------------------------------------------------------
// Assertions communes apres execution bridge
// ---------------------------------------------------------------------------

static bool assertModelCreated(const ScriptingContext& ctx, const std::string& label) {
    if (!ctx.currentModel) {
        MIMIR_FAIL(label + ": currentModel == nullptr apres bridge");
        return false;
    }
    if (ctx.modelType != "basic_mlp") {
        MIMIR_FAIL(label + ": modelType != basic_mlp, got=" + ctx.modelType);
        return false;
    }
    return true;
}

static bool assertParamsAllocated(const ScriptingContext& ctx, const std::string& label) {
    if (!ctx.currentModel) { MIMIR_FAIL(label + ": no model"); return false; }
    const size_t n = ctx.currentModel->totalParamCount();
    if (n == 0) {
        MIMIR_FAIL(label + ": totalParamCount == 0 apres allocate_params");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// ScriptingBridgeCommon: test des builders de donnees
// ---------------------------------------------------------------------------

static int testBridgeCommonData() {
    const std::string avail = ScriptingBridgeCommon::buildAvailableJson();
    MIMIR_ASSERT(!avail.empty(), "buildAvailableJson vide");
    MIMIR_ASSERT(avail.find("basic_mlp") != std::string::npos,
                 "basic_mlp absent de buildAvailableJson");

    const std::string info = ScriptingBridgeCommon::buildArchitecturesInfoJson();
    MIMIR_ASSERT(info.find("description") != std::string::npos,
                 "buildArchitecturesInfoJson sans 'description'");

    // Normalisation: aliases doit etre un array ([ au lieu d'une string "...)
    const std::string dtypes = ScriptingBridgeCommon::buildDtypesJson();
    MIMIR_ASSERT(dtypes.find("float32") != std::string::npos,
                 "buildDtypesJson sans 'float32'");
    MIMIR_ASSERT(dtypes.find("\"aliases\":[") != std::string::npos,
                 "aliases n'est pas un array JSON (normalisation manquante)");
    MIMIR_ASSERT(dtypes.find("\"bytes\":4") != std::string::npos,
                 "bytes=4 absent pour float32");
    MIMIR_ASSERT(dtypes.find("\"kind\":\"float\"") != std::string::npos,
                 "kind=float absent");

    // Verifier la structure complete du premier dtype via parse JSON.
    const json dts = json::parse(dtypes);
    MIMIR_ASSERT(dts.is_array() && dts.size() == 13, "dtypes: 13 entrees attendues");
    const json& f32 = dts[0];  // float32 est le premier.
    MIMIR_ASSERT(f32["name"].get<std::string>() == "float32", "f32 name != float32");
    MIMIR_ASSERT(f32["aliases"].is_array(),                   "f32 aliases n'est pas un array");
    MIMIR_ASSERT(f32["aliases"].size() >= 2,                  "f32 aliases trop court");
    MIMIR_ASSERT(f32["aliases"][0].get<std::string>() == "float", "f32 aliases[0] != 'float'");
    MIMIR_ASSERT(f32["bytes"].get<int>() == 4,                "f32 bytes != 4");
    MIMIR_ASSERT(f32["kind"].get<std::string>() == "float",   "f32 kind != 'float'");

    // Cache: premiere lecture donne un JSON object valide (possiblement vide).
    const std::string cache = ScriptingBridgeCommon::loadArchCacheJson();
    MIMIR_ASSERT(cache.front() == '{', "loadArchCacheJson ne commence pas par '{'");

    return 0;
}

// ---------------------------------------------------------------------------
// Test: bridge commun direct (sans script)
// ---------------------------------------------------------------------------

static int testBridgeDirectCommands() {
    // Simuler l'ecriture d'un fichier de commandes et sa lecture par processBridgeCommands.
    const auto cmd_file = fs::temp_directory_path() /
                          ("mimir_test_cmds_" + std::to_string(currentPid()) + ".cmd");

    {
        std::ofstream f(cmd_file);
        f << "Model.create|basic_mlp\n";
        f << "Model.allocate_params\n";
        f << "Model.init_weights|he|0\n";
        f << "Model.total_params\n";
    }

    // Utiliser un contexte runtime disponible comme contexte generique.
    ScriptingContext* ctx = nullptr;
#if defined(MIMIR_ENABLE_SCRIPTING_JS)
    ctx = &JSContext::getInstance();
#elif defined(MIMIR_ENABLE_SCRIPTING_CSHARP)
    ctx = &CSharpContext::getInstance();
#elif defined(MIMIR_ENABLE_SCRIPTING_RUST)
    ctx = &RustContext::getInstance();
#else
    std::cerr << "[SKIP] BridgeTest.Direct: aucun bridge script activé\n";
    {
        std::error_code skip_ec;
        fs::remove(cmd_file, skip_ec);
    }
    return 0;
#endif
    ctx->resetRuntimeState();

    const bool ok = ScriptingBridgeCommon::processBridgeCommands(*ctx, cmd_file, "[test]");
    std::error_code ec;
    fs::remove(cmd_file, ec);

    MIMIR_ASSERT(ok, "processBridgeCommands a retourne false");
    if (!assertModelCreated(*ctx, "bridge direct")) return 1;
    if (!assertParamsAllocated(*ctx, "bridge direct")) return 1;

    ctx->resetRuntimeState();
    return 0;
}

// ---------------------------------------------------------------------------
// Test: bridge via runtime JS
// ---------------------------------------------------------------------------

static int testJsBridge() {
#ifndef MIMIR_ENABLE_SCRIPTING_JS
    std::cerr << "[SKIP] BridgeTest.JS: bridge JS désactivé à la compilation\n";
    return 0;
#else
    if (!commandAvailable("node")) {
        std::cerr << "[SKIP] BridgeTest.JS: node absent du PATH\n";
        return 0;
    }

    const auto script = writeTempScript("mjs",
        "Model.create('basic_mlp', {});\n"
        "Model.allocate_params();\n"
        "Model.init_weights('he', 0);\n"
        "Model.total_params();\n"
    );

    JSContext& ctx = JSContext::getInstance();
    ctx.resetRuntimeState();

    JSScripting js;
    js.registerAPI();
    js.setArgs(script.string(), {});
    const bool ok = js.loadScript(script.string());

    std::error_code ec;
    fs::remove(script, ec);

    MIMIR_ASSERT(ok, "JS bridge script a echoue");
    if (!assertModelCreated(ctx, "JS bridge")) return 1;
    if (!assertParamsAllocated(ctx, "JS bridge")) return 1;

    ctx.resetRuntimeState();
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// Test: bridge via runtime C#
// ---------------------------------------------------------------------------

static int testCSharpBridge() {
#ifndef MIMIR_ENABLE_SCRIPTING_CSHARP
    std::cerr << "[SKIP] BridgeTest.CSharp: bridge C# désactivé à la compilation\n";
    return 0;
#else
    const bool has_cs = commandAvailable("dotnet-script") || commandAvailable("csi");
    if (!has_cs || !dotnetRuntimeAvailable()) {
        std::cerr << "[SKIP] BridgeTest.CSharp: dotnet-script/csi absent du PATH\n";
        return 0;
    }

    const auto script = writeTempScript("csx",
        "Model.create(\"basic_mlp\", \"\");\n"
        "Model.allocate_params();\n"
        "Model.init_weights(\"he\", 0);\n"
        "long p = (long)Model.total_params();\n"
    );

    CSharpContext& ctx = CSharpContext::getInstance();
    ctx.resetRuntimeState();

    CSharpScripting cs;
    cs.registerAPI();
    cs.setArgs(script.string(), {});
    const bool ok = cs.loadScript(script.string());

    std::error_code ec;
    fs::remove(script, ec);

    MIMIR_ASSERT(ok, "C# bridge script a echoue");
    if (!assertModelCreated(ctx, "CSharp bridge")) return 1;
    if (!assertParamsAllocated(ctx, "CSharp bridge")) return 1;

    ctx.resetRuntimeState();
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// Test: bridge via runtime Rust
// ---------------------------------------------------------------------------

static int testRustBridge() {
#ifndef MIMIR_ENABLE_SCRIPTING_RUST
    std::cerr << "[SKIP] BridgeTest.Rust: bridge Rust désactivé à la compilation\n";
    return 0;
#else
    if (!commandAvailable("rust-script")) {
        std::cerr << "[SKIP] BridgeTest.Rust: rust-script absent du PATH\n";
        return 0;
    }

    const auto script = writeTempScript("rs",
        "Model::create(\"basic_mlp\", \"\");\n"
        "Model::allocate_params();\n"
        "Model::init_weights(\"he\", 0);\n"
        "let _p = Model::total_params();\n"
    );

    RustContext& ctx = RustContext::getInstance();
    ctx.resetRuntimeState();

    RustScripting rs;
    rs.registerAPI();
    rs.setArgs(script.string(), {});
    const bool ok = rs.loadScript(script.string());

    std::error_code ec;
    fs::remove(script, ec);

    MIMIR_ASSERT(ok, "Rust bridge script a echoue");
    if (!assertModelCreated(ctx, "Rust bridge")) return 1;
    if (!assertParamsAllocated(ctx, "Rust bridge")) return 1;

    ctx.resetRuntimeState();
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// Test: Model.create avec config JSON personnalisee
// ---------------------------------------------------------------------------

static int testBridgeCreateWithConfig() {
    const auto cmd_file = fs::temp_directory_path() /
                          ("mimir_test_cfg_" + std::to_string(currentPid()) + ".cmd");
    {
        std::ofstream f(cmd_file);
        // Config minimale valide pour basic_mlp
        f << "Model.create|basic_mlp|{\"hidden_size\":256,\"input_size\":512,\"output_size\":10}\n";
        f << "Model.allocate_params\n";
    }

    ScriptingContext* ctx = nullptr;
#if defined(MIMIR_ENABLE_SCRIPTING_JS)
    ctx = &JSContext::getInstance();
#elif defined(MIMIR_ENABLE_SCRIPTING_CSHARP)
    ctx = &CSharpContext::getInstance();
#elif defined(MIMIR_ENABLE_SCRIPTING_RUST)
    ctx = &RustContext::getInstance();
#else
    std::cerr << "[SKIP] BridgeTest.CreateWithConfig: aucun bridge script activé\n";
    {
        std::error_code skip_ec;
        fs::remove(cmd_file, skip_ec);
    }
    return 0;
#endif
    ctx->resetRuntimeState();

    const bool ok = ScriptingBridgeCommon::processBridgeCommands(*ctx, cmd_file, "[test-cfg]");
    std::error_code ec;
    fs::remove(cmd_file, ec);

    MIMIR_ASSERT(ok, "bridge create+cfg a echoue");
    if (!assertModelCreated(*ctx, "bridge create+cfg")) return 1;

    ctx->resetRuntimeState();
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    int failures = 0;

    failures += testBridgeCommonData();
    failures += testBridgeDirectCommands();
    failures += testBridgeCreateWithConfig();
    failures += testJsBridge();
    failures += testCSharpBridge();
    failures += testRustBridge();

    if (failures == 0)
        std::cerr << "[PASS] BridgeTest: tous les sous-tests reussis\n";
    else
        std::cerr << "[FAIL] BridgeTest: " << failures << " echec(s)\n";

    return failures == 0 ? 0 : 1;
}
