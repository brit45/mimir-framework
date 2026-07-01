#include "JSScripting.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unistd.h>

#include "Models/Registry/ModelArchitectures.hpp"
#include "scriptings/ScriptingBridgeCommon.hpp"

namespace {

static std::string shellQuote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

static bool commandExists(const char* cmd) {
    const std::string probe = std::string("command -v ") + cmd + " >/dev/null 2>&1";
    return std::system(probe.c_str()) == 0;
}

static void setEnvVar(const std::string& k, const std::string& v) {
    setenv(k.c_str(), v.c_str(), 1);
}


static std::string makeBootstrap(const std::string& user_code) {
    std::ostringstream ss;
    ss << "import fs from 'node:fs';\n";
    ss << "const __bridgeCmdFile = process.env.MIMIR_BRIDGE_CMD_FILE || '';\n";
    ss << "const __bridgeEmit = (line) => { if (__bridgeCmdFile) fs.appendFileSync(__bridgeCmdFile, line + '\\n'); };\n";
    ss << "const __archInfo = JSON.parse(process.env.MIMIR_BRIDGE_ARCH_INFO_JSON || '[]');\n";
    ss << "const __archAvail = JSON.parse(process.env.MIMIR_BRIDGE_ARCH_AVAILABLE_JSON || '[]');\n";
    ss << "const __dtypes = JSON.parse(process.env.MIMIR_BRIDGE_DTYPES_JSON || '[]');\n";
    ss << "const __archCache = JSON.parse(process.env.MIMIR_BRIDGE_ARCH_CACHE_JSON || '{}');\n";
    ss << "let __currentModelName = '';\n";
    ss << "const __mimirErr = (name) => { throw new Error(`[mimir-js] API ${name} not yet implemented in JS runtime binding.`); };\n";
    ss << "const __proxyFactory = (path='Mimir') => new Proxy(function(){}, {\n";
    ss << "  get: (_t, p) => __proxyFactory(`${path}.${String(p)}`),\n";
    ss << "  apply: () => __mimirErr(path)\n";
    ss << "});\n";
    ss << "globalThis.arg = JSON.parse(process.env.MIMIR_ARG_JSON || '[]');\n";
    ss << "globalThis.CONF = JSON.parse(process.env.MIMIR_CONF_JSON || '{}');\n";
    ss << "globalThis.CONF_PATH = process.env.MIMIR_CONF_PATH || '';\n";
    ss << "globalThis.CONF_DIR = process.env.MIMIR_CONF_DIR || '';\n";
    ss << "globalThis.Mimir = __proxyFactory(process.env.MIMIR_GLOBAL_NAMESPACE || 'Mimir');\n";
    ss << "globalThis.model = __proxyFactory(process.env.MIMIR_ALIAS_MODEL || 'model');\n";
    ss << "globalThis.architectures = __proxyFactory(process.env.MIMIR_ALIAS_ARCHITECTURES || 'architectures');\n";
    ss << "globalThis.tokenizer = __proxyFactory(process.env.MIMIR_ALIAS_TOKENIZER || 'tokenizer');\n";
    ss << "globalThis.dataset = __proxyFactory(process.env.MIMIR_ALIAS_DATASET || 'dataset');\n";
    ss << "globalThis.Memory = __proxyFactory(process.env.MIMIR_ALIAS_MEMORY || 'Memory');\n";
    ss << "globalThis.MemoryGuard = __proxyFactory(process.env.MIMIR_ALIAS_MEMORY_GUARD || 'MemoryGuard');\n";
    ss << "globalThis.Allocator = __proxyFactory(process.env.MIMIR_ALIAS_ALLOCATOR || 'Allocator');\n";
    ss << "globalThis.htop = __proxyFactory(process.env.MIMIR_ALIAS_HTOP || 'htop');\n";
    ss << "globalThis.viz = __proxyFactory(process.env.MIMIR_ALIAS_VIZ || 'viz');\n\n";
    ss << "globalThis.Architectures = {\n";
    ss << "  info: (name) => {\n";
    ss << "    if (typeof name === 'string' && name.length > 0) {\n";
    ss << "      const e = __archInfo.find(x => x.name === name);\n";
    ss << "      if (!e) throw new Error(`unknown architecture: ${name}`);\n";
    ss << "      return e;\n";
    ss << "    }\n";
    ss << "    return __archInfo;\n";
    ss << "  },\n";
    ss << "  available: () => __archAvail,\n";
    ss << "  default_config: (name) => {\n";
    ss << "    const e = __archInfo.find(x => x.name === name);\n";
    ss << "    if (!e) throw new Error(`unknown architecture: ${name}`);\n";
    ss << "    return e.config;\n";
    ss << "  },\n";
    ss << "  dtypes: () => __dtypes,\n";
    ss << "  create: (name, cfg) => {\n";
    ss << "    __currentModelName = name;\n";
    ss << "    const cfgStr = cfg ? JSON.stringify(cfg) : '';\n";
    ss << "    __bridgeEmit(`Model.create|${String(name || '')}${cfgStr ? '|' + cfgStr : ''}`);\n";
    ss << "    return true;\n";
    ss << "  }\n";
    ss << "};\n";
    ss << "globalThis.architectures = globalThis.Architectures;\n";
    ss << "globalThis.Model = {\n";
    ss << "  create: (name, cfg) => {\n";
    ss << "    __currentModelName = name;\n";
    ss << "    const cfgStr = cfg ? JSON.stringify(cfg) : '';\n";
    ss << "    __bridgeEmit(`Model.create|${String(name || '')}${cfgStr ? '|' + cfgStr : ''}`);\n";
    ss << "    return true;\n";
    ss << "  },\n";
    ss << "  allocate_params: () => { __bridgeEmit('Model.allocate_params'); return true; },\n";
    ss << "  init_weights: (method='he', seed=0) => { __bridgeEmit(`Model.init_weights|${String(method)}|${Number(seed)||0}`); return true; },\n";
    ss << "  total_params: () => { __bridgeEmit('Model.total_params'); return (__archCache[__currentModelName]?.total_params ?? 0); }\n";
    ss << "};\n";
    ss << "globalThis.model = globalThis.Model;\n\n";
    ss << user_code;
    ss << "\n";
    return ss.str();
}

}  // namespace

void JSScripting::setArgs(const std::string& script_path,
                          const std::vector<std::string>& script_args) {
    cacheArgs(script_path, script_args);
}

bool JSScripting::loadScript(const std::string& filepath) {
    std::ifstream f(filepath);
    if (!f.is_open()) {
        JSContext::getInstance().addLog("[js] unable to open script: " + filepath);
        return false;
    }

    std::ostringstream ss;
    ss << f.rdbuf();
    loaded_file_ = filepath;
    return executeScript(ss.str());
}

bool JSScripting::executeScript(const std::string& code) {
    auto& ctx = JSContext::getInstance();
    if (!commandExists("node")) {
        ctx.addLog("[js] node not found in PATH");
        return false;
    }

    const json arg_json = [&]() {
        json a = json::array();
        a.push_back(scriptPath());
        for (const auto& s : scriptArgs()) a.push_back(s);
        return a;
    }();

    setEnvVar("MIMIR_GLOBAL_NAMESPACE", ScriptingContext::kGlobalNamespace);
    setEnvVar("MIMIR_ARG_JSON", arg_json.dump());
    setEnvVar("MIMIR_CONF_JSON", ctx.currentConfig.is_null() ? "{}" : ctx.currentConfig.dump());
    setEnvVar("MIMIR_CONF_PATH", "");
    setEnvVar("MIMIR_CONF_DIR", "");
    setEnvVar("MIMIR_ALIAS_MODEL", ScriptingContext::kAliasModel);
    setEnvVar("MIMIR_ALIAS_ARCHITECTURES", ScriptingContext::kAliasArchitectures);
    setEnvVar("MIMIR_ALIAS_TOKENIZER", ScriptingContext::kAliasTokenizer);
    setEnvVar("MIMIR_ALIAS_DATASET", ScriptingContext::kAliasDataset);
    setEnvVar("MIMIR_ALIAS_MEMORY", ScriptingContext::kAliasMemory);
    setEnvVar("MIMIR_ALIAS_MEMORY_GUARD", ScriptingContext::kAliasMemoryGuard);
    setEnvVar("MIMIR_ALIAS_ALLOCATOR", ScriptingContext::kAliasAllocator);
    setEnvVar("MIMIR_ALIAS_HTOP", ScriptingContext::kAliasHtop);
    setEnvVar("MIMIR_ALIAS_VIZ", ScriptingContext::kAliasViz);
    setEnvVar(ScriptingBridgeCommon::kEnvBridgeArchInfoJson,
              ScriptingBridgeCommon::buildArchitecturesInfoJson());
    setEnvVar(ScriptingBridgeCommon::kEnvBridgeArchAvailJson,
              ScriptingBridgeCommon::buildAvailableJson());
    setEnvVar(ScriptingBridgeCommon::kEnvBridgeDtypesJson,
              ScriptingBridgeCommon::buildDtypesJson());
    setEnvVar(ScriptingBridgeCommon::kEnvBridgeArchCacheJson,
              ScriptingBridgeCommon::loadArchCacheJson());
    {
        const auto dtypes_json = ScriptingBridgeCommon::buildDtypesJson();
        const auto avail_json  = ScriptingBridgeCommon::buildAvailableJson();
        // Counts pre-calcules pour les runtimes sans parser JSON (Rust).
        const size_t dc = static_cast<size_t>(std::count(dtypes_json.begin(), dtypes_json.end(), '{'));
        const size_t ac = ModelArchitectures::available().size();
        setEnvVar(ScriptingBridgeCommon::kEnvBridgeDtypesCount,     std::to_string(dc));
        setEnvVar(ScriptingBridgeCommon::kEnvBridgeArchAvailCount,   std::to_string(ac));
    }

    const auto tmp = std::filesystem::temp_directory_path() /
                     ("mimir_js_" + std::to_string(static_cast<unsigned long long>(::getpid())) + ".mjs");
    const auto cmd_file = std::filesystem::temp_directory_path() /
                          ("mimir_js_bridge_" + std::to_string(static_cast<unsigned long long>(::getpid())) + ".cmd");
    setEnvVar(ScriptingBridgeCommon::kEnvBridgeCmdFile, cmd_file.string());
    {
        std::ofstream out(tmp);
        out << makeBootstrap(code);
    }

    std::string cmd = "node " + shellQuote(tmp.string());
    for (const auto& a : scriptArgs()) cmd += " " + shellQuote(a);

    const int rc = std::system(cmd.c_str());
    const bool bridge_ok =
        (rc == 0) ? ScriptingBridgeCommon::processBridgeCommands(ctx, cmd_file, "[js]") : false;

    std::error_code ec;
    std::filesystem::remove(tmp, ec);
    std::filesystem::remove(cmd_file, ec);

    if (rc != 0) {
        ctx.addLog("[js] script execution failed with status=" + std::to_string(rc));
        return false;
    }
    if (!bridge_ok) {
        ctx.addLog("[js] bridge command processing failed");
        return false;
    }
    return true;
}

void JSScripting::registerAPI() {
    JSContext::getInstance().addLog(
        "[js] registerAPI called (system contract unchanged)");
}
