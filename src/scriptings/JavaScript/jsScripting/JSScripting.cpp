#include "JSScripting.hpp"
#include "scriptings/Lua/luaScripting/LuaScripting.hpp"
#include "scriptings/NativeBridge.hpp"
#include <filesystem>
#include <fstream>

namespace {
const char* bootstrap = R"JS(
import __mimirFs from 'node:fs';
{
const fs = __mimirFs;
const channel = 3;
let pending = Buffer.alloc(0);
function receive() {
  for (;;) {
    const end = pending.indexOf(10);
    if (end >= 0) {
      const line = pending.subarray(0, end).toString('utf8');
      pending = pending.subarray(end + 1);
      return JSON.parse(line);
    }
    const buffer = Buffer.allocUnsafe(65536);
    const count = fs.readSync(channel, buffer, 0, buffer.length, null);
    if (!count) throw new Error('Mimir native bridge disconnected');
    pending = Buffer.concat([pending, buffer.subarray(0, count)]);
  }
}
function normalize(value, ancestors = new Set(), depth = 0) {
  if (depth > 64) throw new TypeError('Mimir value nesting exceeds 64');
  if (ArrayBuffer.isView(value)) {
    if (value instanceof DataView) throw new TypeError('Use a typed array instead of DataView');
    const data = normalize(Array.from(value), ancestors, depth + 1);
    if (value instanceof Float32Array || value instanceof Float64Array)
      return {$mimir_array_kind: 'float', $mimir_session: initial.session, data};
    return data;
  }
  if (typeof value === 'function' || typeof value === 'bigint' || typeof value === 'symbol')
    throw new TypeError('Mimir arguments must be JSON values or typed arrays');
  if (typeof value === 'number' && !Number.isFinite(value))
    throw new TypeError('Mimir arguments must contain finite numbers');
  if (value === null || typeof value !== 'object') return value;
  if (ancestors.has(value)) throw new TypeError('Mimir arguments must not contain cycles');
  ancestors.add(value);
  try {
    if (Array.isArray(value)) return value.map(item => normalize(item, ancestors, depth + 1));
    if (typeof value.toJSON === 'function') return normalize(value.toJSON(), ancestors, depth + 1);
    const result = Object.create(null);
    for (const [key, item] of Object.entries(value)) result[key] = normalize(item, ancestors, depth + 1);
    return result;
  } finally { ancestors.delete(value); }
}
function invoke(id, args, raw = false) {
  // Normalize before JSON.stringify: Buffer.toJSON otherwise masks typed arrays.
  const data = Buffer.from(JSON.stringify({function: id, args: normalize(args)}) + '\n');
  let offset = 0;
  while (offset < data.length) {
    const count = fs.writeSync(channel, data, offset, data.length - offset);
    if (!count) throw new Error('Mimir native bridge disconnected');
    offset += count;
  }
  const reply = receive();
  if (reply.error !== undefined) throw new Error(reply.error);
  const results = reply.results.map(hydrate);
  if (raw) return results;
  // Lua's nil,error convention maps to a catchable JS exception. Successful
  // multiple returns remain tuples, single returns remain ordinary JS values.
  if (results.length >= 2 && (results[0] === null || results[0] === false) && typeof results[1] === 'string')
    throw new Error(results[1]);
  return results.length <= 1 ? results[0] : results;
}
function hydrate(value) {
  if (value === null || typeof value !== 'object') return value;
  if (value.$mimir_session === initial.session && Object.hasOwn(value, '$mimir_function')) {
    const id = value.$mimir_function;
    const fn = (...args) => invoke(id, args);
    fn.results = (...args) => invoke(id, args, true);
    return fn;
  }
  if (Array.isArray(value)) return value.map(hydrate);
  const result = {};
  for (const [key, item] of Object.entries(value))
    Object.defineProperty(result, key, {value: hydrate(item), enumerable: true, writable: true, configurable: true});
  return result;
}
const initial = receive();
Object.assign(globalThis, hydrate(initial.globals));
// Keep the original JS dtype representation and architecture creation alias.
const nativeDtypes = Mimir.Architectures.dtypes;
Mimir.Architectures.dtypes = () => nativeDtypes().map(dtype => ({...dtype,
  aliases: typeof dtype.aliases === 'string' ? dtype.aliases.split(',').map(s => s.trim()) : dtype.aliases}));
Mimir.Architectures.dtypes.results = nativeDtypes.results;
Mimir.Architectures.create = Mimir.Model.create;
for (const [name, value] of Object.entries(Mimir)) globalThis[name] = value;
for (const [alias, name] of Object.entries({model:'Model', architectures:'Architectures', tokenizer:'Tokenizer', dataset:'Dataset', Memory:'Memory', MemoryGuard:'MemoryGuard', Allocator:'Allocator', htop:'Htop', viz:'Viz'}))
  globalThis[alias] = Mimir[name];
}
)JS";
} // namespace

JSScripting::JSScripting() : native_(std::make_unique<LuaScripting>()) {}
JSScripting::~JSScripting() = default;
void JSScripting::setArgs(const std::string& path, const std::vector<std::string>& args) { cacheArgs(path, args); }
void JSScripting::setSystemConfig(const json& conf, const std::string& path, const std::string& dir) {
    JSContext::getInstance().currentConfig = conf;
    conf_path_ = path; conf_dir_ = dir;
}
bool JSScripting::loadScript(const std::string& path) {
    std::ifstream file(path);
    if (!file) { JSContext::getInstance().addLog("[js] unable to open script: " + path); return false; }
    file.close();
    return runScript("", std::filesystem::absolute(path).string());
}
bool JSScripting::executeScript(const std::string& code) { return runScript(code, ""); }
bool JSScripting::runScript(const std::string& code, const std::string& source_file) {
    auto& ctx = JSContext::getInstance();
    try {
    NativeBridge::TemporaryDirectory temporary;
    std::string inline_path;
    struct InlineCleanup {
        std::string& path;
        ~InlineCleanup() { if (!path.empty()) { std::error_code ec; std::filesystem::remove(path, ec); } }
    } cleanup{inline_path};
    return NativeBridge::execute(*native_, ctx, scriptPath(), scriptArgs(), ctx.currentConfig,
        conf_path_, conf_dir_, [&](const json&) {
            const auto preload = temporary.write("preload.mjs", bootstrap);
            std::string entry = source_file;
            if (entry.empty()) {
                const auto directory = scriptPath().empty() ? std::filesystem::current_path()
                    : std::filesystem::absolute(scriptPath()).parent_path();
                inline_path = NativeBridge::writeTemporary(directory, ".mjs", code);
                entry = inline_path;
            }
            std::vector<std::string> args{"node", "--import", preload, entry};
            args.insert(args.end(), scriptArgs().begin(), scriptArgs().end());
            return args;
        }, "js");
    } catch (const std::exception& error) {
        ctx.addLog(std::string("[js] execution error: ") + error.what());
        return false;
    }
}
void JSScripting::registerAPI() {}
