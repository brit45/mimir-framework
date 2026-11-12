#include "NativeBridge.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <limits>
#include <random>
#include <stdexcept>
#include <fcntl.h>
#include <spawn.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include "scriptings/Lua/luaScripting/LuaScripting.hpp"
#include "scriptings/ScriptingBridgeCommon.hpp"

extern char** environ;

namespace {
// All external bridges share one lock while borrowing the reference context.
std::mutex execution_mutex;
struct ContextScope {
    ScriptingContext& external;
    ScriptingContext& lua = LuaContext::getInstance();
    void exchange() {
        using std::swap;
        swap(external.currentModel, lua.currentModel);
        swap(external.currentTokenizer, lua.currentTokenizer);
        swap(external.currentEncoder, lua.currentEncoder);
        swap(external.asyncMonitor, lua.asyncMonitor);
        swap(external.currentSequences, lua.currentSequences);
        swap(external.currentConfig, lua.currentConfig);
        swap(external.currentDataset, lua.currentDataset);
        swap(external.datasetPrefetcher, lua.datasetPrefetcher);
        swap(external.modelType, lua.modelType);
        swap(external.modelConfig, lua.modelConfig);
        swap(external.logs, lua.logs);
        swap(external.suppress_stdout_logs, lua.suppress_stdout_logs);
    }
    explicit ContextScope(ScriptingContext& context) : external(context) { exchange(); }
    ~ContextScope() { exchange(); }
};

struct Descriptor {
    int fd = -1;
    ~Descriptor() { if (fd >= 0) close(fd); }
};

struct Values {
    lua_State* L;
    std::vector<int> refs;
    std::string session;
    ~Values() { for (int r : refs) luaL_unref(L, LUA_REGISTRYINDEX, r); }

    json read(int index, int depth = 0) {
        if (depth > 64) throw std::runtime_error("bridge value nesting exceeds 64");
        index = lua_absindex(L, index);
        switch (lua_type(L, index)) {
        case LUA_TNIL: return nullptr;
        case LUA_TBOOLEAN: return bool(lua_toboolean(L, index));
        case LUA_TNUMBER:
            if (lua_isinteger(L, index)) return lua_tointeger(L, index);
            return lua_tonumber(L, index);
        case LUA_TSTRING: {
            size_t n = 0;
            const char* s = lua_tolstring(L, index, &n);
            return std::string(s, n);
        }
        case LUA_TFUNCTION: {
            lua_pushvalue(L, index);
            int ref = luaL_ref(L, LUA_REGISTRYINDEX);
            refs.push_back(ref);
            return json{{"$mimir_function", ref}, {"$mimir_session", session}};
        }
        case LUA_TTABLE: {
            // Lua inspection snapshots are empty read-only proxies. Serialize
            // their detached backing data so JS receives the same layer list.
            if (lua_getmetatable(L, index)) {
                lua_getfield(L, -1, "__mimir_layer_snapshot");
                const bool snapshot = lua_toboolean(L, -1);
                lua_pop(L, 1);
                if (snapshot) {
                    lua_getfield(L, -1, "__index");
                    if (lua_istable(L, -1)) {
                        json value = read(-1, depth + 1);
                        lua_pop(L, 2);
                        return value;
                    }
                    lua_pop(L, 1);
                }
                lua_pop(L, 1);
            }
            json object = json::object();
            size_t count = 0;
            bool sequence = true;
            const size_t length = lua_rawlen(L, index);
            lua_pushnil(L);
            while (lua_next(L, index)) {
                std::string key;
                if (lua_type(L, -2) == LUA_TSTRING) {
                    size_t n = 0;
                    const char* s = lua_tolstring(L, -2, &n);
                    key.assign(s, n);
                    sequence = false;
                } else if (lua_isinteger(L, -2)) {
                    const auto k = lua_tointeger(L, -2);
                    key = std::to_string(k);
                    if (k < 1 || static_cast<size_t>(k) > length) sequence = false;
                } else {
                    throw std::runtime_error("unsupported bridge table key");
                }
                object[key] = read(-1, depth + 1);
                ++count;
                lua_pop(L, 1);
            }
            if (sequence && count > 0 && count == length) {
                json array = json::array();
                for (size_t i = 1; i <= length; ++i) array.push_back(object.at(std::to_string(i)));
                return array;
            }
            return object;
        }
        default: throw std::runtime_error("unsupported native bridge value");
        }
    }

    void push(const json& value, int depth = 0) {
        if (depth > 64) throw std::runtime_error("bridge value nesting exceeds 64");
        if (value.is_object() && value.value("$mimir_session", json{}) == session &&
            value.value("$mimir_array_kind", json{}) == "float" && value.contains("data")) {
            const auto& data = value.at("data");
            if (!data.is_array()) throw std::runtime_error("typed array data must be an array");
            lua_newtable(L);
            for (size_t i = 0; i < data.size(); ++i) {
                lua_pushnumber(L, data.at(i).get<double>());
                lua_rawseti(L, -2, i + 1);
            }
        } else if (value.is_null()) lua_pushnil(L);
        else if (value.is_boolean()) lua_pushboolean(L, value.get<bool>());
        else if (value.is_number_unsigned() && value.get<uint64_t>() > static_cast<uint64_t>(std::numeric_limits<lua_Integer>::max()))
            throw std::runtime_error("integer exceeds native Lua integer range");
        else if (value.is_number_integer()) lua_pushinteger(L, value.get<lua_Integer>());
        else if (value.is_number()) lua_pushnumber(L, value.get<double>());
        else if (value.is_string()) {
            const auto s = value.get<std::string>();
            lua_pushlstring(L, s.data(), s.size());
        } else if (value.is_array()) {
            lua_newtable(L);
            for (size_t i = 0; i < value.size(); ++i) {
                push(value[i], depth + 1);
                lua_rawseti(L, -2, i + 1);
            }
        } else {
            lua_newtable(L);
            for (auto it = value.begin(); it != value.end(); ++it) {
                lua_pushlstring(L, it.key().data(), it.key().size());
                push(it.value(), depth + 1);
                lua_rawset(L, -3);
            }
        }
    }

    json call(const json& request) {
        const int top = lua_gettop(L);
        try {
            const int ref = request.at("function").get<int>();
            if (std::find(refs.begin(), refs.end(), ref) == refs.end())
                throw std::runtime_error("unknown native bridge function");
            lua_rawgeti(L, LUA_REGISTRYINDEX, ref);
            const auto& args = request.at("args");
            if (!args.is_array()) throw std::runtime_error("bridge arguments must be an array");
            for (const auto& arg : args) push(arg);
            if (lua_pcall(L, static_cast<int>(args.size()), LUA_MULTRET, 0) != LUA_OK) {
                const char* error = lua_tostring(L, -1);
                throw std::runtime_error(error ? error : "native API error");
            }
            json results = json::array();
            for (int i = top + 1; i <= lua_gettop(L); ++i) results.push_back(read(i));
            lua_settop(L, top);
            return json{{"results", results}};
        } catch (const std::exception& e) {
            lua_settop(L, top);
            return json{{"error", e.what()}};
        }
    }
};

bool sendLine(int fd, const json& value) {
    const std::string data = value.dump() + "\n";
    size_t offset = 0;
    while (offset < data.size()) {
        const auto n = send(fd, data.data() + offset, data.size() - offset, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        offset += static_cast<size_t>(n);
    }
    return true;
}

} // namespace

bool NativeBridge::execute(LuaScripting& native, ScriptingContext& context,
                           const std::string& script_path, const std::vector<std::string>& script_args,
                           const json& conf, const std::string& conf_path, const std::string& conf_dir,
                           const Prepare& prepare, const std::string& language) {
    std::lock_guard<std::mutex> lock(execution_mutex);
    const json configuration = conf;
    ContextScope scope(context);
    auto& ctx = LuaContext::getInstance();
    try {
        lua_State* L = native.getState();
        struct StackScope {
            lua_State* state;
            int top;
            ~StackScope() { lua_settop(state, top); }
        } stack{L, lua_gettop(L)};
        std::random_device random;
        std::string session;
        for (int i = 0; i < 4; ++i) session += std::to_string(random()) + ":";
        Values values{L, {}, session};
        json globals;
        for (const char* name : {ScriptingContext::kGlobalNamespace, "log", "read_json", "write_json", "ROOTWORK"}) {
            lua_getglobal(L, name);
            globals[name] = values.read(-1);
            lua_pop(L, 1);
        }
        globals[ScriptingContext::kGlobalArg] = json::array();
        globals[ScriptingContext::kGlobalArg].push_back(script_path);
        for (const auto& arg : script_args) globals[ScriptingContext::kGlobalArg].push_back(arg);
        globals[ScriptingContext::kGlobalConf] = configuration.is_null() ? json::object() : configuration;
        globals[ScriptingContext::kGlobalConfPath] = conf_path;
        globals[ScriptingContext::kGlobalConfDir] = conf_dir;
        native.setArgs(script_path, script_args);
        native.setSystemConfig(globals[ScriptingContext::kGlobalConf], conf_path, conf_dir);

        auto arguments = prepare(globals);
        if (arguments.empty()) throw std::runtime_error("empty bridge command");
        int pair[2];
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, pair)) throw std::runtime_error("cannot create native bridge channel");
        Descriptor parent{pair[0]}, child{pair[1]};
        // Move the child endpoint away from fd 3 before installing spawn actions.
        Descriptor endpoint{fcntl(child.fd, F_DUPFD_CLOEXEC, 10)};
        if (endpoint.fd < 0) throw std::runtime_error("cannot duplicate native bridge channel");
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_addclose(&actions, parent.fd);
        posix_spawn_file_actions_addclose(&actions, child.fd);
        posix_spawn_file_actions_adddup2(&actions, endpoint.fd, 3);
        posix_spawn_file_actions_addclose(&actions, endpoint.fd);
        std::vector<char*> argv;
        for (auto& arg : arguments) argv.push_back(arg.data());
        argv.push_back(nullptr);
        pid_t pid;
        std::vector<std::string> environment;
        for (char** e = environ; *e; ++e)
            if (std::string(*e).rfind("MIMIR_NATIVE_BRIDGE=", 0) != 0) environment.emplace_back(*e);
        environment.emplace_back("MIMIR_NATIVE_BRIDGE=1");
        std::vector<char*> env;
        for (auto& e : environment) env.push_back(e.data());
        env.push_back(nullptr);
        const int error = posix_spawnp(&pid, arguments.front().c_str(), &actions, nullptr, argv.data(), env.data());
        posix_spawn_file_actions_destroy(&actions);
        if (error) throw std::runtime_error("cannot start " + arguments.front() + ": " + std::string(strerror(error)));
        struct ChildProcess {
            pid_t pid;
            int channel;
            ~ChildProcess() {
                if (pid > 0) {
                    if (shutdown(channel, SHUT_RDWR) != 0) kill(pid, SIGTERM);
                    int status;
                    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
                }
            }
        } process{pid, parent.fd};
        close(child.fd); child.fd = -1;
        close(endpoint.fd); endpoint.fd = -1;
        bool ok = sendLine(parent.fd, json{{"globals", globals}, {"session", session}});
        std::string pending;
        char buffer[65536];
        while (ok) {
            const auto n = recv(parent.fd, buffer, sizeof(buffer), 0);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) { if (n < 0) ok = false; break; }
            pending.append(buffer, static_cast<size_t>(n));
            size_t end;
            while ((end = pending.find('\n')) != std::string::npos) {
                json reply;
                try { reply = values.call(json::parse(pending.substr(0, end))); }
                catch (const std::exception& e) { reply = json{{"error", e.what()}}; }
                pending.erase(0, end + 1);
                if (!sendLine(parent.fd, reply)) { ok = false; break; }
            }
        }
        if (!ok) kill(pid, SIGTERM);
        shutdown(parent.fd, SHUT_RDWR);
        int status = 0;
        pid_t waited;
        do { waited = waitpid(pid, &status, 0); } while (waited < 0 && errno == EINTR);
        process.pid = -1;
        ok = ok && waited == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0;
        if (!ok) {
            std::string detail;
            if (waited == pid && WIFEXITED(status)) detail = " (exit=" + std::to_string(WEXITSTATUS(status)) + ")";
            else if (waited == pid && WIFSIGNALED(status)) detail = " (signal=" + std::to_string(WTERMSIG(status)) + ")";
            ctx.addLog("[" + language + "] script execution failed" + detail);
        }
        return ok;
    } catch (const std::exception& e) {
        ctx.addLog(("[" + language + "] execution error: ") + e.what());
        return false;
    }
}

std::string NativeBridge::writeTemporary(const std::filesystem::path& directory,
                                          const std::string& suffix, const std::string& contents) {
    std::string pattern = (directory / (".mimir-XXXXXX" + suffix)).string();
    std::vector<char> filename(pattern.begin(), pattern.end());
    filename.push_back('\0');
    Descriptor fd{mkstemps(filename.data(), static_cast<int>(suffix.size()))};
    if (fd.fd < 0) throw std::runtime_error("cannot create bridge temporary: " + std::string(strerror(errno)));
    try {
        size_t offset = 0;
        while (offset < contents.size()) {
            auto n = write(fd.fd, contents.data() + offset, contents.size() - offset);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) throw std::runtime_error("cannot write bridge temporary");
            offset += static_cast<size_t>(n);
        }
    } catch (...) { unlink(filename.data()); throw; }
    return filename.data();
}
NativeBridge::TemporaryDirectory::TemporaryDirectory() {
    auto pattern = (std::filesystem::temp_directory_path() / "mimir-bridge-XXXXXX").string();
    std::vector<char> name(pattern.begin(), pattern.end()); name.push_back('\0');
    if (!mkdtemp(name.data())) throw std::runtime_error("cannot create bridge directory");
    path = name.data();
}
NativeBridge::TemporaryDirectory::~TemporaryDirectory() {
    std::error_code ec; std::filesystem::remove_all(path, ec);
}
std::string NativeBridge::TemporaryDirectory::write(const std::string& name, const std::string& contents) const {
    const auto file = path / name;
    std::ofstream out(file, std::ios::binary);
    out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!out) throw std::runtime_error("cannot write bridge file: " + file.string());
    return file.string();
}
