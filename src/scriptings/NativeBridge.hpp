#pragma once
#include "scriptings/ScriptingContext.hpp"
#include <functional>
#include <filesystem>
#include <string>
#include <vector>
class LuaScripting;
namespace NativeBridge {
using Prepare = std::function<std::vector<std::string>(const json&)>;
bool execute(LuaScripting& native, ScriptingContext& context,
             const std::string& script_path, const std::vector<std::string>& script_args,
             const json& conf, const std::string& conf_path, const std::string& conf_dir,
             const Prepare& prepare, const std::string& language);
std::string writeTemporary(const std::filesystem::path& directory,
                           const std::string& suffix, const std::string& contents);
struct TemporaryDirectory {
    std::filesystem::path path;
    TemporaryDirectory();
    ~TemporaryDirectory();
    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
    std::string write(const std::string& name, const std::string& contents) const;
};
}
