#pragma once

#include <string>
#include <memory>
class LuaScripting;
#include <vector>

#include "scriptings/ScriptingRuntime.hpp"
#include "scriptings/ScriptingContext.hpp"

class RustScripting : public ScriptingRuntime {
public:
    RustScripting();
    ~RustScripting() override;

    void setArgs(const std::string& script_path,
                 const std::vector<std::string>& script_args) override;
    bool loadScript(const std::string& filepath) override;
    bool executeScript(const std::string& code) override;
    void registerAPI() override;
    void setSystemConfig(const json& conf, const std::string& path, const std::string& dir);

private:
    std::unique_ptr<LuaScripting> native_;
    std::string conf_path_, conf_dir_;
    bool runScript(const std::string& code, const std::string& source_file);
};

class RustContext : public ScriptingContext {
public:
    static RustContext& getInstance() {
        static RustContext instance;
        return instance;
    }

private:
    RustContext() = default;
    ~RustContext() = default;
    RustContext(const RustContext&) = delete;
    RustContext& operator=(const RustContext&) = delete;
};
