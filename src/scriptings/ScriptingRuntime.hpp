#pragma once

#include <string>
#include <vector>

class ScriptingRuntime {
public:
    virtual ~ScriptingRuntime() = default;

    virtual void setArgs(const std::string& script_path,
                         const std::vector<std::string>& script_args) = 0;
    virtual bool loadScript(const std::string& filepath) = 0;
    virtual bool executeScript(const std::string& code) = 0;
    virtual void registerAPI() = 0;

protected:
    void cacheArgs(const std::string& script_path,
                   const std::vector<std::string>& script_args) {
        script_path_ = script_path;
        script_args_ = script_args;
    }

    const std::string& scriptPath() const { return script_path_; }
    const std::vector<std::string>& scriptArgs() const { return script_args_; }

private:
    std::string script_path_;
    std::vector<std::string> script_args_;
};
