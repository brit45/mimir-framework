#pragma once

#include <string>
#include <vector>

#include "scriptings/ScriptingRuntime.hpp"
#include "scriptings/ScriptingContext.hpp"

class JSScripting : public ScriptingRuntime {
public:
    JSScripting() = default;
    ~JSScripting() override = default;

    void setArgs(const std::string& script_path,
                 const std::vector<std::string>& script_args) override;
    bool loadScript(const std::string& filepath) override;
    bool executeScript(const std::string& code) override;
    void registerAPI() override;

private:
    std::string loaded_file_;
};

class JSContext : public ScriptingContext {
public:
    static JSContext& getInstance() {
        static JSContext instance;
        return instance;
    }

private:
    JSContext() = default;
    ~JSContext() = default;
    JSContext(const JSContext&) = delete;
    JSContext& operator=(const JSContext&) = delete;
};
