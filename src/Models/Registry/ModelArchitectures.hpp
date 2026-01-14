#pragma once

#include "../Model.hpp"
#include "../include/json.hpp"

#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ModelArchitectures {

using json = nlohmann::json;

using CreateFn = std::function<std::shared_ptr<Model>(const json&)>;

struct Entry {
    std::string name;
    std::string description;
    json default_config;
    CreateFn create;
};

class Registry {
public:
    static Registry& instance();

    void registerArchitecture(Entry entry);

    const Entry* find(const std::string& name) const;

    std::vector<std::string> list() const;

    json defaultConfig(const std::string& name) const;

    std::shared_ptr<Model> create(const std::string& name, const json& config) const;

private:
    Registry() = default;
    void ensureBuiltinsRegistered() const;

    mutable std::once_flag builtins_once_;
    mutable std::unordered_map<std::string, Entry> entries_;
};

inline std::vector<std::string> available() {
    return Registry::instance().list();
}

inline json defaultConfig(const std::string& name) {
    return Registry::instance().defaultConfig(name);
}

inline std::shared_ptr<Model> create(const std::string& name, const json& config = json::object()) {
    return Registry::instance().create(name, config);
}

} // namespace ModelArchitectures

