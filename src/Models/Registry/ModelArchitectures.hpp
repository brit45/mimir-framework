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

// Helpers: resolve/merge a *full* config (with parent keys) into a model config.
// Convention:
//  - Top-level key `architecture` selects the registry entry (fallback: `type`).
//  - Model overrides can live in `config.model` and/or `config[architecture]`.
//  - All parent keys from the full config are preserved inside the returned cfg
//    (as nested objects) so the framework/Lua can access them later.
std::string resolveArchitectureFromConfig(const json& full_config, const std::string& default_arch = "t2i_autoencoder");

json cfgFromConfig(const json& full_config, std::string* out_arch = nullptr, const std::string& default_arch = "t2i_autoencoder");

std::shared_ptr<Model> createFromConfig(const json& full_config,
                                        json* out_cfg = nullptr,
                                        std::string* out_arch = nullptr,
                                        const std::string& default_arch = "t2i_autoencoder");

} // namespace ModelArchitectures

