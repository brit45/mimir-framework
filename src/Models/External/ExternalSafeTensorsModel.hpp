#pragma once

#include "../../Model.hpp"

#include <string>
#include <vector>

class ExternalSafeTensorsModel : public Model {
public:
    struct Config {
        std::string source_safetensors;
        std::vector<std::string> include_prefixes;
        std::vector<std::string> exclude_prefixes;
        int max_tensors = 0;
    };

    ExternalSafeTensorsModel();

    void buildFromConfig(const Config& cfg);
    const Config& getConfig() const { return cfg_; }

    static void buildInto(Model& model, const Config& cfg);

private:
    Config cfg_;
};