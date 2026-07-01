#pragma once

#include <algorithm>
#include <cctype>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "Model.hpp"
#include "Tokenizer.hpp"
#include "Encoder.hpp"
#include "include/json.hpp"

class AsyncMonitor;

using json = nlohmann::json;

class ScriptingContext {
public:
    virtual ~ScriptingContext() = default;

    // Contrat API système partagé entre tous les langages de scripting.
    static constexpr const char* kGlobalNamespace = "Mimir";
    static constexpr const char* kGlobalArg = "arg";
    static constexpr const char* kGlobalConf = "CONF";
    static constexpr const char* kGlobalConfPath = "CONF_PATH";
    static constexpr const char* kGlobalConfDir = "CONF_DIR";

    static constexpr const char* kAliasModel = "model";
    static constexpr const char* kAliasArchitectures = "architectures";
    static constexpr const char* kAliasTokenizer = "tokenizer";
    static constexpr const char* kAliasDataset = "dataset";
    static constexpr const char* kAliasMemory = "Memory";
    static constexpr const char* kAliasMemoryGuard = "MemoryGuard";
    static constexpr const char* kAliasAllocator = "Allocator";
    static constexpr const char* kAliasHtop = "htop";
    static constexpr const char* kAliasViz = "viz";

    void resetRuntimeState() {
        currentModel.reset();
        currentTokenizer.reset();
        currentEncoder.reset();
        asyncMonitor.reset();
        currentSequences.clear();
        currentDataset.clear();
        currentConfig = json{};
        modelType.clear();
        modelConfig = json{};
    }

    // Stockage des objets C++ accessibles depuis les bridges de scripting
    std::shared_ptr<Model> currentModel;
    std::shared_ptr<Tokenizer> currentTokenizer;
    std::shared_ptr<ConditioningEncoder> currentEncoder;

    // AsyncMonitor pour htop et viz
    std::shared_ptr<AsyncMonitor> asyncMonitor;
    std::vector<std::vector<int>> currentSequences;
    json currentConfig;

    // Dataset stocke
    std::vector<DatasetItem> currentDataset;

    // Configuration du modele
    std::string modelType;
    json modelConfig;

    // Logs
    std::vector<std::string> logs;
    bool suppress_stdout_logs = false;

    void addLog(const std::string& msg) {
        logs.push_back(msg);
        if (suppress_stdout_logs) return;

        auto is_error_like = [](const std::string& s) -> bool {
            if (s.empty()) return false;
            auto lower = s;
            std::transform(lower.begin(), lower.end(), lower.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return (lower.find("erreur") != std::string::npos) ||
                   (lower.find("error") != std::string::npos) ||
                   (lower.find("failed") != std::string::npos) ||
                   (lower.find("echec") != std::string::npos) ||
                   (lower.find("\xE2\x9D\x8C") != std::string::npos) ||
                   (lower.find("\xE2\x9B\x94") != std::string::npos);
        };

        const bool has_cfg = (modelConfig.is_object() && !modelConfig.empty()) ||
                             (currentConfig.is_object() && !currentConfig.empty());
        const bool is_err = is_error_like(msg);

        const char* ANSI_RESET = "\033[0m";
        const char* ANSI_GREEN_BOLD = "\033[1;4;32m";
        const char* ANSI_RED = "\033[1;4;31m";
        const char* ANSI_BLUE = "\033[1;4;36m";

        const char* color = ANSI_GREEN_BOLD;
        if (is_err) color = ANSI_RED;
        else if (has_cfg) color = ANSI_BLUE;

        std::string out = msg;
        if (!modelType.empty()) {
            const std::string plain_prefix = "[" + modelType + "]";
            const std::string colored_prefix = std::string("[") + color + modelType + ANSI_RESET + "]";

            if (out.rfind(plain_prefix, 0) == 0) {
                out = colored_prefix + out.substr(plain_prefix.size());
            } else {
                out = colored_prefix + " " + out;
            }
        }

        std::cerr << out << std::endl;
    }
};
