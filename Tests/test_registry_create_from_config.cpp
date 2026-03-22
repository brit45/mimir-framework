#include "test_utils.hpp"

#include "Models/Registry/ModelArchitectures.hpp"
#include "include/json.hpp"

using json = nlohmann::json;

int main() {
    // Use a lightweight architecture for unit tests.
    json full = {
        {"architecture", "basic_mlp"},
        {"basic_mlp", {{"input_dim", 2}, {"hidden_dim", 4}, {"output_dim", 1}, {"hidden_layers", 1}, {"dropout", 0.0}}},
        {"tokenizer", {{"max_vocab", 128}, {"max_sequence_length", 16}}},
        {"encoder", {{"embedding_dim", 32}, {"use_positional_encoding", true}}},
        {"training", {{"optimizer", "adamw"}, {"learning_rate", 1e-3}}},
    };

    json out_cfg;
    std::string out_arch;
    auto m = ModelArchitectures::createFromConfig(full, &out_cfg, &out_arch);

    TASSERT_TRUE(static_cast<bool>(m));
    TASSERT_TRUE(out_arch == "basic_mlp");

    // The model stores the merged cfg (with standardized type).
    TASSERT_TRUE(m->modelConfig.is_object());
    TASSERT_TRUE(m->modelConfig.contains("type"));
    TASSERT_TRUE(m->modelConfig["type"].get<std::string>() == "basic_mlp");

    // Parent keys are preserved in the final config.
    TASSERT_TRUE(m->modelConfig.contains("tokenizer") && m->modelConfig["tokenizer"].is_object());
    TASSERT_TRUE(m->modelConfig.contains("encoder") && m->modelConfig["encoder"].is_object());
    TASSERT_TRUE(m->modelConfig.contains("training") && m->modelConfig["training"].is_object());

    // Flattened overrides still present at root (derived from basic_mlp section).
    TASSERT_TRUE(m->modelConfig.contains("input_dim") && m->modelConfig["input_dim"].get<int>() == 2);
    TASSERT_TRUE(m->modelConfig.contains("hidden_dim") && m->modelConfig["hidden_dim"].get<int>() == 4);

    return 0;
}
