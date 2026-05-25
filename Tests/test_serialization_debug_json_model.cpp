#include "test_utils.hpp"

#include "Models/Registry/ModelArchitectures.hpp"
#include "Serialization/Serialization.hpp"

#include <filesystem>
#include <fstream>
#include <string>

static json read_json_file(const std::filesystem::path& p) {
    std::ifstream f(p);
    TASSERT_TRUE((bool)f);
    json j;
    f >> j;
    return j;
}

int main() {
    using namespace Mimir::Serialization;

    json cfg = {
        {"input_dim", 4},
        {"hidden_dim", 8},
        {"output_dim", 2},
        {"hidden_layers", 1},
        {"dropout", 0.0},
    };

    auto model = ModelArchitectures::create("basic_mlp", cfg);
    TASSERT_TRUE(model != nullptr);

    model->setDefaultDType("bfloat16");
    model->allocateParams();
    model->initializeWeights("xavier", 123u);

    const std::filesystem::path tmp = std::filesystem::temp_directory_path();
    const std::filesystem::path p = tmp / "mimir_test_debug_json_dtype.json";
    std::filesystem::remove(p);

    SaveOptions sopts;
    sopts.format = CheckpointFormat::DebugJson;
    sopts.include_git_info = false;
    sopts.save_tokenizer = false;
    sopts.save_encoder = false;
    sopts.include_optimizer_state = false;
    sopts.include_checksums = false;
    sopts.include_weight_deltas = false;

    std::string err;
    TASSERT_TRUE(save_checkpoint(*model, p.string(), sopts, &err));

    json j = read_json_file(p);

    TASSERT_TRUE(j.contains("format"));
    TASSERT_TRUE(j["format"].get<std::string>() == "mimir_debug_dump");

    TASSERT_TRUE(j.contains("format_version"));
    TASSERT_TRUE(j["format_version"].get<std::string>() == "1.1.0");

    TASSERT_TRUE(j.contains("default_dtype"));
    TASSERT_TRUE(j["default_dtype"].get<std::string>() == "bfloat16");

    TASSERT_TRUE(j.contains("model_config"));
    TASSERT_TRUE(j["model_config"].is_object());
    TASSERT_TRUE(j["model_config"].contains("dtype"));
    TASSERT_TRUE(j["model_config"]["dtype"].get<std::string>() == "bfloat16");

    std::filesystem::remove(p);
    return 0;
}
