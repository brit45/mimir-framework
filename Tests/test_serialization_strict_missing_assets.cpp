#include "test_utils.hpp"

#include "Models/Registry/ModelArchitectures.hpp"
#include "Serialization/Serialization.hpp"

#include <filesystem>
#include <string>

int main() {
    using namespace Mimir::Serialization;

    json cfg = {
        {"input_dim", 4},
        {"hidden_dim", 8},
        {"output_dim", 2},
        {"hidden_layers", 1},
        {"dropout", 0.0}
    };

    // 1) SafeTensors: save without tokenizer, then strict-load with load_tokenizer=true should fail.
    {
        auto modelA = ModelArchitectures::create("basic_mlp", cfg);
        TASSERT_TRUE(modelA != nullptr);
        modelA->allocateParams();
        modelA->initializeWeights("xavier", 123u);

        const std::filesystem::path tmp = std::filesystem::temp_directory_path();
        const std::filesystem::path p = tmp / "mimir_test_missing_tokenizer.safetensors";

        SaveOptions sopts;
        sopts.format = CheckpointFormat::SafeTensors;
        sopts.save_tokenizer = false;
        sopts.save_encoder = false;
        sopts.save_optimizer = false;

        std::string err;
        TASSERT_TRUE(save_checkpoint(*modelA, p.string(), sopts, &err));

        auto modelB = ModelArchitectures::create("basic_mlp", cfg);
        TASSERT_TRUE(modelB != nullptr);
        modelB->allocateParams();
        modelB->initializeWeights("xavier", 999u);

        LoadOptions lopts;
        lopts.format = CheckpointFormat::SafeTensors;
        lopts.strict_mode = true;
        lopts.load_tokenizer = true;
        lopts.load_encoder = false;
        lopts.load_optimizer = false;

        err.clear();
        const bool ok = load_checkpoint(*modelB, p.string(), lopts, &err);
        TASSERT_TRUE(!ok);
        TASSERT_TRUE(!err.empty());

        std::filesystem::remove(p);
    }

    // 2) RawFolder: save without encoder, then strict-load with load_encoder=true should fail.
    {
        auto modelA = ModelArchitectures::create("basic_mlp", cfg);
        TASSERT_TRUE(modelA != nullptr);
        modelA->allocateParams();
        modelA->initializeWeights("xavier", 321u);

        // Save a tokenizer but not an encoder.
        Tokenizer tok(64);
        tok.setMaxSequenceLength(8);
        tok.tokenizeEnsure("alpha beta");
        modelA->setTokenizer(tok);

        const std::filesystem::path tmp = std::filesystem::temp_directory_path();
        const std::filesystem::path dir = tmp / "mimir_test_missing_encoder";

        std::error_code ec;
        std::filesystem::remove_all(dir, ec);

        SaveOptions sopts;
        sopts.format = CheckpointFormat::RawFolder;
        sopts.save_tokenizer = true;
        sopts.save_encoder = false;
        sopts.save_optimizer = false;

        std::string err;
        TASSERT_TRUE(save_checkpoint(*modelA, dir.string(), sopts, &err));

        auto modelB = ModelArchitectures::create("basic_mlp", cfg);
        TASSERT_TRUE(modelB != nullptr);
        modelB->allocateParams();
        modelB->initializeWeights("xavier", 111u);

        LoadOptions lopts;
        lopts.format = CheckpointFormat::RawFolder;
        lopts.strict_mode = true;
        lopts.load_tokenizer = true;
        lopts.load_encoder = true;  // required but missing
        lopts.load_optimizer = false;

        err.clear();
        const bool ok = load_checkpoint(*modelB, dir.string(), lopts, &err);
        TASSERT_TRUE(!ok);
        TASSERT_TRUE(!err.empty());

        std::filesystem::remove_all(dir, ec);
    }

    return 0;
}
