#include "test_utils.hpp"

#include "Models/Registry/ModelArchitectures.hpp"
#include "Serialization/Serialization.hpp"

#include <filesystem>
#include <string>

int main() {
    using namespace Mimir::Serialization;

    json cfg = {
        {"input_dim", 3},
        {"hidden_dim", 5},
        {"output_dim", 2},
        {"hidden_layers", 2},
        {"dropout", 0.0}
    };

    auto modelA = ModelArchitectures::create("basic_mlp", cfg);
    TASSERT_TRUE(modelA != nullptr);
    modelA->allocateParams();
    modelA->initializeWeights("xavier", 321u);

    Tokenizer tok(64);
    tok.setMaxSequenceLength(8);
    tok.tokenizeEnsure("alpha beta");

    Encoder enc(8, 64);
    enc.ensureVocabSize(tok.getVocabSize(), 9u);
    enc.ensureSpecialEmbeddings(10u);

    modelA->setTokenizer(tok);
    modelA->setEncoder(enc);
    modelA->setHasEncoder(true);

    const std::filesystem::path tmp = std::filesystem::temp_directory_path();
    const std::filesystem::path dir = tmp / "mimir_test_rawfolder";

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);

    SaveOptions sopts;
    sopts.format = CheckpointFormat::RawFolder;
    sopts.save_tokenizer = true;
    sopts.save_encoder = true;

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
    lopts.load_encoder = true;

    TASSERT_TRUE(load_checkpoint(*modelB, dir.string(), lopts, &err));

    // Basic sanity: model config restored and embeddings present.
    TASSERT_TRUE(modelB->modelConfig.contains("type"));
    TASSERT_TRUE(modelB->getEncoder().dim == modelA->getEncoder().dim);
    TASSERT_TRUE(!modelB->getEncoder().token_embeddings.empty());

    std::filesystem::remove_all(dir, ec);
    return 0;
}
