#include "test_utils.hpp"

#include "Models/Registry/ModelArchitectures.hpp"
#include "Serialization/Serialization.hpp"

#include <algorithm>
#include <filesystem>
#include <string>

static int compare_model_weights(const Model& a, const Model& b) {
    const auto& la = a.getLayers();
    const auto& lb = b.getLayers();

    TASSERT_TRUE(la.size() == lb.size());

    for (size_t i = 0; i < la.size(); ++i) {
        const auto& A = la[i];
        const auto& B = lb[i];
        TASSERT_TRUE(A.name == B.name);

        if (!A.weight_block && !B.weight_block) continue;
        TASSERT_TRUE(A.weight_block != nullptr);
        TASSERT_TRUE(B.weight_block != nullptr);

        const size_t nA = A.weight_block->getSize();
        const size_t nB = B.weight_block->getSize();
        TASSERT_TRUE(nA == nB);
        const float* pA = A.weight_block->getData();
        const float* pB = B.weight_block->getData();
        TASSERT_TRUE(pA != nullptr);
        TASSERT_TRUE(pB != nullptr);

        const size_t n = std::min<size_t>(nA, 64);
        for (size_t j = 0; j < n; ++j) {
            TASSERT_NEAR(pA[j], pB[j], 1e-6f);
        }
    }

    return 0;
}

int main() {
    using namespace Mimir::Serialization;

    json cfg = {
        {"input_dim", 4},
        {"hidden_dim", 8},
        {"output_dim", 2},
        {"hidden_layers", 1},
        {"dropout", 0.0}
    };

    auto modelA = ModelArchitectures::create("basic_mlp", cfg);
    TASSERT_TRUE(modelA != nullptr);
    modelA->allocateParams();
    modelA->initializeWeights("xavier", 123u);

    // Attach deterministic-ish tokenizer/encoder state (ensures JSON + embeddings tensors exist).
    Tokenizer tok(128);
    tok.setMaxSequenceLength(16);
    tok.tokenizeEnsure("hello world");

    Encoder enc(8, 128);
    enc.ensureVocabSize(tok.getVocabSize(), 777u);
    enc.ensureSpecialEmbeddings(42u);

    modelA->setTokenizer(tok);
    modelA->setEncoder(enc);
    modelA->setHasEncoder(true);

    const std::filesystem::path tmp = std::filesystem::temp_directory_path();
    const std::filesystem::path p = tmp / "mimir_test_model.safetensors";

    SaveOptions sopts;
    sopts.format = CheckpointFormat::SafeTensors;
    sopts.save_tokenizer = true;
    sopts.save_encoder = true;
    sopts.save_optimizer = false;

    std::string err;
    TASSERT_TRUE(save_checkpoint(*modelA, p.string(), sopts, &err));

    // Load into a fresh model with different init weights, to make sure load overwrites.
    auto modelB = ModelArchitectures::create("basic_mlp", cfg);
    TASSERT_TRUE(modelB != nullptr);
    modelB->allocateParams();
    modelB->initializeWeights("xavier", 999u);

    LoadOptions lopts;
    lopts.format = CheckpointFormat::SafeTensors;
    lopts.strict_mode = true;
    lopts.load_tokenizer = true;
    lopts.load_encoder = true;
    lopts.load_optimizer = false;

    TASSERT_TRUE(load_checkpoint(*modelB, p.string(), lopts, &err));

    // Weights should match.
    TASSERT_TRUE(compare_model_weights(*modelA, *modelB) == 0);

    // Tokenizer/encoder should have been restored.
    TASSERT_TRUE(modelB->getTokenizer().getVocabSize() == modelA->getTokenizer().getVocabSize());
    TASSERT_TRUE(modelB->getEncoder().dim == modelA->getEncoder().dim);
    TASSERT_TRUE(modelB->getEncoder().vocab_size == modelA->getEncoder().vocab_size);
    TASSERT_TRUE(!modelB->getEncoder().token_embeddings.empty());

    std::filesystem::remove(p);
    return 0;
}
