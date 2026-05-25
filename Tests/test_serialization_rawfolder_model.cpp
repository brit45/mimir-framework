#include "test_utils.hpp"

#include "Models/Registry/ModelArchitectures.hpp"
#include "Serialization/Serialization.hpp"

#include <filesystem>
#include <fstream>
#include <string>

static int compare_model_weights(const Model& a, const Model& b, float eps) {
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
            TASSERT_NEAR(pA[j], pB[j], eps);
        }
    }

    return 0;
}

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
        {"input_dim", 3},
        {"hidden_dim", 5},
        {"output_dim", 2},
        {"hidden_layers", 2},
        {"dropout", 0.0}
    };

    struct Case {
        const char* dtype;
        const char* expected_tag;
        float eps;
    };
    const Case cases[] = {
        {"float32", "F32", 1e-6f},
        {"float16", "F16", 5e-3f},
        {"bfloat16", "BF16", 2e-2f},
        {"float64", "F64", 1e-6f},
    };

    const std::filesystem::path tmp = std::filesystem::temp_directory_path();
    std::error_code ec;

    for (const auto& c : cases) {
        json cfgA = cfg;
        cfgA["dtype"] = c.dtype;

        auto modelA = ModelArchitectures::create("basic_mlp", cfgA);
        TASSERT_TRUE(modelA != nullptr);
        TASSERT_TRUE(modelA->getDefaultDType() == std::string(c.dtype));
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

        const std::filesystem::path dir = tmp / (std::string("mimir_test_rawfolder_") + c.expected_tag);
        std::filesystem::remove_all(dir, ec);

        SaveOptions sopts;
        sopts.format = CheckpointFormat::RawFolder;
        sopts.save_tokenizer = true;
        sopts.save_encoder = true;

        std::string err;
        TASSERT_TRUE(save_checkpoint(*modelA, dir.string(), sopts, &err));

        // Metadata JSON should advertise expected dtype on at least one weights tensor.
        {
            const auto& layers = modelA->getLayers();
            std::string first_weight;
            for (const auto& layer : layers) {
                if (layer.weight_block && layer.weight_block->getSize() > 0) {
                    first_weight = layer.name + "_weights";
                    break;
                }
            }
            TASSERT_TRUE(!first_weight.empty());
            const std::filesystem::path meta = dir / "tensors" / (first_weight + ".json");
            json tmeta = read_json_file(meta);
            TASSERT_TRUE(tmeta.contains("dtype"));
            TASSERT_TRUE(tmeta["dtype"].get<std::string>() == c.expected_tag);
        }

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

        // DType should be restored from checkpoint config metadata.
        TASSERT_TRUE(modelB->getDefaultDType() == std::string(c.dtype));

        // Basic sanity: model config restored and embeddings present.
        TASSERT_TRUE(modelB->modelConfig.contains("type"));
        TASSERT_TRUE(modelB->getEncoder().dim == modelA->getEncoder().dim);
        TASSERT_TRUE(!modelB->getEncoder().token_embeddings.empty());

        // Weights should match (within dtype quantization error if any).
        TASSERT_TRUE(compare_model_weights(*modelA, *modelB, c.eps) == 0);

        std::filesystem::remove_all(dir, ec);
    }
    return 0;
}
