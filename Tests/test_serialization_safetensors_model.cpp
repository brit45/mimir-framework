#include "test_utils.hpp"

#include "Models/Registry/ModelArchitectures.hpp"
#include "Serialization/Serialization.hpp"
#include "DType.hpp"

#include <algorithm>
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

static uint64_t read_u64_le(std::ifstream& f) {
    uint8_t bytes[8];
    f.read(reinterpret_cast<char*>(bytes), 8);
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= (static_cast<uint64_t>(bytes[i]) << (i * 8));
    }
    return value;
}

static json read_safetensors_header_json(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    TASSERT_TRUE((bool)f);
    const uint64_t header_len = read_u64_le(f);
    TASSERT_TRUE(header_len > 0);
    std::string header;
    header.resize(static_cast<size_t>(header_len));
    f.read(header.data(), static_cast<std::streamsize>(header.size()));
    TASSERT_TRUE((bool)f);
    return json::parse(header);
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

    struct Case {
        const char* dtype;
        const char* expected_tag;
        float eps;
    };
    const Case cases[] = {
        {"float32", "F32", 1e-6f},
        {"f32", "F32", 1e-6f},
        {"float16", "F16", 5e-3f},
        {"f16", "F16", 5e-3f},
        {"fp16", "F16", 5e-3f},
        {"bfloat16", "BF16", 2e-2f},
        {"bf16", "BF16", 2e-2f},
        {"float64", "F64", 1e-6f},
    };

    const std::filesystem::path tmp = std::filesystem::temp_directory_path();

    for (const auto& c : cases) {
        json cfgA = cfg;
        cfgA["dtype"] = c.dtype;

        auto modelA = ModelArchitectures::create("basic_mlp", cfgA);
        TASSERT_TRUE(modelA != nullptr);
        TASSERT_TRUE(modelA->getDefaultDType() == std::string(c.dtype));
        modelA->allocateParams();
        modelA->initializeWeights("xavier", 123u);

        // Attach deterministic-ish tokenizer/encoder state (ensures JSON + embeddings tensors exist).
        Tokenizer tok(128);
        tok.setMaxSequenceLength(16);
        tok.tokenizeEnsure("hello world");

        ConditioningEncoder enc(8, 128);
        enc.ensureVocabSize(tok.getVocabSize(), 777u);
        enc.ensureSpecialEmbeddings(42u);

        modelA->setTokenizer(tok);
        modelA->setEncoder(enc);
        modelA->setHasEncoder(true);

        const std::filesystem::path p = tmp / (std::string("mimir_test_model_") + c.expected_tag + ".safetensors");
        std::filesystem::remove(p);

        SaveOptions sopts;
        sopts.format = CheckpointFormat::SafeTensors;
        sopts.save_tokenizer = true;
        sopts.save_encoder = true;
        sopts.save_optimizer = false;

        std::string err;
        TASSERT_TRUE(save_checkpoint(*modelA, p.string(), sopts, &err));

        // Header should advertise expected dtype on at least one weights tensor.
        {
            const auto& layers = modelA->getLayers();
            std::string first_weight;
            for (const auto& layer : layers) {
                if (layer.weight_block && layer.weight_block->getSize() > 0) {
                    first_weight = layer.name + "/weights";
                    break;
                }
            }
            TASSERT_TRUE(!first_weight.empty());
            json header = read_safetensors_header_json(p);
            TASSERT_TRUE(header.contains(first_weight));
            TASSERT_TRUE(header[first_weight].contains("dtype"));
            const std::string dt = header[first_weight]["dtype"].get<std::string>();
            TASSERT_TRUE(dt == c.expected_tag);
        }

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

        // DType should be restored semantically (aliases allowed: f16/fp16/float16...).
        TASSERT_TRUE(Mimir::parse_dtype(modelB->getDefaultDType()) == Mimir::parse_dtype(c.dtype));

        // Weights should match (within dtype quantization error if any).
        TASSERT_TRUE(compare_model_weights(*modelA, *modelB, c.eps) == 0);

        // Tokenizer/encoder should have been restored.
        TASSERT_TRUE(modelB->getTokenizer().getVocabSize() == modelA->getTokenizer().getVocabSize());
        TASSERT_TRUE(modelB->getEncoder().dim == modelA->getEncoder().dim);
        TASSERT_TRUE(modelB->getEncoder().vocab_size == modelA->getEncoder().vocab_size);
        TASSERT_TRUE(!modelB->getEncoder().token_embeddings.empty());

        std::filesystem::remove(p);
    }
    return 0;
}
