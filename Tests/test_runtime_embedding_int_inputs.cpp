#include "test_utils.hpp"

#include "Model.hpp"

#include <unordered_map>
#include <vector>

static void fill_embedding_weights(Layer& layer) {
    const int vocab = layer.vocab_size;
    const int dim = layer.embed_dim;
    float* w = layer.getWeights();
    for (int id = 0; id < vocab; ++id) {
        for (int d = 0; d < dim; ++d) {
            w[static_cast<size_t>(id) * static_cast<size_t>(dim) + static_cast<size_t>(d)] =
                static_cast<float>(id * 10 + d);
        }
    }
}

int main() {
    Model m;

    const int vocab = 4;
    const int dim = 4;

    // One Embedding layer consuming int ids via forwardPassNamed(int_inputs).
    m.push("emb", "Embedding", static_cast<size_t>(vocab * dim));
    auto& L = m.getMutableLayers().back();
    L.vocab_size = vocab;
    L.embed_dim = dim;
    L.padding_idx = 0;
    L.seq_len = 5;
    L.inputs = {"ids"};
    L.output = "x";

    m.allocateParams();

    // Deterministic embedding table.
    fill_embedding_weights(L);

    std::unordered_map<std::string, std::vector<float>> fin;
    std::unordered_map<std::string, std::vector<int>> iin;

    // Provide ids as int tensor.
    iin["ids"] = {1, 0, 2};

    auto out = m.forwardPassNamed(fin, iin, /*training=*/false);
    TASSERT_TRUE(out.size() == 3u * static_cast<size_t>(dim));

    // id=1 => [10,11,12,13]
    TASSERT_NEAR(out[0], 10.0f, 1e-6f);
    TASSERT_NEAR(out[1], 11.0f, 1e-6f);
    TASSERT_NEAR(out[2], 12.0f, 1e-6f);
    TASSERT_NEAR(out[3], 13.0f, 1e-6f);

    // id=0 is padding => zeros
    TASSERT_NEAR(out[4], 0.0f, 1e-6f);
    TASSERT_NEAR(out[5], 0.0f, 1e-6f);
    TASSERT_NEAR(out[6], 0.0f, 1e-6f);
    TASSERT_NEAR(out[7], 0.0f, 1e-6f);

    // id=2 => [20,21,22,23]
    TASSERT_NEAR(out[8], 20.0f, 1e-6f);
    TASSERT_NEAR(out[9], 21.0f, 1e-6f);
    TASSERT_NEAR(out[10], 22.0f, 1e-6f);
    TASSERT_NEAR(out[11], 23.0f, 1e-6f);

    // Missing ids should not throw: runtime falls back to pad ids of length seq_len.
    iin.clear();
    out = m.forwardPassNamed(fin, iin, /*training=*/false);
    TASSERT_TRUE(out.size() == static_cast<size_t>(L.seq_len) * static_cast<size_t>(dim));
    for (float v : out) {
        TASSERT_NEAR(v, 0.0f, 1e-6f);
    }

    return 0;
}
