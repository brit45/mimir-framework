#include "test_utils.hpp"

#include "Encoder.hpp"

#include <vector>

static void fill_embeddings(Encoder& enc) {
    // Fill token embeddings with deterministic values: emb[id][d] = id*10 + d.
    for (int id = 0; id < enc.vocab_size; ++id) {
        for (int d = 0; d < enc.dim; ++d) {
            enc.token_embeddings[static_cast<size_t>(id) * static_cast<size_t>(enc.dim) + static_cast<size_t>(d)] =
                static_cast<float>(id * 10 + d);
        }
    }
}

int main() {
    // 1) ensureDim should resize safely when empty.
    {
        Encoder enc(4, 16);
        enc.ensureDim(6, 123);
        TASSERT_TRUE(enc.dim == 6);
        enc.ensureSpecialEmbeddings(456);
        TASSERT_TRUE(enc.getSeqEmbedding().size() == static_cast<size_t>(enc.dim));
        TASSERT_TRUE(enc.getModEmbedding().size() == static_cast<size_t>(enc.dim));
        TASSERT_TRUE(enc.getMagEmbedding().size() == static_cast<size_t>(enc.dim));
    }

    // 2) encode() deterministic averaging with magik prefix weighting.
    {
        Encoder enc(4, 32);
        enc.ensureVocabSize(5, 123);
        enc.ensureSpecialEmbeddings(123);
        enc.setSeqEmbedding({});
        enc.setModEmbedding({});
        enc.setMagEmbedding({});

        fill_embeddings(enc);

        enc.magik_prefix_count = 1;
        enc.magik_prefix_weight = 2.0f;

        // tokens: [1, 2] => out = (2*emb1 + 1*emb2) / 3
        std::vector<int> toks = {1, 2};
        auto out = enc.encode(toks);
        TASSERT_TRUE(out.size() == static_cast<size_t>(enc.dim));

        for (int d = 0; d < enc.dim; ++d) {
            const float e1 = static_cast<float>(1 * 10 + d);
            const float e2 = static_cast<float>(2 * 10 + d);
            const float expected = (2.0f * e1 + 1.0f * e2) / 3.0f;
            TASSERT_NEAR(out[static_cast<size_t>(d)], expected, 1e-6f);
        }
    }

    // 3) JSON round-trip preserves metadata and special embeddings.
    {
        Encoder enc(8, 64);
        enc.ensureVocabSize(10, 42);
        enc.ensureSpecialEmbeddings(7);
        enc.magik_prefix_count = 3;
        enc.magik_prefix_weight = 1.5f;

        // Make special embeddings deterministic.
        std::vector<float> seq(8, 0.1f);
        std::vector<float> mod(8, 0.2f);
        std::vector<float> mag(8, 0.3f);
        enc.setSeqEmbedding(seq);
        enc.setModEmbedding(mod);
        enc.setMagEmbedding(mag);

        json j = enc.to_json();

        Encoder dst(1, 1);
        dst.from_json(j);

        TASSERT_TRUE(dst.dim == 8);
        TASSERT_TRUE(dst.vocab_size == 10);
        TASSERT_TRUE(dst.magik_prefix_count == 3);
        TASSERT_NEAR(dst.magik_prefix_weight, 1.5f, 1e-6f);
        TASSERT_TRUE(dst.getSeqEmbedding().size() == 8);
        TASSERT_TRUE(dst.getModEmbedding().size() == 8);
        TASSERT_TRUE(dst.getMagEmbedding().size() == 8);
        TASSERT_NEAR(dst.getSeqEmbedding()[0], 0.1f, 1e-6f);
        TASSERT_NEAR(dst.getModEmbedding()[0], 0.2f, 1e-6f);
        TASSERT_NEAR(dst.getMagEmbedding()[0], 0.3f, 1e-6f);
    }

    return 0;
}
