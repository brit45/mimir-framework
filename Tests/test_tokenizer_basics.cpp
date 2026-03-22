#include "test_utils.hpp"

#include "Tokenizer.hpp"

int main() {
    Tokenizer tok(128);

    // Special tokens should exist and be stable.
    TASSERT_TRUE(tok.getPadId() == 0);
    TASSERT_TRUE(tok.getUnkId() >= 0);
    TASSERT_TRUE(tok.getBosId() >= 0);
    TASSERT_TRUE(tok.getEosId() >= 0);

    const size_t base_vocab = tok.getVocabSize();

    // tokenize() does not add new tokens; unknown words map to <UNK>.
    {
        auto ids = tok.tokenize("hello world");
        TASSERT_TRUE(ids.size() >= 1);
        for (int id : ids) {
            TASSERT_TRUE(id == tok.getUnkId() || id == tok.getPadId() || id >= 0);
        }
    }

    // tokenizeEnsure() should add tokens to vocab.
    {
        auto ids = tok.tokenizeEnsure("hello world");
        TASSERT_TRUE(ids.size() >= 2);
        TASSERT_TRUE(tok.getVocabSize() >= base_vocab + 2);
        std::string decoded = tok.decode(ids);
        TASSERT_TRUE(!decoded.empty());
    }

    // Padding / max sequence length behavior.
    tok.setMaxSequenceLength(4);
    {
        auto ids = tok.tokenizeEnsure("a b c");
        auto padded = tok.padSequence(ids, 4);
        TASSERT_TRUE(padded.size() == 4);

        // If BOS/EOS are enabled, sequence is wrapped and padded in-between.
        TASSERT_TRUE(padded.front() == tok.getBosId());
        TASSERT_TRUE(padded.back() == tok.getEosId());

        // If shorter, must include PAD somewhere (between BOS and EOS).
        auto short_ids = tok.tokenizeEnsure("x");
        auto short_padded = tok.padSequence(short_ids, 4);
        TASSERT_TRUE(short_padded.size() == 4);
        TASSERT_TRUE(short_padded.front() == tok.getBosId());
        TASSERT_TRUE(short_padded.back() == tok.getEosId());
        bool has_pad = false;
        for (size_t i = 0; i < short_padded.size(); ++i) {
            if (short_padded[i] == tok.getPadId()) { has_pad = true; break; }
        }
        TASSERT_TRUE(has_pad);
    }

    // Batch tokenize: fixed width per sample.
    {
        std::vector<std::string> texts = {"one two", "three"};
        auto batch = tok.batchTokenize(texts, 8);
        TASSERT_TRUE(batch.size() == texts.size());
        TASSERT_TRUE(batch[0].size() == 8);
        TASSERT_TRUE(batch[1].size() == 8);
    }

    return 0;
}
