#include "test_utils.hpp"

#include "Tokenizer.hpp"

int main() {
    Tokenizer tok(512);

    const size_t v0 = tok.getVocabSize();

    // Train BPE on a tiny corpus with a very frequent pair: foo + bar.
    std::vector<std::string> corpus = {
        "foo bar foo bar foo bar",
        "foo bar baz",
        "foo bar",
    };

    tok.learnBPEFromCorpus(corpus, 10);
    const size_t v1 = tok.getVocabSize();

    // learnBPEFromCorpus should add at least one merged token when frequency>=2.
    TASSERT_TRUE(v1 > v0);

    // The merged token is a concatenation without separator: "foobar".
    {
        auto ids = tok.tokenizeBPE("foobar");
        TASSERT_TRUE(ids.size() >= 1);
        // Should not map to <UNK> if merge created token.
        TASSERT_TRUE(ids[0] != tok.getUnkId());
    }

    return 0;
}
