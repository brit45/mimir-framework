#include "test_utils.hpp"

#include "Tokenizer.hpp"

int main() {
    Tokenizer tok(256);

    // Add tokens containing control chars; they must survive JSON round-trip.
    const int id_nl = tok.addToken("\n");
    const int id_tab = tok.addToken("\t");

    TASSERT_TRUE(id_nl >= 0);
    TASSERT_TRUE(id_tab >= 0);
    TASSERT_TRUE(id_nl != id_tab);

    const size_t vs_before = tok.getVocabSize();

    json j = tok.to_json();

    Tokenizer dst(1);
    dst.from_json(j);

    TASSERT_TRUE(dst.getVocabSize() == vs_before);
    TASSERT_TRUE(dst.getTokenById(id_nl) == "\n");
    TASSERT_TRUE(dst.getTokenById(id_tab) == "\t");

    return 0;
}
