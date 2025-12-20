#include "../src/Tokenizer.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <string>

// Test 1: Construction et initialisation
TEST(TokenizerTest, Construction) {
    Tokenizer tok(1000);
    EXPECT_GT(tok.getVocabSize(), 0); // Devrait avoir au moins les tokens spéciaux
}

// Test 2: Tokens spéciaux
TEST(TokenizerTest, SpecialTokens) {
    Tokenizer tok;
    tok.ensureSpecialTokens();
    
    int pad_id = tok.getPadId();
    int unk_id = tok.getUnkId();
    int seq_id = tok.getSeqId();
    int mod_id = tok.getModId();
    
    EXPECT_GE(pad_id, 0);
    EXPECT_GE(unk_id, 0);
    EXPECT_GE(seq_id, 0);
    EXPECT_GE(mod_id, 0);
    
    // Vérifier que les tokens sont distincts
    EXPECT_NE(pad_id, unk_id);
    EXPECT_NE(pad_id, seq_id);
    EXPECT_NE(unk_id, mod_id);
}

// Test 3: Ajout de tokens
TEST(TokenizerTest, AddToken) {
    Tokenizer tok;
    size_t initial_size = tok.getVocabSize();
    
    int id1 = tok.addToken("hello");
    int id2 = tok.addToken("world");
    int id3 = tok.addToken("hello"); // Dupliquer devrait retourner le même ID
    
    EXPECT_GE(id1, 0);
    EXPECT_GE(id2, 0);
    EXPECT_EQ(id1, id3); // Même token = même ID
    EXPECT_NE(id1, id2);
    
    EXPECT_EQ(tok.getVocabSize(), initial_size + 2); // +2 nouveaux tokens
}

// Test 4: Tokenization basique
TEST(TokenizerTest, BasicTokenization) {
    Tokenizer tok;
    tok.addToken("hello");
    tok.addToken("world");
    
    std::vector<int> ids = tok.tokenize("hello world");
    EXPECT_GE(ids.size(), 2);
}

// Test 5: Tokenize avec auto-ajout
TEST(TokenizerTest, TokenizeEnsure) {
    Tokenizer tok;
    size_t initial_size = tok.getVocabSize();
    
    std::vector<int> ids = tok.tokenizeEnsure("new tokens here");
    EXPECT_FALSE(ids.empty());
    EXPECT_GT(tok.getVocabSize(), initial_size); // Devrait avoir ajouté des tokens
}

// Test 6: Décodage
TEST(TokenizerTest, Decode) {
    Tokenizer tok;
    int id1 = tok.addToken("hello");
    int id2 = tok.addToken("world");
    
    std::vector<int> ids = {id1, id2};
    std::string decoded = tok.decode(ids);
    
    // Le décodage devrait contenir les mots (avec espaces possibles)
    EXPECT_NE(decoded.find("hello"), std::string::npos);
    EXPECT_NE(decoded.find("world"), std::string::npos);
}

// Test 7: Vocab depuis texte
TEST(TokenizerTest, EnsureVocabFromText) {
    Tokenizer tok;
    size_t initial_size = tok.getVocabSize();
    
    std::string corpus = "this is a test corpus with multiple words and repeated words";
    tok.ensureVocabFromText(corpus);
    
    EXPECT_GT(tok.getVocabSize(), initial_size);
    
    // Vérifier que les mots principaux ont été ajoutés
    std::vector<int> ids = tok.tokenize("test words");
    EXPECT_FALSE(ids.empty());
}

// Test 8: Sérialisation JSON
TEST(TokenizerTest, JSONSerialization) {
    Tokenizer tok1;
    tok1.addToken("alpha");
    tok1.addToken("beta");
    tok1.addToken("gamma");
    
    // Export vers JSON
    json j = tok1.to_json();
    EXPECT_TRUE(j.contains("vocab"));
    // Note: max_vocab n'est pas toujours sérialisé selon l'implémentation
    
    // Import depuis JSON
    Tokenizer tok2;
    tok2.from_json(j);
    
    // Le vocabulaire doit au minimum contenir les tokens ajoutés
    EXPECT_GE(tok2.getVocabSize(), 3);
}

// Test 9: Normalisation de tokens
TEST(TokenizerTest, NormalizeToken) {
    Tokenizer tok;
    
    std::string normalized1 = tok.normalizeToken("  HELLO  ");
    std::string normalized2 = tok.normalizeToken("WoRlD");
    
    EXPECT_EQ(normalized1, "hello");
    EXPECT_EQ(normalized2, "world");
}

// Test 10: Padding de séquences
TEST(TokenizerTest, PadSequence) {
    Tokenizer tok;
    tok.setMaxSequenceLength(10);
    
    std::vector<int> short_seq = {1, 2, 3};
    std::vector<int> padded = tok.padSequence(short_seq, 10);
    
    EXPECT_EQ(padded.size(), 10);
    EXPECT_EQ(padded[0], 1);
    EXPECT_EQ(padded[1], 2);
    EXPECT_EQ(padded[2], 3);
    
    // Les éléments restants doivent être pad_id
    int pad_id = tok.getPadId();
    for (size_t i = 3; i < 10; ++i) {
        EXPECT_EQ(padded[i], pad_id);
    }
}

// Test 11: Batch tokenization
TEST(TokenizerTest, BatchTokenize) {
    Tokenizer tok;
    std::vector<std::string> texts = {
        "first sentence",
        "second longer sentence",
        "third"
    };
    
    auto batched = tok.batchTokenize(texts, 20);
    
    EXPECT_EQ(batched.size(), 3);
    EXPECT_EQ(batched[0].size(), 20);
    EXPECT_EQ(batched[1].size(), 20);
    EXPECT_EQ(batched[2].size(), 20);
}

// Test 12: Extraction de keywords
TEST(TokenizerTest, ExtractKeywords) {
    Tokenizer tok;
    std::string text = "machine learning is the future of artificial intelligence";
    
    auto keywords = tok.extractKeywords(text, 3);
    EXPECT_LE(keywords.size(), 3);
    EXPECT_FALSE(keywords.empty());
}
