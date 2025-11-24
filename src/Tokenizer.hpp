#ifndef __TENSOR_TOKENIZER_HPP__
#define __TENSOR_TOKENIZER_HPP__

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstddef>
#include "include/json.hpp"

using json = nlohmann::json;

class Tokenizer {
public:
    Tokenizer(size_t max_vocab = 4096);

    // vocab manipulation
    int addToken(const std::string &tok);
    void ensureVocabFromText(const std::string &text);

    // tokenization
    std::vector<int> tokenize(const std::string &text) const;
    std::vector<int> tokenizeEnsure(const std::string &text);

    // decode / serialize
    std::string decode(const std::vector<int> &ids) const;
    json to_json() const;
    void from_json(const json &j);

    // special token ids
    int getPadId() const;
    int getUnkId() const;
    int getSeqId() const;
    int getModId() const;
    int getMagId() const;

    // accessors used elsewhere in the project
    size_t getVocabSize() const;
    std::string getTokenById(int id) const;

    // helpers (public for tests)
    void ensureSpecialTokens();
    std::vector<std::string> splitTokens(const std::string &text) const;
    std::string normalizeToken(std::string s) const;

    // Méthodes améliorées pour FLUX
    void learnBPEFromCorpus(const std::vector<std::string>& corpus, int num_merges = 1000);
    std::vector<int> tokenizeBPE(const std::string& text) const;
    void setMaxSequenceLength(int max_len);
    std::vector<int> padSequence(const std::vector<int>& tokens, int target_len = -1) const;
    std::vector<std::vector<int>> batchTokenize(const std::vector<std::string>& texts, int max_len = 512);
    
    // Statistiques
    void printVocabStats() const;
    std::unordered_map<std::string, int> getTokenFrequencies(const std::string& text) const;

    // Compréhension textuelle
    struct TextUnderstanding {
        std::vector<std::string> entities;        // Entités détectées (noms, objets, concepts)
        std::vector<std::string> modifiers;       // Modificateurs (adjectifs, adverbes)
        std::vector<std::string> actions;         // Actions/verbes
        std::unordered_map<std::string, float> sentiment;  // Analyse de sentiment par mot
        std::vector<std::pair<std::string, std::string>> relations;  // Relations entre concepts
        std::string mainSubject;                  // Sujet principal
        std::string context;                      // Contexte général
        int complexity;                           // Complexité du texte (0-10)
    };
    
    TextUnderstanding analyzeText(const std::string& text) const;
    std::vector<std::string> extractKeywords(const std::string& text, int topN = 5) const;
    std::vector<std::string> detectEntities(const std::string& text) const;
    std::string inferContext(const std::string& text) const;
    float computeTextComplexity(const std::string& text) const;
    std::unordered_map<std::string, float> analyzeSentiment(const std::string& text) const;
    
private:
    size_t maxVocab = 4096;
    std::unordered_map<std::string, int> vocab;  // token -> id (structure principale)
    std::unordered_map<int, std::string> reverse_vocab;  // id -> token (cache inverse)
    size_t vocabSize = 0;
    int maxSequenceLength = 512;
    
    struct PairHash {
        size_t operator()(const std::pair<std::string, std::string>& p) const {
            return std::hash<std::string>()(p.first) ^ (std::hash<std::string>()(p.second) << 1);
        }
    };
    
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> bpeMerges;
    
    // Dictionnaires pour la compréhension textuelle
    void initializeUnderstandingDictionaries();
    std::unordered_map<std::string, std::string> entityTypes;     // mot -> type (person, object, color, etc.)
    std::unordered_map<std::string, float> sentimentScores;       // mot -> score (-1.0 à 1.0)
    std::unordered_set<std::string> stopWords;                    // mots vides
    std::unordered_set<std::string> actionVerbs;                  // verbes d'action
};

#endif // __TENSOR_TOKENIZER_HPP__