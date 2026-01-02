#ifndef __TENSOR_ENCODER_HPP__
#define __TENSOR_ENCODER_HPP__

#include "Helpers.hpp"
#include "include/json.hpp"
using json = nlohmann::json;


class Encoder
{
    public:

        Encoder(int d = 64, int Size_Vo = 4096);
        ~Encoder();

        void initRandom(uint64_t seed = 0);

        // ensure token_embeddings can cover new_vocab_size; initialize new embeddings randomly
        void ensureVocabSize(size_t new_vocab_size, uint64_t seed = 0xC0FFEEu);

        // set mag embedding from MagicToken (map 8-d embed -> encoder dim)
        void setMagicFromToken(const MagicToken &mt);

        // expose special embeddings (optional)
        void setSeqEmbedding(const std::vector<float> &v);
        void setModEmbedding(const std::vector<float> &v);
        void setMagEmbedding(const std::vector<float> &v);

        // encode: skip PAD(0), add special embeddings for SEQ/MOD/MAG if present
        std::vector<float> encode(const std::vector<int> &tokens, uint32_t /*seed*/ = 0) const;

        // train embeddings for token ids toward target embedding
        // special tokens are excluded from direct token_embeddings updates (except normal tokens)
        void trainOnTextTokens(const std::vector<int> &token_ids, const std::vector<float> &target, float lr = 0.01f);

        // expose for checkpointing
        int dim;
        int vocab_size;
        std::vector<float> token_embeddings;

        // Serialization (RawFolder + SafeTensors metadata)
        json to_json() const;
        void from_json(const json &j);

    private:
        // special embeddings handled separately
        std::vector<float> seq_embedding;
        std::vector<float> mod_embedding;
        std::vector<float> mag_embedding;
};

#endif //! __TENSOR_ENCODER_HPP__