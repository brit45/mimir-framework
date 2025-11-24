#include "Encoder.hpp"
#include <random>
#include <algorithm>
#include <cstring>

Encoder::Encoder(int d, int Size_Vo)
    : dim(d), vocab_size(0)
{
    if (dim <= 0) dim = 64;
    if (Size_Vo > 0) token_embeddings.reserve(static_cast<size_t>(Size_Vo) * static_cast<size_t>(dim));
    // leave vocab_size == 0 until ensureVocabSize is called
}

Encoder::~Encoder() = default;

void Encoder::initRandom(uint64_t seed)
{
    std::mt19937 rng(static_cast<uint32_t>(seed ^ 0x9e3779b9u));
    std::uniform_real_distribution<float> dist(-0.02f, 0.02f);
    if (vocab_size == 0) return;
    for (size_t i = 0; i < token_embeddings.size(); ++i) token_embeddings[i] = dist(rng);
    // init special embeddings
    seq_embedding.assign(dim, 0.0f);
    mod_embedding.assign(dim, 0.0f);
    mag_embedding.assign(dim, 0.0f);
}

void Encoder::ensureVocabSize(size_t new_vocab_size, uint64_t seed)
{
    if (new_vocab_size <= static_cast<size_t>(vocab_size)) return;
    size_t old = static_cast<size_t>(vocab_size);
    size_t need = new_vocab_size - old;
    token_embeddings.resize(static_cast<size_t>(new_vocab_size) * static_cast<size_t>(dim));
    // init new embeddings randomly
    std::mt19937 rng(static_cast<uint32_t>(seed ^ 0xC0FFEEu));
    std::uniform_real_distribution<float> dist(-0.02f, 0.02f);
    for (size_t i = old * static_cast<size_t>(dim); i < token_embeddings.size(); ++i)
        token_embeddings[i] = dist(rng);
    vocab_size = static_cast<int>(new_vocab_size);
}

void Encoder::setMagicFromToken(const MagicToken & /*mt*/)
{
    // Best-effort: initialize mag_embedding to zeros (user can override via setMagEmbedding)
    mag_embedding.assign(static_cast<size_t>(dim), 0.0f);
    // If MagicToken structure is richer, user may call setMagEmbedding after constructing the mapped vector.
}

void Encoder::setSeqEmbedding(const std::vector<float> &v)
{
    seq_embedding.assign(static_cast<size_t>(dim), 0.0f);
    if (v.empty()) return;
    size_t n = std::min<size_t>(v.size(), static_cast<size_t>(dim));
    std::copy_n(v.data(), n, seq_embedding.data());
}

void Encoder::setModEmbedding(const std::vector<float> &v)
{
    mod_embedding.assign(static_cast<size_t>(dim), 0.0f);
    if (v.empty()) return;
    size_t n = std::min<size_t>(v.size(), static_cast<size_t>(dim));
    std::copy_n(v.data(), n, mod_embedding.data());
}

void Encoder::setMagEmbedding(const std::vector<float> &v)
{
    mag_embedding.assign(static_cast<size_t>(dim), 0.0f);
    if (v.empty()) return;
    size_t n = std::min<size_t>(v.size(), static_cast<size_t>(dim));
    std::copy_n(v.data(), n, mag_embedding.data());
}

std::vector<float> Encoder::encode(const std::vector<int> &tokens, uint32_t /*seed*/) const
{
    std::vector<float> out(static_cast<size_t>(dim), 0.0f);
    if (tokens.empty()) {
        // add special embeddings if present
        for (int i = 0; i < dim; ++i) {
            if (seq_embedding.size() == static_cast<size_t>(dim)) out[i] += seq_embedding[i];
            if (mod_embedding.size() == static_cast<size_t>(dim)) out[i] += mod_embedding[i];
            if (mag_embedding.size() == static_cast<size_t>(dim)) out[i] += mag_embedding[i];
        }
        return out;
    }

    size_t count = 0;
    for (int id : tokens) {
        if (id < 0) continue;
        if (static_cast<size_t>(id) >= static_cast<size_t>(vocab_size)) continue;
        const float *row = token_embeddings.data() + (static_cast<size_t>(id) * static_cast<size_t>(dim));
        for (int d = 0; d < dim; ++d) out[static_cast<size_t>(d)] += row[static_cast<size_t>(d)];
        ++count;
    }

    if (count > 0) {
        float inv = 1.0f / float(count);
        for (auto &x : out) x *= inv;
    }

    // add special embeddings (if set)
    if (seq_embedding.size() == static_cast<size_t>(dim)) {
        for (int d = 0; d < dim; ++d) out[static_cast<size_t>(d)] += seq_embedding[static_cast<size_t>(d)];
    }
    if (mod_embedding.size() == static_cast<size_t>(dim)) {
        for (int d = 0; d < dim; ++d) out[static_cast<size_t>(d)] += mod_embedding[static_cast<size_t>(d)];
    }
    if (mag_embedding.size() == static_cast<size_t>(dim)) {
        for (int d = 0; d < dim; ++d) out[static_cast<size_t>(d)] += mag_embedding[static_cast<size_t>(d)];
    }

    return out;
}

void Encoder::trainOnTextTokens(const std::vector<int> &token_ids, const std::vector<float> &target, float lr)
{
    if (token_ids.empty() || target.size() != static_cast<size_t>(dim) || lr == 0.0f) return;
    for (int id : token_ids) {
        if (id < 0) continue;
        // assume special tokens are at low ids (0..4) — do not train them
        if (id <= 4) continue;
        if (static_cast<size_t>(id) >= static_cast<size_t>(vocab_size)) continue;
        float *row = token_embeddings.data() + (static_cast<size_t>(id) * static_cast<size_t>(dim));
        for (int d = 0; d < dim; ++d) {
            float err = target[static_cast<size_t>(d)] - row[static_cast<size_t>(d)];
            row[static_cast<size_t>(d)] += lr * err;
        }
    }
}