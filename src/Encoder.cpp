#include "Encoder.hpp"
#include <random>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <omp.h>
json Encoder::to_json() const {
    json j;
    j["dim"] = dim;
    j["vocab_size"] = vocab_size;
    // Ne pas inclure token_embeddings ici: c'est volumineux et doit être sauvegardé
    // via les formats binaires (SafeTensors/RawCheckpoint). On garde seulement des métadonnées.
    j["token_embeddings_size"] = token_embeddings.size();
    j["magik_prefix_count"] = magik_prefix_count;
    j["magik_prefix_weight"] = magik_prefix_weight;

    // Snapshot léger: embeddings des ids 0..(N-1) si disponibles.
    // Utile pour inspection/debug (les poids complets restent dans les tensors).
    if (!token_embeddings.empty() && dim > 0 && vocab_size > 0) {
        const int n = std::max(0, std::min(magik_prefix_count, vocab_size));
        if (n > 0) {
            json rows = json::array();
            for (int id = 0; id < n; ++id) {
                json row = json::array();
                const size_t base = static_cast<size_t>(id) * static_cast<size_t>(dim);
                for (int d = 0; d < dim; ++d) {
                    row.push_back(token_embeddings[base + static_cast<size_t>(d)]);
                }
                rows.push_back(row);
            }
            j["magik_token_embeddings_snapshot"] = rows;
        }
    }
    if (!seq_embedding.empty()) j["seq_embedding"] = seq_embedding;
    if (!mod_embedding.empty()) j["mod_embedding"] = mod_embedding;
    if (!mag_embedding.empty()) j["mag_embedding"] = mag_embedding;
    return j;
}

void Encoder::from_json(const json &j) {
    if (j.contains("dim")) dim = j["dim"].get<int>();
    if (j.contains("vocab_size")) vocab_size = j["vocab_size"].get<int>();

    if (j.contains("magik_prefix_count")) magik_prefix_count = j["magik_prefix_count"].get<int>();
    if (j.contains("magik_prefix_weight")) magik_prefix_weight = j["magik_prefix_weight"].get<float>();

    if (j.contains("token_embeddings") && j["token_embeddings"].is_array()) {
        token_embeddings = j["token_embeddings"].get<std::vector<float>>();
    }

    if (j.contains("seq_embedding") && j["seq_embedding"].is_array()) {
        seq_embedding = j["seq_embedding"].get<std::vector<float>>();
    }
    if (j.contains("mod_embedding") && j["mod_embedding"].is_array()) {
        mod_embedding = j["mod_embedding"].get<std::vector<float>>();
    }
    if (j.contains("mag_embedding") && j["mag_embedding"].is_array()) {
        mag_embedding = j["mag_embedding"].get<std::vector<float>>();
    }

    // Cohérence minimale
    if (dim <= 0) throw std::runtime_error("Encoder::from_json: invalid dim");
    if (vocab_size < 0) throw std::runtime_error("Encoder::from_json: invalid vocab_size");
    if (magik_prefix_count < 0) magik_prefix_count = 0;
    if (!(magik_prefix_weight > 0.0f)) magik_prefix_weight = 1.0f;
    if (!token_embeddings.empty()) {
        size_t expected = static_cast<size_t>(std::max(0, vocab_size)) * static_cast<size_t>(dim);
        if (expected > 0 && token_embeddings.size() != expected) {
            // Tolérance: si vocab_size n'était pas fiable, on l'infère.
            if (token_embeddings.size() % static_cast<size_t>(dim) == 0) {
                vocab_size = static_cast<int>(token_embeddings.size() / static_cast<size_t>(dim));
            }
        }
    }
}
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
    // Note: RNG n'est pas thread-safe, on garde séquentiel ici
    for (size_t i = 0; i < token_embeddings.size(); ++i) {
        token_embeddings[i] = dist(rng);
    }
}

void Encoder::ensureSpecialEmbeddings(uint64_t seed)
{
    if (dim <= 0) throw std::runtime_error("Encoder::ensureSpecialEmbeddings: invalid dim");
    std::mt19937 rng(static_cast<uint32_t>(seed ^ 0xA5A5A5A5u));
    std::uniform_real_distribution<float> dist(-0.02f, 0.02f);

    auto init_if_empty = [&](std::vector<float>& emb, uint32_t tweak) {
        if (emb.size() == static_cast<size_t>(dim)) return;
        emb.assign(static_cast<size_t>(dim), 0.0f);
        std::mt19937 lrng(rng());
        lrng.seed(static_cast<uint32_t>(seed ^ tweak));
        for (int d = 0; d < dim; ++d) {
            emb[static_cast<size_t>(d)] = dist(lrng);
        }
    };

    init_if_empty(seq_embedding, 0x13579BDFu);
    init_if_empty(mod_embedding, 0x2468ACE0u);
    init_if_empty(mag_embedding, 0x0F0F0F0Fu);
}

void Encoder::sgdUpdateSpecialEmbeddings(const std::vector<float>& grad_text, float lr,
                                        bool update_seq, bool update_mod, bool update_mag)
{
    if (lr == 0.0f) return;
    if (grad_text.size() != static_cast<size_t>(dim)) return;

    const auto apply = [&](std::vector<float>& emb) {
        if (emb.size() != static_cast<size_t>(dim)) return;
        #pragma omp simd
        for (int d = 0; d < dim; ++d) {
            emb[static_cast<size_t>(d)] -= lr * grad_text[static_cast<size_t>(d)];
        }
    };

    if (update_seq) apply(seq_embedding);
    if (update_mod) apply(mod_embedding);
    if (update_mag) apply(mag_embedding);
}

void Encoder::ensureVocabSize(size_t new_vocab_size, uint64_t seed)
{
    if (new_vocab_size <= static_cast<size_t>(vocab_size)) return;
    size_t old = static_cast<size_t>(vocab_size);
    size_t need = new_vocab_size - old;
    token_embeddings.resize(static_cast<size_t>(new_vocab_size) * static_cast<size_t>(dim));
    // init new embeddings randomly (séquentiel pour RNG)
    std::mt19937 rng(static_cast<uint32_t>(seed ^ 0xC0FFEEu));
    std::uniform_real_distribution<float> dist(-0.02f, 0.02f);
    size_t start = old * static_cast<size_t>(dim);
    size_t end = token_embeddings.size();
    for (size_t i = start; i < end; ++i) {
        token_embeddings[i] = dist(rng);
    }
    vocab_size = static_cast<int>(new_vocab_size);
}

void Encoder::setMagicFromToken(const MagicToken & /*mt*/)
{
    // Best-effort: pas d'initialisation implicite. L'appelant peut définir via setMagEmbedding.
    mag_embedding.clear();
}

void Encoder::setSeqEmbedding(const std::vector<float> &v)
{
    if (v.empty()) {
        seq_embedding.clear();
        return;
    }
    seq_embedding.assign(static_cast<size_t>(dim), 0.0f);
    size_t n = std::min<size_t>(v.size(), static_cast<size_t>(dim));
    std::copy_n(v.data(), n, seq_embedding.data());
}

void Encoder::setModEmbedding(const std::vector<float> &v)
{
    if (v.empty()) {
        mod_embedding.clear();
        return;
    }
    mod_embedding.assign(static_cast<size_t>(dim), 0.0f);
    size_t n = std::min<size_t>(v.size(), static_cast<size_t>(dim));
    std::copy_n(v.data(), n, mod_embedding.data());
}

void Encoder::setMagEmbedding(const std::vector<float> &v)
{
    if (v.empty()) {
        mag_embedding.clear();
        return;
    }
    mag_embedding.assign(static_cast<size_t>(dim), 0.0f);
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

    float weight_sum = 0.0f;
    for (size_t pos = 0; pos < tokens.size(); ++pos) {
        const int id = tokens[pos];
        if (id <= 0) continue; // PAD(0) + ids négatifs
        if (static_cast<size_t>(id) >= static_cast<size_t>(vocab_size)) continue;

        const float w = (static_cast<int>(pos) < magik_prefix_count && magik_prefix_weight > 0.0f) ? magik_prefix_weight : 1.0f;
        const float *row = token_embeddings.data() + (static_cast<size_t>(id) * static_cast<size_t>(dim));
        #pragma omp simd
        for (int d = 0; d < dim; ++d) {
            out[static_cast<size_t>(d)] += row[static_cast<size_t>(d)] * w;
        }
        weight_sum += w;
    }

    if (weight_sum > 0.0f) {
        float inv = 1.0f / weight_sum;
        #pragma omp simd
        for (int i = 0; i < dim; ++i) out[i] *= inv;
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
        #pragma omp simd
        for (int d = 0; d < dim; ++d) {
            float err = target[static_cast<size_t>(d)] - row[static_cast<size_t>(d)];
            row[static_cast<size_t>(d)] += lr * err;
        }
    }
}