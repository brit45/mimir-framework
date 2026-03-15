#pragma once

#include <cstdint>
#include <random>

// Small RNG context used to make stochastic ops deterministic when a seed is provided.
// Thread-local so it works with OpenMP threads and avoids cross-call interference.
namespace MimirRng {

inline thread_local bool g_seeded = false;
inline thread_local bool g_initialized = false;
inline thread_local std::mt19937 g_gen;

inline void setSeed(uint32_t seed) {
    g_gen.seed(seed);
    g_seeded = true;
    g_initialized = true;
}

inline void clearSeed() {
    g_seeded = false;
}

inline bool hasSeed() {
    return g_seeded;
}

inline std::mt19937& generator() {
    if (!g_initialized) {
        std::random_device rd;
        g_gen.seed(rd());
        g_initialized = true;
    }
    return g_gen;
}

class ScopedSeed {
public:
    explicit ScopedSeed(uint32_t seed) : prev_seeded_(g_seeded) {
        // Note: we don't restore the previous generator state (only whether a seed was active).
        // This keeps behavior predictable across calls.
        setSeed(seed);
    }

    ~ScopedSeed() {
        g_seeded = prev_seeded_;
    }

    ScopedSeed(const ScopedSeed&) = delete;
    ScopedSeed& operator=(const ScopedSeed&) = delete;

private:
    bool prev_seeded_;
};

} // namespace MimirRng
