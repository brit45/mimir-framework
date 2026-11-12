#pragma once
#include <cstddef>

namespace CpuAttention {
// Q[q,embed], K/V[keys,embed] may be slices of combined QKV projections.
// dAtt, attended and dQ/dK/dV are row-major with stride embed. All results
// overwrite their buffers. Inputs and outputs must be disjoint and complete.
// Uses the existing native attention softmax/mask and separate FP32 operations.
void backward(const float* q, const float* k, const float* v,
    std::size_t qStride, std::size_t kvStride, const float* dAtt,
    float* attended, float* dQ, float* dK, float* dV,
    std::size_t queries, std::size_t keys, std::size_t embed,
    std::size_t heads, bool causal, int workers);
}
