#ifndef __SIMD_OPS_HPP__
#define __SIMD_OPS_HPP__

#include <immintrin.h>  // AVX2
#include <cstddef>
#include <cstring>
#include <algorithm>

namespace SIMD {

// Matrix multiplication optimisée AVX2: C = A @ B
// A: [M x K], B: [K x N], C: [M x N]
inline void matmul_avx2(float* C, const float* A, const float* B, 
                        size_t M, size_t N, size_t K) {
    // Zéro init
    std::memset(C, 0, M * N * sizeof(float));
    
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; j += 8) {  // Process 8 floats at once with AVX2
            // Vérifier qu'on ne dépasse pas
            size_t limit = std::min(j + 8, N);
            size_t remaining = limit - j;
            
            __m256 sum = _mm256_setzero_ps();
            
            for (size_t k = 0; k < K; ++k) {
                __m256 a_vec = _mm256_set1_ps(A[i * K + k]);
                
                if (remaining == 8) {
                    __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                    sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
                } else {
                    // Traiter les éléments restants
                    float b_vals[8] = {0};
                    for (size_t r = 0; r < remaining; ++r) {
                        b_vals[r] = B[k * N + j + r];
                    }
                    __m256 b_vec = _mm256_loadu_ps(b_vals);
                    sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
                }
            }
            
            // Store result
            if (remaining == 8) {
                _mm256_storeu_ps(&C[i * N + j], sum);
            } else {
                float result[8];
                _mm256_storeu_ps(result, sum);
                for (size_t r = 0; r < remaining; ++r) {
                    C[i * N + j + r] = result[r];
                }
            }
        }
    }
}

// Matrix transpose multiplication optimisée: C = A @ B^T
// A: [M x K], B: [N x K], C: [M x N]
inline void matmul_transpose_avx2(float* C, const float* A, const float* B,
                                   size_t M, size_t N, size_t K) {
    std::memset(C, 0, M * N * sizeof(float));
    
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            __m256 sum = _mm256_setzero_ps();
            
            size_t k = 0;
            // Process 8 elements at a time
            for (; k + 8 <= K; k += 8) {
                __m256 a_vec = _mm256_loadu_ps(&A[i * K + k]);
                __m256 b_vec = _mm256_loadu_ps(&B[j * K + k]);
                sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
            }
            
            // Horizontal sum
            __m128 sum_high = _mm256_extractf128_ps(sum, 1);
            __m128 sum_low = _mm256_castps256_ps128(sum);
            __m128 sum128 = _mm_add_ps(sum_low, sum_high);
            __m128 shuf = _mm_movehdup_ps(sum128);
            __m128 sums = _mm_add_ps(sum128, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            float result = _mm_cvtss_f32(sums);
            
            // Process remaining elements
            for (; k < K; ++k) {
                result += A[i * K + k] * B[j * K + k];
            }
            
            C[i * N + j] = result;
        }
    }
}

// Element-wise operations avec AVX2
inline void add_vectors_avx2(float* C, const float* A, const float* B, size_t N) {
    #pragma omp parallel for
    for (size_t i = 0; i < N; i += 8) {
        if (i + 8 <= N) {
            __m256 a = _mm256_loadu_ps(&A[i]);
            __m256 b = _mm256_loadu_ps(&B[i]);
            __m256 c = _mm256_add_ps(a, b);
            _mm256_storeu_ps(&C[i], c);
        } else {
            // Fallback pour les derniers éléments
            for (size_t j = i; j < N; ++j) {
                C[j] = A[j] + B[j];
            }
        }
    }
}

inline void mul_vectors_avx2(float* C, const float* A, const float* B, size_t N) {
    #pragma omp parallel for
    for (size_t i = 0; i < N; i += 8) {
        if (i + 8 <= N) {
            __m256 a = _mm256_loadu_ps(&A[i]);
            __m256 b = _mm256_loadu_ps(&B[i]);
            __m256 c = _mm256_mul_ps(a, b);
            _mm256_storeu_ps(&C[i], c);
        } else {
            for (size_t j = i; j < N; ++j) {
                C[j] = A[j] * B[j];
            }
        }
    }
}

// GELU activation avec AVX2
inline void gelu_forward_avx2(float* out, const float* in, size_t N) {
    const float c1 = 0.7978845608f;  // sqrt(2/pi)
    const float c2 = 0.044715f;
    
    #pragma omp parallel for
    for (size_t i = 0; i < N; i += 8) {
        if (i + 8 <= N) {
            __m256 x = _mm256_loadu_ps(&in[i]);
            __m256 half = _mm256_set1_ps(0.5f);
            __m256 one = _mm256_set1_ps(1.0f);
            __m256 c1_vec = _mm256_set1_ps(c1);
            __m256 c2_vec = _mm256_set1_ps(c2);
            
            // x^3
            __m256 x2 = _mm256_mul_ps(x, x);
            __m256 x3 = _mm256_mul_ps(x2, x);
            
            // 0.044715 * x^3
            __m256 term = _mm256_mul_ps(c2_vec, x3);
            
            // x + 0.044715 * x^3
            __m256 sum = _mm256_add_ps(x, term);
            
            // 0.7978... * (x + 0.044715 * x^3)
            __m256 arg = _mm256_mul_ps(c1_vec, sum);
            
            // tanh approximation (pas parfait mais rapide)
            // tanh(x) ≈ x pour |x| < 1
            // Pour une vraie impl, utiliser _mm256_tanh_ps si disponible
            // Ici on fait une approximation rapide
            __m256 tanh_arg = arg;
            __m256 tanh_approx = _mm256_max_ps(_mm256_set1_ps(-1.0f), 
                                _mm256_min_ps(_mm256_set1_ps(1.0f), tanh_arg));
            
            // 0.5 * x * (1 + tanh(...))
            __m256 one_plus_tanh = _mm256_add_ps(one, tanh_approx);
            __m256 result = _mm256_mul_ps(half, _mm256_mul_ps(x, one_plus_tanh));
            
            _mm256_storeu_ps(&out[i], result);
        } else {
            // Fallback
            for (size_t j = i; j < N; ++j) {
                float x = in[j];
                out[j] = x * 0.5f * (1.0f + std::tanh(c1 * (x + c2 * x * x * x)));
            }
        }
    }
}

// Softmax avec AVX2
inline void softmax_avx2(float* out, const float* in, size_t N) {
    // Find max for numerical stability
    float max_val = in[0];
    for (size_t i = 1; i < N; ++i) {
        max_val = std::max(max_val, in[i]);
    }
    
    // Compute exp(x - max)
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < N; ++i) {
        out[i] = std::exp(in[i] - max_val);
        sum += out[i];
    }
    
    // Normalize
    float inv_sum = 1.0f / sum;
    #pragma omp parallel for
    for (size_t i = 0; i < N; i += 8) {
        if (i + 8 <= N) {
            __m256 vals = _mm256_loadu_ps(&out[i]);
            __m256 scale = _mm256_set1_ps(inv_sum);
            __m256 result = _mm256_mul_ps(vals, scale);
            _mm256_storeu_ps(&out[i], result);
        } else {
            for (size_t j = i; j < N; ++j) {
                out[j] *= inv_sum;
            }
        }
    }
}

// Dot product optimisé
inline float dot_product_avx2(const float* A, const float* B, size_t N) {
    __m256 sum = _mm256_setzero_ps();
    
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        __m256 a = _mm256_loadu_ps(&A[i]);
        __m256 b = _mm256_loadu_ps(&B[i]);
        sum = _mm256_fmadd_ps(a, b, sum);
    }
    
    // Horizontal sum
    __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(sum_low, sum_high);
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float result = _mm_cvtss_f32(sums);
    
    // Remaining elements
    for (; i < N; ++i) {
        result += A[i] * B[i];
    }
    
    return result;
}

} // namespace SIMD

#endif // __SIMD_OPS_HPP__
