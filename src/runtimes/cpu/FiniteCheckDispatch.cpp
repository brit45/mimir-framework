#include "FiniteCheckDispatch.hpp"
#include "SimdDispatch.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#if (!defined(MIMIR_SIMD_ENABLED) || MIMIR_SIMD_ENABLED) && \
    (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define MIMIR_FINITE_MULTIVERSION 1
#define MIMIR_FINITE_TARGET(isa) __attribute__((target(isa)))
#endif

namespace Mimir::RuntimeChecks {
namespace {
bool scalar(const float* values,size_t count) {
    for(size_t i=0;i<count;++i) {
        if constexpr(sizeof(float)==4 && std::numeric_limits<float>::is_iec559) {
            std::uint32_t bits;std::memcpy(&bits,values+i,sizeof(bits));
            if((bits&0x7f800000u)==0x7f800000u)return false;
        } else if(!std::isfinite(values[i]))return false;
    }
    return true;
}
#ifdef MIMIR_FINITE_MULTIVERSION
MIMIR_FINITE_TARGET("sse2") bool sse(const float* values,size_t count) {
    const __m128i exponent=_mm_set1_epi32(0x7f800000);size_t i=0;
    for(;count-i>=16;i+=16) {
        __m128i invalid=_mm_setzero_si128();
        for(size_t j=0;j<16;j+=4) {
            const __m128i bits=_mm_castps_si128(_mm_loadu_ps(values+i+j));
            invalid=_mm_or_si128(invalid,_mm_cmpeq_epi32(_mm_and_si128(bits,exponent),exponent));
        }
        if(_mm_movemask_epi8(invalid))return false;
    }
    return scalar(values+i,count-i);
}
MIMIR_FINITE_TARGET("avx2") bool avx2(const float* values,size_t count) {
    const __m256i exponent=_mm256_set1_epi32(0x7f800000);size_t i=0;
    for(;count-i>=32;i+=32) {
        __m256i invalid=_mm256_setzero_si256();
        for(size_t j=0;j<32;j+=8) {
            const __m256i bits=_mm256_castps_si256(_mm256_loadu_ps(values+i+j));
            invalid=_mm256_or_si256(invalid,_mm256_cmpeq_epi32(_mm256_and_si256(bits,exponent),exponent));
        }
        if(_mm256_movemask_epi8(invalid))return false;
    }
    return scalar(values+i,count-i);
}
MIMIR_FINITE_TARGET("avx512f") bool avx512(const float* values,size_t count) {
    const __m512i exponent=_mm512_set1_epi32(0x7f800000);size_t i=0;
    for(;count-i>=64;i+=64) {
        __mmask16 invalid=0;
        for(size_t j=0;j<64;j+=16) {
            const __m512i bits=_mm512_castps_si512(_mm512_loadu_ps(values+i+j));
            invalid|=_mm512_cmpeq_epi32_mask(_mm512_and_si512(bits,exponent),exponent);
        }
        if(invalid)return false;
    }
    return scalar(values+i,count-i);
}
#endif
}

bool allFiniteDispatched(const float* values,size_t count) {
    if(!values)return count==0;
    if(count<32)return scalar(values,count);
#ifdef MIMIR_FINITE_MULTIVERSION
    if constexpr(sizeof(float)==4 && std::numeric_limits<float>::is_iec559) {
        switch(RuntimeSimd::selectedIsa()) {
            case RuntimeSimd::Isa::AVX512:return avx512(values,count);
            case RuntimeSimd::Isa::AVX2:return avx2(values,count);
            case RuntimeSimd::Isa::AVX:case RuntimeSimd::Isa::SSE2:return sse(values,count);
            default:break;
        }
    }
#endif
    return scalar(values,count);
}
}
