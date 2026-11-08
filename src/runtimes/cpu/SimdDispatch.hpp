#pragma once

#include <cstddef>
#include <cstdint>

// CPU ISA dispatch is independent of GPU routing. No ISA-specific type escapes
// this header, including when CPU dispatch headers are parsed by nvcc or HIP.
namespace RuntimeSimd {
enum class Isa { Scalar, SSE2, AVX, AVX2, AVX512 };
struct Features {
    bool sse2=false, sse3=false, ssse3=false, sse41=false, sse42=false;
    bool avx=false, avx2=false, avx512f=false, avx512dq=false;
    bool avx512bw=false, avx512vl=false, fma=false, f16c=false;
    bool bmi1=false, bmi2=false;
    std::uint64_t xcr0=0;
};
const Features& features();
bool supported(Isa isa);
const char* isaName(Isa isa);
// MIMIR_CPU_ISA=auto/scalar/sse2/avx/avx256/avx2/avx512 is checked once.
// Invalid or unsupported environment requests throw, never run illegal code.
Isa selectedIsa();
bool requestIsa(Isa isa); // Unsupported requests return false, preserving selection.
void clearIsaRequest();

// Every output accumulates k in increasing order; SIMD lanes are independent
// outputs, never a reassociated horizontal reduction. fused=false uses separate
// FP32 multiply/add. fused=true uses correctly rounded std::fma or hardware FMA
// consistently, including tails and the scalar fallback.
void matmul(float* c,const float* a,const float* b,std::size_t m,std::size_t n,std::size_t k,
            bool accumulate=false,bool fused=false,int workers=1);
void matmulBatched(float* c,const float* a,const float* b,std::size_t batches,
                   std::size_t m,std::size_t n,std::size_t k,bool fused=false,int workers=1);
void matmulTranspose(float* c,const float* a,const float* b,std::size_t m,std::size_t n,
                     std::size_t k,bool fused=false,int workers=1);
// A[m,k], B[m,n] -> C[k,n].
void matmulLeftTranspose(float* c,const float* a,const float* b,std::size_t m,std::size_t n,
                         std::size_t k,bool fused=false,int workers=1);
void add(float* out,const float* a,const float* b,std::size_t count);
void mul(float* out,const float* a,const float* b,std::size_t count);
// out[i] += scale*x[i]; exact in-place aliasing is supported.
void axpy(float* out,const float* x,float scale,std::size_t count,bool fused=false);
float dot(const float* a,const float* b,std::size_t count,bool fused=false);
void fp32ToFp16(std::uint16_t* dst,const float* src,std::size_t count);
void fp16ToFp32(float* dst,const std::uint16_t* src,std::size_t count);
} // namespace RuntimeSimd
