#ifndef __SIMD_OPS_HPP__
#define __SIMD_OPS_HPP__
#include <algorithm>
#include <cmath>
#include <cstddef>
#include "HardwareOpt.hpp"

namespace SIMD {
// Historical API names are adapters to the selected scalar/SSE2/AVX/AVX2/
// AVX-512 implementation; they no longer require AVX2 on the host CPU.
inline void matmul_avx2(float* c,const float* a,const float* b,size_t m,size_t n,size_t k) {
    HardwareOpt::matmul_fma_saturated(c,a,b,m,n,k);
}
inline void matmul_transpose_avx2(float* c,const float* a,const float* b,size_t m,size_t n,size_t k) {
    RuntimeSimd::matmulTranspose(c,a,b,m,n,k,true,HardwareOpt::matmul_workers(m*((n+23)/24),24,k));
}
inline void add_vectors_avx2(float* c,const float* a,const float* b,size_t n){RuntimeSimd::add(c,a,b,n);}
inline void mul_vectors_avx2(float* c,const float* a,const float* b,size_t n){RuntimeSimd::mul(c,a,b,n);}
inline void gelu_forward_avx2(float* out,const float* in,size_t n) {
    // The old vector prefix clamped the tanh argument; use the same accurate
    // function at every position rather than a different activation in tails.
    for(size_t i=0;i<n;++i) {
        const float x=in[i];
        out[i]=x*0.5f*(1.0f+std::tanh(0.7978845608f*(x+0.044715f*x*x*x)));
    }
}
inline void softmax_avx2(float* out,const float* in,size_t n) {
    if(!n)return;
    const float maximum=*std::max_element(in,in+n);
    float sum=0;
    for(size_t i=0;i<n;++i){out[i]=std::exp(in[i]-maximum);sum+=out[i];}
    const float inverse=1.0f/sum;
    for(size_t i=0;i<n;++i)out[i]*=inverse;
}
inline float dot_product_avx2(const float* a,const float* b,size_t n){return RuntimeSimd::dot(a,b,n,true);}
}
#endif
