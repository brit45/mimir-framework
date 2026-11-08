#include "SimdDispatch.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define MIMIR_SIMD_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif
#ifndef MIMIR_SIMD_ENABLED
#define MIMIR_SIMD_ENABLED 1
#endif
#if MIMIR_SIMD_ENABLED && defined(MIMIR_SIMD_X86) && (defined(__GNUC__) || defined(__clang__))
#define MIMIR_SIMD_MULTIVERSION 1
#if defined(__clang__)
#define MIMIR_SIMD_TARGET(isa) __attribute__((target(isa)))
#else
#define MIMIR_SIMD_TARGET(isa) __attribute__((target(isa),optimize("fp-contract=off")))
#endif
#else
#define MIMIR_SIMD_TARGET(isa)
#endif

namespace RuntimeSimd {
namespace {
std::atomic<int> requested{-1};
Features detect() {
    Features f;
#ifdef MIMIR_SIMD_X86
    unsigned eax=0,ebx=0,ecx=0,edx=0;
    auto cpuid=[&](unsigned leaf,unsigned sub) {
#if defined(_MSC_VER)
        int r[4]; __cpuidex(r,int(leaf),int(sub));
        eax=unsigned(r[0]); ebx=unsigned(r[1]); ecx=unsigned(r[2]); edx=unsigned(r[3]);
        return true;
#else
        return __get_cpuid_count(leaf,sub,&eax,&ebx,&ecx,&edx)!=0;
#endif
    };
    if(!cpuid(0,0))return f;
    const unsigned maximum=eax;
    if(maximum<1 || !cpuid(1,0))return f;
    f.sse2=(edx&(1u<<26))!=0; f.sse3=(ecx&1)!=0;
    f.ssse3=(ecx&(1u<<9))!=0; f.sse41=(ecx&(1u<<19))!=0;
    f.sse42=(ecx&(1u<<20))!=0;
    const unsigned flags=ecx;
    // XGETBV is legal only after XSAVE and OSXSAVE have both been checked.
    if((flags&(1u<<26)) && (flags&(1u<<27))) {
#if defined(_MSC_VER)
        f.xcr0=_xgetbv(0);
#else
        unsigned lo=0,hi=0;
        __asm__ volatile("xgetbv" : "=a"(lo),"=d"(hi) : "c"(0));
        f.xcr0=(std::uint64_t(hi)<<32)|lo;
#endif
    }
    f.avx=(flags&(1u<<28)) && ((f.xcr0&0x6)==0x6);
    f.fma=f.avx && (flags&(1u<<12));
    f.f16c=f.avx && (flags&(1u<<29));
    if(maximum>=7 && cpuid(7,0)) {
        f.bmi1=(ebx&(1u<<3))!=0; f.bmi2=(ebx&(1u<<8))!=0;
        f.avx2=f.avx && (ebx&(1u<<5));
        const bool zmm=f.avx && (f.xcr0&0xe6)==0xe6;
        f.avx512f=zmm && (ebx&(1u<<16));
        f.avx512dq=f.avx512f && (ebx&(1u<<17));
        f.avx512bw=f.avx512f && (ebx&(1u<<30));
        f.avx512vl=f.avx512f && (ebx&(1u<<31));
    }
#endif
    return f;
}
Isa best() {
    for(Isa isa:{Isa::AVX512,Isa::AVX2,Isa::AVX,Isa::SSE2})if(supported(isa))return isa;
    return Isa::Scalar;
}
Isa environmentSelection() {
    const char* value=std::getenv("MIMIR_CPU_ISA");
    if(!value || !*value || std::strcmp(value,"auto")==0)return best();
    Isa isa;
    if(std::strcmp(value,"scalar")==0)isa=Isa::Scalar;
    else if(std::strcmp(value,"sse2")==0 || std::strcmp(value,"sse")==0)isa=Isa::SSE2;
    else if(std::strcmp(value,"avx")==0 || std::strcmp(value,"avx256")==0)isa=Isa::AVX;
    else if(std::strcmp(value,"avx2")==0)isa=Isa::AVX2;
    else if(std::strcmp(value,"avx512")==0 || std::strcmp(value,"avx512f")==0)isa=Isa::AVX512;
    else throw std::invalid_argument(std::string("MIMIR_CPU_ISA invalide: ")+value);
    if(!supported(isa))throw std::invalid_argument(std::string("MIMIR_CPU_ISA indisponible pour ce CPU/OS/build: ")+value);
    return isa;
}

// No implicit FMA contraction: selecting a wider ISA never changes rounding.
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize ("fp-contract=off")
#endif
using Row=void(*)(float*,const float*,const float*,std::size_t,std::size_t,std::size_t,std::size_t,bool,bool);
using Binary=void(*)(float*,const float*,const float*,std::size_t);
using Axpy=void(*)(float*,const float*,float,std::size_t);
template<bool Fused>
void rowScalar(float* c,const float* a,const float* b,std::size_t n,std::size_t k,
               std::size_t aStep,std::size_t bStep,bool accumulate,bool resume) {
    for(std::size_t j=0;j<n;++j) {
        float sum=resume?c[j]:0;
        for(std::size_t r=0;r<k;++r) {
            if constexpr(Fused)sum=std::fma(a[r*aStep],b[r*bStep+j],sum);
            else sum+=a[r*aStep]*b[r*bStep+j];
        }
        if(accumulate)c[j]+=sum; else c[j]=sum;
    }
}
void addScalar(float* c,const float* a,const float* b,std::size_t n) {for(std::size_t i=0;i<n;++i)c[i]=a[i]+b[i];}
void mulScalar(float* c,const float* a,const float* b,std::size_t n) {for(std::size_t i=0;i<n;++i)c[i]=a[i]*b[i];}
template<bool Fused>void axpyScalar(float* c,const float* a,float scale,std::size_t n) {
    for(std::size_t i=0;i<n;++i) {if constexpr(Fused)c[i]=std::fma(scale,a[i],c[i]);else c[i]+=scale*a[i];}
}

#ifdef MIMIR_SIMD_MULTIVERSION
// Three independent vector accumulators hide arithmetic latency while each
// output retains exactly the scalar reduction order, including short tails.
#define MIMIR_DEFINE_ROWS(NAME,TARGET,VEC,WIDTH,ZERO,SET,LOAD,STORE,ADD,MUL,FMADD,FUSED) \
MIMIR_SIMD_TARGET(TARGET) void NAME(float* c,const float* a,const float* b,std::size_t n,std::size_t k,std::size_t aStep,std::size_t bStep,bool accumulate,bool resume) { \
    std::size_t j=0; \
    for(;j+3*WIDTH<=n;j+=3*WIDTH) { \
        VEC x=resume?LOAD(c+j):ZERO(),y=resume?LOAD(c+j+WIDTH):ZERO(),z=resume?LOAD(c+j+2*WIDTH):ZERO(); \
        for(std::size_t r=0;r<k;++r) {const VEC v=SET(a[r*aStep]); const float* br=b+r*bStep+j; \
            x=FMADD(v,LOAD(br),x); y=FMADD(v,LOAD(br+WIDTH),y);z=FMADD(v,LOAD(br+2*WIDTH),z);} \
        if(accumulate){x=ADD(LOAD(c+j),x);y=ADD(LOAD(c+j+WIDTH),y);z=ADD(LOAD(c+j+2*WIDTH),z);} \
        STORE(c+j,x);STORE(c+j+WIDTH,y);STORE(c+j+2*WIDTH,z); \
    } \
    for(;j+WIDTH<=n;j+=WIDTH) {VEC sum=resume?LOAD(c+j):ZERO(); \
        for(std::size_t r=0;r<k;++r)sum=FMADD(SET(a[r*aStep]),LOAD(b+r*bStep+j),sum); \
        if(accumulate){sum=ADD(LOAD(c+j),sum);}STORE(c+j,sum);} \
    if(j<n)rowScalar<FUSED>(c+j,a,k?b+j:b,n-j,k,aStep,bStep,accumulate,resume); \
}
#define MIMIR_SEPARATE128(a,b,c) _mm_add_ps(_mm_mul_ps(a,b),c)
#define MIMIR_SEPARATE256(a,b,c) _mm256_add_ps(_mm256_mul_ps(a,b),c)
#define MIMIR_SEPARATE512(a,b,c) _mm512_add_ps(_mm512_mul_ps(a,b),c)
MIMIR_DEFINE_ROWS(rowSse,"sse2",__m128,4,_mm_setzero_ps,_mm_set1_ps,_mm_loadu_ps,_mm_storeu_ps,_mm_add_ps,_mm_mul_ps,MIMIR_SEPARATE128,false)
MIMIR_DEFINE_ROWS(rowAvx,"avx",__m256,8,_mm256_setzero_ps,_mm256_set1_ps,_mm256_loadu_ps,_mm256_storeu_ps,_mm256_add_ps,_mm256_mul_ps,MIMIR_SEPARATE256,false)
MIMIR_DEFINE_ROWS(rowAvx2,"avx2",__m256,8,_mm256_setzero_ps,_mm256_set1_ps,_mm256_loadu_ps,_mm256_storeu_ps,_mm256_add_ps,_mm256_mul_ps,MIMIR_SEPARATE256,false)
MIMIR_DEFINE_ROWS(rowAvx512,"avx512f",__m512,16,_mm512_setzero_ps,_mm512_set1_ps,_mm512_loadu_ps,_mm512_storeu_ps,_mm512_add_ps,_mm512_mul_ps,MIMIR_SEPARATE512,false)
MIMIR_DEFINE_ROWS(rowFma,"avx,fma",__m256,8,_mm256_setzero_ps,_mm256_set1_ps,_mm256_loadu_ps,_mm256_storeu_ps,_mm256_add_ps,_mm256_mul_ps,_mm256_fmadd_ps,true)
MIMIR_DEFINE_ROWS(rowAvx512Fma,"avx512f",__m512,16,_mm512_setzero_ps,_mm512_set1_ps,_mm512_loadu_ps,_mm512_storeu_ps,_mm512_add_ps,_mm512_mul_ps,_mm512_fmadd_ps,true)
#define MIMIR_DEFINE_BINARY(NAME,TARGET,VEC,WIDTH,LOAD,STORE,OP) \
MIMIR_SIMD_TARGET(TARGET) void NAME(float* c,const float* a,const float* b,std::size_t n) {std::size_t i=0;for(;i+WIDTH<=n;i+=WIDTH)STORE(c+i,OP(LOAD(a+i),LOAD(b+i)));for(;i<n;++i)c[i]=OP##Scalar(a[i],b[i]);}
// Use a scalar operator separately, rather than translating intrinsic names.
inline float _mm_add_psScalar(float a,float b){return a+b;}
inline float _mm_mul_psScalar(float a,float b){return a*b;}
inline float _mm256_add_psScalar(float a,float b){return a+b;}
inline float _mm256_mul_psScalar(float a,float b){return a*b;}
inline float _mm512_add_psScalar(float a,float b){return a+b;}
inline float _mm512_mul_psScalar(float a,float b){return a*b;}
MIMIR_DEFINE_BINARY(addSse,"sse2",__m128,4,_mm_loadu_ps,_mm_storeu_ps,_mm_add_ps)
MIMIR_DEFINE_BINARY(mulSse,"sse2",__m128,4,_mm_loadu_ps,_mm_storeu_ps,_mm_mul_ps)
MIMIR_DEFINE_BINARY(addAvx,"avx",__m256,8,_mm256_loadu_ps,_mm256_storeu_ps,_mm256_add_ps)
MIMIR_DEFINE_BINARY(mulAvx,"avx",__m256,8,_mm256_loadu_ps,_mm256_storeu_ps,_mm256_mul_ps)
MIMIR_DEFINE_BINARY(addAvx512,"avx512f",__m512,16,_mm512_loadu_ps,_mm512_storeu_ps,_mm512_add_ps)
MIMIR_DEFINE_BINARY(mulAvx512,"avx512f",__m512,16,_mm512_loadu_ps,_mm512_storeu_ps,_mm512_mul_ps)
#define MIMIR_DEFINE_AXPY(NAME,TARGET,VEC,WIDTH,SET,LOAD,STORE,FMADD,FUSED) \
MIMIR_SIMD_TARGET(TARGET) void NAME(float* c,const float* a,float scale,std::size_t n){std::size_t i=0;const VEC s=SET(scale);for(;i+WIDTH<=n;i+=WIDTH)STORE(c+i,FMADD(s,LOAD(a+i),LOAD(c+i)));axpyScalar<FUSED>(c+i,a+i,scale,n-i);}
MIMIR_DEFINE_AXPY(axpySse,"sse2",__m128,4,_mm_set1_ps,_mm_loadu_ps,_mm_storeu_ps,MIMIR_SEPARATE128,false)
MIMIR_DEFINE_AXPY(axpyAvx,"avx",__m256,8,_mm256_set1_ps,_mm256_loadu_ps,_mm256_storeu_ps,MIMIR_SEPARATE256,false)
MIMIR_DEFINE_AXPY(axpyAvx512,"avx512f",__m512,16,_mm512_set1_ps,_mm512_loadu_ps,_mm512_storeu_ps,MIMIR_SEPARATE512,false)
MIMIR_DEFINE_AXPY(axpyFma,"avx,fma",__m256,8,_mm256_set1_ps,_mm256_loadu_ps,_mm256_storeu_ps,_mm256_fmadd_ps,true)
MIMIR_DEFINE_AXPY(axpyAvx512Fma,"avx512f",__m512,16,_mm512_set1_ps,_mm512_loadu_ps,_mm512_storeu_ps,_mm512_fmadd_ps,true)
#undef MIMIR_DEFINE_ROWS
#undef MIMIR_DEFINE_BINARY
#undef MIMIR_DEFINE_AXPY
#undef MIMIR_SEPARATE128
#undef MIMIR_SEPARATE256
#undef MIMIR_SEPARATE512
#endif

Row rowFunction(bool fused) {
    const Isa isa=selectedIsa();
#ifdef MIMIR_SIMD_MULTIVERSION
    if(fused) {
        if(isa==Isa::AVX512)return rowAvx512Fma;
        if((isa==Isa::AVX || isa==Isa::AVX2) && features().fma)return rowFma;
    } else {
        switch(isa) {
            case Isa::AVX512:return rowAvx512;
            case Isa::AVX2:return rowAvx2;
            case Isa::AVX:return rowAvx;
            case Isa::SSE2:return rowSse;
            default:break;
        }
    }
#else
    (void)isa;
#endif
    return fused?rowScalar<true>:rowScalar<false>;
}
template<class Fn>void forRows(std::size_t rows,int workers,const Fn& fn) {
#ifdef _OPENMP
    workers=std::max(1,std::min(workers,int(std::min(rows,std::size_t(0x7fffffff)))));
    if(workers>1 && !omp_in_parallel()) {
        #pragma omp parallel for num_threads(workers) schedule(static)
        for(std::size_t r=0;r<rows;++r)fn(r);
        return;
    }
#else
    (void)workers;
#endif
    for(std::size_t r=0;r<rows;++r)fn(r);
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
} // namespace

const Features& features(){static const Features f=detect();return f;}
bool supported(Isa isa) {
    if(isa==Isa::Scalar)return true;
#ifdef MIMIR_SIMD_MULTIVERSION
    const auto& f=features();
    switch(isa) {
        case Isa::SSE2:return f.sse2;
        case Isa::AVX:return f.avx;
        case Isa::AVX2:return f.avx2;
        case Isa::AVX512:return f.avx512f;
        default:return false;
    }
#else
    return false;
#endif
}
const char* isaName(Isa isa) {
    switch(isa){case Isa::Scalar:return "scalar";case Isa::SSE2:return "sse2";
        case Isa::AVX:return "avx256";case Isa::AVX2:return "avx2";case Isa::AVX512:return "avx512f";}
    return "unknown";
}
Isa selectedIsa() {
    const int explicitIsa=requested.load(std::memory_order_relaxed);
    if(explicitIsa>=0)return Isa(explicitIsa);
    static const Isa environment=environmentSelection();
    return environment;
}
bool requestIsa(Isa isa) {if(!supported(isa))return false;requested.store(int(isa),std::memory_order_relaxed);return true;}
void clearIsaRequest(){requested.store(-1,std::memory_order_relaxed);}
void matmul(float* c,const float* a,const float* b,std::size_t m,std::size_t n,std::size_t k,bool accumulate,bool fused,int workers) {
    if(!m || !n)return;
    const Row fn=rowFunction(fused);
    // Column tiles provide independent work even for the one-row GEMMs used
    // during generation. Serial execution keeps full rows for minimal overhead.
    const std::size_t tile=workers>1?192:n,tiles=(n+tile-1)/tile;
    forRows(m*tiles,workers,[&](std::size_t task){
        const std::size_t r=task/tiles,j=(task%tiles)*tile;
        fn(c+r*n+j,k?a+r*k:a,k?b+j:b,std::min(tile,n-j),k,1,n,accumulate,false);
    });
}
void matmulBatched(float* c,const float* a,const float* b,std::size_t batches,std::size_t m,std::size_t n,std::size_t k,bool fused,int workers) {
    if(!m || !n || !batches)return;
    const Row fn=rowFunction(fused);
    const std::size_t tile=workers>1?192:n,tiles=(n+tile-1)/tile;
    forRows(batches*m*tiles,workers,[&](std::size_t task){
        const std::size_t r=task/tiles,j=(task%tiles)*tile;
        fn(c+r*n+j,k?a+r*k:a,k?b+(r/m)*k*n+j:b,std::min(tile,n-j),k,1,n,false,false);
    });
}
void matmulTranspose(float* c,const float* a,const float* b,std::size_t m,std::size_t n,std::size_t k,bool fused,int workers) {
    if(!m || !n)return;
    if(!k){matmul(c,a,b,m,n,0,false,fused,workers);return;}
    const Row fn=rowFunction(fused);
    // Fixed scratch: 24 KiB per active worker, independent of matrix sizes.
    // Keep accumulators live numerically across reduction tiles: resume loads
    // the previous FP32 accumulator, rather than summing partial dot products.
    constexpr std::size_t reductionTile=32,columnTile=192,rowTile=16;
    const std::size_t rowTiles=(m+rowTile-1)/rowTile,columnTiles=(n+columnTile-1)/columnTile;
    forRows(rowTiles*columnTiles,workers,[&](std::size_t task){
        std::array<float,reductionTile*columnTile> packed;
        const std::size_t firstRow=(task/columnTiles)*rowTile,firstColumn=(task%columnTiles)*columnTile;
        const std::size_t columns=std::min(columnTile,n-firstColumn),lastRow=std::min(m,firstRow+rowTile);
        for(std::size_t firstReduction=0;firstReduction<k;firstReduction+=reductionTile) {
            const std::size_t reductions=std::min(reductionTile,k-firstReduction);
            for(std::size_t r=0;r<reductions;++r)
                for(std::size_t j=0;j<columns;++j)
                    packed[r*columns+j]=b[(firstColumn+j)*k+firstReduction+r];
            for(std::size_t row=firstRow;row<lastRow;++row)
                fn(c+row*n+firstColumn,a+row*k+firstReduction,packed.data(),columns,reductions,1,columns,false,firstReduction!=0);
        }
    });
}
void matmulLeftTranspose(float* c,const float* a,const float* b,std::size_t m,std::size_t n,std::size_t k,bool fused,int workers) {
    if(!k || !n)return;
    const Row fn=rowFunction(fused);
    const std::size_t tile=workers>1?192:n,tiles=(n+tile-1)/tile;
    forRows(k*tiles,workers,[&](std::size_t task){
        const std::size_t r=task/tiles,j=(task%tiles)*tile;
        fn(c+r*n+j,m?a+r:a,m?b+j:b,std::min(tile,n-j),m,k,n,false,false);
    });
}
void add(float* c,const float* a,const float* b,std::size_t n) {
    Binary fn=addScalar;
#ifdef MIMIR_SIMD_MULTIVERSION
    switch(selectedIsa()){case Isa::AVX512:fn=addAvx512;break;case Isa::AVX:case Isa::AVX2:fn=addAvx;break;case Isa::SSE2:fn=addSse;break;default:break;}
#endif
    if(n)fn(c,a,b,n);
}
void mul(float* c,const float* a,const float* b,std::size_t n) {
    Binary fn=mulScalar;
#ifdef MIMIR_SIMD_MULTIVERSION
    switch(selectedIsa()){case Isa::AVX512:fn=mulAvx512;break;case Isa::AVX:case Isa::AVX2:fn=mulAvx;break;case Isa::SSE2:fn=mulSse;break;default:break;}
#endif
    if(n)fn(c,a,b,n);
}
void axpy(float* c,const float* a,float scale,std::size_t n,bool fused) {
    Axpy fn=fused?axpyScalar<true>:axpyScalar<false>;
#ifdef MIMIR_SIMD_MULTIVERSION
    const Isa isa=selectedIsa();
    if(fused) {
        if(isa==Isa::AVX512)fn=axpyAvx512Fma;
        else if((isa==Isa::AVX || isa==Isa::AVX2) && features().fma)fn=axpyFma;
    } else {
        switch(isa){case Isa::AVX512:fn=axpyAvx512;break;case Isa::AVX:case Isa::AVX2:fn=axpyAvx;break;case Isa::SSE2:fn=axpySse;break;default:break;}
    }
#endif
    if(n)fn(c,a,scale,n);
}
float dot(const float* a,const float* b,std::size_t n,bool fused) {
    float result=0;
    // An independent-output SIMD transform is not available for a single dot;
    // preserve the reduction order rather than changing the numerical result.
    for(std::size_t i=0;i<n;++i){if(fused)result=std::fma(a[i],b[i],result);else result+=a[i]*b[i];}
    return result;
}

namespace {
std::uint16_t encodeHalf(float value) {
    std::uint32_t bits; std::memcpy(&bits,&value,sizeof(bits));
    const std::uint32_t sign=(bits>>16)&0x8000u,exponent=(bits>>23)&0xffu;
    std::uint32_t mantissa=bits&0x7fffffu;
    if(exponent==255)return std::uint16_t(sign|0x7c00u|(mantissa?0x200u:0));
    const int e=int(exponent)-112;
    if(e>=31)return std::uint16_t(sign|0x7c00u);
    if(e<=0) {
        if(e < -10)return std::uint16_t(sign);
        mantissa|=0x800000u;
        const unsigned shift=unsigned(14-e);
        return std::uint16_t(sign|((mantissa+((1u<<(shift-1))-1u)+((mantissa>>shift)&1u))>>shift));
    }
    return std::uint16_t(sign|((std::uint32_t(e)<<10)+((mantissa+0xfffu+((mantissa>>13)&1u))>>13)));
}
float decodeHalf(std::uint16_t half) {
    const std::uint32_t sign=(half&0x8000u)<<16,exponent=(half>>10)&31u;
    std::uint32_t mantissa=half&0x3ffu,bits;
    if(exponent==0) {
        bits=sign;
        if(mantissa){int e=-14;while(!(mantissa&0x400u)){mantissa<<=1;--e;}bits|=(std::uint32_t(e+127)<<23)|((mantissa&0x3ffu)<<13);}
    } else if(exponent==31)bits=sign|0x7f800000u|(mantissa<<13);
    else bits=sign|((exponent+112u)<<23)|(mantissa<<13);
    float result;std::memcpy(&result,&bits,sizeof(result));return result;
}
#ifdef MIMIR_SIMD_MULTIVERSION
MIMIR_SIMD_TARGET("avx,f16c") void encodeF16c(std::uint16_t* dst,const float* src,std::size_t n) {
    std::size_t i=0;
    for(;i+8<=n;i+=8)_mm_storeu_si128(reinterpret_cast<__m128i*>(dst+i),_mm256_cvtps_ph(_mm256_loadu_ps(src+i),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));
    for(;i<n;++i)dst[i]=encodeHalf(src[i]);
}
MIMIR_SIMD_TARGET("avx,f16c") void decodeF16c(float* dst,const std::uint16_t* src,std::size_t n) {
    std::size_t i=0;
    for(;i+8<=n;i+=8)_mm256_storeu_ps(dst+i,_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src+i))));
    for(;i<n;++i)dst[i]=decodeHalf(src[i]);
}
#endif
}
void fp32ToFp16(std::uint16_t* dst,const float* src,std::size_t count) {
#ifdef MIMIR_SIMD_MULTIVERSION
    const Isa isa=selectedIsa();
    if(features().f16c && (isa==Isa::AVX || isa==Isa::AVX2 || isa==Isa::AVX512)){encodeF16c(dst,src,count);return;}
#endif
    for(std::size_t i=0;i<count;++i)dst[i]=encodeHalf(src[i]);
}
void fp16ToFp32(float* dst,const std::uint16_t* src,std::size_t count) {
#ifdef MIMIR_SIMD_MULTIVERSION
    const Isa isa=selectedIsa();
    if(features().f16c && (isa==Isa::AVX || isa==Isa::AVX2 || isa==Isa::AVX512)){decodeF16c(dst,src,count);return;}
#endif
    for(std::size_t i=0;i<count;++i)dst[i]=decodeHalf(src[i]);
}
} // namespace RuntimeSimd
