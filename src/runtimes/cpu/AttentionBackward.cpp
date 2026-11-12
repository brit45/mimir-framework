#include "AttentionBackward.hpp"
#include "ReusableScratch.hpp"
#include "SimdDispatch.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace CpuAttention {
namespace {
size_t product(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max()/a)
        throw std::length_error("attention scratch size overflow");
    return a*b;
}
size_t sum(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max()-a)
        throw std::length_error("attention scratch size overflow");
    return a+b;
}
// Tiny attention fits in two stack panels. Avoid scratch leasing and several
// GEMM dispatches for the short, narrow shapes used by small models.
void smallBackward(const float* q,const float* k,const float* v,size_t qs,size_t ks,
    const float* da,float* attended,float* dq,float* dk,float* dv,
    size_t queries,size_t keys,size_t embed,size_t heads,bool causal,float scale) {
    std::array<float,16*16> probabilities,derivatives;
    const size_t hd=embed/heads;
    for(size_t h=0;h<heads;++h) {
        const size_t off=h*hd;
        for(size_t i=0;i<queries;++i) {
            float* p=probabilities.data()+i*keys;
            float* ds=derivatives.data()+i*keys;
            float maximum=-1e30f;
            for(size_t j=0;j<keys;++j) {
                float score=-1e9f;
                if(!(causal && j>i)) {
                    float dot=0.f;
                    for(size_t d=0;d<hd;++d)dot+=q[i*qs+off+d]*k[j*ks+off+d];
                    score=dot*scale;
                }
                p[j]=score;if(score>maximum)maximum=score;
            }
            float total=0.f;
            for(size_t j=0;j<keys;++j) {p[j]=std::exp(p[j]-maximum);total+=p[j];}
            const float inverse=1.f/(total+1e-9f);
            for(size_t j=0;j<keys;++j)p[j]*=inverse;
            for(size_t d=0;d<hd;++d) {
                float value=0.f;
                for(size_t j=0;j<keys;++j)value+=p[j]*v[j*ks+off+d];
                attended[i*embed+off+d]=value;
            }
            for(size_t j=0;j<keys;++j) {
                float value=0.f;
                for(size_t d=0;d<hd;++d)value+=da[i*embed+off+d]*v[j*ks+off+d];
                ds[j]=value;
            }
            float dot=0.f;
            for(size_t j=0;j<keys;++j)dot+=ds[j]*p[j];
            for(size_t j=0;j<keys;++j)ds[j]=causal && j>i?0.f:(ds[j]-dot)*p[j];
        }
        for(size_t i=0;i<queries;++i)for(size_t d=0;d<hd;++d) {
            float value=0.f;
            for(size_t j=0;j<keys;++j)value+=derivatives[i*keys+j]*k[j*ks+off+d];
            dq[i*embed+off+d]=0.f+value*scale;
        }
        for(size_t j=0;j<keys;++j)for(size_t d=0;d<hd;++d) {
            float key=0.f,value=0.f;
            for(size_t i=0;i<queries;++i) {
                key+=derivatives[i*keys+j]*q[i*qs+off+d];
                value+=probabilities[i*keys+j]*da[i*embed+off+d];
            }
            dk[j*embed+off+d]=0.f+key*scale;
            dv[j*embed+off+d]=0.f+value;
        }
    }
}
}

void backward(const float* q, const float* k, const float* v,
    size_t qStride, size_t kvStride, const float* dAtt,
    float* attended, float* dQ, float* dK, float* dV,
    size_t queries, size_t keys, size_t embed, size_t heads, bool causal, int workers) {
    if (!q || !k || !v || !dAtt || !attended || !dQ || !dK || !dV ||
        !queries || !keys || !embed || !heads || embed%heads || qStride<embed || kvStride<embed)
        throw std::invalid_argument("invalid attention backward geometry");
    (void)product(product(queries,qStride),sizeof(float));
    (void)product(product(keys,kvStride),sizeof(float));
    (void)RuntimeSimd::selectedIsa(); // Validate outside any OpenMP region.
    const size_t headDim=embed/heads;
    const float scale=1.0f/std::sqrt(static_cast<float>(headDim));
    if(queries<=16 && keys<=16 && headDim<=16) {
        smallBackward(q,k,v,qStride,kvStride,dAtt,attended,dQ,dK,dV,
            queries,keys,embed,heads,causal,scale);
        return;
    }
    constexpr size_t queryTile=64;
    const size_t rows=std::min(queries,queryTile);
    // Padding avoids cache-set conflicts when transposed reductions traverse
    // a power-of-two key count. Padding never participates in arithmetic.
    const size_t keyStride=sum(keys,16);
    const size_t scoreCount=product(rows,keyStride), gradCount=product(keys,headDim);
    const size_t slotSize=product(2,sum(scoreCount,gradCount));
    size_t slots=std::min(heads,size_t(std::max(1,workers)));
#ifdef _OPENMP
    if (omp_in_parallel()) slots=1;
#else
    slots=1;
#endif
    // Allocate before parallel execution: allocation failures propagate normally,
    // and no exception or thread-local lease can escape an OpenMP worker.
    RuntimeScratch::FloatArena workspace({product(slots,slotSize)});
    auto storage=workspace.take(product(slots,slotSize));
    auto runHead=[&](size_t head) {
        size_t slot=0;
#ifdef _OPENMP
        if (slots>1) slot=size_t(omp_get_thread_num());
#endif
        float* p=storage.data()+slot*slotSize;
        float* ds=p+scoreCount;
        float* dk=ds+scoreCount;
        float* dv=dk+gradCount;
        std::fill_n(dk,gradCount,0.0f);
        std::fill_n(dv,gradCount,0.0f);
        const size_t offset=head*headDim;
        const float* kh=k+offset;
        const float* vh=v+offset;
        for (size_t begin=0;begin<queries;) {
            const size_t count=std::min(queryTile,queries-begin);
            const float* qh=q+begin*qStride+offset;
            const float* da=dAtt+begin*embed+offset;
            RuntimeSimd::matmulStrided(p,qh,kh,count,keys,headDim,
                qStride,1,1,kvStride,keyStride,false,false);
            for (size_t row=0;row<count;++row) {
                float* probabilities=p+row*keyStride;
                float maximum=-1e30f;
                for (size_t j=0;j<keys;++j) {
                    const float value=causal && j>begin+row ? -1e9f : probabilities[j]*scale;
                    probabilities[j]=value;
                    if (value>maximum) maximum=value;
                }
                float total=0.0f;
                for (size_t j=0;j<keys;++j) {
                    const float value=std::exp(probabilities[j]-maximum);
                    probabilities[j]=value;
                    total+=value;
                }
                const float inverse=1.0f/(total+1e-9f);
                for (size_t j=0;j<keys;++j) probabilities[j]*=inverse;
            }
            RuntimeSimd::matmulStrided(attended+begin*embed+offset,p,vh,count,headDim,keys,
                keyStride,1,kvStride,1,embed,false,false);
            RuntimeSimd::matmulStridedContinue(dv,p,da,keys,headDim,count,
                1,keyStride,embed,1,headDim);
            RuntimeSimd::matmulStrided(ds,da,vh,count,keys,headDim,
                embed,1,1,kvStride,keyStride,false,false);
            for (size_t row=0;row<count;++row) {
                const float* probabilities=p+row*keyStride;
                float* derivative=ds+row*keyStride;
                float dot=0.0f;
                for (size_t j=0;j<keys;++j) dot+=derivative[j]*probabilities[j];
                for (size_t j=0;j<keys;++j)
                    derivative[j]=causal && j>begin+row ? 0.0f : (derivative[j]-dot)*probabilities[j];
            }
            RuntimeSimd::matmulStrided(dQ+begin*embed+offset,ds,kh,count,headDim,keys,
                keyStride,1,kvStride,1,embed,false,false);
            for (size_t row=0;row<count;++row)for (size_t d=0;d<headDim;++d) {
                float& value=dQ[(begin+row)*embed+offset+d];
                value=0.0f+value*scale;
            }
            // Resume each query term in order across tiles. Adding completed
            // partial GEMMs would reassociate these two reductions.
            RuntimeSimd::matmulStridedContinue(dk,ds,qh,keys,headDim,count,
                1,keyStride,qStride,1,headDim);
            begin+=count;
        }
        for (size_t row=0;row<keys;++row)for (size_t d=0;d<headDim;++d) {
            dK[row*embed+offset+d]=0.0f+dk[row*headDim+d]*scale;
            dV[row*embed+offset+d]=0.0f+dv[row*headDim+d];
        }
    };
#ifdef _OPENMP
    if (slots>1) {
        const int team=int(slots);
        // Each thread reuses a scratch slot between heads. This loop must not
        // use omp simd, which could overlap heads sharing that same slot.
        #pragma omp parallel for schedule(static) num_threads(team)
        for (size_t head=0;head<heads;++head) runHead(head);
        return;
    }
#endif
    for (size_t head=0;head<heads;++head) runHead(head);
}
}
