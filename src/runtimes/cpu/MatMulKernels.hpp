#pragma once
#include "SIMD_Ops.hpp"
#include "../LayerOps.hpp"

namespace RuntimeMatMul {
// Each batch-row is independent; the dispatcher creates a single team and
// selects a safe vector width without changing reduction order.
inline void forwardBatched(const float* a,const float* b,float* y,
                           size_t batches,size_t m,size_t k,size_t n) {
    const int workers=HardwareOpt::matmul_workers(batches*m*((n+23)/24),24,k);
    if(RuntimeLayerOps::hardwareAccelerationEnabled()) {
        RuntimeSimd::matmulBatched(y,a,b,batches,m,n,k,true,workers);
    } else {
        RuntimeParallel::forEach(batches*m,workers,[&](size_t row) {
            for(size_t j=0;j<n;++j) {
                float sum=0;
                for(size_t c=0;c<k;++c)sum=std::fma(a[row*k+c],b[(row/m)*k*n+c*n+j],sum);
                y[row*n+j]=sum;
            }
        });
    }
}

inline void backward(const float* a,const float* b,const float* dy,
                     float* da,float* db,size_t m,size_t k,size_t n) {
    if(RuntimeLayerOps::hardwareAccelerationEnabled()) {
        RuntimeSimd::matmulTranspose(da,dy,b,m,k,n,true,HardwareOpt::matmul_workers(m*((k+23)/24),24,n));
        RuntimeSimd::matmulLeftTranspose(db,a,dy,m,n,k,true,HardwareOpt::matmul_workers(k*((n+23)/24),24,m));
    } else {
        for(size_t i=0;i<m;++i)for(size_t c=0;c<k;++c) {
            float sum=0;
            for(size_t j=0;j<n;++j)sum=std::fma(dy[i*n+j],b[c*n+j],sum);
            da[i*k+c]=sum;
        }
        for(size_t c=0;c<k;++c)for(size_t j=0;j<n;++j) {
            float sum=0;
            for(size_t i=0;i<m;++i)sum=std::fma(a[i*k+c],dy[i*n+j],sum);
            db[c*n+j]=sum;
        }
    }
}
}
