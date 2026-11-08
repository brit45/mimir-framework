#pragma once

// HIP compiles the very same gather kernels used by CUDA and the portable
// generator. An ordinary host compiler never advertises this path as native.
#if defined(ENABLE_ROCM) && defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include "runtimes/FiniteCheck.hpp"
#define MIMIR_CONVND_KERNELS_ONLY
#include "runtimes/cuda/CudaConvNDOps.cuh"
#undef MIMIR_CONVND_KERNELS_ONLY
namespace RocmConvNDOps {
struct Buffer {
    float* ptr=nullptr;
    ~Buffer(){if(ptr)hipFree(ptr);}
    Buffer()=default;
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
    bool alloc(size_t n){return n && hipMalloc(&ptr,n*sizeof(float))==hipSuccess;}
    bool upload(const float* x,size_t n){return hipMemcpy(ptr,x,n*sizeof(float),hipMemcpyHostToDevice)==hipSuccess;}
    bool download(float* x,size_t n){return hipMemcpy(x,ptr,n*sizeof(float),hipMemcpyDeviceToHost)==hipSuccess;}
};
inline bool execute(const std::vector<float>& x,const Layer& l,const CudaOperationGeometry::ConvND& g,
    std::vector<float>& y,const std::vector<float>* dy=nullptr,std::vector<float>* dw=nullptr){
    if(!Mimir::RuntimeChecks::allFinite(x.data(),x.size()) ||
       !Mimir::RuntimeChecks::allFinite(l.getWeights(),l.getWeightsSize()) ||
       (dy && !Mimir::RuntimeChecks::allFinite(dy->data(),dy->size())))return false;
    Buffer bx,bw,bg,by,bdw;
    if(!bx.alloc(x.size()) || !bx.upload(x.data(),x.size()) || !bw.alloc(l.getWeightsSize()) || !bw.upload(l.getWeights(),l.getWeightsSize()))return false;
    const int count=dy?g.inputSize():g.outputSize();
    if(!by.alloc(count))return false;
    if(dy){
        if(!dw || dy->size()!=g.outputSize() || !bg.alloc(dy->size()) || !bg.upload(dy->data(),dy->size()) || !bdw.alloc(g.parameterSize()))return false;
        const int parameters=g.parameterSize();
        hipLaunchKernelGGL(CudaNativeOps::convolution_nd_input_grad,dim3(1+(count-1)/256),dim3(256),0,0,bw.ptr,bg.ptr,by.ptr,g,count);
        hipLaunchKernelGGL(CudaNativeOps::convolution_nd_weight_grad,dim3(1+(parameters-1)/256),dim3(256),0,0,bx.ptr,bg.ptr,bdw.ptr,g,parameters);
    }else hipLaunchKernelGGL(CudaNativeOps::convolution_nd_forward,dim3(1+(count-1)/256),dim3(256),0,0,bx.ptr,bw.ptr,by.ptr,g,count);
    if(hipGetLastError()!=hipSuccess || hipDeviceSynchronize()!=hipSuccess)return false;
    std::vector<float> result(count),parameters;
    if(!by.download(result.data(),result.size()))return false;
    if(dy){parameters.resize(g.parameterSize());if(!bdw.download(parameters.data(),parameters.size()))return false;}
    if(!Mimir::RuntimeChecks::allFinite(result.data(),result.size()) ||
       !Mimir::RuntimeChecks::allFinite(parameters.data(),parameters.size()))return false;
    y=std::move(result);
    if(dy)*dw=std::move(parameters);
    return true;
}
}
#endif
