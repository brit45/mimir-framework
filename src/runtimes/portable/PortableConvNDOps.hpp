#pragma once
#ifndef MIMIR_CONVND_KERNELS_ONLY
#include "PortableNativeOps.hpp"
#endif
#include "runtimes/cuda/CudaOperationGeometry.hpp"
#include "runtimes/FiniteCheck.hpp"

namespace PortableNativeOps {
using namespace PortableStaging;
// Every invocation owns its destination. Gather adjoints avoid atomic races and
// preserve a fixed reduction order for both inputs and parameters.



#ifndef MIMIR_CONVND_KERNELS_ONLY
inline bool convolution_nd(const std::vector<float>& x,const Layer& l,const CudaOperationGeometry::ConvND& g,
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
        launch("convolution_nd_input_grad", 1+(count-1)/threads,threads, bw.ptr,bg.ptr,by.ptr,g,count);
        launch("convolution_nd_weight_grad", 1+(parameters-1)/threads,threads, bx.ptr,bg.ptr,bdw.ptr,g,parameters);
    }else launch("convolution_nd_forward", 1+(count-1)/threads,threads, bx.ptr,bw.ptr,by.ptr,g,count);
    if(!finished())return false;
    std::vector<float> result(count),parameters;
    if(!by.download(result.data(),result.size()))return false;
    if(dy){parameters.resize(g.parameterSize());if(!bdw.download(parameters.data(),parameters.size()))return false;}
    if(!Mimir::RuntimeChecks::allFinite(result.data(),result.size()) ||
       !Mimir::RuntimeChecks::allFinite(parameters.data(),parameters.size()))return false;
    y=std::move(result);
    if(dy)*dw=std::move(parameters);
    return true;
}
#endif
} // namespace PortableNativeOps
