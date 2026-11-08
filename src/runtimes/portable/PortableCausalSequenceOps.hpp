#pragma once
// Shared source for generated OpenCL/Vulkan kernels. CUDA capability is not
// advertised by including this source in the portable generator.
#include "PortableNativeOps.hpp"
#include "runtimes/cuda/CudaOperationGeometry.hpp"
#include "runtimes/FiniteCheck.hpp"

namespace PortableNativeOps {
using namespace PortableStaging;
// Each invocation owns its destination; no scatter or floating-point atomics.












inline bool causal_depthwise(const std::vector<float>& x,const Layer& layer,
    const CudaOperationGeometry::CausalSequence& g,std::vector<float>& result,
    const std::vector<float>* dy=nullptr,std::vector<float>* dw=nullptr){
    if(!Mimir::RuntimeChecks::allFinite(x.data(),x.size()) ||
       !Mimir::RuntimeChecks::allFinite(layer.getWeights(),layer.getWeightsSize()) ||
       (dy && (!dw || dy->size()!=x.size() ||
        !Mimir::RuntimeChecks::allFinite(dy->data(),dy->size()))))return false;
    Buffer bx,bw,by,bg,bdw;
    if(!bx.alloc(x.size()) || !bx.upload(x.data(),x.size()) ||
       !bw.alloc(layer.getWeightsSize()) || !bw.upload(layer.getWeights(),layer.getWeightsSize()) ||
       !by.alloc(x.size()))return false;
    if(dy){
        if(!bg.alloc(dy->size()) || !bg.upload(dy->data(),dy->size()) ||
           !bdw.alloc(g.parameters))return false;
        launch("causal_depthwise_input_grad", 1+(g.values-1)/threads,threads, bw.ptr,bg.ptr,by.ptr,g,g.values);
        launch("causal_depthwise_weight_grad", 1+(g.parameters-1)/threads,threads, bx.ptr,bg.ptr,bdw.ptr,g,g.parameters);
    }else{
        launch("causal_depthwise_forward", 1+(g.values-1)/threads,threads, bx.ptr,bw.ptr,by.ptr,g,g.values);
    }
    if(!finished())return false;
    std::vector<float> staged(x.size()),parameters;
    if(!by.download(staged.data(),staged.size()))return false;
    if(dy){
        parameters.resize(g.parameters);
        if(!bdw.download(parameters.data(),parameters.size()))return false;
    }
    if(!Mimir::RuntimeChecks::allFinite(staged.data(),staged.size()) ||
       !Mimir::RuntimeChecks::allFinite(parameters.data(),parameters.size()))return false;
    result=std::move(staged);
    if(dy)*dw=std::move(parameters);
    return true;
}

inline bool swiglu(const std::vector<float>& gate,const std::vector<float>& value,
    std::vector<std::vector<float>>& result,const std::vector<float>* dy=nullptr){
    if(gate.empty() || gate.size()>INT_MAX || gate.size()!=value.size() ||
       !Mimir::RuntimeChecks::allFinite(gate.data(),gate.size()) ||
       !Mimir::RuntimeChecks::allFinite(value.data(),value.size()) ||
       (dy && (dy->size()!=gate.size() ||
        !Mimir::RuntimeChecks::allFinite(dy->data(),dy->size()))))return false;
    Buffer bg,bv,by,bdg,bdu;
    if(!bg.alloc(gate.size()) || !bg.upload(gate.data(),gate.size()) ||
       !bv.alloc(value.size()) || !bv.upload(value.data(),value.size()))return false;
    const int count=static_cast<int>(gate.size());
    if(dy){
        if(!by.alloc(dy->size()) || !by.upload(dy->data(),dy->size()) ||
           !bdg.alloc(gate.size()) || !bdu.alloc(gate.size()))return false;
        launch("swiglu_backward", 1+(count-1)/threads,threads, bg.ptr,bv.ptr,by.ptr,bdg.ptr,bdu.ptr,count);
    }else{
        if(!by.alloc(gate.size()))return false;
        launch("swiglu_forward", 1+(count-1)/threads,threads, bg.ptr,bv.ptr,by.ptr,count);
    }
    if(!finished())return false;
    std::vector<std::vector<float>> staged(dy?2:1,std::vector<float>(gate.size()));
    if(dy){
        if(!bdg.download(staged[0].data(),gate.size()) ||
           !bdu.download(staged[1].data(),gate.size()))return false;
    }else if(!by.download(staged[0].data(),gate.size()))return false;
    for(const auto& values:staged)
        if(!Mimir::RuntimeChecks::allFinite(values.data(),values.size()))return false;
    result=std::move(staged);
    return true;
}
} // namespace PortableNativeOps
