#pragma once
// Shared source for generated OpenCL/Vulkan kernels. CUDA capability is not
// advertised by including this source in the portable generator.
#include "CudaNativeOps.cuh"
#include "CudaOperationGeometry.hpp"
#include "runtimes/FiniteCheck.hpp"

namespace CudaNativeOps {
// Each invocation owns its destination; no scatter or floating-point atomics.
__global__ static void causal_depthwise_forward(const float* x,const float* w,float* y,
    CudaOperationGeometry::CausalSequence g,int count){
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=count)return;
    const int t=index/g.channels,c=index%g.channels;
    const int last=min(g.taps-1,t/g.dilation);
    float sum=0.f;
    for(int i=0;i<=last;++i)
        sum=fmaf(w[c*g.taps+i],x[(t-i*g.dilation)*g.channels+c],sum);
    y[index]=sum+(g.bias?w[g.weights+c]:0.f);
}

__global__ static void causal_depthwise_input_grad(const float* w,const float* dy,float* dx,
    CudaOperationGeometry::CausalSequence g,int count){
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=count)return;
    const int t=index/g.channels,c=index%g.channels;
    const int last=min(g.taps-1,(g.steps-1-t)/g.dilation);
    float sum=0.f;
    for(int i=0;i<=last;++i)
        sum=fmaf(w[c*g.taps+i],dy[(t+i*g.dilation)*g.channels+c],sum);
    dx[index]=sum;
}

__global__ static void causal_depthwise_weight_grad(const float* x,const float* dy,float* dw,
    CudaOperationGeometry::CausalSequence g,int count){
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=count)return;
    const bool is_bias=index>=g.weights;
    const int c=is_bias?index-g.weights:index/g.taps;
    const int i=is_bias?0:index%g.taps;
    float sum=0.f;
    // Division before multiplication keeps even very large dilations safe.
    if(!is_bias && i>(g.steps-1)/g.dilation){dw[index]=0.f;return;}
    const int first=is_bias?0:i*g.dilation;
    for(int t=first;t<g.steps;++t){
        if(is_bias)sum+=dy[t*g.channels+c];
        else sum=fmaf(dy[t*g.channels+c],x[(t-first)*g.channels+c],sum);
    }
    dw[index]=sum;
}

__device__ inline float causal_sequence_sigmoid(float gate){
    if(gate>=0.f)return 1.f/(1.f+expf(-gate));
    const float exponential=expf(gate);
    return exponential/(1.f+exponential);
}

__global__ static void swiglu_forward(const float* gate,const float* value,float* output,int count){
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=count)return;
    const float sig=causal_sequence_sigmoid(gate[index]);
    output[index]=(gate[index]*sig)*value[index];
}

__global__ static void swiglu_backward(const float* gate,const float* value,const float* dy,
    float* dgate,float* dvalue,int count){
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=count)return;
    const float g=gate[index],sig=causal_sequence_sigmoid(g);
    const float silu=g*sig,derivative=sig+g*sig*(1.f-sig);
    dgate[index]=(dy[index]*value[index])*derivative;
    dvalue[index]=dy[index]*silu;
}

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
        causal_depthwise_input_grad<<<1+(g.values-1)/threads,threads>>>(bw.ptr,bg.ptr,by.ptr,g,g.values);
        causal_depthwise_weight_grad<<<1+(g.parameters-1)/threads,threads>>>(bx.ptr,bg.ptr,bdw.ptr,g,g.parameters);
    }else{
        causal_depthwise_forward<<<1+(g.values-1)/threads,threads>>>(bx.ptr,bw.ptr,by.ptr,g,g.values);
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
        swiglu_backward<<<1+(count-1)/threads,threads>>>(bg.ptr,bv.ptr,by.ptr,bdg.ptr,bdu.ptr,count);
    }else{
        if(!by.alloc(gate.size()))return false;
        swiglu_forward<<<1+(count-1)/threads,threads>>>(bg.ptr,bv.ptr,by.ptr,count);
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
} // namespace CudaNativeOps
