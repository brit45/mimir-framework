#pragma once
#ifndef MIMIR_CONVND_KERNELS_ONLY
#include "CudaNativeOps.cuh"
#endif
#include "CudaOperationGeometry.hpp"
#include "runtimes/FiniteCheck.hpp"

namespace CudaNativeOps {
// Every invocation owns its destination. Gather adjoints avoid atomic races and
// preserve a fixed reduction order for both inputs and parameters.
__global__ static void convolution_nd_forward(const float* x,const float* w,float* y,CudaOperationGeometry::ConvND g,int count){
    const int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=count)return;
    const int spatial=index%g.output_spatial,oc=(index/g.output_spatial)%g.oc,n=index/(g.output_spatial*g.oc);
    const int a0=spatial/(g.o1*g.o2*g.o3),a1=(spatial/(g.o2*g.o3))%g.o1,a2=(spatial/g.o3)%g.o2,a3=spatial%g.o3;
    const int first=(oc/g.og)*g.ig;
    float sum=0.f;
    for(int c=0;c<g.ig;++c)for(int k=0;k<g.kernel_spatial;++k){
        const int z0=a0*g.s0-g.p0+(k/(g.k1*g.k2*g.k3))*g.d0;
        const int z1=a1*g.s1-g.p1+((k/(g.k2*g.k3))%g.k1)*g.d1;
        const int z2=a2*g.s2-g.p2+((k/g.k3)%g.k2)*g.d2;
        const int z3=a3*g.s3-g.p3+(k%g.k3)*g.d3;
        if(z0<0 || z0>=g.i0 || z1<0 || z1>=g.i1 || z2<0 || z2>=g.i2 || z3<0 || z3>=g.i3)continue;
        const int source=(n*g.ic+first+c)*g.input_spatial+((z0*g.i1+z1)*g.i2+z2)*g.i3+z3;
        sum=fmaf(x[source],w[(oc*g.ig+c)*g.kernel_spatial+k],sum);
    }
    y[index]=sum+(g.bias?w[g.parameters+oc]:0.f);
}
__global__ static void convolution_nd_input_grad(const float* w,const float* dy,float* dx,CudaOperationGeometry::ConvND g,int count){
    const int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=count)return;
    const int spatial=index%g.input_spatial,ic=(index/g.input_spatial)%g.ic,n=index/(g.input_spatial*g.ic);
    const int a0=spatial/(g.i1*g.i2*g.i3),a1=(spatial/(g.i2*g.i3))%g.i1,a2=(spatial/g.i3)%g.i2,a3=spatial%g.i3;
    const int first=(ic/g.ig)*g.og,channel=ic%g.ig;
    float sum=0.f;
    for(int oc=first;oc<first+g.og;++oc)for(int k=0;k<g.kernel_spatial;++k){
        const int z0=a0+g.p0-(k/(g.k1*g.k2*g.k3))*g.d0;
        const int z1=a1+g.p1-((k/(g.k2*g.k3))%g.k1)*g.d1;
        const int z2=a2+g.p2-((k/g.k3)%g.k2)*g.d2;
        const int z3=a3+g.p3-(k%g.k3)*g.d3;
        if(z0<0 || z1<0 || z2<0 || z3<0 || z0%g.s0 || z1%g.s1 || z2%g.s2 || z3%g.s3)continue;
        const int b0=z0/g.s0,b1=z1/g.s1,b2=z2/g.s2,b3=z3/g.s3;
        if(b0>=g.o0 || b1>=g.o1 || b2>=g.o2 || b3>=g.o3)continue;
        const int dest=(n*g.oc+oc)*g.output_spatial+((b0*g.o1+b1)*g.o2+b2)*g.o3+b3;
        sum=fmaf(dy[dest],w[(oc*g.ig+channel)*g.kernel_spatial+k],sum);
    }
    dx[index]=sum;
}
__global__ static void convolution_nd_weight_grad(const float* x,const float* dy,float* dw,CudaOperationGeometry::ConvND g,int count){
    const int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=count)return;
    const bool bias=index>=g.parameters;
    const int oc=bias?index-g.parameters:index/(g.ig*g.kernel_spatial);
    const int channel=bias?0:(index/g.kernel_spatial)%g.ig,k=bias?0:index%g.kernel_spatial;
    const int ic=(oc/g.og)*g.ig+channel;
    float sum=0.f;
    for(int n=0;n<g.batch;++n)for(int o=0;o<g.output_spatial;++o){
        const int dest=(n*g.oc+oc)*g.output_spatial+o;
        if(bias){sum+=dy[dest];continue;}
        const int z0=(o/(g.o1*g.o2*g.o3))*g.s0-g.p0+(k/(g.k1*g.k2*g.k3))*g.d0;
        const int z1=((o/(g.o2*g.o3))%g.o1)*g.s1-g.p1+((k/(g.k2*g.k3))%g.k1)*g.d1;
        const int z2=((o/g.o3)%g.o2)*g.s2-g.p2+((k/g.k3)%g.k2)*g.d2;
        const int z3=(o%g.o3)*g.s3-g.p3+(k%g.k3)*g.d3;
        if(z0<0 || z0>=g.i0 || z1<0 || z1>=g.i1 || z2<0 || z2>=g.i2 || z3<0 || z3>=g.i3)continue;
        const int source=(n*g.ic+ic)*g.input_spatial+((z0*g.i1+z1)*g.i2+z2)*g.i3+z3;
        sum=fmaf(dy[dest],x[source],sum);
    }
    dw[index]=sum;
}
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
        convolution_nd_input_grad<<<1+(count-1)/threads,threads>>>(bw.ptr,bg.ptr,by.ptr,g,count);
        convolution_nd_weight_grad<<<1+(parameters-1)/threads,threads>>>(bx.ptr,bg.ptr,bdw.ptr,g,parameters);
    }else convolution_nd_forward<<<1+(count-1)/threads,threads>>>(bx.ptr,bw.ptr,by.ptr,g,count);
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
} // namespace CudaNativeOps
