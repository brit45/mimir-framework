#pragma once
// Source of the generated portable device algorithms. Vendor CUDA/HIP runtimes
// do not advertise these operations unless they implement their own dispatch.
#include "CudaNativeOps.cuh"
#include "CudaOperationGeometry.hpp"
#include "runtimes/FiniteCheck.hpp"
#include "runtimes/portable/PortablePrimerValidation.hpp"

namespace CudaNativeOps {
__device__ inline float primer_quiet_nan(){
    return __int_as_float(0x7fc00000);
}
// A zero can hide a device-flushed nonzero result. Mark only that ambiguous
// device result; the router can retry the operation on the CPU unchanged.
__device__ inline float primer_zero_guard(float result,bool expected_nonzero){
    return result==0.f && expected_nonzero?primer_quiet_nan():result;
}

__global__ static void primer_unary_forward(const float* x,float* y,int count,int op){
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=count)return;
    const float value=x[index];
    float result=0.f;
    switch(op){
        case 0:if(value>0.f)result=value*value;break;
        case 1:result=value*value;break;
        case 2:result=sqrtf(fabsf(value));break;
        case 3:result=expf(value);break;
        case 4:result=logf(fabsf(value));break;
        case 5:result=fabsf(value);break;
        case 6:if(value!=0.f)result=1.f/value;break;
        case 7:result=value>0.f?1.f:value<0.f?-1.f:0.f;break;
        case 8:result=sinf(value);break;
        case 9:result=cosf(value);break;
    }
    bool expected_nonzero=value!=0.f;
    if(op==0)expected_nonzero=value>0.f;
    else if(op==3 || op==9)expected_nonzero=true;
    else if(op==4)expected_nonzero=fabsf(value)!=1.f;
    y[index]=primer_zero_guard(result,expected_nonzero);
}

__global__ static void primer_unary_backward(const float* x,const float* dy,float* dx,int count,int op){
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=count)return;
    const float value=x[index],grad=dy[index];
    float result=0.f;
    switch(op){
        case 0:if(value>0.f)result=grad*(2.f*value);break;
        case 1:result=grad*(2.f*value);break;
        case 2:result=((0.5f*grad)*(value>0.f?1.f:-1.f))/sqrtf(fabsf(value));break;
        case 3:result=grad*expf(value);break;
        case 4:result=grad/value;break;
        case 5:result=grad*(value>0.f?1.f:value<0.f?-1.f:0.f);break;
        case 6:if(value!=0.f)result=-(grad/value)/value;break;
        case 7:break;
        case 8:result=grad*cosf(value);break;
        case 9:result=(-grad)*sinf(value);break;
    }
    bool expected_nonzero=grad!=0.f;
    if(op==7)expected_nonzero=false;
    else if(op==0)expected_nonzero=grad!=0.f && value>0.f;
    else if(op==1 || op==5 || op==6 || op==9)expected_nonzero=grad!=0.f && value!=0.f;
    dx[index]=primer_zero_guard(result,expected_nonzero);
}

__device__ inline float primer_binary_gradient(float a,float b,float gradient,int op,bool first){
    if(a==b)return primer_zero_guard(gradient*0.5f,gradient!=0.f);
    const bool select_first=op==0?a>b:a<b;
    return select_first==first?gradient:0.f;
}

__global__ static void primer_binary_forward(const float* a,const float* b,float* y,
    int na,int nb,int count,int op){
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=count)return;
    const float av=a[na==1?0:index],bv=b[nb==1?0:index];
    y[index]=op==0?(av>=bv?av:bv):(av<=bv?av:bv);
}

__global__ static void primer_binary_backward(const float* a,const float* b,const float* dy,
    float* da,float* db,int na,int nb,int count,int op){
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=count)return;
    if(index<na){
        float sum=0.f;
        bool any_nonzero=false;
        if(na==1){
            for(int j=0;j<count;++j){
                const float contribution=primer_binary_gradient(a[0],b[nb==1?0:j],dy[j],op,true);
                any_nonzero=any_nonzero || contribution!=0.f;
                sum+=contribution;
            }
        }else{
            const float contribution=primer_binary_gradient(a[index],b[nb==1?0:index],dy[index],op,true);
            any_nonzero=contribution!=0.f;
            sum+=contribution;
        }
        da[index]=primer_zero_guard(sum,any_nonzero);
    }
    if(index<nb){
        float sum=0.f;
        bool any_nonzero=false;
        if(nb==1){
            for(int j=0;j<count;++j){
                const float contribution=primer_binary_gradient(a[na==1?0:j],b[0],dy[j],op,false);
                any_nonzero=any_nonzero || contribution!=0.f;
                sum+=contribution;
            }
        }else{
            const float contribution=primer_binary_gradient(a[na==1?0:index],b[index],dy[index],op,false);
            any_nonzero=contribution!=0.f;
            sum+=contribution;
        }
        db[index]=primer_zero_guard(sum,any_nonzero);
    }
}

__global__ static void primer_reduce_forward(const float* x,float* y,
    CudaOperationGeometry::PrimerReduction g,int op){
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=g.outputs)return;
    const int base=(index/g.inner)*g.axis_size*g.inner+index%g.inner;
    float result=op==4?1.f:0.f;
    bool any_nonzero=false,zero_factor=false;
    if(op==2 || op==3)result=x[base];
    for(int a=0;a<g.axis_size;++a){
        const float value=x[base+a*g.inner];
        any_nonzero=any_nonzero || value!=0.f;
        zero_factor=zero_factor || value==0.f;
        if(op==0 || op==1)result+=value;
        else if(op==2){if(value<result)result=value;}
        else if(op==3){if(value>result)result=value;}
        else result*=value;
    }
    if(op==1)result/=float(g.axis_size);
    const bool expected_nonzero=op==4?!zero_factor:(op<=1 && any_nonzero);
    y[index]=primer_zero_guard(result,expected_nonzero);
}

__global__ static void primer_reduce_backward(const float* x,const float* dy,float* dx,
    CudaOperationGeometry::PrimerReduction g,int op){
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=g.values)return;
    const int axis=(index/g.inner)%g.axis_size;
    const int group=(index/(g.axis_size*g.inner))*g.inner+index%g.inner;
    const int base=(group/g.inner)*g.axis_size*g.inner+group%g.inner;
    float result=dy[group];
    bool expected_nonzero=result!=0.f;
    if(op==1)result/=float(g.axis_size);
    else if(op==2 || op==3){
        float extreme=x[base];
        for(int a=1;a<g.axis_size;++a){
            const float value=x[base+a*g.inner];
            if(op==2?value<extreme:value>extreme)extreme=value;
        }
        int ties=0;
        for(int a=0;a<g.axis_size;++a)if(x[base+a*g.inner]==extreme)++ties;
        result=x[index]==extreme?result/float(ties):0.f;
        expected_nonzero=x[index]==extreme && dy[group]!=0.f;
    }else if(op==4){
        // Recompute every other factor, preserving zero and underflow behavior.
        float product=1.f;
        bool zero_factor=false;
        for(int a=0;a<g.axis_size;++a)if(a!=axis){
            const float value=x[base+a*g.inner];
            zero_factor=zero_factor || value==0.f;
            product*=value;
        }
        result*=product;
        expected_nonzero=dy[group]!=0.f && !zero_factor;
    }
    dx[index]=primer_zero_guard(result,expected_nonzero);
}

__global__ static void primer_triangle(const float* x,float* y,int rows,int columns,int count){
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=count)return;
    const int row=(index/columns)%rows,column=index%columns;
    y[index]=column<=row?x[index]:0.f;
}

inline bool primer_finite(const std::vector<float>& values){
    return Mimir::RuntimeChecks::allFinite(values.data(),values.size());
}
inline bool primer_finite(const std::vector<std::vector<float>>& values){
    for(const auto& value:values)if(!primer_finite(value))return false;
    return true;
}
inline bool primer_unary(const std::vector<float>& x,std::vector<std::vector<float>>& output,
    int op,const std::vector<float>* dy){
    if(x.empty() || x.size()>INT_MAX || !PortablePrimerValidation::supportedValues(x) ||
       (dy && (dy->size()!=x.size() || !PortablePrimerValidation::supportedValues(*dy))))return false;
    if(op==4 || (op==2 && dy))for(float value:x)if(value==0.f)return false;
    Buffer bx,by,bg;
    if(!bx.alloc(x.size()) || !bx.upload(x.data(),x.size()) || !by.alloc(x.size()))return false;
    const int count=static_cast<int>(x.size());
    if(dy){
        if(!bg.alloc(dy->size()) || !bg.upload(dy->data(),dy->size()))return false;
        primer_unary_backward<<<1+(count-1)/threads,threads>>>(bx.ptr,bg.ptr,by.ptr,count,op);
    }else primer_unary_forward<<<1+(count-1)/threads,threads>>>(bx.ptr,by.ptr,count,op);
    if(!finished())return false;
    std::vector<std::vector<float>> result(1,std::vector<float>(x.size()));
    if(!by.download(result[0].data(),x.size()) || !primer_finite(result))return false;
    output=std::move(result);
    return true;
}

inline bool primer_binary(const std::vector<float>& a,const std::vector<float>& b,
    std::vector<std::vector<float>>& output,int op,const std::vector<float>* dy){
    const size_t n=std::max(a.size(),b.size());
    if(!n || n>INT_MAX || a.empty() || b.empty() ||
       (a.size()!=b.size() && a.size()!=1 && b.size()!=1) ||
       !PortablePrimerValidation::supportedValues(a) || !PortablePrimerValidation::supportedValues(b) ||
       (dy && (dy->size()!=n || !PortablePrimerValidation::supportedValues(*dy))))return false;
    Buffer ba,bb,by,bg,bda,bdb;
    if(!ba.alloc(a.size()) || !ba.upload(a.data(),a.size()) ||
       !bb.alloc(b.size()) || !bb.upload(b.data(),b.size()))return false;
    const int count=static_cast<int>(n),na=static_cast<int>(a.size()),nb=static_cast<int>(b.size());
    if(dy){
        if(!bg.alloc(n) || !bg.upload(dy->data(),n) || !bda.alloc(a.size()) || !bdb.alloc(b.size()))return false;
        primer_binary_backward<<<1+(count-1)/threads,threads>>>(ba.ptr,bb.ptr,bg.ptr,bda.ptr,bdb.ptr,na,nb,count,op);
    }else{
        if(!by.alloc(n))return false;
        primer_binary_forward<<<1+(count-1)/threads,threads>>>(ba.ptr,bb.ptr,by.ptr,na,nb,count,op);
    }
    if(!finished())return false;
    std::vector<std::vector<float>> result;
    if(dy){
        result={std::vector<float>(a.size()),std::vector<float>(b.size())};
        if(!bda.download(result[0].data(),a.size()) || !bdb.download(result[1].data(),b.size()))return false;
    }else{
        result={std::vector<float>(n)};
        if(!by.download(result[0].data(),n))return false;
    }
    if(!primer_finite(result))return false;
    output=std::move(result);
    return true;
}

inline bool primer_reduction(const std::vector<float>& x,std::vector<std::vector<float>>& output,
    const Layer& layer,int op,const std::vector<float>* dy){
    CudaOperationGeometry::PrimerReduction g;
    if(!g.valid(layer,x.size()) || !PortablePrimerValidation::supportedValues(x) ||
       (dy && (dy->size()!=static_cast<size_t>(g.outputs) ||
        !PortablePrimerValidation::supportedValues(*dy))))return false;
    Buffer bx,by,bg;
    const int count=dy?g.values:g.outputs;
    if(!bx.alloc(x.size()) || !bx.upload(x.data(),x.size()) || !by.alloc(count))return false;
    if(dy){
        if(!bg.alloc(dy->size()) || !bg.upload(dy->data(),dy->size()))return false;
        primer_reduce_backward<<<1+(count-1)/threads,threads>>>(bx.ptr,bg.ptr,by.ptr,g,op);
    }else primer_reduce_forward<<<1+(count-1)/threads,threads>>>(bx.ptr,by.ptr,g,op);
    if(!finished())return false;
    std::vector<std::vector<float>> result(1,std::vector<float>(count));
    if(!by.download(result[0].data(),result[0].size()) || !primer_finite(result))return false;
    output=std::move(result);
    return true;
}

inline bool primer_lower_triangle(const std::vector<float>& x,std::vector<std::vector<float>>& output,
    const Layer& layer,const std::vector<float>* dy){
    CudaOperationGeometry::PrimerTriangle g;
    if(!g.valid(layer,x.size()) || !PortablePrimerValidation::supportedValues(x) ||
       (dy && (dy->size()!=x.size() || !PortablePrimerValidation::supportedValues(*dy))))return false;
    const auto& source=dy?*dy:x;
    Buffer bx,by;
    if(!bx.alloc(source.size()) || !bx.upload(source.data(),source.size()) || !by.alloc(source.size()))return false;
    primer_triangle<<<1+(g.values-1)/threads,threads>>>(bx.ptr,by.ptr,g.rows,g.columns,g.values);
    if(!finished())return false;
    std::vector<std::vector<float>> result(1,std::vector<float>(source.size()));
    if(!by.download(result[0].data(),source.size()) || !primer_finite(result))return false;
    output=std::move(result);
    return true;
}

inline bool primer(const std::vector<const std::vector<float>*>& inputs,
    std::vector<std::vector<float>>& output,const Layer& layer,const std::vector<float>* dy=nullptr){
    if(inputs.empty() || !inputs[0] || layer.getWeightsSize()!=0)return false;
    const int unary=RuntimePrimer::unaryCode(layer.type_enum);
    if(unary>=0)return inputs.size()==1 && primer_unary(*inputs[0],output,unary,dy);
    const int binary=RuntimePrimer::binaryCode(layer.type_enum);
    if(binary>=0)return inputs.size()==2 && inputs[1] && primer_binary(*inputs[0],*inputs[1],output,binary,dy);
    const int reduction=RuntimePrimer::reductionCode(layer.type_enum);
    if(reduction>=0)return inputs.size()==1 && primer_reduction(*inputs[0],output,layer,reduction,dy);
    return inputs.size()==1 && RuntimePrimer::lowerTriangularType(layer.type_enum) &&
        primer_lower_triangle(*inputs[0],output,layer,dy);
}
} // namespace CudaNativeOps
