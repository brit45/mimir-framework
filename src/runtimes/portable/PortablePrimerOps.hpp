#pragma once
// Source of the generated portable device algorithms. Vendor CUDA/HIP runtimes
// do not advertise these operations unless they implement their own dispatch.
#include "PortableNativeOps.hpp"
#include "runtimes/cuda/CudaOperationGeometry.hpp"
#include "runtimes/FiniteCheck.hpp"
#include "runtimes/portable/PortablePrimerValidation.hpp"

namespace PortableNativeOps {
using namespace PortableStaging;

// A zero can hide a device-flushed nonzero result. Mark only that ambiguous
// device result; the router can retry the operation on the CPU unchanged.


















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
        launch("primer_unary_backward", 1+(count-1)/threads,threads, bx.ptr,bg.ptr,by.ptr,count,op);
    }else launch("primer_unary_forward", 1+(count-1)/threads,threads, bx.ptr,by.ptr,count,op);
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
        launch("primer_binary_backward", 1+(count-1)/threads,threads, ba.ptr,bb.ptr,bg.ptr,bda.ptr,bdb.ptr,na,nb,count,op);
    }else{
        if(!by.alloc(n))return false;
        launch("primer_binary_forward", 1+(count-1)/threads,threads, ba.ptr,bb.ptr,by.ptr,na,nb,count,op);
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
        launch("primer_reduce_backward", 1+(count-1)/threads,threads, bx.ptr,bg.ptr,by.ptr,g,op);
    }else launch("primer_reduce_forward", 1+(count-1)/threads,threads, bx.ptr,by.ptr,g,op);
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
    launch("primer_triangle", 1+(g.values-1)/threads,threads, bx.ptr,by.ptr,g.rows,g.columns,g.values);
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
} // namespace PortableNativeOps
