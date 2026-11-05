#pragma once
#include "PortableLayerGeometry.hpp"
#include "Planning/ExecutionPolicy.hpp"
#include "runtimes/cuda/CudaOperationGeometry.hpp"
#include "runtimes/LayerOps.hpp"

namespace PortableParallelGeometry {
// Match the parallel kernels' reduction and arena limits. Intrinsically serial
// recurrent/NMS algorithms retain their reference budget and fallback behavior.
inline bool fits(const Layer& layer,const std::vector<const std::vector<float>*>& inputs,bool backward) {
    if(inputs.empty() || !inputs[0] || inputs[0]->empty() || inputs[0]->size()>Mimir::Planning::ExecutionPolicy::Limits::portable_parallel_invocations)return false;
    int op=0;float alpha=0;
    if(RuntimeLayerOps::resolveUnaryOp(layer.type_enum,layer,op,alpha))return true;
    if(layer.type_enum==LayerType::LayerNorm || layer.type_enum==LayerType::RMSNorm){
        const int width=PortableLayerGeometry::rowNormWidth(layer,inputs[0]->size(),backward);
        return width>0 && size_t(width)<=Mimir::Planning::ExecutionPolicy::Limits::portable_reduction_span && inputs[0]->size()/size_t(width)<=Mimir::Planning::ExecutionPolicy::Limits::portable_reduction_span && layer.getWeightsSize()<=Mimir::Planning::ExecutionPolicy::Limits::portable_parallel_invocations;
    }
    if(layer.type_enum==LayerType::Linear){
        return layer.in_features>0 && layer.out_features>0 && inputs[0]->size()%layer.in_features==0 && layer.getWeightsSize()<=Mimir::Planning::ExecutionPolicy::Limits::portable_parallel_invocations;
    }
    // Geometry validation remains in PortableRuntime. These families now use
    // parallel kernels (CAS for scatter), while sequential algorithms stay bounded.
    if(layer.type_enum==LayerType::NMS || CudaOperationGeometry::recurrentType(layer.type_enum))return false;
    size_t total=layer.getWeightsSize();if(total>Mimir::Planning::ExecutionPolicy::Limits::portable_staged_elements)return false;
    for(const auto* input:inputs){if(input){if(input->size()>Mimir::Planning::ExecutionPolicy::Limits::portable_staged_elements-total)return false;total+=input->size();}}
    return total<=Mimir::Planning::ExecutionPolicy::Limits::portable_staged_elements;
}
}
