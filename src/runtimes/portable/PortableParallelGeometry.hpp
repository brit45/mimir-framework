#pragma once
#include "PortableLayerGeometry.hpp"
#include "Planning/ExecutionPolicy.hpp"
#include "runtimes/cuda/CudaOperationGeometry.hpp"
#include "runtimes/LayerOps.hpp"
#include "PortablePrimerValidation.hpp"

namespace PortableParallelGeometry {
inline bool primerFits(const Layer& layer,
    const std::vector<const std::vector<float>*>& inputs,bool backward){
    if(inputs.empty() || !inputs[0] || inputs[0]->empty() ||
       inputs[0]->size()>INT_MAX || layer.getWeightsSize()!=0 ||
       !PortablePrimerValidation::supportedInputs(inputs))return false;
    using Limits=Mimir::Planning::ExecutionPolicy::Limits;
    const uint64_t n=inputs[0]->size();
    auto fits=[](uint64_t count,uint64_t loops){
        return count && count<=Limits::portable_parallel_invocations && loops &&
            loops<=Limits::portable_reduction_span && count<=Limits::portable_parallel_work/loops;
    };
    auto arenaFits=[](uint64_t elements){return elements<Limits::portable_parallel_elements;};
    if(RuntimePrimer::unaryCode(layer.type_enum)>=0)
        return inputs.size()==1 && arenaFits((backward?3:2)*n) && fits(n,32);
    if(RuntimePrimer::binaryCode(layer.type_enum)>=0){
        if(inputs.size()!=2 || !inputs[1] || inputs[1]->empty() || inputs[1]->size()>INT_MAX)return false;
        const uint64_t other=inputs[1]->size(),count=std::max(n,other);
        if(n!=other && n!=1 && other!=1)return false;
        const uint64_t elements=backward?2*(n+other)+count:n+other+count;
        if(!arenaFits(elements) || !fits(count,1))return false;
        // Only one invocation per scalar input performs this gather. Total
        // work stays linear; its individual reduction must remain bounded.
        return !backward || (n!=1 && other!=1) ||
            (count<=Limits::portable_reduction_span && count<=Limits::portable_parallel_work/3);
    }
    const int reduction=RuntimePrimer::reductionCode(layer.type_enum);
    if(reduction>=0){
        if(inputs.size()!=1)return false;
        CudaOperationGeometry::PrimerReduction g;
        if(!g.valid(layer,n))return false;
        uint64_t loops=g.axis_size;
        if(backward)loops=reduction<=1?1:reduction<=3?2ULL*g.axis_size:g.axis_size;
        return arenaFits(backward?2*n+g.outputs:n+g.outputs) &&
            fits(backward?n:g.outputs,loops);
    }
    if(inputs.size()!=1 || !RuntimePrimer::lowerTriangularType(layer.type_enum))return false;
    CudaOperationGeometry::PrimerTriangle g;
    return g.valid(layer,n) && arenaFits(2*n) && fits(n,1);
}
inline size_t primerOutputSize(const Layer& layer,
    const std::vector<const std::vector<float>*>& inputs){
    if(inputs.empty() || !inputs[0])return 0;
    if(RuntimePrimer::binaryCode(layer.type_enum)>=0)
        return inputs.size()==2 && inputs[1]?std::max(inputs[0]->size(),inputs[1]->size()):0;
    if(RuntimePrimer::reductionCode(layer.type_enum)>=0){
        CudaOperationGeometry::PrimerReduction g;
        return g.valid(layer,inputs[0]->size())?static_cast<size_t>(g.outputs):0;
    }
    return inputs[0]->size();
}
inline bool causalSequenceFits(const Layer& layer,
    const std::vector<const std::vector<float>*>& inputs,bool backward){
    if(inputs.size()!=1 || !inputs[0])return false;
    CudaOperationGeometry::CausalSequence g;
    if(!g.valid(layer,inputs[0]->size()))return false;
    using Limits=Mimir::Planning::ExecutionPolicy::Limits;
    auto fits=[](uint64_t count,uint64_t loops){
        return count && count<=Limits::portable_parallel_invocations && loops &&
            loops<=Limits::portable_reduction_span && count<=Limits::portable_parallel_work/loops;
    };
    const uint64_t elements=backward?3ULL*g.values+2ULL*g.parameters:
        2ULL*g.values+g.parameters;
    return elements<Limits::portable_parallel_elements &&
        fits(g.values,std::min(g.taps,1+(g.steps-1)/g.dilation)) &&
        (!backward || fits(g.parameters,g.steps));
}
inline bool swigluFits(const Layer& layer,
    const std::vector<const std::vector<float>*>& inputs,bool backward){
    if(inputs.size()!=2 || !inputs[0] || !inputs[1] || inputs[0]->empty() ||
       inputs[0]->size()!=inputs[1]->size() || layer.getWeightsSize()!=0)return false;
    using Limits=Mimir::Planning::ExecutionPolicy::Limits;
    const uint64_t count=inputs[0]->size();
    return count<=INT_MAX && count<=Limits::portable_parallel_invocations &&
        count<Limits::portable_parallel_elements/(backward?5:3) &&
        count<=Limits::portable_parallel_work/32;
}
inline bool convolutionFits(const Layer& layer,const std::vector<const std::vector<float>*>& inputs,bool backward){
    if(inputs.size()!=1 || !inputs[0])return false;
    CudaOperationGeometry::ConvND g;if(!g.valid(layer,inputs[0]->size()))return false;
    using Limits=Mimir::Planning::ExecutionPolicy::Limits;
    const uint64_t forward_loops=uint64_t(g.ig)*g.kernel_spatial;
    const uint64_t input_loops=uint64_t(g.og)*g.kernel_spatial;
    const uint64_t weight_loops=uint64_t(g.batch)*g.output_spatial;
    auto fits=[](uint64_t count,uint64_t loops){
        return count && count<=Limits::portable_parallel_invocations && loops &&
            loops<=Limits::portable_reduction_span && count<=Limits::portable_parallel_work/loops;
    };
    const uint64_t elements=backward?2ULL*g.inputSize()+2ULL*g.parameterSize()+g.outputSize():
        uint64_t(g.inputSize())+g.parameterSize()+g.outputSize();
    return elements<=Limits::portable_parallel_elements &&
        (backward?(fits(g.inputSize(),input_loops) && fits(g.parameterSize(),weight_loops)):
                  fits(g.outputSize(),forward_loops));
}
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
