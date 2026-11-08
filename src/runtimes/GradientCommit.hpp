#pragma once
#include "Layers.hpp"
#include "runtimes/FiniteCheck.hpp"
#include <cmath>
#include <vector>

namespace RuntimeGradientCommit {
inline bool finite(const std::vector<float>& values){
    return Mimir::RuntimeChecks::allFinite(values.data(),values.size());
}
inline bool finite(const std::vector<std::vector<float>>& gradients){
    for(const auto& gradient:gradients)if(!finite(gradient))return false;
    return true;
}
// Check every FP32 addition before touching authoritative gradients. Device
// results are deltas; a failed route must be safe to retry on another backend.
inline bool parameters(Layer& layer,const std::vector<float>& weights,const std::vector<float>& bias,
                       bool bias_already_packed=false){
    const size_t count=layer.getWeightsSize();
    if(weights.size()>count || bias.size()>count || !finite(weights) || !finite(bias) ||
       !finite(layer.grad_weights) || !finite(layer.grad_bias))return false;
    const size_t offset=count-bias.size();
    const bool have_weights=layer.grad_weights.size()==count;
    const bool have_bias=layer.grad_bias.size()==bias.size();
    for(size_t i=0;i<count;++i){
        float result=have_weights?layer.grad_weights[i]:0.f;
        if(i<weights.size()){result+=weights[i];if(!std::isfinite(result))return false;}
        if(!bias_already_packed && i>=offset){result+=bias[i-offset];if(!std::isfinite(result))return false;}
    }
    for(size_t i=0;i<bias.size();++i)
        if(!std::isfinite((have_bias?layer.grad_bias[i]:0.f)+bias[i]))return false;
    if(!have_weights)layer.grad_weights.assign(count,0.f);
    for(size_t i=0;i<weights.size();++i)layer.grad_weights[i]+=weights[i];
    if(!bias.empty()){
        if(!have_bias)layer.grad_bias.assign(bias.size(),0.f);
        for(size_t i=0;i<bias.size();++i){layer.grad_bias[i]+=bias[i];if(!bias_already_packed)layer.grad_weights[offset+i]+=bias[i];}
    }
    return true;
}
}
