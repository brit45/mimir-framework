#pragma once
#include "Layers.hpp"
#include "Parallel.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <array>
#include <type_traits>

namespace Mimir::CpuFusion {
inline bool supports(LayerType t) {
    switch(t) {
    case LayerType::Identity: case LayerType::ReLU: case LayerType::LeakyReLU:
    case LayerType::GELU: case LayerType::SiLU: case LayerType::Tanh:
    case LayerType::Sigmoid: case LayerType::Softplus: case LayerType::Mish:
    case LayerType::HardSigmoid: case LayerType::HardSwish: return true;
    default: return false;
    }
}
template<LayerType Type> inline float forward(float x, const Layer& l) {
    switch(Type) {
    case LayerType::Identity: return x;
    case LayerType::ReLU: return std::max(0.f,x);
    case LayerType::LeakyReLU: return x>0.f?x:(l.leaky_relu_alpha>0?l.leaky_relu_alpha:.01f)*x;
    case LayerType::GELU: {
        const float c=std::sqrt(2.f/3.14159265359f);
        const float cdf=.5f*(1.f+std::tanh(c*(x+.044715f*x*x*x)));
        return x*cdf;
    }
    case LayerType::SiLU: {const float s=1.f/(1.f+std::exp(-x));return x*s;}
    case LayerType::Tanh: return std::tanh(x);
    case LayerType::Sigmoid: return 1.f/(1.f+std::exp(-x));
    case LayerType::Softplus: return x>20.f?x:std::log1p(std::exp(x));
    case LayerType::Mish: {const float sp=x>20.f?x:std::log1p(std::exp(x));return x*std::tanh(sp);}
    case LayerType::HardSigmoid: return std::max(0.f,std::min(1.f,(x+3.f)/6.f));
    case LayerType::HardSwish: {const float s=std::max(0.f,std::min(1.f,(x+3.f)/6.f));return x*s;}
    default: return x;
    }
}
// Keep the original derivative expressions and multiplication order, including
// multiplication by dy at each logical edge (never multiply derivatives first).
template<LayerType Type> inline float backward(float x,float g,const Layer& l) {
    switch(Type) {
    case LayerType::Identity: return g;
    case LayerType::ReLU: return g*(x>0.f?1.f:0.f);
    case LayerType::LeakyReLU: return g*(x>0.f?1.f:(l.leaky_relu_alpha>0?l.leaky_relu_alpha:.01f));
    case LayerType::Sigmoid: {const float s=1.f/(1.f+std::exp(-x));return g*s*(1.f-s);}
    case LayerType::Tanh: {const float t=std::tanh(x);return g*(1.f-t*t);}
    case LayerType::SiLU: {const float s=1.f/(1.f+std::exp(-x));return g*(s+x*s*(1.f-s));}
    case LayerType::GELU: {
        const float c=std::sqrt(2.f/3.14159265358979323846f),x2=x*x,x3=x2*x;
        const float u=c*(x+.044715f*x3),t=std::tanh(u),sech2=1.f-t*t;
        const float du=c*(1.f+3.f*.044715f*x2);
        const float d=.5f*(1.f+t)+.5f*x*sech2*du;
        return g*d;
    }
    case LayerType::Softplus: return g*(1.f/(1.f+std::exp(-x)));
    case LayerType::Mish: {
        const float sp=x>20.f?x:std::log1p(std::exp(x)),t=std::tanh(sp),s=1.f/(1.f+std::exp(-x));
        const float d=t+x*(1.f-t*t)*s;return g*d;
    }
    case LayerType::HardSigmoid: return g*((x>-3.f&&x<3.f)?1.f/6.f:0.f);
    case LayerType::HardSwish: {
        float d;if(x<=-3.f)d=0;else if(x>=3.f)d=1;else d=x/3.f+.5f;return g*d;
    }
    default: return g;
    }
}

// Resolve the operation once per cache tile, outside the SIMD element loop.
template<class F> inline void visit(LayerType type,const F& f) {
#define MIMIR_POINTWISE_CASE(T) case LayerType::T: f(std::integral_constant<LayerType,LayerType::T>{});break
    switch(type) {
    MIMIR_POINTWISE_CASE(Identity); MIMIR_POINTWISE_CASE(ReLU); MIMIR_POINTWISE_CASE(LeakyReLU);
    MIMIR_POINTWISE_CASE(GELU); MIMIR_POINTWISE_CASE(SiLU); MIMIR_POINTWISE_CASE(Tanh);
    MIMIR_POINTWISE_CASE(Sigmoid); MIMIR_POINTWISE_CASE(Softplus); MIMIR_POINTWISE_CASE(Mish);
    MIMIR_POINTWISE_CASE(HardSigmoid); MIMIR_POINTWISE_CASE(HardSwish);
    default: break;
    }
#undef MIMIR_POINTWISE_CASE
}
inline constexpr size_t tile_size=Mimir::Planning::ExecutionPolicy::Limits::pointwise_tile_elements;
inline void forwardChain(const std::vector<const Layer*>& chain,const std::vector<float>& input,
                         std::vector<std::vector<float>>& outputs,bool retain) {
    outputs.resize(retain?chain.size():1);
    for(auto& output:outputs)output.resize(input.size());
    const size_t tiles=(input.size()+tile_size-1)/tile_size;
    RuntimeParallel::forEach(tiles,RuntimeParallel::workers(input.size()),[&](size_t tile){
        const size_t begin=tile*tile_size,count=std::min(tile_size,input.size()-begin);
        std::array<float,tile_size> scratch;
        const float* source=input.data()+begin;
        for(size_t j=0;j<chain.size();) {
            // Lower adjacent operations into one element loop. Keep the
            // intermediate in F32, and expose it when backward needs it.
            if(j+1<chain.size()) {
                const Layer& first=*chain[j];
                const Layer& second=*chain[j+1];
                float* intermediate=retain?outputs[j].data()+begin:nullptr;
                float* target=retain?outputs[j+1].data()+begin:
                    (j+2==chain.size()?outputs[0].data()+begin:scratch.data());
                visit(first.type_enum,[&](auto first_tag){
                    visit(second.type_enum,[&](auto second_tag){
                        constexpr auto a=decltype(first_tag)::value;
                        constexpr auto b=decltype(second_tag)::value;
                        #pragma omp simd
                        for(size_t i=0;i<count;++i) {
                            const float value=forward<a>(source[i],first);
                            if(intermediate)intermediate[i]=value;
                            target[i]=forward<b>(value,second);
                        }
                    });
                });
                source=target;j+=2;continue;
            }
            float* target=retain?outputs[j].data()+begin:
                (j+1==chain.size()?outputs[0].data()+begin:scratch.data());
            const Layer& layer=*chain[j];
            visit(layer.type_enum,[&](auto tag){
                constexpr auto type=decltype(tag)::value;
                #pragma omp simd
                for(size_t i=0;i<count;++i)target[i]=forward<type>(source[i],layer);
            });
            // A materialized tile retains the reference F32 rounding boundary.
            source=target;
            ++j;
        }
    });
}
inline void backwardChain(const std::vector<const Layer*>& chain,
    const std::vector<const std::vector<float>*>& inputs,const std::vector<float>& grad,std::vector<float>& result) {
    result.resize(grad.size());
    const size_t tiles=(grad.size()+tile_size-1)/tile_size;
    RuntimeParallel::forEach(tiles,RuntimeParallel::workers(grad.size()),[&](size_t tile){
        const size_t begin=tile*tile_size,count=std::min(tile_size,grad.size()-begin);
        const float* source=grad.data()+begin;
        float* target=result.data()+begin;
        for(size_t j=chain.size();j>0;) {
            if(j>=2) {
                const Layer& last=*chain[j-1];
                const Layer& previous=*chain[j-2];
                const float* last_x=inputs[j-1]?inputs[j-1]->data()+begin:nullptr;
                const float* previous_x=inputs[j-2]?inputs[j-2]->data()+begin:nullptr;
                visit(last.type_enum,[&](auto last_tag){
                    visit(previous.type_enum,[&](auto previous_tag){
                        constexpr auto a=decltype(last_tag)::value;
                        constexpr auto b=decltype(previous_tag)::value;
                        #pragma omp simd
                        for(size_t i=0;i<count;++i) {
                            const float value=backward<a>(last_x?last_x[i]:0.f,source[i],last);
                            target[i]=backward<b>(previous_x?previous_x[i]:0.f,value,previous);
                        }
                    });
                });
                source=target;j-=2;continue;
            }
            --j;
            const float* x=inputs[j]?inputs[j]->data()+begin:nullptr;
            const Layer& layer=*chain[j];
            visit(layer.type_enum,[&](auto tag){
                constexpr auto type=decltype(tag)::value;
                #pragma omp simd
                for(size_t i=0;i<count;++i)target[i]=backward<type>(x?x[i]:0.f,source[i],layer);
            });
            source=target;
        }
    });
}
}
