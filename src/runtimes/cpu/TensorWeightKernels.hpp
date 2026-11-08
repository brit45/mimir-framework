#pragma once
#include "tensors.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <utility>

namespace CpuTensorWeights {
inline bool compute(const std::vector<tensor>& tensors,std::vector<uint16_t>& output) {
    // Validate the consumed inputs before evaluating trigonometric functions or
    // quantizing. Tensor payloads and metadata are never modified here.
    for(const auto& t:tensors)
        if(!std::isfinite(t.Pos.X) || !std::isfinite(t.Pos.Y) ||
           !std::isfinite(t.Pos.Z) || !std::isfinite(t.Pos.W))return false;
    std::vector<uint16_t> result(tensors.size());
    for(size_t i=0;i<tensors.size();++i) {
        const tensor& t=tensors[i];
        // Preserve the original CPU float formula and evaluation order.
        const float r2=t.Pos.X*t.Pos.X+t.Pos.Y*t.Pos.Y+t.Pos.Z*t.Pos.Z;
        const float r=std::sqrt(r2);
        const float spatial_nl=std::tanh(std::min(std::max(r*0.5f,-3.0f),3.0f));
        const float v01=static_cast<float>(t.Value)*(1.0f/255.0f);
        const float l01=static_cast<float>(t.Length)*(1.0f/1024.0f);
        const float vel_nl=std::tanh(v01*2.0f-1.0f);
        const float den_nl=std::tanh(std::min(std::max(l01,0.0f),1.0f)*2.0f-1.0f);
        const float w=t.Pos.W;
        const float w_high=w*0.50f,w_mid=w*0.10f,w_low=w*0.02f;
        const float s1=std::sin(w_high),c1=std::cos(w_high);
        const float s2=std::sin(w_mid),c2=std::cos(w_mid);
        const float s3=std::sin(w_low),c3=std::cos(w_low);
        const float ctx_core=std::tanh(w*0.05f);
        const float ctx_feat=0.50f*s1+0.35f*c1+0.30f*s2+0.20f*c2+0.15f*s3+0.10f*c3;
        const float ctx_sum=std::min(std::max(ctx_core+0.6f*ctx_feat,-1.0f),1.0f);
        const float gate=1.0f/(1.0f+std::exp(-2.2f*ctx_sum));
        const float w_spatial=(1.0f-gate)*0.95f+0.05f;
        const float w_dyn=gate*0.90f;
        const float combined=w_spatial*spatial_nl+w_dyn*(vel_nl-0.7f*den_nl);
        float s=1.0f/(1.0f+std::exp(-2.0f*combined));
        s=0.98f*s+0.01f;
        if(!std::isfinite(s))return false;
        int iw=static_cast<int>(std::floor(s*65535.0f+0.5f));
        iw=std::max(0,std::min(iw,65535));
        result[i]=static_cast<uint16_t>(iw);
    }
    output=std::move(result);
    return true;
}
}
