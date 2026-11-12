#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
namespace mimir::viz {
struct HeatspotSample { float x=0.f,y=0.f,intensity=0.f; };
struct HeatspotMap {
    static constexpr unsigned width=64,height=48;
    std::array<float,width*height> values{};
};
// Projected density of measured, normalized |delta weight|. No activity is
// invented for missing baselines, zero changes or non-finite values.
inline HeatspotMap buildHeatspots(const std::vector<HeatspotSample>& samples) {
    HeatspotMap result;
    static const auto kernel=[] {
        std::array<float,169> k{};
        for(int y=-6;y<=6;++y)for(int x=-6;x<=6;++x)
            k[(y+6)*13+x+6]=std::exp(-float(x*x+y*y)/8.f);
        return k;
    }();
    for(const auto& sample:samples) {
        if(!std::isfinite(sample.x)||!std::isfinite(sample.y)||!std::isfinite(sample.intensity)||sample.intensity<=0.f ||
           sample.x<-.15f || sample.x>1.15f || sample.y<-.15f || sample.y>1.15f)continue;
        const int cx=int(std::lround(sample.x*(HeatspotMap::width-1))),cy=int(std::lround(sample.y*(HeatspotMap::height-1)));
        for(int y=std::max(0,cy-6);y<=std::min(int(HeatspotMap::height)-1,cy+6);++y)
            for(int x=std::max(0,cx-6);x<=std::min(int(HeatspotMap::width)-1,cx+6);++x)
                result.values[size_t(y)*HeatspotMap::width+x]=std::min(1.f,
                    result.values[size_t(y)*HeatspotMap::width+x]+std::min(1.f,sample.intensity)*kernel[(y-cy+6)*13+x-cx+6]);
    }
    return result;
}
inline std::array<unsigned char,4> heatspotColor(float value) {
    if(!std::isfinite(value)||value<=.005f)return {{0,0,0,0}};
    const float v=std::clamp(value,0.f,1.f);
    return {{static_cast<unsigned char>(255.f*std::min(1.f,v*3.f)),
             static_cast<unsigned char>(255.f*std::clamp(2.f-v*2.f,0.f,1.f)),
             static_cast<unsigned char>(255.f*std::max(0.f,1.f-v*3.f)),
             static_cast<unsigned char>(145.f*std::sqrt(v))}};
}
}
