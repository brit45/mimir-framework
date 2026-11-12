#pragma once
#include "LatentSpaceProjection.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace mimir::viz {
// A symbolic cortex layout, never a learned spatial embedding. X retains graph
// depth; the two folded Y/Z lobes distribute known graph units deterministically.
// Positions depend on identity, not weights, camera or capture cadence.
struct OrganicNetworkScene {
    std::vector<std::array<float,3>> positions;
    std::vector<float> plasticity;
};
inline uint32_t organicIdentity(const LatentSpacePoint& point) {
    uint32_t hash=2166136261u;
    for (unsigned char byte:point.label) hash=(hash^byte)*16777619u;
    return hash;
}
inline std::array<float,3> organicNetworkPosition(const LatentSpacePoint& point) {
    if (point.neuron_is_operation || point.live_weight_coefficient != std::numeric_limits<size_t>::max())
        return point.position;
    const uint32_t seed=organicIdentity(point);
    const float angle=float(seed&65535u)*(6.28318530718f/65536.f);
    const float shell=.45f+.55f*float((seed>>16)&65535u)/65535.f;
    const float fold=1.f+.12f*std::sin(5.f*angle+float(point.neuron_node_index%17));
    const float hemisphere=(point.neuron_unit_index%2) ? 1.f : -1.f;
    return {{point.position[0], point.position[1]*.35f+std::sin(angle)*shell*fold,
             hemisphere*(.35f+.85f*std::abs(std::cos(angle))*shell*fold)}};
}
inline OrganicNetworkScene buildOrganicNetworkScene(const LatentSpaceProjection& projection) {
    OrganicNetworkScene result;
    result.positions.reserve(projection.points.size());
    result.plasticity.assign(projection.points.size(),0.f);
    const bool topology=projection.neuron_mode || projection.live_weight_mode;
    for (const auto& point:projection.points)
        result.positions.push_back(topology ? organicNetworkPosition(point) : point.position);
    // Brightness means measured parameter change, not firing or activation.
    double max_delta=0.;
    for (const auto& c:projection.live_weight_coefficients)
        if(c.has_delta && std::isfinite(c.value) && std::isfinite(c.delta))
            max_delta=std::max(max_delta,std::abs(c.delta));
    if(!projection.live_weight_mode || !max_delta)return result;
    const auto amplitude=[&](size_t coefficient) {
        if(coefficient>=projection.live_weight_coefficients.size())return 0.f;
        const auto& c=projection.live_weight_coefficients[coefficient];
        return c.has_delta && std::isfinite(c.value) && std::isfinite(c.delta)
            ? float(std::sqrt(std::abs(c.delta)/max_delta)) : 0.f;
    };
    for(const auto& link:projection.live_weight_links) {
        if(link.source>=result.plasticity.size() || link.target>=result.plasticity.size())continue;
        const float value=amplitude(link.coefficient_index);
        result.plasticity[link.source]=std::max(result.plasticity[link.source],value);
        result.plasticity[link.target]=std::max(result.plasticity[link.target],value);
    }
    for(size_t i=0;i<projection.points.size();++i)
        result.plasticity[i]=std::max(result.plasticity[i],amplitude(projection.points[i].live_weight_coefficient));
    return result;
}
// Decorative dendrites: these do not add graph edges or learned connections.
inline std::array<float,3> organicDendriteTip(const LatentSpacePoint& point,
                                           const std::array<float,3>& center,unsigned branch) {
    const float phase=float(organicIdentity(point)%4096u)*.0015339808f+float(branch)*2.39996323f;
    const float reach=.045f+.025f*float((organicIdentity(point)>>(branch%16))&3u);
    return {{center[0]+std::cos(phase)*reach,center[1]+std::sin(phase)*reach,
             center[2]+std::sin(phase*1.7f)*reach}};
}
} // namespace mimir::viz
