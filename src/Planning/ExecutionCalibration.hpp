#pragma once
#include "runtimes/AbstractRuntime.hpp"
#include "Planning/RuntimeRoute.hpp"
#include <functional>
#include <unordered_map>

namespace Mimir::Planning::Calibration {
struct Measurement {
    AbstractRuntime* runtime=nullptr;size_t configuration=0;double seconds=0;
    int cpu_workers=1;ExecutionPolicy::CpuAlgorithm cpu_algorithm=ExecutionPolicy::CpuAlgorithm::Reference;
};
struct Ranking {
    size_t key=0;bool calibrated=false,ordered=false;RuntimeRoute route;
    int cpu_workers=1;ExecutionPolicy::CpuAlgorithm cpu_algorithm=ExecutionPolicy::CpuAlgorithm::Reference;
};
using Probe=std::function<bool(AbstractRuntime&,Layer&,std::vector<std::vector<float>>&)>;
extern thread_local std::unordered_map<const Layer*,std::vector<Measurement>> forward_profiles;
extern thread_local std::unordered_map<const Layer*,std::vector<Measurement>> backward_profiles;
bool enabled();
size_t configuration(const AbstractRuntime& runtime);
void invalidate(size_t key);
size_t signature(const Layer&,bool backward,bool training,uint64_t generation,
    const std::vector<const std::vector<float>*>& inputs,const std::vector<const std::vector<float>*>& gradients);
Ranking rank(const RuntimeRoute& candidates,const Layer& layer,
    const std::vector<const std::vector<float>*>& inputs,const std::vector<const std::vector<float>*>& gradients,
    bool backward,bool training,uint64_t generation,const Probe& execute);
}
