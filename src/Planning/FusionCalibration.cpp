#include "Planning/FusionCalibration.hpp"
#include "Planning/ExecutionCalibration.hpp"
#include "Layers.hpp"
#include "MemoryGuard.hpp"
#include "runtimes/FiniteCheck.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace Mimir::Planning::FusionCalibration {
namespace {
struct Entry {Decision decision;size_t uses=0;};
thread_local std::unordered_map<size_t,Entry> cache;
thread_local std::unordered_map<const Layer*,Decision> forward_profiles,backward_profiles;
void mix(size_t& h,size_t value){h^=value+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2);}
bool finite(const Buffers& outputs) {
    for(const auto& value:outputs)if(!RuntimeChecks::allFinite(value.data(),value.size()))return false;
    return true;
}
bool exact(const Buffers& a,const Buffers& b) {
    if(a.size()!=b.size())return false;
    for(size_t i=0;i<a.size();++i)
        if(a[i].size()!=b[i].size() || (!a[i].empty() &&
            std::memcmp(a[i].data(),b[i].data(),a[i].size()*sizeof(float))))return false;
    return true;
}
}
bool pointwiseCandidate(LayerType type) {
    switch(type) {
    case LayerType::Identity:case LayerType::ReLU:case LayerType::LeakyReLU:
    case LayerType::GELU:case LayerType::SiLU:case LayerType::Tanh:
    case LayerType::Sigmoid:case LayerType::Softplus:case LayerType::Mish:
    case LayerType::HardSigmoid:case LayerType::HardSwish:return true;
    default:return false;
    }
}
void invalidate(size_t key){
    cache.erase(key);
    for(auto* profiles:{&forward_profiles,&backward_profiles})for(auto& entry:*profiles)
        if(entry.second.key==key) {
            entry.second.fused=false;entry.second.runtime=nullptr;entry.second.cache_hit=false;
            entry.second.selected_runtime="unfused (routed)";
            entry.second.reason="native fused execution failed; recalibration required";
        }
}
void clearProfiles(){forward_profiles.clear();backward_profiles.clear();}
const Decision* profile(const Layer* first,bool backward) {
    const auto& profiles=backward?backward_profiles:forward_profiles;
    const auto found=profiles.find(first);return found==profiles.end()?nullptr:&found->second;
}
namespace {
bool prepareKey(const Request& request,const std::vector<Candidate>& candidates,Decision& result) {
    if(request.layers.empty() || !request.input || candidates.size()<2) {
        result.reason="no native fused candidate";return false;
    }
    if(!Calibration::enabled()) {result.reason="latency calibration disabled";return false;}
    size_t key=0;mix(key,request.generation);mix(key,request.backward);mix(key,request.training);
    mix(key,request.retain);mix(key,size_t(request.dtype));mix(key,request.input->size());
    for(const auto* layer:request.layers) {
        if(!layer || !pointwiseCandidate(layer->type_enum) || layer->getWeightsSize()!=0 || layer->inputs.size()>1) {
            result.reason="non replayable program";return false;
        }
        // Complete scalar-program semantics; unrelated convolution/attention
        // fields must not make a cached pointwise dispatch expensive.
        mix(key,size_t(layer->type_enum));mix(key,size_t(layer->dtype));
        mix(key,size_t(layer->accumulation_dtype));
        mix(key,std::hash<float>{}(layer->leaky_relu_alpha));
    }
#ifdef _OPENMP
    mix(key,size_t(omp_get_max_threads()));mix(key,size_t(omp_in_parallel()));
#endif
    for(auto* runtime:request.available) {
        mix(key,reinterpret_cast<size_t>(runtime));mix(key,runtime->isInitialized());
        mix(key,Calibration::configuration(*runtime));
    }
    for(size_t i=1;i<candidates.size();++i) {
        mix(key,reinterpret_cast<size_t>(candidates[i].runtime));mix(key,size_t(candidates[i].cpu_workers));
    }
    result.key=key;return true;
}
bool cachedDecision(const Request& request,size_t key,Decision& result) {
    auto found=cache.find(key);
    if(found==cache.end() || found->second.uses>=ExecutionPolicy::Limits::calibration_refresh)return false;
    ++found->second.uses;
    result=found->second.decision;result.cache_hit=true;
    auto& published=(request.backward?backward_profiles:forward_profiles)[request.layers.front()];
    // A warm decision has identical diagnostics. Republish only on a new key
    // or the first hit after a measurement/profile reset.
    if(published.key!=key || !published.cache_hit)published=result;
    return true;
}
}
bool lookup(const Request& request,const std::vector<Candidate>& candidates,Decision& result) {
    Decision key;
    return prepareKey(request,candidates,key) && cachedDecision(request,key.key,result);
}
Decision select(const Request& request,const std::vector<Candidate>& candidates) {
    Decision result;
    auto publish=[&]() {
        if(!request.layers.empty())(request.backward?backward_profiles:forward_profiles)[request.layers.front()]=result;
        return result;
    };
    if(!prepareKey(request,candidates,result))return publish();
    const size_t key=result.key;
    if(cachedDecision(request,key,result))return result;
    // Bound calibration storage independently of the model's real execution.
    const size_t count=request.input->size(),stages=request.layers.size();
    const size_t budget=ExecutionPolicy::calibrationBudget();
    const size_t bytes_per_element=sizeof(float)*8;
    if(stages>std::numeric_limits<size_t>::max()/(2*sizeof(Layer)) ||
        stages*2*sizeof(Layer)>budget ||
        count>(budget-stages*2*sizeof(Layer))/std::max(size_t(1),stages*bytes_per_element)) {
        result.reason="calibration memory budget";return publish();
    }
    const size_t bytes=count*stages*bytes_per_element+stages*2*sizeof(Layer);
    auto& guard=MemoryGuard::instance();const auto limit=guard.getLimit(),used=guard.getCurrentBytes();
    if(guard.isBlocked() || guard.isFrozen() || (limit && (used>=limit || bytes>limit-used))) {
        result.reason="calibration memory guard";return publish();
    }
    Buffers reference;
    auto probe=[&](const Candidate& candidate,double& seconds,bool compare) {
        std::vector<double> samples;
        for(int i=0;i<6;++i) {
            Buffers output;
            ExecutionPolicy::CpuScope cpu_scope(candidate.cpu_workers);
            const auto start=std::chrono::steady_clock::now();
            bool valid=false;try{valid=candidate.execute(output);}catch(...){return false;}
            const auto duration=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            if(!valid || output.empty() || !finite(output))return false;
            if(!compare && i==0)reference=output;
            if(!exact(output,reference))return false;
            if(i)samples.push_back(duration);
        }
        std::sort(samples.begin(),samples.end());seconds=samples[samples.size()/2];return true;
    };
    if(!probe(candidates[0],result.unfused_seconds,false)) {
        result.reason="unfused reference invalid or non deterministic";return publish();
    }
    result.measured=true;
    result.alternatives.push_back({"unfused (routed)",result.unfused_seconds,true});
    for(size_t i=1;i<candidates.size();++i) {
        double seconds=0;
        if(!candidates[i].runtime || !candidates[i].runtime->isInitialized())continue;
        const bool valid=probe(candidates[i],seconds,true);
        result.alternatives.push_back({candidates[i].runtime->name(),valid?seconds:0,valid,candidates[i].cpu_workers});
        if(!valid)continue;
        result.exact=true;
        if(seconds<result.fused_seconds){result.fused_seconds=seconds;result.runtime=candidates[i].runtime;result.cpu_workers=candidates[i].cpu_workers;}
    }
    // Retain the reference path for ties and timing noise, rather than enabling
    // a fusion whose measured gain does not clear a small safety margin.
    result.fused=result.exact && result.fused_seconds<result.unfused_seconds*.98;
    result.reason=!result.exact?"native fused execution or parity rejected":
        result.fused?"native fusion faster and bit exact":"unfused path faster or tied";
    result.selected_runtime=result.fused?result.runtime->name():"unfused (routed)";
    if(!result.fused)result.runtime=nullptr;
    if(cache.size()>=ExecutionPolicy::Limits::calibration_cache_entries){cache.clear();forward_profiles.clear();backward_profiles.clear();}
    // Several layers can share a semantic key. Refresh their diagnostics on
    // the next hit too, even though only this caller triggered recalibration.
    if(cache.find(key)!=cache.end())
        for(auto* profiles:{&forward_profiles,&backward_profiles})for(auto& entry:*profiles)
            if(entry.second.key==key)entry.second.cache_hit=false;
    cache[key]={result,1};return publish();
}
}
