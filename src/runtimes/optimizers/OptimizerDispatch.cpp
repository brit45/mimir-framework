#include "OptimizerDispatch.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "Model.hpp"
#include "Planning/ExecutionPolicy.hpp"
#include "Planning/ExecutionCalibration.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <map>
#include <tuple>
#include <optional>

namespace {
namespace Policy=Mimir::Planning::ExecutionPolicy;
std::unordered_map<std::string,size_t>& optimizerCounts(){static thread_local std::unordered_map<std::string,size_t> counts;return counts;}
struct Trial {
    RuntimeOptimizerRequest request;
    std::vector<float> w,m,v;
    explicit Trial(const RuntimeOptimizerRequest& r):request(r),w(r.weights,r.weights+r.count) {
        request.weights=w.data();
        if(r.first_moment){m.assign(r.first_moment,r.first_moment+r.count);request.first_moment=m.data();}
        if(r.second_moment){v.assign(r.second_moment,r.second_moment+r.count);request.second_moment=v.data();}
    }
    void reset(const RuntimeOptimizerRequest& r) {
        std::copy_n(r.weights,r.count,w.data());
        if(r.first_moment)std::copy_n(r.first_moment,r.count,m.data());
        if(r.second_moment)std::copy_n(r.second_moment,r.count,v.data());
    }
    void commit(const RuntimeOptimizerRequest& r)const {
        std::copy(w.begin(),w.end(),r.weights);
        if(r.first_moment)std::copy(m.begin(),m.end(),r.first_moment);
        if(r.second_moment)std::copy(v.begin(),v.end(),r.second_moment);
    }
};
bool cpu(const AbstractRuntime* runtime) { return std::strcmp(runtime->name(),"CPU")==0; }
bool update(AbstractRuntime* runtime,const RuntimeOptimizerRequest& r,int workers=1) {
    Policy::CpuScope scope(workers);
    if(cpu(runtime))return runtime->optimizerStep(r);
    // Device failure must never partially commit parameters or moments.
    Trial trial(r);
    if(!runtime->optimizerStep(trial.request))return false;
    trial.commit(r);
    return true;
}
constexpr size_t maxRuntimes=6;
constexpr size_t refreshUpdates=Policy::Limits::calibration_refresh;
using Identity=std::pair<uintptr_t,size_t>;
using Key=std::tuple<int,size_t,unsigned,unsigned>;
struct Route {
    std::array<AbstractRuntime*,maxRuntimes> order{};
    size_t count=0,updates=0;
    int cpu_workers=1;
};
struct Cache {
    uint64_t generation=0;
    std::array<Identity,maxRuntimes> available{};
    std::map<Key,Route> routes;
    std::vector<int> cpu_candidates;
};
// Same algorithm variant, state transfer layout and exact block size share a route.
unsigned variant(const Optimizer& o) {
    if(o.type==OptimizerType::ADAFACTOR)
        return unsigned(o.adafactor_beta1>0.f) | (unsigned(o.adafactor_scale_parameter)<<1) |
            (unsigned(o.adafactor_relative_step)<<2);
    if(o.type==OptimizerType::RADAM)return unsigned(o.step>5);
    return 0;
}
Route calibrate(const RuntimeOptimizerRequest& request,
                const std::array<AbstractRuntime*,maxRuntimes>& candidates,size_t count) {
    Route route;
    if(request.count>Policy::calibrationBudget()/(12*sizeof(float)) || !Mimir::Planning::Calibration::enabled()) {
        // No unmeasured size preference: retain a safe serial CPU reference.
        for(size_t i=0;i<count;++i)if(cpu(candidates[i]))route.order[route.count++]=candidates[i];
        for(size_t i=0;i<count;++i)if(!cpu(candidates[i]))route.order[route.count++]=candidates[i];
        return route;
    }
    struct Timing {double seconds;AbstractRuntime* runtime;int workers;};
    std::vector<Timing> timings;
    Trial reference(request);bool have_reference=false;
    for(size_t i=0;i<count;++i)if(cpu(candidates[i])) {
        have_reference=update(candidates[i],reference.request,1);break;
    }
    const bool ordered=request.optimizer->type==OptimizerType::ADAFACTOR || request.optimizer->type==OptimizerType::LAMB;
    Trial destination(request);
    auto exact=[](const std::vector<float>& a,const std::vector<float>& b) {
        return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0);
    };
    for(size_t index=0;index<count;++index) {
        auto* runtime=candidates[index];std::optional<Timing> best;
        for(int workers:cpu(runtime)?Policy::cpuCandidates(ordered):std::vector<int>{1}) {
            bool ok=true;std::array<double,5> samples{};
            for(int sample=0;sample<6;++sample) {
                destination.reset(request);
                const auto start=std::chrono::steady_clock::now();
                if(!update(runtime,destination.request,workers)){ok=false;break;}
                const auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                if(cpu(runtime) && have_reference && (!exact(destination.w,reference.w) ||
                    !exact(destination.m,reference.m) || !exact(destination.v,reference.v))){ok=false;break;}
                if(sample)samples[size_t(sample-1)]=elapsed;
            }
            if(ok) {
                std::sort(samples.begin(),samples.end());Timing trial{samples[2],runtime,workers};
                if(!best || trial.seconds<best->seconds)best=trial;
            }
        }
        if(best)timings.push_back(*best);
    }
    std::stable_sort(timings.begin(),timings.end(),[](const auto& a,const auto& b){return a.seconds<b.seconds;});
    route.count=timings.size();
    for(size_t i=0;i<timings.size();++i) {
        route.order[i]=timings[i].runtime;
        if(cpu(timings[i].runtime))route.cpu_workers=timings[i].workers;
    }
    if(route.count) {
        std::cerr<<"[optimizer] backend="<<route.order[0]->name()<<" type="<<int(request.optimizer->type)
            <<" parameters="<<request.count<<" selection=median_host_roundtrip samples=5"
            <<" cpu_workers="<<route.cpu_workers<<" refresh_updates="<<refreshUpdates<<" candidates_seconds=";
        for(size_t i=0;i<timings.size();++i)std::cerr<<(i?",":"")<<timings[i].runtime->name()<<":"<<timings[i].seconds;
        std::cerr<<'\n';
    }
    return route;
}
}

bool RuntimeRouter::dispatchOptimizer(const RuntimeOptimizerRequest& request,AbstractRuntime** selected) const {
    if(selected)*selected=nullptr;
    if(!optimizerRequestValid(request))return false;
    ensureActivatedAndComposed();
    std::array<AbstractRuntime*,maxRuntimes> candidates{};
    std::array<Identity,maxRuntimes> identities{};
    size_t count=0;
    for(auto* runtime:runtime_priority_) {
        if(!runtime || !runtime->isInitialized() || runtime->config().disabled ||
           !runtime->supportsOptimizer() || (!cpu(runtime) && !optimizerDeviceSizeFits(request)))continue;
        if(std::find(candidates.begin(),candidates.begin()+count,runtime)!=candidates.begin()+count)continue;
        candidates[count]=runtime;identities[count++]={reinterpret_cast<uintptr_t>(runtime),Mimir::Planning::Calibration::configuration(*runtime)};
    }
    static thread_local Cache cache;
    const auto cpu_candidates=Policy::cpuCandidates();
    if(cache.generation!=optimizer_routing_generation_ || cache.available!=identities || cache.cpu_candidates!=cpu_candidates) {
        cache.routes.clear();cache.generation=optimizer_routing_generation_;cache.available=identities;cache.cpu_candidates=cpu_candidates;
    }
    if(!count)return false;
    const bool ordered=request.optimizer->type==OptimizerType::ADAFACTOR || request.optimizer->type==OptimizerType::LAMB;
    if(count==1 && (!cpu(candidates[0]) || Policy::cpuCandidates(ordered).size()==1)) {
        if(!update(candidates[0],request))return false;
        ++optimizerCounts()[candidates[0]->name()];
        if(selected)*selected=candidates[0];
        return true;
    }
    const unsigned moments=unsigned(request.first_moment!=nullptr) | (unsigned(request.second_moment!=nullptr)<<1);
    const Key key{int(request.optimizer->type),request.count,moments,variant(*request.optimizer)};
    auto found=cache.routes.find(key);
    if(found==cache.routes.end() || found->second.updates>=refreshUpdates) {
        Route route=calibrate(request,candidates,count);
        if(!route.count){cache.routes.erase(key);return false;}
        // Dynamic block shapes must not grow a thread's cache indefinitely.
        if(found==cache.routes.end() && cache.routes.size()>=512)cache.routes.erase(cache.routes.begin());
        found=cache.routes.insert_or_assign(key,route).first;
    }
    auto& route=found->second;
    bool failed=false;
    for(size_t i=0;i<route.count;++i) {
        auto* runtime=route.order[i];
        if(!update(runtime,request,cpu(runtime)?route.cpu_workers:1)){failed=true;continue;}
        ++optimizerCounts()[runtime->name()];
        if(selected)*selected=runtime;
        // A transient failure must not permanently pin the model to its fallback.
        route.updates=failed?refreshUpdates:route.updates+1;
        return true;
    }
    cache.routes.erase(found);
    return false;
}

const std::unordered_map<std::string,size_t>& RuntimeRouter::optimizerDispatchCounts() const { return optimizerCounts(); }
