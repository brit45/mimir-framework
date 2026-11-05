#include "Planning/FusionCalibration.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "Layers.hpp"
#include <chrono>
#include <thread>
#include <iostream>
#include <stdexcept>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif
using Buffers=std::vector<std::vector<float>>;
namespace Fusion=Mimir::Planning::FusionCalibration;
static void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static void env(const char* name,const char* value){
#ifdef _WIN32
    _putenv_s(name,value);
#else
    setenv(name,value,1);
#endif
}
struct Simulated : AbstractRuntime {
    std::string id;bool online=true,bad=false,fused=true;int ordinary_us=150,fused_us=0,probes=0,fail_after=-1;
    explicit Simulated(std::string name):id(std::move(name)){}
    const char* name() const override{return id.c_str();}
    bool initialize(const RuntimeConfig& cfg) override{config_=cfg;return true;}
    void shutdown() override{online=false;}
    bool isInitialized() const override{return online;}
    bool linearForward(const float*,const float*,const float*,float*,int,int,int) override{return false;}
    bool supportsForwardLayerType(LayerType t) const override{return t==LayerType::Identity;}
    bool supportsBackwardLayerType(LayerType t) const override{return t==LayerType::Identity;}
    RuntimeCapabilityLevel queryForwardCapability(LayerType t) const override{return supportsForwardLayerType(t)?RuntimeCapabilityLevel::Native:RuntimeCapabilityLevel::Unsupported;}
    RuntimeCapabilityLevel queryBackwardCapability(LayerType t) const override{return queryForwardCapability(t);}
    bool forwardLayer(const std::vector<const std::vector<float>*>& x,Buffers& y,const Layer&,bool) override {
        if(ordinary_us)std::this_thread::sleep_for(std::chrono::microseconds(ordinary_us));
        y={*x[0]};return true;
    }
    bool backwardLayer(const std::vector<const std::vector<float>*>&,const std::vector<const std::vector<float>*>& g,
        Buffers& dx,Layer& disposable,bool) override {
        if(ordinary_us)std::this_thread::sleep_for(std::chrono::microseconds(ordinary_us));
        disposable.grad_weights={999};dx={*g[0]};return true;
    }
    RuntimeCapabilityLevel queryPointwiseFusionCapability(const std::vector<const Layer*>&,bool,bool) const override {
        return fused?RuntimeCapabilityLevel::NativeOptimized:RuntimeCapabilityLevel::Unsupported;
    }
    bool pointwiseFusionForward(const std::vector<const Layer*>& chain,const std::vector<float>& x,
        Buffers& y,bool retain,bool) override {
        ++probes;if(fail_after>=0 && probes>fail_after)return false;
        if(fused_us)std::this_thread::sleep_for(std::chrono::microseconds(fused_us));
        y.assign(retain?chain.size():1,x);
        if(bad && !x.empty())y.back()[0]=std::nextafter(y.back()[0],100.f);
        return true;
    }
    bool pointwiseFusionBackward(const std::vector<const Layer*>&,const std::vector<const std::vector<float>*>&,
        const std::vector<float>& g,std::vector<float>& dx) override {
        ++probes;if(fused_us)std::this_thread::sleep_for(std::chrono::microseconds(fused_us));
        dx=g;if(bad && !dx.empty())dx[0]=std::nextafter(dx[0],100.f);return true;
    }
};
int main(){try{
    env("MIMIR_RUNTIME_AUTOTUNE","1");env("MIMIR_PLANNER_CALIBRATION_MAX_BYTES","67108864");
    Simulated cpu("CPU"),device("CUDA");cpu.fused_us=700;device.fused_us=0;
    auto& router=RuntimeRouter::instance();router.setActivators({},{},{},{},{});
    auto reset=[&]{router.setRuntimes(nullptr,&device,nullptr,nullptr,&cpu);};reset();
    Layer a("first","Identity",0),b("second","Identity",0);a.inputs={"x"};b.inputs={"x"};
    std::vector<const Layer*> chain{&a,&b};std::vector<float> x{1,2,3},dx;Buffers output;
    AbstractRuntime* selected=nullptr;
    check(router.dispatchPointwiseForward(chain,x,output,true,&selected),"fast native fusion refused");
    check(selected==&device && output==Buffers({x,x}),"planner did not select measured faster device");
    const auto* decision=Fusion::profile(&a,false);
    check(decision && decision->fused && decision->measured && decision->exact,"missing planner proof");
    check(decision->fused_seconds<decision->unfused_seconds && decision->alternatives.size()==3,"missing latency comparison");
    const int warm_probes=device.probes;
    check(router.dispatchPointwiseForward(chain,x,output,true,&selected),"cached fusion refused");
    check(Fusion::profile(&a,false)->cache_hit && device.probes==warm_probes+1,"fusion measured again on cache hit");
    device.fail_after=device.probes;output={{42}};
    check(!router.dispatchPointwiseForward(chain,x,output,true) && output==Buffers{{42}},"native execution failure mutated output");
    check(!Fusion::profile(&a,false)->fused,"failed execution still reported as selected");
    device.fail_after=-1;reset();
    a.grad_weights={7};b.grad_weights={8};
    check(router.dispatchPointwiseBackward(chain,{&x,&x},x,dx,&selected) && selected==&device && dx==x,"backward native selection failed");
    check(a.grad_weights==std::vector<float>{7} && b.grad_weights==std::vector<float>{8},"calibration mutated model gradients");
    // A one-ULP error must be rejected even when it is the fastest candidate.
    device.bad=true;reset();
    router.dispatchPointwiseForward(chain,x,output,true,&selected);
    decision=Fusion::profile(&a,false);
    bool device_rejected=false;
    if(decision)for(const auto& alternative:decision->alternatives)
        if(alternative.runtime=="CUDA")device_rejected=!alternative.exact;
    check(device_rejected && selected!=&device,"inexact fast fusion accepted");
    device.bad=false;cpu.ordinary_us=0;device.ordinary_us=0;cpu.fused_us=device.fused_us=1500;reset();
    output={{42}};
    check(router.dispatchPointwiseForward(chain,x,output,true,&selected) && !selected && output==Buffers({x,x}),"planner did not execute faster ordinary chain");
    decision=Fusion::profile(&a,false);check(decision && decision->measured && !decision->fused,"unfused decision missing");
    const auto key=decision->key;
    const int slow_probes=device.probes;
    check(router.dispatchPointwiseForward(chain,x,output,true) && !Fusion::profile(&a,false)->fused,"cached ordinary program not executed");
    check(device.probes==slow_probes,"unfused winner was remeasured");
    x.push_back(4);router.dispatchPointwiseForward(chain,x,output,true);
    check(Fusion::profile(&a,false)->key!=key && !Fusion::profile(&a,false)->cache_hit,"shape did not invalidate fusion plan");
    auto old_key=Fusion::profile(&a,false)->key;
    router.dispatchPointwiseForward(chain,x,output,true,nullptr,true,Mimir::DType::F16);
    check(Fusion::profile(&a,false)->key!=old_key,"dtype/mode did not invalidate fusion plan");
    RuntimeConfig config;config.device_index=7;device.initialize(config);
    old_key=Fusion::profile(&a,false)->key;
    router.dispatchPointwiseForward(chain,x,output,true,nullptr,true,Mimir::DType::F16);
    check(Fusion::profile(&a,false)->key!=old_key,"device config did not invalidate plan");
#ifdef _OPENMP
    const int threads=omp_get_max_threads();old_key=Fusion::profile(&a,false)->key;
    omp_set_num_threads(threads==1?2:1);
    router.dispatchPointwiseForward(chain,x,output,true,nullptr,true,Mimir::DType::F16);
    check(Fusion::profile(&a,false)->key!=old_key,"thread count did not invalidate plan");
    omp_set_num_threads(threads);
#endif
    env("MIMIR_RUNTIME_AUTOTUNE","0");
    const int before_disabled=device.probes;
    check(!router.dispatchPointwiseForward(chain,x,output,true),"unmeasured fusion enabled when autotune disabled");
    check(device.probes==before_disabled,"disabled calibration executed probes");
    env("MIMIR_RUNTIME_AUTOTUNE","1");env("MIMIR_PLANNER_CALIBRATION_MAX_BYTES","0");reset();
    check(!router.dispatchPointwiseForward(chain,x,output,true),"fusion ignored memory budget");
    check(!Fusion::profile(&a,false)->measured,"budget refusal claims measured latency");
    env("MIMIR_PLANNER_CALIBRATION_MAX_BYTES","67108864");
    Fusion::Request request;request.layers=chain;request.input=&x;request.available={&device};
    request.generation=424242;request.retain=true;request.training=true;
    int evaluations=0;
    auto evaluate=[&](Buffers& values){++evaluations;values.assign(chain.size(),x);return true;};
    std::vector<Fusion::Candidate> executable{{nullptr,evaluate,1},{&device,evaluate,1}};
    std::vector<Fusion::Candidate> metadata{{nullptr,{},1},{&device,{},1}};
    const auto measured=Fusion::select(request,executable);
    check(measured.measured && measured.exact,"lookup reference calibration failed");
    Fusion::Decision cached;const int before_lookup=evaluations;
    check(Fusion::lookup(request,metadata,cached) && cached.cache_hit && cached.key==measured.key,
          "metadata-only cache lookup failed");
    check(evaluations==before_lookup && cached.alternatives.size()==measured.alternatives.size(),
          "cache lookup replayed probes or lost diagnostics");
    Fusion::clearProfiles();
    check(Fusion::lookup(request,metadata,cached) && Fusion::profile(&a,false)->cache_hit,
          "cleared diagnostics were not republished");
    auto other=request;other.generation++;
    check(!Fusion::lookup(other,metadata,cached),"lookup ignored routing generation");
    std::vector<float> different_shape(x.size()+1,1.f);other=request;other.input=&different_shape;
    check(!Fusion::lookup(other,metadata,cached),"lookup ignored tensor shape");
    auto different_workers=metadata;different_workers[1].cpu_workers=2;
    check(!Fusion::lookup(request,different_workers,cached),"lookup ignored worker policy");
    const float alpha=a.leaky_relu_alpha;a.leaky_relu_alpha+=.125f;
    check(!Fusion::lookup(request,metadata,cached),"lookup ignored scalar program changes");a.leaky_relu_alpha=alpha;
    Layer alias_a=a,alias_b=b;
    auto alias_request=request;alias_request.layers={&alias_a,&alias_b};
    check(Fusion::lookup(alias_request,metadata,cached),"equivalent layer did not share calibration");
    Fusion::invalidate(measured.key);
    check(!Fusion::lookup(request,metadata,cached),"invalidated decision returned by lookup");
    const auto refreshed=Fusion::select(request,executable);
    check(refreshed.measured && !refreshed.cache_hit && evaluations>before_lookup,"invalidated program was not remeasured");
    check(Fusion::lookup(alias_request,metadata,cached) && Fusion::profile(&alias_a,false)->measured &&
          Fusion::profile(&alias_a,false)->reason==refreshed.reason,"shared profile stayed invalid after recalibration");
    for(size_t i=2;i<Mimir::Planning::ExecutionPolicy::Limits::calibration_refresh;++i)
        check(Fusion::lookup(request,metadata,cached),"decision expired before refresh interval");
    const int before_refresh=evaluations;
    check(!Fusion::lookup(request,metadata,cached) && evaluations==before_refresh,"expired lookup replayed probes");
    const auto periodic=Fusion::select(request,executable);
    check(periodic.measured && evaluations>before_refresh,"expired decision never recalibrated");
    check(Fusion::lookup(alias_request,metadata,cached) &&
          Fusion::profile(&alias_a,false)->fused_seconds==periodic.fused_seconds,
          "shared profile retained old latency after periodic refresh");
    std::cout<<"PLANNER_FUSION_CALIBRATION_OK: runtimes, forward/backward medians, strict parity, cache and limits\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
