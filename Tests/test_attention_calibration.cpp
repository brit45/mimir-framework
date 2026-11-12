#include "test_utils.hpp"
#include "Layers.hpp"
#include "Planning/ExecutionCalibration.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"
#ifdef ENABLE_OPENCL
#include "runtimes/opencl/OpenCLRuntime.hpp"
#endif
#ifdef ENABLE_VULKAN
#include "runtimes/vulkan/VulkanRuntime.hpp"
#endif
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace Policy=Mimir::Planning::ExecutionPolicy;
namespace Calibration=Mimir::Planning::Calibration;
using Buffers=std::vector<std::vector<float>>;
static void check(bool ok,const char* message) { if(!ok)throw std::runtime_error(message); }
static bool exact(const std::vector<float>& a,const std::vector<float>& b) {
    return a.size()==b.size() && (a.empty() || !std::memcmp(a.data(),b.data(),a.size()*sizeof(float)));
}

class ProbeRuntime final:public AbstractRuntime {
public:
    const char* label;int delay=0,forward_calls=0,backward_calls=0,last_workers=0;
    bool fail=false,nonfinite_gradient=false;
    explicit ProbeRuntime(const char* name):label(name) {}
    const char* name() const override{return label;}
    bool initialize(const RuntimeConfig& cfg) override{config_=cfg;return true;}
    void shutdown() override{}
    bool isInitialized() const override{return true;}
    bool linearForward(const float*,const float*,const float*,float*,int,int,int) override{return false;}
    RuntimeCapabilityLevel queryForwardCapability(LayerType) const override{return RuntimeCapabilityLevel::Native;}
    RuntimeCapabilityLevel queryBackwardCapability(LayerType) const override{return RuntimeCapabilityLevel::Native;}
    RuntimeCapabilityLevel queryForwardOperationCapability(const Layer&,const std::vector<const std::vector<float>*>&,bool) const override{return RuntimeCapabilityLevel::Native;}
    RuntimeCapabilityLevel queryBackwardOperationCapability(const Layer&,const std::vector<const std::vector<float>*>&,
        const std::vector<const std::vector<float>*>&,bool) const override{return RuntimeCapabilityLevel::Native;}
    bool cpu() const{return std::strcmp(label,"CPU")==0;}
    void begin() {
        last_workers=Policy::cpuChoice().workers;
        const int us=delay+(cpu() && last_workers==1?300:0);
        if(us)std::this_thread::sleep_for(std::chrono::microseconds(us));
    }
    bool forwardLayer(const std::vector<const std::vector<float>*>& inputs,Buffers& out,const Layer&,bool) override {
        ++forward_calls;begin();if(fail)return false;
        out={*inputs[0]};for(float& value:out[0])value*=2.f;
        if(cpu() && last_workers>2)out[0][0]=std::nextafter(out[0][0],100.f);
        return true;
    }
    bool backwardLayer(const std::vector<const std::vector<float>*>&,
        const std::vector<const std::vector<float>*>& grad,Buffers& out,Layer& layer,bool) override {
        ++backward_calls;begin();
        if(fail){layer.grad_weights[0]=999.f;return false;}
        out={*grad[0]};for(float& value:out[0])value*=2.f;
        layer.grad_weights[0]+=.5f;layer.grad_bias[0]+=.25f;
        if(nonfinite_gradient)layer.grad_weights.back()=std::numeric_limits<float>::quiet_NaN();
        // Even a one-ULP error in parameters must reject a fast CPU variant.
        if(cpu() && last_workers>2)layer.grad_weights[0]=std::nextafter(layer.grad_weights[0],100.f);
        return true;
    }
};

static void policy() {
    ProbeRuntime cpu("CPU"),gpu("VULKAN");gpu.delay=1200;
    auto& router=RuntimeRouter::instance();router.setActivators({},{},{},{},{});
    for(const char* type:{"SelfAttention","CrossAttention"}) {
        cpu.delay=0;gpu.delay=1200;gpu.fail=false;
        router.setRuntimes(nullptr,nullptr,&gpu,nullptr,&cpu);
        Layer layer("attention",type,0);layer.embed_dim=layer.in_features=4;layer.seq_len=2;layer.num_heads=2;
        layer.weights.assign(64,.01f);layer.grad_weights.assign(64,.125f);layer.grad_bias.assign(4,.25f);
        std::vector<float> x(8,1.f),kv(12,.5f),dy(8,.25f);
        std::vector<const std::vector<float>*> in{&x};if(layer.type_enum==LayerType::CrossAttention)in.push_back(&kv);
        Buffers out;AbstractRuntime* selected=nullptr;
        check(router.dispatchForwardLayer(in,out,layer,true,&selected) && selected==&cpu,"attention ignored measured CPU route");
        check(out[0]==std::vector<float>(8,2.f) && layer.grad_weights[0]==.125f,"forward probes mutated model state");
        const int expected_workers=Policy::cpuCandidates().size()>1?2:1;
        check(cpu.last_workers==expected_workers,"attention accepted inexact fastest workers");
        check(Calibration::forward_profiles.at(&layer).front().cpu_workers==expected_workers,"forward worker profile missing");
        int before=cpu.forward_calls,device_before=gpu.forward_calls;
        check(router.dispatchForwardLayer(in,out,layer,true,&selected) && cpu.forward_calls==before+1 && gpu.forward_calls==device_before,
            "warm attention replayed calibration");
        layer.causal=true;before=cpu.forward_calls;
        check(router.dispatchForwardLayer(in,out,layer,true) && cpu.forward_calls>before+1,"causal mode reused stale costs");
        layer.num_heads=4;before=cpu.forward_calls;
        check(router.dispatchForwardLayer(in,out,layer,true) && cpu.forward_calls>before+1,"head count reused stale costs");
        check(router.dispatchBackwardLayer(in,{&dy},out,layer,true,&selected) && selected==&cpu,"attention backward not calibrated");
        check(layer.grad_weights[0]==.625f && layer.grad_bias[0]==.5f,"backward probes leaked gradient accumulation");
        check(Calibration::backward_profiles.at(&layer).front().cpu_workers==expected_workers,"inexact parameter gradient accepted");
        check(router.dispatchBackwardLayer(in,{&dy},out,layer,true) && layer.grad_weights[0]==1.125f,"warm gradient update was not applied once");
        cpu.delay=2500;gpu.delay=0;router.setRuntimes(nullptr,nullptr,&gpu,nullptr,&cpu);
        check(router.dispatchBackwardLayer(in,{&dy},out,layer,true,&selected) && selected==&gpu,"attention ignored measured device route");
        check(layer.grad_weights[0]==1.625f,"measured GPU probes mutated real gradients");
        gpu.fail=true;
        check(router.dispatchBackwardLayer(in,{&dy},out,layer,true,&selected) && selected==&cpu && layer.grad_weights[0]==2.125f,
            "failed attention runtime leaked partial gradients");
        gpu.fail=false;
        check(router.dispatchBackwardLayer(in,{&dy},out,layer,true,&selected) && selected==&gpu && layer.grad_weights[0]==2.625f,
            "valid device was not remeasured after failure");
        gpu.nonfinite_gradient=true;
        check(router.dispatchBackwardLayer(in,{&dy},out,layer,true,&selected) && selected==&cpu &&
            layer.grad_weights[0]==3.125f && layer.grad_weights.back()==.125f,
            "nonfinite attention parameter gradient was committed before fallback");
        gpu.nonfinite_gradient=false;cpu.delay=0;gpu.delay=1200;
        setTestEnvironment("MIMIR_PLANNER_CALIBRATION_MAX_BYTES","0");layer.seq_len=3;
        device_before=gpu.forward_calls;before=cpu.forward_calls;
        check(router.dispatchForwardLayer(in,out,layer,true,&selected) && selected==&cpu && cpu.last_workers==1 &&
            cpu.forward_calls==before+1 && gpu.forward_calls==device_before,"attention ignored calibration memory budget");
        check(!Calibration::forward_profiles.count(&layer),"memory refusal published stale costs");
        setTestEnvironment("MIMIR_PLANNER_CALIBRATION_MAX_BYTES","67108864");
        // Every field used by attention geometry must distinguish cached plans.
        const auto signature=[&](const Layer& value){return Calibration::signature(value,true,true,42,in,{&dy});};
        const size_t key=signature(layer);
        for(int field=0;field<5;++field) {
            Layer changed=layer;
            if(field==0)++changed.num_heads;
            if(field==1)++changed.num_kv_heads;
            if(field==2)++changed.head_dim;
            if(field==3)changed.causal=!changed.causal;
            if(field==4)changed.rope_theta=10000.f;
            check(signature(changed)!=key,"attention signature omitted geometry");
        }
    }
    Calibration::RuntimeRoute available;available.push_back(&cpu);available.push_back(&gpu);
    for(auto type:{LayerType::Dropout,LayerType::Dropout2d,LayerType::AlphaDropout,LayerType::Reparameterize}) {
        Layer stochastic;stochastic.type_enum=type;int probes=0;
        auto ranked=Calibration::rank(available,stochastic,{}, {},false,true,1,
            [&](AbstractRuntime&,Layer&,Buffers&){++probes;return true;});
        check(probes==0 && !ranked.calibrated,"stochastic layer became replayable");
    }
    router.setRuntimes(nullptr,nullptr,nullptr,nullptr,nullptr);
}

static void close(const std::vector<float>& actual,const std::vector<float>& expected,bool same_cpu) {
    check(actual.size()==expected.size(),"native attention shape mismatch");
    if(same_cpu){check(exact(actual,expected),"measured CPU changed attention bits");return;}
    for(size_t i=0;i<actual.size();++i)check(std::isfinite(actual[i]) &&
        std::abs(actual[i]-expected[i])<=4e-4f*(1+std::abs(expected[i])),"measured GPU changed attention values");
}

static void native(AbstractRuntime* device=nullptr) {
    CpuRuntime cpu;cpu.initialize({});auto& router=RuntimeRouter::instance();
    router.setActivators({},{},{},{},{});
    router.setRuntimes(nullptr,nullptr,device && std::string(device->name())=="VULKAN"?device:nullptr,
        device && std::string(device->name())=="OPENCL"?device:nullptr,&cpu);
    for(const char* type:{"SelfAttention","CrossAttention"})for(bool causal:{false,true}) {
        Layer actual("native",type,0);actual.embed_dim=actual.in_features=8;actual.seq_len=17;
        actual.num_heads=2;actual.causal=causal;actual.use_bias=false;
        actual.weights.resize(256);for(size_t i=0;i<256;++i)actual.weights[i]=float(int(i%19)-9)*.013f;
        actual.grad_weights.assign(256,.125f);Layer expected=actual;
        std::vector<float> x(17*8),kv(20*8),dy(x.size());
        for(size_t i=0;i<x.size();++i){x[i]=std::sin(float(i)*.13f);dy[i]=std::cos(float(i)*.17f)*.1f;}
        for(size_t i=0;i<kv.size();++i)kv[i]=std::cos(float(i)*.21f);
        std::vector<const std::vector<float>*> inputs{&x};if(actual.type_enum==LayerType::CrossAttention)inputs.push_back(&kv);
        Buffers want,got;AbstractRuntime* selected=nullptr;
        {Policy::CpuScope serial(1);check(cpu.forwardLayer(inputs,want,expected,true),"serial attention forward failed");}
        check(router.dispatchForwardLayer(inputs,got,actual,true,&selected),"native measured attention forward failed");
        check(Calibration::forward_profiles.count(&actual)==1,"native attention forward was excluded from calibration");
        const auto& forward=Calibration::forward_profiles.at(&actual);
        check(selected==forward.front().runtime,"native route did not use measured winner");
        check(forward.size()==(device?2u:1u),"native attention candidate was not validated");
        close(got[0],want[0],selected==&cpu);
        {Policy::CpuScope serial(1);check(cpu.backwardLayer(inputs,{&dy},want,expected,true),"serial attention backward failed");}
        check(router.dispatchBackwardLayer(inputs,{&dy},got,actual,true,&selected),"native measured attention backward failed");
        const auto& backward=Calibration::backward_profiles.at(&actual);
        check(selected==backward.front().runtime && backward.size()==(device?2u:1u),"native backward candidates missing");
        check(got.size()==want.size(),"native gradient count");
        for(size_t i=0;i<got.size();++i)close(got[i],want[i],selected==&cpu);
        close(actual.grad_weights,expected.grad_weights,selected==&cpu);
        std::cout<<"ATTENTION_MEASURED_NATIVE_OK "<<type<<" causal="<<causal<<" backend="<<selected->name()
            <<" cpu_workers="<<backward.front().cpu_workers
            <<" device_candidate="<<(device?device->name():"none")
            <<" validated_forward="<<forward.size()<<" validated_backward="<<backward.size()<<'\n';
    }
    router.setRuntimes(nullptr,nullptr,nullptr,nullptr,nullptr);
}

int main(int argc,char** argv) {
    try {
        setTestEnvironment("MIMIR_RUNTIME_AUTOTUNE","1");setTestEnvironment("MIMIR_PLANNER_CALIBRATION_MAX_BYTES","67108864");
#ifdef _OPENMP
        omp_set_num_threads(4);
#endif
        if(argc>1) {
            std::unique_ptr<AbstractRuntime> device;
            const std::string backend=argv[1];
#ifdef ENABLE_OPENCL
            if(backend=="OPENCL")device=std::make_unique<OpenCLRuntime>();
#endif
#ifdef ENABLE_VULKAN
            if(backend=="VULKAN")device=std::make_unique<VulkanRuntime>();
#endif
            if(!device){std::cout<<"SKIP backend not compiled\n";return 77;}
            RuntimeConfig cfg;cfg.linear_enabled=cfg.attention_enabled=true;
            if(!device->initialize(cfg)){std::cout<<"SKIP device unavailable\n";return 77;}
            native(device.get());
        } else {
            policy();
#ifdef _OPENMP
            native();
#endif
        }
        check(Policy::cpuChoice().workers==1,"calibration worker scope leaked");
        std::cout<<"ATTENTION_CALIBRATION_OK exact CPU choices, isolated gradients, cache, geometry, budget, stochastic exclusion\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
