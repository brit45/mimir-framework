#include "test_utils.hpp"
#include "Layers.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "Planning/ExecutionCalibration.hpp"
#include <thread>
#include <iostream>
#include <stdexcept>
using namespace Mimir::Planning;
static void check(bool b,const char* why){if(!b)throw std::runtime_error(why);}
class Timed final:public AbstractRuntime {
public:
    const char* label;int delay=0,forward_calls=0,backward_calls=0;bool fail=false,bad=false,ready=true;float shift=0;
    explicit Timed(const char* s):label(s){}
    const char* name()const override{return label;}
    bool initialize(const RuntimeConfig& c)override{config_=c;return true;}
    void shutdown()override{ready=false;}
    bool isInitialized()const override{return ready;}
    bool linearForward(const float*,const float*,const float*,float*,int,int,int)override{return false;}
    RuntimeCapabilityLevel queryForwardCapability(LayerType)const override{return RuntimeCapabilityLevel::Native;}
    RuntimeCapabilityLevel queryBackwardCapability(LayerType)const override{return RuntimeCapabilityLevel::Native;}
    RuntimeCapabilityLevel queryForwardOperationCapability(const Layer&,const std::vector<const std::vector<float>*>&,bool)const override{return RuntimeCapabilityLevel::Native;}
    RuntimeCapabilityLevel queryBackwardOperationCapability(const Layer&,const std::vector<const std::vector<float>*>&,const std::vector<const std::vector<float>*>&,bool)const override{return RuntimeCapabilityLevel::Native;}
    bool forwardLayer(const std::vector<const std::vector<float>*>& x,std::vector<std::vector<float>>& y,const Layer&,bool)override{
        ++forward_calls;if(delay)std::this_thread::sleep_for(std::chrono::microseconds(delay));
        if(fail)return false;
        y={*x[0]};
        for(auto& v:y[0]){v*=bad?3:2;v+=shift;}
        return true;
    }
    bool backwardLayer(const std::vector<const std::vector<float>*>&,const std::vector<const std::vector<float>*>& g,std::vector<std::vector<float>>& dx,Layer& l,bool)override{
        ++backward_calls;if(delay)std::this_thread::sleep_for(std::chrono::microseconds(delay));
        if(fail){l.grad_weights[0]=900;return false;}
        dx={*g[0]};for(auto& v:dx[0])v*=2;
        l.grad_weights[0]+=.5f;l.grad_bias[0]+=.25f;return true;
    }
};
int main(){try{
    unsetTestEnvironment("MIMIR_RUNTIME_AUTOTUNE");
    setTestEnvironment("MIMIR_PLANNER_MODE","cost");setTestEnvironment("MIMIR_PLANNER_AUTOTUNE","1");
    Timed gpu("VULKAN"),cpu("CPU");gpu.delay=2000;
    auto& router=RuntimeRouter::instance();router.setActivators({},{},{},{},{});router.setRuntimes(nullptr,nullptr,&gpu,nullptr,&cpu);
    Layer layer;layer.name="dense";layer.type_enum=LayerType::Linear;layer.in_features=layer.out_features=1;layer.weights={2};layer.grad_weights={.125f};layer.grad_bias={.25f};
    std::vector<float>x{1,-2},dy{.1f,-.2f};std::vector<std::vector<float>> out;AbstractRuntime* selected=nullptr;
    check(router.dispatchForwardLayerPlanned(&gpu,{&x},out,layer,true,&selected) && selected==&cpu,"planner ignored measured CPU winner");
    check(out[0]==std::vector<float>({2,-4}) && layer.grad_weights[0]==.125f,"forward calibration mutated state");
    // Graph planner switches must not force slow host/device round trips.
    setTestEnvironment("MIMIR_ENABLE_PLANNER","0");
    setTestEnvironment("MIMIR_PLANNER_COST_MODEL","0");
    setTestEnvironment("MIMIR_PLANNER_MODE","legacy");
    check(Calibration::enabled(),"graph planner disabled runtime calibration");
    check(router.dispatchForwardLayer({&x},out,layer,true,&selected) && selected==&cpu,
          "planner-off forward ignored measured CPU winner");
    unsetTestEnvironment("MIMIR_ENABLE_PLANNER");
    unsetTestEnvironment("MIMIR_PLANNER_COST_MODEL");
    setTestEnvironment("MIMIR_PLANNER_MODE","cost");
    const int probes=gpu.forward_calls;
    check(router.dispatchForwardLayer({&x},out,layer,true,&selected) && selected==&cpu && gpu.forward_calls==probes,"cached placement reprobed losing runtime");
    check(router.dispatchBackwardLayer({&x},{&dy},out,layer,true,&selected) && selected==&cpu,"backward cost selection failed");
    check(layer.grad_weights[0]==.625f && layer.grad_bias[0]==.5f,"backward calibration accumulated real gradients");
    check(router.dispatchBackwardLayer({&x},{&dy},out,layer,true,&selected) && layer.grad_weights[0]==1.125f,"cached backward did not accumulate once");
    for(int i=0;i<1021;++i)check(router.dispatchForwardLayer({&x},out,layer,true,&selected) && selected==&cpu,"cached forward failed");
    gpu.delay=0;cpu.delay=2000;
    check(router.dispatchForwardLayer({&x},out,layer,true,&selected) && selected==&gpu,"periodic calibration never adapted to changed performance");
    RuntimeConfig config;config.device_index=1;gpu.initialize(config);
    check(router.dispatchForwardLayerPlanned(&cpu,{&x},out,layer,true,&selected) && selected==&gpu,"device/configuration change reused stale costs");
    check(router.dispatchBackwardLayer({&x},{&dy},out,layer,true,&selected) && selected==&gpu,"backward configuration not refreshed");
    gpu.fail=true;
    check(router.dispatchBackwardLayer({&x},{&dy},out,layer,true,&selected) && selected==&cpu,"failed measured runtime did not fall back");
    check(layer.grad_weights[0]==2.125f,"failed runtime leaked gradient writes");
    gpu.fail=false;gpu.bad=true;router.setRuntimes(nullptr,nullptr,&gpu,nullptr,&cpu);
    check(router.dispatchForwardLayer({&x},out,layer,true,&selected) && selected==&cpu,"numerically invalid fastest candidate accepted");
    gpu.bad=false;cpu.delay=0;
    setTestEnvironment("MIMIR_PLANNER_CALIBRATION_MAX_BYTES","0");std::vector<float> larger(7,1);const int before=gpu.forward_calls;
    check(router.dispatchForwardLayer({&larger},out,layer,true,&selected) && selected==&cpu && gpu.forward_calls==before,"unmeasured execution did not retain the CPU reference");
    Layer single_vector=layer;
    single_vector.in_features=7;
    single_vector.weights.resize(7,2.f);
    single_vector.grad_weights.resize(7,0.f);
    const int forward_before=gpu.forward_calls;
    check(router.dispatchForwardLayerPlanned(&gpu,{&larger},out,single_vector,true,&selected) && selected==&cpu && gpu.forward_calls==forward_before,
          "memory-limited vector placement chose an unmeasured GPU");
    const int backward_before=gpu.backward_calls;
    check(router.dispatchBackwardLayer({&larger},{&dy},out,single_vector,true,&selected) && selected==&cpu && gpu.backward_calls==backward_before,
          "memory-limited backward chose an unmeasured GPU");
    check(single_vector.grad_weights[0]==2.625f && layer.grad_weights[0]==2.125f,
          "memory-limited backward did not accumulate once in its own layer");
    setTestEnvironment("MIMIR_ENABLE_PLANNER","0");
    setTestEnvironment("MIMIR_PLANNER_MODE","static");
    check(router.dispatchBackwardLayer({&larger},{&dy},out,single_vector,true,&selected) && selected==&cpu,
          "planner-off large vector backward chose an unmeasured GPU");
    unsetTestEnvironment("MIMIR_ENABLE_PLANNER");
    setTestEnvironment("MIMIR_PLANNER_MODE","cost");
    unsetTestEnvironment("MIMIR_PLANNER_CALIBRATION_MAX_BYTES");
    // Newly parallel families must use measured host-to-host costs too.
    gpu.delay=2000;
    for(auto type:{LayerType::ConvTranspose2d,LayerType::Embedding,LayerType::MaxPool2d,
                   LayerType::GlobalAvgPool2d,LayerType::Permute,LayerType::UpsampleBilinear,
                   LayerType::ZeroPad2d,LayerType::Constant,LayerType::NMS}){
        layer.type_enum=type;
        check(router.dispatchForwardLayerPlanned(&gpu,{&larger},out,layer,true,&selected) && selected==&cpu,
              "generalized operation ignored measured CPU winner");
    }
    gpu.delay=0;cpu.delay=2000;layer.nms_max_detections=1;
    check(router.dispatchForwardLayer({&larger},out,layer,true,&selected) && selected==&gpu,
          "NMS controls reused stale calibration");
    std::vector<float> high_indices(8,10000.f);gpu.shift=1.f;
    check(router.dispatchForwardLayer({&high_indices},out,layer,true,&selected) && selected==&cpu,
          "NMS calibration accepted different detection indices");
    gpu.shift=0.f;
    gpu.delay=2000;cpu.delay=0;
    layer.type_enum=LayerType::Dropout;const int stateful_before=gpu.forward_calls;
    check(router.dispatchForwardLayer({&larger},out,layer,true,&selected) && gpu.forward_calls==stateful_before+1,"stateful operation was replayed");
    setTestEnvironment("MIMIR_PLANNER_AUTOTUNE","0");
    check(!Calibration::enabled(),"legacy explicit tuning switch ignored");
    setTestEnvironment("MIMIR_RUNTIME_AUTOTUNE","1");
    check(Calibration::enabled(),"runtime tuning override ignored");
    setTestEnvironment("MIMIR_RUNTIME_AUTOTUNE","0");
    check(!Calibration::enabled(),"runtime tuning disable ignored");
    unsetTestEnvironment("MIMIR_RUNTIME_AUTOTUNE");
    unsetTestEnvironment("MIMIR_PLANNER_AUTOTUNE");
    router.setRuntimes(nullptr,nullptr,nullptr,nullptr,nullptr);
    std::cout<<"EXECUTION_CALIBRATION_OK measured routes, gradient isolation, invalidation, parity, memory, stateful exclusion\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
