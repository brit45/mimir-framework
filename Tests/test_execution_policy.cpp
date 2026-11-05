#include "Planning/ExecutionPolicy.hpp"
#include "Model.hpp"
#include "Planning/ExecutionCalibration.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "runtimes/cpu/Parallel.hpp"
#include "viz/ModelGraph.hpp"
#include <thread>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <set>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace Policy=Mimir::Planning::ExecutionPolicy;
namespace Calibration=Mimir::Planning::Calibration;
static void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static void env(const char* name,const char* value){
#ifdef _WIN32
    _putenv_s(name,value);
#else
    setenv(name,value,1);
#endif
}
struct ProbeCpu : AbstractRuntime {
    std::set<int> measured;int last_workers=0;
    const char* name()const override{return "CPU";}
    bool initialize(const RuntimeConfig&)override{return true;}
    void shutdown()override{}
    bool isInitialized()const override{return true;}
    bool linearForward(const float*,const float*,const float*,float*,int,int,int)override{return false;}
    bool supportsForwardLayerType(LayerType)const override{return true;}
    bool supportsBackwardLayerType(LayerType)const override{return true;}
    RuntimeCapabilityLevel queryForwardCapability(LayerType)const override{return RuntimeCapabilityLevel::Native;}
    RuntimeCapabilityLevel queryBackwardCapability(LayerType)const override{return RuntimeCapabilityLevel::Native;}
    bool evaluate(const std::vector<float>& x,std::vector<std::vector<float>>& result){
        last_workers=Policy::cpuChoice().workers;measured.insert(last_workers);
        if(last_workers==1)std::this_thread::sleep_for(std::chrono::microseconds(200));
        result={x};
        // A faster but numerically wrong worker configuration must be refused.
        if(last_workers>2 && !x.empty())result[0][0]=std::nextafter(x[0],100.f);
        return true;
    }
    bool forwardLayer(const std::vector<const std::vector<float>*>& input,
        std::vector<std::vector<float>>& output,const Layer&,bool)override{return evaluate(*input[0],output);}
    bool backwardLayer(const std::vector<const std::vector<float>*>&,
        const std::vector<const std::vector<float>*>& grad,std::vector<std::vector<float>>& output,Layer&,bool)override {
        return evaluate(*grad[0],output);
    }
};
int main(){try{
    env("MIMIR_RUNTIME_AUTOTUNE","1");env("MIMIR_PLANNER_CALIBRATION_MAX_BYTES","67108864");
    for(size_t work:{size_t(1),size_t(63),size_t(131073),size_t(1000000000)})
        check(RuntimeParallel::workers(work)==1,"unmeasured work guessed a CPU threshold");
    check(RuntimeConfig{}.linear_min_ops==0 && RuntimeConfig{}.conv_min_ops==0 &&
        RuntimeConfig{}.norm_min_elements==0 && RuntimeConfig{}.attention_min_ops==0,"hidden default runtime threshold");
    env("MIMIR_CUDA_LINEAR_MIN_OPS","37");
    const auto configured=RuntimeConfig::fromEnv("CUDA");
    check(configured.linear_min_ops==37 && !Policy::runtimeWorkAllowed(36,configured.linear_min_ops) &&
        Policy::runtimeWorkAllowed(37,configured.linear_min_ops),"explicit minimum is not centralized");
    env("MIMIR_CUDA_LINEAR_MIN_OPS","0");
    ProbeCpu cpu;auto& router=RuntimeRouter::instance();router.setActivators({},{},{},{},{});
    router.setRuntimes(nullptr,nullptr,nullptr,nullptr,&cpu);
    std::vector<Layer> layers;layers.emplace_back("policy","Identity",0);auto& layer=layers[0];
    layer.inputs={"x"};layer.output="x";
    std::vector<float> input{1,2,3};std::vector<std::vector<float>> output;
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    check(router.dispatchForwardLayer({&input},output,layer,true),"CPU policy forward failed");
    check(output==std::vector<std::vector<float>>{input},"worker probes changed output");
#ifdef _OPENMP
    check(cpu.measured==std::set<int>({1,2,4}) && cpu.last_workers==2,"planner did not choose exact fastest worker count");
    check(Calibration::forward_profiles.at(&layer)[0].cpu_workers==2,"forward worker decision missing");
#endif
    check(RuntimeParallel::workers(1000000000)==1,"CPU scope leaked after dispatch");
    check(router.dispatchBackwardLayer({&input},{&input},output,layer,true),"CPU policy backward failed");
#ifdef _OPENMP
    check(cpu.last_workers==2 && Calibration::backward_profiles.at(&layer)[0].cpu_workers==2,"backward did not use planner worker decision");
    {
        Policy::CpuScope outer(4);
        check(RuntimeParallel::workers(1)==1 && RuntimeParallel::workers(1000000)==4,"graph geometry cap failed");
        std::vector<int> nested(16);
        RuntimeParallel::forEach(nested.size(),RuntimeParallel::workers(nested.size()),[&](size_t i){nested[i]=RuntimeParallel::workers(1000000);});
        for(int value:nested)check(value==1,"nested OpenMP team was permitted");
    }
#endif
    const auto plan=Mimir::Planning::build_execution_plan_global(layers,true);
    const auto graph=Mimir::Viz::modelGraphSnapshot(layers,&plan,"cost",true,true,true);
    check(graph["execution_policy"]["selection"]=="measured_latency","GraphSet lacks central execution policy");
#ifdef _OPENMP
    check(graph["nodes"][0]["forward_execution_choices"][0]["cpu_workers"]==2,"GraphSet shows a different worker policy");
#endif
    env("MIMIR_RUNTIME_AUTOTUNE","0");
    check(router.dispatchForwardLayer({&input},output,layer,true) && cpu.last_workers==1,"unmeasured fallback is not serial reference");
    check(Calibration::forward_profiles.count(&layer)==0,"GraphSet retained a stale measured worker decision");
    check(Policy::cpuCandidates(true)==std::vector<int>{1},"ordered reduction can change arithmetic order");
    // Large seeded initialization must not pick another RNG stream at a size cutoff.
    Model initialized;
    initialized.push("seeded", "Linear", 256*257);
    auto& seeded=initialized.getMutableLayers()[0];
    seeded.in_features=256;seeded.out_features=257;seeded.use_bias=false;
    initialized.allocateParams();
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    initialized.initializeWeights("xavier",123);
    const std::vector<float> first(seeded.getWeights(),seeded.getWeights()+256*257);
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    initialized.initializeWeights("xavier",123);
    check(std::equal(first.begin(),first.end(),seeded.getWeights()),"initialization RNG depends on worker count");
    std::cout<<"EXECUTION_POLICY_OK: one policy, measured CPU threads, exact parity, geometry and GraphSet\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
