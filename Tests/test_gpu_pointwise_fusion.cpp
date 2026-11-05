#include "test_utils.hpp"
#include "runtimes/GpuPointwiseFusion.hpp"
#include "runtimes/portable/PortableWorkBudget.hpp"
#include "runtimes/portable/PointwiseFusionIds.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"
#include "runtimes/cuda/CudaRuntime.hpp"
#include "runtimes/rocm/RocmRuntime.hpp"
#include "runtimes/opencl/OpenCLRuntime.hpp"
#include "runtimes/vulkan/VulkanRuntime.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "Planning/FusionCalibration.hpp"
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>

static void check(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
static double maximum_error=0;
static size_t exact_buffers=0,total_buffers=0;
static void compare(const std::vector<float>& a,const std::vector<float>& b) {
    check(a.size()==b.size(),"fusion shape mismatch");
    ++total_buffers;
    if(a.empty() || !std::memcmp(a.data(),b.data(),a.size()*sizeof(float)))++exact_buffers;
    for(size_t i=0;i<a.size();++i) {
        const double error=std::abs(double(a[i])-double(b[i]));
        maximum_error=std::max(maximum_error,error);
        check(std::isfinite(a[i]) && error<=3e-5*(1+std::abs(double(b[i]))),"fusion numeric mismatch");
    }
}
static std::vector<Layer> layers(const std::vector<std::string>& types) {
    std::vector<Layer> result;
    for(const auto& type:types) {result.emplace_back(type,type,0);result.back().inputs={"x"};result.back().leaky_relu_alpha=.13f;}
    return result;
}
static void parity(AbstractRuntime& gpu,CpuRuntime& cpu,const std::vector<std::string>& types,size_t n) {
    auto nodes=layers(types);std::vector<const Layer*> chain;
    for(const auto& node:nodes)chain.push_back(&node);
    check(runtimeCapabilityIsNative(gpu.queryPointwiseFusionCapability(chain,false,true)),"missing native forward fusion");
    check(runtimeCapabilityIsNative(gpu.queryPointwiseFusionCapability(chain,true,true)),"missing native backward fusion");
    std::vector<float> x(n),dy(n);
    const float special[]={-30,-3,-1,-0.f,0,.1f,1,3,30};
    for(size_t i=0;i<n;++i){x[i]=i<9?special[i]:float(int(i%101)-50)*.137f;dy[i]=float(int(i%17)-8)*.125f;}
    std::vector<std::vector<float>> reference,temp;
    for(auto& node:nodes) {
        const auto& source=reference.empty()?x:reference.back();
        check(cpu.forwardLayer({&source},temp,node,true),"CPU reference forward");
        reference.push_back(std::move(temp[0]));
    }
    for(bool retain:{false,true}) {
        std::vector<std::vector<float>> result{{42}};
        check(gpu.pointwiseFusionForward(chain,x,result,retain,true),"native fused forward failed");
        check(result.size()==(retain?chain.size():1),"retained output count");
        for(size_t j=0;j<result.size();++j)compare(result[j],reference[retain?j:reference.size()-1]);
    }
    // Use the GPU's own retained snapshots to check the adjoint independently.
    std::vector<std::vector<float>> saved;
    check(gpu.pointwiseFusionForward(chain,x,saved,true,true),"GPU snapshots");
    std::vector<const std::vector<float>*> snapshots;
    for(size_t j=0;j<chain.size();++j)snapshots.push_back(j?&saved[j-1]:&x);
    auto gradient=dy;
    for(size_t j=chain.size();j-->0;) {
        check(cpu.backwardLayer({snapshots[j]},{&gradient},temp,nodes[j],true),"CPU reference backward");
        gradient=std::move(temp[0]);
    }
    std::vector<float> dx;
    check(gpu.pointwiseFusionBackward(chain,snapshots,dy,dx),"native fused backward failed");compare(dx,gradient);
    // A declined call must leave the caller's buffer intact.
    snapshots[0]=nullptr;dx={42};
    if(nodes[0].type_enum!=LayerType::Identity)
        check(!gpu.pointwiseFusionBackward(chain,snapshots,dy,dx) && dx==std::vector<float>{42},"invalid snapshot mutated output");
}
static void routing(AbstractRuntime& gpu,CpuRuntime& cpu) {
    auto nodes=layers({"ReLU","HardSigmoid","HardSwish"});
    std::vector<const Layer*> chain;for(auto& node:nodes)chain.push_back(&node);
    auto& router=RuntimeRouter::instance();const std::string backend=gpu.name();
    router.setRuntimes(backend=="ROCM"?&gpu:nullptr,backend=="CUDA"?&gpu:nullptr,
        backend=="VULKAN"?&gpu:nullptr,backend=="OPENCL"?&gpu:nullptr,&cpu);
    std::vector<float> x(65539,.125f),dx;std::vector<std::vector<float>> y;
    AbstractRuntime* selected=nullptr;
    check(router.dispatchPointwiseForward(chain,x,y,true,&selected,true),"routed forward failed");
    auto validate=[&](bool backward) {
        const auto* decision=Mimir::Planning::FusionCalibration::profile(chain[0],backward);
        check(decision && decision->measured,"fusion calibration missing");
        bool found=false;
        for(const auto& candidate:decision->alternatives)if(candidate.runtime==backend)found=true;
        check(found,"GPU missing from fusion candidates");
        if(selected==&gpu)check(decision->fused && decision->exact,"inexact GPU selected");
        std::cout<<"FUSION_SELECTION "<<backend<<" backward="<<backward<<" selected="<<decision->selected_runtime<<'\n';
    };
    validate(false);
    check(router.dispatchPointwiseBackward(chain,{&x,&y[0],&y[1]},x,dx,&selected),"routed backward failed");
    validate(true);
    router.setRuntimes(nullptr,nullptr,nullptr,nullptr,&cpu);
}
static void packing() {
    auto nodes=layers({"Identity","LeakyReLU","ReLU"});
    std::vector<const Layer*> chain;for(auto& node:nodes)chain.push_back(&node);
    std::vector<float> x{-.5f,0.f,1.f},short_input{1.f};
    Mimir::GpuFusion::Program program;
    check(program.prepare(chain,x,true),"forward packing failed");
    check(program.output_count==9 && program.memory[program.operations]==10.f &&
          program.memory[program.operations+2]==1.f && program.memory[program.operations+3]==.13f,"program semantics");
    for(size_t i=0;i<program.output_count;++i)program.memory[program.output+i]=float(i);
    std::vector<std::vector<float>> output;
    program.forwardResult(output);
    check(output==std::vector<std::vector<float>>{{0,1,2},{3,4,5},{6,7,8}},"retained output layout");
    std::vector<const std::vector<float>*> saved{nullptr,&x,&x};
    check(program.prepare(chain,x,false,&saved),"Identity null snapshot rejected");
    check(program.memory[program.snapshots+3]==x[0] && program.output_count==3,"snapshot layout");
    saved[1]=&short_input;check(!program.prepare(chain,x,false,&saved),"short snapshot accepted");
    saved.pop_back();check(!program.prepare(chain,x,false,&saved),"missing stage snapshot accepted");
    Layer reduction("softmax","Softmax",0);
    check(!program.prepare({&nodes[0],&reduction},x,false),"reduction packing accepted");
    std::vector<const Layer*> oversized(Mimir::Planning::ExecutionPolicy::Limits::portable_reduction_span+1,&nodes[0]);
    check(!program.prepare(oversized,x,false),"unbounded program accepted");
    std::vector<int> forward_args{Mimir::GpuFusion::forward_kernel_id,256,1,1,4,10,3,3,1};
    std::vector<int> backward_args{Mimir::GpuFusion::backward_kernel_id,256,1,1,4,13,19,3,3};
    check(PortableStaging::parallelDispatchFits(32,forward_args) &&
          PortableStaging::parallelDispatchFits(32,backward_args),"fusion parallel budget rejected");
    forward_args.pop_back();check(!PortableStaging::parallelDispatchFits(32,forward_args),"malformed program budget accepted");
    backward_args[8]=0;check(!PortableStaging::parallelDispatchFits(32,backward_args),"zero-depth budget accepted");
    backward_args[8]=1024*1024;backward_args[2]=1024;
    check(!PortableStaging::parallelDispatchFits(32,backward_args),"unbounded kernel work accepted");
    std::cout<<"GPU_FUSION_PACKING_OK outputs, snapshots, opcodes, bounds\n";
}
int main(int argc,char** argv) {try {
    if(argc>1 && std::string(argv[1])=="--packing"){packing();return 0;}
    const std::string backend=argc>1?argv[1]:"VULKAN";
    std::unique_ptr<AbstractRuntime> gpu;
#ifdef ENABLE_VULKAN
    if(backend=="VULKAN")gpu=std::make_unique<VulkanRuntime>();
#endif
#ifdef ENABLE_OPENCL
    if(backend=="OPENCL")gpu=std::make_unique<OpenCLRuntime>();
#endif
#ifdef ENABLE_CUDA
    if(backend=="CUDA")gpu=std::make_unique<CudaRuntime>();
#endif
#ifdef ENABLE_ROCM
    if(backend=="ROCM")gpu=std::make_unique<RocmRuntime>();
#endif
    if(!gpu){std::cout<<"SKIP "<<backend<<" not compiled\n";return 77;}
    RuntimeConfig config;config.linear_enabled=true;config.linear_min_ops=0;
    if(!gpu->initialize(config)){std::cout<<"SKIP "<<backend<<" unavailable\n";return 77;}
    CpuRuntime cpu;check(cpu.initialize(config),"CPU initialize");
    const std::vector<std::string> types={"Identity","ReLU","LeakyReLU","GELU","SiLU","Tanh","Sigmoid","Softplus","Mish","HardSigmoid","HardSwish"};
    for(const auto& a:types)for(const auto& b:types)parity(*gpu,cpu,{a,b},257);
    for(size_t size:{size_t(0),size_t(1),size_t(4099),size_t(65539)}) {
        parity(*gpu,cpu,{"LeakyReLU","HardSwish","Sigmoid"},size);
        parity(*gpu,cpu,{"ReLU","SiLU","GELU","Tanh","Identity"},size);
    }
    Layer reduction("softmax","Softmax",0),relu("relu","ReLU",0);
    check(!runtimeCapabilityIsNative(gpu->queryPointwiseFusionCapability({&relu,&reduction},false,false)),"reduction fused as pointwise");
    routing(*gpu,cpu);
    std::cout<<"GPU_POINTWISE_FUSION_OK "<<backend<<" pairs=121 forward/backward retained/final-only max_error="
        <<maximum_error<<" exact_cpu_buffers="<<exact_buffers<<'/'<<total_buffers<<'\n';
    return 0;
} catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
