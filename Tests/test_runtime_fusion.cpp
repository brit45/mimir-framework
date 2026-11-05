#include "test_utils.hpp"
#include "Model.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "Planning/FusionCalibration.hpp"
#include "RngContext.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"
#include "runtimes/cpu/PointwiseFusion.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

static void require(bool ok,const std::string& message) {if(!ok)throw std::runtime_error(message);}
static bool exact(const std::vector<float>& a,const std::vector<float>& b) {
    return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0);
}
static void fusion(bool enabled) {
#ifdef _WIN32
    _putenv_s("MIMIR_PLANNER_FUSION",enabled?"1":"0");
#else
    setenv("MIMIR_PLANNER_FUSION",enabled?"1":"0",1);
#endif
}
static std::vector<Layer> pointwise(const std::vector<std::string>& types) {
    std::vector<Layer> result;
    for(const auto& type:types) {
        result.emplace_back(type,type,0);result.back().inputs={"x"};result.back().output="x";
        result.back().leaky_relu_alpha=.13f;
    }
    return result;
}
static std::vector<const Layer*> pointers(const std::vector<Layer>& layers) {
    std::vector<const Layer*> result;for(const auto& l:layers)result.push_back(&l);return result;
}
static void nativeParity(CpuRuntime& cpu) {
    auto& router=RuntimeRouter::instance();
    std::vector<std::string> types;
    for(int kind=0;kind<static_cast<int>(LayerType::UNKNOWN);++kind)
        if(Mimir::CpuFusion::supports(static_cast<LayerType>(kind)))types.push_back(LayerRegistry::type_to_string(static_cast<LayerType>(kind)));
    for(const auto& first:types)for(const auto& second:types) {
        auto ls=pointwise({first,second});const auto chain=pointers(ls);
        for(size_t n:{size_t(0),size_t(1),size_t(63),size_t(4099),size_t(131073)}) {
            std::vector<float> x(n),dy(n);
            const float special[]={-30.f,-3.f,-1.f,-0.f,0.f,.1f,1.f,3.f,30.f};
            for(size_t i=0;i<n;++i){x[i]=special[i%9];dy[i]=float(int(i%17)-8)*.125f;}
            std::vector<std::vector<float>> reference,temporary;
            std::vector<const std::vector<float>*> snapshots;
            for(size_t i=0;i<ls.size();++i) {
                const auto& input=i?reference.back():x;
                require(cpu.forwardLayer({&input},temporary,ls[i],true),"reference forward");
                reference.push_back(std::move(temporary[0]));
            }
            std::vector<std::vector<float>> outputs;
            AbstractRuntime* selected=nullptr;
            require(cpu.pointwiseFusionForward(chain,x,outputs,true,true),"native forward kernel");
            for(size_t i=0;i<reference.size();++i)require(exact(outputs[i],reference[i]),"forward mismatch: "+first+"/"+second);
            require(cpu.pointwiseFusionForward(chain,x,outputs,false,false),"final-only native route");
            require(exact(outputs[0],reference.back()),"final-only parity");
            auto gradient=dy;
            for(size_t i=ls.size();i-->0;) {
                const auto& input=i?reference[i-1]:x;
                require(cpu.backwardLayer({&input},{&gradient},temporary,ls[i],true),"reference backward");
                gradient=std::move(temporary[0]);
            }
            for(size_t i=0;i<ls.size();++i)snapshots.push_back(i?&reference[i-1]:&x);
            std::vector<float> dx;
            require(cpu.pointwiseFusionBackward(chain,snapshots,dy,dx),"native backward route");
            require(exact(dx,gradient),"backward mismatch: "+first+"/"+second);
        }
    }
    std::cout<<"POINTWISE_PAIRS_OK "<<types.size()*types.size()<<" pairs, exact forward/backward\n";
}
static void chainTailParity(CpuRuntime& cpu) {
    // Exercise paired lowering across tile boundaries and an unpaired tail.
    for(size_t depth=1;depth<=7;++depth) {
        std::vector<std::string> types;
        for(size_t i=0;i<depth;++i)types.push_back(i%2?"HardSwish":"LeakyReLU");
        auto layers=pointwise(types);const auto chain=pointers(layers);
        std::vector<float> input(Mimir::CpuFusion::tile_size*2+3),gradient(input.size(),.3f);
        for(size_t i=0;i<input.size();++i)input[i]=float(int(i%101)-50)*.125f;
        std::vector<std::vector<float>> reference,temporary,actual;
        for(const auto& layer:layers) {
            const auto& x=reference.empty()?input:reference.back();
            require(cpu.forwardLayer({&x},temporary,layer,true),"tail reference forward");
            reference.push_back(std::move(temporary[0]));
        }
        for(bool retain:{false,true}) {
            require(cpu.pointwiseFusionForward(chain,input,actual,retain,true),"tail fused forward");
            require(actual.size()==(retain?depth:1),"tail output count");
            for(size_t i=0;i<actual.size();++i)
                require(exact(actual[i],reference[retain?i:depth-1]),"tail forward parity");
        }
        std::vector<const std::vector<float>*> snapshots;
        for(size_t i=0;i<depth;++i)snapshots.push_back(i?&reference[i-1]:&input);
        auto expected=gradient;
        for(size_t i=depth;i-->0;) {
            require(cpu.backwardLayer({snapshots[i]},{&expected},temporary,layers[i],true),"tail reference backward");
            expected=std::move(temporary[0]);
        }
        std::vector<float> dx;
        require(cpu.pointwiseFusionBackward(chain,snapshots,gradient,dx),"tail fused backward");
        require(exact(dx,expected),"tail backward parity");
    }
    std::cout<<"PAIRED_KERNEL_TAILS_OK depths 1-7, retained/final-only, multiple tiles\n";
}
static void routingBoundaries(CpuRuntime& cpu) {
    auto& router=RuntimeRouter::instance();
    auto ls=pointwise({"ReLU","Tanh"});auto chain=pointers(ls);
    // Runtime-latency selection is covered by PlannerTest.FusionCalibration.
    // Raw kernels are checked independently of whether they win calibration.
    std::vector<std::vector<float>> outputs{{42}};
    std::vector<float> input{.3f,-.4f},dx{42};
    router.setRuntimes(nullptr,nullptr,nullptr,nullptr,&cpu);
    Layer reduce("reduce","Softmax",0);
    require(!router.dispatchPointwiseForward({&reduce},input,outputs,true),"reduction incorrectly entered elementwise kernel");
    input[0]=std::numeric_limits<float>::infinity();
    require(!router.dispatchPointwiseForward(chain,input,outputs,true),"non-finite input accepted");
    // Validate the vector classification path and its scalar tail without
    // touching caller-owned output on rejection.
    for(size_t position:{size_t(0),size_t(31),size_t(32),size_t(64)}) {
        input.assign(65,.25f);input[position]=std::numeric_limits<float>::quiet_NaN();outputs={{42}};
        require(!router.dispatchPointwiseForward(chain,input,outputs,true) && outputs==std::vector<std::vector<float>>{{42}},
                "non-finite SIMD lane or tail accepted");
    }
    std::cout<<"FUSION_ROUTING_BOUNDARIES_OK reductions, non-finite inputs\n";
}
static void modelParity() {
    for(const char* dtype:{"float32","float16"}) for(bool stochastic:{false,true}) {
        Model model;model.setDefaultDType(dtype);
        for(const char* type:{"Identity","Linear","LayerNorm","GELU","SiLU","Tanh","Sigmoid",
                "Add","Multiply","Dropout","Split","Identity"})model.push(type,type,0);
        auto& ls=model.getMutableLayers();
        for(size_t i=0;i<ls.size();++i){ls[i].inputs={i?"x":"__input__"};ls[i].output="x";}
        ls[1].in_features=4;ls[1].out_features=4;ls[1].use_bias=true;ls[1].params_count=20;
        ls[2].in_features=4;ls[2].params_count=8;
        ls[7].inputs={"x","side"};ls[8].inputs={"side","x"};
        ls[9].dropout_p=stochastic?.25f:0.f;
        ls[10].split_sizes={2,2};ls[10].num_splits=2;ls[10].output="parts";
        ls[11].inputs={"parts_1"};
        model.allocateParams();model.initializeWeights("xavier",42);
        std::vector<std::vector<float>> initial;
        for(const auto& l:ls)initial.emplace_back(l.getWeightsSize()?std::vector<float>(l.getWeights(),l.getWeights()+l.getWeightsSize()):std::vector<float>{});
        std::vector<std::vector<float>> expected;
        for(bool enabled:{false,true}) {
            fusion(enabled);
            for(size_t i=0;i<ls.size();++i)if(!initial[i].empty())std::copy(initial[i].begin(),initial[i].end(),ls[i].getWeights());
            Optimizer opt;opt.type=OptimizerType::ADAM;
            size_t value_index=0;
            auto check=[&](const std::vector<float>& value) {
                if(!enabled)expected.push_back(value);
                else require(exact(expected.at(value_index),value),"mixed graph parity at "+std::to_string(value_index));
                ++value_index;
            };
            for(int step=0;step<3;++step) {
                MimirRng::ScopedSeed seed(42+step);
                model.zeroGradients();
                const auto out=model.forwardPassNamed({{"__input__",{-.4f,.3f,.8f,-.1f}},{"side",{.2f,.4f,.1f,.5f}}},{},true);
                check(out);model.backwardPass({.3f,-.1f});check(model.getLastInputGradient());
                for(const auto& l:ls)check(l.grad_weights);
                model.optimizerStep(opt,.001f);
                for(const auto& l:ls)if(l.getWeightsSize())check(std::vector<float>(l.getWeights(),l.getWeights()+l.getWeightsSize()));
            }
            MimirRng::ScopedSeed seed(99);
            check(model.forwardPassNamed({{"__input__",{-.4f,.3f,.8f,-.1f}},{"side",{.2f,.4f,.1f,.5f}}},{},false));
        }
    }
    // In-place tensor names represent versions, not multiple live consumers.
    auto inplace=pointwise({"ReLU","Tanh","Sigmoid","Identity"});
    inplace.front().inputs={"__input__"};
    auto plan=Mimir::Planning::build_execution_plan_global(inplace,true);
    require(plan.skip_layer[1] && plan.skip_layer[3],"in-place chain not fused");
    inplace.emplace_back("side","Identity",0);inplace.back().inputs={"__input__"};inplace.back().output="side";
    // Named edges and early inference outputs must remain observable.
    Model named;
    named.push("seed","Identity",0);named.push("relu","ReLU",0);named.push("tanh","Tanh",0);
    auto& nodes=named.getMutableLayers();
    nodes[0].inputs={"__input__"};nodes[0].output="seed";
    nodes[1].inputs={"seed"};nodes[1].output="activated";
    nodes[2].inputs={"activated"};nodes[2].output="x";
    named.allocateParams();fusion(true);
    named.forwardPass(std::vector<float>{-1,2},false);
    require(exact(named.getTensor("seed"),{-1,2}) && exact(named.getTensor("activated"),{0,2}),"named intermediate lost");
    named.modelConfig["inference_output_tensor"]="seed";
    require(exact(named.forwardPass(std::vector<float>{-2,3},false),{-2,3}),"early inference output changed");
    std::cout<<"MIXED_GRAPH_FUSION_OK train/inference, F32/F16, masks, named side inputs, secondary split, Adam\n";
}
static std::vector<std::vector<float>> branchedTrainingParity(Model& model,const std::vector<float>& input,int only=-1) {
    auto& layers=model.getMutableLayers();
    std::vector<std::vector<float>> initial,reference;
    for(const auto& l:layers)initial.emplace_back(l.getWeightsSize()?std::vector<float>(l.getWeights(),l.getWeights()+l.getWeightsSize()):std::vector<float>{});
    for(bool enabled:{false,true}) {
        if(only>=0 && enabled!=(only!=0))continue;
        fusion(enabled);
        for(size_t i=0;i<layers.size();++i)if(!initial[i].empty())std::copy(initial[i].begin(),initial[i].end(),layers[i].getWeights());
        Optimizer optimizer;optimizer.type=OptimizerType::LION;
        size_t index=0;
        auto check=[&](const std::vector<float>& value) {
            if(!enabled || only>=0)reference.push_back(value);
            else require(exact(value,reference.at(index)),"branched training parity at "+std::to_string(index));
            ++index;
        };
        for(int step=0;step<3;++step) {
            MimirRng::ScopedSeed seed(42+step);
            model.zeroGradients();
            const auto output=model.forwardPass(input,true);check(output);
            const auto graph=model.runtimeConfiguration()->modelGraph();
            require(graph && (*graph)["fusion"]==enabled,"GraphSet fusion flag disagrees with requested training mode");
            if(enabled)require(!(*graph)["fusion_groups"].empty(),"branches disabled every fusion group");
            model.backwardPass(std::vector<float>(output.size(),.001f));check(model.getLastInputGradient());
            for(const auto& layer:layers)check(layer.grad_weights);
            model.optimizerStep(optimizer,.0001f);
            for(const auto& layer:layers)if(layer.getWeightsSize())check(std::vector<float>(layer.getWeights(),layer.getWeights()+layer.getWeightsSize()));
        }
    }
    return reference;
}
// CPU Reparameterize owns a private RNG. Stochastic off/on comparisons run
// in separate fresh processes so they start with identical RNG state.
static std::vector<std::vector<float>> branchedVae(bool stochastic,int only=-1) {
    auto vae=ModelArchitectures::create("vae_conv",{
        {"image_w",32},{"image_h",32},{"image_c",3},{"latent_w",2},{"latent_h",2},{"latent_c",32},
        {"base_channels",8},{"latent_spatial_context",true},{"latent_detail_path",true},{"latent_context_channels",32},
        {"stochastic_latent",stochastic},{"resnet",false},{"attention",false},{"enc_norm","groupnorm"},{"dec_norm","layernorm"},
        {"decoder_upsample","nearest_conv"},{"skip_connections_enabled",false},{"d_model",0},{"encoder_prior",false}});
    require(std::any_of(vae->getLayers().begin(),vae->getLayers().end(),[](const Layer& l){return l.requiresBranchComputation();}),"VAE reproduction has no branch metadata");
    vae->allocateParams();vae->initializeWeights("xavier",42);
    std::vector<float> input(32*32*3);for(size_t i=0;i<input.size();++i)input[i]=.3f*std::sin(float(i)*.1f);
    return branchedTrainingParity(*vae,input,only);
}
static void branchBoundaries() {
    Model legacy;
    for(const char* type:{"Identity","Linear","ReLU","Tanh","Identity","Identity","Identity","Sigmoid","Tanh"})
        legacy.push("node"+std::to_string(legacy.getLayers().size()),type,0);
    auto& ls=legacy.getMutableLayers();
    for(size_t i=0;i<ls.size();++i){ls[i].inputs={i?"x":"__input__"};ls[i].output="x";}
    ls[1].in_features=ls[1].out_features=2;ls[1].params_count=6;ls[1].use_bias=true;
    ls[6].branch_type=BranchType::RESIDUAL;ls[6].branch_sources={0};
    // Deliberately leave source 0 unflagged: implicit sources are boundaries too.
    auto plan=Mimir::Planning::build_execution_plan_global(ls,true);
    require(plan.fuse_chain_next[0]<0 && plan.fuse_chain_next[6]<0 && !plan.skip_layer[6],"legacy branch endpoints fused");
    require(plan.skip_layer[2] && plan.skip_layer[8],"independent chains around branch lost");
    legacy.allocateParams();legacy.initializeWeights("xavier",42);
    branchedTrainingParity(legacy,{-.3f,.7f});

    branchedVae(false);
    std::cout<<"BRANCHED_TRAIN_FUSION_OK legacy merges, VAEConv detail/context, exact outputs/gradients/Lion, GraphSet on\n";
}
static double median(std::function<void()> run,int repetitions) {
    for(int i=0;i<3;++i)run();
    std::vector<double> samples;
    for(int round=0;round<5;++round) {
        const auto start=std::chrono::steady_clock::now();for(int i=0;i<repetitions;++i)run();
        samples.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/repetitions);
    }
    std::sort(samples.begin(),samples.end());return samples[2];
}
static nlohmann::json benchmark() {
    nlohmann::json report;
    for(size_t size:{size_t(4096),size_t(262144)}) {
        Model model;for(const char* type:{"Identity","ReLU","HardSigmoid","HardSwish","LeakyReLU","ReLU"})model.push(type,type,0);
        auto& layers=model.getMutableLayers();layers[0].inputs={"__input__"};
        for(auto& l:layers)l.output="x";
        std::vector<float> x(size),g(size,.3f);for(size_t i=0;i<size;++i)x[i]=float(int(i%101)-50)*.1f;
        model.allocateParams();
        for(bool training:{false,true}) {
            double timing[2],cold[2];
            for(int enable=0;enable<2;++enable) {
                fusion(enable);
                auto run=[&]{model.forwardPassView(x,training);if(training)model.backwardPass(g);};
                const auto start=std::chrono::steady_clock::now();run();
                cold[enable]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
                timing[enable]=median(run,size<5000?100:10);
            }
            report[training?"train":"inference"][std::to_string(size)]={{"unfused_us",timing[0]},
                {"fused_us",timing[1]},{"speedup",timing[0]/timing[1]},
                {"unfused_cold_us",cold[0]},{"planned_cold_us",cold[1]}};
            auto& decisions=report[training?"train":"inference"][std::to_string(size)]["planner_decisions"];
            decisions=nlohmann::json::array();
            for(const auto& layer:layers)for(bool backward:{false,true}) {
                const auto* plan=Mimir::Planning::FusionCalibration::profile(&layer,backward);
                if(!plan)continue;
                nlohmann::json alternatives=nlohmann::json::array();
                for(const auto& candidate:plan->alternatives)alternatives.push_back({
                    {"runtime",candidate.runtime},{"seconds",candidate.seconds},{"bit_exact",candidate.exact}});
                decisions.push_back({{"layer",layer.name},{"backward",backward},{"measured",plan->measured},
                    {"fused",plan->fused},{"selected_runtime",plan->selected_runtime},
                    {"reason",plan->reason},{"alternatives",std::move(alternatives)}});
            }
        }
    }
    return report;
}
int main(int argc,char** argv) {
    CpuRuntime cpu;RuntimeConfig config;require(cpu.initialize(config),"CPU initialization");
    RuntimeRouter::instance().setRuntimes(nullptr,nullptr,nullptr,nullptr,&cpu);
    if(argc==4 && std::string(argv[1])=="--vae-branch-parity") {
        const auto values=branchedVae(true,std::string(argv[2])=="1"?1:0);
        std::ofstream(argv[3])<<nlohmann::json(values).dump();return 0;
    }
    nativeParity(cpu);chainTailParity(cpu);routingBoundaries(cpu);modelParity();branchBoundaries();
    nlohmann::json audit=nlohmann::json::array();
    for(int kind=0;kind<static_cast<int>(LayerType::UNKNOWN);++kind) {
        Layer l;l.type_enum=static_cast<LayerType>(kind);
        audit.push_back({{"layer",LayerRegistry::type_to_string(l.type_enum)},
            {"composable_consumer",Mimir::Planning::is_fusible_layer(l)||Mimir::Planning::is_fusible_split_layer(l)},
            {"cpu_pointwise_forward_backward",Mimir::CpuFusion::supports(l.type_enum)}});
    }
    std::cout<<"LAYER_FUSION_AUDIT_OK "<<audit.size()<<" registered layers\n";
    if(argc>1) {
        auto report=benchmark();report["audit"]=audit;report["backend"]="CPU";
        report["training_measurement"]="forward + backward; optimizer excluded";
        report["statistic"]="median of 5 samples after 3 warmups";
#ifdef _OPENMP
        report["threads"]=omp_get_max_threads();
#endif
        std::ofstream(argv[1])<<report.dump(2)<<'\n';std::cout<<report.dump(2)<<'\n';
    }
}
