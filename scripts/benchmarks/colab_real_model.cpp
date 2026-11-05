// Native public Model API counterpart of colab_real_worker.py.
#include "Model.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <cmath>
#include <numeric>
#include <sstream>
#include <unordered_map>
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
static double rss_mib(){
    std::ifstream f("/proc/self/status");std::string line;
    while(std::getline(f,line))if(line.rfind("VmRSS:",0)==0){std::istringstream value(line.substr(6));double kb=0;value>>kb;return kb/1024.;}
    return 0.;
}
static double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
void framework_log_write_file_only(const char*,size_t){}
void framework_log_write(const char*,size_t){}
template<class T>static std::vector<T> read(const std::string& path){
    std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("Missing file: "+path);
    const auto size=f.tellg();if(size<0 || size%sizeof(T))throw std::runtime_error("Invalid file: "+path);
    std::vector<T> v(static_cast<size_t>(size)/sizeof(T));f.seekg(0);f.read(reinterpret_cast<char*>(v.data()),size);
    if(!f)throw std::runtime_error("Cannot read: "+path);return v;
}
static Json run_architecture_benchmark(const Json& c,const std::string& backend){
    const Json options=c.value("mimir",Json::object());
    const bool allow_cpu=options.value("allow_cpu_fallback",backend=="cpu");
    setenv("MIMIR_DISABLE_CPU",backend=="cpu" || allow_cpu?"0":"1",1);
    const std::string planner=options.value("planner_mode",std::string("cost"));
    if(planner!="cost" && planner!="static" && planner!="legacy")throw std::runtime_error("Invalid planner mode");
    const bool planner_enabled=options.value("planner_enabled",true);
    setenv("MIMIR_ENABLE_PLANNER",planner_enabled?"1":"0",1);
    setenv("MIMIR_PLANNER_MODE",planner.c_str(),1);
    auto flag=[&](const char* environment,const char* option,bool fallback){
        setenv(environment,options.value(option,fallback)?"1":"0",1);
    };
    flag("MIMIR_PLANNER_FUSION","fusion",true);
    flag("MIMIR_ENABLE_FUSION","fusion",true);
    flag("MIMIR_ENABLE_FUSION_TRAIN","fusion_train",true);
    flag("MIMIR_PLANNER_BUFFER_REUSE","buffer_reuse",true);
    flag("MIMIR_PLANNER_DEVICE_RESIDENCY","device_residency",true);
    flag("MIMIR_RUNTIME_AUTOTUNE","runtime_autotune",true);
    flag("MIMIR_PLANNER_COST_MODEL","cost_model",true);
    flag("MIMIR_PLANNER_DUMP","planner_dump",false);

    Model::setFrameworkLogsSuppressed(true);
    const std::string architecture=c.at("architecture");
    const Json model_config=c.at("model_config");
    const auto init=Clock::now();
    auto model=ModelArchitectures::create(architecture,model_config);
    if(!model)throw std::runtime_error("Architecture factory returned null: "+architecture);
    model->allocateParams();
    model->initializeWeights("he",c.value("seed",1u));

    const size_t input_size=c.value("input_size",0u);
    const size_t token_count=c.value("token_count",0u);
    if((c.value("input_kind",std::string("float"))=="tokens" && token_count==0) ||
       (c.value("input_kind",std::string("float"))!="tokens" && input_size==0))
        throw std::runtime_error("Architecture input must not be empty");
    std::vector<float> input;
    for(size_t i=0;i<input_size;++i)
        input.push_back((static_cast<int>(i%31)-15)/15.f);
    std::vector<int> tokens;
    for(size_t i=0;i<token_count;++i)
        tokens.push_back(static_cast<int>((i*7)%std::max(1,c.value("vocab_size",1))));
    const auto forward=[&](Model* target){
        if(c.value("input_kind",std::string("float"))=="tokens"){
            std::unordered_map<std::string,std::vector<float>> floats;
            const int d_model=model_config.value("d_model",1);
            floats["mag"]=std::vector<float>(static_cast<size_t>(d_model),0.f);
            floats["mod"]=std::vector<float>(static_cast<size_t>(d_model),0.f);
            std::vector<float> token_values(tokens.begin(),tokens.end());
            floats["__input__"]=token_values;
            floats["x"]=std::move(token_values);
            return target->forwardPassNamed(floats,{{"seq",tokens}},false);
        }
        return target->forwardPass(input,false);
    };
    const double initialization_s=seconds(init);
    const auto cold=Clock::now();
    auto output=forward(model.get());
    const double cold_forward_ms=seconds(cold)*1000;
    const size_t expected=c.at("expected_output_size");
    if(output.size()!=expected)throw std::runtime_error("Wrong output size for "+architecture+
        ": expected "+std::to_string(expected)+", got "+std::to_string(output.size()));
    for(float value:output)if(!std::isfinite(value))throw std::runtime_error("Non-finite architecture output");
    std::vector<double> samples;
    const int warmup=c.value("warmup",3),iterations=c.value("iterations",15);
    if(warmup<0 || iterations<=0)throw std::runtime_error("Invalid architecture benchmark iteration count");
    for(int i=0;i<warmup;++i)output=forward(model.get());
    for(int i=0;i<iterations;++i){
        const auto tick=Clock::now();output=forward(model.get());
        samples.push_back(seconds(tick)*1000);
    }
    for(float value:output)if(!std::isfinite(value))throw std::runtime_error("Non-finite architecture output");
    const std::vector<float> measured_output=output;
    const size_t parameter_count=model->totalParamCount();
    Json inference_plan;
    if(auto snapshot=model->runtimeConfiguration()->modelGraph())inference_plan=*snapshot;
    model.reset();
    setenv("MIMIR_ENABLE_PLANNER","0",1);
    setenv("MIMIR_PLANNER_FUSION","0",1);
    setenv("MIMIR_ENABLE_FUSION","0",1);
    std::ostringstream audit;
    auto* old_cerr=std::cerr.rdbuf(audit.rdbuf());
    const char* trace=std::getenv("MIMIR_RUNTIME_TRACE");
    const bool had_trace=trace!=nullptr;
    const std::string previous_trace=trace?trace:"";
    const char* profile=std::getenv("MIMIR_LAYER_PROFILE");
    const bool had_profile=profile!=nullptr;
    const std::string previous_profile=profile?profile:"";
    auto restore_diagnostics=[&]{
        if(had_trace)setenv("MIMIR_RUNTIME_TRACE",previous_trace.c_str(),1);
        else unsetenv("MIMIR_RUNTIME_TRACE");
        if(had_profile)setenv("MIMIR_LAYER_PROFILE",previous_profile.c_str(),1);
        else unsetenv("MIMIR_LAYER_PROFILE");
    };
    try{
        setenv("MIMIR_RUNTIME_TRACE","1",1);
        setenv("MIMIR_LAYER_PROFILE","1",1);
        Model::setFrameworkLogsSuppressed(false);
        auto profile_model=ModelArchitectures::create(architecture,model_config);
        if(!profile_model)throw std::runtime_error("Diagnostic architecture factory returned null: "+architecture);
        profile_model->allocateParams();
        profile_model->initializeWeights("he",c.value("seed",1u));
        output=forward(profile_model.get());
        if(output.size()!=measured_output.size())
            throw std::runtime_error("Diagnostic profile output size differs from measured model");
        for(size_t i=0;i<output.size();++i)
            if(!std::isfinite(output[i]) ||
               std::fabs(output[i]-measured_output[i])>1e-3f+1e-3f*std::fabs(measured_output[i]))
                throw std::runtime_error("Planner-disabled diagnostic output differs from measured model");
        profile_model.reset();
        Model::setFrameworkLogsSuppressed(true);
        std::cerr.rdbuf(old_cerr);
        restore_diagnostics();
        setenv("MIMIR_ENABLE_PLANNER",planner_enabled?"1":"0",1);
        flag("MIMIR_PLANNER_FUSION","fusion",true);
        flag("MIMIR_ENABLE_FUSION","fusion",true);
    }catch(...){
        Model::setFrameworkLogsSuppressed(true);
        std::cerr.rdbuf(old_cerr);
        restore_diagnostics();
        setenv("MIMIR_ENABLE_PLANNER",planner_enabled?"1":"0",1);
        flag("MIMIR_PLANNER_FUSION","fusion",true);
        flag("MIMIR_ENABLE_FUSION","fusion",true);
        throw;
    }
    Json layer_profile=Json::array();
    std::istringstream audit_lines(audit.str());
    std::string audit_line;
    constexpr const char* profile_prefix="[layer-profile] ";
    while(std::getline(audit_lines,audit_line)){
        if(audit_line.rfind(profile_prefix,0)==0)
            layer_profile.push_back(Json::parse(audit_line.substr(std::char_traits<char>::length(profile_prefix))));
    }
    if(layer_profile.empty())throw std::runtime_error("Diagnostic forward produced no per-layer timing samples");
    std::sort(samples.begin(),samples.end());
    const double median=samples.size()%2?samples[samples.size()/2]:
        (samples[samples.size()/2-1]+samples[samples.size()/2])/2.;
    const auto percentile=[&](double p){
        const size_t index=std::min(samples.size()-1,
            static_cast<size_t>(std::ceil(p*samples.size())-1));
        return samples.at(index);
    };
    Json report={{"framework","mimir"},{"backend",backend},{"case",c.at("case")},
        {"architecture",architecture},{"status","measured"},
        {"mode","ModelArchitectures::create + Model::forwardPass"},
        {"version",c.value("mimir_version",std::string("see protocol"))},
        {"device","target="+backend+((backend=="cpu" || allow_cpu)?"; CPU routes allowed by planner":"; CPU routes disabled")},
        {"initialization_s",initialization_s},{"setup_s",initialization_s},
        {"cold_forward_ms",cold_forward_ms},
        {"forward_samples_ms",samples},{"forward_median_ms",median},
        {"forward_p95_ms",percentile(.95)},{"input_size",c.value("input_size",0u)},
        {"token_count",c.value("token_count",0u)},{"output_size",measured_output.size()},
        {"parameter_count",parameter_count},{"first_output",measured_output},
        {"optimizations_requested",options},{"runtime_audit",audit.str()},
        {"layer_profile_scope","separate planner-disabled diagnostic forward; per-layer wall time includes dispatch/materialization"},
        {"layer_profile_optimization","planner and fusion disabled; output checked against measured forward"},
        {"layer_profile",layer_profile},
        {"routing_policy",backend=="cpu"?"CPU target selected":
            (allow_cpu?"CPU fallback allowed":"CPU routes disabled")}};
    if(!inference_plan.is_null())report["inference_plan"]=std::move(inference_plan);
    return report;
}
int main(int argc,char** argv){
 if(argc!=4)return 2;
 const auto whole=Clock::now();Json report={{"framework","mimir"},{"backend",argv[1]},{"status","error"}};
 try{
    Json c;std::ifstream(argv[2])>>c;const std::string backend=argv[1];
    if(c.value("benchmark_kind",std::string())=="architecture"){
        report=run_architecture_benchmark(c,backend);
        report["worker_wall_s"]=seconds(whole);
        std::ofstream(argv[3])<<report.dump(2);
        return 0;
    }
    const std::string dir=c.at("data_dir");
    const int batch=c.at("batch"),hidden=c.at("hidden"),steps=c.at("steps");const float lr=c.at("lr");
    if(batch<=0 || hidden<=0 || steps<=0)throw std::runtime_error("Invalid dimensions");
    // Use the production planner and its supported optimizations, rather than
    // forcing every layer onto the requested accelerator independently.
    const Json options=c.value("mimir",Json::object());
    const std::string planner=options.value("planner_mode",std::string("cost"));
    if(planner!="cost" && planner!="static" && planner!="legacy")throw std::runtime_error("Invalid planner mode");
    const bool planner_enabled=options.value("planner_enabled",true);
    auto flag=[&](const char* environment,const char* option,bool fallback){
        setenv(environment,options.value(option,fallback)?"1":"0",1);
    };
    const bool allow_cpu=options.value("allow_cpu_fallback",true);
    setenv("MIMIR_DISABLE_CPU",backend=="cpu" || allow_cpu?"0":"1",1);
    setenv("MIMIR_ENABLE_PLANNER",planner_enabled?"1":"0",1);setenv("MIMIR_PLANNER_MODE",planner.c_str(),1);
    flag("MIMIR_PLANNER_FUSION","fusion",true);flag("MIMIR_ENABLE_FUSION","fusion",true);
    flag("MIMIR_ENABLE_FUSION_TRAIN","fusion_train",true);
    flag("MIMIR_PLANNER_BUFFER_REUSE","buffer_reuse",true);
    flag("MIMIR_PLANNER_DEVICE_RESIDENCY","device_residency",true);
    flag("MIMIR_RUNTIME_AUTOTUNE","runtime_autotune",true);
    flag("MIMIR_PLANNER_COST_MODEL","cost_model",true);
    flag("MIMIR_PLANNER_DUMP","planner_dump",false);
    report["optimizations_requested"]={{"planner_enabled",planner_enabled},{"planner_mode",planner},
        {"fusion",options.value("fusion",true)},{"fusion_train",options.value("fusion_train",true)},
        {"buffer_reuse",options.value("buffer_reuse",true)},
        {"runtime_autotune",options.value("runtime_autotune",true)},
        {"cost_model",options.value("cost_model",true)},
        {"planner_dump",options.value("planner_dump",false)},
        {"device_residency",options.value("device_residency",true)},{"allow_cpu_fallback",allow_cpu}};
    Model::setFrameworkLogsSuppressed(true);
    const auto data_started=Clock::now();
    std::vector<std::vector<float>> initial;
    for(int i=0;i<4;++i)initial.push_back(read<float>(dir+"/weights-"+std::to_string(hidden)+"-"+std::to_string(i)+".f32"));
    struct Data{std::vector<uint8_t> x,y;};std::map<std::string,Data> data;
    for(const std::string split:{"train","valid","test"}){
        auto& d=data[split];d.x=read<uint8_t>(dir+"/"+split+".images.u8");d.y=read<uint8_t>(dir+"/"+split+".labels.u8");
        if(d.y.empty() || d.x.size()!=d.y.size()*784)throw std::runtime_error("Dataset shape mismatch");
    }
    report["data_allocation_s"]=seconds(data_started);
    const auto initialization=Clock::now();
    Model model;model.modelConfig={{"loss","mse"},{"grad_clip_norm",0}};
    model.push("hidden","Linear",hidden*784+hidden);
    model.push("relu","ReLU",0);model.push("logits","Linear",10*hidden+10);
    auto& layers=model.getMutableLayers();
    layers[0].inputs={"__input__"};layers[0].output="hidden";
    layers[0].in_features=784;layers[0].out_features=hidden;layers[0].seq_len=batch;layers[0].use_bias=true;
    layers[1].inputs={"hidden"};layers[1].output="activation";
    layers[2].inputs={"activation"};layers[2].output="x";
    layers[2].in_features=hidden;layers[2].out_features=10;layers[2].seq_len=batch;layers[2].use_bias=true;
    const auto allocation=Clock::now();model.allocateParams();
    report["parameter_allocation_s"]=seconds(allocation);
    Optimizer opt;const std::string optimizer=c.value("optimizer",std::string("sgd"));
    if(optimizer!="sgd" && optimizer!="adam" && optimizer!="adamw")throw std::runtime_error("Invalid optimizer");
    opt.type=optimizer=="sgd"?OptimizerType::SGD:optimizer=="adam"?OptimizerType::ADAM:OptimizerType::ADAMW;opt.initial_lr=lr;opt.min_lr=lr;
    opt.decay_strategy=LRDecayStrategy::NONE;opt.weight_decay=c.value("weight_decay",0.f);opt.warmup_steps=0;
    auto reset=[&]{
        for(int block=0;block<2;++block){auto& l=layers[block==0?0:2];
            if(l.getWeightsSize()!=initial[2*block].size()+initial[2*block+1].size())throw std::runtime_error("Parameter shape mismatch");
            std::copy(initial[2*block].begin(),initial[2*block].end(),l.getWeights());
            std::copy(initial[2*block+1].begin(),initial[2*block+1].end(),l.getWeights()+initial[2*block].size());
        }opt.step=0;opt.mv_by_param_ptr.clear();opt.m.clear();opt.v.clear();
    };
    auto probe=[&]{
        std::vector<float> flat;
        for(int i:{0,2})flat.insert(flat.end(),layers[i].getWeights(),layers[i].getWeights()+layers[i].getWeightsSize());
        std::vector<float> result;for(size_t index:c.at("probe_indices").get<std::vector<size_t>>())result.push_back(flat.at(index));return result;
    };
    std::vector<float> x(batch*784),y(batch*10);
    auto get_batch=[&](const std::string& split,size_t offset){
        const auto& d=data.at(split);std::fill(y.begin(),y.end(),0.f);
        for(int row=0;row<batch;++row){const size_t index=(offset+row)%d.y.size();
            if(d.y[index]>=10)throw std::runtime_error("Invalid label");y[row*10+d.y[index]]=1.f;
            for(int j=0;j<784;++j)x[row*784+j]=d.x[index*784+j]/255.f;
        }
    };
    auto train=[&](){
        Model::TrainStepRequest request;request.float_inputs["__input__"]=&x;request.target=&y;
        request.optimizer=&opt;request.learning_rate=lr;request.capture_prediction=false;
        auto result=model.trainStep(request);if(!result)throw std::runtime_error("Model.trainStep unavailable");
        if(!std::isfinite(result->loss))throw std::runtime_error("Non-finite loss");return *result;
    };
    report["initialization_s"]=seconds(initialization);
    report["rss_after_initialization_mib"]=rss_mib();
    // Native public model tensors reside on host; accelerator scratch is lazy
    // and therefore included in the first synchronized trainStep.
    {
        const double before=rss_mib();const auto tick=Clock::now();std::vector<float> probe_allocation(4*1024*1024,1.f);
        volatile float touched=probe_allocation.back();(void)touched;
        report["allocation_probe_s"]=seconds(tick);report["allocation_probe_bytes"]=16*1024*1024;
        report["allocation_probe_rss_delta_mib"]=rss_mib()-before;
        report["allocation_probe_device"]="host; lazy device scratch included in cold_step_ms";
    }
    reset();get_batch("train",0);auto cold=Clock::now();auto first=train();
    report["cold_step_ms"]=seconds(cold)*1000;report["first_loss"]=first.loss;
    report["first_prediction"]=model.getTensor("x");report["first_parameter_probe"]=probe();
    if(auto snapshot=model.runtimeConfiguration()->modelGraph())report["training_plan"]=*snapshot;
    reset();cold=Clock::now();model.forwardPass(x,false);report["cold_predict_ms"]=seconds(cold)*1000;
    if(auto snapshot=model.runtimeConfiguration()->modelGraph())report["inference_plan"]=*snapshot;
    // Route diagnostics are deliberately outside the timed training/evaluation.
    std::ostringstream audit;
    auto* old_cerr=std::cerr.rdbuf(audit.rdbuf());
    const char* trace=std::getenv("MIMIR_RUNTIME_TRACE");
    const bool had_trace=trace!=nullptr;
    const std::string previous_trace=trace?trace:"";
    auto restore_trace=[&]{if(had_trace)setenv("MIMIR_RUNTIME_TRACE",previous_trace.c_str(),1);else unsetenv("MIMIR_RUNTIME_TRACE");};
    try {
        setenv("MIMIR_RUNTIME_TRACE","1",1);
        Model::setFrameworkLogsSuppressed(false);
        train();model.forwardPass(x,false);
        Model::setFrameworkLogsSuppressed(true);std::cerr.rdbuf(old_cerr);restore_trace();
    } catch(...) {
        Model::setFrameworkLogsSuppressed(true);std::cerr.rdbuf(old_cerr);restore_trace();throw;
    }
    report["runtime_audit"]=audit.str();
    std::ofstream(std::string(argv[3])+".runtime-audit.log")<<audit.str();
    reset();

    auto evaluate=[&](const std::string& split){
        const auto& d=data.at(split);size_t correct=0;double mse=0;std::vector<double> samples;
        for(size_t offset=0;offset<d.y.size();offset+=batch){
            auto tick=Clock::now();get_batch(split,offset);const auto out=model.forwardPass(x,false);
            samples.push_back(seconds(tick)*1000);
            if(out.size()!=static_cast<size_t>(batch*10))throw std::runtime_error("Wrong prediction shape");
            const size_t count=std::min<size_t>(batch,d.y.size()-offset);
            for(size_t row=0;row<count;++row){
                auto begin=out.begin()+row*10;const int predicted=std::max_element(begin,begin+10)-begin;
                correct+=predicted==d.y[offset+row];
                for(int j=0;j<10;++j){if(!std::isfinite(begin[j]))throw std::runtime_error("Non-finite predictions");
                    const double diff=begin[j]-y[row*10+j];mse+=diff*diff;}
            }
        }
        return Json{{"accuracy",double(correct)/d.y.size()},{"mse",mse/(d.y.size()*10)},
            {"samples_ms",samples},{"examples",d.y.size()}};
    };
    const int eval_every=c.at("eval_every");const double target=c.at("target_accuracy");
    Json history=Json::array(),goal=nullptr;std::vector<double> samples,losses;
    auto started=Clock::now();
    for(int step=0;step<steps;++step){
        auto tick=Clock::now();get_batch("train",size_t(step)*batch);const auto result=train();
        samples.push_back(seconds(tick)*1000);losses.push_back(result.loss);
        if((step+1)%eval_every==0 || step+1==steps){
            const auto valid=evaluate("valid");const double elapsed=seconds(started);
            history.push_back({{"step",step+1},{"elapsed_s",elapsed},{"accuracy",valid.at("accuracy")},{"mse",valid.at("mse")}});
            if(goal.is_null() && valid.at("accuracy").get<double>()>=target)goal=elapsed;
            Json progress={{"status","running"},{"steps_completed",step+1},{"history",history}};
            std::ofstream(std::string(argv[3])+".progress.json")<<progress.dump();
        }
    }
    report["train_wall_s"]=seconds(started);report["train_step_samples_ms"]=samples;report["train_losses"]=losses;
    report["history"]=history;report["time_to_goal_s"]=goal;report["steps_completed"]=steps;
    report["examples_trained"]=size_t(steps)*batch;
    started=Clock::now();report["test"]=evaluate("test");report["inference_wall_s"]=seconds(started);
    report["final_parameter_probe"]=probe();report["case"]=c.at("case");report["status"]="measured";
    report["mode"]="Model.trainStep + Model.forwardPass; planner="+planner+"; production optimizations (see plans/audit)";
    report["device"]="target="+backend+(allow_cpu?"; CPU routes allowed by planner":"; CPU routes disabled");report["optimizer"]=optimizer;
    report["reconstruction_loss_runtime_counts"]=RuntimeRouter::instance().reconstructionLossDispatchCounts();
    report["optimizer_runtime_counts"]=RuntimeRouter::instance().optimizerDispatchCounts();
    report["version"]=c.value("mimir_version",std::string("see protocol"));
 }catch(const std::exception& e){report["error"]=e.what();}
 report["worker_wall_s"]=seconds(whole);std::ofstream(argv[3])<<report.dump(2);
 return 0;
}
