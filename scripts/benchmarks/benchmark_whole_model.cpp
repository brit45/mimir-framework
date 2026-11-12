// Bounded whole-model latency probe, using the unchanged public training API.
// cmake --build build --target mimir_bench_whole_model
// CPU example (set these environment variables before running):
// OMP_NUM_THREADS=1 MIMIR_DISABLE_VULKAN=1 MIMIR_DISABLE_OPENCL=1
// ./build/Tests/mimir_bench_whole_model float32 200 2
// Optional trailing arguments: features (64), batch (16), layer/SiLU pairs (8),
// layer type (Linear or SelfAttention, default Linear).
// Optimizer IDs: 0 SGD, 1 Adam, 2 AdamW, 3 Lion, 4 Adafactor,
// 5 RAdam, 6 NAdam, 7 RMSProp, 8 LAMB (default: SGD).
#include "Model.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "Planning/ExecutionCalibration.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <new>
#if defined(__linux__)
#include <sys/resource.h>
#endif
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
namespace {
thread_local bool count_allocations=false;
thread_local size_t allocation_count=0,allocation_bytes=0;
size_t peakRssKiB() {
#if defined(__linux__)
    rusage usage{};return getrusage(RUSAGE_SELF,&usage)==0?size_t(usage.ru_maxrss):0;
#else
    return 0;
#endif
}
}
void* operator new(size_t size) {
    if(void* p=std::malloc(size?size:1)) {
        if(count_allocations){++allocation_count;allocation_bytes+=size;}
        return p;
    }
    throw std::bad_alloc();
}
void* operator new[](size_t n){return ::operator new(n);}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete[](void* p) noexcept {std::free(p);}
void operator delete(void* p,size_t) noexcept {std::free(p);}
void operator delete[](void* p,size_t) noexcept {std::free(p);}


int main(int argc, char** argv) {
    try {
        const std::string dtype = argc > 1 ? argv[1] : "float32";
        const int steps = argc > 2 ? std::stoi(argv[2]) : 200;
        if (steps < 1 || steps > 10000) throw std::invalid_argument("steps must be in [1,10000]");
        const int features=argc>4?std::stoi(argv[4]):64;
        const int batch=argc>5?std::stoi(argv[5]):16;
        const int pairs=argc>6?std::stoi(argv[6]):8;
        const std::string layer_type=argc>7?argv[7]:"Linear";
        if(layer_type!="Linear" && layer_type!="SelfAttention")
            throw std::invalid_argument("layer type must be Linear or SelfAttention");
        if(features<1 || features>4096 || batch<1 || batch>1024 || pairs<1 || pairs>32)
            throw std::invalid_argument("features must be in [1,4096], batch in [1,1024], pairs in [1,32]");
        Model::setFrameworkLogsSuppressed(true);
        const auto initialize_start=std::chrono::steady_clock::now();
        Model model;
        model.setDefaultDType(dtype);
        model.modelConfig = {{"loss", "mse"}};
        for (int i = 0; i < pairs; ++i) {
            const bool attention=layer_type=="SelfAttention";
            model.push("fc" + std::to_string(i),layer_type,
                attention?4*features*features:features*features+features);
            auto& layer = model.getMutableLayers().back();
            layer.in_features = layer.out_features = features;
            layer.seq_len = batch;
            layer.use_bias = !attention;
            if(attention) {
                layer.embed_dim=features;
                layer.num_heads=features%8==0?8:1;
            }
            model.push("act" + std::to_string(i), "SiLU", 0);
        }
        model.getMutableLayers()[0].inputs = {"__input__"};
        model.allocateParams();
        for (auto& layer : model.getMutableLayers()) {
            auto* weights = layer.getWeights();
            if (weights) for (size_t i = 0; i < layer.getWeightsSize(); ++i)
                weights[i] = float(int(i % 19)-9)*0.002f;
        }
        const double initialize_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-initialize_start).count();
        std::vector<float> input(batch*features, 0.3f), target(input.size(), 0.2f);
        Optimizer optimizer;
        const int optimizer_kind = argc > 3 ? std::stoi(argv[3]) : 0;
        if (optimizer_kind < 0 || optimizer_kind > 8) throw std::invalid_argument("optimizer must be in [0,8]");
        optimizer.type = static_cast<OptimizerType>(optimizer_kind);
        optimizer.weight_decay = 0;
        Model::TrainStepRequest request;
        request.float_inputs["__input__"] = &input;
        request.target = &target;
        request.optimizer = &optimizer;
        request.learning_rate = 0.001f;
        request.capture_prediction = false;
        auto train = [&]() {
            auto result = model.trainStep(request);
            if (!result || !std::isfinite(result->loss)) throw std::runtime_error("Invalid training result");
            return result->loss;
        };
        using Clock = std::chrono::steady_clock;
        const auto cold_start=Clock::now();train();
        const double cold_trainstep_ms=std::chrono::duration<double,std::milli>(Clock::now()-cold_start).count();
        for (int i = 1; i < 10; ++i) train();
        auto elapsed = [](Clock::time_point start) {
            return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        };
        auto measure = [&](auto operation,auto prepare) {
            std::vector<double> samples;
            samples.reserve(steps);
            allocation_count=allocation_bytes=0;
            for (int i = 0; i < steps; ++i) {
                prepare();
                auto start = Clock::now();
                count_allocations=true;
                try {operation();}catch(...){count_allocations=false;throw;}
                count_allocations=false;
                samples.push_back(elapsed(start));
            }
            const auto raw_samples=samples;
            std::sort(samples.begin(), samples.end());
            return json{{"median_ms",samples[samples.size()/2]},
                        {"p95_ms",samples[std::min(samples.size()-1,samples.size()*95/100)]},{"samples_ms",raw_samples},
                        {"new_allocations_per_call",double(allocation_count)/steps},
                        {"new_bytes_per_call",double(allocation_bytes)/steps}};
        };
        float loss = 0;
        const auto training = measure([&]() { loss = train(); },[]{});
        const auto cold_inference_start=Clock::now();
        (void)model.forwardPass(input,false);
        const double cold_inference_ms=elapsed(cold_inference_start);
        const auto inference = measure([&]() { (void)model.forwardPass(input, false); },[]{});
        // Keep the public trainStep above as the main throughput measurement.
        // These synchronized stages locate overhead without timing Lua copies.
        model.zeroGradients();
        auto forward_stage=measure([&] {(void)model.forwardPassView(input,true);},[&]{model.zeroGradients();});
        const auto& prediction=model.forwardPassView(input,true);
        std::vector<float> gradient(prediction.size());
        for(size_t i=0;i<gradient.size();++i)
            gradient[i]=2.f*(prediction[i]-target[i])/float(gradient.size());
        auto backward_stage=measure([&] {model.backwardPass(gradient);},[&]{model.zeroGradients();(void)model.forwardPassView(input,true);});
        auto optimizer_stage=measure([&] {model.optimizerStep(optimizer,request.learning_rate);},[]{});
        // Capture actual routed choices before the optional inference dump can
        // publish another profile. OMP_NUM_THREADS alone is not a worker choice.
        json execution_profiles=json::array();
        for(const auto& layer:model.getLayers()) {
            auto choices=[&](const auto& profiles) {
                json rows=json::array();
                const auto at=profiles.find(&layer);
                if(at!=profiles.end())for(const auto& cost:at->second)
                    rows.push_back({{"runtime",cost.runtime->name()},{"cpu_workers",cost.cpu_workers},
                        {"median_host_ms",cost.seconds*1000.0}});
                return rows;
            };
            execution_profiles.push_back({{"name",layer.name},{"type",layer.type},
                {"forward",choices(Mimir::Planning::Calibration::forward_profiles)},
                {"backward",choices(Mimir::Planning::Calibration::backward_profiles)}});
        }
        std::ofstream values;
        if(const char* path=std::getenv("MIMIR_BENCH_VALUES")) {values.open(path,std::ios::binary);if(!values)throw std::runtime_error("Cannot open benchmark value dump");}
        if(values.is_open()) {
            const auto& output=model.forwardPassView(input,false);
            values.write(reinterpret_cast<const char*>(output.data()),std::streamsize(output.size()*sizeof(float)));
        }
        double checksum = 0;
        for (const auto& layer : model.getLayers())
            if (layer.getWeights()) {
                if(values.is_open())values.write(reinterpret_cast<const char*>(layer.getWeights()),std::streamsize(layer.getWeightsSize()*sizeof(float)));
                for (size_t i = 0; i < layer.getWeightsSize(); ++i)
                    checksum += double(i+1)*layer.getWeights()[i];
            }
        std::cout << json{{"dtype",dtype},{"steps",steps},{"layers",pairs*2},{"batch",batch},{"features",features},{"layer_type",layer_type},
            {"initialize_ms",initialize_ms},{"cold_trainstep_ms",cold_trainstep_ms},{"cold_inference_ms",cold_inference_ms},{"peak_rss_kib",peakRssKiB()},{"training",training},{"inference",inference},{"final_loss",loss},
            {"parameter_checksum",checksum},{"forward_stage",forward_stage},
            {"backward_stage",backward_stage},{"optimizer_stage",optimizer_stage},
            {"runtime_autotune",Mimir::Planning::Calibration::enabled()},{"execution_profiles",execution_profiles},
            {"allocation_scope","calling-thread C++ new/new[]; excludes malloc/device allocations"},
            {"optimizer_kind",optimizer_kind},{"optimizer_backends",RuntimeRouter::instance().optimizerDispatchCounts()},
            {"loss_backends",RuntimeRouter::instance().reconstructionLossDispatchCounts()}}.dump() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
