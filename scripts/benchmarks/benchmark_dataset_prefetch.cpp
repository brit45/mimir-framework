// Synthetic native dataset -> normalization -> Model::trainStep pipeline.
// Args: look-ahead (0=sync), steps (24), target width/height (128), features (512).
// Run through run_mimir.sh; MIMIR_BENCH_VALUES enables the final binary dump.
#include "DatasetPrefetcher.hpp"
#include "Model.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "runtimes/cpu/SimdDispatch.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#if defined(__linux__)
#include <sys/resource.h>
#endif

using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
static size_t peakRssKiB() {
#if defined(__linux__)
    rusage usage{};if(getrusage(RUSAGE_SELF,&usage)==0)return size_t(usage.ru_maxrss);
#endif
    return 0;
}
static json summary(std::vector<double> samples) {
    const auto raw=samples;std::sort(samples.begin(),samples.end());
    return {{"median_ms",samples[samples.size()/2]},{"total_ms",std::accumulate(samples.begin(),samples.end(),0.)},{"samples_ms",raw}};
}
int main(int argc,char** argv) {
    try {
        const int ahead=argc>1?std::stoi(argv[1]):8,steps=argc>2?std::stoi(argv[2]):24,side=argc>3?std::stoi(argv[3]):128;
        const int features=argc>4?std::stoi(argv[4]):512;
        if(ahead<0 || ahead>256 || steps<1 || steps>2000 || side<16 || side>512 || side%16 ||
            features<1 || features>2048 || (size_t(side)*side*3)%size_t(features))
            throw std::invalid_argument("ahead [0,256], steps [1,2000], side [16,512] multiple of 16, features [1,2048] dividing side*side*3");
        Model::setFrameworkLogsSuppressed(true);
        const auto root=fs::temp_directory_path()/("mimir_dataset_bench_"+std::to_string(Clock::now().time_since_epoch().count()));
        fs::create_directories(root);
        struct Cleanup {fs::path path;~Cleanup(){std::error_code error;fs::remove_all(path,error);}}cleanup{root};
        std::vector<DatasetItem> items(16);
        for(size_t i=0;i<items.size();++i) {
            items[i].image_file=(root/(std::to_string(i)+".ppm")).string();
            std::ofstream out(items[i].image_file,std::ios::binary);out<<"P6\n"<<side*2<<' '<<side*2<<"\n255\n";
            for(int p=0;p<side*side*4*3;++p)out.put(char((size_t(p)*17+i*31)%256));
        }
        Model model;model.setDefaultDType("float32");model.modelConfig={{"loss","mse"}};
        model.push("linear","Linear",size_t(features)*features+features);auto& layer=model.getMutableLayers().back();
        layer.in_features=layer.out_features=features;layer.inputs={"__input__"};
        model.push("activation","SiLU",0);model.allocateParams();
        for(auto& value:model.getMutableLayers())if(auto* weights=value.getWeights())
            for(size_t i=0;i<value.getWeightsSize();++i)weights[i]=float(int(i%19)-9)*.002f;
        std::vector<float> input(size_t(side)*side*3),target(input.size(),.2f);
        Optimizer optimizer;optimizer.type=OptimizerType::ADAMW;optimizer.weight_decay=0;
        Model::TrainStepRequest request;request.float_inputs["__input__"]=&input;request.target=&target;
        request.optimizer=&optimizer;request.learning_rate=.001f;request.capture_prediction=false;
        std::vector<size_t> order;for(int i=0;i<steps;++i)order.push_back(size_t((i*7+3)%16));
        DatasetPrefetcher::Options options;options.width=options.height=side;options.ahead=size_t(ahead);
        DatasetPrefetcher queue(items,options);
        std::vector<double> loading,training,batches;
        uint64_t input_hash=1469598103934665603ULL;float final_loss=0;
        const auto total_start=Clock::now();queue.setOrder(order,DatasetPrefetcher::Format::HWC);
        for(size_t index:order) {
            const auto batch_start=Clock::now();
            {
            const auto load_start=Clock::now();
            auto sample=queue.acquire(index,DatasetPrefetcher::Format::HWC);
            auto& item=sample?sample->item:items[index];
            DatasetItemUnloadGuard release(item,!sample);
            if(!sample && !item.loadImageRGB(side,side))throw std::runtime_error("dataset decode failed");
            if(sample && sample->input.size()==input.size())input.swap(sample->input);
            else {
                if(item.img.size()!=input.size())throw std::runtime_error("invalid input size");
                for(size_t i=0;i<input.size();++i)input[i]=float(double(item.img[i])/127.5-1.0);
            }
            loading.push_back(elapsed(load_start));
            // Constant-size order/content probes avoid adding a full-buffer
            // hash pass that would artificially hide decoder time.
            for(size_t probe:{size_t(0),input.size()/2,input.size()-1}) {
                const auto* bytes=reinterpret_cast<const unsigned char*>(&input[probe]);
                for(size_t i=0;i<sizeof(float);++i){input_hash^=bytes[i];input_hash*=1099511628211ULL;}
            }
            const auto train_start=Clock::now();auto result=model.trainStep(request);
            if(!result || !std::isfinite(result->loss))throw std::runtime_error("training failed");
            final_loss=result->loss;training.push_back(elapsed(train_start));
            }
            // Include synchronous unloading or asynchronous lease handoff.
            batches.push_back(elapsed(batch_start));
        }
        const double total_ms=elapsed(total_start);queue.setOrder({});queue.stop();const auto stats=queue.stats();
        if(const char* path=std::getenv("MIMIR_BENCH_VALUES")) {
            std::ofstream dump(path,std::ios::binary);const auto& prediction=model.forwardPassView(input,false);
            dump.write(reinterpret_cast<const char*>(prediction.data()),std::streamsize(prediction.size()*sizeof(float)));
            for(const auto& value:model.getLayers())if(value.getWeights())
                dump.write(reinterpret_cast<const char*>(value.getWeights()),std::streamsize(value.getWeightsSize()*sizeof(float)));
            if(!dump)throw std::runtime_error("benchmark dump failed");
        }
        std::cout<<json{{"ahead",ahead},{"steps",steps},{"width",side},{"height",side},{"features",features},{"total_ms",total_ms},
            {"load",summary(loading)},{"training",summary(training)},{"batch",summary(batches)},
            {"input_probe_hash",input_hash},{"final_loss",final_loss},
            {"cpu_isa",RuntimeSimd::isaName(RuntimeSimd::selectedIsa())},{"peak_rss_kib",peakRssKiB()},
            {"optimizer_backends",RuntimeRouter::instance().optimizerDispatchCounts()},
            {"loss_backends",RuntimeRouter::instance().reconstructionLossDispatchCounts()},
            {"prefetch",{{"hits",stats.hits},{"waits",stats.waits},{"prepared",stats.prepared},{"buffer_growths",stats.buffer_growths},
                {"peak_bytes",stats.peak_bytes},{"prepare_ms",stats.prepare_ms},{"wait_ms",stats.wait_ms}}}}.dump()<<'\n';
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
