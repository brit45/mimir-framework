#include "Planning/ExecutionCalibration.hpp"
// Late binding uses actual host-to-host operations: allocations, device transfers
// and synchronization are part of each sample. No stochastic/stateful op is replayed.
#include "Layers.hpp"
#include "MemoryGuard.hpp"
#include "runtimes/LayerOps.hpp"
#include "runtimes/AbstractRuntime.hpp"
#include "runtimes/FiniteCheck.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace Mimir::Planning::Calibration {
struct Entry { std::vector<Measurement> costs; size_t uses=0; };
thread_local std::unordered_map<size_t,Entry> cache;
thread_local std::unordered_map<const Layer*,std::vector<Measurement>> forward_profiles;
thread_local std::unordered_map<const Layer*,std::vector<Measurement>> backward_profiles;
void mix(size_t& h,size_t v){h^=v+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2);}
bool enabled(){
    // Graph construction/fusion and host-to-host backend selection are
    // independent. Disabling the graph planner must not blindly prefer a GPU
    // whose transfers cost far more than the CPU operation.
    const char* value=std::getenv("MIMIR_RUNTIME_AUTOTUNE");
    if(!value)value=std::getenv("MIMIR_PLANNER_AUTOTUNE"); // legacy tuning switch
    return !value || std::string(value)!="0";
}
size_t configuration(const AbstractRuntime& rt){
    const auto& c=rt.config();size_t h=0;
    for(auto v:{c.device_index,int(c.disabled),int(c.linear_enabled),c.linear_min_ops,int(c.conv_enabled),c.conv_min_ops,int(c.norm_enabled),c.norm_min_elements,int(c.attention_enabled),c.attention_min_ops})mix(h,size_t(v));
    if(std::string(rt.name())=="CPU") {
        mix(h,RuntimeLayerOps::hardwareAccelerationEnabled());
        mix(h,ExecutionPolicy::cacheBytes());
        mix(h,MemoryGuard::instance().getLimit());
        if(const char* scratch=std::getenv("MIMIR_PLANNER_SCRATCH_BYTES"))mix(h,std::hash<std::string>{}(scratch));
    }
    return h;
}
bool replayable(LayerType type){
    switch(type){
        case LayerType::Linear:case LayerType::MatMul:case LayerType::BatchMatMul:
        case LayerType::Conv2d:case LayerType::LayerNorm:case LayerType::RMSNorm:
        case LayerType::GroupNorm:case LayerType::InstanceNorm2d:
        case LayerType::ReLU:case LayerType::LeakyReLU:case LayerType::Sigmoid:case LayerType::Tanh:
        case LayerType::SiLU:case LayerType::GELU:case LayerType::Softplus:case LayerType::Mish:
        case LayerType::HardSigmoid:case LayerType::HardSwish:
        case LayerType::Add:case LayerType::Subtract:case LayerType::Multiply:case LayerType::Divide:
        case LayerType::Identity:case LayerType::Flatten:case LayerType::Reshape:
        case LayerType::Softmax:case LayerType::LogSoftmax:case LayerType::GEGLU:
        case LayerType::Conv1d:case LayerType::ConvTranspose2d:case LayerType::DepthwiseConv2d:
        case LayerType::Bilinear:case LayerType::Embedding:case LayerType::EmbeddingBag:case LayerType::PatchEmbed:
        case LayerType::MaxPool2d:case LayerType::AvgPool2d:case LayerType::AdaptiveAvgPool2d:
        case LayerType::GlobalAvgPool2d:case LayerType::MaxPool1d:case LayerType::AvgPool1d:case LayerType::TokenMeanPool:
        case LayerType::Transpose:case LayerType::Permute:case LayerType::Squeeze:case LayerType::Unsqueeze:case LayerType::View:
        case LayerType::Concat:case LayerType::Split:case LayerType::Chunk:case LayerType::Stack:
        case LayerType::UpsampleNearest:case LayerType::UpsampleBilinear:case LayerType::UpsampleBicubic:case LayerType::PixelShuffle:
        case LayerType::ZeroPad2d:case LayerType::ReflectionPad2d:case LayerType::ReplicationPad2d:case LayerType::Constant:
        case LayerType::NMS:return true; // Deterministic selection and stop-gradient.
        default:return false;
    }
}
bool finite(const std::vector<std::vector<float>>& outputs){
    for(const auto& o:outputs)if(!RuntimeChecks::allFinite(o.data(),o.size()))return false;
    return true;
}
bool close(const std::vector<float>& a,const std::vector<float>& b){
    if(a.size()!=b.size())return false;
    for(size_t i=0;i<a.size();++i)if(!std::isfinite(a[i]) || !std::isfinite(b[i]) || std::abs(a[i]-b[i])>4e-4f*(1+std::abs(b[i])))return false;
    return true;
}
bool close(const std::vector<std::vector<float>>& a,const std::vector<std::vector<float>>& b){
    if(a.size()!=b.size())return false;
    for(size_t i=0;i<a.size();++i)if(!close(a[i],b[i]))return false;
    return true;
}
size_t signature(const Layer& l,bool backward,bool training,uint64_t generation,
    const std::vector<const std::vector<float>*>& inputs,const std::vector<const std::vector<float>*>& gradients){
    size_t h=0;mix(h,generation);mix(h,l.weight_block!=nullptr);mix(h,backward);mix(h,training);mix(h,size_t(l.type_enum));
    mix(h,size_t(l.dtype));mix(h,size_t(l.accumulation_dtype));mix(h,l.getWeightsSize());mix(h,l.trainable_parameter);
    if(backward){mix(h,l.grad_weights.size());mix(h,l.grad_bias.size());}
    for(int v:{l.in_features,l.out_features,l.in_channels,l.out_channels,l.input_height,l.input_width,l.output_height,l.output_width,l.kernel_size,l.kernel_h,l.kernel_w,l.stride,l.stride_h,l.stride_w,l.padding,l.pad_h,l.pad_w,l.dilation,l.dilation_h,l.dilation_w,l.groups,l.num_groups,l.embed_dim,l.seq_len,l.axis,int(l.affine),int(l.use_bias),int(l.activation),l.skip_input_index})mix(h,size_t(v));
    for(float v:{l.eps,l.alpha,l.negative_slope,l.leaky_relu_alpha,l.activation_param})mix(h,std::hash<float>{}(v));
    for(int v:{l.padding_idx,l.concat_axis,l.num_splits,l.split_axis,l.num_chunks,l.stack_axis,l.squeeze_dim,l.unsqueeze_dim,l.out_h,l.out_w,l.patch_dim,l.num_patches,l.seq_text})mix(h,size_t(v));
    for(float v:{l.scale_h,l.scale_w})mix(h,std::hash<float>{}(v));
    if(l.type_enum==LayerType::NMS){
        mix(h,std::hash<float>{}(l.nms_iou_threshold));mix(h,std::hash<float>{}(l.nms_score_threshold));
        mix(h,size_t(l.nms_max_detections));mix(h,size_t(l.nms_class_agnostic));
    }
    for(const auto* dims:{&l.shape,&l.target_shape,&l.permute_dims,&l.split_sizes}){mix(h,dims->size());for(int v:*dims)mix(h,size_t(v));}
    for(const auto* tensors:{&inputs,&gradients}){mix(h,tensors->size());for(const auto* v:*tensors)mix(h,v?v->size():size_t(-1));}
    return h;
}
void invalidate(size_t key){cache.erase(key);}
// Execute callbacks receive a disposable layer for every probe. The authoritative
// model and its accumulated gradients are only used by the final dispatch.
Ranking rank(const std::vector<AbstractRuntime*>& candidates,const Layer& layer,
    const std::vector<const std::vector<float>*>& inputs,const std::vector<const std::vector<float>*>& gradients,
    bool backward,bool training,uint64_t generation,const Probe& execute){
    Ranking result;result.route=candidates;
    auto& profiles=backward?backward_profiles:forward_profiles;
    auto use_cpu_choice=[&](const std::vector<Measurement>& costs) {
        for(const auto& cost:costs)if(std::string(cost.runtime->name())=="CPU") {
            result.cpu_workers=cost.cpu_workers;result.cpu_algorithm=cost.cpu_algorithm;break;
        }
    };
    if(!enabled() || !replayable(layer.type_enum)){profiles.erase(&layer);return result;}
    std::vector<AbstractRuntime*> eligible;
    for(auto* rt:candidates)if(rt && rt->isInitialized() && runtimeCapabilityIsNative(backward?
        rt->queryBackwardOperationCapability(layer,inputs,gradients,training):rt->queryForwardOperationCapability(layer,inputs,training)))eligible.push_back(rt);
    if(eligible.empty())return result;
    const auto worker_candidates=ExecutionPolicy::cpuCandidates();
    const bool pointwise_conv=layer.type_enum==LayerType::Conv2d && layer.in_channels==1 && layer.get_kernel_h()==1 &&
        layer.get_kernel_w()==1 && layer.get_stride_h()==1 && layer.get_pad_h()==0;
    const auto algorithms=ExecutionPolicy::cpuAlgorithms(pointwise_conv && !backward);
    if(eligible.size()==1 && (std::string(eligible[0]->name())!="CPU" ||
        (worker_candidates.size()==1 && algorithms.size()==1))) {
        result.route=eligible;result.ordered=true;profiles.erase(&layer);return result;
    }
    result.key=signature(layer,backward,training,generation,inputs,gradients);
#ifdef _OPENMP
    mix(result.key,size_t(omp_get_max_threads()));mix(result.key,size_t(omp_in_parallel()));
#endif
    for(auto* rt:eligible){mix(result.key,reinterpret_cast<size_t>(rt));mix(result.key,configuration(*rt));}
    auto found=cache.find(result.key);
    if(found!=cache.end() && found->second.uses<ExecutionPolicy::Limits::calibration_refresh){
        ++found->second.uses;result.route.clear();result.calibrated=true;result.ordered=true;
        for(const auto& cost:found->second.costs)result.route.push_back(cost.runtime);
        profiles[&layer]=found->second.costs;use_cpu_choice(found->second.costs);
        return result;
    }
    profiles.erase(&layer);
    size_t bytes=sizeof(float)*(layer.weights.size()+layer.bias.size()+layer.grad_weights.size()+layer.grad_bias.size()+layer.running_mean.size()+layer.running_var.size());
    if(layer.getWeightsSize()>std::numeric_limits<size_t>::max()/32)return result;
    bytes+=layer.getWeightsSize()*32;
    for(const auto* tensors:{&inputs,&gradients})for(const auto* v:*tensors)if(v){if(v->size()>std::numeric_limits<size_t>::max()/32)return result;bytes+=v->size()*32;}
    const size_t budget=ExecutionPolicy::calibrationBudget();
    auto& guard=MemoryGuard::instance();const size_t limit=guard.getLimit(),used=guard.getCurrentBytes();
    if(bytes>budget || guard.isBlocked() || guard.isFrozen() || (limit && (used>=limit || bytes>limit-used))){
        // If measurement is disallowed, keep the explicit serial CPU reference
        // for every graph shape, rather than inventing an arithmetic cutoff.
        result.route=eligible;
        std::stable_sort(result.route.begin(),result.route.end(),[](auto* a,auto* b){
            return std::string(a->name())=="CPU" && std::string(b->name())!="CPU";
        });
        result.ordered=true;
        return result;
    }
    // Host arithmetic is the numerical reference whenever available.
    std::stable_sort(eligible.begin(),eligible.end(),[](auto* a,auto* b){return std::string(a->name())=="CPU" && std::string(b->name())!="CPU";});
    Entry entry;std::vector<std::vector<float>> reference;std::vector<float> reference_w,reference_b;bool have_reference=false;
    auto exact=[](const std::vector<float>& a,const std::vector<float>& b) {
        return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0);
    };
    auto exact_buffers=[&](const auto& a,const auto& b) {
        if(a.size()!=b.size())return false;
        for(size_t i=0;i<a.size();++i)if(!exact(a[i],b[i]))return false;
        return true;
    };
    for(auto* rt:eligible) {
        const bool cpu=std::string(rt->name())=="CPU";
        std::optional<Measurement> best;
        for(auto algorithm:cpu?algorithms:std::vector<ExecutionPolicy::CpuAlgorithm>{ExecutionPolicy::CpuAlgorithm::Reference})
        for(int workers:cpu?worker_candidates:std::vector<int>{1}) {
            bool valid=true;std::vector<double> samples;samples.reserve(5);
            for(int sample=0;sample<6 && valid;++sample) {
                Layer disposable;std::vector<std::vector<float>> outputs;
                try{disposable=layer;}catch(...){return result;}
                ExecutionPolicy::CpuScope scope(workers,algorithm);
                const auto start=std::chrono::steady_clock::now();
                try{valid=execute(*rt,disposable,outputs) && (!outputs.empty() || (backward && inputs.empty())) && finite(outputs) &&
                    (!backward || (RuntimeChecks::allFinite(disposable.grad_weights.data(),disposable.grad_weights.size()) &&
                    RuntimeChecks::allFinite(disposable.grad_bias.data(),disposable.grad_bias.size())));}catch(...){valid=false;}
                const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                if(!valid)break;
                if(!have_reference) {reference=outputs;reference_w=disposable.grad_weights;reference_b=disposable.grad_bias;have_reference=true;}
                valid=(cpu?exact_buffers(outputs,reference):layer.type_enum==LayerType::NMS?outputs==reference:close(outputs,reference)) &&
                    (!backward || (cpu?(exact(disposable.grad_weights,reference_w)&&exact(disposable.grad_bias,reference_b)):
                        (close(disposable.grad_weights,reference_w)&&close(disposable.grad_bias,reference_b))));
                if(sample>0)samples.push_back(seconds);
            }
            if(valid && samples.size()==5) {
                std::sort(samples.begin(),samples.end());
                Measurement cost{rt,configuration(*rt),samples[2],workers,algorithm};
                if(!best || cost.seconds<best->seconds)best=cost;
            }
        }
        if(best)entry.costs.push_back(*best);
    }
    if(entry.costs.empty())return result;
    std::stable_sort(entry.costs.begin(),entry.costs.end(),[](const auto& a,const auto& b){return a.seconds<b.seconds;});
    result.route.clear();for(const auto& cost:entry.costs)result.route.push_back(cost.runtime);
    entry.uses=1;result.calibrated=true;result.ordered=true;
    if(cache.size()>=ExecutionPolicy::Limits::calibration_cache_entries){cache.clear();forward_profiles.clear();backward_profiles.clear();}
    profiles[&layer]=entry.costs;use_cpu_choice(entry.costs);
    if(std::getenv("MIMIR_ACCEL_VERBOSE")){std::cerr<<"[planner-autotune] direction="<<(backward?"backward":"forward")<<" layer="<<layer.name<<" selected="<<entry.costs.front().runtime->name()<<" median_host_roundtrip="<<entry.costs.front().seconds<<" samples=5 cpu_workers="<<result.cpu_workers<<"\n";}
    cache[result.key]=std::move(entry);return result;
}
}
