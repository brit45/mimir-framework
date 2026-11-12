#include "runtimes/FiniteCheck.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"
#include "tensors.hpp"
#include "Planning/FusionCalibration.hpp"

#include "Layers.hpp"
#include "Planning/ExecutionCalibration.hpp"

#include <cmath>
#include <cstddef>
#include <array>
#include <algorithm>

namespace {

bool tensorsAreFinite(const std::vector<std::vector<float>>& tensors) {
    for (const auto& tensor : tensors) {
        if (!Mimir::RuntimeChecks::allFinite(tensor.data(), tensor.size())) return false;
    }
    return true;
}

auto rankForward(const std::vector<AbstractRuntime*>& candidates,const Layer& layer,
    const std::vector<const std::vector<float>*>& inputs,bool training,uint64_t generation,const RuntimeForwardContext& context){
    return Mimir::Planning::Calibration::rank(candidates,layer,inputs,{},false,training,generation,
        [&](AbstractRuntime& rt,Layer& disposable,std::vector<std::vector<float>>& outputs){
            auto probe=context;std::vector<uint8_t> mask;if(context.output_mask)probe.output_mask=&mask;
            return rt.forwardLayerWithContext(inputs,outputs,disposable,training,probe);
        });
}

} // namespace

RuntimeRouter& RuntimeRouter::instance() {
    static RuntimeRouter router;
    return router;
}

bool RuntimeRouter::dispatchTensorWeights(const std::vector<tensor>& tensors,
    std::vector<uint16_t>& output,AbstractRuntime** selected) const {
    if(selected)*selected=nullptr;
    ensureActivatedAndComposed();
    auto candidates=runtime_priority_;
    static thread_local CpuRuntime standalone;
    // Match reconstruction-loss dispatch: standalone APIs work without a model,
    // while an explicitly configured runtime set remains authoritative.
    if(candidates.empty()) {
        if(!standalone.isInitialized() && !standalone.initialize(RuntimeConfig{}))return false;
        candidates.push_back(&standalone);
    }
    for(auto* runtime:candidates) {
        if(!runtime || !runtime->isInitialized() || runtime->config().disabled ||
           !runtime->supportsTensorWeights())continue;
        std::vector<uint16_t> result;
        if(!runtime->computeTensorWeights(tensors,result) || result.size()!=tensors.size())continue;
        output=std::move(result);
        if(selected)*selected=runtime;
        return true;
    }
    return false;
}

size_t RuntimeRouter::layerTypeIndex(const LayerType type) {
    const size_t idx = static_cast<size_t>(type);
    const size_t max_idx = static_cast<size_t>(LayerType::UNKNOWN);
    return (idx <= max_idx) ? idx : max_idx;
}

void RuntimeRouter::composeRoutes() const {
    const size_t type_count = static_cast<size_t>(LayerType::UNKNOWN) + 1;

    forward_vote_.assign(type_count, 0);
    backward_vote_.assign(type_count, 0);
    forward_routes_.assign(type_count, {});
    backward_routes_.assign(type_count, {});

    for (AbstractRuntime* rt : runtime_priority_) {
        if (!rt || !rt->isInitialized()) continue;

        for (size_t i = 0; i < type_count; ++i) {
            const LayerType type = static_cast<LayerType>(i);

            if (runtimeCapabilityIsNative(rt->queryForwardCapability(type))) {
                forward_vote_[i] = 1;
                forward_routes_[i].push_back(rt);
            }
            if (runtimeCapabilityIsNative(rt->queryBackwardCapability(type))) {
                backward_vote_[i] = 1;
                backward_routes_[i].push_back(rt);
            }
        }
    }
}

void RuntimeRouter::setRuntimes(
    AbstractRuntime* rocm,
    AbstractRuntime* cuda,
    AbstractRuntime* vulkan,
    AbstractRuntime* opencl,
    AbstractRuntime* cpu
) {
    setRuntimes(rocm, cuda, vulkan, opencl, nullptr, cpu);
}

void RuntimeRouter::setRuntimes(
    AbstractRuntime* rocm,
    AbstractRuntime* cuda,
    AbstractRuntime* vulkan,
    AbstractRuntime* opencl,
    AbstractRuntime* tpu,
    AbstractRuntime* cpu
) {
    ++optimizer_routing_generation_;
    Mimir::Planning::FusionCalibration::clearProfiles();
    Mimir::Planning::Calibration::forward_profiles.clear();
    Mimir::Planning::Calibration::backward_profiles.clear();
    runtime_priority_.clear();
    runtime_priority_.reserve(6);

    switch (rocm != nullptr) {
        case true: runtime_priority_.push_back(rocm); break;
        case false: break;
    }
    switch (cuda != nullptr) {
        case true: runtime_priority_.push_back(cuda); break;
        case false: break;
    }
    switch (vulkan != nullptr) {
        case true: runtime_priority_.push_back(vulkan); break;
        case false: break;
    }
    switch (opencl != nullptr) {
        case true: runtime_priority_.push_back(opencl); break;
        case false: break;
    }
    switch (tpu != nullptr) {
        case true: runtime_priority_.push_back(tpu); break;
        case false: break;
    }
    switch (cpu != nullptr) {
        case true: runtime_priority_.push_back(cpu); break;
        case false: break;
    }

    runtimes_activated_ = false;
    forward_layer_routes_.clear();
    backward_layer_routes_.clear();

    composeRoutes();
}

AbstractRuntime* RuntimeRouter::selectForwardRuntimeForLayer(
    const Layer& layer, const bool allow_host_fallback) const {
    ensureActivatedAndComposed();
    AbstractRuntime* fallback = nullptr;
    for (AbstractRuntime* runtime : runtime_priority_) {
        if (!runtime || !runtime->isInitialized()) continue;
        const RuntimeCapabilityLevel capability = runtime->queryForwardCapability(layer.type_enum);
        if (runtimeCapabilityIsNative(capability)) return runtime;
        if (allow_host_fallback && !fallback && capability == RuntimeCapabilityLevel::HostFallback) {
            fallback = runtime;
        }
    }
    return fallback;
}

AbstractRuntime* RuntimeRouter::selectBackwardRuntimeForLayer(
    const Layer& layer, const bool allow_host_fallback) const {
    ensureActivatedAndComposed();
    AbstractRuntime* fallback = nullptr;
    for (AbstractRuntime* runtime : runtime_priority_) {
        if (!runtime || !runtime->isInitialized()) continue;
        const RuntimeCapabilityLevel capability = runtime->queryBackwardCapability(layer.type_enum);
        if (runtimeCapabilityIsNative(capability)) return runtime;
        if (allow_host_fallback && !fallback && capability == RuntimeCapabilityLevel::HostFallback) {
            fallback = runtime;
        }
    }
    return fallback;
}

void RuntimeRouter::setActivators(
    std::function<bool()> rocm,
    std::function<bool()> cuda,
    std::function<bool()> vulkan,
    std::function<bool()> opencl,
    std::function<bool()> cpu
) {
    ++optimizer_routing_generation_;
    activate_rocm_ = std::move(rocm);
    activate_cuda_ = std::move(cuda);
    activate_vulkan_ = std::move(vulkan);
    activate_opencl_ = std::move(opencl);
    activate_cpu_ = std::move(cpu);
    runtimes_activated_ = false;
    forward_layer_routes_.clear();
    backward_layer_routes_.clear();
}

void RuntimeRouter::activateAvailableRuntimes() const {
    switch (static_cast<bool>(activate_rocm_)) {
        case true: (void)activate_rocm_(); break;
        case false: break;
    }
    switch (static_cast<bool>(activate_cuda_)) {
        case true: (void)activate_cuda_(); break;
        case false: break;
    }
    switch (static_cast<bool>(activate_vulkan_)) {
        case true: (void)activate_vulkan_(); break;
        case false: break;
    }
    switch (static_cast<bool>(activate_opencl_)) {
        case true: (void)activate_opencl_(); break;
        case false: break;
    }
    switch (static_cast<bool>(activate_cpu_)) {
        case true: (void)activate_cpu_(); break;
        case false: break;
    }
}

void RuntimeRouter::ensureActivatedAndComposed() const {
    if (!runtimes_activated_) {
        activateAvailableRuntimes();
        runtimes_activated_ = true;
        composeRoutes();
    }
    switch (forward_vote_.empty() || backward_vote_.empty()) {
        case true:
            composeRoutes();
            break;
        case false:
            break;
    }
}

std::vector<AbstractRuntime*> RuntimeRouter::buildForwardRouteForLayer(const Layer& layer) const {
    const size_t idx = layerTypeIndex(layer.type_enum);
    const std::vector<AbstractRuntime*>& base_route =
        (idx < forward_routes_.size()) ? forward_routes_[idx] : runtime_priority_;

    std::vector<AbstractRuntime*> route;
    route.reserve(base_route.size());
    for (AbstractRuntime* rt : base_route) {
        if (!rt) continue;
        if (!rt->isInitialized()) continue;
        if (!runtimeCapabilityIsNative(rt->queryForwardCapability(layer.type_enum))) continue;
        route.push_back(rt);
    }
    return route;
}

std::vector<AbstractRuntime*> RuntimeRouter::buildBackwardRouteForLayer(const Layer& layer) const {
    const size_t idx = layerTypeIndex(layer.type_enum);
    const std::vector<AbstractRuntime*>& base_route =
        (idx < backward_routes_.size()) ? backward_routes_[idx] : runtime_priority_;

    std::vector<AbstractRuntime*> route;
    route.reserve(base_route.size());
    for (AbstractRuntime* rt : base_route) {
        if (!rt) continue;
        if (!rt->isInitialized()) continue;
        if (!runtimeCapabilityIsNative(rt->queryBackwardCapability(layer.type_enum))) continue;
        route.push_back(rt);
    }
    return route;
}

void RuntimeRouter::planForwardLayerRoutes(const std::vector<Layer>& layers) const {
    ensureActivatedAndComposed();

    bool reuse = forward_layer_routes_.size() == layers.size();
    if (reuse) {
        for (const Layer& layer : layers)
            if (forward_layer_routes_.find(&layer) == forward_layer_routes_.end()) {
                reuse = false;
                break;
            }
    }
    if (!reuse) {
        forward_layer_routes_.clear();
        forward_layer_routes_.reserve(layers.size());
    }
    for (const Layer& layer : layers) {
        auto& route = forward_layer_routes_[&layer];
        route.clear();
        const size_t idx = layerTypeIndex(layer.type_enum);
        const auto& base = idx < forward_routes_.size() ? forward_routes_[idx] : runtime_priority_;
        route.reserve(base.size());
        // Refresh capability and initialization even when storage is reused.
        for (AbstractRuntime* rt : base)
            if (rt && rt->isInitialized() &&
                runtimeCapabilityIsNative(rt->queryForwardCapability(layer.type_enum)))
                route.push_back(rt);
    }
}

bool RuntimeRouter::hasForwardRouteForLayer(const Layer& layer) const {
    ensureActivatedAndComposed();

    auto it = forward_layer_routes_.find(&layer);
    if (it == forward_layer_routes_.end()) {
        auto inserted = forward_layer_routes_.emplace(&layer, buildForwardRouteForLayer(layer));
        it = inserted.first;
    }
    return !it->second.empty();
}

bool RuntimeRouter::voteForwardLayerType(const LayerType type) const {
    ensureActivatedAndComposed();
    const size_t idx = layerTypeIndex(type);
    return idx < forward_vote_.size() && forward_vote_[idx] != 0;
}

bool RuntimeRouter::voteBackwardLayerType(const LayerType type) const {
    ensureActivatedAndComposed();
    const size_t idx = layerTypeIndex(type);
    return idx < backward_vote_.size() && backward_vote_[idx] != 0;
}

bool RuntimeRouter::dispatchForwardLayer(
    const std::vector<const std::vector<float>*>& inputs,
    std::vector<std::vector<float>>& outputs,
    const Layer& layer,
    bool training,
    AbstractRuntime** selected_runtime,
    const RuntimeForwardContext& context
) const {
    const int skip = layer.skipInputIndex();
    if (!context.skip_connections_enabled && skip >= 0 &&
        static_cast<size_t>(skip) < inputs.size() && inputs[skip]) {
        auto gated_inputs = inputs;
        std::vector<float> zeros(inputs[skip]->size(), 0.0f);
        gated_inputs[skip] = &zeros;
        auto gated_context = context;
        gated_context.skip_connections_enabled = true;
        return dispatchForwardLayer(gated_inputs, outputs, layer, training, selected_runtime, gated_context);
    }
    ensureActivatedAndComposed();

    if (selected_runtime) *selected_runtime = nullptr;
    outputs.clear();

    // Autocast is capability-routed independently of the regular FP32 planner
    // route. A backend must explicitly advertise the requested precision.
    if (context.autocast_dtype != Mimir::DType::F32) {
        for (AbstractRuntime* runtime : runtime_priority_) {
            if (!runtime || !runtime->isInitialized() ||
                !runtime->supportsAutocast(layer, context.autocast_dtype)) continue;
            std::vector<std::vector<float>> local_outputs;
            if (runtime->forwardLayerWithContext(inputs, local_outputs, layer,
                    training, context) && !local_outputs.empty() &&
                tensorsAreFinite(local_outputs)) {
                outputs = std::move(local_outputs);
                if (selected_runtime) *selected_runtime = runtime;
                return true;
            }
        }
        // Only use ordinary FP32 dispatch when this operation has no AMP-capable
        // backend; a failed AMP-capable kernel is treated as a real failure.
        if (std::any_of(runtime_priority_.begin(), runtime_priority_.end(),
                [&](AbstractRuntime* runtime) {
                    return runtime && runtime->isInitialized() &&
                        runtime->supportsAutocast(layer, context.autocast_dtype);
                })) return false;
        if (layer.type_enum == LayerType::Linear) return false;
    }

    auto it = forward_layer_routes_.find(&layer);
    if (it == forward_layer_routes_.end()) {
        auto inserted = forward_layer_routes_.emplace(&layer, buildForwardRouteForLayer(layer));
        it = inserted.first;
    }

    auto ranking=rankForward(it->second,layer,inputs,training,optimizer_routing_generation_,context);
    Mimir::Planning::ExecutionPolicy::CpuScope cpu_scope(ranking.cpu_workers,ranking.cpu_algorithm);
    auto& route=ranking.route;
    if (route.empty()) return false;

    size_t i = 0;
    while (i < route.size()) {
        AbstractRuntime* rt = route[i];
        if (!rt || !rt->isInitialized()) {
            route.erase(route.begin() + static_cast<std::ptrdiff_t>(i));
            continue;
        }
        if (!runtimeCapabilityIsNative(
                rt->queryForwardOperationCapability(layer, inputs, training))) {
            ++i;
            continue;
        }

        std::vector<std::vector<float>> local_outputs;
        if (rt->forwardLayerWithContext(inputs, local_outputs, layer, training, context) &&
            !local_outputs.empty() &&
            tensorsAreFinite(local_outputs)) {
            outputs = std::move(local_outputs);
            if (selected_runtime) *selected_runtime = rt;
            return true;
        }

        if(ranking.calibrated)Mimir::Planning::Calibration::invalidate(ranking.key);
        // Keep the failure in the layer cache as well as the measured local order.
        // planForwardLayerRoutes/setRuntimes can explicitly make it eligible again.
        auto& cached = it->second;
        cached.erase(std::remove(cached.begin(), cached.end(), rt), cached.end());
        route.erase(route.begin() + static_cast<std::ptrdiff_t>(i));
    }

    return false;
}

bool RuntimeRouter::dispatchForwardLayerPlanned(
    AbstractRuntime* preferred,
    const std::vector<const std::vector<float>*>& inputs,
    std::vector<std::vector<float>>& outputs,
    const Layer& layer,
    const bool training,
    AbstractRuntime** selected_runtime,
    const RuntimeForwardContext& context
) const {
    const int skip = layer.skipInputIndex();
    if (!context.skip_connections_enabled && skip >= 0 &&
        static_cast<size_t>(skip) < inputs.size() && inputs[skip]) {
        auto gated_inputs = inputs;
        std::vector<float> zeros(inputs[skip]->size(), 0.0f);
        gated_inputs[skip] = &zeros;
        auto gated_context = context;
        gated_context.skip_connections_enabled = true;
        return dispatchForwardLayerPlanned(preferred, gated_inputs, outputs, layer, training, selected_runtime, gated_context);
    }
    ensureActivatedAndComposed();
    if (context.autocast_dtype != Mimir::DType::F32 &&
        layer.type_enum == LayerType::Linear) {
        return dispatchForwardLayer(inputs, outputs, layer, training,
                                    selected_runtime, context);
    }
    if (selected_runtime) *selected_runtime = nullptr;
    outputs.clear();

    auto route_it = forward_layer_routes_.find(&layer);
    if (route_it == forward_layer_routes_.end()) {
        route_it = forward_layer_routes_.emplace(
            &layer, buildForwardRouteForLayer(layer)).first;
    }
    auto ranking=rankForward(route_it->second,layer,inputs,training,optimizer_routing_generation_,context);
    Mimir::Planning::ExecutionPolicy::CpuScope cpu_scope(ranking.cpu_workers,ranking.cpu_algorithm);
    auto& route=ranking.route;
    if(ranking.ordered && !route.empty())preferred=route.front();

    auto is_eligible = [&](AbstractRuntime* runtime) {
        return runtime && runtime->isInitialized() &&
            runtimeCapabilityIsNative(runtime->queryForwardOperationCapability(
                layer, inputs, training));
    };
    auto execute = [&](AbstractRuntime* runtime) -> bool {
        std::vector<std::vector<float>> local;
        if (!runtime->forwardLayerWithContext(inputs, local, layer, training, context) ||
            local.empty() || !tensorsAreFinite(local)) return false;
        outputs = std::move(local);
        if (selected_runtime) *selected_runtime = runtime;
        return true;
    };

    auto preferred_it = std::find(route.begin(), route.end(), preferred);
    if (preferred_it != route.end()) {
        if (is_eligible(preferred)) {
            if (execute(preferred)) return true;
            if(ranking.calibrated)Mimir::Planning::Calibration::invalidate(ranking.key);
            auto& cached = route_it->second;
            cached.erase(std::remove(cached.begin(), cached.end(), preferred), cached.end());
            route.erase(preferred_it);
        }
    }

    size_t index = 0;
    while (index < route.size()) {
        AbstractRuntime* runtime = route[index];
        if (!runtime || !runtime->isInitialized()) {
            route.erase(route.begin() + static_cast<std::ptrdiff_t>(index));
            continue;
        }
        if (!is_eligible(runtime)) {
            ++index;
            continue;
        }
        if (execute(runtime)) return true;
        if(ranking.calibrated)Mimir::Planning::Calibration::invalidate(ranking.key);
        auto& cached = route_it->second;
        cached.erase(std::remove(cached.begin(), cached.end(), runtime), cached.end());
        route.erase(route.begin() + static_cast<std::ptrdiff_t>(index));
    }
    return false;
}

bool RuntimeRouter::dispatchBackwardLayer(
    const std::vector<const std::vector<float>*>& inputs,
    const std::vector<const std::vector<float>*>& grad_outputs,
    std::vector<std::vector<float>>& grad_inputs,
    Layer& layer,
    bool training,
    AbstractRuntime** selected_runtime,
    const RuntimeBackwardContext& context
) const {
    const int skip = layer.skipInputIndex();
    if (!context.skip_connections_enabled && skip >= 0 &&
        static_cast<size_t>(skip) < inputs.size() && inputs[skip]) {
        auto gated_inputs = inputs;
        std::vector<float> zeros(inputs[skip]->size(), 0.0f);
        gated_inputs[skip] = &zeros;
        auto gated_context = context;
        gated_context.skip_connections_enabled = true;
        const bool ok = dispatchBackwardLayer(gated_inputs, grad_outputs,
            grad_inputs, layer, training, selected_runtime, gated_context);
        if (ok && static_cast<size_t>(skip) < grad_inputs.size())
            std::fill(grad_inputs[skip].begin(), grad_inputs[skip].end(), 0.0f);
        return ok;
    }
    ensureActivatedAndComposed();

    if (selected_runtime) *selected_runtime = nullptr;
    grad_inputs.clear();

    // Temporary/rewritten layers can reuse an address. Backward capability is
    // cheap to resolve and must describe this operation, not a previous layer.
    const size_t idx = layerTypeIndex(layer.type_enum);
    const auto& base = idx < backward_routes_.size() ? backward_routes_[idx] : runtime_priority_;
    // Snapshot before callbacks, without a heap allocation for the six backends.
    std::array<AbstractRuntime*, 6> inline_route{};
    std::vector<AbstractRuntime*> overflow;
    if (base.size() > inline_route.size()) overflow.resize(base.size());
    AbstractRuntime** route = overflow.empty() ? inline_route.data() : overflow.data();
    size_t count = 0;
    for (AbstractRuntime* rt : base)
        if (rt && rt->isInitialized() &&
            runtimeCapabilityIsNative(rt->queryBackwardCapability(layer.type_enum)))
            route[count++] = rt;

    auto ranking=Mimir::Planning::Calibration::rank(Mimir::Planning::Calibration::RuntimeRoute(route,route+count),layer,inputs,grad_outputs,true,training,optimizer_routing_generation_,
        [&](AbstractRuntime& rt,Layer& disposable,std::vector<std::vector<float>>& outputs){return rt.backwardLayerWithContext(inputs,grad_outputs,outputs,disposable,training,context);});
    Mimir::Planning::ExecutionPolicy::CpuScope cpu_scope(ranking.cpu_workers,ranking.cpu_algorithm);
    for (size_t i = 0; i < ranking.route.size(); ++i) {
        AbstractRuntime* rt = ranking.route[i];
        if (!rt->isInitialized()) continue;
        if (!runtimeCapabilityIsNative(rt->queryBackwardOperationCapability(
                layer, inputs, grad_outputs, training))) {
            continue;
        }

        std::vector<std::vector<float>> local_grad_inputs;
        Layer disposable;
        Layer* target=&layer;
        if(ranking.calibrated){disposable=layer;target=&disposable;}
        if (rt->backwardLayerWithContext(inputs, grad_outputs, local_grad_inputs, *target, training, context) &&
            (layer.type_enum == LayerType::Constant || !local_grad_inputs.empty()) &&
            (!ranking.calibrated || (Mimir::RuntimeChecks::allFinite(target->grad_weights.data(),target->grad_weights.size()) && Mimir::RuntimeChecks::allFinite(target->grad_bias.data(),target->grad_bias.size()))) &&
            tensorsAreFinite(local_grad_inputs)) {
            if(ranking.calibrated){layer.grad_weights=std::move(disposable.grad_weights);layer.grad_bias=std::move(disposable.grad_bias);}
            grad_inputs = std::move(local_grad_inputs);
            if (selected_runtime) *selected_runtime = rt;
            return true;
        }

        if(ranking.calibrated)Mimir::Planning::Calibration::invalidate(ranking.key);
        // Try the next backend if execution or its finiteness check failed.
    }

    return false;
}

bool RuntimeRouter::dispatchBranchMerge(const std::vector<float>& left,
                                        const std::vector<float>& right,
                                        std::vector<float>& output, const Layer& layer) const {
    ensureActivatedAndComposed();
    for (auto* runtime : runtime_priority_) {
        if (!runtime || !runtime->isInitialized()) continue;
        std::vector<float> candidate;
        if (runtime->mergeBranches(left, right, candidate, layer)) {
            output = std::move(candidate);
            return true;
        }
    }
    return false;
}

bool RuntimeRouter::dispatchPointwiseForward(const std::vector<const Layer*>& chain,
    const std::vector<float>& input,std::vector<std::vector<float>>& outputs,bool retain,
    AbstractRuntime** selected,bool training,Mimir::DType dtype) const {
    if(selected)*selected=nullptr;
    ensureActivatedAndComposed();
    if(!Mimir::RuntimeChecks::allFinite(input.data(),input.size()))return false;
    namespace Fusion=Mimir::Planning::FusionCalibration;
    Fusion::Request request;request.layers=chain;request.input=&input;request.training=training;
    request.retain=retain;request.dtype=dtype;request.generation=optimizer_routing_generation_;
    for(auto* rt:runtime_priority_)if(rt && rt->isInitialized() && !rt->config().disabled)request.available.push_back(rt);
    auto ordinary=[&](Fusion::Buffers& result) {
        const std::vector<float>* value=&input;
        std::vector<float> current;
        result.clear();if(retain)result.reserve(chain.size());
        for(const auto* layer:chain) {
            Fusion::Buffers next;
            if(!dispatchForwardLayer({value},next,*layer,training) || next.size()!=1)return false;
            if(retain){result.push_back(std::move(next[0]));value=&result.back();}
            else {current=std::move(next[0]);value=&current;}
        }
        if(!retain)result.push_back(std::move(current));
        return true;
    };
    std::vector<Fusion::Candidate> candidates;
    const auto worker_candidates=Mimir::Planning::ExecutionPolicy::cpuCandidates();
    candidates.reserve(1+request.available.size()*worker_candidates.size());
    candidates.push_back({nullptr,{},1});
    for(auto* rt:request.available)
        if(runtimeCapabilityIsNative(rt->queryPointwiseFusionCapability(chain,false,training))) {
            if(std::string(rt->name())=="CPU") {
                for(int workers:worker_candidates)candidates.push_back({rt,{},workers});
            } else candidates.push_back({rt,{},1});
        }
    Fusion::Decision decision;
    if(!Fusion::lookup(request,candidates,decision)) {
        candidates[0].execute=ordinary;
        for(size_t i=1;i<candidates.size();++i) {
            auto* rt=candidates[i].runtime;
            candidates[i].execute=[&,rt](Fusion::Buffers& result) {
                return rt->pointwiseFusionForward(chain,input,result,retain,training) && tensorsAreFinite(result);
            };
        }
        decision=Fusion::select(request,candidates);
    }
    if(!decision.measured)return false;
    Fusion::Buffers result;
    if(!decision.fused) {
        if(!ordinary(result) || result.empty() || !tensorsAreFinite(result))return false;
        outputs=std::move(result);return true;
    }
    if(!decision.runtime)return false;
    Mimir::Planning::ExecutionPolicy::CpuScope cpu_scope(decision.cpu_workers);
    if(!decision.runtime->pointwiseFusionForward(chain,input,result,retain,training) ||
        result.empty() || !tensorsAreFinite(result)) {Fusion::invalidate(decision.key);return false;}
    outputs=std::move(result);if(selected)*selected=decision.runtime;return true;
}
bool RuntimeRouter::dispatchPointwiseBackward(const std::vector<const Layer*>& chain,
    const std::vector<const std::vector<float>*>& inputs,const std::vector<float>& grad,
    std::vector<float>& result,AbstractRuntime** selected,Mimir::DType dtype) const {
    if(selected)*selected=nullptr;
    ensureActivatedAndComposed();
    if(chain.size()!=inputs.size())return false;
    for(size_t i=0;i<chain.size();++i)
        if(!chain[i] || (chain[i]->type_enum!=LayerType::Identity && (!inputs[i] || inputs[i]->size()!=grad.size())))return false;
    namespace Fusion=Mimir::Planning::FusionCalibration;
    Fusion::Request request;request.layers=chain;request.input=&grad;request.snapshots=inputs;
    request.training=true;request.backward=true;request.dtype=dtype;request.generation=optimizer_routing_generation_;
    for(auto* rt:runtime_priority_)if(rt && rt->isInitialized() && !rt->config().disabled)request.available.push_back(rt);
    std::vector<Layer> disposable;
    auto ordinary=[&](Fusion::Buffers& output) {
        // Disposable gradients are part of every ordinary program execution,
        // so their copies are included in each latency sample as well.
        disposable.clear();disposable.reserve(chain.size());
        for(const auto* layer:chain)disposable.push_back(*layer);
        const std::vector<float>* value=&grad;std::vector<float> current;
        for(size_t i=chain.size();i-->0;) {
            // Identity backward needs only a shape, exactly as the normal
            // Model backward path supplies the current gradient as that view.
            const std::vector<const std::vector<float>*> x{inputs[i]?inputs[i]:value},g{value};
            const auto route=buildBackwardRouteForLayer(*chain[i]);
            const auto ranked=Mimir::Planning::Calibration::rank(route,*chain[i],x,g,true,true,
                optimizer_routing_generation_,[&](AbstractRuntime& rt,Layer& copy,Fusion::Buffers& dx) {
                    return rt.backwardLayerWithContext(x,g,dx,copy,true,{});
                });
            Mimir::Planning::ExecutionPolicy::CpuScope cpu_scope(ranked.cpu_workers,ranked.cpu_algorithm);
            bool valid=false;
            for(auto* rt:ranked.route) {
                if(!rt || !rt->isInitialized() || !runtimeCapabilityIsNative(
                    rt->queryBackwardOperationCapability(*chain[i],x,g,true)))continue;
                disposable[i].grad_weights.clear();disposable[i].grad_bias.clear();
                Fusion::Buffers dx;
                if(rt->backwardLayerWithContext(x,g,dx,disposable[i],true,{}) && dx.size()==1 && tensorsAreFinite(dx)) {
                    current=std::move(dx[0]);value=&current;valid=true;break;
                }
            }
            if(!valid)return false;
        }
        output.clear();output.push_back(std::move(current));return true;
    };
    std::vector<Fusion::Candidate> candidates;
    const auto worker_candidates=Mimir::Planning::ExecutionPolicy::cpuCandidates();
    candidates.reserve(1+request.available.size()*worker_candidates.size());
    candidates.push_back({nullptr,{},1});
    for(auto* rt:request.available)
        if(runtimeCapabilityIsNative(rt->queryPointwiseFusionCapability(chain,true,true))) {
            if(std::string(rt->name())=="CPU") {
                for(int workers:worker_candidates)candidates.push_back({rt,{},workers});
            } else candidates.push_back({rt,{},1});
        }
    Fusion::Decision decision;
    if(!Fusion::lookup(request,candidates,decision)) {
        candidates[0].execute=ordinary;
        for(size_t i=1;i<candidates.size();++i) {
            auto* rt=candidates[i].runtime;
            candidates[i].execute=[&,rt](Fusion::Buffers& output) {
                std::vector<float> dx;
                if(!rt->pointwiseFusionBackward(chain,inputs,grad,dx) ||
                   !Mimir::RuntimeChecks::allFinite(dx.data(),dx.size()))return false;
                output.clear();output.push_back(std::move(dx));return true;
            };
        }
        decision=Fusion::select(request,candidates);
    }
    if(!decision.measured)return false;
    if(!decision.fused) {
        Fusion::Buffers output;
        if(!ordinary(output) || output.size()!=1 || !tensorsAreFinite(output))return false;
        result=std::move(output[0]);return true;
    }
    if(!decision.runtime)return false;
    Mimir::Planning::ExecutionPolicy::CpuScope cpu_scope(decision.cpu_workers);
    std::vector<float> dx;
    if(!decision.runtime->pointwiseFusionBackward(chain,inputs,grad,dx) ||
        !Mimir::RuntimeChecks::allFinite(dx.data(),dx.size())) {Fusion::invalidate(decision.key);return false;}
    result=std::move(dx);if(selected)*selected=decision.runtime;return true;
}
