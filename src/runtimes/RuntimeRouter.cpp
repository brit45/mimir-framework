#include "runtimes/RuntimeRouter.hpp"

#include "Layers.hpp"

#include <cmath>
#include <cstddef>

namespace {

bool tensorsAreFinite(const std::vector<std::vector<float>>& tensors) {
    for (const auto& tensor : tensors) {
        for (const float value : tensor) {
            if (!std::isfinite(value)) return false;
        }
    }
    return true;
}

} // namespace

RuntimeRouter& RuntimeRouter::instance() {
    static RuntimeRouter router;
    return router;
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
        switch (rt == nullptr) {
            case true:
                continue;
            case false:
                break;
        }

        for (size_t i = 0; i < type_count; ++i) {
            const LayerType type = static_cast<LayerType>(i);

            switch (rt->supportsForwardLayerType(type)) {
                case true:
                    forward_vote_[i] = 1;
                    forward_routes_[i].push_back(rt);
                    break;
                case false:
                    break;
            }
            switch (rt->supportsBackwardLayerType(type)) {
                case true:
                    backward_vote_[i] = 1;
                    backward_routes_[i].push_back(rt);
                    break;
                case false:
                    break;
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
    AbstractRuntime* fpga,
    AbstractRuntime* cpu
) {
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
    switch (fpga != nullptr) {
        case true: runtime_priority_.push_back(fpga); break;
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
    activate_rocm_ = std::move(rocm);
    activate_cuda_ = std::move(cuda);
    activate_vulkan_ = std::move(vulkan);
    activate_opencl_ = std::move(opencl);
    activate_cpu_ = std::move(cpu);
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

    forward_layer_routes_.clear();
    forward_layer_routes_.reserve(layers.size());
    for (const Layer& layer : layers) {
        forward_layer_routes_.emplace(&layer, buildForwardRouteForLayer(layer));
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
    AbstractRuntime** selected_runtime
) const {
    ensureActivatedAndComposed();

    if (selected_runtime) *selected_runtime = nullptr;
    outputs.clear();

    auto it = forward_layer_routes_.find(&layer);
    if (it == forward_layer_routes_.end()) {
        auto inserted = forward_layer_routes_.emplace(&layer, buildForwardRouteForLayer(layer));
        it = inserted.first;
    }

    std::vector<AbstractRuntime*>& route = it->second;
    if (route.empty()) return false;

    size_t i = 0;
    while (i < route.size()) {
        AbstractRuntime* rt = route[i];
        if (!rt || !rt->isInitialized() || !rt->supportsForwardLayerType(layer.type_enum)) {
            route.erase(route.begin() + static_cast<std::ptrdiff_t>(i));
            continue;
        }

        std::vector<std::vector<float>> local_outputs;
        if (rt->forwardLayer(inputs, local_outputs, layer, training) &&
            !local_outputs.empty() &&
            tensorsAreFinite(local_outputs)) {
            outputs = std::move(local_outputs);
            if (selected_runtime) *selected_runtime = rt;
            return true;
        }

        // Runtime a échoué à l'exécution: on invalide cette route pour ce layer.
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
    AbstractRuntime** selected_runtime
) const {
    ensureActivatedAndComposed();
    if (selected_runtime) *selected_runtime = nullptr;
    outputs.clear();

    auto execute = [&](AbstractRuntime* runtime) -> bool {
        if (!runtime || !runtime->isInitialized() ||
            !runtimeCapabilityIsNative(runtime->queryForwardCapability(layer.type_enum))) return false;
        std::vector<std::vector<float>> local;
        if (!runtime->forwardLayer(inputs, local, layer, training) ||
            local.empty() || !tensorsAreFinite(local)) return false;
        outputs = std::move(local);
        if (selected_runtime) *selected_runtime = runtime;
        return true;
    };

    if (execute(preferred)) return true;

    auto it = forward_layer_routes_.find(&layer);
    if (it == forward_layer_routes_.end()) {
        it = forward_layer_routes_.emplace(&layer, buildForwardRouteForLayer(layer)).first;
    }
    for (AbstractRuntime* runtime : it->second) {
        if (runtime == preferred) continue;
        if (execute(runtime)) return true;
    }
    return false;
}

bool RuntimeRouter::dispatchBackwardLayer(
    const std::vector<const std::vector<float>*>& inputs,
    const std::vector<const std::vector<float>*>& grad_outputs,
    std::vector<std::vector<float>>& grad_inputs,
    Layer& layer,
    bool training,
    AbstractRuntime** selected_runtime
) const {
    ensureActivatedAndComposed();

    if (selected_runtime) *selected_runtime = nullptr;
    grad_inputs.clear();

    auto it = backward_layer_routes_.find(&layer);
    if (it == backward_layer_routes_.end()) {
        auto inserted = backward_layer_routes_.emplace(&layer, buildBackwardRouteForLayer(layer));
        it = inserted.first;
    }

    std::vector<AbstractRuntime*>& route = it->second;
    if (route.empty()) return false;

    size_t i = 0;
    while (i < route.size()) {
        AbstractRuntime* rt = route[i];
        if (!rt || !rt->isInitialized() || !rt->supportsBackwardLayerType(layer.type_enum)) {
            route.erase(route.begin() + static_cast<std::ptrdiff_t>(i));
            continue;
        }

        std::vector<std::vector<float>> local_grad_inputs;
        if (rt->backwardLayer(inputs, grad_outputs, local_grad_inputs, layer, training) &&
            !local_grad_inputs.empty() &&
            tensorsAreFinite(local_grad_inputs)) {
            grad_inputs = std::move(local_grad_inputs);
            if (selected_runtime) *selected_runtime = rt;
            return true;
        }

        // Runtime a échoué à l'exécution: on invalide cette route pour ce layer.
        route.erase(route.begin() + static_cast<std::ptrdiff_t>(i));
    }

    return false;
}
