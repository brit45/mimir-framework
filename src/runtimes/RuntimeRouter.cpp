#include "runtimes/RuntimeRouter.hpp"

RuntimeRouter& RuntimeRouter::instance() {
    static RuntimeRouter router;
    return router;
}

void RuntimeRouter::setRuntimes(AbstractRuntime* rocm, AbstractRuntime* cuda, AbstractRuntime* cpu) {
    runtime_priority_.clear();
    runtime_priority_.reserve(3);

    if (rocm) runtime_priority_.push_back(rocm);
    if (cuda) runtime_priority_.push_back(cuda);
    if (cpu) runtime_priority_.push_back(cpu);
}

void RuntimeRouter::setActivators(
    std::function<bool()> rocm,
    std::function<bool()> cuda,
    std::function<bool()> cpu
) {
    activate_rocm_ = std::move(rocm);
    activate_cuda_ = std::move(cuda);
    activate_cpu_ = std::move(cpu);
}

void RuntimeRouter::activateAvailableRuntimes() const {
    if (activate_rocm_) (void)activate_rocm_();
    if (activate_cuda_) (void)activate_cuda_();
    if (activate_cpu_) (void)activate_cpu_();
}

bool RuntimeRouter::dispatchForwardLayer(
    const std::vector<const std::vector<float>*>& inputs,
    std::vector<std::vector<float>>& outputs,
    const Layer& layer,
    bool training,
    AbstractRuntime** selected_runtime
) const {
    activateAvailableRuntimes();
    return AbstractRuntime::dispatchForwardLayer(
        runtime_priority_, inputs, outputs, layer, training, selected_runtime);
}

bool RuntimeRouter::dispatchBackwardLayer(
    const std::vector<const std::vector<float>*>& inputs,
    const std::vector<const std::vector<float>*>& grad_outputs,
    std::vector<std::vector<float>>& grad_inputs,
    Layer& layer,
    bool training,
    AbstractRuntime** selected_runtime
) const {
    activateAvailableRuntimes();
    return AbstractRuntime::dispatchBackwardLayer(
        runtime_priority_, inputs, grad_outputs, grad_inputs, layer, training, selected_runtime);
}
