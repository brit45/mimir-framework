#pragma once

#include <functional>
#include <vector>

#include "runtimes/AbstractRuntime.hpp"

class RuntimeRouter {
public:
    static RuntimeRouter& instance();

    // Priority is fixed internally as [ROCM, CUDA, CPU].
    void setRuntimes(AbstractRuntime* rocm, AbstractRuntime* cuda, AbstractRuntime* cpu);

    // Activation callbacks are also ordered [ROCM, CUDA, CPU].
    // They may initialize runtimes lazily before dispatch.
    void setActivators(
        std::function<bool()> rocm,
        std::function<bool()> cuda,
        std::function<bool()> cpu
    );

    void activateAvailableRuntimes() const;

    bool dispatchForwardLayer(
        const std::vector<const std::vector<float>*>& inputs,
        std::vector<std::vector<float>>& outputs,
        const Layer& layer,
        bool training,
        AbstractRuntime** selected_runtime = nullptr
    ) const;

    bool dispatchBackwardLayer(
        const std::vector<const std::vector<float>*>& inputs,
        const std::vector<const std::vector<float>*>& grad_outputs,
        std::vector<std::vector<float>>& grad_inputs,
        Layer& layer,
        bool training,
        AbstractRuntime** selected_runtime = nullptr
    ) const;

private:
    RuntimeRouter() = default;

    std::vector<AbstractRuntime*> runtime_priority_;
    std::function<bool()> activate_rocm_;
    std::function<bool()> activate_cuda_;
    std::function<bool()> activate_cpu_;
};
