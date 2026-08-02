#pragma once

#include "runtimes/AbstractRuntime.hpp"

class CudaRuntime final : public AbstractRuntime {
public:
    ~CudaRuntime() override { shutdown(); }
    const char* name() const override { return "CUDA"; }

    bool initialize(const RuntimeConfig& cfg) override;
    void shutdown() override;

    bool isInitialized() const override;

    bool linearForward(
        const float* input,
        const float* weights,
        const float* bias_or_null,
        float* output,
        int batch,
        int in_f,
        int out_f
    ) override;

    bool forwardLayer(
        const std::vector<const std::vector<float>*>& inputs,
        std::vector<std::vector<float>>& outputs,
        const Layer& layer,
        bool training
    ) override;

    bool supportsForwardLayerType(LayerType type) const override;
    bool supportsBackwardLayerType(LayerType type) const override;

    bool backwardLayer(
        const std::vector<const std::vector<float>*>& inputs,
        const std::vector<const std::vector<float>*>& grad_outputs,
        std::vector<std::vector<float>>& grad_inputs,
        Layer& layer,
        bool training
    ) override;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};
