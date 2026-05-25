#pragma once

#include "runtimes/AbstractRuntime.hpp"

class CpuRuntime final : public AbstractRuntime {
public:
    ~CpuRuntime() override = default;

    const char* name() const override { return "CPU"; }

    bool initialize(const RuntimeConfig& cfg) override;
    void shutdown() override;

    bool isInitialized() const override { return initialized_; }

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

private:
    bool initialized_ = false;
};
