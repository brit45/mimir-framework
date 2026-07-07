#pragma once

#include "runtimes/AbstractRuntime.hpp"

class VulkanRuntime final : public AbstractRuntime {
public:
    ~VulkanRuntime() override { shutdown(); }

    const char* name() const override { return "VULKAN"; }

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

private:
    struct Impl;
    Impl* impl_ = nullptr;
};
