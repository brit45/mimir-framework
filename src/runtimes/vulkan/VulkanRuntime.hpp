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

    bool supportsForwardLayerType(LayerType type) const override;
    RuntimeCapabilityLevel queryForwardCapability(LayerType type) const override;
    RuntimeCapabilityLevel queryBackwardCapability(LayerType type) const override;

    // Initial residency slice: one upload, N native Vulkan unary kernels and
    // one final download. No intermediate std::vector is materialized.
    bool unaryChainForwardResident(
        const float* input,
        float* output,
        int elements,
        const std::vector<LayerType>& operations
    );

private:
    struct Impl;
    Impl* impl_ = nullptr;
};
