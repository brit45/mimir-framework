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

    bool linearForwardTyped(
        const Mimir::TypedTensor& input,
        const Mimir::TypedTensor& weights,
        const Mimir::TypedTensor* bias_or_null,
        Mimir::TypedTensor& output
    ) override;

    bool mergeBranches(const std::vector<float>& left,
                       const std::vector<float>& right,
                       std::vector<float>& output, const Layer& layer) override;

    bool forwardLayer(
        const std::vector<const std::vector<float>*>& inputs,
        std::vector<std::vector<float>>& outputs,
        const Layer& layer,
        bool training
    ) override;

    bool forwardLayerWithContext(
        const std::vector<const std::vector<float>*>& inputs,
        std::vector<std::vector<float>>& outputs, const Layer& layer, bool training,
        const RuntimeForwardContext& context = {}) override;

    bool supportsForwardLayerType(LayerType type) const override;
    bool supportsBackwardLayerType(LayerType type) const override;
    RuntimeCapabilityLevel queryForwardCapability(LayerType type) const override;
    RuntimeCapabilityLevel queryBackwardCapability(LayerType type) const override;

    bool backwardLayer(
        const std::vector<const std::vector<float>*>& inputs,
        const std::vector<const std::vector<float>*>& grad_outputs,
        std::vector<std::vector<float>>& grad_inputs,
        Layer& layer,
        bool training
    ) override;

    RuntimeCapabilityLevel queryPointwiseFusionCapability(
        const std::vector<const Layer*>&,bool backward,bool training) const override;
    bool pointwiseFusionForward(const std::vector<const Layer*>&,const std::vector<float>&,
        std::vector<std::vector<float>>&,bool retain,bool training) override;
    bool pointwiseFusionBackward(const std::vector<const Layer*>&,
        const std::vector<const std::vector<float>*>&,const std::vector<float>&,std::vector<float>&) override;

private:
    bool initialized_ = false;
};
