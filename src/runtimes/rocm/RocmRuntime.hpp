#pragma once

#include "runtimes/AbstractRuntime.hpp"

class RocmRuntime final : public AbstractRuntime {
public:
    bool supportsReconstructionLoss() const override { return isInitialized() && !config_.disabled; }
    bool reconstructionLoss(const RuntimeLossGrad::Request&, RuntimeLossGrad::LossWithGrad&) override;
    bool supportsOptimizer() const override;
    bool optimizerStep(const RuntimeOptimizerRequest&) override;
    ~RocmRuntime() override { shutdown(); }
    const char* name() const override { return "ROCM"; }

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
    RuntimeCapabilityLevel queryForwardCapability(LayerType type) const override;
    RuntimeCapabilityLevel queryBackwardCapability(LayerType type) const override;
    RuntimeCapabilityLevel queryForwardOperationCapability(
        const Layer& layer,
        const std::vector<const std::vector<float>*>& inputs,
        bool training) const override;

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
    struct Impl;
    Impl* impl_ = nullptr;
};
