#pragma once

#include "ScriptingContext.hpp"
#include "AsyncMonitor.hpp"
#include "runtimes/DistributedRuntime.hpp"

namespace Mimir::Scripting {
// Called only by the scripting/model thread. The UI exchanges requests and
// immutable snapshots; it never owns or reads a live model.
inline void bindModelProjection(ScriptingContext& context, bool new_session = false,
                                bool capture_current = false) {
    if (!context.asyncMonitor || !context.asyncMonitor->isRunning()) {
        if (context.currentModel) context.currentModel->bindLatentSpaceControl(nullptr);
        return;
    }
    if (!context.currentModel) {
        if (new_session) context.asyncMonitor->configureLatentSpace(json::object());
        return;
    }
    auto& model = *context.currentModel;
    if (!DistributedRuntime::instance().isMainProcess()) {
        model.bindLatentSpaceControl(nullptr);
        return;
    }
    auto control = context.asyncMonitor->latentSpaceControl();
    if (new_session || !control->available.load(std::memory_order_acquire)) {
        auto config = model.modelConfig;
        if (!config.is_object()) config = json::object();
        if (!config.contains("type")) config["type"] = context.modelType;
        config["model_name"] = model.getModelName();
        context.asyncMonitor->configureLatentSpace(config);
    }
    model.bindLatentSpaceControl(control);
    if (capture_current && context.asyncMonitor->getViz())
        model.captureRequestedLatentSpace(model.latentSpaceOptimizerStep(), true);
}

// Covers all returns of the shared training entry point, including specialized
// VAE/GAN/diffusion branches. Capture runs before leaving the model thread.
class ProjectionTrainingScope {
public:
    explicit ProjectionTrainingScope(ScriptingContext& context)
        : context_(context), model_(context.currentModel) {
        bindModelProjection(context_);
        if (context_.asyncMonitor && context_.asyncMonitor->isRunning() && model_ &&
            DistributedRuntime::instance().isMainProcess()) {
            control_ = context_.asyncMonitor->latentSpaceControl();
            control_->training_active.store(true, std::memory_order_release);
        }
    }
    ~ProjectionTrainingScope() {
        if (!model_ || !control_) return;
        control_->training_active.store(false, std::memory_order_release);
        model_->captureRequestedLatentSpace(model_->latentSpaceOptimizerStep(),
            context_.asyncMonitor && context_.asyncMonitor->getViz());
    }
private:
    ScriptingContext& context_;
    std::shared_ptr<Model> model_;
    std::shared_ptr<mimir::viz::LatentSpaceControl> control_;
};
} // namespace Mimir::Scripting
