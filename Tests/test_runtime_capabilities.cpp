#include "test_utils.hpp"

#include "Layers.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"
#include "runtimes/opencl/OpenCLRuntime.hpp"
#include "runtimes/vulkan/VulkanRuntime.hpp"

class CapabilityRuntime final : public AbstractRuntime {
public:
    CapabilityRuntime(const char* runtime_name,
                      RuntimeCapabilityLevel forward,
                      RuntimeCapabilityLevel backward)
        : name_(runtime_name), forward_(forward), backward_(backward) {}

    const char* name() const override { return name_; }
    bool initialize(const RuntimeConfig& cfg) override { config_ = cfg; initialized_ = true; return true; }
    void shutdown() override { initialized_ = false; }
    bool isInitialized() const override { return initialized_; }
    bool linearForward(const float*, const float*, const float*, float*, int, int, int) override { return false; }
    bool forwardLayer(const std::vector<const std::vector<float>*>& inputs,
                      std::vector<std::vector<float>>& outputs,
                      const Layer&, bool) override {
        if (!runtimeCapabilityIsNative(forward_) || inputs.empty() || !inputs[0]) return false;
        outputs = {{static_cast<float>(name_[0]), inputs[0]->front()}};
        return true;
    }
    bool supportsForwardLayerType(LayerType type) const override {
        return type == LayerType::Linear && forward_ != RuntimeCapabilityLevel::Unsupported;
    }
    bool supportsBackwardLayerType(LayerType type) const override {
        return type == LayerType::Linear && backward_ != RuntimeCapabilityLevel::Unsupported;
    }
    RuntimeCapabilityLevel queryForwardCapability(LayerType type) const override {
        return type == LayerType::Linear ? forward_ : RuntimeCapabilityLevel::Unsupported;
    }
    RuntimeCapabilityLevel queryBackwardCapability(LayerType type) const override {
        return type == LayerType::Linear ? backward_ : RuntimeCapabilityLevel::Unsupported;
    }

private:
    const char* name_;
    RuntimeCapabilityLevel forward_;
    RuntimeCapabilityLevel backward_;
    bool initialized_ = false;
};

int main() {
    CpuRuntime concrete_cpu;
    TASSERT_TRUE(concrete_cpu.queryForwardCapability(LayerType::Linear) ==
                 RuntimeCapabilityLevel::Native);

#ifdef ENABLE_VULKAN
    VulkanRuntime concrete_vulkan;
    TASSERT_TRUE(concrete_vulkan.queryForwardCapability(LayerType::Add) ==
                 RuntimeCapabilityLevel::NativeOptimized);
    TASSERT_TRUE(concrete_vulkan.queryForwardCapability(LayerType::Subtract) ==
                 RuntimeCapabilityLevel::HostFallback);
    TASSERT_TRUE(concrete_vulkan.queryBackwardCapability(LayerType::Linear) ==
                 RuntimeCapabilityLevel::Unsupported);
#endif

#ifdef ENABLE_OPENCL
    OpenCLRuntime concrete_opencl;
    TASSERT_TRUE(concrete_opencl.queryForwardCapability(LayerType::Add) ==
                 RuntimeCapabilityLevel::Native);
    TASSERT_TRUE(concrete_opencl.queryBackwardCapability(LayerType::Add) ==
                 RuntimeCapabilityLevel::HostFallback);
#endif

    CapabilityRuntime cuda("CUDA", RuntimeCapabilityLevel::HostFallback,
                           RuntimeCapabilityLevel::HostFallback);
    CapabilityRuntime vulkan("VULKAN", RuntimeCapabilityLevel::Native,
                             RuntimeCapabilityLevel::Unsupported);
    CapabilityRuntime cpu("CPU", RuntimeCapabilityLevel::Native,
                          RuntimeCapabilityLevel::Native);
    RuntimeConfig cfg;
    TASSERT_TRUE(cuda.initialize(cfg));
    TASSERT_TRUE(vulkan.initialize(cfg));
    TASSERT_TRUE(cpu.initialize(cfg));

    Layer linear;
    linear.type_enum = LayerType::Linear;

    auto& router = RuntimeRouter::instance();
    router.setRuntimes(nullptr, &cuda, &vulkan, nullptr, nullptr, &cpu);

    // A higher-priority host fallback must not hide a lower native route.
    TASSERT_TRUE(router.selectForwardRuntimeForLayer(linear) == &vulkan);
    TASSERT_TRUE(router.selectBackwardRuntimeForLayer(linear) == &cpu);

    const std::vector<float> input = {3.0f};
    const std::vector<const std::vector<float>*> inputs = {&input};
    std::vector<std::vector<float>> outputs;
    AbstractRuntime* executed = nullptr;
    TASSERT_TRUE(router.dispatchForwardLayerPlanned(
        &cpu, inputs, outputs, linear, false, &executed));
    TASSERT_TRUE(executed == &cpu);
    TASSERT_TRUE(outputs.size() == 1 && outputs[0].size() == 2);

    // Compatibility mode may still select a host fallback when no native
    // implementation exists.
    router.setRuntimes(nullptr, &cuda, nullptr, nullptr, nullptr, nullptr);
    TASSERT_TRUE(router.selectForwardRuntimeForLayer(linear) == nullptr);
    TASSERT_TRUE(router.selectForwardRuntimeForLayer(linear, true) == &cuda);

    return 0;
}
