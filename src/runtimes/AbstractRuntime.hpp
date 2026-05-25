#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

struct Layer;

struct RuntimeConfig {
    // Nom du backend (ex: "CUDA", "ROCM", "VULKAN", "OPENCL").
    std::string backend;

    // Kill-switch runtime (ex: MIMIR_DISABLE_CUDA=1)
    bool disabled = false;

    // Logs best-effort (ex: MIMIR_ACCEL_VERBOSE=1)
    bool verbose = false;

    // Fast-paths (opt-in)
    bool linear_enabled   = false;
    int  linear_min_ops   = 1 << 20;

    bool conv_enabled     = false;
    int  conv_min_ops     = 1 << 18;  // ~256K MACs

    bool norm_enabled         = false;
    int  norm_min_elements    = 1 << 12; // 4096 éléments

    bool attention_enabled  = false;
    int  attention_min_ops  = 1 << 18;  // ~256K MACs

    // Optionnel: sélection de device (ex: MIMIR_CUDA_DEVICE=0)
    int device_index = 0;

    // Construit une config commune depuis les variables d'environnement.
    // Exemples pour backend="CUDA":
    // - MIMIR_DISABLE_CUDA
    // - MIMIR_CUDA_LINEAR
    // - MIMIR_CUDA_LINEAR_MIN_OPS
    // - MIMIR_CUDA_DEVICE
    // - MIMIR_ACCEL_VERBOSE
    static RuntimeConfig fromEnv(const char* backend_upper);
};

class AbstractRuntime {
public:
    virtual ~AbstractRuntime() = default;

    virtual const char* name() const = 0;

    // Initialisation/shutdown du backend. La config est conservée pour pilotage runtime.
    virtual bool initialize(const RuntimeConfig& cfg) = 0;
    virtual void shutdown() = 0;

    virtual bool isInitialized() const = 0;

    const RuntimeConfig& config() const { return config_; }

    // API minimale alignée avec l'usage actuel (fast-path Linear)
    virtual bool linearForward(
        const float* input,
        const float* weights,
        const float* bias_or_null,
        float* output,
        int batch,
        int in_f,
        int out_f
    ) = 0;

    // API générique (objectif: couvrir tous les LayerType via un switch). 
    // - `inputs`: tenseurs float (multi-input)
    // - `outputs`: 1 ou plusieurs tenseurs float produits (Split/Chunk peuvent produire N sorties)
    // Retourne false si le runtime ne supporte pas ce layer.
    virtual bool forwardLayer(
        const std::vector<const std::vector<float>*>& inputs,
        std::vector<std::vector<float>>& outputs,
        const Layer& layer,
        bool training
    ) = 0;

protected:
    RuntimeConfig config_{};
};
