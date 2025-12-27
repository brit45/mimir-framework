#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include "include/json.hpp"
#include "Helpers.hpp"    // contient MagicToken, DatasetItem, loadDataset, imageToEmbedding, write_u32_le
#include "tensors.hpp"
#include "Tokenizer.hpp"
#include "Encoder.hpp"
#include "Autograd.hpp"   // Pour la structure Gradients
#include "HardwareOpt.hpp" // Optimisations hardware avancées
#include "Layers.hpp"      // Pour la structure Layer

using json = nlohmann::json;
namespace fs = std::filesystem;

// LR Decay strategies
enum class LRDecayStrategy {
    NONE,           // Pas de decay
    COSINE,         // Cosine annealing
    STEP,           // Step decay (réduction par paliers)
    EXPONENTIAL,    // Exponential decay
    LINEAR          // Linear decay
};

// Optimizer types
enum class OptimizerType {
    SGD,      // Stochastic Gradient Descent
    ADAM,     // Adam optimizer
    ADAMW     // Adam with weight decay (L2 regularization)
};

// Optimizer (Adam-like) state with LR decay
struct Optimizer {
    OptimizerType type = OptimizerType::ADAM;
    std::vector<float> m;
    std::vector<float> v;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.01f;  // Pour AdamW
    size_t step = 0;
    
    // LR Decay parameters
    LRDecayStrategy decay_strategy = LRDecayStrategy::COSINE;
    float initial_lr = 5e-5f;
    float min_lr = 1e-6f;          // Learning rate minimum
    float decay_rate = 0.95f;       // Pour exponential/step decay
    int decay_steps = 100;          // Nombre de steps entre chaque decay (step decay)
    int total_steps = 1000;         // Total steps pour cosine/linear
    int warmup_steps = 0;           // Warmup optionnel
    
    void ensure(size_t n) {
        if (m.size() < n) m.resize(n, 0.0f);
        if (v.size() < n) v.resize(n, 0.0f);
    }
    
    // Calcule le learning rate actuel avec decay
    float getCurrentLR() const {
        if (step < warmup_steps) {
            // Warmup linéaire
            return initial_lr * (static_cast<float>(step) / warmup_steps);
        }
        
        int effective_step = step - warmup_steps;
        int effective_total = total_steps - warmup_steps;
        
        switch (decay_strategy) {
            case LRDecayStrategy::NONE:
                return initial_lr;
                
            case LRDecayStrategy::COSINE: {
                // Cosine annealing: lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(π * t / T))
                float progress = std::min(1.0f, static_cast<float>(effective_step) / effective_total);
                float cosine = 0.5f * (1.0f + std::cos(3.14159265359f * progress));
                return min_lr + (initial_lr - min_lr) * cosine;
            }
            
            case LRDecayStrategy::STEP: {
                // Step decay: lr *= decay_rate chaque decay_steps
                int num_decays = effective_step / decay_steps;
                return std::max(min_lr, initial_lr * std::pow(decay_rate, static_cast<float>(num_decays)));
            }
            
            case LRDecayStrategy::EXPONENTIAL: {
                // Exponential decay: lr = initial_lr * decay_rate^(step / decay_steps)
                float exponent = static_cast<float>(effective_step) / decay_steps;
                return std::max(min_lr, initial_lr * std::pow(decay_rate, exponent));
            }
            
            case LRDecayStrategy::LINEAR: {
                // Linear decay: lr décroit linéairement de initial_lr à min_lr
                float progress = std::min(1.0f, static_cast<float>(effective_step) / effective_total);
                return initial_lr - (initial_lr - min_lr) * progress;
            }
            
            default:
                return initial_lr;
        }
    }
};

// -------------------- Model class --------------------
class Model {
public:
    Model();
    ~Model();

    void setDensity(double d);
    double getDensity() const;

    void build();
    void autoBuildFromDataset(const std::string &dataset_dir);

    // topology hooks (to be overridden)
    virtual void buildBackboneUNet(int stages, int blocks_per_stage, int bottleneck_depth);
    virtual void injectMagicToken(const MagicToken &tok);
    virtual void buildTextBranch(const MagicToken &tok);
    virtual void buildAudioBranch(const MagicToken &tok);
    virtual void buildImageBranch(const MagicToken &tok);
    virtual void buildVideoBranch(const MagicToken &tok);

    std::vector<uint16_t> getWeights() const;
    void setTokenizer(const Tokenizer &t);
    void setEncoder(const Encoder &e);

    size_t totalParamCount() const;
    void allocateParams();
    void initializeWeights(const std::string &method = "xavier", unsigned int seed = 0);
    void updateWeightsWithNoise(float learning_rate, float noise_std = 0.01f);
    void forward(std::vector<uint8_t> &) const;
    
    // Nouveau forward/backward pass complet
    std::vector<float> forwardPass(const std::vector<float> &input, bool training = true);
    Gradients backwardPass(const std::vector<float> &loss_gradient);
    void zeroGradients();  // Réinitialise tous les gradients à zéro
    Gradients getGradients() const;  // Récupère les gradients actuels
    float computeLoss(const std::vector<float> &prediction, const std::vector<float> &target, const std::string &loss_type = "mse");
    std::vector<float> computeLossGradient(const std::vector<float> &prediction, const std::vector<float> &target, const std::string &loss_type = "mse");
    
    void push(const std::string &name, const std::string &type, size_t params_count);
    void setOutputTarget(const std::vector<uint8_t> &target);
    void applyParamUpdate(float learning_rate);
    void optimizerStep(Optimizer &opt, float learning_rate, const Gradients* gradients = nullptr);

    struct DecoderOutput {
        std::vector<int> tokens;
        double mse = -1.0;
        std::vector<float> logits;
    };


    DecoderOutput eval(const std::vector<uint8_t> &target) const;
    void setLastEncoding(const std::vector<float> &e);

    // accessors used elsewhere
    int width() const { return tw; }
    int height() const { return th; }
    const Tokenizer &getTokenizer() const { return tokenizer; }
    Tokenizer &getMutableTokenizer() { return tokenizer; }
    const Encoder &getEncoder() const { return encoder; }
    Encoder &getMutableEncoder() { return encoder; }
    std::vector<tensor>& getMutableParams() { return params; }  // Accès aux paramètres

    // static helpers for saving/loading
    bool saveCheckpoint(const Tokenizer &tokenizer, const std::vector<MagicToken> &magic_tokens, const fs::path &dir, int epoch);
    bool packToSafetensor(const fs::path &outpath, const std::unordered_map<std::string, std::vector<float>> &tensors) const;
    bool tryLoadExistingModel(const fs::path &ckdir, const fs::path &safep, Tokenizer &outTok, Encoder &outEnc, std::vector<MagicToken> &outMagic);
    
    // Nouvelles méthodes pour sauvegarder/charger la structure complète
    bool saveLayersStructure(const fs::path &filepath) const;
    bool loadLayersStructure(const fs::path &filepath);
    bool saveEmbeddings(const fs::path &filepath) const;
    bool loadEmbeddings(const fs::path &filepath);
    bool saveParamsData(const fs::path &filepath) const;
    bool loadParamsData(const fs::path &filepath);

    // ============================= 
    //           Helpers
    // =============================

    // fonction utilitaire : convertit Weight (uint16) -> float [-1,1]
    static inline float weightToFloat(uint16_t w)
    {
        return (static_cast<float>(w) / 65535.0f) * 2.0f - 1.0f;
    }

    static inline float sigmoidf(float v) { return 1.0f / (1.0f + std::exp(-v)); }
    
    // =============================
    // Hardware Acceleration
    // =============================
    
    // Détection des capacités CPU au runtime
    static bool hasAVX2();
    static bool hasFMA();
    static bool hasF16C();
    static bool hasBMI2();
    
    bool hasVulkanCompute() const;
    bool initializeComputeEngine();
    void shutdownComputeEngine();
    
    // =============================
    // Layer Operations (Hardware/Software Dispatch)
    // =============================
    
    // Structure pour paramètres de layer
    struct LayerParams {
        std::vector<float> weights;
        std::vector<float> bias;
        int in_features = 0;
        int out_features = 0;
        int kernel_size = 3;
        int stride = 1;
        int padding = 0;
        int dilation = 1;
        int groups = 1;
        bool use_hardware = true;  // Dynamic dispatch
    };
    
    // Convolution 2D avec dispatch hardware/software
    static void computeConv2D(const std::vector<float>& input, std::vector<float>& output,
                             const LayerParams& params, int in_h, int in_w, int in_c, int out_c,
                             bool use_hardware = true);
    
    // Linear/Dense avec dispatch
    static void computeLinear(const std::vector<float>& input, std::vector<float>& output,
                             const LayerParams& params, bool use_hardware = true);
    
    // Pooling avec dispatch
    static void computeMaxPool2D(const std::vector<float>& input, std::vector<float>& output,
                                int in_h, int in_w, int channels, int kernel_size, int stride,
                                bool use_hardware = true);
    
    static void computeAvgPool2D(const std::vector<float>& input, std::vector<float>& output,
                                int in_h, int in_w, int channels, int kernel_size, int stride,
                                bool use_hardware = true);
    
    // Activation avec dispatch
    static void computeActivation(std::vector<float>& data, const std::string& activation_type,
                                 float param = 0.0f, bool use_hardware = true);
    
    // Batch Normalization avec dispatch
    static void computeBatchNorm(std::vector<float>& data, const std::vector<float>& gamma,
                                const std::vector<float>& beta, const std::vector<float>& running_mean,
                                const std::vector<float>& running_var, int batch_size, int channels,
                                int spatial_size, float eps = 1e-5f, bool training = false,
                                bool use_hardware = true);
    
    // Layer Normalization avec dispatch
    static void computeLayerNorm(std::vector<float>& data, const std::vector<float>& gamma,
                                const std::vector<float>& beta, int normalized_size,
                                float eps = 1e-5f, bool use_hardware = true);
    
    // Transpose Convolution avec dispatch
    static void computeConvTranspose2D(const std::vector<float>& input, std::vector<float>& output,
                                      const LayerParams& params, int in_h, int in_w, int in_c, int out_c,
                                      bool use_hardware = true);
    
    // Attention mechanism (pour transformers)
    static void computeAttention(const std::vector<float>& query, const std::vector<float>& key,
                                const std::vector<float>& value, std::vector<float>& output,
                                int seq_len, int d_model, int num_heads, bool use_hardware = true);
    
    // =============================
    // Branch Operations (pour résiduals, skip connections, etc.)
    // =============================
    
    // Calcul des opérations de branche avec dispatch automatique
    static void computeBranchMerge(const std::vector<float>& branch1, 
                                   const std::vector<float>& branch2,
                                   std::vector<float>& output,
                                   MergeOperation merge_op,
                                   bool use_hardware = true);
    
    // Split d'un tensor en plusieurs branches
    static void computeBranchSplit(const std::vector<float>& input,
                                   std::vector<std::vector<float>>& outputs,
                                   const std::vector<int>& split_sizes);
    
    // Détection et exécution automatique des branches pendant forward/backward
    void detectAndSetupBranches();
    void executeBranchComputation(int layer_idx, 
                                  std::vector<std::vector<float>>& layer_outputs,
                                  bool training = false);
    
    // Gestion des gradients pour les branches pendant le backward pass
    void backpropThroughBranch(int layer_idx,
                              const std::vector<float>& grad_output,
                              std::vector<std::vector<float>>& layer_gradients);
    
    // Configuration globale hardware
    static inline bool global_use_hardware = true;
    static void setHardwareAcceleration(bool enable) { global_use_hardware = enable; }

    // Configuration du modèle (pour dimensionnement dynamique des layers)
    json modelConfig;

    static void conv2d_same(const std::vector<float> &in, std::vector<float> &out, int W, int H, const std::vector<float> &kernel, int ksize);

    static inline void add_inplace(std::vector<float> &a, const std::vector<float> &b)
    {
        const size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; ++i)
            a[i] += b[i];
    }

    static inline void relu_inplace(std::vector<float> &x)
    {
        for (auto &v : x)
            if (v < 0.0f)
                v = 0.0f;
    }

    static inline void leaky_relu_inplace(std::vector<float> &x, float slope = 0.01f)
    {
        for (auto &v : x)
            if (v < 0.0f)
                v *= slope;
    }

    static inline void tanh_inplace(std::vector<float> &x)
    {
        for (auto &v : x)
            v = std::tanh(v);
    }

    static inline void elu_inplace(std::vector<float> &x, float alpha = 1.0f)
    {
        for (auto &v : x)
            v = (v >= 0.0f) ? v : alpha * (std::exp(v) - 1.0f);
    }

    // Sigmoid « image » (centre 128, échelle 32 -> [0..255])
    static inline void sigmoid_image_inplace(std::vector<float> &x)
    {
        for (auto &v : x)
        {
            float z = (v - 128.0f) / 32.0f;
            v = 255.0f * sigmoidf(z);
        }
    }

    // Softmax in-place (stable) sur un petit vecteur
    static inline void softmax_inplace(std::vector<float> &x)
    {
        if (x.empty())
            return;
        float m = *std::max_element(x.begin(), x.end());
        float s = 0.f;
        for (auto &v : x)
        {
            v = std::exp(v - m);
            s += v;
        }
        for (auto &v : x)
            v /= s;
    }

    // BatchNorm « globale » : (x-mean)/std * gamma + beta
    static inline void batchnorm_global_inplace(std::vector<float> &x, float gamma, float beta, float eps = 1e-5f)
    {
        if (x.empty())
            return;
        double mean = 0.0;
        for (float v : x)
            mean += v;
        mean /= double(x.size());
        double var = 0.0;
        for (float v : x)
        {
            double d = v - mean;
            var += d * d;
        }
        var = var / double(x.size());
        float inv_std = 1.0f / std::sqrt(float(var) + eps);
        for (auto &v : x)
            v = gamma * ((v - float(mean)) * inv_std) + beta;
    }

    // ANCIEN: std::vector<tensor> params;  // 1 paramètre = 1 tensor
    // NOUVEAU: Chaque layer a son propre bloc de poids (weight_block)
    std::vector<tensor> layer_weight_blocks;  // 1 tensor = tous les poids d'un layer
    
    // Rétrocompatibilité: accès aux paramètres via une interface unifiée
    std::vector<tensor> params;  // Conservé temporairement pour transition

    void setName(std::string name) {

        model_name = name;
    }
    
    // Activations du forward pass (pour le backward)
    struct ForwardState {
        std::vector<std::vector<float>> layer_inputs;
        std::vector<std::vector<float>> layer_outputs;
        std::vector<std::vector<float>> activations;
        std::vector<float> final_output;
        bool is_valid = false;
        
        void clear() {
            layer_inputs.clear();
            layer_outputs.clear();
            activations.clear();
            final_output.clear();
            is_valid = false;
        }
    };
    ForwardState forward_state;

protected:
    std::vector<Layer> layers;
    int tw = 64, th = 64;
    Tokenizer tokenizer;
    Encoder encoder;
    bool hasTokenizer = false;
    bool hasEncoder = false;
    std::vector<float> lastEncoding;
    double densityFactor = 1.0;
    std::string model_name;
};