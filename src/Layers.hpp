#pragma once

#include <cstddef>
#include <vector>
#include <string>
#include "LayerTypes.hpp"  // Enum central des types de layers

// Needed for inline access to weight_block storage
#include "tensors.hpp"
#include "DType.hpp"

// ============================================================================
// Énumérations et structures
// ============================================================================

enum class ActivationType {
    NONE,
    RELU,
    RELU6,
    LEAKY_RELU,
    ELU,
    SELU,
    GELU,
    SWISH,
    MISH,
    TANH,
    SIGMOID,
    SOFTMAX,
    SOFTPLUS,
    SOFTSIGN
};

enum class PoolingType {
    MAX,
    AVERAGE,
    MIN,
    SUM
};

enum class PaddingMode {
    ZEROS,
    REFLECT,
    REPLICATE,
    CIRCULAR
};

enum class NormalizationType {
    BATCH_NORM,
    LAYER_NORM,
    INSTANCE_NORM,
    GROUP_NORM,
    RMS_NORM
};

// Type de branche pour les layers avec bifurcations
enum class BranchType {
    NONE,              // Pas de branche
    RESIDUAL,          // Connexion résiduelle (y = F(x) + x)
    SKIP_CONNECTION,   // Skip connection (concaténation)
    DENSE_CONNECTION,  // DenseNet-style (concat avec tous les précédents)
    ATTENTION_BRANCH,  // Branche d'attention parallèle
    MULTI_SCALE,       // Branches multi-échelles (Inception-style)
    GATE,              // Gating mechanism (LSTM-style)
    SPLIT,             // Division en branches parallèles
    MERGE              // Fusion de plusieurs branches
};

// Opération de fusion pour les branches
enum class MergeOperation {
    ADD,               // Addition élément par élément
    MULTIPLY,          // Multiplication élément par élément
    CONCATENATE,       // Concaténation le long du canal
    MAX,               // Maximum élément par élément
    AVERAGE,           // Moyenne élément par élément
    GATED,             // Fusion avec gating
    ATTENTION_WEIGHTED // Fusion pondérée par attention
};

// ============================================================================
// Layer de base
// ============================================================================

struct Layer {
    std::string name;
    std::string type;  // String type for backward compat
    LayerType type_enum = LayerType::UNKNOWN;  // Enum type (NEW)
    size_t params_count;
    Mimir::DType dtype = Mimir::DType::F32;
    Mimir::DType accumulation_dtype = Mimir::DType::F32;
    
    // NOUVEAU: Pointeur vers le tensor de poids unifié pour ce layer
    tensor* weight_block = nullptr;  // Tous les poids du layer dans un seul tensor
    std::string shared_weights_from; // Optional source layer for tied parameters
    
    // Données des paramètres (weights, bias) - conservé pour compatibilité
    std::vector<float> weights;
    std::vector<float> bias;
    
    // Gradients (pour backprop)
    std::vector<float> grad_weights;
    std::vector<float> grad_bias;

    // Les couches Constant sont fixes par défaut. Une Constant marquée
    // trainable_parameter se comporte comme un tenseur paramètre sans entrée.
    bool trainable_parameter = false;
    
    // État interne (pour BatchNorm, etc.)
    std::vector<float> running_mean;
    std::vector<float> running_var;
    
    // ========================================================================
    // PARAMETRES UNIVERSELS (tous les layers)
    // ========================================================================
    
    // === Dimensions de base ===
    int in_features = 0;          // Linear, Embedding; C for token-major CausalDepthwiseConv1d
    int out_features = 0;         // Linear
    int in_channels = 0;          // Conv, Norm
    int out_channels = 0;         // Conv
    
    // === Dimensions spatiales (pour 2D) ===
    int input_height = 0;
    int input_width = 0;
    int output_height = 0;
    int output_width = 0;

    // Conv3d: D,H,W; Conv4d: T,D,H,W. Explicit vectors take precedence over
    // scalar defaults. `shape` may also describe [N,C,...] or [C,...].
    std::vector<int> input_spatial_shape;
    std::vector<int> kernel_shape;
    std::vector<int> stride_shape;
    std::vector<int> padding_shape;
    std::vector<int> dilation_shape;
    int input_depth = 0;
    int input_time = 0;
    int output_depth = 0;
    int output_time = 0;
    
    // === Convolution / Pooling ===
    int kernel_size = 0;          // Kernel unique (carré)
    int kernel_h = 0;             // Kernel height (rectangulaire)
    int kernel_w = 0;             // Kernel width
    int kernel_d = 0;
    int kernel_t = 0;
    int stride = 1;               // Stride unique
    int stride_h = 1;             // Stride height
    int stride_w = 1;             // Stride width
    int stride_d = 1;
    int stride_t = 1;
    int padding = 0;              // Padding unique
    int pad_h = 0;                // Padding height
    int pad_w = 0;                // Padding width
    int pad_d = 0;
    int pad_t = 0;
    int dilation = 1;             // Dilation
    int dilation_h = 1;
    int dilation_w = 1;
    int dilation_d = 1;
    int dilation_t = 1;
    int groups = 1;               // Grouped convolution
    
    // === Normalization ===
    float eps = 1e-5f;            // Epsilon pour stabilité numérique
    int num_groups = 1;           // GroupNorm
    float momentum = 0.1f;        // BatchNorm momentum
    bool affine = true;           // Affine transform (gamma/beta)
    bool track_running_stats = true;  // BatchNorm stats tracking
    
    // === Dropout ===
    float dropout_p = 0.5f;       // Dropout probability
    
    // === Embedding ===
    int vocab_size = 0;           // Embedding vocabulary size
    int embed_dim = 0;            // Embedding dimension
    int padding_idx = -1;         // Padding index for embedding
    
    // === Softmax / LogSoftmax ===
    int axis = -1;                // Axis pour softmax (default: dernier)
    bool use_mask = false;        // Causal mask pour attention

    // === Non-Maximum Suppression ===
    // Entrées: boxes [N,4] en xyxy, scores [N], classes [N] optionnelles.
    // Sortie: indices originaux conservés, encodés en float.
    float nms_iou_threshold = 0.5f;
    float nms_score_threshold = 0.0f;
    int nms_max_detections = 0;       // 0 = aucune limite
    bool nms_class_agnostic = false;   // classes ignorées si true
    
    // === Reshape / View ===
    std::vector<int> target_shape;  // Shape cible pour reshape
    std::vector<int> shape;         // Shape actuel (pour permute, etc.)
    std::vector<int> permute_dims;  // Ordre des dimensions pour permute [0,2,1]
    
    // === Concat / Split ===
    // Explicit graph annotation; -1 preserves ordinary merge semantics.
    int skip_input_index = -1;
    int skipInputIndex() const {
        if (type_enum != LayerType::Concat && type_enum != LayerType::Add) return -1;
        if (skip_input_index >= 0 && static_cast<size_t>(skip_input_index) < inputs.size())
            return skip_input_index;
        // Older checkpoints predate the annotation. Recognize only native
        // encoder-decoder skips, never arbitrary concatenations/residuals.
        const bool legacy_vae_decoder = name.rfind("vae_conv/dec/", 0) == 0 ||
            name.rfind("/dec/", 0) == 0;
        const bool legacy_unet_decoder = name.rfind("unet/up", 0) == 0 ||
            name.rfind("/up", 0) == 0;
        if (type_enum == LayerType::Concat && inputs.size() == 2 &&
            ((legacy_vae_decoder && name.size() >= 9 &&
              name.compare(name.size() - 9, 9, "/skip_cat") == 0) ||
             (legacy_unet_decoder && name.size() >= 7 &&
              name.compare(name.size() - 7, 7, "/concat") == 0))) return 1;
        return -1;
    }
    int concat_axis = 1;               // Axis de concatenation (default: channels)
    int num_splits = 2;                // Nombre de splits (splits égaux si split_sizes vide)
    std::vector<int> split_sizes;      // Tailles explicites de chaque split (si vide: splits égaux)
    int split_axis = 0;                // Axis de split (default: 0 = batch dimension)
    
    // === Upsample ===
    float scale_h = 2.0f;         // Scale factor height
    float scale_w = 2.0f;         // Scale factor width
    int out_h = 0;                // Output height explicit
    int out_w = 0;                // Output width explicit
    
    // === Attention ===
    int num_heads = 8;            // Multi-head attention
    int num_kv_heads = 0;         // 0 = num_heads; smaller enables GQA
    int head_dim = 64;            // Dimension par head
    int seq_len = 0;              // Sequence length pour attention
    bool causal = false;          // Causal mask
    float rope_theta = 0.0f;      // > 0 applies RoPE to Q and K

    // === Custom: PatchEmbed (projection learnable des patches) ===
    int patch_dim = 0;            // Dimension d'un patch aplati (p*p)
    int num_patches = 0;          // Nombre total de patches
    int seq_text = 0;             // Longueur de la partie texte (sans le time token)
    
    // === Activation ===
    float alpha = 0.01f;          // LeakyReLU alpha
    float negative_slope = 0.01f; // Alias pour alpha
    float leaky_relu_alpha = 0.01f;  // LeakyReLU alpha (explicit)
    
    // === Shape Operations (Strict Mode) ===
    int squeeze_dim = -1;         // Dimension à squeeze (-1 = auto)
    int unsqueeze_dim = -1;       // Dimension où unsqueeze
    int num_chunks = 2;           // Nombre de chunks pour Chunk operation
    int stack_axis = 0;           // Axis pour Stack operation
    
    // === Bias ===
    bool use_bias = true;
    
    // ========================================================================
    // MULTI-INPUT / TENSOR ROUTING (Nouveau système)
    // ========================================================================
    
    std::vector<std::string> inputs;   // Noms des tensors d'entrée (vide = {"x"} par défaut)
    std::string output = "x";          // Nom du tensor de sortie (default "x")
    
    // ========================================================================
    // CONFIGURATION DES BRANCHES
    // ========================================================================
    
    ActivationType activation = ActivationType::NONE;
    float activation_param = 0.0f; // Pour LeakyReLU alpha, ELU alpha, etc.
    
    // Configuration des branches (pour les architectures avec skip connections, etc.)
    BranchType branch_type = BranchType::NONE;
    MergeOperation merge_op = MergeOperation::ADD;
    std::vector<int> branch_sources;  // Indices des layers sources pour les branches
    int branch_target = -1;           // Indice du layer cible pour cette branche
    bool is_branch_point = false;     // Ce layer est un point de bifurcation
    bool is_merge_point = false;      // Ce layer est un point de fusion
    
    Layer() = default;
    
    Layer(const std::string& n, const std::string& t, size_t pc)
        : name(n), type(t), params_count(pc) {
        // Convertir string -> enum automatiquement
        std::string normalized = LayerRegistry::normalize_type(t);
        type_enum = LayerRegistry::string_to_type(normalized);
        type = normalized;  // Stocker le type normalisé
    }
    
    // Accesseur pour récupérer les données du weight_block
    inline float* getWeights() {
        if (weight_block) return weight_block->getData();
        return weights.data();
    }

    inline const float* getWeights() const {
        if (weight_block) return weight_block->getData();
        return weights.data();
    }

    inline size_t getWeightsSize() const {
        // Hot-path: when using unified weight blocks, params_count is the authoritative size.
        // This avoids calling tensor::getSize() in tight loops.
        if (weight_block) return params_count ? params_count : weight_block->getSize();
        return weights.size();
    }
    
    // Helper: obtenir kernel effectif (kernel_size prend priorité)
    int get_kernel_h() const { return kernel_h > 0 ? kernel_h : (kernel_size > 0 ? kernel_size : 3); }
    int get_kernel_w() const { return kernel_w > 0 ? kernel_w : (kernel_size > 0 ? kernel_size : 3); }
    // NOTE: stride_h/stride_w valent 1 par défaut. Si on n'a configuré que `stride`,
    // on doit utiliser `stride` (sinon le stride reste bloqué à 1).
    int get_stride_h() const { return (stride_h != 1) ? stride_h : stride; }
    int get_stride_w() const { return (stride_w != 1) ? stride_w : stride; }
    int get_pad_h() const { return pad_h > 0 ? pad_h : padding; }
    int get_pad_w() const { return pad_w > 0 ? pad_w : padding; }
    
    // Détection automatique du type de branche basé sur le nom du layer
    void detectBranchType() {
        if (name.find("residual") != std::string::npos || 
            name.find("shortcut") != std::string::npos ||
            name.find("skip") != std::string::npos) {
            branch_type = BranchType::RESIDUAL;
            merge_op = MergeOperation::ADD;
        }
        else if (name.find("concat") != std::string::npos) {
            branch_type = BranchType::SKIP_CONNECTION;
            merge_op = MergeOperation::CONCATENATE;
        }
        else if (name.find("dense") != std::string::npos && name.find("connect") != std::string::npos) {
            branch_type = BranchType::DENSE_CONNECTION;
            merge_op = MergeOperation::CONCATENATE;
        }
        else if (name.find("attention") != std::string::npos && name.find("branch") != std::string::npos) {
            branch_type = BranchType::ATTENTION_BRANCH;
            merge_op = MergeOperation::ATTENTION_WEIGHTED;
        }
        else if (name.find("inception") != std::string::npos || name.find("multiscale") != std::string::npos) {
            branch_type = BranchType::MULTI_SCALE;
            merge_op = MergeOperation::CONCATENATE;
        }
        else if (name.find("gate") != std::string::npos) {
            branch_type = BranchType::GATE;
            merge_op = MergeOperation::GATED;
        }
        else if (name.find("split") != std::string::npos) {
            branch_type = BranchType::SPLIT;
            is_branch_point = true;
        }
        else if (name.find("merge") != std::string::npos || name.find("fusion") != std::string::npos) {
            branch_type = BranchType::MERGE;
            is_merge_point = true;
            // L'opération de fusion par défaut peut être précisée
            if (name.find("add") != std::string::npos) {
                merge_op = MergeOperation::ADD;
            } else if (name.find("mul") != std::string::npos) {
                merge_op = MergeOperation::MULTIPLY;
            } else if (name.find("concat") != std::string::npos) {
                merge_op = MergeOperation::CONCATENATE;
            }
        }
    }
    
    // Détermine si ce layer nécessite un calcul de branche
    bool requiresBranchComputation() const {
        return branch_type != BranchType::NONE || is_branch_point || is_merge_point;
    }
};
