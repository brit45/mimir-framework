# 📚 Référence des Layers - Mímir Framework v2.1

Ce document liste tous les types de layers supportés par le système unifié.

---

## 🎯 Légende

- ✅ **Implémenté** : Forward pass complet avec optimisations
- 🔄 **Préservé** : Code original conservé de l'ancien système
- ⚠️  **Pass-through** : Non implémenté, pass-through avec warning
- 🔜 **Planifié** : À implémenter prochainement

---

## 📋 Liste Complète (67 Types)

### 🔲 Convolutional Layers

| Type | Status | Optimisations | Notes |
|------|--------|---------------|-------|
| `Conv2d` | 🔄 Préservé | OpenMP | 2D convolution standard |
| `ConvTranspose2d` | 🔄 Préservé | OpenMP | Transposed convolution (upsampling) |
| `Conv1d` | ⚠️ Pass-through | - | 1D convolution |
| `Conv3d` | ⚠️ Pass-through | - | 3D convolution |
| `ConvDepthwise` | ⚠️ Pass-through | - | Depthwise separable convolution |

**Paramètres:**
- `kernel_h`, `kernel_w` (ou `kernel_size`)
- `stride_h`, `stride_w` (ou `stride`)
- `pad_h`, `pad_w` (ou `padding`)
- `dilation`, `groups`
- `in_channels`, `out_channels`

**Exemple Lua:**
```lua
Mimir.Model.push_layer("conv1", "Conv2d", params_count)
-- Configuration automatique via architectures
```

---

### 📊 Normalization Layers

| Type | Status | Optimisations | Notes |
|------|--------|---------------|-------|
| `BatchNorm2d` | 🔄 Préservé | OpenMP | Batch normalization 2D |
| `BatchNorm1d` | 🔄 Préservé | OpenMP | Batch normalization 1D |
| `LayerNorm` | ✅ Implémenté | OpenMP | Layer normalization (complète) |
| `GroupNorm` | ✅ Implémenté | OpenMP | Group normalization |
| `InstanceNorm` | ⚠️ Pass-through | - | Instance normalization |
| `RMSNorm` | ⚠️ Pass-through | - | Root Mean Square normalization |

**Paramètres:**
- `eps` (epsilon, défaut: 1e-5)
- `num_groups` (pour GroupNorm, défaut: 32)
- `affine` (learnable γ, β, défaut: true)
- `momentum` (pour BatchNorm, défaut: 0.1)
- `track_running_stats` (défaut: true)

**Exemple Lua:**
```lua
Mimir.Model.push_layer("ln", "LayerNorm", params_count)
Mimir.Model.push_layer("gn", "GroupNorm", params_count)
```

---

### 🎯 Activation Layers

| Type | Status | Optimisations | Notes |
|------|--------|---------------|-------|
| `ReLU` | ✅ Implémenté | AVX2, OpenMP | Rectified Linear Unit |
| `GELU` | ✅ Implémenté | OpenMP | Gaussian Error Linear Unit |
| `SiLU` | ✅ Implémenté | OpenMP | Sigmoid Linear Unit (Swish) |
| `Tanh` | ✅ Implémenté | OpenMP | Hyperbolic tangent |
| `Sigmoid` | ✅ Implémenté | OpenMP | Sigmoid activation |
| `Softmax` | ✅ Implémenté | OpenMP | Softmax (numerically stable) |
| `LogSoftmax` | ✅ Implémenté | OpenMP | Log-Softmax |
| `LeakyReLU` | ⚠️ Pass-through | - | Leaky ReLU |
| `ELU` | ⚠️ Pass-through | - | Exponential Linear Unit |
| `PReLU` | ⚠️ Pass-through | - | Parametric ReLU |
| `Mish` | ⚠️ Pass-through | - | Mish activation |

**Paramètres:**
- `alpha` (pour LeakyReLU, PReLU, ELU)
- `axis` (pour Softmax, défaut: -1)

**Exemple Lua:**
```lua
Mimir.Model.push_layer("relu", "ReLU", 0)
Mimir.Model.push_layer("gelu", "GELU", 0)
Mimir.Model.push_layer("silu", "SiLU", 0)
```

---

### 🏊 Pooling Layers

| Type | Status | Optimisations | Notes |
|------|--------|---------------|-------|
| `MaxPool2d` | 🔄 Préservé | OpenMP | Max pooling 2D |
| `AvgPool2d` | ✅ Implémenté | OpenMP | Average pooling 2D |
| `GlobalAvgPool2d` | ✅ Implémenté | OpenMP | Global average pooling |
| `AdaptiveAvgPool2d` | ✅ Implémenté | OpenMP | Adaptive average pooling |
| `AdaptiveMaxPool2d` | ⚠️ Pass-through | - | Adaptive max pooling |
| `MaxPool1d` | ⚠️ Pass-through | - | Max pooling 1D |
| `AvgPool1d` | ⚠️ Pass-through | - | Average pooling 1D |

**Paramètres:**
- `kernel_h`, `kernel_w` (ou `kernel_size`)
- `stride_h`, `stride_w` (ou `stride`)
- `pad_h`, `pad_w` (ou `padding`)
- `out_h`, `out_w` (pour Adaptive)

**Exemple Lua:**
```lua
Mimir.Model.push_layer("maxpool", "MaxPool2d", 0)
Mimir.Model.push_layer("globalavg", "GlobalAvgPool2d", 0)
```

---

### 🔗 Linear Layers

| Type | Status | Optimisations | Notes |
|------|--------|---------------|-------|
| `Linear` | ✅ Implémenté | AVX2, OpenMP | Fully connected (GEMM) |
| `Bilinear` | ⚠️ Pass-through | - | Bilinear transformation |

**Paramètres:**
- `in_features`
- `out_features`
- `use_bias` (défaut: true)

**Exemple Lua:**
```lua
-- Note: in_features et out_features doivent être configurés
-- par l'architecture ou manuellement
Mimir.Model.push_layer("fc", "Linear", params_count)
```

---

### 📐 Shape Operations

| Type | Status | Optimisations | Notes |
|------|--------|---------------|-------|
| `Flatten` | ✅ Implémenté | - | Flatten dimensions |
| `Reshape` | ✅ Implémenté | - | Reshape tensor |
| `View` | ✅ Implémenté | - | Alias for Reshape |
| `Transpose` | ✅ Implémenté | OpenMP | Transpose 2D matrix |
| `Permute` | ⚠️ Pass-through | - | Permute dimensions |
| `Squeeze` | ⚠️ Pass-through | - | Remove dimensions of size 1 |
| `Unsqueeze` | ⚠️ Pass-through | - | Add dimension of size 1 |

**Paramètres:**
- `target_shape` (pour Reshape/View)
- `axis` (pour Transpose/Squeeze/Unsqueeze)
- `in_features`, `out_features` (pour Transpose - rows×cols)

**Exemple Lua:**
```lua
Mimir.Model.push_layer("flatten", "Flatten", 0)
Mimir.Model.push_layer("reshape", "Reshape", 0)
-- Transpose nécessite configuration dimensions
```

---

### 🧠 Attention Layers

| Type | Status | Optimisations | Notes |
|------|--------|---------------|-------|
| `SelfAttention` | ⚠️ Pass-through | - | Self-attention mechanism |
| `MultiHeadAttention` | ⚠️ Pass-through | - | Multi-head attention |
| `CrossAttention` | ⚠️ Pass-through | - | Cross-attention mechanism |

**Paramètres:**
- `num_heads` (défaut: 8)
- `head_dim` (défaut: 64)
- `causal` (défaut: false)
- `dropout_p` (défaut: 0.0)

**Exemple Lua:**
```lua
Mimir.Model.push_layer("attn", "SelfAttention", params_count)
```

---

### ➕ Element-wise Operations

| Type | Status | Optimisations | Notes |
|------|--------|---------------|-------|
| `Add` | ✅ Implémenté | AVX2 | Element-wise addition (residual support) |
| `Multiply` | ✅ Implémenté | AVX2 | Element-wise multiplication |
| `Subtract` | ⚠️ Pass-through | - | Element-wise subtraction |
| `Divide` | ⚠️ Pass-through | - | Element-wise division |

**Paramètres:**
- `use_branch` (pour combiner avec branches)

**Note:** Add et Multiply implémentés avec support résiduel.
Fonctionnent en combinant le layer actuel avec le layer précédent.

**Exemple Lua:**
```lua
-- Add est utilisé automatiquement dans les architectures avec residual
local config = {
    use_residual = true  -- Active les connexions Add
}
Mimir.Architectures.unet(config)
```

---

### 📦 Embedding & Lookup

| Type | Status | Optimisations | Notes |
|------|--------|---------------|-------|
| `Embedding` | ⚠️ Pass-through | OpenMP ready | Embedding lookup table |
| `PositionalEncoding` | ⚠️ Pass-through | - | Positional encoding |

**Paramètres:**
- `vocab_size`
- `embed_dim`
- `padding_idx` (défaut: -1)

**Note:** Implémentation disponible mais nécessite input int[]
au lieu de float[]. Adaptation API nécessaire.

---

### 🔀 Tensor Manipulation

| Type | Status | Optimisations | Notes |
|------|--------|---------------|-------|
| `Concat` | ⚠️ Pass-through | - | Concatenate tensors (needs branches) |
| `Split` | ✅ Implémenté | - | Split tensor along axis |
| `Chunk` | ⚠️ Pass-through | - | Chunk tensor into N parts |
| `Stack` | ⚠️ Pass-through | - | Stack tensors |

**Paramètres:**
- `concat_axis` (pour Concat)
- `num_splits` (pour Split/Chunk)

**Note:** Concat nécessite support branches multiples (pas encore implémenté).
Split implémenté et retourne vecteur de tensors.

---

### 📈 Upsampling & Interpolation

| Type | Status | Optimisations | Notes |
|------|--------|---------------|-------|
| `UpsampleNearest` | ✅ Implémenté | OpenMP | Nearest neighbor upsampling |
| `UpsampleBilinear` | ✅ Implémenté | OpenMP | Bilinear interpolation |
| `UpsampleBicubic` | ⚠️ Pass-through | - | Bicubic interpolation |

**Paramètres:**
- `scale_h`, `scale_w` (défaut: 2) - pour Nearest
- `out_h`, `out_w` (output size) - pour Bilinear
- `in_channels`, `out_h`, `out_w` (dimensions actuelles)

**Exemple Lua:**
```lua
-- Utilisé automatiquement dans ConvTranspose2d (UNet decoder)
Mimir.Architectures.unet(config)
```

---

### 🧮 Matrix Operations

| Type | Status | Optimisations | Notes |
|------|--------|---------------|-------|
| `MatMul` | ✅ Implémenté | AVX2, OpenMP | Matrix multiplication (GEMM) |
| `BatchMatMul` | ⚠️ Pass-through | - | Batched matrix multiplication |

**Paramètres:**
- `M`, `K`, `N` (dimensions: A=M×K, B=K×N)

**Note:** MatMul implémenté avec optimisations AVX2 (FMA).
Utilisé dans Linear layer pour GEMM.

**Exemple Lua:**
```lua
-- MatMul utilisé automatiquement dans Linear
Mimir.Model.push_layer("fc", "Linear", params_count)
```

---

### 🎲 Regularization

| Type | Status | Optimisations | Notes |
|------|--------|---------------|-------|
| `Dropout` | ✅ Implémenté | OpenMP | Dropout with scaling |
| `Dropout2d` | ✅ Implémenté | OpenMP | 2D dropout (spatial) |
| `AlphaDropout` | ⚠️ Pass-through | - | Alpha dropout (SELU) |

**Paramètres:**
- `dropout_p` (probability, défaut: 0.5)

**Exemple Lua:**
```lua
Mimir.Model.push_layer("dropout", "Dropout", 0)
```

---

### 🔧 Utility Layers

| Type | Status | Optimisations | Notes |
|------|--------|---------------|-------|
| `Identity` | ✅ Implémenté | - | Pass-through layer |
| `Copy` | ⚠️ Pass-through | - | Copy tensor |
| `Clone` | ⚠️ Pass-through | - | Clone tensor |

**Exemple Lua:**
```lua
Mimir.Model.push_layer("identity", "Identity", 0)
```

---

## 📝 Aliases Supportés

Le système accepte plusieurs variantes pour chaque type:

```cpp
// Convolution
"conv2d", "Conv2d", "conv", "Conv" → LayerType::Conv2d

// Linear
"linear", "Linear", "fc", "FC", "dense", "Dense" → LayerType::Linear

// Activation
"relu", "ReLU", "RELU" → LayerType::ReLU
"gelu", "GELU" → LayerType::GELU
"silu", "SiLU", "swish", "Swish" → LayerType::SiLU

// Normalization
"batchnorm2d", "bn2d", "BatchNorm2d" → LayerType::BatchNorm2d
"layernorm", "ln", "LayerNorm" → LayerType::LayerNorm
"groupnorm", "gn", "GroupNorm" → LayerType::GroupNorm

// Pooling
"maxpool2d", "maxpool", "MaxPool2d" → LayerType::MaxPool2d
"avgpool2d", "avgpool", "AvgPool2d" → LayerType::AvgPool2d
"globalavgpool", "gap", "GlobalAvgPool2d" → LayerType::GlobalAvgPool2d

// Shape
"flatten", "Flatten" → LayerType::Flatten
"reshape", "Reshape", "view", "View" → LayerType::Reshape

// Attention
"selfattention", "attention", "SelfAttention" → LayerType::SelfAttention
"multiheadattention", "mha", "MultiHeadAttention" → LayerType::MultiHeadAttention

// Embedding
"embedding", "embed", "Embedding" → LayerType::Embedding

// Element-wise
"add", "Add", "+" → LayerType::Add
"multiply", "Multiply", "mul", "*" → LayerType::Multiply

// Dropout
"dropout", "Dropout" → LayerType::Dropout
```

---

## 🎯 Statistiques

### Par Catégorie

| Catégorie | Total | Implémenté | Pass-through |
|-----------|-------|------------|--------------|
| Convolution | 5 | 2 (🔄) | 3 |
| Normalization | 6 | 4 (2🔄 + 2✅) | 2 |
| Activation | 11 | 7 ✅ | 4 |
| Pooling | 7 | 4 (1🔄 + 3✅) | 3 |
| Linear | 2 | 1 ✅ | 1 |
| Shape Operations | 7 | 3 ✅ | 4 |
| Attention | 3 | 0 | 3 ⚠️ |
| Element-wise | 4 | 0 | 4 ⚠️ |
| Embedding | 2 | 0 | 2 ⚠️ |
| Tensor Manipulation | 4 | 0 | 4 |
| Upsampling | 3 | 0 | 3 |
| Matrix Operations | 2 | 0 | 2 |
| Regularization | 3 | 2 ✅ | 1 |
| Utility | 3 | 1 ✅ | 2 |
| **TOTAL** | **67** | **24** | **43** |

### Détail Status

- ✅ **Implémenté:** 24 layers
- 🔄 **Préservé (ancien code):** 4 layers
- ⚠️ **Pass-through:** 39 layers
- **TOTAL fonctionnel:** 28 layers (42%)

---

## 🚀 Prochaines Priorités

### Phase 1 (Critical) 🔴
1. ~~**Add**~~ ✅ Element-wise addition (implémenté avec residual)
2. **Concat** - Tensor concatenation (nécessite support branches multiples)
3. **Embedding** - Embedding lookup (nécessite input integer)

### Phase 2 (Important) 🟡
4. **SelfAttention** - Self-attention mechanism
5. **MultiHeadAttention** - Multi-head attention
6. ~~**UpsampleNearest**~~ ✅ Upsampling nearest neighbor (implémenté)
7. ~~**MatMul**~~ ✅ Matrix multiplication (implémenté avec AVX2)

### Phase 3 (Nice to have) 🟢
8. ~~**Transpose**~~ ✅ 2D matrix transpose (implémenté)
9. ~~**Split**~~ ✅ Tensor splitting (implémenté)
10. **LeakyReLU**, **ELU**, **PReLU** - Additional activations

---

## 📚 Ressources

- **Guide d'ajout:** [HOWTO_ADD_LAYER.md](HOWTO_ADD_LAYER.md)
- **Implémentations:** [src/LayerOps.hpp](../src/LayerOps.hpp)
- **Types:** [src/LayerTypes.hpp](../src/LayerTypes.hpp)
- **Dispatch:** [src/Model.cpp](../src/Model.cpp) (ligne ~1850)

---

**Version:** v2.1.0  
**Last Update:** 2025-01  
**Total Layers Defined:** 67  
**Implemented:** 19 (28%)
