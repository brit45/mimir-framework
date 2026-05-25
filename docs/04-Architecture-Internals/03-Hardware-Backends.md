# Backends hardware : CPU / CUDA / ROCm / Vulkan / OpenCL

Mímir est **CPU-first** : le CPU reste le chemin de référence (justesse + compatibilité maximale). Des backends d'accélération optionnels activent des fast-paths pour les opérations lourdes.

Sources de vérité :

- `src/runtimes/AbstractRuntime.hpp` — interface commune + `RuntimeConfig`
- `src/runtimes/cpu/CpuRuntime.cpp` — implémentation CPU
- `src/runtimes/cuda/CudaRuntime.cpp` — fast-paths cuBLAS (CUDA)
- `src/runtimes/rocm/RocmRuntime.cpp` — fast-paths rocBLAS (ROCm/HIP)
- `src/VulkanCompute.hpp` — compute Vulkan (legacy, Linear uniquement)
- `src/OpenCLCompute.hpp` — compute OpenCL (legacy)
- `CMakeLists.txt` — flags : `ENABLE_CUDA`, `ENABLE_ROCM`, `ENABLE_VULKAN`

---

## 0) Architecture générale des runtimes

Chaque runtime implémente `AbstractRuntime` :

```cpp
class AbstractRuntime {
    virtual bool initialize(const RuntimeConfig& cfg) = 0;
    virtual bool forwardLayer(
        const std::vector<const std::vector<float>*>& inputs,
        std::vector<std::vector<float>>& outputs,
        const Layer& layer,
        bool training) = 0;
};
```

Le dispatch dans `Model.cpp` cherche le backend dans l'ordre :
1. CUDA (si `ENABLE_CUDA` et initialisé)
2. ROCm (si `ENABLE_ROCM` et initialisé)
3. CPU (toujours disponible, fallback universel)

Tout fast-path GPU échoue silencieusement vers CPU via un bloc `do { ... } while(false)` avec `break`.

---

## 0.1) `RuntimeConfig` — configuration commune

```cpp
struct RuntimeConfig {
    bool disabled           = false;
    bool verbose            = false;

    bool linear_enabled     = false;
    int  linear_min_ops     = 1 << 20;   // ~1M MACs

    bool conv_enabled       = false;
    int  conv_min_ops       = 1 << 18;   // ~256K MACs

    bool norm_enabled         = false;
    int  norm_min_elements    = 1 << 12; // 4096 éléments

    bool attention_enabled  = false;
    int  attention_min_ops  = 1 << 18;  // ~256K MACs

    int device_index = 0;

    static RuntimeConfig fromEnv(const char* backend_upper);
};
```

### Variables d'environnement

Pour CUDA (remplacer `CUDA` par `ROCM` pour le backend ROCm) :

| Variable | Effet |
|---|---|
| `MIMIR_CUDA=0` | Désactive le runtime CUDA |
| `MIMIR_CUDA_VERBOSE=1` | Active le mode verbeux |
| `MIMIR_CUDA_DEVICE=N` | Choisit le device (défaut 0) |
| `MIMIR_CUDA_LINEAR=1` | Active fast-path Linear |
| `MIMIR_CUDA_LINEAR_MIN_OPS=N` | Seuil min MACs pour Linear |
| `MIMIR_CUDA_CONV=1` | Active fast-path Conv2d |
| `MIMIR_CUDA_CONV_MIN_OPS=N` | Seuil min MACs pour Conv |
| `MIMIR_CUDA_NORM=1` | Active fast-path LayerNorm/RMSNorm |
| `MIMIR_CUDA_NORM_MIN_ELEMS=N` | Seuil min éléments pour Norm |
| `MIMIR_CUDA_ATTENTION=1` | Active fast-path Attention |
| `MIMIR_CUDA_ATTENTION_MIN_OPS=N` | Seuil min MACs pour Attention |

Activation complète CUDA :

```bash
export MIMIR_CUDA=1
export MIMIR_CUDA_LINEAR=1
export MIMIR_CUDA_CONV=1
export MIMIR_CUDA_NORM=1
export MIMIR_CUDA_ATTENTION=1
./bin/mimir --lua scripts/training/ponyxl_ddpm_train.lua
```

---

## 1) CUDA Runtime (cuBLAS)

Build flag : `-DENABLE_CUDA=ON`.
Dépendances : `CUDA::cudart`, `CUDA::cublas`.

### Fast-paths implémentés

Tous les fast-paths suivent le même patron :
1. Vérifier les seuils d'activation (`config_.xxx_enabled`, seuil d'ops)
2. Allouer des buffers device (`Impl::DeviceBuf`)
3. Appeler les primitives cuBLAS
4. `cudaDeviceSynchronize()` + copie host
5. En cas d'échec → `break` → fallback CPU

#### 1.1 Linear (SGEMM)

`output[M,N] = input[M,K] × W[K,N]` via `cublasSgemm`. Biais via `cublasSaxpy`.

Seuil par défaut : `1 << 20` MACs. Réglable via `MIMIR_CUDA_LINEAR_MIN_OPS`.

#### 1.2 Conv2d (im2col + SGEMM)

1. **im2col sur host** — `input[C,H,W]` → `col[C·kH·kW, H_out·W_out]`
2. **SGEMM sur device** — `output = W × col`
3. **Biais via `cublasSger`** — outer product

Supporte : `stride`, `padding`, `dilation`. Désactivé en mode `training=true`.

#### 1.3 LayerNorm / RMSNorm (hybride CPU+GPU)

1. **Normalisation sur host** — `x_hat = (x - mean) / sqrt(var + eps)` ou RMS
2. **Affine sur GPU** — `y = gamma ⊙ x_hat + beta` via `cublasSdgmm` + `cublasSaxpy`

GroupNorm/BatchNorm restent sur CPU (layout incompatible).
Seuil par défaut : `1 << 12` éléments.

#### 1.4 SelfAttention / MultiHeadAttention

1. QKV projection sur GPU (SGEMM)
2. Split Q/K/V sur host
3. Par tête : scores (SGEMM OP_T), masque causal optionnel, softmax, contexte (SGEMM)
4. Projection de sortie sur GPU

#### 1.5 CrossAttention

Même principe avec `Q` et `KV` depuis deux sources différentes (`qlen` ≠ `kvlen` possible).

### DeviceBuf helper

```cpp
struct Impl::DeviceBuf {
    bool alloc(size_t bytes);
    bool copyFromHost(const void* src, size_t bytes);
    bool copyToHost(void* dst, size_t bytes);
    ~DeviceBuf(); // cudaFree automatique
};
```

---

## 2) ROCm Runtime (rocBLAS)

Build flag : `-DENABLE_ROCM=ON`.
Dépendances : `hip::host`, `roc::rocblas`.

Interface **identique** au runtime CUDA. API rocBLAS calquée :
- `hipMalloc/hipFree/hipMemcpy/hipDeviceSynchronize` → `cudaMalloc/…`
- `rocblas_sgemm`, `rocblas_sger`, `rocblas_sdgmm`, `rocblas_saxpy`
- Bloc conditionnel : `#ifdef ENABLE_ROCM` / `#endif`

Variables d'environnement : `MIMIR_ROCM_*` (mêmes suffixes que CUDA).

---

## 3) CPU Runtime

Toujours actif. Fallback universel via `RuntimeLayerDispatch::cpu_forward_layer` qui délègue aux fonctions `LayerOps.hpp` et `LayerOpsExt.hpp`.

Détection CPU SIMD :
```cpp
bool Model::hasAVX2();  // AVX2
bool Model::hasFMA();   // FMA
bool Model::hasF16C();  // float16 conversion native
bool Model::hasBMI2();  // bit manipulation
```

OpenMP activé au build pour les boucles parallèles dans LayerOps.

---

## 4) Tableau de synthèse

| Layer | CPU | CUDA | ROCm | Vulkan |
|---|---|---|---|---|
| Linear | ✓ ref | ✓ cuBLAS | ✓ rocBLAS | ✓ shader |
| Conv2d | ✓ ref | ✓ im2col+SGEMM | ✓ im2col+SGEMM | ✗ |
| LayerNorm | ✓ ref | ✓ hybride | ✓ hybride | ✗ |
| RMSNorm | ✓ ref | ✓ hybride | ✓ hybride | ✗ |
| GroupNorm | ✓ ref | ✗ fallback | ✗ fallback | ✗ |
| BatchNorm2d | ✓ ref | ✗ fallback | ✗ fallback | ✗ |
| SelfAttention | ✓ ref | ✓ multi-SGEMM | ✓ multi-SGEMM | ✗ |
| MultiHeadAttention | ✓ ref | ✓ multi-SGEMM | ✓ multi-SGEMM | ✗ |
| CrossAttention | ✓ ref | ✓ multi-SGEMM | ✓ multi-SGEMM | ✗ |
| Autres | ✓ ref | ✗ fallback | ✗ fallback | ✗ |

**Règle** : tout fast-path GPU absent ou dont les conditions ne sont pas remplies retombe silencieusement sur le CPU.

---

## 5) Vulkan Compute (legacy)

### Build (Vulkan)

- flag CMake : `ENABLE_VULKAN`
- dépendance : `find_package(Vulkan)`

### Compute engine (Vulkan)

Le backend Vulkan est implémenté dans `src/VulkanCompute.hpp` via `VulkanCompute::ComputeEngine`.

Init (simplifié) :

- crée `VkInstance`
- choisit le premier device avec `VK_QUEUE_COMPUTE_BIT`
- crée `VkDevice` + récupère la queue compute
- crée un `VkCommandPool`

### Kernel ciblé (Vulkan) : `linearForward`

- `ensureLinearKernel()` prépare pipeline/descriptor sets
- `linearForward(input, weights, bias, output, batch, in_f, out_f)`
  - alloue des buffers temporaires
  - upload input/weights/bias
  - dispatch compute
  - readback output

Note : allocation par appel (pas de pooling).

### Shaders SPIR-V

Le shader attendu : `linear_forward.comp.spv`.

Compilation (CMake, best-effort) :
- source : `shaders/linear_forward.comp`
- output : `${CMAKE_BINARY_DIR}/shaders/linear_forward.comp.spv`

Lookup runtime :
- variable d'environnement : `MIMIR_VULKAN_LINEAR_SPV` (chemin direct)
- sinon candidats relatifs au cwd

Si le SPIR-V n'est pas trouvé, le runtime log et retombe sur CPU.

---

## 6) OpenCL Compute (legacy)

### Build (OpenCL)

- flag CMake : `ENABLE_OPENCL`
- dépendance : librairie `OpenCL`

### Compute engine (OpenCL)

`src/OpenCLCompute.hpp` via `OpenCLCompute::ComputeEngine`.

Init :
- détecte les plateformes
- choisit un device : GPU d'abord, sinon CPU
- crée `cl_context` + `cl_command_queue`
- compile le programme OpenCL embarqué (source string dans le header)

### Kernel ciblé (OpenCL) : `linearForward`

- `linearForward(input, weights, bias_or_null, output, batch, in_f, out_f)`
- alloue des `cl_mem` par appel
- `clEnqueueNDRangeKernel` sur une grille 2D `(batch, out_f)`
- readback output

---

## 7) Notes et pièges

- La justesse doit être validée sur CPU d'abord : GPU est un accélérateur, pas le chemin de référence.
- Les fast-paths GPU sont **désactivés par défaut** : activer via les variables d'environnement.
- La perf GPU peut être limitée par :
  - allocations temporaires par appel (DeviceBuf non poolé)
  - transferts host↔device
  - seuils d'opérations non atteints (réduire `*_MIN_OPS` si nécessaire)
  - disponibilité/compilation des shaders (Vulkan)
- Le mode `training=true` désactive certains fast-paths (Conv2d) car le backward GPU n'est pas implémenté.

## 8) API Lua

- `Mimir.Model.hardware_caps()` — capacités CPU détectées
- `Mimir.Model.set_hardware(true/false)` — activer/désactiver les chemins hardware

Détails dans `src/LuaScripting.cpp`.
