# Internals : GPU Runtimes — CUDA & ROCm (C++)

## Pour qui

Développeur avancé qui modifie le moteur C/C++.

## Objectif

Comprendre le fonctionnement interne exact des composants runtime.

## Avant de commencer

Connaître les bases C++ et la structure du dépôt.

## Résultat attendu

Tu peux modifier le code interne en limitant les régressions.


Cette page documente l'implémentation interne des backends GPU de Mímir : architecture, fast-paths, conventions de code.

Sources de vérité :

- `src/runtimes/AbstractRuntime.hpp` — interface + `RuntimeConfig`
- `src/runtimes/cuda/CudaRuntime.hpp` / `.cpp` — CUDA (cuBLAS)
- `src/runtimes/rocm/RocmRuntime.hpp` / `.cpp` — ROCm (rocBLAS)
- `src/runtimes/cpu/CpuRuntime.hpp` / `.cpp` — CPU référence

Guide utilisateur associé : [docs/05-Advanced/05-GPU-Acceleration.md](../05-Advanced/05-GPU-Acceleration.md).

---

## 1) Interface commune (`AbstractRuntime`)

```cpp
class AbstractRuntime {
public:
    virtual const char* name() const = 0;
    virtual bool initialize(const RuntimeConfig& cfg) = 0;
    virtual void shutdown() = 0;
    virtual bool isInitialized() const = 0;

    // Fast-path pour Linear (SGEMM)
    virtual bool linearForward(const float* input, const float* weights,
                               const float* bias_or_null, float* output,
                               int batch, int in_f, int out_f) = 0;

    // Dispatch générique par type de layer
    virtual bool forwardLayer(
        const std::vector<const std::vector<float>*>& inputs,
        std::vector<std::vector<float>>& outputs,
        const Layer& layer,
        bool training) = 0;
};
```

### Patron de dispatch (Model.cpp)

```cpp
// Dans Model::forward pour chaque layer :
if (cuda_runtime_ && cuda_runtime_->isInitialized()) {
    if (cuda_runtime_->forwardLayer(inputs, outputs, layer, training))
        continue; // fast-path OK, on passe au layer suivant
}
if (rocm_runtime_ && rocm_runtime_->isInitialized()) {
    if (rocm_runtime_->forwardLayer(inputs, outputs, layer, training))
        continue;
}
// Fallback CPU
cpu_forward_layer(inputs, outputs, layer, training);
```

`forwardLayer` retourne `true` si le fast-path a réussi, `false` pour fallback.

---

## 2) `RuntimeConfig` — paramètres et parsing

### Structure

```cpp
struct RuntimeConfig {
    std::string backend;   // "CUDA" ou "ROCM"
    bool disabled         = false;
    bool verbose          = false;

    bool linear_enabled   = false;
    int  linear_min_ops   = 1 << 20;

    bool conv_enabled     = false;
    int  conv_min_ops     = 1 << 18;

    bool norm_enabled         = false;
    int  norm_min_elements    = 1 << 12;

    bool attention_enabled  = false;
    int  attention_min_ops  = 1 << 18;

    int device_index = 0;
};
```

### Parsing depuis l'environnement

```cpp
RuntimeConfig RuntimeConfig::fromEnv(const char* backend_upper) {
    RuntimeConfig cfg;
    cfg.backend = backend_upper;
    // Préfixe : "MIMIR_CUDA_" ou "MIMIR_ROCM_"
    std::string pfx = std::string("MIMIR_") + backend_upper + "_";

    auto get_bool = [&](const char* key, bool def) -> bool { ... };
    auto get_int  = [&](const char* key, int  def) -> int  { ... };

    cfg.disabled    = !get_bool("", true);     // MIMIR_CUDA=0 → disabled
    cfg.verbose     = get_bool("VERBOSE", false);
    cfg.device_index = get_int("DEVICE", 0);

    cfg.linear_enabled    = get_bool("LINEAR",    false);
    cfg.linear_min_ops    = get_int ("LINEAR_MIN_OPS", 1 << 20);
    cfg.conv_enabled      = get_bool("CONV",      false);
    cfg.conv_min_ops      = get_int ("CONV_MIN_OPS",  1 << 18);
    cfg.norm_enabled      = get_bool("NORM",      false);
    cfg.norm_min_elements = get_int ("NORM_MIN_ELEMS", 1 << 12);
    cfg.attention_enabled = get_bool("ATTENTION", false);
    cfg.attention_min_ops = get_int ("ATTENTION_MIN_OPS", 1 << 18);
    return cfg;
}
```

---

## 3) CudaRuntime — implémentation

### Structure interne

```cpp
class CudaRuntime final : public AbstractRuntime {
    struct Impl {
        cublasHandle_t handle = nullptr;
        RuntimeConfig  config;

        // Buffers temporaires réutilisables (arène simple)
        struct DeviceBuf {
            void*  ptr   = nullptr;
            size_t bytes = 0;
            bool alloc(size_t n);
            bool copyFromHost(const void* src, size_t n);
            bool copyToHost(void* dst, size_t n) const;
            ~DeviceBuf();
        };

        DeviceBuf d_A, d_B, d_C, d_bias, d_col, d_scratch;
    };
    Impl* impl_ = nullptr;
};
```

### Initialisation

```cpp
bool CudaRuntime::initialize(const RuntimeConfig& cfg) {
    impl_ = new Impl{};
    impl_->config = cfg;
    if (cfg.disabled) return false;
    cudaError_t err = cudaSetDevice(cfg.device_index);
    if (err != cudaSuccess) { /* log, return false */ }
    cublasStatus_t st = cublasCreate(&impl_->handle);
    return st == CUBLAS_STATUS_SUCCESS;
}
```

---

## 4) Fast-paths détaillés

### 4.1 Pattern commun

Chaque fast-path dans `CudaRuntime::forwardLayer` suit ce squelette :

```cpp
case LayerType::Linear:
{
    if (!impl_->config.linear_enabled) break;
    int M = batch, K = in_f, N = out_f;
    long long ops = (long long)M * K * N * 2;
    if (ops < impl_->config.linear_min_ops) break;

    // 1. Allouer device
    if (!impl_->d_A.alloc(M*K*sizeof(float))) break;
    if (!impl_->d_B.alloc(K*N*sizeof(float))) break;
    if (!impl_->d_C.alloc(M*N*sizeof(float))) break;

    // 2. Upload
    if (!impl_->d_A.copyFromHost(input.data(), M*K*sizeof(float))) break;
    if (!impl_->d_B.copyFromHost(weights, K*N*sizeof(float))) break;

    // 3. SGEMM
    float alpha=1.f, beta=0.f;
    auto st = cublasSgemm(impl_->handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        (float*)impl_->d_B.ptr, N,
        (float*)impl_->d_A.ptr, K,
        &beta,
        (float*)impl_->d_C.ptr, N);
    if (st != CUBLAS_STATUS_SUCCESS) break;

    // 4. Biais (optionnel)
    if (bias && out_f > 0) { /* cublasSaxpy par ligne */ }

    // 5. Sync + download
    if (cudaDeviceSynchronize() != cudaSuccess) break;
    if (!impl_->d_C.copyToHost(output.data(), M*N*sizeof(float))) break;

    return true; // ← succès
}
while(false); // break → fallback CPU
return false;
```

### 4.2 Conv2d : im2col détail

```
input[B, C_in, H, W]  →  col[B, C_in·kH·kW, H_out·W_out]
                          (im2col, CPU, avec stride/padding/dilation)

output[B, C_out, H_out·W_out]
  = W[C_out, C_in·kH·kW]  ×  col[C_in·kH·kW, H_out·W_out]
    (cublasSgemm, device)
```

Im2col implémenté avec des boucles C++ standard. Pas de CUDA pour im2col (coût faible, évite les kernels supplémentaires).

### 4.3 LayerNorm / RMSNorm : hybride

**Pourquoi hybride ?** La normalisation implique une réduction (mean/variance) qui est efficace sur CPU avec peu de code. Seule la partie affine (multiplication + addition par vecteur) est déportée sur GPU car c'est un SGEMM/AXPY.

```
x_hat = (x - mean) / sqrt(var + eps)    ← sur host (O(N))
y     = gamma ⊙ x_hat + beta            ← sur device (cublasSdgmm + cublasSaxpy)
```

`cublasSdgmm(LEFT, N, M, x_hat, N, gamma, 1, y, N)` = mise à l'échelle par `gamma`.

### 4.4 Attention : boucle par tête

```
Pour chaque tête h (0..num_heads-1) :
  Q_h, K_h, V_h = split(QKV_projected)      ← sur host

  scores[h] = Q_h × K_h^T / sqrt(d_head)    ← SGEMM OP_T sur device
  if causal: masque supérieur                 ← sur host
  scores[h] = softmax(scores[h])              ← sur host

  ctx[h] = scores[h] × V_h                   ← SGEMM sur device

concat(ctx_0..ctx_h)                          ← sur host
output = concat × W_out                       ← SGEMM sur device
```

**Note** : le calcul de softmax reste sur host pour simplifier (pas de kernel softmax GPU custom). Cela implique un aller-retour host↔device pour les scores, ce qui est un overhead mesurable sur de longues séquences.

---

## 5) ROCm Runtime — différences avec CUDA

L'implémentation ROCm est un miroir de CUDA, avec substitution d'API :

| CUDA | ROCm |
|---|---|
| `cudaMalloc` | `hipMalloc` |
| `cudaFree` | `hipFree` |
| `cudaMemcpy` | `hipMemcpy` |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` |
| `cublasCreate` | `rocblas_create_handle` |
| `cublasSgemm` | `rocblas_sgemm` |
| `cublasSaxpy` | `rocblas_saxpy` |
| `cublasSger` | `rocblas_sger` |
| `cublasSdgmm` | `rocblas_sdgmm` |
| `CUBLAS_STATUS_SUCCESS` | `rocblas_status_success` |
| `CUBLAS_OP_N` | `rocblas_operation_none` |
| `CUBLAS_OP_T` | `rocblas_operation_transpose` |
| `CUBLAS_SIDE_LEFT` | `rocblas_side_left` |

Les fast-paths activés, seuils, et logique de fallback sont strictement identiques.

---

## 6) Invariants et gotchas

**Aucun état persistent entre les couches**

Les `DeviceBuf` sont alloués/libérés à chaque appel de fast-path. Il n'y a pas de pool de buffers GPU. C'est délibéré : simplicité d'implémentation. L'overhead d'allocation est limité car `cudaMalloc` est relativement rapide pour des buffers de même taille (le driver CUDA met en cache).

**`training=true` désactive Conv2d**

Le backward de Conv2d n'est pas implémenté côté GPU. Si le fast-path Conv2d était utilisé en training, le gradient serait incorrect. Le check `if (training) break;` est donc critique.

**Pas de mixed precision**

Tous les fast-paths travaillent en `float32`. Les types `F16`/`BF16` du modèle ne sont pas utilisés ici (la conversion est faite ailleurs, si applicable).

**Masque causal sur host**

Le masque causal de l'attention (upper triangle → -∞) est appliqué sur le CPU après download des scores. Pour de très longues séquences (>1K tokens), cela représente un overhead notable. À optimiser si besoin.

**Ordre des builds**

CUDA et ROCm sont mutuellement exclusifs (liés par `#ifdef`). On ne peut pas compiler les deux ensemble dans le même binaire.

---

## 7) Ajouter un nouveau fast-path GPU

1. Ajouter les flags dans `RuntimeConfig` (`src/runtimes/AbstractRuntime.hpp`)
2. Ajouter le parsing `fromEnv` correspondant
3. Implémenter le `case` dans `CudaRuntime::forwardLayer`
4. Implémenter le `case` miroir dans `RocmRuntime::forwardLayer`
5. Mettre à jour le tableau de synthèse dans `docs/04-Architecture-Internals/03-Hardware-Backends.md`
