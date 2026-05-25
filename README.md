# Mímir Framework

![logo](./logo.png)

Version framework : **2.4.0**  
Révision documentation : **2026-05-25**

Mímir est un framework de deep learning en **C++17** orienté **CPU-first** (SIMD/OpenMP) avec une API **Lua** pour prototyper rapidement, un registre d’architectures (Vision/NLP/Diffusion) et un système de **sérialisation** (SafeTensors + formats debug). Une accélération **Vulkan Compute** est disponible pour certains chemins.

## Objectifs

- Construire des modèles composables (layers → architectures → scripts).
- Entraîner et exécuter localement avec garde-fous mémoire.
- Sauvegarder/charger des checkpoints et échanger des poids (interop SafeTensors).

## Démarrage rapide

### Compiler

```bash
cmake -S . -B build
cmake --build build -j
```

## Accélération GPU (optionnelle)

Le framework reste **CPU-first**. Les backends GPU sont **optionnels**, **désactivés par défaut** au runtime, et ne couvrent actuellement qu’un **fast-path** ciblé.

### Ce qui est accéléré

- `Linear` (inference only) : tentative GPU → sinon fallback CPU.
- Backends disponibles (si compilés) : **CUDA (cuBLAS)**, **ROCm (rocBLAS/HIP)**, **Vulkan Compute**, **OpenCL**.

Priorité actuelle : CUDA → ROCm → Vulkan → OpenCL → CPU.

### Build (CMake)

- `-DENABLE_VULKAN=ON|OFF`
- `-DENABLE_OPENCL=ON|OFF`
- `-DENABLE_CUDA=ON|OFF` (requiert `CUDAToolkit`)
- `-DENABLE_ROCM=ON|OFF` (requiert HIP + rocBLAS)

Si CUDA/ROCm sont installés dans un chemin non standard, tu peux aider la détection :

- `-DMIMIR_CUDA_ROOT=/chemin/vers/cuda` (ex: `/usr/local/cuda`)
- `-DMIMIR_ROCM_ROOT=/chemin/vers/rocm` (ex: `/opt/rocm`)

Alternativement (sans toucher à la ligne CMake), tu peux définir :

- `CUDA_HOME` / `CUDA_PATH`
- `ROCM_PATH` / `ROCM_HOME`

### Runtime (variables d’environnement)

- Activer par backend (opt-in) :
  - `MIMIR_CUDA_LINEAR=1`
  - `MIMIR_ROCM_LINEAR=1`
  - `MIMIR_VULKAN_LINEAR=1`
  - `MIMIR_OPENCL_LINEAR=1`
- Seuil minimal (évite d’accélérer les petits GEMM) :
  - `MIMIR_CUDA_LINEAR_MIN_OPS` (défaut: $2^{20}$)
  - `MIMIR_ROCM_LINEAR_MIN_OPS` (défaut: $2^{20}$)
  - `MIMIR_VULKAN_LINEAR_MIN_OPS` (défaut: $2^{20}$)
  - `MIMIR_OPENCL_LINEAR_MIN_OPS` (défaut: $2^{20}$)
- Debug : `MIMIR_ACCEL_VERBOSE=1`
- Kill-switch build-time OK mais runtime OFF :
  - `MIMIR_DISABLE_CUDA=1`
  - `MIMIR_DISABLE_ROCM=1`

### Lancer un script Lua

```bash
./bin/mimir --lua scripts/examples/vae_text_sample.lua -- --help
```

## Planners & fusion (CPU)

Le runtime inclut un mini “framework planner” (scheduling statique, analyse de lifetimes, plan mémoire best-effort) et une première fusion d’opérations/kernels sur CPU.

- `MIMIR_ENABLE_PLANNER=1` : active la construction du plan statique (par défaut: on).
- `MIMIR_ENABLE_FUSION=1` : active les fusions conservatrices (par défaut: on).
- `MIMIR_PLANNER_DUMP=1` : log une fois des infos planner (lifetimes + scratch Conv2d).

Fusion actuellement branchée :

- `Conv2d + ReLU` fusionné (inference only) dans le scatter du fast-path im2col+GEMM.

## DTypes

La gestion des dtypes est centralisée (parsing + taille en octets) via `src/DType.hpp`.

- `RuntimeAllocator::allocate_tensor(...)` reste volontairement limité à `float32`.
- Pour d’autres dtypes, utiliser `RuntimeAllocator::allocate_raw_tensor(shape, dtype, name)`.

### `dtype` (contrat framework)

- Le runtime reste **float32-first** (poids et activations en RAM).
- Le champ `dtype` sert principalement à :
  - fixer le **dtype “par défaut”** du modèle (`Model.default_dtype`),
  - choisir le **dtype de stockage sur disque** pour les tenseurs float lors des `save()` (SafeTensors/RawFolder : f32 → f16/bf16/f64 selon le choix).

Propagation garantie :

- `cfg.dtype` → appliqué automatiquement lors de `Model.create(...)` (via le registre d’architectures).
- `Serialization.load(...)` → si le checkpoint embarque `model_config.dtype`, il est réappliqué au runtime (donc les `save()` suivants restent cohérents).

Régler le dtype :

- Recommandé : mettre `cfg.dtype = "float16"|"bfloat16"|"float32"|"float64"` avant `Model.create(...)`.
- Ou : `Mimir.Model.dtype("float16")` pour override explicite (et synchronisation de `model_config.dtype`).

Note : un `dtype` inconnu est considéré comme une erreur (pour éviter les runs “silencieux” en `float32`).

## Documentation

La documentation est dans le dossier `docs/`.

- Point d’entrée : `docs/00-INDEX.md`
- Installation & build : `docs/01-Getting-Started/`
- Guide utilisateur : `docs/02-User-Guide/`
- Référence API : `docs/03-API-Reference/`
- Internals : `docs/04-Architecture-Internals/`
- Performance & tuning : `docs/05-Advanced/`
- Contribution : `docs/06-Contributing/`

## Notes (limites)

- **CPU-first** : adapté au prototypage et à des modèles modestes ; pour des LLM très gros, il faut des optimisations spécifiques (batching, KV-cache, quantization).
- Tous les layers ne sont pas au même niveau de maturité : la doc distingue **stable** vs **expérimental**.

## Licence

Voir `LICENSE`.
