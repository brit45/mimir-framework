# Performance (CPU)

Cette page liste les leviers **réels** (implémentés) pour améliorer les perfs, et où les retrouver côté C/C++.

Point d’entrée conseillé : [04-Source-Code-Map.md](04-Source-Code-Map.md).

## 1) Où sont les hotspots ?

Dans ce repo, les goulots typiques côté CPU sont :

- `Linear` / `MatMul` (projection, MLP, embeddings) : `src/LayerOps.hpp`, `src/SIMD_Ops.hpp`, et le dispatch dans `src/Model.cpp`.
- Attention (QKV + softmax + matmul) : `src/LayerOps.hpp`.
- Conv / ops “image” (si workload diffusion/vae) : `src/Layers.hpp`, `src/LayerOpsExt.hpp`.
- Allocations temporaires / copies : `src/RuntimeAllocator.hpp`, `src/tensors.cpp`, `src/DynamicTensorAllocator.*`.

## 2) OpenMP (threads)

Le code utilise OpenMP (si compilé avec support) pour paralléliser certaines boucles (matmul/conv/ops) avec des seuils de travail.

- Au démarrage, le binaire affiche le nombre de threads disponibles et appelle `omp_set_num_threads(omp_get_max_threads())` (voir `src/main.cpp`).
- Les kernels ont souvent des directives du type `#pragma omp parallel for ... if(work >= 262144)` (par ex. dans `src/LayerOps.hpp`, `src/SIMD_Ops.hpp`, `src/LayerOpsExt.hpp`).

Conseils pratiques :

- Si vous voulez *forcer* un nombre de threads, utilisez les mécanismes OpenMP habituels (ex: `OMP_NUM_THREADS`) avant de lancer le binaire.
- Faites varier `seq_len`, `d_model`, `num_heads` : l’activation des chemins parallèles dépend de la taille du travail.

## 3) SIMD (AVX2/FMA/F16C/BMI)

Le runtime expose des kernels AVX2 sur certaines ops (et un fallback sinon) :

- Détection des capacités + affichage au démarrage : `src/main.cpp` + helpers `Model::hasAVX2()` etc.
- Kernels SIMD : `src/SIMD_Ops.hpp`, et certaines routines dans `src/Layers.hpp` / `src/HardwareOpt.hpp`.

Conseils pratiques :

- Favoriser des dimensions multiples de 8 (AVX2 traite 8 floats) quand c’est pertinent (ex: `d_model`, `mlp_hidden`).
- Vérifier dans les logs de démarrage que l’AVX2 est bien détecté (sinon, vous êtes sur le fallback scalar/SIMD réduit).

## 4) Allocations et “strict mode”

Ce qui compte ici est moins “malloc vs new” et plus “qui alloue, quand, et sous quel garde-fou” :

- Les tenseurs “dynamiques” (utilisés dans des chemins runtime) passent par `DynamicTensorAllocator` et sont comptabilisés par `MemoryGuard` (voir `src/tensors.cpp`, `src/DynamicTensorAllocator.*`, `src/MemoryGuard.hpp`).
- Les scratchpads/temporaires RAII sont gérés par `RuntimeAllocator` (voir `src/RuntimeAllocator.hpp`) — utile pour limiter les pics de mémoire et stabiliser les perfs.

Conseils pratiques :

- Éviter de créer/détruire des gros buffers dans des boucles chaudes si un scratchpad/pool existe déjà.
- Sur OOM, préférez d’abord réduire batch/seq/dims avant d’essayer d’“optimiser le CPU”.

## 5) Offload Vulkan/OpenCL (Linear, inférence)

Le code peut dispatcher certaines couches `Linear` vers Vulkan ou OpenCL **uniquement en inférence** et **si explicitement activé**.

Pré-requis : le binaire doit avoir été compilé avec le support backend correspondant (gating via `ENABLE_VULKAN` / `ENABLE_OPENCL` dans le code).

Variables d’environnement (voir `src/Model.cpp`) :

- `MIMIR_VULKAN_LINEAR=1` active Vulkan pour `Linear`.
- `MIMIR_VULKAN_LINEAR_MIN_OPS` (défaut `1<<20`) fixe le seuil minimal d’opérations.
- `MIMIR_VULKAN_LINEAR_SPV=/chemin/vers/linear_forward.comp.spv` force le chemin du shader SPIR-V (sinon, le runtime cherche dans quelques emplacements standards, voir `src/VulkanCompute.hpp`).
- `MIMIR_OPENCL_LINEAR=1` active OpenCL pour `Linear`.
- `MIMIR_OPENCL_LINEAR_MIN_OPS` (défaut `1<<20`) fixe le seuil minimal d’opérations.
- `MIMIR_ACCEL_VERBOSE=1` logge quand un offload est réellement utilisé.
- `MIMIR_DISABLE_VULKAN=1` / `MIMIR_DISABLE_OPENCL=1` désactive le backend correspondant.

Notes :

- Si le seuil est trop haut, vous n’observerez aucun gain (le CPU restera le chemin pris).
- Les gains seront surtout visibles sur gros `out_features` / gros batch (ou `seq_len` si batch=seq).

## 6) Méthode de mesure (reproductible)

Pour éviter les faux gains :

- Mesurez séparément “forward seul” vs “train step” (backward + optimizer).
- Isolez le coût tokenization/IO (Lua/dataset) du coût compute (layers).
- Activez `MIMIR_ACCEL_VERBOSE` quand vous testez Vulkan/OpenCL, sinon vous ne savez pas si c’est réellement pris.
