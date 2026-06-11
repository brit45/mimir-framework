# Mímir Framework

![logo](./logo.png)

Version framework : **3.0.1**  
Révision documentation : **2026-06-05**

Mímir est un framework de deep learning en **C++17** orienté **CPU-first** (SIMD/OpenMP) avec une API **Lua** pour prototyper rapidement, un registre d’architectures (Vision/NLP/Diffusion) et un système de **sérialisation** (SafeTensors + formats debug). Une accélération **Vulkan Compute** est disponible pour certains chemins.

En pratique, le workflow recommandé ressemble à ceci :

1. créer une config canonique via le registre,
2. créer puis construire le modèle,
3. allouer et initialiser les poids, ou charger un checkpoint,
4. exécuter un `forward()` ou lancer une boucle d’entraînement.

## Objectifs

- Construire des modèles composables (layers → architectures → scripts).
- Entraîner et exécuter localement avec garde-fous mémoire.
- Sauvegarder/charger des checkpoints et échanger des poids (interop SafeTensors).

## Démarrage rapide

Si ton objectif est simplement de vérifier que le dépôt est exploitable, suis cet ordre : build, aide CLI, template minimal, smoke test de sérialisation.

### Compiler

```bash
cmake -S . -B build
cmake --build build -j
```

### Vérifier que le binaire démarre

```bash
./bin/mimir --help
./bin/mimir --lua scripts/tests/test_serialization_smoke.lua
```

Le premier appel vérifie que le binaire et les dépendances minimales sont présents. Le second valide un parcours court de sérialisation, utile pour confirmer que l’environnement est exploitable avant d’attaquer un entraînement plus lourd.

### Quelques commandes utiles dès le départ

```bash
# 1) Créer un modèle minimal via le registre
./bin/mimir --lua scripts/templates/template_new_model.lua

# 2) Vérifier save/load
./bin/mimir --lua scripts/tests/test_serialization_smoke.lua

# 3) Lister les architectures + dtypes disponibles
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -a

# 4) Inspecter un vrai script du dépôt
./bin/mimir --lua scripts/examples/classify_vgg16_feat.lua -- --help
```

Lecture rapide de ces commandes :

- `template_new_model.lua` valide le lifecycle `create -> build -> allocate -> init -> forward`.
- `test_serialization_smoke.lua` valide la partie checkpoint et compat de format.
- `classify_vgg16_feat.lua` sert de point d’entrée plus réaliste si tu veux passer d’un exemple minimal à un script orienté usage.

### Exemple minimal de workflow Lua

```lua
local cfg, err = Mimir.Architectures.default_config("transformer")
assert(cfg, err)

cfg.vocab_size = 4096
cfg.seq_len = 64
cfg.d_model = 128
cfg.num_layers = 2
cfg.num_heads = 4
cfg.mlp_hidden = 512

assert(Mimir.Model.create("transformer", cfg))
assert(Mimir.Model.build())
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier", 42))

local ids = {}
for i = 1, cfg.seq_len do ids[i] = 1 end

local out = Mimir.Model.forward({ __input__ = ids }, false)
assert(out)
print("forward ok, taille sortie =", #out)
```

Cet exemple illustre le chemin le plus stable de la version actuelle : registre d’architectures, entrées nommées, puis `forward()` explicite.

### Variante avec le wrapper du projet

Si tu préfères passer par le wrapper fourni par le dépôt :

```bash
./run_mimir.sh --lua scripts/templates/template_new_model.lua
```

Ce wrapper est pratique quand tu veux centraliser la manière de lancer le binaire sans réécrire toujours la commande complète.

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
./bin/mimir --lua scripts/templates/template_pipeline_only.lua
./bin/mimir --lua scripts/templates/template_pipeline_args.lua -- --no-train
```

Ces deux commandes correspondent mieux à l’état actuel du dépôt que les anciens exemples historiques :

- `template_pipeline_only.lua` montre le plus petit chemin Pipeline API.
- `template_pipeline_args.lua` montre la même logique avec arguments et overrides de runtime.

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

- Point d’entrée : [docs/00-INDEX.md](docs/00-INDEX.md)
- Installation & build : [docs/01-Getting-Started/](docs/01-Getting-Started)
- Guide utilisateur : [docs/02-User-Guide/](docs/02-User-Guide)
- Référence API : [docs/03-API-Reference/](docs/03-API-Reference)
- Internals : [docs/04-Architecture-Internals/](docs/04-Architecture-Internals)
- Performance & tuning : [docs/05-Advanced/](docs/05-Advanced)
- Contribution : [docs/06-Contributing/](docs/06-Contributing)

Ordre de lecture conseillé :

1. [docs/01-Getting-Started/01-Quick-Start.md](docs/01-Getting-Started/01-Quick-Start.md) pour faire tourner quelque chose tout de suite,
2. [docs/02-User-Guide/02-Model-Lifecycle.md](docs/02-User-Guide/02-Model-Lifecycle.md) pour comprendre le pipeline runtime,
3. [docs/03-API-Reference/10-Model.md](docs/03-API-Reference/10-Model.md) et [docs/03-API-Reference/16-Serialization.md](docs/03-API-Reference/16-Serialization.md) pour les détails d’API.

## Scripts utiles

- Templates et points de départ : [scripts/README.md](scripts/README.md)
- Smoketests Lua : [scripts/tests/](scripts/tests)
- Benchmarks : [scripts/benchmarks/](scripts/benchmarks)
- Entraînement : [scripts/training/](scripts/training)

Si tu découvres le dépôt, commence par les templates avant de passer aux scripts d’entraînement complets : ils exposent moins d’état implicite et sont plus rapides à déboguer.

## Notes (limites)

- **CPU-first** : adapté au prototypage et à des modèles modestes ; pour des LLM très gros, il faut des optimisations spécifiques (batching, KV-cache, quantization).
- Tous les layers ne sont pas au même niveau de maturité : la doc distingue **stable** vs **expérimental**.

## Licence

Voir `LICENSE`.
