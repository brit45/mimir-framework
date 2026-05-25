# Accélération GPU : guide utilisateur

Ce guide explique comment activer et configurer les backends GPU de Mímir (CUDA, ROCm, Vulkan) pour accélérer l'entraînement et l'inférence.

Référence interne : [docs/04-Architecture-Internals/03-Hardware-Backends.md](../04-Architecture-Internals/03-Hardware-Backends.md).

---

## Prérequis

### CUDA

- GPU NVIDIA avec support CUDA (Compute Capability ≥ 5.0 recommandé)
- CUDA Toolkit installé (`nvcc`, `libcudart`, `libcublas`)
- Build avec `-DENABLE_CUDA=ON`

```bash
cmake -DENABLE_CUDA=ON ..
cmake --build . -j$(nproc)
```

### ROCm

- GPU AMD avec support ROCm (RX 5000+, instinct MI)
- ROCm 5.x+ installé (`hipcc`, `libhip`, `librocblas`)
- Build avec `-DENABLE_ROCM=ON`

```bash
cmake -DENABLE_ROCM=ON ..
cmake --build . -j$(nproc)
```

### Vulkan (legacy)

- GPU avec support Vulkan Compute (pratiquement tous les GPUs modernes)
- Vulkan SDK installé, `glslangValidator` disponible
- Build avec `-DENABLE_VULKAN=ON`

---

## Activation par variables d'environnement

Tous les fast-paths sont **désactivés par défaut**. Il faut les activer explicitement.

### Activer tout sur CUDA

```bash
export MIMIR_CUDA=1
export MIMIR_CUDA_LINEAR=1
export MIMIR_CUDA_CONV=1
export MIMIR_CUDA_NORM=1
export MIMIR_CUDA_ATTENTION=1
./bin/mimir --lua scripts/training/ponyxl_ddpm_train.lua
```

### Activer tout sur ROCm

```bash
export MIMIR_ROCM=1
export MIMIR_ROCM_LINEAR=1
export MIMIR_ROCM_CONV=1
export MIMIR_ROCM_NORM=1
export MIMIR_ROCM_ATTENTION=1
./bin/mimir --lua scripts/training/ponyxl_ddpm_train.lua
```

### Mode verbeux (diagnostic)

```bash
export MIMIR_CUDA_VERBOSE=1
```

Affiche un log chaque fois qu'un fast-path GPU est utilisé ou skippé.

---

## Comprendre les seuils d'opérations

Chaque fast-path a un seuil minimal d'opérations MACs (Multiply-Accumulate) en dessous duquel le CPU est préféré (le transfert mémoire host↔device serait plus coûteux que le calcul).

| Fast-path | Variable de seuil | Défaut |
|---|---|---|
| Linear | `MIMIR_CUDA_LINEAR_MIN_OPS` | `1048576` (~1M) |
| Conv2d | `MIMIR_CUDA_CONV_MIN_OPS` | `262144` (~256K) |
| LayerNorm/RMSNorm | `MIMIR_CUDA_NORM_MIN_ELEMS` | `4096` |
| Attention | `MIMIR_CUDA_ATTENTION_MIN_OPS` | `262144` (~256K) |

**Si un fast-path n'est jamais déclenché** (visible en mode verbeux), essayez de réduire les seuils :

```bash
export MIMIR_CUDA_LINEAR_MIN_OPS=65536   # 64K au lieu de 1M
export MIMIR_CUDA_CONV_MIN_OPS=16384     # 16K au lieu de 256K
```

---

## Fast-paths disponibles par type de layer

| Layer | CUDA/ROCm | Notes |
|---|---|---|
| `Linear` | ✓ SGEMM | Toujours, si seuil atteint |
| `Conv2d` | ✓ im2col+SGEMM | Désactivé en mode `training=true` |
| `LayerNorm` | ✓ hybride | Normalisation CPU, affine GPU |
| `RMSNorm` | ✓ hybride | Idem LayerNorm |
| `GroupNorm` | ✗ CPU | Layout incompatible |
| `BatchNorm2d` | ✗ CPU | Layout incompatible |
| `SelfAttention` | ✓ multi-SGEMM | Scores+contexte sur GPU |
| `MultiHeadAttention` | ✓ multi-SGEMM | Par tête, boucle GPU |
| `CrossAttention` | ✓ multi-SGEMM | `qlen` ≠ `kvlen` supporté |
| Autres layers | ✗ CPU | Fallback silencieux |

---

## Sélection du device

Par défaut, le premier device GPU (index 0) est utilisé.

```bash
export MIMIR_CUDA_DEVICE=1   # Utiliser le 2ème GPU
```

---

## Recommandations selon le modèle

### PonyXL / DDPM

Profite fortement de l'attention (beaucoup de `SelfAttention` / `CrossAttention`) et des `Linear` dans les blocs UNet :

```bash
export MIMIR_CUDA=1
export MIMIR_CUDA_LINEAR=1
export MIMIR_CUDA_ATTENTION=1
export MIMIR_CUDA_NORM=1
```

### VAEConv

Profite surtout de Conv2d (encodeur/décodeur convolutionnel) :

```bash
export MIMIR_CUDA=1
export MIMIR_CUDA_CONV=1
export MIMIR_CUDA_NORM=1
```

### VGG16 / Tags

Principalement Conv2d et Linear :

```bash
export MIMIR_CUDA=1
export MIMIR_CUDA_CONV=1
export MIMIR_CUDA_LINEAR=1
```

---

## Pièges courants

**Fast-path Conv2d désactivé en training**

Le Conv2d GPU est uniquement disponible en mode inférence (`training=false`). Pendant l'entraînement, Conv2d retombe sur le CPU.

**Transferts mémoire trop fréquents**

Si les layers sont petits (en dessous des seuils), chaque opération génère un aller-retour host↔device coûteux. Dans ce cas, réduire les seuils ou accepter le CPU pour ces layers.

**Plusieurs GPUs**

Le framework n'a pas de support multi-GPU natif (pas de pipeline parallèle). Chaque runtime utilise un device unique. Pour multi-GPU, il faudrait des instances séparées.

**Validation correcte**

Avant de comparer des résultats, validez la justesse sur CPU d'abord. Les fast-paths GPU doivent produire des sorties numériquement proches du CPU (différences de flottant normal).

---

## Introspection Lua

```lua
-- Capacités CPU détectées
local caps = Mimir.Model.hardware_caps()

-- Activer/désactiver le chemin hardware
Mimir.Model.set_hardware(true)   -- activer
Mimir.Model.set_hardware(false)  -- forcer CPU
```
