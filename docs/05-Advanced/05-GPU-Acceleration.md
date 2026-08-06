# Accélération GPU

Optimiser, diagnostiquer et stabiliser des runs complexes.

**Public concerné :** Utilisateur intermédiaire à avancé.

> **Prérequis**
>
> Avoir déjà exécuté au moins un pipeline complet.

## Sur cette page

- [Lecture rapide](#lecture-rapide)
- [Comment ça marche](#comment-ça-marche)
- [Prérequis](#prérequis)
- [Activation pas à pas](#activation-pas-à-pas)
- [Comprendre les seuils d'opérations](#comprendre-les-seuils-dopérations)
- [Matrice de couverture runtime (etat actuel)](#matrice-de-couverture-runtime-etat-actuel)
- [Recettes par modèle](#recettes-par-modèle)
- [Diagnostic guidé](#diagnostic-guidé)
- [Sélection du device GPU](#sélection-du-device-gpu)
- [Diagnostic](#diagnostic)
- [Problèmes courants](#problèmes-courants)
- [Voir aussi](#voir-aussi)
- [Étapes suivantes](#étapes-suivantes)

## Lecture rapide

Choix recommandé :

1. Vous voulez la stabilité maximale: commence en CPU-only.
2. Vous voulez accélérer progressivement: active d'abord `Linear`, puis `Conv2d`, puis `Attention`.
3. Vous voulez diagnostiquer: active les traces runtime.

Ce guide est volontairement pratique: chaque section te donne une action concrète et un résultat attendu.

Par défaut, Mímir exécute tous les calculs sur le **CPU**. C'est intentionnel : le CPU garantit la portabilité maximale et sert de référence pour la correction numérique. Mais pour les grands modèles, comme VAEConv 512 px ou les Transformers profonds, le CPU devient rapidement le goulot d'étranglement.

Ce guide explique comment activer les **fast-paths GPU** : des chemins d'exécution spécialisés qui délèguent les opérations lourdes (multiplications matricielles, convolutions, attention) à cuBLAS (NVIDIA) ou rocBLAS (AMD). Le reste des layers continue de s'exécuter sur CPU, sans aucun changement dans vos scripts Lua.

> **Conseil :** si vous voulez juste démarrer rapidement, allez directement à la section [Recettes par modèle](#recettes-par-modèle).

---

## Comment ça marche

Mímir maintient une **pile de runtimes** ordonnée par priorité :

```text
1. ROCm Runtime
2. CUDA Runtime
3. Vulkan Runtime
4. OpenCL Runtime
5. CPU Runtime (toujours actif, fallback universel)
```

Pour chaque layer à exécuter, le runtime le plus prioritaire *tente* de le prendre en charge. Il vérifie deux conditions :

1. **A-t-il un fast-path pour ce type de layer ?** (Linear, Conv2d, etc.)
2. **La taille du tenseur dépasse-t-elle le seuil minimal ?** (en dessous du seuil, le transfert mémoire host↔device coûterait plus que le calcul lui-même)

Si l'une des deux conditions échoue, le runtime passe la main au suivant. Ce **fallback est silencieux et automatique** — vous n'avez pas à le gérer.

Point clé : l'objectif n'est pas de forcer "100% GPU", mais d'obtenir le meilleur compromis perf/stabilité par type de layer.

**Exemple :**

```text
[Layer Linear 4096×4096] → CUDA actif ? Seuil atteint ? → oui → cuBLAS SGEMM ✓
[Layer GroupNorm]         → CUDA actif ? Fast-path dispo ? → non → CPU       ✓
[Layer Linear 16×16]      → CUDA actif ? Seuil atteint ?  → non → CPU        ✓
```

> **Note :** les fast-paths GPU ne modifient pas les résultats numériques au-delà de la précision float32 habituelle. Si vous observez des divergences par rapport à une exécution CPU-seul, activez le [mode verbeux](#diagnostic) pour voir exactement quels layers passent par GPU.

---

## Prérequis

### Pour NVIDIA (CUDA / cuBLAS)

Vous avez besoin de :

- Un GPU NVIDIA avec support CUDA (Compute Capability ≥ 5.0 recommandé)
- CUDA Toolkit installé (`nvcc`, `libcudart`, `libcublas`)
- Mímir compilé avec `-DENABLE_CUDA=ON`

```bash
# Dans le dossier build/
cmake -DENABLE_CUDA=ON ..
cmake --build . -j$(nproc)
```

### Pour AMD (ROCm / rocBLAS)

Vous avez besoin de :

- Un GPU AMD compatible ROCm (RX 5000 ou supérieur, gamme Instinct)
- ROCm 5.x ou supérieur installé (`hipcc`, `libhip`, `librocblas`)
- Mímir compilé avec `-DENABLE_ROCM=ON`

```bash
cmake -DENABLE_ROCM=ON ..
cmake --build . -j$(nproc)
```

> **Avertissement :** si le build n'inclut pas `ENABLE_CUDA` ou `ENABLE_ROCM`, les variables d'environnement correspondantes seront ignorées sans message d'erreur. Vérifiez la sortie de `cmake` pour confirmer que le backend est activé.

Checklist rapide avant toute investigation performance :

1. Le backend est compilé (`ENABLE_*`).
2. Le binaire démarre sans crash en CPU-only.
3. Les scripts smoke passent avant activation GPU.

### Pour Vulkan

Vulkan est disponible via le runtime routeur avec des kernels compute dédiés.

Scope actuel Vulkan (forward):

- `Linear`
- `MatMul`
- `BatchMatMul`
- `Conv2d`
- `ConvTranspose2d`
- `Add`
- `Multiply`
- `ReLU`

Pour les autres layers, le fallback routeur vers CPU/CUDA/ROCm reste automatique.

- Vulkan SDK installé, `glslangValidator` disponible
- Compilé avec `-DENABLE_VULKAN=ON`

---

## Activation pas à pas

Les fast-paths matériels sont **activés par défaut** quand le backend est compilé et le device détecté. Vous pouvez toujours affiner via variables d'environnement ou forcer la désactivation avec `MIMIR_DISABLE_*`.

### Activer tous les fast-paths sur CUDA

```bash
export MIMIR_CUDA_LINEAR=1     # Linear → cuBLAS SGEMM
export MIMIR_CUDA_CONV=1       # Conv2d → im2col + SGEMM
export MIMIR_CUDA_NORM=1       # LayerNorm/RMSNorm → hybride GPU
export MIMIR_CUDA_ATTENTION=1  # Attention → multi-SGEMM

./bin/mimir --lua scripts/training/train_vae_conv.lua
```

### Activer tous les fast-paths sur ROCm

```bash
export MIMIR_ROCM_LINEAR=1
export MIMIR_ROCM_CONV=1
export MIMIR_ROCM_NORM=1
export MIMIR_ROCM_ATTENTION=1

./bin/mimir --lua scripts/training/train_vae_conv.lua
```

> **Conseil :** ajoutez ces exports dans votre script de lancement (`.sh`) ou dans votre `.env` pour ne pas les réécrire à chaque fois. Voir `run_mimir.sh` à la racine du projet comme exemple.

Parcours de mise en route conseillé :

1. Activer un seul backend GPU.
2. Activer `Linear` uniquement.
3. Mesurer.
4. Ajouter `Conv2d`.
5. Mesurer à nouveau.
6. Ajouter `Attention` en dernier.

---

## Comprendre les seuils d'opérations

Chaque fast-path possède un **seuil minimal de MACs** (Multiply-Accumulate operations). En dessous de ce seuil, Mímir préfère le CPU, car transférer de petits tenseurs entre la RAM et la VRAM prendrait plus de temps que le calcul lui-même.

| Fast-path | Variable de seuil | Valeur par défaut |
| --- | --- | --- |
| `Linear` | `MIMIR_CUDA_LINEAR_MIN_OPS` | `0` |
| `Conv2d` | `MIMIR_CUDA_CONV_MIN_OPS` | `0` |
| `LayerNorm` / `RMSNorm` | `MIMIR_CUDA_NORM_MIN_ELEMS` | `0` |
| `Attention` | `MIMIR_CUDA_ATTENTION_MIN_OPS` | `0` |

Pour les variables ROCm, remplacez `CUDA` par `ROCM`.

**Quand réduire les seuils ?** Si le mode verbeux indique qu'un fast-path est systématiquement ignoré parce que les tenseurs sont petits, abaissez le seuil :

```bash
export MIMIR_CUDA_LINEAR_MIN_OPS=65536   # 64 K au lieu de 1 M
export MIMIR_CUDA_CONV_MIN_OPS=16384     # 16 K au lieu de 256 K
```

> **Note :** des seuils trop bas sur des micro-layers peuvent dégrader les performances à cause de l'overhead de synchronisation (`cudaDeviceSynchronize`). En cas de doute, comparez les temps d'exécution avec et sans les réductions de seuil.

Méthode simple de tuning :

1. Commencer avec les seuils par défaut.
2. Baisser un seul seuil à la fois.
3. Rejouer le même script (mêmes entrées, même seed).
4. Garder la valeur seulement si le temps global baisse de manière stable.

---

## Matrice de couverture runtime (etat actuel)

Objectif: visualiser clairement ce qui est utilisable aujourd'hui, par runtime et par famille d'ops.

Legende des statuts:

- **Complet**: implementation native dans le runtime pour cette direction (forward ou backward).
- **Partiel**: seulement une sous-partie de la famille d'ops est native.
- **Conditionnel**: implementation stricte disponible, mais necessite un etat/cache/intermediaire explicite.
- **Placeholder**: chemin branche mais non natif (souvent fallback CPU), ou branche qui ne couvre pas la famille de facon exploitable en GPU.
- **Absent**: pas de support explicite dans le runtime pour cette direction.

Convention de validation backward CPU:

- Le switch backward CPU est en mode **strict explicite**: `1 LayerType = 1 case` (pas de regroupement de labels).
- Les ops non supportees en strict retournent explicitement `false` (pas d'approximation silencieuse).

### Forward

| Famille d'ops | CPU | CUDA | ROCm | OpenCL | Vulkan |
| --- | --- | --- | --- | --- | --- |
| Linear / MatMul / BatchMatMul | Complet | Complet | Complet | Complet | Complet |
| Convolutions (Conv2d, ConvTranspose2d, Conv1d, Depthwise) | Complet | Partiel (Conv2d) | Partiel (Conv2d) | Absent | Partiel (Conv2d + ConvTranspose2d) |
| Normes (BatchNorm, LayerNorm, GroupNorm, InstanceNorm, RMSNorm) | Complet | Partiel (LayerNorm + RMSNorm) | Partiel (LayerNorm + RMSNorm) | Absent | Absent |
| Element-wise binaires (Add/Subtract/Multiply/Divide) | Complet | Complet | Complet | Complet | Partiel (Add + Multiply) |
| Activations unaires (ReLU, LeakyReLU, GELU, SiLU, Sigmoid, Tanh, Softplus, Mish, Hard*) | Complet | Complet | Complet | Complet | Partiel (ReLU, SiLU, GELU, Sigmoid, Tanh) |
| Attention (Self/MultiHead/Cross) | Complet | Complet | Complet | Absent | Absent |
| Pooling / Reshape / Routing (Flatten, View, Permute, Concat, Split, etc.) | Complet | Absent (fallback router) | Absent (fallback router) | Absent | Absent |
| RNN (LSTM/GRU/RNN) | Complet | Absent | Absent | Absent | Absent |

### Backward

| Famille d'ops | CPU | CUDA | ROCm | OpenCL | Vulkan |
| --- | --- | --- | --- | --- | --- |
| Convolutions (Conv2d, ConvTranspose2d, Conv1d, Depthwise) | Complet | Placeholder (fallback CPU selon op) | Placeholder (fallback CPU selon op) | Absent | Absent |
| Linear / Bilinear / MatMul / BatchMatMul | Complet | Complet | Placeholder | Placeholder | Absent |
| Embedding / EmbeddingBag | Complet | Placeholder | Placeholder | Absent | Absent |
| Element-wise binaires (Add/Subtract/Multiply/Divide) | Complet | Complet | Placeholder | Placeholder | Absent |
| Activations unaires (ReLU, LeakyReLU, GELU, GEGLU, SiLU, Sigmoid, Tanh, Softplus, Mish, Hard*) | Complet | Complet | Placeholder | Placeholder | Absent |
| Normes (BatchNorm/LayerNorm/RMSNorm/GroupNorm/InstanceNorm) | Complet | Placeholder | Placeholder | Absent | Absent |
| Pooling (2D/1D/Global/Adaptive/TokenMean) | Complet | Placeholder | Placeholder | Absent | Absent |
| Shape/Tensor ops (Flatten/Reshape/View/Squeeze/Unsqueeze/Transpose/Permute/Concat/Split/Chunk/Stack) | Complet | Placeholder | Placeholder | Absent | Absent |
| Upsampling (Nearest/Bilinear/Bicubic/PixelShuffle) | Complet | Placeholder | Placeholder | Absent | Absent |
| PatchEmbed | Complet | Placeholder | Placeholder | Absent | Absent |
| Reparameterize | Complet | Placeholder | Placeholder | Absent | Absent |
| Dropout / Dropout2d / AlphaDropout | Complet | Placeholder | Placeholder | Absent | Absent |
| Attention (Self/MultiHead/Cross) | Complet | Placeholder | Placeholder | Absent | Absent |
| RNN (LSTM/GRU/RNN) | Complet | Absent | Absent | Absent | Absent |
| Constant / Lambda | Complet | Absent | Absent | Absent | Absent |

Notes de lecture importantes:

- Le routeur runtime peut rendre une operation "utilisable" globalement via fallback CPU, meme si le runtime GPU courant est en **Absent** ou **Placeholder**.
- Cote backward, le runtime CPU couvre explicitement la majorite des familles (conv, matmul, activations, normes, pooling, shape/tensor, upsampling, patch embed), avec dispatch strict `1:1` par `LayerType`.
- Cote CPU, les familles backward Reparameterize, Dropout, Attention, RNN, Constant et Lambda sont maintenant implementees explicitement dans le runtime.
- Vulkan n'override pas `backwardLayer`, donc backward Vulkan est **Absent** (retour `false` de l'implementation par defaut).

Comment utiliser la matrice :

1. Vérifier d'abord le support forward de la famille d'ops dominante de votre modèle.
2. Si vous entraînez le modèle, vérifiez ensuite le support de la passe arrière.
3. En absence de support GPU backward, prévoir fallback CPU explicite.

---

## Recettes par modèle

### VAEConv (encodeur/décodeur convolutionnel)

VAEConv est dominé par des blocs Conv2d. Le fast-path Conv2d est utile en entraînement comme en génération.

```bash
export MIMIR_CUDA_CONV=1
export MIMIR_CUDA_NORM=1
export MIMIR_CUDA_LINEAR=1
```

### VGG16 / Classification par tags

Ce modèle mélange Conv2d (feature extraction) et Linear (classifier).

```bash
export MIMIR_CUDA_CONV=1
export MIMIR_CUDA_LINEAR=1
```

## Diagnostic guidé

Activer les logs pour comprendre le dispatch réel :

```bash
export MIMIR_ACCEL_VERBOSE=1
export MIMIR_RUNTIME_TRACE=1
```

Interprétation rapide :

1. Si un layer reste CPU: vérifier support op et seuil.
2. Si un backend échoue tôt: vérifier son initialisation et ses bibliothèques.
3. Si les performances empirent: relever les seuils pour éviter les petits offloads coûteux.

---

## Sélection du device GPU

Par défaut, Mímir utilise le premier GPU (index 0). Pour choisir un autre device (sur une machine multi-GPU) :

```bash
export MIMIR_CUDA_DEVICE=1   # Utiliser le 2ème GPU NVIDIA
export MIMIR_ROCM_DEVICE=0   # Utiliser le 1er GPU AMD
```

---

## Diagnostic

### Mode verbeux

Activez le mode verbeux pour savoir exactement ce qui passe par GPU :

```bash
export MIMIR_ACCEL_VERBOSE=1
export MIMIR_RUNTIME_TRACE=1
```

Exemple de sortie :

```text
[CUDA] Linear 4096x4096: SGEMM (ops=16777216 >= 0) ✓
[CUDA] Conv2d 64x3x3: SGEMM fast-path ✓
[CUDA] LayerNorm 512: hybride (512 >= 4096 ? non) → fallback CPU
```

Chaque ligne indique le type de layer, sa taille, la décision prise et la raison.

Avec `MIMIR_RUNTIME_TRACE=1`, vous obtenez en plus la trace d'execution reelle layer-par-layer:

```text
[runtime-trace] layer#12 name='unet/down1/conv' type='Conv2d' backend=CUDA call=runtime_router.dispatchForwardLayer output_size=262144
[runtime-trace] layer#13 name='unet/down1/relu' type='ReLU' backend=CPU call=cpu_switch_kernel output_size=262144
[runtime-trace] layer#14 name='unet/down1/proj' type='Linear' backend=CUDA call=linear_accel_chain output_size=65536
```

Cette vue permet de distinguer facilement les layers effectivement offloades de ceux qui restent en fallback CPU.

---

## Problèmes courants

### "Comment forcer CPU-only malgré l'auto-détection matériel ?"

Utilisez les kill-switchs runtime:

```bash
export MIMIR_DISABLE_CUDA=1
export MIMIR_DISABLE_ROCM=1
export MIMIR_DISABLE_VULKAN=1
export MIMIR_DISABLE_OPENCL=1
```

### "Mon fast-path Conv2d n'est jamais utilisé pendant l'entraînement"

Ce n'est plus le comportement attendu. Vérifiez:

1. `MIMIR_CUDA_CONV=1` (ou `MIMIR_ROCM_CONV=1`).
2. Le seuil (`MIMIR_CUDA_CONV_MIN_OPS` / `MIMIR_ROCM_CONV_MIN_OPS`) n'est pas trop haut.
3. Le backend est bien compilé (`ENABLE_CUDA` / `ENABLE_ROCM`) et initialisé.

### "Les performances sont pires qu'en CPU-only"

Les layers trop petits génèrent de nombreux aller-retours host↔device. Essayez d'augmenter les seuils ou de n'activer que les fast-paths pour les layers vraiment lourds (typiquement `Linear` et `Attention`).

### "J'obtiens des résultats numériquement différents avec GPU"

Les opérations en virgule flottante ne sont pas strictement associatives — l'ordre des additions change légèrement les résultats. C'est normal et attendu. Si les différences sont supérieures à ~1e-4 en float32, activez le mode verbeux et vérifiez qu'aucun layer ne produit des NaN.

### "Le build ne détecte pas CUDA alors que le Toolkit est installé"

Vérifiez que `nvcc` est dans votre `PATH` et que `CUDA_TOOLKIT_ROOT_DIR` est défini si CMake ne le trouve pas automatiquement :

```bash
cmake -DENABLE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda ..
```

---

## Voir aussi

- [Architecture des backends hardware](../04-Architecture-Internals/03-Hardware-Backends.md) — détails d'implémentation C++
- [Internals GPU Runtimes](../04-Architecture-Internals/21-GPU-Runtimes.md) — guide pour étendre les runtimes
- [Entraînement](../02-User-Guide/04-Training.md) — workflow d'entraînement complet

## Étapes suivantes

- [Page précédente : Advanced — Carte du code source (C/C++)](04-Source-Code-Map.md)
- [Index de la documentation](../00-INDEX.md)
- [Revenir à la documentation](../00-INDEX.md)
