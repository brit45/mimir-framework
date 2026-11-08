# Layers, opérations et disposition des poids

Comprendre le fonctionnement interne exact des composants runtime.

**Public concerné :** Développeur avancé qui modifie le moteur C/C++.

> **Prérequis**
>
> Connaître les bases C++ et la structure du dépôt.


Cette page documente l’API interne des layers : la structure `Layer`, le registre de types, les conventions de routing `inputs/output`, et les implémentations de forward/backward.

Source de vérité :

- Enum & mapping : `src/LayerTypes.hpp`
- Structure `Layer` : `src/Layers.hpp`
- Forward ops (principal) : `src/runtimes/cpu/LayerOps.hpp`
- Forward ops (extensions) : `src/runtimes/cpu/LayerOpsExt.hpp`
- Routage entre backends : `src/runtimes/RuntimeRouter.cpp`
- Dispatch CPU : `src/runtimes/cpu/RuntimeLayerDispatch.hpp`
- Façade des opérations : `src/runtimes/LayerOps.hpp` et `LayerOps.cpp`
- Noyaux CPU partagés : `src/runtimes/cpu/ConvolutionKernels.hpp`, `NormalizationKernels.hpp` et `PoolingKernels.hpp`
- Primitives Primer : `src/runtimes/PrimerGeometry.hpp`, `src/runtimes/cpu/PrimerKernels.hpp` et `src/runtimes/portable/PortablePrimerOps.hpp`

## Sur cette page

- [0) TL;DR](#0-tldr)
- [1) LayerType et normalisation de type](#1-layertype-et-normalisation-de-type)
  - [Opérations de séquence causale](#opérations-de-séquence-causale)
  - [Primitives issues de l’étude Primer](#primitives-issues-de-létude-primer)
- [2) struct Layer : champs “universels”](#2-struct-layer-champs-universels)
- [3) Helpers dans Layer](#3-helpers-dans-layer)
- [4) LayerOps : conventions et performance](#4-layerops-conventions-et-performance)
- [5) Attention : layout des poids (runtime)](#5-attention-layout-des-poids-runtime)
- [6) LayerOpsExt : extensions et statut](#6-layeropsext-extensions-et-statut)
- [7) Backward : où regarder](#7-backward-où-regarder)
- [8) Checklist lorsque vous ajoutez un nouveau layer](#8-checklist-lorsque-vous-ajoutez-un-nouveau-layer)
- [Étapes suivantes](#étapes-suivantes)

## 0) TL;DR

- Un `Layer` = un nœud du graphe, avec un `type_enum` (canonique) + des champs de config (dimensions, hyperparams).
- Le runtime dispatch en `switch (LayerType)` et appelle `LayerOps::*_forward`.
- Les poids d’un layer sont stockés dans **un bloc unifié** (`Layer::weight_block`) quand disponible, sinon fallback `weights/bias`.
- Le routing (graphe non-linéaire) se fait via `Layer.inputs` / `Layer.output` et un `TensorStore` interne “par nom”.

## 1) `LayerType` et normalisation de type

Dans `src/LayerTypes.hpp` :

- `enum class LayerType` liste les types supportés (Conv/Linear/Norm/Activation/Attention/Shape ops/…)
- `LayerRegistry::normalize_type(string)` gère les alias (ex: `Swish` → `SiLU`, `BN` → `BatchNorm2d`, `ConvTranspose` → `ConvTranspose2d`, …)
- `LayerRegistry::string_to_type(str)` donne l’enum.

Invariant : un type inconnu doit finir en `LayerType::UNKNOWN` et être rejeté lors de la construction (`Model::push`).

### Opérations de séquence causale

`CausalDepthwiseConv1d` et `SwiGLU` sont ajoutés à la fin de l’enum pour préserver
les identifiants des anciens types. Les mathématiques restent dans les runtimes.

| Type | Entrées et paramètres | Sortie |
| --- | --- | --- |
| `CausalDepthwiseConv1d` | Une séquence token-major `[T,C]`, `in_features=C`, `kernel_size=K`, `dilation=D`. Poids `[C,K]`, puis `C` biais si `use_bias=true`. | `[T,C]`, `y[t,c]=somme_i w[c,i]*x[t-i*D,c]`, zéro avant le début. |
| `SwiGLU` | Deux tenseurs de mêmes tailles, gate et value, aucun paramètre. | `SiLU(gate)*value`, même taille. |

La géométrie vérifiée est dans `src/runtimes/CausalSequenceGeometry.hpp` et les
kernels CPU dans `src/runtimes/cpu/CausalSequenceKernels.hpp`. Les kernels
OpenCL/Vulkan sont générés depuis `src/runtimes/cuda/CudaCausalSequenceOps.cuh`
via `scripts/dev/generate_portable_reference.py`. Cette source de génération
n’annonce pas de capacité native CUDA ou ROCm pour ces types.

Le routage CPU et Portable valide les formes et la finitude avant publication.
Le backward de la convolution rassemble indépendamment `dX` et `dW/db`, sans
atomiques flottantes ; SwiGLU renvoie `dGate` et `dValue`. Le planner conserve
des formes de sortie dynamiques et la calibration utilise les entrées réelles.
Les métadonnées complètes sont sauvegardées et restaurées en RawFolder et
SafeTensors, y compris lors d’une reconstruction depuis un graphe vide.

### Primitives issues de l’étude Primer

L’[étude Primer](https://arxiv.org/html/2109.08668v2), annexe A.1, décrit
le vocabulaire d’opérations utilisé par sa recherche d’architectures. Mímir ajoute
18 types de ce vocabulaire, avec forward et backward CPU et Vulkan/OpenCL.
`SquaredReLU` provient aussi de l’architecture simplifiée Primer-EZ.
Cet ajout étend les opérations disponibles ; il ne transforme pas les modèles
existants en Primer. En particulier, la convolution causale après chaque projection
Q, K et V de Primer n’est pas introduite dans leur graphe par cet ajout.

| Types | Calcul et contrat |
| --- | --- |
| `SquaredReLU`, `Square` | `max(0,x)^2`, `x*x`. Le gradient de `SquaredReLU` est nul en zéro. |
| `SqrtAbs`, `LogAbs` | `sqrt(abs(x))`, `log(abs(x))`. `LogAbs` refuse zéro ; `SqrtAbs` accepte zéro en forward mais refuse son gradient singulier. Aucun epsilon implicite. |
| `Exp`, `Abs`, `ReciprocalNoNan` | `exp(x)`, `abs(x)`, `x==0 ? 0 : 1/x`. Les gradients de `Abs` et `ReciprocalNoNan` en zéro valent zéro. |
| `Sign`, `Sin`, `Cos` | Signe, sinus et cosinus, élément par élément. Le gradient de `Sign` est défini nul. |
| `Maximum`, `Minimum` | Deux entrées de mêmes tailles, ou une entrée scalaire diffusée. En cas d’égalité, le gradient est partagé en deux. |
| `ReduceSum`, `ReduceMean`, `ReduceMin`, `ReduceMax`, `ReduceProd` | Réduction d’un axe de `shape` ; `axis` est indexé depuis zéro, avec indices négatifs acceptés. La dimension réduite reste présente et vaut 1. Sans `shape`, le tenseur est un vecteur réduit sur son unique axe. |
| `LowerTriangular` | `shape` obligatoire, de rang au moins 2. Conserve `col<=row` sur les deux derniers axes, met le reste à zéro ; les axes précédents représentent des matrices indépendantes. Le backward applique le même masque. |

Ces types sont sans paramètres (`params_count=0`). Les réductions min/max partagent
le gradient entre tous les extrema égaux. Le backward du produit multiplie les
autres facteurs, sans division par le produit total : les facteurs nuls sont
ainsi traités correctement. Son coût est quadratique dans la longueur de l’axe ;
le runtime portable borne ce travail avant lancement.

Les noms du tableau sont canoniques. Le registre accepte notamment `ReLU2`
ou `ReLUSquared` pour `SquaredReLU`, `AbsRoot` pour `SqrtAbs`,
`reciprocal_no_nan` pour `ReciprocalNoNan`, `reduce_sum` pour `ReduceSum`
et `lower_triangular` pour `LowerTriangular`. Les filtres d’inspection
[`Mimir.Layers`](../03-API-Reference/18-Layers-Module.md) utilisent les noms
canoniques ou leur variante en minuscules.

`LowerTriangular` est un masque multiplicatif. Appliqué aux logits avant softmax,
il ne suffit pas à annuler la probabilité des positions futures : l’attention
causale doit conserver son masquage adapté aux logits.

La classe C++ `PrimerReductionModel`, dérivée de `Model`, est définie dans le
[contrat des architectures](../07-Devs/08-Architecture-Contract.md). Extrait
d’utilisation après inclusion de cette définition :

```cpp
PrimerReductionModel model;
model.allocateParams();
const auto output = model.forwardPass(
    std::vector<float>{-1.f, 2.f, 3.f, 4.f, -5.f, 6.f}, false);
// Six valeurs donnent deux moyennes, de forme [2,1] : 13/3 et 52/3.
```

Les calculs sont dans `src/runtimes/cpu/PrimerKernels.hpp` et les kernels portables
sont générés depuis `src/runtimes/cuda/CudaPrimerOps.cuh`. Ce dernier est une source
de génération ; les backends CUDA/ROCm ne déclarent pas de support natif pour ces
18 types. `RuntimeRouter` choisit le backend ou le repli CPU. Les entrées,
gradients et résultats non finis sont refusés avant publication.

La préservation des subnormaux FP32 n’étant pas garantie par tous les pilotes,
le chemin portable refuse explicitement les entrées ou gradients subnormaux.
Les kernels signalent aussi un zéro calculé ambigu pouvant masquer un sous-flux,
afin que le routeur réessaie sur CPU. Ce repli conservateur peut concerner une
annulation exacte. Il évite de publier silencieusement un zéro sous l’étiquette
de calcul natif GPU.

Les métadonnées `shape`, `axis` et les connexions sont conservées dans les
checkpoints RawFolder/SafeTensors, y compris lors de la reconstruction d’un graphe
vide. Les tests `RuntimeTest.PrimerOps`, `RuntimeTest.VulkanPrimerOps`,
`ModelTest.PrimerOps` et `LuaTest.PrimerOps` couvrent les gradients, les domaines,
les formes, le routage et la sérialisation.

## 2) `struct Layer` : champs “universels”

Définition : `src/Layers.hpp`.

Ce fichier contient les types, la configuration des couches et les accesseurs
de stockage. Les calculs sur les tenseurs sont définis dans `src/runtimes/`.
Les anciennes fonctions globales `Activation`, `Conv`, `Pooling`,
`Normalization` et `Operations` ont été retirées de `Layers.hpp` : les appelants
utilisent la façade `RuntimeLayerOps` ou les noyaux du backend concerné.

Champs structurants :

- `name` : identifiant stable.
- `type` : string normalisée (compat).
- `type_enum` : version canonique.
- `params_count` : taille du bloc de poids attendu.

Stockage des paramètres :

- `tensor* weight_block` : bloc unifié (recommandé).
- `std::vector<float> weights/bias` : fallback compat.
- `std::vector<float> grad_weights/grad_bias` : gradients.

Dimensions (extraits, non exhaustif) :

- Linear : `in_features`, `out_features`, `use_bias`, `seq_len` (mode séquence)
- Conv2d : `in_channels`, `out_channels`, `input_height`, `input_width`, `kernel_size`, `stride`, `padding`
- Conv3d / Conv4d : `input_spatial_shape`, `kernel_shape`, `stride_shape`, `padding_shape`, `dilation_shape`, `in_channels`, `out_channels`, `groups`, `use_bias`
- Embedding : `vocab_size`, `embed_dim`, `padding_idx`
- Attention : `seq_len`, `embed_dim`, `num_heads`, `causal`

Routing :

- `inputs : vector<string>` (vide ⇒ `{ "x" }`)
- `output : string` (vide ⇒ `"x"`)

## 3) Helpers dans `Layer`

- `getWeights()` / `getWeightsSize()` :
  - si `weight_block` est présent : pointer + taille = `params_count` (hot-path).
  - sinon : pointer = `weights.data()`.

- `get_stride_h/get_stride_w` :
  - gère la compat entre `stride` et `stride_h/stride_w`.

## 4) `LayerOps` : conventions et performance

### 4.1 Linear

Implémentation : `LayerOps::linear_forward`.

- Poids = matrice `[out_f, in_f]` *row-major par sortie*.
- Bias optionnel stocké juste après les poids :
  - `bias = weights + in_f*out_f`.

Support “séquence” :

- si `layer.seq_len > 0` et `input.size() == seq_len * in_f` :
  - l’entrée est interprétée comme `[seq_len, in_f]` aplati,
  - la sortie est `[seq_len, out_f]` aplati.

Optimisations :

- Sélection SIMD à l'exécution selon le CPU et l'état OS (CPUID/XCR0), OpenMP si la taille le justifie.
- MatMul/Linear, leurs gradients transposés, Add/Multiply, AXPY et conversions FP16 utilisent le dispatch CPU partagé. Les réductions gardent leur ordre par sortie; les queues scalaires suivent la même politique FMA que les blocs SIMD.

### 4.2 Convolutions 3D et 4D

Types canoniques : `Conv3d`, `Conv4d`; alias `Conv3D`, `Conv4D`, `conv3d`, `conv4d`, `Convolution3D`, `Convolution4D`.
La dimension désigne les axes spatiaux : `D,H,W` ou `T,D,H,W`.
Les tenseurs sont contigus en ordre `[N,C,D,H,W]` ou `[N,C,T,D,H,W]`.
Les poids suivent `[Cout,Cin/groups,k...]`, puis les biais optionnels `[Cout]`.
Les canaux d'entrée et de sortie doivent être divisibles par `groups`.

Les vecteurs kernel/stride/padding/dilation acceptent une valeur commune ou une par axe; `input_spatial_shape` exige tous les axes.
Pour chaque axe, `output = floor((input + 2*padding - dilation*(kernel-1) - 1)/stride) + 1`.
La géométrie commune vérifie les dimensions et les débordements avant allocation ou appel backend.
Une `shape` complète fixe le batch; autrement, le runtime le déduit de la taille effective et le planner conserve une taille inconnue.
`allocateParams()` déduit le nombre de paramètres lorsque `params_count=0`, ou rejette une taille fournie incohérente.

Une architecture native déclare ces champs dans sa classe dérivée de `Model` :

```cpp
#include "Model.hpp"

class VolumeProjectionModel final : public Model {
public:
    VolumeProjectionModel() {
        push("volume/conv", "Conv3d", 0);
        auto* conv = getLayerByName("volume/conv");
        conv->inputs = {"__input__"};
        conv->output = "x";
        conv->shape = {2, 4, 8, 16, 16};
        conv->input_spatial_shape = {8, 16, 16};
        conv->in_channels = 4;
        conv->out_channels = 8;
        conv->groups = 2;
        conv->kernel_shape = {3};
        conv->stride_shape = {1};
        conv->padding_shape = {1};
        conv->dilation_shape = {1};
        conv->use_bias = true;
    }
};
```

Les forwards et backwards passent par `RuntimeRouter` et le dispatcher du backend.
Les implémentations CPU, Vulkan/OpenCL, CUDA et HIP incluent les gradients d'entrée, de poids et de biais; le worker TPU utilise JAX/XLA.
La disponibilité effective dépend du binaire, du périphérique et des limites du backend; le router conserve le fallback CPU.
Les kernels GPU calculent les gradients par destination sans atomiques et valident les résultats avant de publier les gradients.
L'ordre des réductions GPU peut différer du CPU : les tests GPU vérifient la tolérance numérique et le backend sélectionné.
Les checkpoints RawFolder/SafeTensors enregistrent une version complète des métadonnées structurelles, y compris les couches voisines d'un graphe mixte.

### 4.3 Variantes SIMD CPU

`src/runtimes/cpu/SimdDispatch.cpp` contient les variantes scalar, SSE2, AVX, AVX2 et AVX512F.
`MIMIR_CPU_ISA=auto|scalar|sse2|avx|avx256|avx2|avx512` permet de les sélectionner; `avx256` est un alias d'AVX.
Une demande invalide ou incompatible avec le CPU/OS est refusée avant exécution.
FMA, F16C, BMI et les extensions AVX512 sont détectés séparément; leur présence ne remplace pas une vérification du support de l'OS.
`Mimir.Model.hardware_caps().simd` expose ces capacités et la variante sélectionnée.

Avec GCC/Clang x86, les fonctions portent leurs cibles ISA propres; le reste du binaire reste compilé pour une cible de base portable.
Les compilateurs sans ces attributs utilisent le chemin scalaire sûr.
Les opérations d'activation conservent les fonctions mathématiques de référence, sans approximation tanh tronquée.
Les tests `RuntimeTest.SimdDispatch` comparent les résultats bit à bit entre les ISA disponibles pour une même politique arithmétique, y compris les queues scalaires.

Pour contrôler la frontière entre le modèle et les runtimes et enregistrer l'inventaire des sources et appels système :

```bash
python3 scripts/dev/audit_runtime_dispatch.py --output build/runtime_dispatch_audit.json
ctest --test-dir build --output-on-failure -R 'RuntimeTest\.(ConvND|SimdDispatch|DispatchBoundary)|LuaTest.ConvND'
```

L'audit de source détecte les appels directs hors runtime; les tests de calcul et de sélection du backend restent nécessaires pour valider l'exécution.

### 4.4 Ops élément-wise

- `Add` : support broadcast minimal (scalaire, répétition si divisible).
- `Multiply/Subtract/Divide` : tailles égales (Divide protège division par ~0).

### 4.5 Concat/Split/Chunk/Stack

`LayerOps::concat_forward` concatène simplement les buffers 1D (actuellement l’`axis` est surtout informatif).

`LayerOpsExt` contient des implémentations d’ops manquantes et des simplifications (1D/axis=0), utiles pour compléter le runtime.

## 5) Attention : layout des poids (runtime)

Pour `SelfAttention` / `MultiHeadAttention` :

- le runtime attend un bloc de poids contenant `Wqkv` puis `Wout`.
- tailles typiques :
  - `Wqkv` : `3 * E * E`
  - `Wout` : `E * E`
  - total : `4 * E * E`

Dans certains chemins (forward int), le runtime reconstruit des `std::vector` temporaires `qkv_weight/out_weight` à partir du bloc.

⚠️ Implication : `params_count` doit être configuré correctement, sinon lecture hors limites.

## 6) `LayerOpsExt` : extensions et statut

Ce fichier contient les kernels CPU d'extension, raccordés par `RuntimeLayerDispatch.hpp`.

Exemples :

- activations : `LeakyReLU`, `Mish`, `Softplus`, `HardSigmoid`, `HardSwish`
- élément-wise : `Subtract`, `Divide`
- shape ops : `Squeeze`, `Unsqueeze`
- tensor ops : `Chunk`, `Stack`

La présence dans `LayerOpsExt.hpp` ne garantit pas qu'une opération est routée : il faut aussi un `case LayerType::...` dans le dispatcher et une capacité backend exacte.

## 7) Backward : où regarder

- Backward principal : `Model::backwardPass` dans `src/Model.cpp`.
- Les calculs de backward sont implémentés dans les runtimes; `Model::backwardPass` orchestre les dépendances et snapshots.

Conventions :

- gradients d’un layer : écrits dans `layer.grad_weights` (taille = `getWeightsSize()`), et éventuellement `layer.grad_bias`.
- routing des gradients : via un `grad_store` par nom de tensor (miroir du forward).

## 8) Checklist lorsque vous ajoutez un nouveau layer

1) Ajouter l’enum dans `LayerType` (si absent).
2) Ajouter mapping string↔enum dans `LayerRegistry::string_to_type`.
3) Ajouter les champs nécessaires dans `Layer` (si pas déjà couvert par les champs “universels”).
4) Implémenter `LayerOps::<layer>_forward` (ou dans `LayerOpsExt`).
5) Ajouter le `case LayerType::<...>` dans le switch du forward runtime (et backward si nécessaire).
6) Définir le layout de poids + formule `params_count` et documenter.

## Étapes suivantes

- [Page précédente : Internals : Autograd + gradients + backward (C++)](13-Autograd-Gradients.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Internals : sérialisation (SafeTensors / RawFolder / DebugJson)](15-Serialization-Internals.md)
