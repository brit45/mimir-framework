# Runtimes GPU : Vulkan, OpenCL, CUDA et ROCm

Comprendre le fonctionnement interne exact des composants runtime.

**Public concerné :** Développeur avancé qui modifie le moteur C/C++.

> **Prérequis**
>
> Connaître les bases C++ et la structure du dépôt.

Cette page documente l'implémentation interne des backends GPU de Mímir : architecture, fast-paths, conventions de code.

Sources de vérité :

- `src/runtimes/AbstractRuntime.hpp` — interface + `RuntimeConfig`
- `src/runtimes/cuda/CudaRuntime.hpp` / `.cpp` — CUDA (cuBLAS)
- `src/runtimes/rocm/RocmRuntime.hpp` / `.cpp` — ROCm (rocBLAS)
- `src/runtimes/cpu/CpuRuntime.hpp` / `.cpp` — CPU référence
- `src/runtimes/vulkan/VulkanRuntime.hpp` / `.cpp` — Vulkan
- `src/runtimes/opencl/OpenCLRuntime.hpp` / `.cpp` — OpenCL
- `src/runtimes/PointwiseFusionKernels.cuh` — kernels pointwise partagés
- `src/runtimes/portable/PortablePointwiseFusion.hpp` — soumission fusionnée Vulkan/OpenCL
- `src/runtimes/portable/PortableRuntime.cpp` — capacités et dispatch portables
- `src/runtimes/PrimerGeometry.hpp` et `src/runtimes/portable/PortablePrimerValidation.hpp` — contrats et validation des opérations Primer

Guide utilisateur associé : [docs/05-Advanced/05-GPU-Acceleration.md](../05-Advanced/05-GPU-Acceleration.md).

---

## Sur cette page

- [1) Interface commune (AbstractRuntime)](#1-interface-commune-abstractruntime)
- [2) RuntimeConfig — paramètres et parsing](#2-runtimeconfig--paramètres-et-parsing)
- [3) CudaRuntime — implémentation](#3-cudaruntime-implémentation)
- [4) Fast-paths détaillés](#4-fast-paths-détaillés)
- [5) ROCm Runtime — différences avec CUDA](#5-rocm-runtime-différences-avec-cuda)
- [6) Vulkan Runtime — kernels compute réels (SPIR-V)](#6-vulkan-runtime--kernels-compute-réels-spir-v)
- [7) Invariants et gotchas](#7-invariants-et-gotchas)
- [8) Ajouter un nouveau fast-path GPU](#8-ajouter-un-nouveau-fast-path-gpu)
- [OpenCL et Vulkan : kernels de référence partagés](#opencl-et-vulkan-kernels-de-référence-partagés)
- [Opérations Primer et séquences causales](#opérations-primer-et-séquences-causales)
- [Fusion native des chaînes pointwise](#fusion-native-des-chaînes-pointwise)
- [Validation de la fusion au 28 octobre 2026](#validation-de-la-fusion-au-28-octobre-2026)
- [Étapes suivantes](#étapes-suivantes)

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

### Patron de dispatch (Model.cpp + RuntimeRouter)

```cpp
// Le forward route via RuntimeRouter.
// Le router essaie chaque backend initialisé dans l'ordre,
// et appelle forwardLayer(...).
// Chaque runtime sélectionne les LayerType supportés via
// switch (layer.type_enum) et retourne false sinon.
```

Le dispatch `forwardLayer(...)` est utilisé dans le forward principal pour Linear, Convolution, Normes, Attention, et autres ops supportées par les backends.

---

## 2) `RuntimeConfig` — paramètres et parsing

`RuntimeConfig` porte les flags `linear_enabled`, `conv_enabled`,
`norm_enabled`, `attention_enabled`, leurs minima de travail, `disabled`,
`verbose` et `device_index`. Les minima valent `0` par défaut : la politique
centrale compare les latences, sans seuil de vitesse implicite par backend.
Les limites de protection des kernels restent applicables.

`RuntimeConfig::fromEnv("CUDA")` lit notamment `MIMIR_DISABLE_CUDA`,
`MIMIR_ACCEL_VERBOSE`, `MIMIR_CUDA_DEVICE`, `MIMIR_CUDA_LINEAR` et
`MIMIR_CUDA_LINEAR_MIN_OPS`. Les mêmes suffixes existent pour les autres
backends. Il n’existe pas de kill-switch global `MIMIR_CUDA=0`.

Les flags sont initialement `false` dans une configuration construite directement
et dans `fromEnv()` en l’absence de variable correspondante. L’initialisation
des runtimes par `Model` applique ensuite ses valeurs par défaut, notamment
l’activation des fast-paths CUDA/ROCm. Le lanceur peut aussi désactiver un backend.
Consulter la [référence des variables](../03-API-Reference/22-Environment-Variables.md)
pour distinguer ces niveaux.

---

## 3) CudaRuntime — implémentation

L’état backend est défini dans `CudaRuntime.cpp`, derrière le pointeur
`Impl*` du header. Il conserve le handle cuBLAS, six buffers `linear_scratch`,
deux buffers de training et les mutex correspondants. `DeviceBuf::alloc()`
réutilise la capacité disponible, et ne réalloue que lorsqu’elle est insuffisante.
La configuration est stockée dans `AbstractRuntime::config_`.

L’initialisation vérifie le device CUDA et crée le handle cuBLAS. Les entrées et
les poids sont réuploadés selon l’opération ; la réutilisation d’une allocation
ne signifie pas que son contenu reste valable après une mise à jour des poids.

---

## 4) Fast-paths détaillés

> **Important :** la section ci-dessous décrit l'implémentation de `CudaRuntime::forwardLayer` / `RocmRuntime::forwardLayer`, qui est effectivement sollicitée par le dispatch principal selon le type de layer et les flags runtime.

### 4.1 Pattern commun

Un chemin natif valide son état, ses flags et sa géométrie, réserve ses
buffers, charge les données requises, lance les kernels ou les opérations BLAS,
puis restitue le résultat à l’hôte. Les copies D2H bloquantes des buffers CUDA
complètent les kernels précédents du flux par défaut ; elles évitent une
synchronisation globale supplémentaire sur ces chemins.

Un retour `false` permet au routeur d’essayer une autre exécution. La présence
d’un backend en tête de priorité ne force pas sa sélection :
`ExecutionCalibration` compare les candidats natifs pour les opérations
rejouables. Les refus de géométrie et les échecs ne sont pas attribués à une
exécution GPU réussie.

### 4.2 Convolutions et normalisations natives

`CudaNativeOps.cuh` contient les kernels Conv2d et les normalisations
LayerNorm/RMSNorm/GroupNorm/InstanceNorm. `CudaTrainingOps.cuh` couvre les
variantes BatchNorm et les opérations stochastiques. Les convolutions étendues
(Conv1d, DepthwiseConv2d, ConvTranspose2d) et leurs gradients utilisent également
des kernels device. Les calculs de réduction et de convolution des chemins
natifs restent sur GPU ; les tenseurs sont transférés pour le staging.

### 4.3 Attention, récurrence et NMS

`CudaAttentionOps.cuh` fournit les projections, scores, masque causal,
softmax et gradients d'attention sur le device. `CudaRecurrentOps.cuh` couvre
RNN, GRU et LSTM selon les formes admises par le contrat CPU.

`CudaDetectionOps.cuh` effectue la sélection NMS sur GPU et fournit le backward
stop-gradient : un gradient nul de la taille de chaque entrée. Le dispatcher
CPU utilise le même contrat. Le backward `Lambda` reste indisponible en CUDA.
La présence de ces kernels ne prouve pas leur exécution sur GPU NVIDIA :
les validations locales récentes portent sur CPU, OpenCL et Vulkan.

---

## 5) ROCm Runtime — différences avec CUDA

Les chemins historiques ROCm utilisent les équivalents HIP/rocBLAS des API
CUDA. Les extensions natives CUDA récentes ne sont pas automatiquement
portées dans `RocmRuntime` :

| CUDA | ROCm |
| --- | --- |
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

Les flags et seuils sont propres à chaque backend. Consulter les capacités
déclarées et les traces pour distinguer calcul natif et repli CPU.

---

## 6) Vulkan Runtime — kernels compute réels (SPIR-V)

Vulkan est branché via `VulkanRuntime` + `VulkanCompute::ComputeEngine` et dispatché par `RuntimeRouter`.

Quelques chemins spécialisés historiques de `VulkanCompute::ComputeEngine` :

- `Linear`
- `MatMul`
- `BatchMatMul`
- `Add`
- `Multiply`
- `ReLU`

Shaders associés (compilés au build):

- `shaders/linear_forward.comp`
- `shaders/add_forward.comp`
- `shaders/mul_forward.comp`
- `shaders/relu_forward.comp`

Pipeline technique:

1. CMake compile les `.comp` en `.spv` via `glslangValidator`. `mimir_core` dépend de la cible `mimir_shaders`, y compris lorsqu'un client SDK ou un test est compilé sans l'exécutable `mimir`.
2. Les `.spv` sont produits dans le dossier `shaders` du build. `MIMIR_VULKAN_SHADER_DIR` permet un dossier explicite ; sous Linux, le runtime cherche ensuite près de l'exécutable et de son parent avant les anciens chemins relatifs au répertoire courant. `MIMIR_VULKAN_LINEAR_SPV` garde la priorité pour Linear.
3. `VulkanCompute.hpp` charge les SPIR-V, crée pipeline/layout/descriptor pool.
4. `VulkanRuntime::forwardLayer` route les layers compatibles vers ces kernels.
5. Les autres opérations et le backward passent par `PortableRuntime`, décrit ci-dessous.
6. En cas d'indisponibilité shader/device, le runtime retourne `false` et le routeur propose l'opération au backend suivant.

`RuntimeTest.VulkanShaderPaths` vérifie les onze fichiers, les remplacements
explicites et les anciens chemins. Sous Linux, `RuntimeTest.VULKANNativeOps`
s'exécute depuis un dossier vide : une validation ne doit pas dépendre du
répertoire du dépôt. Ce test exige la parité GPU native et échoue si un shader
manque ; l'initialisation du device seule ne suffit pas.

---

## 7) Invariants et gotchas

### Buffers réutilisables et contenu à jour

Les runtimes CUDA/ROCm réutilisent des buffers de scratch protégés par mutex.
Le staging portable garde aussi des allocations device réutilisables. Les
résultats et paramètres sont transférés selon le contrat de chaque opération ;
un cache d’allocation ne constitue pas un cache immuable des poids.

### Forward, backward et contexte sauvegardé

Les backends exposent des capacités forward et backward distinctes. La fusion
pointwise dispose de kernels pour les deux directions. Pour les autres
opérations, vérifier `queryBackwardOperationCapability()` et le runtime
réellement sélectionné : la couverture CUDA n’implique pas une couverture ROCm
identique. L’entraînement conserve les snapshots et masques nécessaires aux
opérations qui les utilisent.

### Pas de mixed precision

Tous les fast-paths travaillent en `float32`. Les types `F16`/`BF16` du modèle ne sont pas utilisés ici (la conversion est faite ailleurs, si applicable).

### Masque causal sur le device

Les kernels d'attention de référence appliquent le masque causal sur le device.
Le coût du staging et des kernels séquentiels doit être mesuré pour les longues
séquences ; la couverture native ne constitue pas une garantie de performance.

### Ordre des builds

CUDA et ROCm sont deux options CMake indépendantes (`ENABLE_CUDA` et
`ENABLE_ROCM`). Les deux backends peuvent donc être présents dans le même
binaire si leurs bibliothèques sont détectées. Au runtime, le routeur essaie
ROCm avant CUDA, puis Vulkan, OpenCL et CPU ; une opération refusée par un
backend est proposée au suivant.

---

## 8) Ajouter un nouveau fast-path GPU

Les layers du modèle passent par `RuntimeRouter` en forward et backward. Ajouter le calcul dans le runtime, déclarer sa capacité réelle et tester le runtime sélectionné ; aucun nouveau switch mathématique ne doit être ajouté dans `Model.cpp`.

1. Ajouter les flags dans `RuntimeConfig` (`src/runtimes/AbstractRuntime.hpp`)
2. Ajouter le parsing `fromEnv` correspondant
3. Implémenter le `case` dans `CudaRuntime::forwardLayer`
4. Implémenter le `case` miroir dans `RocmRuntime::forwardLayer`
5. Mettre à jour le tableau de synthèse dans `docs/04-Architecture-Internals/03-Hardware-Backends.md`

## Étapes suivantes

- [Page précédente : Internals : CLI (binaire `mimir`) et points d’entrée](20-CLI-EntryPoints.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Internals : Execution Planner (C++)](22-Planning.md)

## OpenCL et Vulkan : kernels de référence partagés

Les backends Vulkan et OpenCL conservent leurs chemins spécialisés et utilisent
`PortableRuntime` pour les autres opérations et les passes arrière. Leur
couverture déclarée par type correspond au dispatcher CPU exécutable : **94 types en
forward et 95 en backward** dans le registre actuel, dont NMS avec stop-gradient.
Ces déclarations supposent le backend compilé et activé ; elles ne prouvent pas
l'acceptation d'une opération particulière ni son exécution sur un périphérique.
`Lambda` est refusé
en forward par le dispatcher CPU lui-même ; son backward identité est supporté
sur CPU, Vulkan et OpenCL. CUDA ne prend pas en charge le backward `Lambda` et
ne déclare pas de kernels natifs pour les 18 opérations Primer,
`CausalDepthwiseConv1d` ou `SwiGLU` : ces types restent en `HostFallback` sur
CUDA et ROCm. La présence d'une source CUDA utilisée par le générateur portable
ne suffit pas à publier une capacité native dans ces backends.

Les familles couvertes comprennent convolutions, normalisations, attention,
projections/embeddings, pooling, transformations de tenseurs, récurrence,
upsampling, dropout, reparamétrisation, convolutions ND, séquences causales
et primitives Primer. Les configurations, formes de tenseurs
et limites du device restent vérifiées ; cette couverture par type ne garantit
pas que toute forme soit acceptée.

Les kernels OpenCL/Vulkan à sorties indépendantes utilisent une grille
parallèle, et les réductions prises en charge utilisent des groupes coopératifs.
Les algorithmes intrinsèquement séquentiels conservent un chemin de référence
borné. `PortableWorkBudget.hpp` vérifie séparément les limites de travail et de
staging. La couverture numérique ne garantit pas des performances supérieures
au CPU. Le staging des tenseurs et l'accumulation des
gradients de paramètres retournés restent sur l'hôte. Les calculs forward et
backward ne sont pas remplacés par un appel arithmétique CPU caché.

### Durée de vie du cache de staging

La table locale au thread conserve une référence faible au périphérique.
`PortableRuntime` et les scopes de calcul actifs en gardent les références
fortes ; le chemin matriciel spécialisé Vulkan conserve aussi sa référence
tant que son `ComputeEngine` existe. Les buffers et pipelines restent donc
réutilisés entre les appels d'un runtime vivant. Le dernier propriétaire
libère le périphérique pendant sa fermeture, avant la destruction du stockage
local au thread et des ressources internes du pilote.

Les périphériques portables d'optimiseurs et de pertes appartiennent également
au runtime. Leurs caches par backend/device sont protégés séparément pendant
les calculs et sont libérés avec lui, sans cache GPU propriétaire dans le
stockage local au thread.

`PortableWorkBudget` vérifie que le cache seul ne prolonge pas cette durée de
vie. Les tests de parité ferment explicitement le runtime et vérifient qu'il
ne reste ni allocation de staging, ni commande en attente, ni périphérique
détenu par le cache avant d'imprimer `GPU_NATIVE_PARITY_COMPLETE`.

### Sources et génération

- `src/runtimes/portable/PortableRuntime.cpp` valide et orchestre les opérations.
- `PortableStaging.hpp` prépare les buffers et les arguments des kernels.
- `PortableDevice.cpp` soumet les calculs à OpenCL compute ou Vulkan compute.
- `reference.cl` et `reference.comp` contiennent les kernels générés.
- `PortableShaderData.hpp` embarque le source OpenCL et le SPIR-V Vulkan.

Le générateur réutilise les algorithmes GPU existants alignés sur le contrat
CPU. Pour régénérer les fichiers, il faut Python 3, `glslangValidator` et
`spirv-val` :

```bash
python3 scripts/dev/generate_portable_reference.py
```

Cette génération a été exécutée lors de l'implémentation. Les builds ordinaires
utilisent les sources/SPIR-V embarqués ; les kernels de référence ne dépendent
pas d'un toolkit CUDA.

### Validation

Recette CMake/CTest pour un nouveau build avec tests (à adapter ; commandes
non rejouées sous le nom `build-gpu-tests` pour cette mise à jour). Les preuves
matérielles locales sont précisées ci-dessous :

```bash
cmake -S . -B build-gpu-tests -DENABLE_OPENCL=ON -DENABLE_VULKAN=ON -DMIMIR_ENABLE_TESTS=ON -DBUILD_TESTING=ON
cmake --build build-gpu-tests --target PortableNativeOps RuntimeTest.NMS GpuBackwardParity -j2
ctest --test-dir build-gpu-tests --output-on-failure -R '^RuntimeTest\.(NMS|(OPENCL|VULKAN)(NativeOps|BackwardParity))$'
```

Le 11 octobre 2026, les exécutables de tests ont été compilés et lancés
directement contre la bibliothèque locale : 165 cas numériques et quatre
configurations NMS par backend passent avec OpenCL/Rusticl sur le GPU AMD
Radeon Graphics et Vulkan/RADV RENOIR sur ce même GPU. Les régressions de
gradients historiques et le test CPU NMS passent également. Vulkan/Lavapipe
a aussi passé les comparaisons numériques pendant l'implémentation.

Les tests NMS incluent les classes optionnelles, les entrées vides, les formes
invalides, le stop-gradient et la sélection du runtime. Le backward CPU est
la référence des comparaisons GPU. CUDA dispose du kernel NMS forward/backward
et des tests correspondants ; son exécution sur GPU NVIDIA reste à valider.
Ces résultats prouvent la conformité et le routage, sans mesure de performance.

Voir aussi [le contrat NMS](../03-API-Reference/01-Layers.md#contrat-de-nms)
et [les sources portables](../../src/runtimes/portable/README.md).

## Opérations Primer et séquences causales

Les 18 nouveaux types proviennent du vocabulaire étudié dans Primer. Leurs
[contrats mathématiques et exemples Lua](14-Layers-And-Ops.md#primitives-issues-de-létude-primer)
définissent les domaines, axes, égalités et gradients ; cet ajout n'installe pas
automatiquement une architecture Primer dans les modèles existants.

| Famille | Types |
| --- | --- |
| Fonctions élémentaires | `SquaredReLU`, `Square`, `SqrtAbs`, `Exp`, `LogAbs`, `Abs`, `ReciprocalNoNan`, `Sign`, `Sin`, `Cos` |
| Comparaisons | `Maximum`, `Minimum` |
| Réductions | `ReduceSum`, `ReduceMean`, `ReduceMin`, `ReduceMax`, `ReduceProd` |
| Masque de matrices | `LowerTriangular` |

| Backend | Forward / backward de ces 18 types | Limites et preuve locale |
| --- | --- | --- |
| CPU | Natif | Référence FP32, domaines et formes contrôlés, tests de gradients et checkpoints. |
| Vulkan | Natif via `PortableRuntime` | Exécuté sur AMD Radeon Graphics/RADV RENOIR le 8 novembre 2026 ; les tests contrôlent le runtime sélectionné. |
| OpenCL | Natif via `PortableRuntime` | Kernels générés et syntaxe OpenCL 1.2 validée ; exécution matérielle de ces 18 types non vérifiée lors de cet ajout. |
| CUDA / ROCm | `HostFallback` | Le routeur peut sélectionner un autre backend natif disponible, puis le CPU. Aucun calcul de ces types n'est attribué à un kernel CUDA/HIP natif. |

Les calculs CPU sont dans `cpu/PrimerKernels.hpp`. Les algorithmes device sont
définis dans `cuda/CudaPrimerOps.cuh`, puis traduits en OpenCL/SPIR-V par
`scripts/dev/generate_portable_reference.py`. `portable/PortablePrimerOps.hpp`
prépare les appels et publie les résultats seulement après validation complète.
Les kernels attribuent chaque sortie ou gradient d'entrée à une invocation,
sans atomique flottante. Le backward de `ReduceProd` recalcule le produit des
autres facteurs sans division par zéro ; son travail quadratique dans la
longueur de l'axe est borné avant lancement par les politiques portables.

Le chemin portable refuse les entrées et gradients subnormaux FP32, car leur
préservation n'est pas garantie par tous les pilotes. Un zéro device pouvant
masquer un résultat non nul est également signalé et refusé avant publication.
Le routeur réessaie alors sur CPU ; cette protection conservatrice peut aussi
concerner une annulation exacte. Les résultats non finis ou les domaines
invalides restent des échecs si aucun backend ne peut produire un résultat valide.

`CausalDepthwiseConv1d` et `SwiGLU` suivent un chemin natif CPU et portable
distinct, documenté dans [les opérations de séquence causale](14-Layers-And-Ops.md#opérations-de-séquence-causale).
Ces deux types et les 18 primitives Primer ne sont pas ajoutés à la liste
historique des activations admises par la fusion pointwise ci-dessous.

### Vérification ciblée des primitives Primer

Les cibles et tests sont déclarés dans `Tests/CMakeLists.txt`. Les recettes
suivantes supposent un build courant avec `mimir` et `RuntimeTest.PrimerOps` ;
elles ne reconstruisent pas le binaire :

```bash
env MIMIR_DISABLE_CUDA=1 MIMIR_DISABLE_ROCM=1 MIMIR_DISABLE_VULKAN=1 MIMIR_DISABLE_OPENCL=1 \
  ctest --test-dir build --output-on-failure \
  -R '^(RuntimeTest|ModelTest|LuaTest)\.PrimerOps$'
ctest --test-dir build --output-on-failure -R '^RuntimeTest\.VulkanPrimerOps$'
```

`RuntimeTest.VulkanPrimerOps` retourne `77` lorsque Vulkan est absent ou
indisponible ; CTest le marque alors ignoré, ce qui ne constitue pas une preuve
GPU. Les cas ordinaires exigent l'exécution Vulkan native. Les cas numériques
protégés vérifient le refus sans publication partielle et le repli CPU explicite.
Les tests utilisent des oracles indépendants, des différences finies et des
comparaisons adaptées au FP32, y compris pour les très petites valeurs.

Lors de l'intégration du 8 novembre 2026, les 11 tests CPU ciblés et les trois
tests Vulkan `VULKANNativeOps`, `VulkanCausalSequence` et `VulkanPrimerOps` passent.
Le scénario `scripts/tests/test_primer_ops.lua` passe aussi via `run_mimir.sh`,
avec backward, mise à jour des poids et rechargement du checkpoint. Ces résultats
ne valident ni les performances, ni une exécution OpenCL/CUDA/ROCm de ces types.


## Benchmarks Colab et validation du 18 octobre 2026

Les 165 cas de parité native par backend ont été revérifiés sur AMD Radeon
Graphics avec Vulkan/RADV et OpenCL/Rusticl. Le contrôle des budgets portables
passe aussi. Le profil Colab `portable` valide sept cas par runtime, avec
sélection explicite de `VULKAN` ou `OPENCL`, sans repli CPU attribué au GPU.
Cette validation locale ne prouve pas la disponibilité du pilote NVIDIA dans Colab.

Le microbenchmark peut mesurer plusieurs runtimes Mímir dans le même run, chacun
dans son processus et sous son identité. Le profil portable partage les mêmes
formes entre tous les concurrents ; les budgets de sécurité restent actifs.
Voir [Benchmarks Colab](../05-Advanced/07-Colab-Benchmarks.md) pour les profils,
la provenance, le RSS et le dépannage des anciennes cellules.


## Fusion native des chaînes pointwise

Les quatre backends implémentent les hooks `queryPointwiseFusionCapability()`,
`pointwiseFusionForward()` et `pointwiseFusionBackward()` d’`AbstractRuntime`.
La fusion porte sur `Identity`, `ReLU`, `LeakyReLU`, `GELU`, `SiLU`, `Tanh`,
`Sigmoid`, `Softplus`, `Mish`, `HardSigmoid` et `HardSwish`, dans des chaînes
unaires sans poids d’au moins deux étapes. Un backend non initialisé, désactivé
ou sans compilation native appropriée refuse la capacité.

`Mimir::GpuFusion::Program` prépare une arène bornée contenant les données,
les opcodes/paramètres, les snapshots backward et les sorties. La chaîne entière
est traitée dans une invocation par élément, avec un seul lancement de kernel
par direction. `retain=true` conserve les sorties de toutes les étapes ; sinon
seule la sortie finale est restituée. Les transferts nécessaires au résultat
hôte restent présents : ce mécanisme ne rend pas automatiquement tout le modèle
résident sur le GPU.

| Backend | Source et soumission |
| --- | --- |
| Vulkan | `PointwiseFusionKernels.cuh` traduit en SPIR-V ; `PortableExecutor` soumet un seul dispatch. |
| OpenCL | Même source traduite en OpenCL ; `PortableExecutor` soumet un seul kernel. |
| CUDA | Source partagée compilée avec CUDA ; lancement natif et buffers de scratch réutilisés. |
| ROCm | Source partagée compilée avec HIP ; lancement natif et buffers de scratch réutilisés. |

Les flags `linear_enabled` et `linear_min_ops` du backend s’appliquent aussi à
ces programmes, avec le nombre d’éléments comme travail pour ce contrôle. La
calibration reste nécessaire : elle impose la parité exacte avec la chaîne
ordinaire et un gain mesuré supérieur à la marge de sélection. Les 121 couples
d’activations disponibles ne seront donc pas tous systématiquement sélectionnés
sur GPU. Voir [le planner](22-Planning.md#sélection-dune-fusion).

### Compilation CUDA et HIP

CUDA nécessite un toolkit avec compilateur CUDA ; détecter uniquement les
bibliothèques ne suffit pas. CMake compile `CudaRuntime.cpp` avec le langage
CUDA lorsque ce backend est disponible.

Pour ROCm, `ENABLE_ROCM=ON` recherche HIP et rocBLAS. Avec CMake 3.21 ou plus et
un compilateur HIP détecté, `RocmRuntime.cpp` est compilé en langage HIP, en
C++17. Sans ce compilateur, la fusion native ROCm n’est pas annoncée comme
supportée. `CMAKE_HIP_ARCHITECTURES` permet de cibler l’architecture de compilation
sans constituer une preuve d’exécution sur ce matériel.

### Validation de la fusion au 28 octobre 2026

| Backend | Preuve locale dans ce checkout |
| --- | --- |
| CPU | 121 couples, chaînes de longueur 1 à 7, parité bit à bit des sorties et gradients, branches VAEConv et tests du cache. |
| Vulkan | Fusion exécutée sur AMD Radeon Graphics via RADV RENOIR ; 121 couples, forward/backward, rétention/final seul et tailles irrégulières. |
| OpenCL | Même couverture exécutée sur AMD Radeon Graphics via Rusticl/radeonsi. |
| ROCm | Runtime compilé en HIP, y compris via CMake pour `gfx90c` ; aucune exécution ROCm pour cette validation. |
| CUDA | Implémentation présente ; compilation et exécution non vérifiées ici, `nvcc` absent. |

Les tests directs GPU comparent leurs résultats au CPU avec une tolérance
numérique ; la sélection d’une fusion par le planner conserve son exigence
bit à bit avec la chaîne routée. Les tests de routage vérifient que le candidat
GPU est évalué, mais autorisent le CPU ou la chaîne non fusionnée à gagner.
Sur les cas de calibration locaux, ces chemins ordinaires/CPU ont été retenus.
Aucun gain de vitesse GPU global n’est déduit de ces validations.

Recette pour un nouveau dossier de build (à adapter ; cette forme de commande
n’a pas été rejouée sous le nom `build-gpu-tests` lors de la validation locale) :

```bash
cmake -S . -B build-gpu-tests -DENABLE_OPENCL=ON -DENABLE_VULKAN=ON -DMIMIR_ENABLE_TESTS=ON -DBUILD_TESTING=ON
cmake --build build-gpu-tests --target RuntimeTest.GpuPointwiseFusion -j4
ctest --test-dir build-gpu-tests --output-on-failure -R '^RuntimeTest.(VULKAN|OPENCL)PointwiseFusion$'
```

`RuntimeTest.CUDAPointwiseFusion` et `RuntimeTest.ROCMPointwiseFusion` utilisent
le même exécutable avec le backend demandé. Un backend non compilé ou un device
indisponible donne le code de skip `77` ; ce résultat ne vaut pas validation
matérielle. Les tests `RuntimeTest.GpuFusionPacking`,
`PlannerTest.FusionCalibration` et `PlannerTest.ExecutionPolicy` couvrent les
budgets, les décisions, l’isolation, l’invalidation et les choix CPU.
