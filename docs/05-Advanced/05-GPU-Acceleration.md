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
- [Primitives Primer et séquences causales](#primitives-primer-et-séquences-causales)
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

Le **CPU** reste la référence numérique et le repli des opérations refusées par
les autres runtimes. Les backends GPU compilés et initialisés peuvent être
utilisés automatiquement selon la configuration, les formes et le plan
d'exécution. La présence d'un GPU ne signifie donc pas que chaque layer y passe.

Ce guide explique le routage vers les kernels natifs CUDA, ROCm, Vulkan et
OpenCL, ainsi que vers cuBLAS/rocBLAS pour les opérations prises en charge.
Chaque backend publie des capacités forward et backward propres ; les autres
layers peuvent être exécutés par un autre runtime disponible.

> **Conseil :** si vous voulez juste démarrer rapidement, allez directement à la section [Recettes par modèle](#recettes-par-modèle).

---

## Comment ça marche

Mímir maintient une **pile de runtimes** ordonnée par priorité :

```text
1. ROCm Runtime
2. CUDA Runtime
3. Vulkan Runtime
4. OpenCL Runtime
5. TPU Runtime (lorsqu'il est fourni au routeur)
6. CPU Runtime (repli lorsqu'il est activé)
```

Pour chaque layer, le routeur examine les candidats et leurs capacités :

1. Le backend est-il compilé, initialisé et activé ?
2. Le forward ou backward demandé possède-t-il une implémentation native ?
3. La forme, le domaine numérique, les flags et les limites de travail sont-ils acceptés ?

Le planner et la calibration peuvent classer les candidats selon leurs coûts.
Un seuil configuré peut exclure une opération, mais sa valeur par défaut est
zéro : il ne remplace pas une mesure de vitesse. Si un candidat refuse le
calcul, le routeur propose l'opération à un autre backend, puis au CPU.
Les traces et les tests contrôlant `selected_runtime` permettent d'observer ce repli.

Point clé : l'objectif n'est pas de forcer "100% GPU", mais d'obtenir le meilleur compromis perf/stabilité par type de layer.

**Exemple :**

```text
[Layer Linear 4096×4096] → CUDA actif ? Seuil atteint ? → oui → cuBLAS SGEMM ✓
[Layer GroupNorm]         → CUDA actif ? Forme acceptée ? → oui → kernel CUDA ✓
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

- Un périphérique et une pile ROCm compatibles entre eux ; la génération
  commerciale du GPU seule ne garantit pas cette compatibilité
- Un toolkit HIP et rocBLAS détectables par CMake (`hipcc`, `libhip`, `librocblas`)
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

Chemins spécialisés Vulkan (forward), complétés par les kernels de référence
pour les autres types et le backward :

- `Linear`
- `MatMul`
- `BatchMatMul`
- `Conv2d`
- `ConvTranspose2d`
- `Add`
- `Multiply`
- `ReLU`

Les autres layers peuvent passer par les kernels portables Vulkan, y compris
en backward. Un refus permet au routeur de sélectionner un autre candidat.

- Vulkan SDK installé, `glslangValidator` disponible
- Compilé avec `-DENABLE_VULKAN=ON`

---

## Activation pas à pas

Les fast-paths matériels sont **activés par défaut** quand le backend est compilé et le device détecté. Vous pouvez toujours affiner via variables d'environnement ou forcer la désactivation avec `MIMIR_DISABLE_*`.

### Activer tous les fast-paths sur CUDA

```bash
export MIMIR_CUDA_LINEAR=1     # Linear → cuBLAS SGEMM
export MIMIR_CUDA_CONV=1       # Conv2d → im2col + SGEMM
export MIMIR_CUDA_NORM=1       # LayerNorm/RMSNorm → kernels CUDA
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

Les familles de fast-paths possèdent des minima de travail configurables,
exprimés en opérations ou en éléments selon le champ. Leur valeur par défaut
est zéro. Augmenter un minimum exclut les petits calculs de cette famille ;
la calibration et les mesures du run déterminent si cela améliore la vitesse.

| Fast-path | Variable de seuil | Valeur par défaut |
| --- | --- | --- |
| `Linear` | `MIMIR_CUDA_LINEAR_MIN_OPS` | `0` |
| `Conv2d` | `MIMIR_CUDA_CONV_MIN_OPS` | `0` |
| `LayerNorm` / `RMSNorm` | `MIMIR_CUDA_NORM_MIN_ELEMS` | `0` |
| `Attention` | `MIMIR_CUDA_ATTENTION_MIN_OPS` | `0` |

Pour les variables ROCm, remplacez `CUDA` par `ROCM`.

Exemple de réglage manuel, à comparer aux valeurs par défaut :

```bash
export MIMIR_CUDA_LINEAR_MIN_OPS=65536
export MIMIR_CUDA_CONV_MIN_OPS=16384
```

Les transferts et les soumissions GPU peuvent coûter plus que le calcul sur
un petit tenseur. Comparez le temps global, forward, backward et optimiseur,
avec les mêmes données et la même configuration.

Méthode simple de tuning :

1. Commencer avec les seuils par défaut.
2. Modifier un seul seuil à la fois.
3. Rejouer le même script (mêmes entrées, même seed).
4. Garder la valeur seulement si le temps global baisse de manière stable.

---

## Matrice de couverture runtime (etat actuel)

La couverture suivante décrit les implémentations du checkout courant. Les
capacités publiées par le binaire dépendent des backends compilés. Le dispatcher
vérifie ensuite configuration, formes et disponibilité du device.

### Couverture actuelle par type

| Backend | Forward | Backward | Particularités |
| --- | --- | --- | --- |
| CPU | 94 types exécutables | 95 types | Registre de 95 types ; `Lambda` forward refusé, backward identité ; NMS stop-gradient. |
| CUDA | Familles historiques natives si compilées | Capacités propres à chaque type | Les 18 primitives Primer, `CausalDepthwiseConv1d` et `SwiGLU` restent en `HostFallback` ; `Lambda` indisponible. |
| OpenCL | 94 types déclarés natifs | 95 types déclarés natifs | Chemins spécialisés et kernels portables ; formes, domaines et budgets contrôlés. |
| Vulkan | 94 types déclarés natifs | 95 types déclarés natifs | Même contrat portable ; la capacité par type n'est pas une preuve d'exécution. |
| ROCm | Couverture partielle | Couverture partielle | BLAS et convolutions ND selon compilation ; autres familles en repli selon la capacité publiée. |

Les backends Vulkan et OpenCL conservent leurs chemins spécialisés et utilisent
`PortableRuntime` pour les autres opérations et les passes arrière. Leur
couverture déclarée par type correspond au dispatcher CPU exécutable : **94 types en
forward et 95 en backward**, dont NMS avec stop-gradient. `Lambda` est refusé
en forward par le dispatcher CPU lui-même ; son backward identité est supporté
sur CPU, Vulkan et OpenCL. Ces nombres décrivent le registre courant avec le
backend activé ; ils ne garantissent ni l'acceptation de toute forme, ni
l'exécution sur le matériel installé. CUDA et ROCm ont une couverture native distincte.

Les familles couvertes comprennent convolutions, normalisations, attention,
projections/embeddings, pooling, transformations de tenseurs, récurrence,
upsampling, dropout, reparamétrisation, convolutions ND, séquences causales
et primitives Primer. Les configurations, formes de tenseurs
et limites du device restent vérifiées ; cette couverture par type ne garantit
pas que toute forme soit acceptée.

Les kernels OpenCL/Vulkan à sorties indépendantes utilisent une grille
parallèle ; certaines réductions utilisent des groupes coopératifs. Les chemins
séquentiels restants sont bornés. Cette couverture ne garantit pas des
performances supérieures au CPU. Le staging des tenseurs et l'accumulation des
gradients de paramètres retournés restent sur l'hôte. Les calculs forward et
backward ne sont pas remplacés par un appel arithmétique CPU caché.

NMS renvoie des indices discrets en forward. Son backward stop-gradient
produit des zéros pour les boîtes, scores et classes ; placez les losses de
détection avant NMS. Ce contrat est commun à CPU, CUDA, OpenCL et Vulkan.

Pour consulter les déclarations du binaire :

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- --runtime
```

`Native` signifie un calcul natif ; `NativeOptimized` identifie un chemin
optimisé. `HostFallback` et `Unsupported` restent distincts. Les tests
`RuntimeTest.OPENCLNativeOps` et `RuntimeTest.VULKANNativeOps` comparent les
résultats au CPU et contrôlent le backend effectivement exécuté.

Voir [les kernels et validations](../04-Architecture-Internals/21-GPU-Runtimes.md#opencl-et-vulkan-kernels-de-référence-partagés).

## Primitives Primer et séquences causales

Les opérations ajoutées au vocabulaire du framework comprennent :

- `SquaredReLU`, `Square`, `SqrtAbs`, `Exp`, `LogAbs`, `Abs`, `ReciprocalNoNan`, `Sign`, `Sin`, `Cos` ;
- `Maximum`, `Minimum` avec deux entrées de même taille ou une entrée scalaire ;
- `ReduceSum`, `ReduceMean`, `ReduceMin`, `ReduceMax`, `ReduceProd` sur un axe ;
- `LowerTriangular`, masque multiplicatif des deux derniers axes.

Ces 18 types possèdent un forward et un backward CPU et Vulkan/OpenCL. Les
[contrats et l'exemple de graphe C++](../04-Architecture-Internals/14-Layers-And-Ops.md#primitives-issues-de-létude-primer)
précisent les domaines, les gradients aux égalités et la configuration `shape`/`axis`.
Le masque triangulaire met les éléments supérieurs à zéro ; il ne remplace pas
le masque des logits d'une attention causale avant softmax.

Les primitives Primer utilisent le routeur habituel. Les formes trop coûteuses,
les entrées ou gradients subnormaux FP32 et les zéros device ambigus peuvent
déclencher un repli CPU explicite. Les résultats sont validés avant publication,
sans remplacer le calcul natif déclaré par une référence CPU cachée. Les autres
erreurs de domaine ou résultats non finis ne sont pas corrigés par un epsilon
implicite.

`CausalDepthwiseConv1d` et `SwiGLU` sont aussi disponibles sur CPU et via le
runtime portable. CUDA/ROCm déclarent `HostFallback` pour ces deux types et les
18 primitives ci-dessus ; le routeur peut choisir Vulkan/OpenCL s'ils sont
disponibles, puis le CPU. Les types ajoutés ne font pas partie du programme
historique de fusion pointwise.

La validation du 8 novembre 2026 comprend des tests de sorties et de gradients
sur CPU et Vulkan/RADV RENOIR sur GPU AMD, avec contrôle du backend sélectionné.
La syntaxe des kernels OpenCL a été validée ; leur exécution matérielle pour
ces nouvelles primitives reste à vérifier. Voir les
[recettes CTest ciblées et les limites](../04-Architecture-Internals/21-GPU-Runtimes.md#vérification-ciblée-des-primitives-primer).

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

Les journaux de configuration et de calibration exposent les candidats et
leurs décisions. Ils ne suffisent pas à attribuer chaque opération au GPU.

Avec `MIMIR_RUNTIME_TRACE=1`, vous obtenez en plus la trace d'execution reelle layer-par-layer:

Lire notamment les champs `backend`, `call`, `type` et `output_size` de
`[runtime-trace]`. La valeur de `call` dépend du chemin direct, planifié ou
fusionné réellement exécuté ; une ancienne trace illustrant un autre build
ne démontre pas la couverture du run courant.

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

L'ordre des réductions et la précision des fonctions élémentaires peuvent
différer selon le backend. Comparez les sorties et gradients avec les tolérances
propres à l'opération, à leur échelle et aux très petites valeurs ; un seuil
absolu unique ne convient pas à tous les cas. Activez les traces pour vérifier
le backend exécuté, puis utilisez les tests ciblés de parité et de domaine.

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
