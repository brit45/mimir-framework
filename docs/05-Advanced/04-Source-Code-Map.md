# Carte du code source (C/C++)

Optimiser, diagnostiquer et stabiliser des runs complexes.

**Public concerné :** Utilisateur intermédiaire à avancé.

> **Prérequis**
>
> Avoir déjà exécuté au moins un pipeline complet.


Cette page sert de « table des matières technique » : comment le framework est structuré côté C/C++, fichier par fichier.

Pages Advanced associées (explications orientées usage) :

- [01-Performance.md](01-Performance.md)
- [02-Debugging.md](02-Debugging.md)
- [03-LLM-Readiness.md](03-LLM-Readiness.md)

Objectif : quand une fonctionnalité est mentionnée dans la doc (mémoire stricte, forward tokens, multi-input, sérialisation…), vous devez pouvoir retrouver rapidement **le fichier source qui fait foi**, et comprendre le rôle exact de chaque composant.

> Notes
> - La plupart des features sont exposées via l’API Lua (bindings dans `src/scriptings/Lua/luaScripting/LuaScripting.cpp`).
> - Les modèles « prêts à l’emploi » sont construits via le registre d’architectures (`src/Models/Registry/ModelArchitectures.*`).

## Sur cette page

- [1) Point d’entrée et exposition API](#1-point-dentrée-et-exposition-api)
- [2) Noyau du runtime : Model + exécution](#2-noyau-du-runtime-model-exécution)
- [3) Tenseurs : stockage et allocations](#3-tenseurs-stockage-et-allocations)
- [4) Tokenizer et encodage](#4-tokenizer-et-encodage)
- [5) Registry d’architectures et modèles built-in](#5-registry-darchitectures-et-modèles-built-in)
- [6) Sérialisation / debug](#6-sérialisation-debug)
- [7) Hardware, SIMD et backends compute](#7-hardware-simd-et-backends-compute)
- [8) Monitoring / visualisation](#8-monitoring-visualisation)
- [9) Utilitaires](#9-utilitaires)
- [Lecture guidée (par tâche)](#lecture-guidée-par-tâche)
- [Étapes suivantes](#étapes-suivantes)

## 1) Point d’entrée et exposition API

### `src/main.cpp`

- Rôle : point d’entrée du binaire (init, chargement, exécution Lua / modes CLI selon build).
- À lire quand : vous voulez comprendre comment le runtime démarre, et à quel moment les sous-systèmes (Lua, compute backends, etc.) sont initialisés.

### `src/scriptings/Lua/luaScripting/LuaScripting.hpp` / `src/scriptings/Lua/luaScripting/LuaScripting.cpp`

- Rôle : **API publique** (bindings Lua) sous la table globale `Mimir`.
- Contient :
  - `registerAPI()` : enregistre `Mimir.Model`, `Mimir.Architectures`, `Mimir.Tokenizer`, `Mimir.MemoryGuard`, `Mimir.Allocator`, `Mimir.Serialization`, `Mimir.Htop`, `Mimir.Viz`, etc.
  - Forward : `Mimir.Model.forward(input, training)` route vers :
    - `Model::forwardPass(input_ids)` si `input` est une table d’entiers
    - `Model::forwardPass(input)` si `input` est une table de floats
    - `Model::forwardPassNamed(float_inputs, int_inputs)` si `input` est une map `{name -> table}` (multi-input mixte)
  - Mémoire :
    - `Mimir.MemoryGuard.setLimit(...)`, `getCurrentUsage()`, `getPeakUsage()`, `getLimit()`, `printStats()`, etc.
    - `Mimir.Allocator.configure({max_ram_gb, enable_compression})` configure le `DynamicTensorAllocator`
  - Debug JSON : `Mimir.Serialization.save_enhanced_debug(path, options)` (v1.3 inclut `framework_state`) (voir aussi `src/Serialization/DebugJsonDump.*`).

À lire quand :

- vous cherchez “comment un appel Lua devient un appel C++”
- vous voulez des signatures exactes, erreurs retournées, comportements de conversion table->vector, etc.

## 2) Noyau du runtime : `Model` + exécution

### `src/Model.hpp` / `src/Model.cpp`

- Rôle : cœur du framework.
- Responsabilités :
  - stocker la liste des layers (`std::vector<Layer> layers`)
  - construire/exécuter le forward (float, tokens int, ou named inputs)
  - gérer l’entraînement (loss, backward, optimizer)
  - orchestrer la mémoire runtime via `MemoryGuard` + allocateurs
  - déléguer les opérations forward/backward au routeur des runtimes

Points à connaître :

- **Trois APIs forward** importantes :
  - `forwardPass(const std::vector<float>&, training)`
  - `forwardPass(const std::vector<int>&, training)` (chemin tokens : `Embedding` lit dans l’IntTensorStore)
  - `forwardPassNamed(float_inputs, int_inputs, training)` (multi-input)
- **Optimiseur** : `optimizerStep(Optimizer&, learning_rate)` gère SGD, Adam, AdamW, Lion, Adafactor, RAdam, NAdam, RMSprop et LAMB, avec protections (epsilon non-fini -> fallback), warmup/decay, et *grad clipping optionnel* via `modelConfig` (`grad_clip_norm` / `clip_norm`).
- **Accélération compute** : le routeur choisit un backend selon ses capacités,
  la configuration, les entrées réelles et le plan, en entraînement comme en inférence.
- **Capture 3D commune** : `captureLatentSpaceSnapshot()` copie des vecteurs du
  graphe; `bindLatentSpaceControl()` et `captureRequestedLatentSpace()` relient
  les demandes Viz aux points sûrs du modèle, après un pas optimiseur ou un
  forward d'inférence. Cette API appartient à `Model`, y compris pour les
  architectures sans dimension latente explicite.
- **Observations de corrélation** : `observeCompletedForwardCorrelations()`
  collecte une fois par forward réussi quand le mode et la vue sont actifs ;
  `captureLatentCorrelationObservation()` copie les composantes internes à ce
  point sûr, avant la mise à jour des poids.

À lire quand : vous voulez comprendre l’ordre d’exécution des layers, les invariants de shapes, la logique training vs inference, ou les routes “float vs token ids”.

### `src/Layers.hpp`

- Rôle : structure “Layer” unifiée, avec un grand nombre de paramètres optionnels.
- Contient :
  - la struct `Layer` (nom, type string + enum, `inputs`/`output` pour le routing)
  - les champs “universels” (dimensions, conv, norm, dropout, attention, reshape, split/concat, etc.)
  - le lien poids unifié : `tensor* weight_block` (poids du layer dans un bloc unique)

À lire quand : vous documentez un type de layer, ses champs, ou pourquoi une op lit tel paramètre (ex: `seq_len`, `embed_dim`, `num_heads`, `split_sizes`, …).

### `src/LayerTypes.hpp`

- Rôle : enum central `LayerType` + normalisation d’alias (rétrocompat).
- À lire quand : vous voulez la **liste canonique** des layers supportés par le runtime, et les alias acceptés.

### `src/runtimes/cpu/LayerOps.hpp` / `src/runtimes/cpu/LayerOpsExt.hpp`

- Rôle : implémentations CPU des opérations (forward/backward) utilisées par `Model`.
- `LayerOpsExt` regroupe souvent des implémentations plus spécialisées/étendues.
- À lire quand : vous cherchez “où est l’implémentation réelle de Conv/Linear/Norm/Activation/Attention”.

### Routage et primitives Primer

- `src/runtimes/RuntimeRouter.cpp` orchestre la sélection et les replis en
  forward/backward ; `src/runtimes/cpu/RuntimeLayerDispatch.hpp` relie les types
  aux noyaux CPU. Les calculs restent dans les runtimes.
- `src/runtimes/PrimerGeometry.hpp` classe les 18 types Primer et vérifie les
  formes des réductions et du masque triangulaire.
- `src/runtimes/cpu/PrimerKernels.hpp` implémente leurs valeurs et gradients,
  avec validation des domaines et publication après contrôle de finitude.
- `src/runtimes/cuda/CudaPrimerOps.cuh` est la source de génération portable,
  utilisée par `scripts/dev/generate_portable_reference.py` pour produire
  `PortablePrimerOps.hpp`, `reference.comp`, `reference.cl` et les shaders
  embarqués sous `src/runtimes/portable/`. Ce chemin de génération ne déclare
  pas de prise en charge native CUDA/ROCm pour ces types.
- `src/runtimes/portable/PortableParallelGeometry.hpp` borne le travail et le
  stockage ; `PortablePrimerValidation.hpp` vérifie les valeurs prises en charge.
- `src/Serialization/LayerMetadata.hpp` préserve formes, axes et connexions
  pour les graphes restaurés depuis RawFolder ou SafeTensors.
- Tests : `Tests/test_runtime_primer.cpp` couvre les runtimes et les
  checkpoints ; `scripts/tests/test_primer_ops.lua` exerce l’API de graphe,
  le backward et une mise à jour des poids.

Les [contrats des couches et opérations](../04-Architecture-Internals/14-Layers-And-Ops.md)
décrivent les domaines numériques, les gradients aux égalités et le repli CPU.

### `src/Autograd.hpp`

- Rôle : structures/utilitaires pour gradients (type `Gradients`, helpers).
- À lire quand : vous voulez comprendre la représentation des gradients, et comment ils sont manipulés.

## 3) Tenseurs : stockage et allocations

### `src/tensors.hpp` / `src/tensors.cpp`

- Rôle : type `tensor` (bloc float), et intégration optionnelle avec l’allocation dynamique.
- Point clé : un `tensor` peut être créé en mode “dynamic” et utiliser `DynamicTensorAllocator`.
- À lire quand : vous cherchez pourquoi certaines allocations passent (ou non) par le guard/allocator.

### `src/MemoryGuard.hpp`

- Rôle : garde-fou strict (comptabilisation courant/pic + limite) pour éviter OOM silencieux.
- Utilisé par : runtime, allocateurs, wrappers (ex: allocations stb_image routées).

### `src/runtimes/cpu/DynamicTensorAllocator.hpp`

- Rôle : allocateur dynamique global (singleton) utilisé par `tensor(dynamic=true)` et certains chemins runtime.
- Fonctionnalités : réservations, stats, (optionnel) compression/éviction selon config.

### `src/runtimes/cpu/RuntimeAllocator.hpp`

- Rôle : allocateur runtime RAII (handles auto-release) + pool scratchpad.
- Utilisé dans des chemins `Model.cpp` pour gérer buffers temporaires (ex: conv im2col/tiles, scratch buffers).

### `src/AdvancedRAMManager.hpp`

- Rôle : gestion RAM avancée / cache (configuration, statistiques, clear).
- Exposé via `Mimir.Memory.*` dans Lua (stats, limite en GB, etc.).

### `src/MemorySafety.hpp`

- Rôle : helpers/guardrails “safety” (assertions, checks, comportements stricts) selon build.

### `src/stb_image_impl.cpp` + `src/stb_image.h`

- Rôle : chargement image via stb_image.
- Particularité : **certaines allocations** sont routées via `MemoryGuard` pour éviter des pics incontrôlés.

## 4) Tokenizer et encodage

### `src/Tokenizer.hpp` / `src/Tokenizer.cpp`

- Rôle : tokenizer (dont BPE), vocab, encode/decode.
- Exposé via `Mimir.Tokenizer.*`.
- À lire quand : vous voulez comprendre `tokenize_bpe`, `decode`, ou les IDs spéciaux.

### `src/Encoder.hpp` / `src/Encoder.cpp` (ConditioningEncoder)

- Rôle : embeddings / encodeur “mag/mod” et embeddings tokens.
- Important : le `Model` maintient la compatibilité `Tokenizer` <-> `ConditioningEncoder` (ex: ensure vocab size).

## 5) Registry d’architectures et modèles built-in

### `src/Models/Registry/ModelArchitectures.hpp` / `src/Models/Registry/ModelArchitectures.cpp`

- Rôle : registre d’architectures (noms -> (default_config, factory)).
- Comportement important :
  - canonicalisation d’alias (`canonicalArchName`)
  - merge récursif `default_config` + overrides
  - impose `cfg["type"] = <nom canonique>`
  - écrit `model->modelConfig = cfg`

À lire quand : vous voulez la liste des archis et leurs configs par défaut, ou comprendre pourquoi `cfg.type` est standardisé.

### `src/Models/**` (implémentations de modèles)

Chaque modèle “builder” sait pousser une séquence de layers cohérente (noms, routage, paramètres) :

- NLP :
  - `src/Models/NLP/CausalLMModel.hpp` / `.cpp` : decoder-only causal (RoPE/GQA/SwiGLU/RMSNorm)
  - `src/Models/NLP/TransformerModel.hpp` / `.cpp` : builder Transformer (peut être causal)
  - `src/Models/NLP/VAETextModel.hpp` / `.cpp` : VAE sur texte (encode/decode)
- Vision :
  - `src/Models/Vision/ViTModel.*`, `VAEModel.*`, `VAEConvModel.*`, `UNetModel.*`
  - CNN : `ResNetModel.*`, `MobileNetModel.*`, `VGG16Model.*`, `VGG19Model.*`
- Diffusion :
  - `src/Models/Diffusion/DiffusionModel.*`, `CondDiffusionModel.*`, `SD35Model.*`

À lire quand : vous voulez voir comment une architecture “réaliste” se traduit en layers, et quelles conventions de noms sont utilisées.

## 6) Sérialisation / debug

### `src/Serialization/Serialization.hpp` / `src/Serialization/Serialization.cpp`

- Rôle : façade de sérialisation (sauvegarde/chargement) utilisée par l’API Lua `Mimir.Serialization.*`.

### `src/Serialization/SafeTensorsReader.*` et `src/Serialization/SafeTensorsWriter.*`

- Rôle : IO `safetensors`.
- Important : peut inclure des payloads extra (ex: tokenizer JSON) selon options.

### `src/Serialization/RawCheckpointReader.*` / `src/Serialization/RawCheckpointWriter.*`

- Rôle : format legacy/raw (compatibilité / debug).

### `src/Serialization/DebugJsonDump.hpp` / `.cpp`

- Rôle : export debug JSON “amélioré” (statistiques, gradients, checksums, infos git, tokenizer/encoder… selon options).
- Depuis v1.3 : ajoute `framework_state` (runtime, mémoire, registre layers, état modèle) au moment du dump.
- Exposé via `Mimir.Serialization.save_enhanced_debug(path, options)`.

## 7) Hardware, SIMD et backends compute

### `src/runtimes/cpu/HardwareOpt.hpp`

- Rôle : détection capacités CPU et helpers (AVX2/FMA/F16C/BMI2, etc.).
- Exposé via `Mimir.Model.hardware_caps()`.

### `src/runtimes/cpu/SIMD_Ops.hpp`

- Rôle : adaptateurs SIMD utilisés par des ops. La sélection effective des
  primitives multiversionnées est dans `src/runtimes/cpu/SimdDispatch.cpp`.
  `MIMIR_CPU_ISA` sélectionne une ISA disponible pour le CPU, l'OS et le build ;
  les capacités détectées ne garantissent pas un kernel dédié pour chaque op.

### `src/runtimes/vulkan/VulkanCompute.hpp` et `src/runtimes/opencl/OpenCLCompute.hpp`

- Rôle : engines compute GPU pour Vulkan/OpenCL (si compilés).
- Build flags : `ENABLE_VULKAN`, `ENABLE_OPENCL`.
- Variables d'environnement (Vulkan/OpenCL) : `MIMIR_VULKAN_LINEAR_SPV`, `MIMIR_VULKAN_LINEAR`, `MIMIR_OPENCL_LINEAR`.

Ces moteurs compute ne constituent pas l’inventaire complet des couches.
`src/runtimes/portable/PortableRuntime.cpp`, partagé par les runtimes Vulkan et
OpenCL, déclare les capacités par type et par opération et lance les kernels
générés, dont ceux des 18 types Primer. La disponibilité dépend des formes,
des valeurs, des limites de travail et du périphérique.

### `src/runtimes/AbstractRuntime.hpp`

- Rôle : interface commune `AbstractRuntime`, capacités par type/opération et
  structure `RuntimeConfig` (Linear, convolution, normalisation et attention).
- `RuntimeConfig::fromEnv(backend_upper)` : parse les variables `MIMIR_CUDA_*` / `MIMIR_ROCM_*`.

### `src/runtimes/cuda/CudaRuntime.hpp` / `.cpp`

- Rôle : BLAS pour Linear/MatMul et convolution, et kernels device pour
  convolutions, normes, attention, récurrence et autres familles natives.
  Les calculs Conv2d et leurs gradients restent sur device ; voir les gardes
  de `CudaOperationGeometry.hpp` pour les formes admises.
- Build flag : `ENABLE_CUDA`.
- Variables : `MIMIR_CUDA_LINEAR=1`, `MIMIR_CUDA_CONV=1`, `MIMIR_CUDA_NORM=1`, `MIMIR_CUDA_ATTENTION=1`.
- Internals : `src/runtimes/cuda/CudaRuntime.cpp` (Impl, DeviceBuf, fast-path par LayerType).

### `src/runtimes/rocm/RocmRuntime.hpp` / `.cpp`

- Rôle : chemins rocBLAS/HIP ; couverture distincte de CUDA, à vérifier dans
  les capacités de `RocmRuntime` et par des tests sur le matériel concerné.
- Build flag : `ENABLE_ROCM`.
- Variables : `MIMIR_ROCM_*` (mêmes suffixes que CUDA).

### `src/runtimes/cpu/CpuRuntime.hpp` / `.cpp`

- Rôle : référence et repli CPU via `RuntimeLayerDispatch::cpu_forward_layer`
  et `cpu_backward_layer`.
- Délègue à `LayerOps.hpp`, `LayerOpsExt.hpp` et aux noyaux spécialisés,
  dont `PrimerKernels.hpp`.

Documentation détaillée : [docs/04-Architecture-Internals/03-Hardware-Backends.md](../04-Architecture-Internals/03-Hardware-Backends.md), [docs/04-Architecture-Internals/21-GPU-Runtimes.md](../04-Architecture-Internals/21-GPU-Runtimes.md), [docs/05-Advanced/05-GPU-Acceleration.md](05-GPU-Acceleration.md).

### `src/runtimes/fpga/FpgaSession.cpp` et `FpgaIo.cpp`

- Rôle : session matérielle, buffering de dataset et conversions/contrôle
  d'intégrité des checkpoints selon les capacités du firmware.
- Ces sources et `FpgaRuntime.cpp` sont intégrées dans `MIMIR_SOURCES`.
  Le service FPGA n'est pas une route de layers : les votes forward/backward
  restent négatifs. Voir [l'état d'intégration FPGA](06-FPGA-iCESugar-Pro-Runtime-Study.md#état-dintégration-du-checkout-actuel).

### `src/Planning/Planner.hpp`

- Rôle : analyse statique du graphe — durées de vie des tenseurs, fusions Conv2d+ReLU, pré-calcul des scratchpads.
- Au lancement : le plan est construit au premier forward (ou reconstruit si invalidé), puis réutilisé tant que le graphe et le mode training restent compatibles.
- Documentation : [docs/04-Architecture-Internals/22-Planning.md](../04-Architecture-Internals/22-Planning.md).

## 8) Monitoring / visualisation

### `src/AsyncMonitor.hpp`

- Rôle : collecte asynchrone métriques + frames (si activé).
- Projection 3D : `configureLatentSpace()` utilise la configuration effective de
  toute architecture native; le worker `latentSpaceWorkerLoop()` consomme les
  snapshots copiés, vérifie leur génération/révision, calcule l'ACP puis publie
  un résultat immuable. Il ne lit pas un `Model` actif.
- En mode corrélations, le même worker possède la fenêtre d’observations,
  calcule Pearson puis projette les profils de composantes. La file d’attente
  est bornée et les résultats d’une session ou d’un schéma dépassé sont rejetés.

### `src/HtopDisplay.hpp`

- Rôle : affichage/collecte style htop (CPU/RAM/loss/bps…).
- Exposé via `Mimir.Htop.*`.

### `src/Visualizer.hpp` / `src/Visualizer.cpp`

- Rôle : scène graphique commune (taps d'images par bloc/layer, métriques et
  projection 3D), avec le backend choisi à la compilation.
- Exposé via `Mimir.Viz.*`.
- `renderLatentSpaceOverlay()` rend le nom du modèle courant, les groupes et
  dimensions réellement présents, la rotation, le zoom et le survol. Les
  marqueurs et leurs zones de survol restent dans le canvas, même après zoom.

### `src/viz/LatentSpaceProjection.hpp`

- [Source](../../src/viz/LatentSpaceProjection.hpp) : structures
  `LatentVectorSample`, `LatentSpaceRawSnapshot`, `LatentSpaceProjection` et
  contrôle partagé `LatentSpaceControl`.
- `projectLatentSpace3D()` normalise en L2, complète les vecteurs courts par zéro
  puis centre les échantillons pour l'ACP commune. L'échelle globale de rendu
  préserve les rapports entre les trois axes; elle borne le rayon à 2,5.
- Un ou deux vecteurs et les poids nuls sont affichables. Les métadonnées
  `model_name`, `architecture`, `feature_dimension`, `original_dimension` et
  `feature_offset` décrivent la capture; `d_model` et `latent_dim` restent
  facultatifs.
- `resetSession()` invalide le modèle précédent. La génération protège aussi
  les transitions d'état du worker; la révision rejette une ACP dépassée par une
  capture plus récente du même modèle. Les snapshots publiés sont constants et
  possèdent leurs copies de valeurs.

### `src/viz/LatentCorrelation.hpp` / `LatentCorrelationCapture.hpp`

- [LatentCorrelation.hpp](../../src/viz/LatentCorrelation.hpp) définit les
  observations, la fenêtre bornée à 128 observations × 128 composantes, les
  profils centrés normalisés et les liens de Pearson en double précision.
  Les composantes constantes sont exclues. `edges` conserve les 512 liens les
  plus forts satisfaisant `|r| ≥ 0,30` ; `candidate_edges` conserve tous les
  candidats valides de la fenêtre (au plus 8 128). La case **Tous les candidats
  valides** choisit cet ensemble pour le rendu, sans supprimer les budgets de
  capture ni transformer une corrélation en preuve de compréhension.
- [LatentCorrelationCapture.hpp](../../src/viz/LatentCorrelationCapture.hpp)
  sélectionne et regroupe les activations réelles du graphe principal, avec
  prise en compte des masques de séquence et des géométries spatiales connues.
  La capture ne relit pas de corpus et ne modifie pas les tenseurs du modèle.
- `LatentSpaceControl` pilote l’activation, l’effacement et les epochs de cette
  fenêtre ; `projectLatentCorrelations3D()` adapte ses profils à l’ACP commune.

### `src/viz/NeuronNetwork.hpp`

- [NeuronNetwork.hpp](../../src/viz/NeuronNetwork.hpp) décrit les couches,
  unités et liens de routage capturés par `Model::captureNeuronNetworkSnapshot()`.
  `layoutNeuronNetwork3D()` dispose les couches selon leur profondeur dans le
  graphe et répartit les unités/canaux en grilles ; les largeurs inconnues
  restent des nœuds d’opération. Les liens sont des flux structurels agrégés.
- `projectNeuronNetwork3D()` adapte cette disposition au rendu commun sans
  appliquer d’ACP. `LatentSpaceControl::setNeuronsEnabled()` rend ce mode
  exclusif des corrélations et invalide les résultats du mode précédent.

### `src/viz/LiveWeightNetwork.hpp`

- [LiveWeightNetwork.hpp](../../src/viz/LiveWeightNetwork.hpp) contient les
  coefficients signés et leurs identités stables. La capture native
  `Model::captureLiveWeightNetworkSnapshot()` copie au plus 2 048 coefficients
  résidents, sans charger les blocs froids. Elle conserve le routage neuronal
  seulement lorsque le stockage du coefficient permet de le vérifier.
- `projectLiveWeightNetwork3D()` garde la disposition du graphe et calcule les
  variations entre deux captures de même session et de même schéma. Le worker
  d’`AsyncMonitor` conserve la dernière capture publiée comme référence ; le
  thread graphique reçoit uniquement des données immuables.
- Les [tests de capture](../../Tests/test_live_weight_network.cpp) vérifient
  valeurs, indices, groupes, biais, partages et budgets. Le
  [test du contrôle](../../Tests/test_live_weight_control.cpp) vérifie les
  changements de mode, les variations et deux pas d’entraînement natifs,
  comparés à un modèle identique exécuté sans observation.

### `src/Model.cpp` — capture des vecteurs

- [API déclarée dans Model.hpp](../../src/Model.hpp) :
  `Model::captureLatentSpaceSnapshot(optimizer_step, max_token_vectors,
  max_sequence_positions, max_vectors_per_matrix) const`.
- La capture lit les poids hôte déjà résidents et les activations encore
  disponibles du graphe principal, sans modifier les paramètres ni recharger
  un bloc dynamique froid/compressé. Les poids partagés sont dédupliqués;
  les paramètres fixes et les activations d'un graphe sans poids sont couverts.
- Les copies sont bornées à 512 points, 256 K valeurs et 2 048 composantes par
  vecteur. Une fenêtre d'un vecteur large porte sa plage et sa dimension
  d'origine. Les types inconnus peuvent être représentés par leurs valeurs
  aplaties réelles, sans inventer une géométrie de couche.
- Le parcours ne descend pas dans les sous-modèles privés : les VAE encodeur et
  décodeur de Lumen détenus hors du graphe principal ne sont pas inclus.

### `src/scriptings/ModelVizProjection.hpp`

- [Source](../../src/scriptings/ModelVizProjection.hpp) : helper commun
  `Mimir::Scripting::bindModelProjection()` et garde RAII
  `ProjectionTrainingScope`.
- Lie le modèle courant au contrôle d'`AsyncMonitor`, démarre une nouvelle
  session lors d'un remplacement et capture depuis le thread des scripts.
  Le processus principal DDP réalise la publication.
- Les bindings Lua de modèle/Viz l'utilisent pour la création, le chargement,
  l'initialisation et l'entraînement. `Mimir.Viz.update()` peut servir une demande
  hors entraînement; les chemins Lua de `src/main.cpp` forcent l'état courant avant d'attendre la
  fermeture d'une Viz encore ouverte.

Cette projection est une vue structurelle des vecteurs échantillonnés, pas une
mesure de proximité sémantique entre couches. Pour le pipeline complet, les
invariants et un exemple C++ natif, lire
[Monitoring : projection 3D](../04-Architecture-Internals/04-Monitoring-Htop-Visualizer.md#projection-3d-du-modèle-courant).
Pour l'utilisation du bouton, lire
[l'API Viz/Htop](../03-API-Reference/15-Viz-Htop.md#projection-3d-du-modèle-courant).

### Tests de la projection générique

- [Tests/test_latent_space_generic.cpp](../../Tests/test_latent_space_generic.cpp)
  (`VizTest.GenericLatentSpace`) : registre, graphes personnalisés, dimensions
  spatiales, poids partagés/fixes, limites de copies et absence de mutation.
- [Tests/test_latent_space_control.cpp](../../Tests/test_latent_space_control.cpp)
  (`VizTest.GenericLatentControl`) : publication, points sûrs, petits modèles,
  dimensions variables et courses entre générations/révisions.
- [Tests/test_viz_events.cpp](../../Tests/test_viz_events.cpp)
  (`VizTest.GenericProjection`, mode `--projection-only`) : interaction réelle
  du bouton et de l'overlay, rotation, zoom, survol, seuils des liens et bornes du canvas.
- [Tests/test_latent_correlations.cpp](../../Tests/test_latent_correlations.cpp)
  (`VizTest.LatentCorrelationMath`) : oracles Pearson, constantes, finitude,
  fenêtres, schémas et sélection déterministe des liens.
- [Tests/test_latent_correlation_control.cpp](../../Tests/test_latent_correlation_control.cpp)
  (`VizTest.LatentCorrelationControl`) : worker, publication, actualisation
  manuelle/automatique, sessions et refus des observations périmées.
- [Tests/test_model_latent_correlations.cpp](../../Tests/test_model_latent_correlations.cpp)
  (`ModelTest.LatentCorrelations`) : capture native, padding, géométrie spatiale,
  limites et absence de mutation des poids/gradients.

L'enregistrement des cibles est dans
[Tests/CMakeLists.txt](../../Tests/CMakeLists.txt). Les tests de contrôle et de
capture sont headless; le test graphique exige un backend utilisable et peut
être ignoré dans un environnement sans affichage.

## 9) Utilitaires

### `src/Helpers.hpp`

- Rôle : types et helpers transverses (dataset items, IO, magic tokens, etc.).

### `src/RngContext.hpp`

- Rôle : PRNG/seed management (déterminisme best-effort selon ops).

### `src/Sha256.hpp` / `src/Sha256.cpp`

- Rôle : checksum (utilisé par debug/serialization selon options).

---

## Lecture guidée (par tâche)

- "Je veux activer l'accélération GPU" : lire [docs/05-Advanced/05-GPU-Acceleration.md](05-GPU-Acceleration.md), puis variables d'environnement `MIMIR_CUDA_*` / `MIMIR_ROCM_*`, puis internals dans `src/runtimes/cuda/CudaRuntime.cpp`.
- "Je veux comprendre l'exécution d'un forward multi-input" : `src/scriptings/Lua/luaScripting/LuaScripting.cpp` (forward), puis `src/Model.cpp` (forwardPassNamed), puis `src/Layers.hpp` (inputs/output).
- "Je veux comprendre la mémoire stricte (OOM explicites)" : `src/MemoryGuard.hpp`, `src/runtimes/cpu/DynamicTensorAllocator.hpp`, `src/runtimes/cpu/RuntimeAllocator.hpp`, et les usages dans `src/Model.cpp` et `src/tensors.cpp`.
- "Je veux comprendre le Transformer causal" : `src/Models/NLP/TransformerModel.cpp` (builder) + ops dans `src/LayerOps*` + forward tokens dans `src/Model.cpp`.
- "Je veux un dump exploitable pour debug" : `src/Serialization/DebugJsonDump.*` + binding `Mimir.Serialization.save_enhanced_debug` dans `src/scriptings/Lua/luaScripting/LuaScripting.cpp`.
- "Je veux comprendre le nuage 3D du modèle courant" : `src/Model.cpp`
  (`captureLatentSpaceSnapshot`) → `src/scriptings/ModelVizProjection.hpp`
  (liaison et points sûrs) → `src/viz/LatentSpaceProjection.hpp` (copie/ACP/contrôle)
  → `src/AsyncMonitor.hpp` (worker) → `src/Visualizer.cpp` (rendu et événements).
- "Je veux ajouter un nouveau fast-path GPU" : `src/runtimes/AbstractRuntime.hpp` (RuntimeConfig) → `src/runtimes/cuda/CudaRuntime.cpp` (case + DeviceBuf) → `src/runtimes/rocm/RocmRuntime.cpp` (miroir).

## Étapes suivantes

- [Page précédente : LLM readiness (état réel)](03-LLM-Readiness.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Accélération GPU](05-GPU-Acceleration.md)
