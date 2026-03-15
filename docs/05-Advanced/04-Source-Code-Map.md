# Advanced — Carte du code source (C/C++)

Cette page sert de « table des matières technique » : comment le framework est structuré côté C/C++, fichier par fichier.

Pages Advanced associées (explications orientées usage) :

- [01-Performance.md](01-Performance.md)
- [02-Debugging.md](02-Debugging.md)
- [03-LLM-Readiness.md](03-LLM-Readiness.md)

Objectif : quand une fonctionnalité est mentionnée dans la doc (mémoire stricte, forward tokens, multi-input, sérialisation…), vous devez pouvoir retrouver rapidement **le fichier source qui fait foi**, et comprendre le rôle exact de chaque composant.

> Notes
> - La plupart des features sont exposées via l’API Lua (bindings dans `src/LuaScripting.cpp`).
> - Les modèles « prêts à l’emploi » sont construits via le registre d’architectures (`src/Models/Registry/ModelArchitectures.*`).

## 1) Point d’entrée et exposition API

### `src/main.cpp`

- Rôle : point d’entrée du binaire (init, chargement, exécution Lua / modes CLI selon build).
- À lire quand : vous voulez comprendre comment le runtime démarre, et à quel moment les sous-systèmes (Lua, compute backends, etc.) sont initialisés.

### `src/LuaScripting.hpp` / `src/LuaScripting.cpp`

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
  - Debug JSON : `Mimir.Serialization.save_enhanced_debug(path, options)` (voir aussi `src/Serialization/DebugJsonDump.*`).

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
  - (optionnel) orchestrer des backends compute (Vulkan/OpenCL) pour certaines ops en inférence

Points à connaître :

- **Trois APIs forward** importantes :
  - `forwardPass(const std::vector<float>&, training)`
  - `forwardPass(const std::vector<int>&, training)` (chemin tokens : `Embedding` lit dans l’IntTensorStore)
  - `forwardPassNamed(float_inputs, int_inputs, training)` (multi-input)
- **Optimiseur** : `optimizerStep(Optimizer&, learning_rate)` gère SGD/Adam/AdamW, avec protections (epsilon non-fini -> fallback), warmup/decay, et *grad clipping optionnel* via `modelConfig` (`grad_clip_norm` / `clip_norm`).
- **Accélération compute** (en inférence, opt-in) :
  - certaines ops (ex: `Linear`) peuvent être dispatchées vers Vulkan/OpenCL si activé via variables d’environnement (voir plus bas).

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

### `src/LayerOps.hpp` / `src/LayerOpsExt.hpp`

- Rôle : implémentations CPU des opérations (forward/backward) utilisées par `Model`.
- `LayerOpsExt` regroupe souvent des implémentations plus spécialisées/étendues.
- À lire quand : vous cherchez “où est l’implémentation réelle de Conv/Linear/Norm/Activation/Attention”.

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

### `src/DynamicTensorAllocator.hpp`

- Rôle : allocateur dynamique global (singleton) utilisé par `tensor(dynamic=true)` et certains chemins runtime.
- Fonctionnalités : réservations, stats, (optionnel) compression/éviction selon config.

### `src/RuntimeAllocator.hpp`

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

### `src/Encoder.hpp` / `src/Encoder.cpp`

- Rôle : embeddings / encodeur “mag/mod” et embeddings tokens.
- Important : le `Model` maintient la compatibilité `Tokenizer` <-> `Encoder` (ex: ensure vocab size).

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
  - `src/Models/NLP/TransformerModel.hpp` / `.cpp` : builder Transformer (peut être causal)
  - `src/Models/NLP/VAETextModel.hpp` / `.cpp` : VAE sur texte (encode/decode)
- Vision :
  - `src/Models/Vision/ViTModel.*`, `VAEModel.*`, `VAEConvModel.*`, `UNetModel.*`
  - CNN : `ResNetModel.*`, `MobileNetModel.*`, `VGG16Model.*`, `VGG19Model.*`
- Diffusion :
  - `src/Models/Diffusion/DiffusionModel.*`, `PonyXLDDPMModel.*`, `SD35Model.*`
- Autres : `src/Models/NeuroPulse.*`

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
- Exposé via `Mimir.Serialization.save_enhanced_debug(path, options)`.

## 7) Hardware, SIMD et backends compute

### `src/HardwareOpt.hpp`

- Rôle : détection capacités CPU et helpers (AVX2/FMA/F16C/BMI2, etc.).
- Exposé via `Mimir.Model.hardware_caps()`.

### `src/SIMD_Ops.hpp`

- Rôle : primitives SIMD (CPU) utilisées par des ops.

### `src/VulkanCompute.hpp` et `src/OpenCLCompute.hpp`

- Rôle : engines compute (si compilés) pour accélérer certaines opérations.
- Politique observée : opt-in par variables d’environnement.

Variables d’environnement (exemples observés) :

- `MIMIR_DISABLE_VULKAN` / `MIMIR_DISABLE_OPENCL` : désactiver le backend.
- `MIMIR_VULKAN_LINEAR=1` : autoriser `Linear` via Vulkan en inférence.
- `MIMIR_OPENCL_LINEAR=1` : autoriser `Linear` via OpenCL en inférence.
- `MIMIR_VULKAN_LINEAR_MIN_OPS` / `MIMIR_OPENCL_LINEAR_MIN_OPS` : seuil d’opérations avant offload.
- `MIMIR_ACCEL_VERBOSE=1` : logs d’accélération.

## 8) Monitoring / visualisation

### `src/AsyncMonitor.hpp`

- Rôle : collecte asynchrone métriques + frames (si activé).

### `src/HtopDisplay.hpp`

- Rôle : affichage/collecte style htop (CPU/RAM/loss/bps…).
- Exposé via `Mimir.Htop.*`.

### `src/Visualizer.hpp` / `src/Visualizer.cpp`

- Rôle : visualisation SFML (taps d’images par bloc/layer, métriques).
- Exposé via `Mimir.Viz.*`.

## 9) Utilitaires

### `src/Helpers.hpp`

- Rôle : types et helpers transverses (dataset items, IO, magic tokens, etc.).

### `src/RngContext.hpp`

- Rôle : PRNG/seed management (déterminisme best-effort selon ops).

### `src/Sha256.hpp` / `src/Sha256.cpp`

- Rôle : checksum (utilisé par debug/serialization selon options).

---

## Lecture guidée (par tâche)

- “Je veux comprendre l’exécution d’un forward multi-input” : `src/LuaScripting.cpp` (forward), puis `src/Model.cpp` (forwardPassNamed), puis `src/Layers.hpp` (inputs/output).
- “Je veux comprendre la mémoire stricte (OOM explicites)” : `src/MemoryGuard.hpp`, `src/DynamicTensorAllocator.hpp`, `src/RuntimeAllocator.hpp`, et les usages dans `src/Model.cpp` et `src/tensors.cpp`.
- “Je veux comprendre le Transformer causal” : `src/Models/NLP/TransformerModel.cpp` (builder) + ops dans `src/LayerOps*` + forward tokens dans `src/Model.cpp`.
- “Je veux un dump exploitable pour debug” : `src/Serialization/DebugJsonDump.*` + binding `Mimir.Serialization.save_enhanced_debug` dans `src/LuaScripting.cpp`.
