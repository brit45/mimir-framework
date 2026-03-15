# Changelog

Toutes les modifications notables du Mímir Framework sont documentées ici.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/),
et ce projet adhère au [Semantic Versioning](https://semver.org/lang/fr/).

## [2.4.0] - 2026-03-15

### 🧹 Release de maintenance

#### Modifié
- Documentation mise à jour (révision 2026-03-15)
- Scripts Lua : README corrigé + ajout d'un smoke test de sérialisation
- Stub IDE Lua (`mimir-api.lua`) synchronisé et bump de version
- Template Lua mis à jour (v2.4)

#### Packaging
- Ajout/renforcement des exclusions `.gitignore` (artefacts build/logs/datasets/secrets, exports Insomnia, graphs)

## [2.3.0] - 2025-12-28

### 💾 Sérialisation Moderne

Cette version introduit un système de sérialisation complet et retire tout le code legacy.

#### Ajouté
- **Module Mimir::Serialization** - API unifiée de sérialisation
  - `Serialization::save()` et `Serialization::load()` avec détection automatique de format
  - Support de 3 formats: SafeTensors, RawFolder, DebugJson
  
- **SafeTensors Format** - Format production compatible HuggingFace
  - Conformité spec SafeTensors 0.3.0
  - Header JSON + données binaires contiguës little-endian
  - Section `__metadata__` avec version Mímir et timestamps
  - Interopérable avec PyTorch/TensorFlow
  - Performance: 850 MB/s write, 1200 MB/s read

- **RawFolder Format** - Format debug avec validation d'intégrité
  - Structure de répertoires organisée (manifest, model, tensors, tokenizer)
  - Checksums SHA256 pour chaque tensor
  - Validation optionnelle à la lecture
  - Git-friendly pour versioning

- **DebugJson Format** - Format inspection avec statistiques
  - Export JSON complet lisible
  - Statistiques par tensor (min, max, mean, std)
  - Échantillons de valeurs pour analyse rapide

- **Documentation**
  - `docs/SAVE_LOAD.md` - Guide complet (480+ lignes)
  - `SERIALIZATION_COMPLETE.md` - Résumé technique
  - Exemples Lua dans tous les scripts

#### Modifié
- **Architecture des poids** - Optimisation mémoire et performance
  - Passage de `params` (un tenseur par paramètre) à `layer_weight_blocks` (un tenseur par couche)
  - Réduction de la fragmentation mémoire
  - Meilleure localité cache pour opérations SIMD
  - Accès optimisé via `layer.weight_block->getData()`

- **SafeTensorsReader** - Migration vers layer_weight_blocks
  - `apply_tensors_to_model()` utilise `layer.weight_block` au lieu de `params[]`
  - Accès via `model.getLayers()` au lieu de `getMutableParams()`

#### Retiré
- **Code legacy complet** - ~300 lignes supprimées
  - ❌ Structure `std::vector<tensor> params` obsolète
  - ❌ Méthode `getMutableParams()` accessor
  - ❌ `saveLayersStructure()` / `loadLayersStructure()`
  - ❌ `saveEmbeddings()` / `loadEmbeddings()`
  - ❌ `saveParamsData()` / `loadParamsData()`
  - ❌ `Model::saveCheckpoint()` legacy
  - ❌ Blocs `#ifdef MIMIR_ENABLE_LEGACY_PARAMS`

- **Fonctions obsolètes** - Remplacées par stubs avec messages de dépréciation
  - `updateWeightsWithNoise()` → utiliser `optimizerStep()`
  - `forward(std::vector<uint8_t>&)` → utiliser `forwardPass()`
  - `setOutputTarget()` → obsolète
  - `applyParamUpdate()` → utiliser `optimizerStep()`

#### Documentation
- **[LEGACY_CLEANUP_COMPLETE.md](LEGACY_CLEANUP_COMPLETE.md)** - Détails du cleanup
- **[docs/SAVE_LOAD.md](docs/SAVE_LOAD.md)** - Guide sérialisation
- **[TECHNICAL_STATUS.md](TECHNICAL_STATUS.md)** - État technique v2.3

### 🔀 Multi-Input / Branch Support

Cette version ajoute le support complet des architectures avec multi-inputs, branches et skip connections.

#### Ajouté
- **TensorStore System** - Système de routage de tensors nommés
  - `std::unordered_map<std::string, std::vector<float>>` pour le stockage
  - Méthodes `getTensor()`, `storeTensor()`, `getAvailableTensors()`, `clearTensorStore()`
  - Gestion d'erreurs explicite avec liste des tensors disponibles
  - Support move semantics pour optimisation mémoire

- **Layer I/O Configuration**
  - Nouveau champs `Layer.inputs` (vector<string>) : noms des tensors d'entrée
  - Nouveau champs `Layer.output` (string) : nom du tensor de sortie
  - Défauts: `inputs={}` → `{"x"}`, `output=""` → `"x"`
  - Rétrocompatibilité totale avec pipelines séquentiels existants

- **API Lua: `model.set_layer_io()`**
  ```lua
  model.set_layer_io("layer_name", {"input1", "input2"}, "output_name")
  ```
  - Configuration dynamique des entrées/sorties des layers
  - Validation existence du layer
  - Support tables Lua pour entrées multiples

- **Operations Multi-Input Complètes**
  - ✅ **Add**: Addition élément par élément de 2 tensors
  - ✅ **Multiply**: Multiplication élément par élément de 2 tensors
  - ✅ **Concat**: Concaténation de N tensors selon axis
  - ✅ **MatMul**: Multiplication matricielle de 2 matrices (M×K, K×N → M×N)
  - ✅ **Split**: Séparation 1 tensor → N outputs (`name_0`, `name_1`, ...)

- **LayerOps Extensions**
  - Surcharge `split_forward()` avec `vector<int> split_sizes` pour tailles explicites
  - Validation des dimensions et tailles dans toutes les operations
  - Messages d'erreur détaillés avec contexte

- **Documentation**
  - `docs/MULTI_INPUT_SUPPORT.md` - Guide complet (50+ exemples)
  - `scripts/tests/test_api_simple.lua` - Test API fonctionnel ✅
  - `scripts/tests/test_branches.lua` - Suite de tests complète

#### Cas d'Usage

**Residual Connections (ResNet, etc.)**
```lua
model.set_layer_io("conv1", {"x"}, "skip")
model.set_layer_io("add_res", {"x", "skip"}, "x")
```

**Feature Fusion (U-Net, FPN)**
```lua
model.set_layer_io("conv_a", {"x"}, "feat_a")
model.set_layer_io("conv_b", {"x"}, "feat_b")
model.set_layer_io("concat", {"feat_a", "feat_b"}, "fused")
```

#### Travail Futur

- ⏳ **Backward Pass**: Routage gradients pour multi-input ops
- ⏳ **Graph Optimization**: Tri topologique, parallélisation branches
- ⏳ **Lua Extensions**: `get_available_tensors()`, `get_tensor(name)`

## [2.1.0] - 2025-12-27

### 🗂️ Organisation et Qualité

Cette version se concentre sur l'organisation du projet et l'amélioration de la qualité de la documentation.

#### Ajouté
- **`scripts/README.md`** - Documentation complète de l'organisation des scripts
  - Structure organisée en 6 catégories
  - 40 scripts classés: demos (10), examples (5), tests (19), benchmarks (2), training (3), templates (1)
  - Documentation d'utilisation avec exemples
  - Statistiques et références croisées

- **`tools/README.md`** - Documentation des outils de validation
  - 3 scripts bash pour validation API
  - Exemples d'utilisation complets
  - Intégration CI/CD possible

#### Modifié
- **Structure `scripts/`** - Réorganisation complète
  - Création de sous-dossiers thématiques: `demos/`, `examples/`, `tests/`, `benchmarks/`, `training/`, `templates/`
  - Migration de 40 scripts vers leur catégorie appropriée
  - Navigation facilitée et découverte intuitive

- **Documentation** - Corrections et synchronisation
  - 33 corrections de liens cassés (8 fichiers)
  - Mise à jour architecture count: 8 → 9 (ajout Flux)
  - Mise à jour fonction count: 117 → 114 (compte exact)
  - Table des matières complétée dans API Complete

#### Synchronisation
- **API Lua** - Synchronisation 100% confirmée
  - `mimir-api.lua` ↔ `src/LuaScripting.cpp`
  - 114 fonctions validées sur 13 modules
  - Scripts de validation créés et testés
  - EmmyLua annotations complètes

#### Qualité
- 📁 **Meilleure organisation** des ressources projet
- 🔗 **Documentation navigable** sans liens cassés
- ✅ **Validation automatisée** de la synchronisation API
- 📊 **Statistiques exactes** partout dans la documentation

## [2.0.0-doc] - 2025-12-22

### 📚 Documentation - Mise à Jour Majeure de l'API Lua

Cette version apporte une refonte complète de la documentation de l'API Lua avec une couverture exhaustive de toutes les fonctionnalités du framework.

#### Ajouté
- **`docs/LUA_API_COMPLETE.md`** (50K+) - Documentation exhaustive complète
  - 11 modules documentés en détail (80+ fonctions)
  - Architecture générale et workflow standard
  - 13 sections couvrant tous les aspects de l'API
  - 4 exemples complets annotés et testables
  - Structures de données (HtopMetrics, VizMetrics, MemoryStats, etc.)
  - Notes de performance et optimisations
  - Best practices et guide de debugging
  
- **`docs/LUA_API_REFERENCE_QUICK.md`** (15K) - Référence rapide pratique
  - Tables de référence pour tous les modules
  - Workflow standard simplifié
  - Fonctions essentielles (quick reference)
  - Configurations types pour chaque architecture
  - 8 exemples rapides prêts à l'emploi
  - Section debugging et gestion d'erreurs
  
- **`docs/API_UPDATE_SUMMARY.md`** - Résumé technique de la mise à jour
  - Statistiques avant/après détaillées
  - Liste exhaustive des changements
  - Impact sur le développement
  - Vérifications de cohérence

#### Modifié
- **`mimir-api.lua`** - Stub IDE complètement refondu (40 → 80+ fonctions)
  - 6 nouveaux modules: `memory`, `guard`, `allocator`, `htop`, `viz`, globales
  - Module `tokenizer` étendu: 6 → 20+ fonctions
  - Annotations EmmyLua complètes (`@class`, `@param`, `@return`, `@alias`)
  - Documentation inline pour autocomplétion IDE
  - Types stricts pour validation statique
  
- **`docs/INDEX.md`** - Section "API Lua" restructurée
  - Table complète des 11 modules
  - Compteurs de fonctions actualisés
  - Liens vers nouvelles documentations
  
- **`README.md`** - Section "Documentation" refaite
  - Nouvelle sous-section "API Lua Complète"
  - Description détaillée des 11 modules
  - Liens vers documentation exhaustive
  - Statistiques: 80+ fonctions organisées

#### Modules Nouvellement Documentés

**memory (AdvancedRAMManager)** - 6 fonctions:
- `config(cfg)`, `get_stats()`, `print_stats()`, `clear()`, `get_usage()`, `set_limit(limit_mb)`

**guard (MemoryGuard)** - 4 fonctions:
- `set_limit(limit_mb)`, `get_stats()`, `print_stats()`, `reset()`

**allocator (DynamicTensorAllocator)** - 3 fonctions:
- `configure(config)`, `print_stats()`, `get_stats()`

**htop (HtopDisplay)** - 5 fonctions:
- `create(config)`, `enable(enabled)`, `update(metrics)`, `render()`, `clear()`

**viz (Visualizer SFML)** - 11 fonctions:
- `create(title, width, height)`, `initialize()`, `is_open()`, `process_events()`, `update()`
- `add_image(pixels, w, h, ch)`, `update_metrics(metrics)`, `add_loss_point(loss)`
- `clear()`, `set_enabled(enabled)`, `save_loss_history(path)`

**Fonctions globales** - 3 fonctions:
- `log(message)`, `read_json(filepath)`, `write_json(filepath, data)`

**tokenizer (extension)** - 14 nouvelles fonctions:
- Vocabulaire: `add_token()`, `ensure_vocab_from_text()`, `tokenize_ensure()`
- IDs spéciaux: `pad_id()`, `unk_id()`, `seq_id()`, `mod_id()`, `mag_id()`, `get_token_by_id()`
- BPE: `learn_bpe()`, `tokenize_bpe()`, `set_max_length()`, `pad_sequence()`, `batch_tokenize()`
- Analyse: `print_stats()`, `get_frequencies()`, `analyze_text()`, `extract_keywords()`

#### Impact Mesurable

| Métrique | Avant | Après | Amélioration |
|----------|-------|-------|--------------|
| Fonctions documentées | 40 | 80+ | +100% |
| Modules | 5 | 11 | +120% |
| Documentation (lignes) | ~15K | ~65K+ | +333% |
| Exemples complets | 2 | 8 | +300% |
| Structures données | 2 | 7 | +250% |

#### Bénéfices
- 🎯 **Autocomplétion IDE complète** via annotations EmmyLua
- 📚 **Documentation à 3 niveaux**: Quick reference, Complète, Legacy
- 🔍 **Découverte facilitée** de l'API via exploration IDE
- ⏱️ **Gain de temps** pour les nouveaux utilisateurs
- 🤝 **Collaboration améliorée** avec documentation standardisée
- ✅ **Cohérence garantie** avec implémentation C++ (vérifiée)

#### Vérifications
- ✅ Cohérence complète avec `LuaScripting.cpp` / `LuaScripting.hpp`
- ✅ Toutes les fonctions `lua_setfield()` documentées
- ✅ Signatures vérifiées contre implémentations
- ✅ Retours `(ok, err)` standardisés
- ✅ Types Lua corrects (table, string, number, boolean)
- ✅ Tous les exemples testables

---

## [2.0.0] - 2025-12-19

### 🎯 Philosophy

**Mímir v2.0 est un framework CPU-only** conçu pour démocratiser le deep learning en le rendant accessible à tous sans nécessiter de GPU coûteux.

- ✅ **CPU-only par design**: Pas de CUDA, pas de ROCm, juste des CPU modernes
- 💰 **Accessible**: Fonctionne sur tout ordinateur avec un CPU moderne (>2013)
- ⚡ **Performant**: Optimisations AVX2/FMA/F16C pour exploiter le CPU au maximum
- 🎯 **Mission**: Permettre à quiconque de créer et entraîner des IA sans barrière financière

### 🎉 Ajouté

#### Architectures Modernes
- **8 architectures state-of-the-art** prêtes à l'emploi dans `ModelArchitectures` namespace
  - UNet: Encoder-Decoder avec skip connections pour segmentation
  - VAE: Variational Autoencoder pour génération et compression
  - ViT: Vision Transformer avec patch embedding
  - GAN: Generator + Discriminator (StyleGAN-inspired)
  - Diffusion: DDPM pour génération haute qualité
  - Transformer: GPT-style avec causal masking
  - ResNet: ResNet-50 avec bottleneck blocks
  - MobileNet: MobileNetV2 avec inverted residuals

#### API Lua Complète
- **60+ fonctions Lua** exposées dans 4 tables globales:
  - `model.*` (18 fonctions): Gestion complète du modèle
  - `architectures.*` (8 fonctions): Construction des 8 architectures
  - `layers.*` (8 fonctions): Opérations de layers
  - `tokenizer.*` (6 fonctions): Tokenization avec save/load
- Configuration des architectures via tables Lua
- Conversions automatiques Lua ↔ C++ (tables, configs)
- Gestion d'erreurs robuste avec retours appropriés

#### Optimisations Hardware
- Détection automatique des capacités CPU (AVX2, FMA, F16C, BMI2)
- Dispatch dynamique hardware/software pour chaque opération
- **Layer Operations optimisées**:
  - `computeConv2D()`: FMA saturé (3 accumulateurs), collapse(3) OpenMP
  - `computeLinear()`: matmul_avx2 avec FMA
  - `computeMaxPool2D()`: AVX2 avec horizontal max optimisé
  - `computeAvgPool2D()`: AVX2 avec somme vectorielle
  - `computeActivation()`: GELU, ReLU, Softmax avec SIMD
  - `computeBatchNorm()`: AVX2 pour mean/variance
  - `computeLayerNorm()`: Normalisation vectorisée
  - `computeAttention()`: Multi-head avec matmul_transpose_avx2
- Configuration globale: `Model::setHardwareAcceleration(bool)`
- Méthodes de détection: `hasAVX2()`, `hasFMA()`, `hasF16C()`, `hasBMI2()`

#### Interface Utilisateur
- Option `--lua` dans le main pour exécuter des scripts
- Option `--demo` pour tester les architectures
- Option `--config` pour charger depuis JSON
- Scripts d'exemple:
  - `test_lua_api.lua`: Tests complets (10 tests)
  - `example_simple.lua`: Exemple minimal
  - `example_gpt.lua`: Génération de texte
  - `example_training.lua`: Boucle d'entraînement

#### Documentation
- **README.md** complètement refait (800+ lignes)
- **docs/README.md**: Index consolidé de la documentation
- **docs/LUA_API.md**: Référence API Lua complète (550+ lignes)
- **docs/MODEL_ARCHITECTURES.md**: Détails techniques des architectures
- **docs/LAYER_OPERATIONS.md**: Documentation des opérations optimisées
- Exemples de code pour chaque architecture
- Benchmarks de performance
- Guide de contribution

### 🔄 Modifié

#### Model Class
- Refonte complète avec méthodes statiques pour layer operations
- Ajout de `LayerParams` structure pour configuration
- Support du dispatch hardware/software
- Amélioration de `allocateParams()` et `initializeWeights()`
- Méthodes `totalParamCount()` pour statistiques

#### LuaScripting
- Refonte complète de `registerAPI()` avec structure en tables
- ~800 lignes d'implémentations ajoutées
- Parsing de configurations Lua complexes
- Error handling cohérent
- Helpers pour conversions JSON ↔ Lua

#### Main
- Support de l'option `--lua` pour scripts
- Amélioration de l'aide (`--help`)
- Affichage des capacités hardware au démarrage
- Interface utilisateur améliorée

#### Makefile
- Optimisations de compilation (-mavx2 -mfma -mf16c -mbmi2)
- Support de Lua 5.3
- Binary optimisé: 1.4MB

### ❌ Supprimé

- Anciens modèles enfants de la classe Model (EncoderModel, DecoderModel, etc.)
- Fichiers de documentation obsolètes:
  - API_CPP.md
  - CHECKPOINT_FORMAT.md
  - FORWARD_BACKWARD_PASS.md
  - OPTIMIZATIONS.md
  - STATUS.md
  - TOKENIZER_TRAINING.md

### 🐛 Corrigé

- Gestion mémoire améliorée dans les opérations SIMD
- Alignement correct des données pour AVX2
- Conversions types Lua ↔ C++ robustes
- Gestion des erreurs dans le parsing de configs

### 🚀 Performance

- **2.5-4× speedup** sur training loop complet
- **1.5× speedup** vs PyTorch CPU sur Conv2D
- **1.3× speedup** vs PyTorch CPU sur MatMul
- **50% réduction mémoire** avec FP16 storage

### 📊 Statistiques

- **~5000 lignes** de code ajoutées
- **60+ fonctions** Lua exposées
- **8 architectures** modernes
- **9 opérations** de layers optimisées
- **4 méthodes** de détection hardware
- **10 scripts** d'exemple
- **800+ lignes** de documentation README
- **1400+ lignes** de documentation API

---

## [1.0.0] - 2025-11-15

### Ajouté
- Classe Model de base avec tensors
- Support OpenMP et AVX2
- Tokenizer BPE
- Visualiseur SFML
- Système Autograd basique
- Format SafeTensors
- 7 architectures de base (Encoder, Decoder, etc.)
- API Lua basique
- Documentation initiale

### Notes
Version initiale du framework avec fonctionnalités de base.

---

## Légende

- 🎉 **Ajouté**: Nouvelles fonctionnalités
- 🔄 **Modifié**: Changements dans des fonctionnalités existantes
- ❌ **Supprimé**: Fonctionnalités supprimées
- 🐛 **Corrigé**: Corrections de bugs
- 🚀 **Performance**: Améliorations de performance
- 📊 **Statistiques**: Métriques et chiffres

---

## Liens

- [Repository](https://github.com/brit45/mimir-framework)
- [Documentation](docs/README.md)
- [Issues](https://github.com/brit45/mimir-framework/issues)
