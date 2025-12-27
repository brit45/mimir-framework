# Scripts Mímir v2.1.0

Organisation des scripts Lua pour le framework Mímir.

## Structure

```
scripts/
├── demos/           # Démonstrations des architectures (10 fichiers)
├── examples/        # Exemples d'utilisation (5 fichiers)
├── tests/           # Scripts de tests et validation (19 fichiers)
├── benchmarks/      # Scripts de benchmarking (2 fichiers)
├── training/        # Scripts d'entraînement (3 fichiers)
└── templates/       # Templates pour nouveaux modèles (1 fichier)
```

## Catégories

### 📊 Démonstrations (`demos/`)

Démonstrations des 9 architectures prédéfinies et fonctionnalités avancées:

- `demo_unet.lua` - Architecture UNet pour segmentation
- `demo_vae.lua` - Variational AutoEncoder
- `demo_vit.lua` - Vision Transformer
- `demo_gan.lua` - Generative Adversarial Network
- `demo_diffusion.lua` - Modèle de diffusion
- `demo_resnet.lua` - ResNet pour classification
- `demo_mobilenet.lua` - MobileNet optimisé
- `demo_branches.lua` - Système de branches conditionnelles
- `demo_memory_blocking.lua` - Système de blocking mémoire
- `demo_10gb_limit.lua` - Démonstration limite mémoire 10GB

### 💡 Exemples (`examples/`)

Exemples pratiques d'utilisation de l'API:

- `example_simple.lua` - Exemple basique
- `example_gpt.lua` - Modèle GPT-like
- `example_layer_ops.lua` - Opérations sur les couches
- `example_training.lua` - Boucle d'entraînement
- `pipeline_api.lua` - API Pipeline

### 🧪 Tests (`tests/`)

Scripts de validation et tests:

- `test_lua_api.lua` - Test de l'API Lua complète
- `test_autocompletion.lua` - Test autocomplétion IDE (114 fonctions)
- `test_flux.lua` / `test_flux_complete.lua` / `test_fluxmodel_api.lua` - Tests Flux
- `test_conv2d.lua` / `test_conv2d_dynamic.lua` / `test_conv2d_simple.lua` - Tests convolution
- `test_memory_limit_10gb.lua` / `test_10gb_limit.lua` / `test_memory_safety.lua` - Tests mémoire
- `test_gradients.lua` - Test système de gradients
- `test_tokenizer.lua` - Test tokenizer
- `test_dataset_get.lua` - Test chargement dataset
- `test_all_pipelines.lua` - Test toutes les pipelines
- `test_params.lua` / `test_simple.lua` / `test_verbose.lua` - Tests divers
- `validate_memory_fixes.lua` - Validation corrections mémoire

### ⚡ Benchmarks (`benchmarks/`)

Scripts de performance:

- `benchmark.lua` - Benchmark standard
- `benchmark_stress.lua` - Test de stress

### 🎓 Training (`training/`)

Scripts d'entraînement:

- `train_complete.lua` - Entraînement complet
- `train_flux.lua` - Entraînement Flux
- `train_llm.lua` - Entraînement LLM

### 📝 Templates (`templates/`)

Templates pour développement:

- `template_new_model.lua` - Template pour nouveau modèle

## Utilisation

### Exécution depuis la racine du projet

```bash
# Démonstration
./bin/mimir --lua scripts/demos/demo_resnet.lua

# Exemple
./bin/mimir --lua scripts/examples/example_simple.lua

# Test
./bin/mimir --lua scripts/tests/test_lua_api.lua

# Benchmark
./bin/mimir --lua scripts/benchmarks/benchmark.lua

# Training
./bin/mimir --lua scripts/training/train_complete.lua
```

### Avec run_mimir.sh

```bash
./run_mimir.sh --lua scripts/demos/demo_vit.lua
```

## Statistiques

- **Total**: 40 scripts Lua
- **Démonstrations**: 10 fichiers
- **Exemples**: 5 fichiers
- **Tests**: 19 fichiers
- **Benchmarks**: 2 fichiers
- **Training**: 3 fichiers
- **Templates**: 1 fichier

## Voir aussi

- [Documentation complète](../docs/00-INDEX.md)
- [Guide de démarrage rapide](../docs/01-Getting-Started/01-Quick-Start.md)
- [Référence API Lua](../docs/03-API-Reference/00-API-Complete.md)
- [Architectures prédéfinies](../docs/02-User-Guide/03-Predefined-Architectures.md)

---

**Version**: 2.1.0 | **Date**: 27 décembre 2025
