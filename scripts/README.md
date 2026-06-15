# Scripts Mímir v3.0.1

Organisation des scripts Lua pour le framework Mímir.

Cette page décrit les scripts utiles dans le workspace actuel. Elle ne reflète pas l’ancienne arborescence documentaire archivée.

## Structure

```text
scripts/
├── demos/           # (réservé) Démonstrations (actuellement vide)
├── examples/        # (réservé) Exemples (actuellement vide)
├── tests/           # Tests et validation
├── benchmarks/      # Benchmarks
├── training/        # Scripts d'entraînement
├── templates/       # Templates pour nouveaux modèles
├── modules/         # Modules partagés (args, ws server, tokenizer, etc.)
└── tools/           # Outils divers (inspection d'archis, analyse de modèles)
```

## Catégories

### 📊 Démonstrations (`demos/`)

(Dossier réservé — vide dans ce workspace.)

### 💡 Exemples (`examples/`)

(Dossier réservé — vide dans ce workspace.)

### 🧪 Tests (`tests/`)

Scripts de validation et tests:

- `test_vae_conv_generate.lua` - Génération VAE Conv (smoke)
- `test_serialization_smoke.lua` - Smoke test sérialisation (SafeTensors)

Usage recommandé : commence par `test_serialization_smoke.lua` si tu veux un signal rapide que le binaire, l’API Lua et la sérialisation sont tous opérationnels.

### 🛠️ Outils (`tools/`)

Outils d'inspection et d'analyse:

- `inspect_architectures.lua` - Liste les architectures + dtypes, et affiche les paramètres d'une archi (`-a`, `-l <arch> -p`, `-d`)
- `analyze_model.lua` - Analyse un checkpoint/modèle (SafeTensors / RawFolder / DebugJson)
- `build_tags_vocab.lua` - Construit un vocab de tags depuis un dataset (`.txt` séparés par des points)
- `convert_checkpoint2safetensor.lua` - Convertit un checkpoint RawFolder → SafeTensors
- `convert_safetensors2raw_folder.lua` - Convertit un checkpoint SafeTensors → RawFolder

### ⚡ Benchmarks (`benchmarks/`)

Scripts de performance:

- `benchmark_official.lua` - Benchmark standard
- `benchmark_stress.lua` - Test de stress
- `benchmark.lua` / `benchmark_complet.lua` / `benchmark_conv_train.lua` - Benchmarks complémentaires

### 🎓 Training (`training/`)

Scripts d'entraînement:

- `ponyxl_ddpm_train.lua` - Entraînement PonyXL-DDPM (diffusion)
- `ponyxl_ddpm_direct_train.lua` - Entraînement PonyXL-DDPM direct sur une image + un prompt, sans loader de dataset
- `train_vae_conv.lua` - Entraînement VAE Conv
- `train_vae_texte.lua` - Entraînement VAE Texte

### 📝 Templates (`templates/`)

Templates pour développement:

- `template_new_model.lua` - Template pour nouveau modèle
- `template_pipeline_only.lua` - Template minimal (pipeline API uniquement)
- `template_pipeline_args.lua` - Template (args + overrides + pipeline API)

Choix rapide :

- `template_new_model.lua` si tu veux comprendre le lifecycle complet d’un modèle.
- `template_pipeline_only.lua` si tu veux juste tester l’enchaînement pipeline sans couche d’arguments.
- `template_pipeline_args.lua` si tu veux un point de départ plus réaliste pour un script maintenable.

## Utilisation

### Exécution depuis la racine du projet

```bash
# Template
./bin/mimir --lua scripts/templates/template_new_model.lua
./bin/mimir --lua scripts/templates/template_pipeline_only.lua
./bin/mimir --lua scripts/templates/template_pipeline_args.lua -- --no-train

# Tests
./bin/mimir --lua scripts/tests/test_serialization_smoke.lua

# Outils
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -a

# Benchmark
./bin/mimir --lua scripts/benchmarks/benchmark_official.lua -- --safe --iters 1

# Training
./bin/mimir --lua scripts/training/ponyxl_ddpm_train.lua -- --help
./bin/mimir --lua scripts/training/ponyxl_ddpm_direct_train.lua -- --help
```

Lecture pratique de ces commandes :

- templates : découverte et prototypage,
- tests : validation ciblée d’un sous-système,
- benchmarks : mesure rapide et régression perf,
- training : scripts complets avec plus de paramètres et d’état.

### Avec run_mimir.sh

```bash
./run_mimir.sh --lua scripts/templates/template_new_model.lua
```

## Statistiques

- **NB** : ce README reflète l'état du dossier `scripts/` à la date de la release.

En cas de doute sur un script, la référence principale reste le code Lua lui-même. Plusieurs fichiers sont pensés comme exemples exécutables avant d’être des tutoriels exhaustifs.

## Voir aussi

- [Documentation complète](../docs/00-INDEX.md)
- [Guide de démarrage rapide](../docs/01-Getting-Started/01-Quick-Start.md)
- [Vue d’ensemble API Lua](../docs/03-API-Reference/00-API-Overview.md)
- [Workflow modèle](../docs/02-User-Guide/02-Model-Lifecycle.md)

---

**Version**: 3.0.1 | **Date**: 5 juin 2026
