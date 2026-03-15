# Scripts Mímir v2.4.0

Organisation des scripts Lua pour le framework Mímir.

## Structure

```
scripts/
├── demos/           # (réservé) Démonstrations (actuellement vide)
├── examples/        # (réservé) Exemples (actuellement vide)
├── tests/           # Tests et validation
├── benchmarks/      # Benchmarks
├── training/        # Scripts d'entraînement
├── templates/       # Templates pour nouveaux modèles
├── modules/         # Modules partagés (args, ws server, tokenizer, etc.)
└── tools/           # (réservé) Outils divers (actuellement vide)
```

## Catégories

### 📊 Démonstrations (`demos/`)

(Dossier réservé — vide dans ce workspace.)

### 💡 Exemples (`examples/`)

(Dossier réservé — vide dans ce workspace.)

### 🧪 Tests (`tests/`)

Scripts de validation et tests:

- `test_list_archi_conf.lua` - Liste les architectures + configs par défaut
- `test_vae_conv_generate.lua` - Génération VAE Conv (smoke)
- `test_serialization_smoke.lua` - Smoke test sérialisation (SafeTensors)

### ⚡ Benchmarks (`benchmarks/`)

Scripts de performance:

- `benchmark_official.lua` - Benchmark standard
- `benchmark_stress.lua` - Test de stress
- `benchmark.lua` / `benchmark_complet.lua` / `benchmark_conv_train.lua` - Benchmarks complémentaires

### 🎓 Training (`training/`)

Scripts d'entraînement:

- `ponyxl_ddpm_train.lua` - Entraînement PonyXL-DDPM (diffusion)
- `train_vae_conv.lua` - Entraînement VAE Conv
- `train_vae_texte.lua` - Entraînement VAE Texte

### 📝 Templates (`templates/`)

Templates pour développement:

- `template_new_model.lua` - Template pour nouveau modèle

## Utilisation

### Exécution depuis la racine du projet

```bash
# Template
./bin/mimir --lua scripts/templates/template_new_model.lua

# Tests
./bin/mimir --lua scripts/tests/test_list_archi_conf.lua
./bin/mimir --lua scripts/tests/test_serialization_smoke.lua

# Benchmark
./bin/mimir --lua scripts/benchmarks/benchmark_official.lua -- --safe --iters 1

# Training
./bin/mimir --lua scripts/training/ponyxl_ddpm_train.lua -- --help
```

### Avec run_mimir.sh

```bash
./run_mimir.sh --lua scripts/templates/template_new_model.lua
```

## Statistiques

- **NB** : ce README reflète l'état du dossier `scripts/` à la date de la release.

## Voir aussi

- [Documentation complète](../docs/00-INDEX.md)
- [Guide de démarrage rapide](../docs/01-Getting-Started/01-Quick-Start.md)
- [Référence API Lua](../docs/03-API-Reference/00-API-Complete.md)
- [Architectures prédéfinies](../docs/02-User-Guide/03-Predefined-Architectures.md)

---

**Version**: 2.4.0 | **Date**: 15 mars 2026
