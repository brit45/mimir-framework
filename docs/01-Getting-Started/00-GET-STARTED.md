# 🚀 GET STARTED — Démarrage rapide Mímir

## Pour qui

Débutant complet (aucune expérience framework requise).

## Objectif

Compiler Mímir et exécuter un premier test fonctionnel en quelques minutes.

## Avant de commencer

Avoir un terminal Linux et les droits pour compiler le projet.

## Résultat attendu

Tu obtiens un binaire exécutable et un smoketest qui passe.


Cette page te montre comment **compiler**, **tester** et **utiliser** Mímir en 10 minutes.

---

## ⚡ En 5 étapes

### 1️⃣ Vérifier les prérequis

```bash
# Vérifier CMake
cmake --version  # ≥ 3.15

# Vérifier le compilateur C++17
g++ --version    # ou clang++, autre C++17-compatible

# Vérifier Lua
lua -v            # Lua 5.3+ (optionnel si système pas l'a, CMake téléchargera)
```

### 2️⃣ Compiler

```bash
# Cloner et accéder au répo
cd ~/path/to/tensor-2

# Compiler
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

**Durée:** 2–5 minutes selon la machine.  
**Résultat:** binaire `./bin/mimir`

### 3️⃣ Vérifier l'installation

```bash
# Afficher l'aide et capacités hardware détectées
./bin/mimir --help

# Exécuter un test rapide (smoke test)
./bin/mimir --lua scripts/tests/test_serialization_smoke.lua

# ✅ Si pas d'erreur → tout est bon!
```

### 4️⃣ Créer ton premier modèle

```bash
# Exécuter le template minimal
./bin/mimir --lua scripts/templates/template_new_model.lua
```

**Ce qui se passe:**
- Charge une config Transformer depuis le registre
- Crée le modèle + alloue les poids
- Lance un `forward()` test
- Affiche le résultat

### 5️⃣ Sauvegarder et charger un checkpoint

```bash
# Exécuter un script d'exemple
./bin/mimir --lua scripts/templates/template_pipeline_args.lua \
  --arch transformer --d-model 256 --layers 4 --epochs 1 --save-checkpoint

# ✅ Checkpoint sauvegardé dans `checkpoints/`
```

---

## 📚 Prochaines étapes

| Objectif | Ressource |
| --- | --- |
| **Comprendre le cycle de vie** | [Model Lifecycle](../02-User-Guide/02-Model-Lifecycle.md) |
| **Entraîner un modèle** | [Training Guide](../02-User-Guide/04-Training.md) |
| **Utiliser un modèle pré-entraîné** | [Inference Guide](../02-User-Guide/05-Inference.md) |
| **Sauvegarder/charger des poids** | [Serialization API](../03-API-Reference/02-Serialization.md) |
| **Explorer les architectures** | `./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -a` |

---

## ❓ Problèmes courants

### Problème: CMake ne trouve pas Lua

**Solution:**
```bash
# Installer Lua dev
# Ubuntu/Debian:
sudo apt-get install lua5.3 liblua5.3-dev

# macOS:
brew install lua@5.3

# RedHat:
sudo dnf install lua-devel
```

### Problème: Compilation échoue avec erreur OpenMP

**Solution:**
```bash
# Installer OpenMP
# Ubuntu/Debian:
sudo apt-get install libomp-dev

# Compiler sans OpenMP:
cmake -S . -B build -DENABLE_OPENMP=OFF
cmake --build build -j
```

### Problème: OpenCL/Vulkan non détecté

**Info:** Ces accélérations sont **optionnelles**. La CPU suffit pour débuter.  
Passer si tu n'en as pas besoin. Pour plus tard, voir [Installation détaillée](./02-Installation.md#dependencies-optionnelles).

### Problème: `./bin/mimir` n'existe pas après build

**Vérifier:**
```bash
# Vérifier le build
cmake --build build --verbose

# Vérifier le contenu de bin/
ls -la bin/

# Si absent, nettoyer et recommencer
rm -rf build
cmake -S . -B build
cmake --build build -j
```

---

## 🎯 Commandes utiles

```bash
# Lister toutes les architectures disponibles
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -a

# Créer un modèle MLP simple
./bin/mimir --lua scripts/templates/template_new_model.lua \
  --arch basic_mlp --hidden-dim 512 --hidden-layers 3

# Inspecter un checkpoint
./bin/mimir --lua scripts/tools/inspect_checkpoint.lua \
  checkpoints/my_model.safetensors

# Exécuter tous les tests
ctest --test-dir build
```

---

## 📖 Pour aller plus loin

- **[Installation complète](./02-Installation.md)** — toutes les options CMake et dépendances
- **[CLI Reference](./03-CLI.md)** — tous les flags `./bin/mimir`
- **[Repo Layout](./04-Repo-Layout.md)** — structure du projet
- **[Lua API Reference](../03-API-Reference/)** — toutes les APIs Lua disponibles
