# 📚 Index Complet - Documentation Mímir Framework v2.0

**Guide de navigation complet pour toute la documentation du framework**

Date de mise à jour: 19 décembre 2025  
Version du framework: 2.0.0

---

## 🎯 Démarrage Rapide (5 minutes)

| Document | Description | Taille |
|----------|-------------|--------|
| **[README.md](../README.md)** | Vue d'ensemble du framework | 16K |
| **[docs/QUICKSTART.md](QUICKSTART.md)** | Premier modèle en 5 minutes | 7.0K |
| **[docs/INSTALLATION.md](INSTALLATION.md)** | Guide d'installation | 9.7K |

**Workflow recommandé**: README → QUICKSTART → Créer votre premier script Lua

---

## 💡 Philosophie CPU-Only

| Document | Description | Taille |
|----------|-------------|--------|
| **[WHY_CPU_ONLY.md](WHY_CPU_ONLY.md)** | 🎯 Pourquoi Mímir est CPU-only (arguments économiques, accessibilité, simplicité) | 12K |
| **[ROADMAP.md](ROADMAP.md)** | Roadmap future (CPU-only forever, pas de GPU) | 8K |

**Mission**: Démocratiser l'IA sans barrière financière - 0€ de GPU requis

---

## 📖 Référence API Complète

### API Lua

| Document | Contenu | Taille |
|----------|---------|--------|
| **[docs/LUA_API.md](LUA_API.md)** | 📘 Référence complète (60+ fonctions) | 14K |
| **[docs/API_LUA.md](API_LUA.md)** | 📗 Version alternative avec exemples | 13K |

**Tables exposées**:

- `model.*` - 18 fonctions de gestion
- `architectures.*` - 8 constructeurs d'architectures
- `layers.*` - 8 opérations de layers
- `tokenizer.*` - 6 fonctions de tokenization
- `dataset.*` - 2 fonctions de données

### Architectures

| Document | Contenu | Taille |
|----------|---------|--------|
| **[docs/MODEL_ARCHITECTURES.md](MODEL_ARCHITECTURES.md)** | Détails des 8 architectures | 9.2K |

**Architectures disponibles**:

1. UNet - Segmentation/Génération (~15.6M params)
2. VAE - Compression/Génération (~1.3M params)
3. ViT - Vision Transformer (~86M params)
4. GAN - Génération adversariale (~6.3M params)
5. Diffusion - DDPM (~35M params)
6. Transformer - LLMs GPT-style (~117M params)
7. ResNet - Classification (~25M params)
8. MobileNet - Mobile/Embarqué (~3.5M params)

---

## 🏗️ Architecture Technique

| Document | Contenu | Taille |
|----------|---------|--------|
| **[docs/ARCHITECTURE.md](ARCHITECTURE.md)** | Architecture interne du framework | 19K |
| **[docs/TECHNICAL_SPECS.md](TECHNICAL_SPECS.md)** | Spécifications techniques complètes | 21K |
| **[docs/HARDWARE_OPTIMIZATIONS.md](HARDWARE_OPTIMIZATIONS.md)** | Optimisations AVX2/FMA/F16C/BMI2 | 14K |
| **[docs/LAYER_OPERATIONS.md](LAYER_OPERATIONS.md)** | Opérations de layers optimisées | 11K |

**Sujets couverts**:

- Détection hardware (AVX2, FMA, F16C, BMI2)
- Dispatch dynamique hardware/software
- Optimisations SIMD détaillées
- Benchmarks de performance
- Système de tenseurs et autograd

---

## 📝 Scripts et Exemples

### Scripts Lua d'Exemple

| Script | Description | Taille |
|--------|-------------|--------|
| **[scripts/test_lua_api.lua](../scripts/test_lua_api.lua)** | Tests complets de l'API (10 tests) | 16K |
| **[scripts/example_simple.lua](../scripts/example_simple.lua)** | Exemple minimal Transformer | 5.3K |
| **[scripts/example_gpt.lua](../scripts/example_gpt.lua)** | Génération de texte GPT | 1.6K |
| **[scripts/example_training.lua](../scripts/example_training.lua)** | Boucle d'entraînement | 4.2K |
| **[scripts/example_layer_ops.lua](../scripts/example_layer_ops.lua)** | Démonstration opérations | 8.1K |

**Usage**:

```bash
./bin/mimir --lua scripts/example_simple.lua
```

---

## 🛠️ Développement et Contribution

| Document | Contenu | Taille |
|----------|---------|--------|
| **[CONTRIBUTING.md](../CONTRIBUTING.md)** | Guide de contribution | 6.9K |
| **[CHANGELOG.md](../CHANGELOG.md)** | Historique des versions | 5.6K |
| **[VERSION](../VERSION)** | Version actuelle | 6 bytes |
| **[docs/ROADMAP.md](ROADMAP.md)** | Roadmap et futures features | - |

---

## 📊 Documentation par Thème

### 🚀 Pour les Débutants

1. **[README.md](../README.md)** - Vue d'ensemble complète
2. **[QUICKSTART.md](QUICKSTART.md)** - Premier modèle
3. **[INSTALLATION.md](INSTALLATION.md)** - Installation
4. **[example_simple.lua](../scripts/example_simple.lua)** - Script minimal

### 🎓 Pour les Développeurs

1. **[LUA_API.md](LUA_API.md)** - API complète
2. **[MODEL_ARCHITECTURES.md](MODEL_ARCHITECTURES.md)** - Architectures
3. **[example_training.lua](../scripts/example_training.lua)** - Training

### 🔬 Pour les Experts

1. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Internals
2. **[TECHNICAL_SPECS.md](TECHNICAL_SPECS.md)** - Spécifications complètes
3. **[HARDWARE_OPTIMIZATIONS.md](HARDWARE_OPTIMIZATIONS.md)** - Optimisations
4. **[LAYER_OPERATIONS.md](LAYER_OPERATIONS.md)** - Opérations optimisées

### 📐 Par Architecture

| Architecture | Documentation | Script Exemple |
|--------------|---------------|----------------|
| **UNet** | [MODEL_ARCHITECTURES.md](MODEL_ARCHITECTURES.md#unet) | test_lua_api.lua (Test 1) |
| **VAE** | [MODEL_ARCHITECTURES.md](MODEL_ARCHITECTURES.md#vae) | test_lua_api.lua (Test 5) |
| **ViT** | [MODEL_ARCHITECTURES.md](MODEL_ARCHITECTURES.md#vit) | test_lua_api.lua (Test 4) |
| **GAN** | [MODEL_ARCHITECTURES.md](MODEL_ARCHITECTURES.md#gan) | test_lua_api.lua (Test 6) |
| **Diffusion** | [MODEL_ARCHITECTURES.md](MODEL_ARCHITECTURES.md#diffusion) | test_lua_api.lua (Test 7) |
| **Transformer** | [MODEL_ARCHITECTURES.md](MODEL_ARCHITECTURES.md#transformer) | example_gpt.lua |
| **ResNet** | [MODEL_ARCHITECTURES.md](MODEL_ARCHITECTURES.md#resnet) | test_lua_api.lua (Test 8) |
| **MobileNet** | [MODEL_ARCHITECTURES.md](MODEL_ARCHITECTURES.md#mobilenet) | test_lua_api.lua (Test 9) |

---

## 🔍 Recherche par Mot-Clé

### Performance

- **Benchmarks**: [README.md § Performance](../README.md#-performance)
- **Optimisations**: [HARDWARE_OPTIMIZATIONS.md](HARDWARE_OPTIMIZATIONS.md)
- **Layer ops**: [LAYER_OPERATIONS.md](LAYER_OPERATIONS.md)

### API

- **Référence Lua**: [LUA_API.md](LUA_API.md)
- **Exemples**: [scripts/](../scripts/)
- **C++ API**: [TECHNICAL_SPECS.md § Spécifications C++](TECHNICAL_SPECS.md#spécifications-c)

### Architectures


- **Liste complète**: [MODEL_ARCHITECTURES.md](MODEL_ARCHITECTURES.md)
- **Implémentations**: [src/Models/ModelArchitectures.hpp](../src/Models/ModelArchitectures.hpp)
- **Tests**: [test_lua_api.lua](../scripts/test_lua_api.lua)

### Hardware

- **Détection**: [HARDWARE_OPTIMIZATIONS.md § Détection](HARDWARE_OPTIMIZATIONS.md)
- **AVX2/FMA**: [LAYER_OPERATIONS.md](LAYER_OPERATIONS.md)
- **Benchmarks**: [TECHNICAL_SPECS.md § Performance](TECHNICAL_SPECS.md#performance)

---

## 📦 Structure Complète

```txt
mimir-framework/
├── 📄 README.md                    (16K)   Vue d'ensemble
├── 📄 CHANGELOG.md                 (5.6K)  Historique versions
├── 📄 CONTRIBUTING.md              (6.9K)  Guide contribution
├── 📄 VERSION                      (6)     Version actuelle: 2.0.0
│
├── 📁 docs/                                Documentation
│   ├── 📄 README.md                (5.0K)  Index documentation
│   ├── 📄 QUICKSTART.md            (7.0K)  Démarrage rapide
│   ├── 📄 INSTALLATION.md          (9.7K)  Installation
│   ├── 📄 LUA_API.md               (14K)   API Lua complète
│   ├── 📄 API_LUA.md               (13K)   API Lua alternative
│   ├── 📄 MODEL_ARCHITECTURES.md   (9.2K)  8 architectures
│   ├── 📄 ARCHITECTURE.md          (19K)   Architecture interne
│   ├── 📄 TECHNICAL_SPECS.md       (21K)   Spécifications complètes
│   ├── 📄 HARDWARE_OPTIMIZATIONS.md (14K)  Optimisations hardware
│   ├── 📄 LAYER_OPERATIONS.md      (11K)   Opérations layers
│   └── 📄 ROADMAP.md               (-)     Roadmap futures features
│
├── 📁 scripts/                             Exemples Lua
│   ├── 📄 test_lua_api.lua         (16K)   Tests complets (10 tests)
│   ├── 📄 example_simple.lua       (5.3K)  Exemple minimal
│   ├── 📄 example_gpt.lua          (1.6K)  Génération texte
│   ├── 📄 example_training.lua     (4.2K)  Boucle training
│   └── 📄 example_layer_ops.lua    (8.1K)  Démo opérations
│
├── 📁 src/                                 Code source
│   ├── main.cpp                            Point d'entrée
│   ├── Model.hpp/cpp                       Classe Model
│   ├── LuaScripting.hpp/cpp                Bindings Lua
│   ├── tensors.hpp/cpp                     Système tenseurs
│   ├── Autograd.hpp                        Backpropagation
│   ├── Layers.hpp                          Définitions layers
│   ├── SIMD_Ops.hpp                        Optimisations SIMD
│   ├── Tokenizer.hpp/cpp                   BPE Tokenizer
│   ├── Encoder.hpp/cpp                     Encoder utilities
│   ├── Visualizer.hpp/cpp                  Visualisation
│   ├── Sha256.hpp/cpp                      Hashing
│   ├── Models/
│   │   └── ModelArchitectures.hpp          8 architectures
│   └── include/
│       └── json.hpp                        nlohmann/json
│
├── 📁 checkpoints/                         Modèles sauvegardés
│   └── epoch_*/                            Checkpoints par epoch
│
├── 📁 bin/
│   └── mimir                               Binary compilé (1.4MB)
│
└── Makefile                                Build system
```

**Total documentation**: ~135K (10 fichiers)  
**Total scripts**: ~51K (5 fichiers)  
**Total code source**: ~15,000 lignes C++

---

## 💡 Workflows Recommandés

### 1. Premier Modèle (Débutant)

```txt
1. README.md - Vue d'ensemble
2. QUICKSTART.md - Guide rapide
3. example_simple.lua - Copier et modifier
4. ./bin/mimir --lua my_model.lua - Exécuter
```

### 2. Créer une Architecture (Intermédiaire)

```
1. MODEL_ARCHITECTURES.md - Choisir architecture
2. LUA_API.md - Consulter API
3. test_lua_api.lua - Voir exemples
4. Créer script personnalisé
```

### 3. Optimiser Performance (Avancé)

```
1. HARDWARE_OPTIMIZATIONS.md - Comprendre optimisations
2. LAYER_OPERATIONS.md - Détails opérations
3. TECHNICAL_SPECS.md - Benchmarks
4. Profiler et optimiser
```

### 4. Contribuer au Projet (Expert)

```
1. CONTRIBUTING.md - Guide contribution
2. ARCHITECTURE.md - Comprendre internals
3. TECHNICAL_SPECS.md - Spécifications
4. Coder et PR
```

---

## 🆘 FAQ Documentation

**Q: Par où commencer?**  
A: [README.md](../README.md) puis [QUICKSTART.md](QUICKSTART.md)

**Q: Comment utiliser l'API Lua?**  
A: [LUA_API.md](LUA_API.md) - référence complète

**Q: Quelle architecture choisir?**  
A: [MODEL_ARCHITECTURES.md](MODEL_ARCHITECTURES.md) - 8 architectures comparées

**Q: Comment optimiser les performances?**  
A: [HARDWARE_OPTIMIZATIONS.md](HARDWARE_OPTIMIZATIONS.md) + [LAYER_OPERATIONS.md](LAYER_OPERATIONS.md)

**Q: Exemples de code?**  
A: [scripts/](../scripts/) - 5 scripts d'exemple

**Q: Spécifications techniques complètes?**  
A: [TECHNICAL_SPECS.md](TECHNICAL_SPECS.md) - 21K de documentation technique

---

## 📊 Statistiques Documentation

| Catégorie | Fichiers | Taille Totale |
|-----------|----------|---------------|
| **Documentation** | 10 | ~135K |
| **Scripts Lua** | 5 | ~51K |
| **Guides** | 4 | ~39K |
| **Total** | **19** | **~225K** |

**Couverture**:
- ✅ API Lua: 100% (60+ fonctions documentées)
- ✅ Architectures: 100% (8/8 documentées)
- ✅ Optimisations: 100% (toutes expliquées)
- ✅ Exemples: 100% (tous les cas d'usage)

---

## 🔗 Liens Rapides

### Documentation Essentielle

- [README](../README.md) - Vue d'ensemble
- [Quickstart](QUICKSTART.md) - Démarrage rapide
- [API Lua](LUA_API.md) - Référence API
- [Architectures](MODEL_ARCHITECTURES.md) - 8 architectures

### Code Source

- [GitHub Repository](https://github.com/brit45/mimir-framework)
- [src/Model.hpp](../src/Model.hpp) - Classe Model
- [src/Models/ModelArchitectures.hpp](../src/Models/ModelArchitectures.hpp) - Architectures
- [src/LuaScripting.cpp](../src/LuaScripting.cpp) - Bindings Lua

### Exemples

- [test_lua_api.lua](../scripts/test_lua_api.lua) - Tests complets
- [example_simple.lua](../scripts/example_simple.lua) - Exemple minimal
- [example_gpt.lua](../scripts/example_gpt.lua) - GPT

---

<div align="center">

**Documentation mise à jour: 19 décembre 2025**  
**Framework version: 2.0.0**

[⬆ Retour en haut](#-index-complet---documentation-mímir-framework-v20)

</div>
