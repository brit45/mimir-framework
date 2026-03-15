# 🎯 DÉMARRER AVEC MÍMIR FRAMEWORK

**Version:** 2.3.0  
**Date:** 28 décembre 2025

Bienvenue dans **Mímir Framework** ! Voici votre guide de démarrage rapide.

---

## 📍 Vous êtes ici...

```
Mímir Framework v2.3.0
├── README.md              ← Vue d'ensemble du projet
├── DOCS_INDEX.md          ← Index de toute la documentation
├── TECHNICAL_STATUS.md    ← État technique détaillé
└── docs/                  ← Documentation complète
```

---

## 🚀 Démarrage en 3 Étapes

### 1️⃣ Lire la Vue d'Ensemble (5 min)

**Fichier:** [README.md](../../README.md)

**Ce que vous allez apprendre:**
- Qu'est-ce que Mímir ?
- Pourquoi CPU-only ?
- Fonctionnalités principales
- Comment installer
- Premier exemple de code

**Action:** Lisez maintenant !

---

### 2️⃣ Compiler et Tester (5 min)

```bash
# Installation des dépendances (Ubuntu/Debian)
sudo apt-get install -y g++ make liblua5.3-dev libsfml-dev liblz4-dev

# Compilation
make -j$(nproc)

# Test rapide
LUA_PATH='scripts/modules/?.lua;;' ./bin/mimir --lua scripts/demos/demo_unet.lua
```

**Résultat attendu:** Le binaire `bin/mimir` est créé (~2.2 MB) et une démo Transformer s'exécute.

---

### 3️⃣ Premier Modèle Lua (10 min)

**Fichier:** Créer `my_first_model.lua`

```lua
-- Configuration mémoire (OBLIGATOIRE)
Mimir.Allocator.configure({
    max_ram_gb = 10.0,
    enable_compression = true
})

-- Créer un modèle simple via le registre d'architectures (v2.3+)
local cfg, err = Mimir.Architectures.default_config("transformer")
if not cfg then
    print("❌ default_config failed: " .. tostring(err))
    os.exit(1)
end

cfg.vocab_size = 10000
cfg.d_model = 256
cfg.num_layers = 4
cfg.num_heads = 8

local ok, create_err = Mimir.Model.create("transformer", cfg)
if not ok then
    print("❌ create failed: " .. tostring(create_err))
    os.exit(1)
end

-- Allouer et initialiser
local success, params = model.allocate_params()
if not success then
    print("❌ Modèle trop grand!")
    os.exit(1)
end

print(string.format("✅ Paramètres: %d (%.1f MB)", 
    params, params * 4 / 1024 / 1024))

model.init_weights("xavier", 42)

-- Sauvegarder avec nouveau système v2.3
Mimir.Serialization.save("my_model.safetensors", "safetensors")
print("✅ Modèle sauvegardé!")
```

**Exécuter:**
```bash
./bin/mimir --lua my_first_model.lua
```

---

## 📚 Prochaines Étapes

### Explorer la Documentation

**Fichier principal:** [DOCS_INDEX.md](../../DOCS_INDEX.md)

Ce fichier contient TOUT :
- Navigation par cas d'usage
- Index complet de la documentation
- Guides de référence rapide
- FAQ

### Apprendre les Bases

**Dossier:** [docs/01-Getting-Started/](docs/01-Getting-Started/)

Guides disponibles:
1. ✅ [Quick Start](01-Quick-Start.md) - Premier modèle en 5 minutes
2. 📦 [Installation](02-Installation.md) - Guide complet
3. 🛠️ [IDE Setup](03-IDE-Setup.md) - VSCode, CLion, Vim
4. ❓ [Why CPU-Only](04-Why-CPU-Only.md) - Philosophie
5. 🏗️ [Build Instructions](05-Build-Instructions.md) - Compilation détaillée
6. 🔧 [Rebuild and Test](06-Rebuild-And-Test.md) - Tests

### Créer des Modèles Avancés

**Dossier:** [docs/02-User-Guide/](docs/02-User-Guide/)

Guides avancés:
- 🤖 [Model Creation](../02-User-Guide/02-Model-Creation.md)
- 🎨 [Predefined Architectures](../02-User-Guide/03-Predefined-Architectures.md)
- 🏋️ [Training](../02-User-Guide/06-Training.md)
- 💾 [Save/Load](../02-User-Guide/08-Save-Load.md)

### Consulter les Références

**Fichiers clés:**
- 📘 [docs/03-API-Reference/](docs/03-API-Reference/) - 114 fonctions documentées
- 📚 [docs/LAYERS_REFERENCE.md](../03-API-Reference/LAYERS_REFERENCE.md) - 67 layers
- 💾 [docs/SAVE_LOAD.md](docs/01-Getting-Started/05-Save-Load.md) - Système de sérialisation v2.3
- ⚡ [docs/PERFORMANCE.md](docs/05-Advanced/PERFORMANCE.md) - Benchmarks

---

## 🎓 Parcours d'Apprentissage Recommandé

### Débutant (1-2 heures)
1. ✅ Lire [README.md](../../README.md)
2. ✅ Compiler et tester
3. ✅ Créer un premier modèle
4. ✅ Parcourir [docs/01-Getting-Started/](docs/01-Getting-Started/)

### Intermédiaire (1 journée)
1. ✅ Lire [TECHNICAL_STATUS.md](../04-Architecture-Internals/TECHNICAL_STATUS.md)
2. ✅ Explorer [docs/02-User-Guide/](docs/02-User-Guide/)
3. ✅ Essayer les templates dans [scripts/templates/](scripts/templates/)
4. ✅ Créer une architecture custom

### Avancé (1 semaine)
1. ✅ Étudier [docs/04-Architecture-Internals/](docs/04-Architecture-Internals/)
2. ✅ Lire [docs/HOWTO_ADD_LAYER.md](../06-Contributing/HOWTO_ADD_LAYER.md)
3. ✅ Explorer le code source dans [src/](src/)
4. ✅ Contribuer au projet

---

## 🔑 Documents Essentiels

| Document | Contenu | Quand le lire |
|----------|---------|---------------|
| [README.md](../../README.md) | Vue d'ensemble | **Premier document** |
| [DOCS_INDEX.md](../../DOCS_INDEX.md) | Index complet | Navigation |
| [TECHNICAL_STATUS.md](../04-Architecture-Internals/TECHNICAL_STATUS.md) | État technique | Comprendre le projet |
| [CHANGELOG.md](../../CHANGELOG.md) | Historique | Évolution du projet |
| [docs/SAVE_LOAD.md](docs/01-Getting-Started/05-Save-Load.md) | Sérialisation v2.3 | Sauvegarder/charger |
| [docs/LAYERS_REFERENCE.md](../03-API-Reference/LAYERS_REFERENCE.md) | 67 layers | Référence layers |

---

## 🆘 Besoin d'Aide ?

### Documentation
- **Index complet:** [DOCS_INDEX.md](../../DOCS_INDEX.md)
- **FAQ:** Voir [DOCS_INDEX.md#-faq](../../DOCS_INDEX.md#-faq)
- **Guides:** [docs/](docs/)

### Support
- **Issues:** [GitHub Issues](https://github.com/votre-repo/mimir/issues)
- **Discussions:** [GitHub Discussions](https://github.com/votre-repo/mimir/discussions)

### Exemples
- **Templates:** [scripts/templates/](scripts/templates/)
- **Démos:** [scripts/demos/](scripts/demos/)
- **Exemples:** [scripts/examples/](scripts/examples/)

---

## ✨ Nouveautés v2.3.0

### 💾 Sérialisation Moderne
- ✅ 3 formats: SafeTensors (production), RawFolder (debug), DebugJson (inspection)
- ✅ Compatible HuggingFace (import/export PyTorch/TensorFlow)
- ✅ Checksums SHA256, validation d'intégrité
- ✅ Performance: 850 MB/s write, 1200 MB/s read

**Guide complet:** [docs/SAVE_LOAD.md](docs/01-Getting-Started/05-Save-Load.md)

### 🧹 Legacy Cleanup
- ✅ ~300 lignes de code legacy supprimées
- ✅ Architecture optimisée `layer_weight_blocks`
- ✅ API moderne uniquement

**Détails:** [LEGACY_CLEANUP_COMPLETE.md](../Archives/LEGACY_CLEANUP_COMPLETE.md)

---

## 🎯 Objectifs de Mímir

**Vision:** Démocratiser l'IA en permettant à quiconque avec un CPU moderne de créer et entraîner des modèles performants.

**Philosophie:**
- 💰 Accessible (pas de GPU requis)
- 🌍 Universel (fonctionne partout)
- ⚡ Optimisé (AVX2, FMA, F16C)
- 🎯 Pratique (API Lua simple)

**Résultat:** Framework CPU-only haute performance, production-ready, avec 67 layers, 8 architectures, et API complète.

---

**Prêt à commencer ?** → Lisez [README.md](../../README.md) maintenant ! 🚀

**Dernière mise à jour:** 28 décembre 2025
