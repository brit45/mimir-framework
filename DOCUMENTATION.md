# 📚 Guide de la Documentation - Mímir Framework v2.0

**Navigation rapide vers toute la documentation du framework**

---

## 🎯 Démarrage Rapide

1. **[README.md](README.md)** - Commencez ici ! Vue d'ensemble complète du framework
2. **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - Créez votre premier modèle en 5 minutes
3. **[docs/INSTALLATION.md](docs/INSTALLATION.md)** - Guide d'installation détaillé

## 📖 Documentation Complète

### Index et Navigation
- **[docs/INDEX.md](docs/INDEX.md)** - Index complet de toute la documentation (recommandé)
- **[docs/README.md](docs/README.md)** - Table des matières principale

### API et Référence
- **[docs/LUA_API.md](docs/LUA_API.md)** - Référence complète de l'API Lua (60+ fonctions)
- **[docs/MODEL_ARCHITECTURES.md](docs/MODEL_ARCHITECTURES.md)** - Détails des 8 architectures

### Architecture Technique
- **[docs/TECHNICAL_SPECS.md](docs/TECHNICAL_SPECS.md)** - Spécifications techniques complètes (21K)
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Architecture interne du framework
- **[docs/HARDWARE_OPTIMIZATIONS.md](docs/HARDWARE_OPTIMIZATIONS.md)** - Optimisations AVX2/FMA/F16C/BMI2
- **[docs/LAYER_OPERATIONS.md](docs/LAYER_OPERATIONS.md)** - Opérations de layers optimisées

## 💻 Exemples et Scripts

### Scripts Lua
- **[scripts/test_lua_api.lua](scripts/test_lua_api.lua)** - Tests complets de l'API (10 tests)
- **[scripts/example_simple.lua](scripts/example_simple.lua)** - Exemple minimal de Transformer
- **[scripts/example_gpt.lua](scripts/example_gpt.lua)** - Génération de texte avec GPT
- **[scripts/example_training.lua](scripts/example_training.lua)** - Boucle d'entraînement complète

### Exécution
```bash
./bin/mimir --lua scripts/example_simple.lua
./bin/mimir --demo transformer
./bin/mimir --help
```

## 🛠️ Développement

- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Guide de contribution au projet
- **[CHANGELOG.md](CHANGELOG.md)** - Historique des versions et changements
- **[docs/ROADMAP.md](docs/ROADMAP.md)** - Roadmap et features à venir

## 📊 Structure de la Documentation

```
Documentation/
│
├── Niveau 1: Démarrage (README, Quickstart, Installation)
├── Niveau 2: API et Référence (Lua API, Architectures)
├── Niveau 3: Technique (Specs, Architecture, Optimisations)
└── Niveau 4: Développement (Contributing, Changelog, Roadmap)
```

**Total**: ~135K de documentation dans 10 fichiers principaux

## 🎓 Parcours Recommandés

### Pour Débutants
1. README.md → docs/QUICKSTART.md → scripts/example_simple.lua

### Pour Développeurs
1. docs/LUA_API.md → docs/MODEL_ARCHITECTURES.md → scripts/test_lua_api.lua

### Pour Experts
1. docs/ARCHITECTURE.md → docs/TECHNICAL_SPECS.md → docs/HARDWARE_OPTIMIZATIONS.md

## 🔗 Liens Utiles

- **Index Complet**: [docs/INDEX.md](docs/INDEX.md)
- **GitHub**: https://github.com/brit45/mimir-framework
- **Issues**: https://github.com/brit45/mimir-framework/issues

---

**Version**: 2.0.0  
**Dernière mise à jour**: 19 décembre 2025

**Navigation**: Consultez [docs/INDEX.md](docs/INDEX.md) pour une navigation détaillée par thème.
