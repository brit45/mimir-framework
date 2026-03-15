# 📚 Guide de la Documentation - Mímir Framework v2.0

**Navigation rapide vers toute la documentation du framework**

---

## 🎯 Démarrage Rapide

1. **[README.md](../../README.md)** - Commencez ici ! Vue d'ensemble complète du framework
2. **[docs/01-Getting-Started/01-Quick-Start.md](../01-Getting-Started/01-Quick-Start.md)** - Créez votre premier modèle en 5 minutes
3. **[docs/01-Getting-Started/02-Installation.md](../01-Getting-Started/02-Installation.md)** - Guide d'installation détaillé

## 📖 Documentation Complète

### Index et Navigation
- **[docs/00-INDEX.md](../00-INDEX.md)** - Index complet de toute la documentation (recommandé)
- **README principal**: [README.md](../../README.md)

### API et Référence
- **[docs/03-API-Reference/00-API-Complete.md](../03-API-Reference/00-API-Complete.md)** - Référence complète de l'API Lua (114 fonctions)
- **[docs/02-User-Guide/03-Predefined-Architectures.md](../02-User-Guide/03-Predefined-Architectures.md)** - Détails des 9 architectures

### Architecture Technique
- **[docs/04-Architecture-Internals/03-Technical-Specifications.md](../04-Architecture-Internals/03-Technical-Specifications.md)** - Spécifications techniques complètes
- **[docs/04-Architecture-Internals/01-System-Architecture.md](../04-Architecture-Internals/01-System-Architecture.md)** - Architecture interne du framework
- **[docs/04-Architecture-Internals/07-Hardware-Optimizations.md](../04-Architecture-Internals/07-Hardware-Optimizations.md)** - Optimisations AVX2/FMA/F16C/BMI2
- **[docs/05-Advanced/04-Layer-Operations.md](../05-Advanced/04-Layer-Operations.md)** - Opérations de layers optimisées

## 💻 Exemples et Scripts

### Scripts Lua
- **[scripts/test_lua_api.lua](../../scripts/tests/test_lua_api.lua)** - Tests complets de l'API (10 tests)
- **[scripts/example_simple.lua](../../scripts/examples/example_simple.lua)** - Exemple minimal de Transformer
- **[scripts/example_gpt.lua](../../scripts/examples/example_gpt.lua)** - Génération de texte avec GPT
- **[scripts/example_training.lua](../../scripts/examples/example_training.lua)** - Boucle d'entraînement complète

### Exécution
```bash
./bin/mimir --lua scripts/example_simple.lua
./bin/mimir --demo transformer
./bin/mimir --help
```

## 🛠️ Développement

- **[CONTRIBUTING.md](../../build/_deps/googletest-src/CONTRIBUTING.md)** - Guide de contribution au projet
- **[CHANGELOG.md](../../CHANGELOG.md)** - Historique des versions et changements
- **[docs/06-Contributing/05-Roadmap.md](05-Roadmap.md)** - Roadmap et features à venir

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

- **Index Complet**: [docs/00-INDEX.md](../00-INDEX.md)
- **GitHub**: https://github.com/brit45/mimir-framework
- **Issues**: https://github.com/brit45/mimir-framework/issues

---

**Version**: 2.0.0  
**Dernière mise à jour**: 27 décembre 2025

**Navigation**: Consultez [docs/00-INDEX.md](../00-INDEX.md) pour une navigation détaillée par thème.
