# 📚 Documentation Principale - Mímir Framework

**Version:** 2.3.0  
**Date:** 28 décembre 2025  
**Statut:** Production Ready ✅

---

## 🚀 Points d'Entrée Rapides

### Pour Débuter
- **[README.md](README.md)** - Vue d'ensemble du projet et quickstart
- **[docs/04-Architecture-Internals/TECHNICAL_STATUS.md](docs/04-Architecture-Internals/TECHNICAL_STATUS.md)** - État technique détaillé
- **[docs/00-INDEX.md](docs/00-INDEX.md)** - Index complet de la documentation

### Pour les Développeurs
- **[CHANGELOG.md](CHANGELOG.md)** - Historique des versions
- **[docs/03-API-Reference/LAYERS_REFERENCE.md](docs/03-API-Reference/LAYERS_REFERENCE.md)** - Référence des 67 layers
- **[docs/03-API-Reference/SAVE_LOAD.md](docs/03-API-Reference/SAVE_LOAD.md)** - Guide sérialisation
- **[docs/06-Contributing/HOWTO_ADD_LAYER.md](docs/06-Contributing/HOWTO_ADD_LAYER.md)** - Ajouter un layer

### Pour les Contributeurs
- **[docs/06-Contributing/](docs/06-Contributing/)** - Guides de contribution
- **[LICENSE](LICENSE)** - Licence GPL-2.0

---

## 📦 Versions et Fonctionnalités

### v2.3.0 (Actuel) - Décembre 2025
**Thème:** Sérialisation Moderne & Legacy Cleanup

**Nouveautés:**
- ✅ Module de sérialisation complet (SafeTensors, RawFolder, DebugJson)
- ✅ Compatibilité HuggingFace (import/export SafeTensors)
- ✅ Suppression complète du code legacy (~300 lignes retirées)
- ✅ Architecture optimisée `layer_weight_blocks` (un tenseur par couche)
- ✅ Checksums SHA256 et validation d'intégrité
- ✅ Performance: 850 MB/s write, 1200 MB/s read

**Documents:**
- [docs/Archives/LEGACY_CLEANUP_COMPLETE.md](docs/Archives/LEGACY_CLEANUP_COMPLETE.md) - Détails du cleanup
- [docs/04-Architecture-Internals/SERIALIZATION_COMPLETE.md](docs/04-Architecture-Internals/SERIALIZATION_COMPLETE.md) - Système de sérialisation
- [docs/03-API-Reference/SAVE_LOAD.md](docs/03-API-Reference/SAVE_LOAD.md) - Guide utilisateur complet

### v2.1.0 - Janvier 2025
**Thème:** Système Unifié des Layers

**Nouveautés:**
- ✅ Architecture refactorisée avec enum-based dispatch (10-40× plus rapide)
- ✅ 67 layer types définis, 19 fonctionnels avec optimisations AVX2
- ✅ Type-safe, pas de silent fallback
- ✅ Documentation complète (1600+ lignes)
- ✅ Scripts réorganisés en 6 catégories

**Documents:**
- [docs/Archives/MISSION_COMPLETE.md](docs/Archives/MISSION_COMPLETE.md) - Mission "nettoyage" accomplie
- [docs/Archives/REFONTE_COMPLETE.md](docs/Archives/REFONTE_COMPLETE.md) - Architecture unifiée
- [docs/03-API-Reference/LAYERS_REFERENCE.md](docs/03-API-Reference/LAYERS_REFERENCE.md) - Référence complète
- [docs/05-Advanced/PERFORMANCE.md](docs/05-Advanced/PERFORMANCE.md) - Benchmarks et optimisations

### v2.0.0 - Décembre 2024
**Thème:** Sécurité Mémoire & Threading Asynchrone

**Nouveautés:**
- ✅ Limite stricte 10 GB avec protection OOM
- ✅ Compression LZ4 automatique (~50% économie)
- ✅ Threading asynchrone pour monitoring/visualisation
- ✅ Support Vulkan Compute (optionnel)
- ✅ 8 architectures pré-définies (UNet, VAE, Transformer, etc.)

**Documents:**
- [docs/02-User-Guide/10-Memory-Best-Practices.md](docs/02-User-Guide/10-Memory-Best-Practices.md)
- [docs/Archives/RELEASE_NOTES_2.1.0.md](docs/Archives/RELEASE_NOTES_2.1.0.md)

---

## 🗂️ Organisation de la Documentation

### Documentation Principale (`/`)
```
README.md                    # Vue d'ensemble et quickstart
TECHNICAL_STATUS.md          # État technique détaillé
CHANGELOG.md                 # Historique des versions
LICENSE                      # Licence GPL-2.0
```

### Documentation Organisée (`docs/`)
```
docs/
├── 00-INDEX.md                      # Index principal
├── 01-Getting-Started/              # Démarrage (6 guides)
├── 02-User-Guide/                   # Guide utilisateur (10 guides)
├── 03-API-Reference/                # API complète (114 fonctions)
├── 04-Architecture-Internals/       # Architecture interne (11 docs)
├── 05-Advanced/                     # Sujets avancés (11 guides)
└── 06-Contributing/                 # Contribution (5 guides)
```

### Documents de Référence (`docs/`)
```
LAYERS_REFERENCE.md          # 67 layers avec détails
SAVE_LOAD.md                 # Sérialisation complète
HOWTO_ADD_LAYER.md          # Guide développeur layers
MULTI_INPUT_SUPPORT.md      # Branches et multi-inputs
PERFORMANCE.md              # Benchmarks et optimisations
OP_COVERAGE.md              # Audit des opérations
```

### Documents Historiques (Référence)
```
LEGACY_CLEANUP_COMPLETE.md   # Cleanup v2.3.0
SERIALIZATION_COMPLETE.md    # Sérialisation v2.3.0
MISSION_COMPLETE.md          # Refonte layers v2.1.0
RELEASE_NOTES_2.1.0.md       # Notes v2.1.0
```

---

## 🎯 Navigation par Cas d'Usage

### Je veux créer mon premier modèle
1. Lire [README.md](README.md#-quickstart)
2. Suivre [docs/01-Getting-Started/01-Quick-Start.md](docs/01-Getting-Started/01-Quick-Start.md)
3. Utiliser le template [scripts/templates/template_new_model.lua](scripts/templates/template_new_model.lua)

### Je veux comprendre l'architecture du framework
1. Lire [TECHNICAL_STATUS.md](TECHNICAL_STATUS.md)
2. Explorer [docs/04-Architecture-Internals/](docs/04-Architecture-Internals/)
3. Consulter [docs/LAYERS_REFERENCE.md](docs/LAYERS_REFERENCE.md)

### Je veux sauvegarder/charger des modèles
1. Lire [docs/SAVE_LOAD.md](docs/SAVE_LOAD.md)
2. Voir [SERIALIZATION_COMPLETE.md](SERIALIZATION_COMPLETE.md) pour les détails techniques
3. Consulter [docs/02-User-Guide/08-Save-Load.md](docs/02-User-Guide/08-Save-Load.md)

### Je veux optimiser les performances
1. Lire [docs/PERFORMANCE.md](docs/PERFORMANCE.md)
2. Explorer [docs/04-Architecture-Internals/07-Hardware-Optimizations.md](docs/04-Architecture-Internals/07-Hardware-Optimizations.md)
3. Consulter [docs/02-User-Guide/10-Memory-Best-Practices.md](docs/02-User-Guide/10-Memory-Best-Practices.md)

### Je veux ajouter un nouveau layer
1. Lire [docs/HOWTO_ADD_LAYER.md](docs/HOWTO_ADD_LAYER.md)
2. Consulter [docs/LAYERS_REFERENCE.md](docs/LAYERS_REFERENCE.md) pour les exemples
3. Voir [docs/OP_COVERAGE.md](docs/OP_COVERAGE.md) pour le statut actuel

### Je veux contribuer au projet
1. Lire [docs/06-Contributing/01-Contributing-Guide.md](docs/06-Contributing/01-Contributing-Guide.md)
2. Consulter [CHANGELOG.md](CHANGELOG.md) pour le contexte
3. Utiliser [docs/06-Contributing/02-Pull-Request-Template.md](docs/06-Contributing/02-Pull-Request-Template.md)

---

## 🔧 Outils et Scripts

### Scripts Utiles
```bash
# Compilation et tests
make                                 # Build optimisé
make test                           # Build et tests
make clean                          # Nettoyage

# Vérifications
./scripts/quick_check.sh            # Vérification rapide
./scripts/check_ready_for_strict.sh # Vérif pré-migration
./scripts/generate_op_coverage.sh   # Génération coverage

# Exemples et démos
./bin/mimir --demo transformer      # Démo Transformer
./bin/mimir --lua script.lua        # Exécuter script Lua
```

### Templates
```
scripts/templates/
├── template_new_model.lua          # Template modèle de base
├── template_training_loop.lua      # Template boucle entraînement
└── template_architecture.lua       # Template architecture custom
```

---

## 📊 Statistiques du Projet

### Code Base
- **Lignes de code C++:** ~35,000 lignes
- **Modules Lua:** 13 modules, 114 fonctions
- **Tests:** 18 suites de tests
- **Documentation:** 80+ fichiers markdown

### Performance
- **Dispatch layers:** 10-40× plus rapide (enum vs if/else)
- **Linear AVX2:** 2.6× speedup
- **ReLU AVX2:** 5.2× speedup
- **OpenMP scaling:** 8-10× sur 12 threads
- **Sérialisation:** 850 MB/s write, 1200 MB/s read

### Architectures
- **Layers définis:** 67 types
- **Layers fonctionnels:** 19 avec optimisations AVX2
- **Architectures pré-définies:** 8 (UNet, VAE, Transformer, etc.)
- **Formats de sérialisation:** 3 (SafeTensors, RawFolder, DebugJson)

---

## ❓ FAQ

### Où commencer ?
Commencez par [README.md](README.md), puis suivez [docs/01-Getting-Started/01-Quick-Start.md](docs/01-Getting-Started/01-Quick-Start.md).

### Comment sauvegarder un modèle ?
Utilisez le nouveau système de sérialisation. Voir [docs/SAVE_LOAD.md](docs/SAVE_LOAD.md).

### Où sont les exemples ?
Dans `scripts/examples/` et `scripts/demos/`. Utilisez les templates dans `scripts/templates/`.

### Comment migrer du code legacy ?
Consultez [LEGACY_CLEANUP_COMPLETE.md](LEGACY_CLEANUP_COMPLETE.md) pour les fonctions obsolètes et leurs remplacements.

### Quelles sont les performances CPU ?
Voir [docs/PERFORMANCE.md](docs/PERFORMANCE.md) pour les benchmarks détaillés. En résumé : 2.5-4× speedup global sur training complet.

### Le framework supporte-t-il les GPU ?
Mímir est volontairement CPU-only. Support Vulkan Compute optionnel pour accélération sur certaines opérations.

### Quelle est la limite mémoire ?
10 GB par défaut (configurable). Voir [docs/02-User-Guide/10-Memory-Best-Practices.md](docs/02-User-Guide/10-Memory-Best-Practices.md).

---

## 📞 Support et Contact

- **Issues:** [GitHub Issues](https://github.com/votre-repo/mimir/issues)
- **Discussions:** [GitHub Discussions](https://github.com/votre-repo/mimir/discussions)
- **Email:** support@mimir-framework.org

---

**Dernière mise à jour:** 28 décembre 2025  
**Version:** 2.3.0
