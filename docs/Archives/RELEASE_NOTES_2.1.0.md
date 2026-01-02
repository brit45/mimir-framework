# Version 2.1.0 - Release Notes

**Date**: 27 décembre 2025  
**Type**: Minor Release

## 🎯 Objectifs de cette version

Cette version se concentre sur l'organisation du projet et l'amélioration de la qualité de la documentation.

## ✨ Nouveautés principales

### 🗂️ Réorganisation du dossier `scripts/`

Le dossier scripts a été complètement restructuré en catégories claires:

```
scripts/
├── README.md         # Documentation complète
├── demos/            # 10 démonstrations d'architectures
├── examples/         # 5 exemples d'utilisation
├── tests/            # 19 scripts de validation
├── benchmarks/       # 2 scripts de performance
├── training/         # 3 scripts d'entraînement
└── templates/        # 1 template pour développement
```

**Total**: 40 scripts organisés

### 📚 Corrections de documentation

- **33 corrections** de liens cassés dans 8 fichiers
- **Architecture count** corrigé: 8 → 9 (ajout Flux)
- **Function count** corrigé: 117 → 114 (compte exact)
- **Table des matières** complétée dans API Complete
- **Navigation** fluide sans erreurs 404

### ✅ Synchronisation API confirmée

- **114 fonctions** validées sur **13 modules**
- Synchronisation 100% entre `mimir-api.lua` et `src/LuaScripting.cpp`
- Scripts de validation créés dans `tools/`
- EmmyLua annotations maintenues et vérifiées

## 📊 Statistiques

### Changements de version

- **VERSION**: 2.0.0 → 2.1.0
- **CMakeLists.txt**: 2.0 → 2.1
- **README.md**: Ajout section v2.1
- **mimir-api.lua**: Mise à jour header et features
- **Documentation**: Mise à jour de tous les fichiers de référence

### Organisation des fichiers

- **131 fichiers modifiés**
- **22 501 insertions**
- **3 797 suppressions**
- **40 scripts Lua** réorganisés
- **8 fichiers** de documentation corrigés

## 🔧 Améliorations techniques

### Scripts organisés par catégorie

| Catégorie | Nombre | Description |
|-----------|--------|-------------|
| demos | 10 | Démonstrations des 9 architectures + features avancées |
| examples | 5 | Exemples pratiques d'utilisation de l'API |
| tests | 19 | Scripts de validation et tests |
| benchmarks | 2 | Tests de performance |
| training | 3 | Scripts d'entraînement complets |
| templates | 1 | Template pour nouveaux modèles |

### Documentation navigable

Tous les liens internes fonctionnent correctement:
- 23 liens cassés corrigés
- 7 statistiques inexactes mises à jour
- 3 modules manquants ajoutés à la table des matières

## 🚀 Migration depuis v2.0.0

Aucun changement breaking. La structure des scripts a changé:

**Ancien**:
```bash
./bin/mimir scripts/test_lua_api.lua
```

**Nouveau**:
```bash
./bin/mimir scripts/tests/test_lua_api.lua
```

## ✅ Validation

Tous les tests passent:

```bash
# Test de synchronisation API
./tools/verify_api_sync.sh
# ✅ SYNCHRONISATION COMPLÈTE - 114 fonctions validées

# Comptage des fonctions
./tools/count_lua_functions.sh
# ✅ 114 = 114

# Liste des fonctions
./tools/list_lua_api.sh
# ✅ 13 modules listés correctement
```

## 📝 Fichiers ajoutés

### Scripts
- `scripts/README.md` - Documentation complète de l'organisation
- Tous les scripts dans leurs nouvelles catégories respectives

### Documentation
- Organisation complète avec 6 dossiers thématiques
- Archives pour anciens documents

### Outils
- Scripts de validation dans `tools/`

## 🔍 Détails des corrections

### Liens corrigés (23)
- docs/00-INDEX.md: 7 liens
- docs/06-Contributing/*.md: 14 liens
- docs/03-API-Reference/00-API-Complete.md: 2 ajouts

### Statistiques corrigées (7)
- Architectures: 8 → 9 (2 fichiers)
- Fonctions: 117 → 114 (5 fichiers)

## 🎓 Prochaines étapes

1. Tester l'autocomplétion avec VS Code + Lua extension
2. Créer des snippets pour workflows courants
3. Ajouter CI/CD avec validation automatique
4. Générer documentation HTML depuis EmmyLua
5. Enrichir les exemples avec cas d'usage réels

## 📖 Documentation

- [Guide complet](docs/00-INDEX.md)
- [Guide de démarrage](docs/01-Getting-Started/01-Quick-Start.md)
- [API Lua complète](docs/03-API-Reference/00-API-Complete.md)
- [Scripts README](scripts/README.md)
- [CHANGELOG](CHANGELOG.md)

---

**Framework**: Mímir v2.1.0  
**License**: MIT  
**Author**: bri45  
**Date**: 27 décembre 2025
