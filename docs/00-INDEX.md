# 📚 Mímir Framework - Documentation Complète

**Version:** 2.3.0  
**Licence:** GPL-2.0 (Open Source) / Commercial (avec licence)  
**Date:** Décembre 2025

Bienvenue dans la documentation complète du **Mímir Framework**, un framework de deep learning moderne, optimisé pour CPU, écrit en C++17 avec une API Lua intuitive.

---

## ⚡ NOUVEAU: Sérialisation Moderne v2.3

Le framework dispose maintenant d'un **système de sérialisation complet** avec 3 formats :

- 💾 **[SafeTensors](03-API-Reference/SAVE_LOAD.md#1-safetensors-production-)** - Format production compatible HuggingFace
- 🔍 **[RawFolder](03-API-Reference/SAVE_LOAD.md#2-rawfolder-debug--development-)** - Format debug avec checksums SHA256
- 📊 **[DebugJson](03-API-Reference/SAVE_LOAD.md#3-debugjson-inspection-)** - Dump JSON avec statistiques
- 🧹 **[Legacy Cleanup](Archives/LEGACY_CLEANUP_COMPLETE.md)** - Suppression code legacy (v2.3)
- 📖 **[Guide Complet](03-API-Reference/SAVE_LOAD.md)** - Documentation sérialisation
- 🚀 **[API Lua v2.3](03-API-Reference/API_LUA_UPDATE_v2.3.0.md)** - Nouvelle API Mimir.Serialization
- ✅ **[Validation API](03-API-Reference/SERIALIZATION_API_VALIDATED.md)** - Tests et validation

**Bénéfices :** Interopérabilité Python/PyTorch, validation d'intégrité, performance optimale (850 MB/s write).

---

## ⚡ Système Unifié des Layers v2.1

Le framework a été complètement refactorisé avec un **nouveau système unifié de layers** :

- 🎯 **[Vue d'Ensemble](Archives/MISSION_COMPLETE.md)** - Mission "nettoyage" accomplie
- 📋 **[Détails Techniques](Archives/REFONTE_COMPLETE.md)** - Architecture unifiée
- 📚 **[Référence des Layers](03-API-Reference/LAYERS_REFERENCE.md)** - 67 layers supportés
- 🔧 **[Guide Développeur](06-Contributing/HOWTO_ADD_LAYER.md)** - Ajouter un nouveau layer
- ⚡ **[Performance](05-Advanced/PERFORMANCE.md)** - Optimisations et benchmarks

**Bénéfices :** Dispatch 10-40x plus rapide, AVX2, OpenMP, type-safe, maintenable.

---

## 🎯 Navigation Rapide

### Pour Débuter (5-30 minutes)
- 🚀 **[Démarrage Rapide](01-Getting-Started/01-Quick-Start.md)** - Premier modèle en 5 minutes
- 📦 **[Installation](01-Getting-Started/02-Installation.md)** - Guide d'installation complet
- 🛠️ **[Configuration IDE](01-Getting-Started/03-IDE-Setup.md)** - VSCode, CLion, Vim
- ❓ **[Pourquoi CPU-only?](01-Getting-Started/04-Why-CPU-Only.md)** - Philosophie du framework
- 🏗️ **[Instructions de Build](01-Getting-Started/05-Build-Instructions.md)** - Compilation détaillée
- 🔧 **[Rebuild et Tests](01-Getting-Started/06-Rebuild-And-Test.md)** - Recompilation et validation
- 📖 **[Guide de Démarrage](01-Getting-Started/START_HERE.md)** - Introduction complète

### Guide Utilisateur (1-3 heures)
- 📖 **[Concepts Fondamentaux](02-User-Guide/01-Core-Concepts.md)** - Architecture et concepts clés
- 🤖 **[Création de Modèles](02-User-Guide/02-Model-Creation.md)** - Créer et configurer des modèles
- 🎨 **[Architectures Pré-définies](02-User-Guide/03-Predefined-Architectures.md)** - UNet, VAE, Transformer, etc.
- 🔤 **[Tokenization](02-User-Guide/04-Tokenization.md)** - Gestion du vocabulaire et BPE
- 💾 **[Gestion des Données](02-User-Guide/05-Data-Management.md)** - Datasets et préparation
- 🏋️ **[Entraînement](02-User-Guide/06-Training.md)** - Boucles d'entraînement et optimisation
- 🔮 **[Inférence](02-User-Guide/07-Inference.md)** - Génération et prédiction
- 💾 **[Sauvegarde/Chargement](02-User-Guide/08-Save-Load.md)** - Checkpoints et persistence
- 📊 **[Monitoring](02-User-Guide/09-Monitoring.md)** - Htop et visualisation
- 🛡️ **[Bonnes Pratiques Mémoire](02-User-Guide/10-Memory-Best-Practices.md)** - Gestion sécurisée de la RAM

### Référence API (consultation)
- 📘 **[API Complète](03-API-Reference/00-API-Complete.md)** - Documentation complète (117 fonctions)
- 🚀 **[Référence Rapide](03-API-Reference/01-API-Quick-Reference.md)** - Guide rapide par module
- 📚 **[Référence Layers](03-API-Reference/LAYERS_REFERENCE.md)** - 67 layers supportés
- 💾 **[Guide Sérialisation](03-API-Reference/SAVE_LOAD.md)** - SafeTensors, RawFolder, DebugJson
- 🔗 **[Support Multi-Input](03-API-Reference/MULTI_INPUT_SUPPORT.md)** - Entrées multiples pour modèles
- 🚀 **[API Lua v2.3](03-API-Reference/API_LUA_UPDATE_v2.3.0.md)** - Mimir.Serialization API
- ✅ **[Validation Sérialisation](03-API-Reference/SERIALIZATION_API_VALIDATED.md)** - Tests complets

**Modules disponibles (14 modules, 117 fonctions):**
- `model` (17 fonctions) - Gestion du modèle et entraînement
- `architectures` (9 fonctions) - Builders d'architectures pré-définies
- `Mimir.Serialization` (3 fonctions) - **NOUVEAU v2.3** - Sérialisation moderne
- `flux` (5 fonctions) - API fonctionnelle Flux (génération texte→image)
- `FluxModel` (12 fonctions) - API orientée objet Flux
- `layers` (8 fonctions) - Opérations de couches (placeholders)
- `tokenizer` (24 fonctions) - Tokenization word-level et BPE
- `dataset` (3 fonctions) - Chargement et préparation des données
- `memory` (6 fonctions) - AdvancedRAMManager
- `guard` (4 fonctions) - MemoryGuard API ancienne
- `MemoryGuard` (7 fonctions) - MemoryGuard API moderne (recommandée)
- `allocator` (3 fonctions) - DynamicTensorAllocator avec compression
- `htop` (5 fonctions) - Monitoring temps réel terminal
- `viz` (11 fonctions) - Visualisation SFML (images, métriques, loss)
- Globales (3 fonctions) - `log()`, `read_json()`, `write_json()`

**💡 Note:** Le stub API ([mimir-api.lua](../mimir-api.lua)) est synchronisé à 100% avec l'implémentation C++ ([src/LuaScripting.cpp](../src/LuaScripting.cpp)). Voir [API_STUB_UPDATE.md](../API_STUB_UPDATE.md) pour les détails techniques.

### Architecture et Internals (avancé)
- 🏛️ **[Architecture Système](04-Architecture-Internals/01-System-Architecture.md)** - Vue d'ensemble système
- ⚙️ **[Runtime Engine](04-Architecture-Internals/02-Runtime-Engine.md)** - Moteur d'exécution interne
- 📊 **[Spécifications Techniques](04-Architecture-Internals/03-Technical-Specifications.md)** - Détails techniques
- 💾 **[Gestion Mémoire](04-Architecture-Internals/04-Memory-Management.md)** - Système de gestion RAM 10GB
- 🧵 **[Threading et Compute](04-Architecture-Internals/05-Threading-And-Compute.md)** - Multithreading et GPU
- 🛡️ **[Correctifs Sécurité Mémoire](04-Architecture-Internals/06-Memory-Safety-Fixes.md)** - Fixes OOM
- ⚡ **[Optimisations Hardware](04-Architecture-Internals/07-Hardware-Optimizations.md)** - AVX2, FMA, SIMD
- 🧮 **[Système de Tenseurs](04-Architecture-Internals/08-Tensor-System.md)** - Refactoring tenseurs
- 🔌 **[Mises à Jour API](04-Architecture-Internals/09-API-Updates.md)** - Évolution de l'API
- 🌊 **[Modèle Flux](04-Architecture-Internals/10-Flux-Model.md)** - Architecture Flux complète
- 🌊 **[Implémentation Flux](04-Architecture-Internals/10-Flux-Implementation.md)** - Résumé implémentation
- 📊 **[Couverture Opérations](04-Architecture-Internals/OP_COVERAGE.md)** - Liste complète des opérations
- 💾 **[Sérialisation Complète](04-Architecture-Internals/SERIALIZATION_COMPLETE.md)** - Implémentation système
- 🔧 **[État Technique](04-Architecture-Internals/TECHNICAL_STATUS.md)** - Status v2.3.0

### Sujets Avancés
- 🚀 **[API Pipeline](05-Advanced/01-Pipeline-API.md)** - Pipelines haut niveau
- 🔄 **[Pipeline Complet](05-Advanced/02-Pipeline-Complete.md)** - Workflow complet
- 🏋️ **[Training Complet](05-Advanced/03-Training-Complete.md)** - Entraînement avancé
- 🧩 **[Opérations Layers](05-Advanced/04-Layer-Operations.md)** - Couches personnalisées
- 🏗️ **[Architectures Modèles](05-Advanced/05-Model-Architectures.md)** - Architectures avancées
- 🧱 **[Memory Blocking](05-Advanced/06-Memory-Blocking.md)** - Système de blocking mémoire
- 🌿 **[Branch Operations](05-Advanced/07-Branch-Operations.md)** - Opérations de branches
- 🌳 **[Branch System](05-Advanced/08-Branch-System.md)** - Système de branches
- ⚡ **[Layer Fusion](05-Advanced/09-Layer-Fusion.md)** - Fusion de couches
- 🔲 **[Conv2D Completion](05-Advanced/10-Conv2D-Completion.md)** - Convolution 2D complète
- 🔳 **[Conv2D Improvements](05-Advanced/11-Conv2D-Improvements.md)** - Améliorations Conv2D
- 🔄 **[Migration Strict Mode](05-Advanced/MIGRATION_STRICT_MODE.md)** - Migration v2.1 strict
- ⚡ **[Performance](05-Advanced/PERFORMANCE.md)** - Optimisations et benchmarks
- 📊 **[Migration Benchmarks](05-Advanced/BENCHMARKS_MIGRATION.md)** - Migration benchmarks v2.3

### Contribution
- 🤝 **[Guide de Contribution](06-Contributing/01-Contributing-Guide.md)** - Comment contribuer
- 📝 **[Template Pull Request](06-Contributing/02-Pull-Request-Template.md)** - Modèle de PR
- 🔄 **[Guide de Migration](06-Contributing/03-Migration-Guide.md)** - Migration entre versions
- 📚 **[Guide Documentation](06-Contributing/04-Documentation-Guide.md)** - Standards de documentation
- 🗺️ **[Roadmap](06-Contributing/05-Roadmap.md)** - Évolution future
- 🔄 **[Mises à Jour Documentation](06-Contributing/06-Documentation-Updates.md)** - Changelog docs
- 📜 **[Mises à Jour Scripts](06-Contributing/07-Scripts-Updates.md)** - Changelog scripts
- 🔧 **[Ajouter un Layer](06-Contributing/HOWTO_ADD_LAYER.md)** - Guide développeur layers

### Archives et Historique
- 📦 **[Archives/](Archives/)** - Documents historiques et rapports
- 🎯 **[Mission Complete](Archives/MISSION_COMPLETE.md)** - Refonte système layers v2.1
- 📋 **[Refonte Complete](Archives/REFONTE_COMPLETE.md)** - Détails refonte
- 🧹 **[Legacy Cleanup](Archives/LEGACY_CLEANUP_COMPLETE.md)** - Nettoyage code v2.3
- 📝 **[Release Notes 2.1.0](Archives/RELEASE_NOTES_2.1.0.md)** - Notes de version
- 📊 **[Documentation Update](Archives/DOCUMENTATION_UPDATE_v2.3.0.md)** - Mise à jour docs v2.3

---

## 📊 Statistiques du Framework

| Métrique | Valeur |
|----------|--------|
| **Version** | 2.3.0 |
| **Lignes de code C++** | ~15,000 |
| **Fichiers source** | 30 |
| **Fonctions API Lua** | 117 (100% synchronisées) |
| **Modules API** | 14 (+ Mimir.Serialization) |
| **Layers supportés** | 67 |
| **Architectures pré-définies** | 9 |
| **Formats sérialisation** | 3 (SafeTensors, RawFolder, DebugJson) |
| **Optimisations SIMD** | AVX2, FMA, F16C, BMI2 |
| **Licence** | GPL-2.0 / Commercial |

---

## 🎯 Philosophie du Framework

**Mímir est volontairement CPU-only.** Pas de CUDA, pas de ROCm, juste des CPU modernes.

### Pourquoi CPU-only?

- 💰 **Accessibilité** : Pas besoin de GPU à 1000€+
- 🌍 **Universalité** : Fonctionne partout (laptop, serveur, edge)
- 🔧 **Simplicité** : Pas de drivers complexes
- ⚡ **Performance** : Optimisations SIMD exploitent le CPU au max
- 🎯 **Mission** : Démocratiser l'IA sans barrière financière

### Points Forts

✅ **Performance CPU maximale** - AVX2/FMA exploitent chaque cycle  
✅ **API Lua intuitive** - Prototypage rapide sans recompilation  
✅ **9 architectures modernes** - UNet, VAE, ViT, GAN, Diffusion, Transformer, ResNet, MobileNet, Flux  
✅ **Gestion mémoire avancée** - Compression LZ4, offload, allocation dynamique  
✅ **Monitoring intégré** - Htop terminal + visualisation SFML  
✅ **Open Source** - GPL-2.0, communauté active

---

## 🚀 Démarrage Ultra-Rapide (2 minutes)

```bash
# 1. Installer
git clone https://github.com/your-repo/mimir-framework
cd mimir-framework
./build.sh

# 2. Premier modèle
cat > hello_mimir.lua << 'EOF'
-- Créer un petit transformer
model.create("transformer", {vocab_size = 1000, embed_dim = 128, num_layers = 2})
model.build()
print("✅ Modèle créé avec", model.total_params(), "paramètres!")
EOF

# 3. Exécuter
./bin/mimir hello_mimir.lua
```

**Résultat** : Modèle transformer créé et prêt en moins de 30 secondes! 🎉

---

## 📚 Organisation de la Documentation

### Niveau 1️⃣ : Getting Started (Débutant)
**Durée** : 5-30 minutes  
**Public** : Nouveaux utilisateurs, découverte  
**Objectif** : Installer et créer son premier modèle

### Niveau 2️⃣ : User Guide (Intermédiaire)
**Durée** : 1-3 heures  
**Public** : Utilisateurs réguliers  
**Objectif** : Maîtriser toutes les fonctionnalités

### Niveau 3️⃣ : API Reference (Consultation)
**Durée** : Variable  
**Public** : Développeurs actifs  
**Objectif** : Référence complète de l'API

### Niveau 4️⃣ : Architecture & Internals (Expert)
**Durée** : 3-10 heures  
**Public** : Contributors, experts  
**Objectif** : Comprendre l'implémentation interne

### Niveau 5️⃣ : Advanced Topics (Expert+)
**Durée** : Variable  
**Public** : Power users, optimisation  
**Objectif** : Maîtrise complète et personnalisation

### Niveau 6️⃣ : Contributing (Contributeur)
**Durée** : 1-2 heures  
**Public** : Contributeurs potentiels  
**Objectif** : Standards et processus de contribution

---

## 🔍 Comment Utiliser Cette Documentation

### Je débute complètement
1. Lire [Démarrage Rapide](01-Getting-Started/01-Quick-Start.md)
2. Suivre [Installation](01-Getting-Started/02-Installation.md)
3. Configurer votre [IDE](01-Getting-Started/03-IDE-Setup.md)
4. Consulter [Pourquoi CPU-only?](01-Getting-Started/04-Why-CPU-Only.md) pour comprendre la philosophie

### Je veux créer un modèle spécifique
1. Parcourir [Architectures Prédéfinies](02-User-Guide/03-Predefined-Architectures.md)
2. Lire [Création de Modèles](02-User-Guide/02-Model-Creation.md)
3. Consulter [API Complète](03-API-Reference/00-API-Complete.md)
4. Voir exemples dans `/scripts`

### Je veux optimiser les performances
1. Lire [API Pipeline](05-Advanced/01-Pipeline-API.md)
2. Consulter [Hardware Optimizations](04-Architecture-Internals/07-Hardware-Optimizations.md)
3. Voir [Memory Blocking](05-Advanced/06-Memory-Blocking.md)

### Je veux comprendre l'interne
1. Lire [Architecture Système](04-Architecture-Internals/01-System-Architecture.md)
2. Étudier [Runtime Engine](04-Architecture-Internals/02-Runtime-Engine.md)
3. Approfondir chaque composant dans Architecture-Internals/

### Je veux contribuer
1. Lire [Guide de Contribution](06-Contributing/01-Contributing-Guide.md)
2. Suivre [Template Pull Request](06-Contributing/02-Pull-Request-Template.md)
3. Consulter [Roadmap](06-Contributing/05-Roadmap.md)

---

## 📖 Formats de Documentation

### Markdown (.md)
Tous les documents sont en Markdown pour faciliter la lecture sur GitHub et dans les IDEs.

### Code Samples
Tous les exemples de code sont testés et fonctionnels. Vous pouvez les copier-coller directement.

### Diagrammes
Les diagrammes sont en ASCII art pour une compatibilité maximale.

---

## 🆘 Support et Communauté

- 📧 **Email** : support@mimir-framework.org
- 💬 **Discord** : [discord.gg/mimir](https://discord.gg/mimir)
- 🐛 **Issues** : [GitHub Issues](https://github.com/your-repo/mimir/issues)
- 📚 **Forum** : [forum.mimir-framework.org](https://forum.mimir-framework.org)

---

## 📝 Licence

Mímir Framework est sous **double licence** :

- **GPL-2.0** : Usage gratuit pour projets open source, éducation, recherche
- **Commercial** : Licence commerciale disponible pour usage propriétaire

Voir [LICENSE](../LICENSE) pour les détails complets.

---

## 🙏 Remerciements

Merci à tous les contributeurs et utilisateurs du Mímir Framework!

---

**© 2025 Mímir Framework - Documentation v2.0**  
*Généré automatiquement le 22 décembre 2025*
