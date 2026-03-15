# 🛡️ Guide de Recompilation et Test Post-Correctifs

## 🎯 Correctifs Appliqués

Les correctifs critiques de sécurité mémoire ont été appliqués pour empêcher les dépassements RAM qui causaient des crashs OOM. Voir [MEMORY_SAFETY_FIXES.md](MEMORY_SAFETY_FIXES.md) pour les détails complets.

---

## 📋 Étapes de Recompilation

### 1️⃣ Nettoyage du Build Précédent

```bash
cd <repo-root>
rm -rf build
mkdir -p build
```

### 2️⃣ Configuration CMake (Production - Recommandé)

```bash
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMIMIR_ENABLE_LEGACY_PARAMS=OFF \
  -DENABLE_SIMD=ON \
  -DENABLE_OPENMP=ON
```

**Note**: `MIMIR_ENABLE_LEGACY_PARAMS=OFF` est **critique** pour éviter l'explosion mémoire.

### 3️⃣ Compilation

```bash
make -j$(nproc)
```

### 4️⃣ Vérification du Build

```bash
./bin/mimir --help
```

Vous devriez voir au démarrage:
```
🛡️  Vérification de la sécurité mémoire...
✅ Structure legacy désactivée (configuration optimale)
🧪 TEST D'INTÉGRITÉ MÉMOIRE
...
```

---

## 🧪 Tests de Validation

### Test 1: Intégrité du Système Mémoire

```bash
./bin/mimir --lua scripts/tests/test_memory_safety.lua
```

**Attendu**: 
- ✅ Tous les tests passent
- Aucune allocation ne dépasse 10 GB
- Refus propre si limite atteinte

### Test 2: Limite 10GB Stricte

```bash
./bin/mimir --lua scripts/tests/test_10gb_limit.lua
```

**Attendu**:
- Limite respectée
- Stats mémoire affichées
- Pas de dépassement

### Test 3: Modèle Réaliste (UNet)

```bash
LUA_PATH='scripts/modules/?.lua;;' ./bin/mimir --lua scripts/demos/demo_unet.lua
```

**Attendu**:
- Création du modèle sans erreur Lua
- Allocation des paramètres sans dépasser la limite

> Note: `scripts/tests/test_flux_complete.lua` existe mais fait un **skip** sur le binaire v2.3 si `Mimir.FluxModel` n'est pas exporté.

---

## 🔍 Vérification des Correctifs

### ✅ Vérifier Allocation Dynamique des Poids

Chercher dans les logs au démarrage:
```
📦 Allocation de X blocs de poids
⚠️ CRITIQUE: Allocation dynamique via MemoryGuard
```

### ✅ Vérifier Structure Legacy Désactivée

Au démarrage:
```
✅ Structure legacy désactivée (configuration optimale)
```

Si vous voyez:
```
⚠️⚠️⚠️  ATTENTION: LEGACY PARAMS ACTIVÉ!
```

➡️ **Recompiler avec `-DMIMIR_ENABLE_LEGACY_PARAMS=OFF`**

### ✅ Vérifier stb_image Protégé

Charger une image et vérifier les logs:
```
📊 Allocation: X MB (stb_image) - Total: Y MB / 10240 MB
```

---

## 🐛 Debugging en Cas de Problème

### Problème: RAM dépasse toujours 10 GB

**Causes possibles**:
1. Legacy params activé par erreur
2. Code legacy non migré utilisant `tensor(size)` au lieu de `tensor(size, true)`
3. Allocations directes `std::vector` quelque part

**Solution**:
```bash
# Chercher les allocations non-dynamiques
grep -rn "tensor(" src/ | grep -v "tensor(.*true"

# Vérifier la config CMake
cmake .. -LA | grep LEGACY
```

### Problème: Crash au démarrage

**Cause probable**: Manque d'en-têtes ou linking

**Solution**:
```bash
# Vérifier les dépendances
ldd ./bin/mimir

# Recompiler en clean
rm -rf build/*
cmake .. && make -j$(nproc)
```

### Problème: "requestAllocation failed"

**C'est normal !** Si la limite est atteinte, l'allocation est refusée proprement.

**Logs attendus**:
```
❌ MemoryGuard: ALLOCATION REFUSÉE!
   Demandé: X MB
   Actuel: Y MB
   Limite: 10240 MB
🚨 ARRÊT CONTRÔLÉ POUR ÉVITER CRASH OS
```

---

## 📊 Monitoring de la RAM

### En temps réel

```bash
# Terminal 1: Lancer Mímir
./bin/mimir --lua scripts/demo_flux.lua

# Terminal 2: Monitorer la RAM
watch -n 1 'ps aux | grep mimir | grep -v grep'
```

### Avec MemorySafety utilities

Dans votre code C++ ou Lua:
```cpp
#include "MemorySafety.hpp"

// Afficher rapport complet
MemorySafety::printMemoryReport();

// Vérifier si proche de la limite
MemorySafety::checkMemoryPressure(0.8f); // Alert à 80%

// Test d'intégrité
MemorySafety::runMemoryIntegrityTest();
```

---

## 🎛️ Options de Configuration

### Build Debug (avec plus de logs)

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DMIMIR_ENABLE_LEGACY_PARAMS=OFF
```

Les logs `TRACE_ALLOC` seront activés pour tracer chaque allocation.

### Build avec Legacy (déconseillé)

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMIMIR_ENABLE_LEGACY_PARAMS=ON
```

⚠️ **Attention**: Consomme beaucoup plus de RAM ! Utiliser seulement pour compatibilité temporaire.

---

## ✅ Checklist de Validation

Après recompilation, vérifier:

- [ ] `./bin/mimir` démarre sans erreur
- [ ] Message "Structure legacy désactivée" au démarrage
- [ ] Tests d'intégrité mémoire passent
- [ ] RAM ne dépasse jamais 10 GB en monitoring
- [ ] Refus propre si limite atteinte (pas de crash OS)
- [ ] Stats MemoryGuard accessibles

---

## 📚 Documentation Complète

- [MEMORY_SAFETY_FIXES.md](MEMORY_SAFETY_FIXES.md) - Détails des correctifs
- [MEMORY_10GB_IMPLEMENTATION.md](MEMORY_10GB_IMPLEMENTATION.md) - Architecture système mémoire
- [docs/03-API-Reference/](docs/03-API-Reference/) - API complète

---

## 🆘 Support

En cas de problème persistant:

1. Afficher les stats mémoire:
   ```bash
   ./bin/mimir --lua -e "MemorySafety.printMemoryReport()"
   ```

2. Vérifier la config CMake:
   ```bash
   cat build/CMakeCache.txt | grep LEGACY
   ```

3. Tester avec modèle minimal:
   ```bash
   ./bin/mimir --lua scripts/example_simple.lua
   ```

---

**Date**: 27 Décembre 2025  
**Version**: Mímir 2.0 Post-Security-Fix  
**Status**: ✅ Production Ready
