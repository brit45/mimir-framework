# Limitation RAM à 10 Go

## Vue d'ensemble

Le système de gestion mémoire de Mímir permet de **bloquer strictement** l'allocation RAM à une limite définie (par défaut 10 Go). Cette fonctionnalité empêche le système de consommer plus de mémoire que spécifié, protégeant ainsi contre les dépassements mémoire (OOM).

## Configuration de la Limite

### Via Lua

```lua
-- Définir la limite à 10 Go
MemoryGuard.setLimit(10 * 1024 * 1024 * 1024)  -- en bytes

-- Ou simplement (accepte GB si valeur < 1000)
MemoryGuard.setLimit(10)  -- 10 GB
```

### Via C++

```cpp
#include "MemoryGuard.hpp"

// Définir la limite à 10 Go
auto& guard = MemoryGuard::instance();
guard.setLimit(10ULL * 1024 * 1024 * 1024);  // 10 GB en bytes
```

## API Lua Complète

### Configuration

```lua
-- Définir la limite (bytes ou GB si < 1000)
MemoryGuard.setLimit(bytes_or_gb)

-- Obtenir la limite configurée (en bytes)
local limit = MemoryGuard.getLimit()

-- Obtenir l'utilisation courante (en bytes)
local current = MemoryGuard.getCurrentUsage()

-- Obtenir le pic d'utilisation (en bytes)
local peak = MemoryGuard.getPeakUsage()

-- Obtenir les statistiques complètes (table)
local stats = MemoryGuard.getStats()

-- Afficher les statistiques
MemoryGuard.printStats()

-- Réinitialiser les compteurs
MemoryGuard.reset()
```

### Exemple Complet

```lua
-- Configuration initiale
print("Configuration MemoryGuard: Limite 10 Go")
MemoryGuard.setLimit(10 * 1024 * 1024 * 1024)

-- Vérifier la configuration
local limit = MemoryGuard.getLimit()
print("Limite: " .. string.format("%.2f", limit / 1024 / 1024 / 1024) .. " GB")

-- Faire des allocations...
local my_model = model.create("MonModele")
-- ... opérations ...

-- Vérifier l'utilisation
local current = MemoryGuard.getCurrentUsage()
local peak = MemoryGuard.getPeakUsage()

print("RAM courante: " .. string.format("%.2f", current / 1024 / 1024 / 1024) .. " GB")
print("RAM pic: " .. string.format("%.2f", peak / 1024 / 1024 / 1024) .. " GB")

-- Rapport final
if peak < limit then
    print("✅ Limite respectée!")
else
    print("⚠️  Limite dépassée!")
end
```

## API C++ Complète

### Include

```cpp
#include "MemoryGuard.hpp"
```

### Configuration

```cpp
auto& guard = MemoryGuard::instance();

// Définir la limite
guard.setLimit(10ULL * 1024 * 1024 * 1024);  // 10 GB

// Obtenir les informations
size_t limit = guard.getLimit();
size_t current = guard.getCurrentBytes();
size_t peak = guard.getPeakBytes();
double usage = guard.getUsagePercent();

// Statistiques
guard.printStats();

// Réinitialiser
guard.reset();
```

### Requête d'Allocation

```cpp
// Demander une allocation
size_t bytes_needed = 1024 * 1024 * 100;  // 100 MB
bool allowed = guard.requestAllocation(bytes_needed, "MonTensor");

if (allowed) {
    // Allocation autorisée
    float* data = new float[bytes_needed / sizeof(float)];
    
    // ... utiliser les données ...
    
    // Libérer
    guard.releaseAllocation(bytes_needed);
    delete[] data;
} else {
    std::cerr << "❌ Allocation refusée: limite dépassée" << std::endl;
}
```

## Fonctionnalités de Blocage

### Bloquer Temporairement

```cpp
// Bloquer toutes les nouvelles allocations
guard.blockAllocations(true);

// Les allocations sont maintenant refusées
// ...

// Débloquer
guard.blockAllocations(false);
```

### Blocage Temporaire avec Timeout

```cpp
// Bloquer pendant 5 secondes
guard.blockTemporary(5000);  // millisecondes

// Après 5 secondes, le blocage est automatiquement levé
```

### Mode Freeze

```cpp
// Mode freeze: bloque nouvelles allocations mais autorise les libérations
guard.freezeAllocations(true);

// Les nouvelles allocations sont bloquées
// Les libérations fonctionnent normalement
// Permet de réduire l'utilisation sans crasher

guard.freezeAllocations(false);  // Désactiver
```

## Intégration avec AdvancedRAMManager

Le `AdvancedRAMManager` respecte automatiquement les limites de `MemoryGuard` :

```cpp
#include "AdvancedRAMManager.hpp"

auto& ram_manager = AdvancedRAMManager::instance();

// Configuration
AdvancedRAMManager::Config config;
config.max_ram_bytes = 10ULL * 1024 * 1024 * 1024;  // 10 GB
config.enable_compression = true;
config.enable_prediction = true;

ram_manager.configure(config);

// Les allocations via AdvancedRAMManager respectent la limite
std::vector<uint8_t> data(1024 * 1024);  // 1 MB
bool success = ram_manager.allocate("my_data", data, true);

if (!success) {
    std::cerr << "Allocation refusée: limite RAM atteinte" << std::endl;
}
```

## Hiérarchie de Gestion Mémoire

```
┌─────────────────────────────────────┐
│         MemoryGuard                 │  ← Limite stricte 10 GB
│      (Garde-fou global)             │
├─────────────────────────────────────┤
│     AdvancedRAMManager              │  ← Compression LZ4
│   (Gestion avancée + cache)         │     Éviction LRU
├─────────────────────────────────────┤
│   DynamicTensorAllocator            │  ← Pooling
│    (Allocation tenseurs)            │     Recyclage
└─────────────────────────────────────┘
```

**Tous les niveaux respectent la limite de 10 Go définie dans MemoryGuard.**

## Statistiques

### Format d'Affichage

```
╔═══════════════════════════════════════════════════════╗
║           MEMORY GUARD - STATISTIQUES                ║
╠═══════════════════════════════════════════════════════╣
║ Limite:          10240 MB                         ║
║ Actuel:           2456 MB                         ║
║ Pic:              3892 MB                         ║
║ Utilisation:     38.0 %                          ║
║ Allocations:      1523                            ║
║ Libérations:       892                            ║
╠═══════════════════════════════════════════════════════╣
║ État:       🔓 ACTIF                             ║
╚═══════════════════════════════════════════════════════╝
```

### Via Lua

```lua
local stats = MemoryGuard.getStats()
-- stats.current_mb   : RAM courante en MB
-- stats.peak_mb      : Pic en MB
-- stats.limit_mb     : Limite en MB
-- stats.usage_percent: Pourcentage d'utilisation
```

## Scripts de Test

### Demo Simple

```bash
./bin/mimir --lua scripts/demo_10gb_limit.lua
```

**Contenu du script** : `scripts/demo_10gb_limit.lua`

Vérifie :
- ✅ Configuration de la limite
- ✅ Surveillance des allocations
- ✅ Rapport d'utilisation
- ✅ Respect de la limite

### Test Intensif

```bash
./bin/mimir --lua scripts/test_memory_limit_10gb.lua
```

Teste :
- 📊 Petites allocations (<100 MB)
- 📊 Allocations moyennes (~500 MB)
- 📊 Grandes allocations (~2 GB)
- 📊 Vérification limite non dépassée

## Cas d'Usage

### 1. Serveur de Production

```lua
-- Limiter à 10 Go pour éviter d'impacter le système
MemoryGuard.setLimit(10 * 1024 * 1024 * 1024)

-- Charger et exécuter le modèle
local my_model = model.create("ProductionModel")
-- ... setup ...

-- Vérifier régulièrement
local function check_memory()
    local current = MemoryGuard.getCurrentUsage()
    local limit = MemoryGuard.getLimit()
    if current > limit * 0.9 then
        print("⚠️  Approche de la limite mémoire!")
    end
end
```

### 2. Développement/Debug

```cpp
// En mode debug, limite stricte pour détecter les fuites
#ifdef DEBUG
    MemoryGuard::instance().setLimit(5ULL * 1024 * 1024 * 1024);  // 5 GB
#else
    MemoryGuard::instance().setLimit(10ULL * 1024 * 1024 * 1024); // 10 GB
#endif
```

### 3. Multi-Modèles

```lua
-- Partager 10 Go entre plusieurs modèles
MemoryGuard.setLimit(10 * 1024 * 1024 * 1024)

local model1 = model.create("Model1")  -- ~3 GB
local model2 = model.create("Model2")  -- ~3 GB
local model3 = model.create("Model3")  -- ~3 GB

-- Total: ~9 GB, sous la limite de 10 GB
print("✅ Tous les modèles chargés sous la limite")
```

## Protection OOM (Out Of Memory)

### Mécanisme

1. **Prévention** : Allocation refusée avant dépassement
2. **Détection précoce** : Alerte à 90% de la limite
3. **Éviction automatique** : LRU cache si activé
4. **Compression** : LZ4 si activé (ratio ~0.4×)

### Exemple avec Éviction

```cpp
auto& ram = AdvancedRAMManager::instance();

// Tenter d'allouer
bool success = ram.allocate("big_data", large_data);

if (!success) {
    // Éviction LRU automatique
    // Puis retry
    std::cout << "⚠️  Éviction en cours..." << std::endl;
}
```

## Performance

### Impact sur les Performances

- **Vérification limite** : ~10 ns par allocation
- **Compteurs atomiques** : Thread-safe sans mutex
- **Lock-free** : Pas de contention entre threads

### Overhead Mémoire

- **MemoryGuard** : ~1 KB (compteurs + flags)
- **Par allocation** : ~0 byte (pas de métadonnées supplémentaires)

## Monitoring en Temps Réel

### Via Script Lua

```lua
-- Boucle de monitoring
while true do
    local current = MemoryGuard.getCurrentUsage()
    local limit = MemoryGuard.getLimit()
    local percent = (current / limit) * 100
    
    print(string.format("RAM: %.2f / %.2f GB (%.1f%%)", 
          current/1e9, limit/1e9, percent))
    
    -- Attendre 1 seconde
    os.execute("sleep 1")
end
```

### Avec AsyncMonitor

```cpp
#include "AsyncMonitor.hpp"

AsyncMonitor monitor;
monitor.start();

// Le monitoring RAM se fait automatiquement
// Statistiques disponibles en temps réel
```

## Recommandations

### Limites Suggérées

| Environnement | Limite RAM | Raison |
|---------------|------------|--------|
| Dev/Local | 5-10 GB | Développement |
| CI/CD | 4 GB | Ressources limitées |
| Production | 10-20 GB | Haute disponibilité |
| Serveur dédié | 20-50 GB | Performance maximale |

### Bonnes Pratiques

1. **Toujours définir une limite** dès le début
2. **Monitorer régulièrement** l'utilisation
3. **Activer la compression** pour données volumineuses
4. **Utiliser l'éviction LRU** si cache important
5. **Tester avec limite stricte** en développement

## Dépannage

### Problème : "Allocation bloquée"

```
🔒 MemoryGuard: ALLOCATION BLOQUÉE
```

**Solutions** :
1. Vérifier si allocations bloquées : `guard.isBlocked()`
2. Débloquer : `guard.blockAllocations(false)`
3. Augmenter la limite si nécessaire

### Problème : "Limite dépassée"

```
❌ MemoryGuard: Limite dépassée
```

**Solutions** :
1. Activer la compression : `ram.config.enable_compression = true`
2. Libérer mémoire inutilisée : `model.clear()`
3. Augmenter la limite : `guard.setLimit(15ULL * 1024 * 1024 * 1024)`

### Problème : "Pic > Limite"

**Diagnostic** :
```lua
local peak = MemoryGuard.getPeakUsage()
local limit = MemoryGuard.getLimit()
if peak > limit then
    print("⚠️  Pic a dépassé la limite")
    MemoryGuard.printStats()
end
```

**Solutions** :
- Analyser le moment du pic
- Optimiser les allocations temporaires
- Utiliser le pooling de tenseurs

## Conclusion

Le système de limitation RAM à 10 Go de Mímir offre :

- ✅ **Protection stricte** contre les OOM
- ✅ **API simple** Lua et C++
- ✅ **Monitoring en temps réel**
- ✅ **Thread-safe** avec atomics
- ✅ **Zero overhead** en production
- ✅ **Intégration complète** avec tout le framework

**Status : Production-Ready** 🚀
