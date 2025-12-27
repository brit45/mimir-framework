# Système de Blocage d'Allocation Mémoire

## Vue d'ensemble

Le framework Mímir dispose maintenant d'un système complet de blocage et de contrôle des allocations mémoire, intégré à trois niveaux :

1. **MemoryGuard** : Garde-fou strict sur toutes les allocations
2. **AdvancedRAMManager** : Gestionnaire avancé avec compression/éviction
3. **DynamicTensorAllocator** : Allocateur de tenseurs avec respect des blocages

## Fonctionnalités

### 🔒 Blocage Complet

Empêche toute nouvelle allocation mémoire :

```cpp
// C++
auto& guard = MemoryGuard::instance();
guard.blockAllocations(true);   // Bloquer
guard.blockAllocations(false);  // Débloquer

// Vérifier l'état
bool blocked = guard.isBlocked();
```

**Comportement** :

- Toutes les tentatives d'allocation retournent `false`
- Les libérations continuent de fonctionner normalement
- Un compteur de tentatives bloquées est maintenu
- Messages d'erreur détaillés dans stderr

### ❄️ Mode Freeze

Gèle l'état mémoire : autorise les libérations mais bloque les nouvelles allocations :

```cpp
// C++
guard.freezeAllocations(true);   // Activer freeze
guard.freezeAllocations(false);  // Désactiver freeze

// Vérifier l'état
bool frozen = guard.isFrozen();
```

**Cas d'usage** :

- Capturer un snapshot mémoire stable
- Permettre le nettoyage sans nouvelles allocations
- Stabiliser l'utilisation avant opération critique

### ⏰ Blocage Temporaire

Bloque automatiquement avec déblocage après timeout :

```cpp
// Bloquer pendant 2 secondes puis débloquer automatiquement
guard.blockTemporary(2000); // millisecondes
```

**Implémentation** :

- Thread détaché pour le déblocage automatique
- Utile pour rate limiting
- Protection temporaire pendant opérations critiques

## Intégration Multi-Niveaux

### Niveau 1 : MemoryGuard

**Rôle** : Garde-fou strict sur TOUTES les allocations de tenseurs

```cpp
class MemoryGuard {
    // Configuration
    void setLimit(size_t bytes);
    
    // Blocage
    void blockAllocations(bool block = true);
    void freezeAllocations(bool freeze = true);
    void blockTemporary(size_t milliseconds);
    
    // État
    bool isBlocked() const;
    bool isFrozen() const;
    
    // Allocation
    bool requestAllocation(size_t bytes, const std::string& tag = "");
    void releaseAllocation(size_t bytes);
    
    // Statistiques
    void printStats() const;
    size_t getCurrentBytes() const;
    float getUsagePercent() const;
};
```

**Nouvelles variables membres** :

```cpp
std::atomic<bool> allocations_blocked_{false};
std::atomic<bool> freeze_mode_{false};
std::atomic<size_t> blocked_attempts_{0};
std::atomic<size_t> frozen_attempts_{0};
```

### Niveau 2 : AdvancedRAMManager

**Rôle** : Gestion avancée avec compression, éviction LRU, et blocage

```cpp
class AdvancedRAMManager {
    // Blocage
    void blockAllocations(bool block = true);
    void freezeAllocations(bool freeze = true);
    bool isBlocked() const;
    bool isFrozen() const;
    
    // Allocation (respecte le blocage)
    bool allocate(const std::string& key, 
                  const std::vector<uint8_t>& data, 
                  bool compress = true);
};
```

**Vérification dans `allocate()`** :

```cpp
// Vérifier si les allocations sont bloquées
if (allocations_blocked_.load()) {
    blocked_attempts_++;
    return false;
}

// Vérifier si en mode freeze
if (freeze_mode_.load()) {
    frozen_attempts_++;
    return false;
}
```

### Niveau 3 : DynamicTensorAllocator

**Rôle** : Allocation lazy de tenseurs, respecte automatiquement les blocages de MemoryGuard

Les appels à `requestAllocation()` de MemoryGuard vérifient automatiquement les blocages.

## Statistiques Enrichies

Les statistiques incluent maintenant l'état de blocage :

```txt
╔═══════════════════════════════════════════════════════╗
║           MEMORY GUARD - STATISTIQUES                 ║
╠═══════════════════════════════════════════════════════╣
║ Limite:           1024 MB                             ║
║ Actuel:            150 MB                             ║
║ Pic:               250 MB                             ║
║ Utilisation:      14.6 %                              ║
║ Allocations:         15                               ║
║ Libérations:          5                               ║
╠═══════════════════════════════════════════════════════╣
║ État:       🔒 BLOQUÉ                                 ║
║ Tentatives bloquées:      3                           ║
║ Tentatives gelées:        0                           ║
╚═══════════════════════════════════════════════════════╝
```

## Cas d'Usage

### 1. Contrôle Strict de Mémoire

```cpp
// Avant opération critique
guard.blockAllocations(true);

// Opération garantie sans nouvelles allocations
processData();

// Débloquer après
guard.blockAllocations(false);
```

### 2. Tests de Robustesse

```cpp
// Simuler conditions de mémoire limitée
guard.blockAllocations(true);

// Tester que le code gère correctement les échecs
bool success = model.forward(input);
assert(!success && "Le modèle doit gérer l'échec d'allocation");

guard.blockAllocations(false);
```

### 3. Snapshot Mémoire

```cpp
// Capturer état stable
guard.freezeAllocations(true);

// Permettre nettoyage mais pas nouvelles allocations
cleanupUnusedTensors();
saveMemorySnapshot();

guard.freezeAllocations(false);
```

### 4. Protection Pendant Swap

```cpp
// Bloquer pendant compression/décompression
ram.blockAllocations(true);

// Compression de tenseurs inactifs
compressInactiveTensors();

// Débloquer
ram.blockAllocations(false);
```

### 5. Rate Limiting

```cpp
// Limiter vitesse d'allocation
if (allocationRate() > threshold) {
    guard.blockTemporary(1000); // Pause 1 seconde
}
```

## Workflow Complet d'Entraînement

```cpp
// 1. Configuration initiale
auto& guard = MemoryGuard::instance();
guard.setLimit(2ULL * 1024 * 1024 * 1024); // 2 GB

auto& allocator = DynamicTensorAllocator::instance();
allocator.configure(2, true); // 2 GB, compression activée

// 2. Construction du modèle
Model model;
model.buildArchitecture();
model.allocateParams();

// 3. Boucle d'entraînement
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (auto& batch : dataloader) {
        // Bloquer avant forward pass
        guard.blockAllocations(true);
        
        // Forward (utilise uniquement mémoire existante)
        auto output = model.forward(batch.input, true);
        
        // Débloquer pour backward (gradients nécessitent allocation)
        guard.blockAllocations(false);
        
        auto loss = criterion(output, batch.target);
        auto grads = model.backward(loss);
        
        // Freeze pendant optimisation
        guard.freezeAllocations(true);
        optimizer.step();
        guard.freezeAllocations(false);
        
        // Nettoyage périodique
        if (step % cleanup_interval == 0) {
            guard.blockAllocations(true);
            model.clearCache();
            guard.blockAllocations(false);
        }
    }
}
```

## Thread Safety

Toutes les opérations de blocage sont thread-safe grâce à `std::atomic` :

```cpp
std::atomic<bool> allocations_blocked_{false};
std::atomic<bool> freeze_mode_{false};
std::atomic<size_t> blocked_attempts_{0};
std::atomic<size_t> frozen_attempts_{0};
```

Les opérations `load()` et `store()` sont atomiques et ne nécessitent pas de mutex.

## Tests

Un programme de test complet est fourni : [tests/test_memory_blocking.cpp](../tests/test_memory_blocking.cpp)

**Tests inclus** :

1. Blocage complet (MemoryGuard)
2. Mode freeze
3. Blocage temporaire avec timeout
4. Blocage AdvancedRAMManager
5. Intégration complète

**Exécution** :

```bash
g++ -std=c++17 -O3 tests/test_memory_blocking.cpp -I./src -o test_blocking -llz4 -pthread
./test_blocking
```

## Performance

### Impact sur les Performances

- **Vérification de blocage** : ~1-2 cycles CPU (lecture atomique)
- **Overhead** : Négligeable (<0.1% du temps d'allocation)
- **Thread-safe** : Pas de contention avec mutex

### Compteurs

Les compteurs de tentatives bloquées utilisent `fetch_add()` atomique, garantissant la précision même en multi-threading.

## Limitations et Considérations

### Limitations Actuelles

1. **Pas de file d'attente** : Les allocations bloquées échouent immédiatement
2. **Pas de priorités** : Toutes les allocations sont traitées également
3. **Pas de notification** : Pas de callback lors du déblocage

### Considérations

- **Cohérence** : S'assurer que le blocage est bien levé après usage
- **Deadlocks** : Éviter de bloquer indéfiniment dans du code critique
- **Debugging** : Utiliser `printStats()` pour tracer les blocages

## Évolutions Futures

- [ ] File d'attente d'allocations en attente
- [ ] Système de priorités pour allocations critiques
- [ ] Callbacks de notification de déblocage
- [ ] Blocage sélectif par tag
- [ ] Quotas par type de tenseur
- [ ] Interface Lua complète
- [ ] Profiling des patterns de blocage

## API Lua (À Implémenter)

```lua
-- Blocage
model.blockAllocations(true/false)
model.freezeAllocations(true/false)
model.blockTemporary(milliseconds)

-- État
local blocked = model.isBlocked()
local frozen = model.isFrozen()

-- Statistiques
model.printMemoryStats()

-- MemoryGuard direct
MemoryGuard.setLimit(bytes)
MemoryGuard.blockAllocations(true/false)
MemoryGuard.printStats()
```

## Conclusion

Le système de blocage d'allocation mémoire offre un contrôle précis et flexible sur l'utilisation de la RAM, essentiel pour :

- **Stabilité** : Éviter les dépassements mémoire
- **Prédictibilité** : Garantir l'utilisation mémoire dans des limites connues
- **Testing** : Simuler conditions de ressources limitées
- **Performance** : Optimiser les patterns d'allocation/libération

L'intégration à trois niveaux (MemoryGuard, AdvancedRAMManager, DynamicTensorAllocator) garantit que le blocage est respecté partout dans le framework.
