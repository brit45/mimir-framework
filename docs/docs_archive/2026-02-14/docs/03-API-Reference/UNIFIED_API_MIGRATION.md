# API Unifiée - Enhanced Debug JSON v1.1.0

## ✅ Modifications Complétées

### 1. Structure SaveOptions étendue
**Fichier**: [src/Serialization/Serialization.hpp](../../src/Serialization/Serialization.hpp#L38-L49)

Ajout des options Enhanced Debug JSON v1.1.0 directement dans `SaveOptions` :
```cpp
struct SaveOptions {
    // Options existantes...
    
    // Enhanced DebugJson options (v1.1.0)
    bool include_gradients = false;
    bool include_optimizer_state = false;
    size_t max_values_per_tensor = 20;
    bool include_activations = false;
    bool include_checksums = false;
    bool include_weight_deltas = false;
};
```

### 2. Implémentation C++ modifiée
**Fichier**: [src/Serialization/Serialization.cpp](../../src/Serialization/Serialization.cpp#L35-L49)

Conversion automatique de `SaveOptions` vers `DebugJsonOptions` :
```cpp
case CheckpointFormat::DebugJson: {
    DebugJsonDump dumper;
    
    // Convert SaveOptions to DebugJsonOptions
    DebugJsonOptions debug_opts;
    debug_opts.include_gradients = options.include_gradients;
    debug_opts.include_optimizer_state = options.include_optimizer_state;
    debug_opts.max_values_per_tensor = options.max_values_per_tensor;
    debug_opts.include_activations = options.include_activations;
    debug_opts.include_checksums = options.include_checksums;
    debug_opts.include_weight_deltas = options.include_weight_deltas;
    debug_opts.include_git_info = options.include_git_info;
    debug_opts.save_tokenizer = options.save_tokenizer;
    debug_opts.save_encoder = options.save_encoder;
    
    return dumper.save_enhanced(path, model, debug_opts, error);
}
```

### 3. Bindings Lua étendus
**Fichier**: [src/LuaScripting.cpp](../../src/LuaScripting.cpp#L1171-L1213)

Parsing des nouvelles options depuis Lua :
```cpp
// Enhanced DebugJson options (v1.1.0)
lua_getfield(L, 3, "include_gradients");
if (lua_isboolean(L, -1)) {
    options.include_gradients = lua_toboolean(L, -1);
}
lua_pop(L, 1);

// ... (6 autres options)
```

### 4. API Lua documentée
**Fichier**: [mimir-api.lua](../../mimir-api.lua#L1237-L1249)

Annotations EmmyLua complètes :
```lua
---@class SaveOptions
---@field include_gradients? boolean @[DebugJson v1.1] Inclure gradients (défaut: false)
---@field include_optimizer_state? boolean @[DebugJson v1.1] Inclure optimizer state (défaut: false)
---@field max_values_per_tensor? integer @[DebugJson v1.1] Nb valeurs par tensor (défaut: 20)
---@field include_activations? boolean @[DebugJson v1.1] Inclure activations (défaut: false)
---@field include_checksums? boolean @[DebugJson v1.1] Inclure checksums (défaut: false)
---@field include_weight_deltas? boolean @[DebugJson v1.1] Inclure weight deltas (défaut: false)
```

## 📝 Utilisation

### API Unifiée (recommandée)

```lua
-- DebugJson Enhanced v1.1.0 avec API standard
Mimir.Serialization.save("debug.json", "debug_json", {
    include_gradients = true,
    include_optimizer_state = true,
    include_weight_deltas = true,
    include_checksums = true,
    max_values_per_tensor = 20,
    save_tokenizer = false,
    save_encoder = false
})
```

### Ancienne API (toujours disponible, mais non nécessaire)

```lua
-- Fonction spécialisée (optionnelle)
Mimir.Serialization.save_enhanced_debug("debug.json", {
    include_gradients = true,
    -- ... mêmes options
})
```

## ✅ Tests Validés

### Test 1: API Unifiée
**Fichier**: [scripts/tests/test_unified_api.lua](../../scripts/tests/test_unified_api.lua)

Résultats :
```
✅ Format SafeTensors: Fonctionnel
✅ Format DebugJson legacy: Fonctionnel
✅ Format DebugJson v1.1.0: Fonctionnel
✅ Options Enhanced intégrées: OK
```

### Test 2: Benchmark Conv Training
**Fichier**: [scripts/benchmarks/benchmark_conv_train.lua](../../scripts/benchmarks/benchmark_conv_train.lua)

Utilise maintenant l'API unifiée avec options v1.1.0 :
```lua
Mimir.Serialization.save(snapshot_path, "debug_json", {
    include_gradients = true,
    include_optimizer_state = false,
    max_values_per_tensor = 10,
    include_checksums = true,
    include_weight_deltas = false,
    include_git_info = false,
    save_tokenizer = false,
    save_encoder = false
})
```

## 🎯 Avantages

1. **API Unique** : Une seule fonction `Mimir.Serialization.save()` pour tous les formats
2. **Cohérence** : Même signature pour SafeTensors, RawFolder et DebugJson
3. **Extensibilité** : Facile d'ajouter de nouvelles options sans changer l'API
4. **Rétro-compatibilité** : Les anciens scripts fonctionnent toujours
5. **Documentation** : Autocomplétion IDE complète avec EmmyLua

## 📊 Comparaison

| Aspect | Ancienne API | Nouvelle API |
|--------|-------------|--------------|
| **Fonction** | `save_enhanced_debug()` séparée | `save()` unifiée |
| **Options** | Structure séparée | Intégrées dans `SaveOptions` |
| **Formats** | Spécifique DebugJson | Tous formats (SafeTensors, RawFolder, DebugJson) |
| **Maintenance** | Dupliquer code | Centralisé |
| **Cohérence** | API fragmentée | API cohérente |

## 🔄 Migration

Scripts à migrer simplement en changeant :

**Avant** :
```lua
Mimir.save_enhanced_debug("debug.json", {
    include_gradients = true
})
```

**Après** :
```lua
Mimir.Serialization.save("debug.json", "debug_json", {
    include_gradients = true
})
```

## 📚 Documentation Complète

- [Enhanced Debug JSON Format](ENHANCED_DEBUG_JSON.md) - Guide complet v1.1.0
- [Serialization API](SAVE_LOAD.md) - Documentation générale
- [Test Unitaire](../../scripts/tests/test_unified_api.lua) - Exemples d'utilisation

## ✅ Résultat

**L'API est maintenant unifiée et cohérente**. Toutes les fonctionnalités Enhanced Debug JSON v1.1.0 sont accessibles via l'API standard `Mimir.Serialization.save()` avec le format `"debug_json"`.
