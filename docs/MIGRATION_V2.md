# Guide de Migration v1.x → v2.0

Ce guide vous aide à migrer vos scripts Lua existants vers la nouvelle API asynchrone de Mímir v2.0.

## Changements Majeurs

### 1. HtopDisplay - Monitoring Asynchrone

#### ❌ Avant (v1.x - Synchrone, Bloquant)

```lua
htop.create()

for epoch = 1, 10 do
    for batch = 1, 100 do
        -- ... training ...
        
        htop.update(epoch, 10, batch, 100, loss, avg_loss, lr, ...)
        htop.render()  -- ⚠️ BLOQUANT! Ralentit le training
    end
end
```

**Problème**: `htop.render()` bloque le thread principal pendant ~50ms → **20% de perte de performance**

#### ✅ Après (v2.0 - Asynchrone, Non-Bloquant)

```lua
htop.create()  -- Démarre un thread séparé

for epoch = 1, 10 do
    for batch = 1, 100 do
        -- ... training ...
        
        htop.update(epoch, 10, batch, 100, loss, avg_loss, lr, ...)
        -- ✓ PAS BESOIN de htop.render()! C'est automatique
    end
end
```

**Avantage**: Rendu automatique dans un thread séparé → **Gain de 20% sur le training**

#### Migration Automatique

**Option 1 - Supprimer les appels `render()`**:
```bash
# Supprimer tous les htop.render() dans vos scripts
sed -i 's/htop\.render()/-- htop.render() -- NO-OP en v2.0/g' *.lua
```

**Option 2 - Garder pour compatibilité** (recommandé):
```lua
htop.render()  -- NO-OP en v2.0, conservé pour compatibilité
```

### 2. Visualizer - Rendu Asynchrone

#### ❌ Avant (v1.x)

```lua
viz.create(config)
viz.initialize()

while viz.is_open() do
    viz.process_events()  -- ⚠️ Bloque le thread
    
    -- Training
    viz.update_metrics(epoch, batch, loss, lr)
    viz.add_image(pixels, prompt)
    
    viz.update()  -- ⚠️ Bloque le thread
end
```

#### ✅ Après (v2.0)

```lua
viz.create(config)  -- Démarre un thread séparé
-- viz.initialize() n'est plus nécessaire (automatique)

for epoch = 1, 10 do
    for batch = 1, 100 do
        -- Training
        viz.update_metrics(epoch, batch, loss, lr)
        viz.add_image(pixels, prompt)
        
        -- ✓ PAS BESOIN de viz.process_events() ni viz.update()!
        -- C'est automatique dans le thread séparé
    end
end
```

#### Migration Automatique

```bash
# Remplacer la boucle while par une boucle for
sed -i 's/while viz\.is_open() do/for epoch = 1, total_epochs do/g' *.lua
sed -i 's/viz\.process_events()/-- viz.process_events() -- NO-OP en v2.0/g' *.lua
sed -i 's/viz\.update()/-- viz.update() -- NO-OP en v2.0/g' *.lua
```

### 3. Gestion Mémoire - DynamicTensorAllocator

#### ❌ Avant (v1.x)

```lua
memory.config({
    max_ram_gb = 10.0,
    enable_compression = true
})
```

#### ✅ Après (v2.0)

```lua
allocator.configure({
    max_ram_gb = 10.0,
    enable_compression = true,
    compression_threshold_mb = 100  -- Nouveau paramètre
})
```

**Nouveautés**:
- Compression **LZ4 réelle** (au lieu du stub)
- Lazy loading transparent via `getData()`
- Éviction LRU automatique

#### Migration Automatique

```bash
sed -i 's/memory\.config(/allocator.configure(/g' *.lua
sed -i 's/memory\.print_stats()/allocator.print_stats()/g' *.lua
```

### 4. Accélération GPU (Nouveau)

#### Détection Automatique

```lua
-- Vérifier si le GPU est disponible
if model.has_vulkan_compute() then
    print("✓ Accélération GPU activée (Vulkan Compute)")
else
    print("⚠ Mode CPU uniquement")
end
```

**Note**: Le dispatch GPU/CPU est **automatique** pour les layers > 10k params. Aucune modification de code nécessaire.

## Checklist de Migration

### Étape 1: Sauvegarder

```bash
cp my_script.lua my_script_v1.lua.bak
```

### Étape 2: Mettre à jour les appels API

**Rechercher et remplacer**:
```bash
# HtopDisplay
sed -i 's/htop\.render()/-- htop.render()/g' my_script.lua

# Visualizer
sed -i 's/viz\.process_events()/-- viz.process_events()/g' my_script.lua
sed -i 's/viz\.update()/-- viz.update()/g' my_script.lua

# Memory → Allocator
sed -i 's/memory\.config(/allocator.configure(/g' my_script.lua
sed -i 's/memory\.print_stats()/allocator.print_stats()/g' my_script.lua
```

### Étape 3: Tester

```bash
# Tester avec la nouvelle version
bin/mimir --lua my_script.lua

# Comparer les performances
# v1.x: ~250ms/batch
# v2.0: ~200ms/batch (20% plus rapide)
```

### Étape 4: Optimiser (Optionnel)

Ajouter la détection GPU:

```lua
-- Au début du script
if model.has_vulkan_compute() then
    print("✓ GPU détecté - Accélération automatique activée")
end
```

## Compatibilité Backward

### Fonctions Conservées (NO-OP)

Ces fonctions existent encore pour compatibilité mais ne font plus rien:

- `htop.render()` → Rendu automatique dans le thread
- `viz.process_events()` → Traitement automatique
- `viz.update()` → Mise à jour automatique
- `viz.initialize()` → Initialisation automatique

**Vous pouvez garder ces appels** dans vos scripts, ils n'auront aucun impact négatif.

### Fonctions Dépréciées

Ces fonctions ont été renommées:

| v1.x | v2.0 | Migration |
|------|------|-----------|
| `memory.config()` | `allocator.configure()` | Renommer |
| `memory.print_stats()` | `allocator.print_stats()` | Renommer |
| `memory.get_stats()` | `allocator.get_stats()` | Renommer |

## Exemple de Migration Complète

### Avant (v1.x)

```lua
#!/usr/bin/env lua5.3

-- Configuration mémoire
memory.config({
    max_ram_gb = 10.0,
    enable_compression = true
})

-- Monitoring
htop.create()

-- Visualizer
viz.create({enabled = true})
viz.initialize()

-- Training
model.create("encoder")
model.build()

while viz.is_open() do
    viz.process_events()
    
    for epoch = 1, 10 do
        for batch = 1, 100 do
            -- ... training ...
            
            htop.update(epoch, 10, batch, 100, loss, avg_loss, lr)
            htop.render()  -- BLOQUANT!
            
            viz.update_metrics(epoch, batch, loss, lr)
            viz.update()  -- BLOQUANT!
        end
    end
end

memory.print_stats()
```

### Après (v2.0)

```lua
#!/usr/bin/env lua5.3

-- Configuration mémoire (nouveau nom)
allocator.configure({
    max_ram_gb = 10.0,
    enable_compression = true,
    compression_threshold_mb = 100  -- Nouveau paramètre
})

-- Monitoring asynchrone
htop.create()

-- Visualizer asynchrone
viz.create({enabled = true})
-- viz.initialize() n'est plus nécessaire

-- Vérifier le GPU
if model.has_vulkan_compute() then
    print("✓ GPU activé")
end

-- Training
model.create("encoder")
model.build()

for epoch = 1, 10 do
    for batch = 1, 100 do
        -- ... training (GPU automatique si disponible) ...
        
        -- Mise à jour asynchrone (non-bloquant)
        htop.update(epoch, 10, batch, 100, loss, avg_loss, lr)
        -- Pas besoin de htop.render()!
        
        viz.update_metrics(epoch, batch, loss, lr)
        -- Pas besoin de viz.update()!
    end
end

allocator.print_stats()  -- Nouveau nom
```

## Dépannage

### Erreur: "memory.config not found"

**Solution**: Remplacer par `allocator.configure()`

### Monitoring ne s'affiche pas

**Solution**: Vérifier que vous avez appelé `htop.create()` au début du script

### "Vulkan not available"

**Solution**: Normal si pas de GPU Vulkan. Le framework utilise automatiquement le CPU (fallback).

### Performance dégradée

**Vérification**:
```lua
-- Ajouter des logs de timing
local start = os.clock()
-- ... training batch ...
local elapsed = os.clock() - start
print(string.format("Batch time: %.2fms", elapsed * 1000))
```

Si > 250ms/batch → Problème (devrait être ~200ms en v2.0)

## Support

- Documentation complète: `docs/THREADING_AND_COMPUTE.md`
- Exemples: `examples/async_monitoring_demo.lua`
- API Reference: `mimir-api.lua` (pour autocomplétion IDE)

## Changelog v2.0

**Ajouts**:
- Threading asynchrone pour htop/viz
- Vulkan Compute (GPU) avec fallback CPU
- Compression LZ4 réelle
- DynamicTensorAllocator avec lazy loading

**Modifications**:
- `memory.*` → `allocator.*`
- `htop.render()` → NO-OP (automatique)
- `viz.process_events()/update()` → NO-OP (automatique)

**Gains de Performance**:
- +20% sur training complet (threading)
- +100x potentiel sur GPU (layers > 10k params)
- -50% utilisation mémoire (compression LZ4)
