# Guide de Migration v1.x → v2.0

Ce guide vous aide à migrer vos scripts Lua existants vers la nouvelle API asynchrone de Mímir v2.0.

## Changements Majeurs

### 1. HtopDisplay - Monitoring Asynchrone

#### ❌ Avant (v1.x - Synchrone, Bloquant)

```lua
Mimir.Htop.create()

for epoch = 1, 10 do
    for batch = 1, 100 do
        -- ... training ...
        
        Mimir.Htop.update(epoch, 10, batch, 100, loss, avg_loss, lr, ...)
        Mimir.Htop.render()  -- ⚠️ BLOQUANT! Ralentit le training
    end
end
```

**Problème**: `Mimir.Htop.render()` bloque le thread principal pendant ~50ms → **20% de perte de performance**

#### ✅ Après (v2.0 - Asynchrone, Non-Bloquant)

```lua
Mimir.Htop.create()  -- Démarre un thread séparé

for epoch = 1, 10 do
    for batch = 1, 100 do
        -- ... training ...
        
        Mimir.Htop.update(epoch, 10, batch, 100, loss, avg_loss, lr, ...)
        -- ✓ PAS BESOIN de Mimir.Htop.render()! C'est automatique
    end
end
```

**Avantage**: Rendu automatique dans un thread séparé → **Gain de 20% sur le training**

#### Migration Automatique

**Option 1 - Supprimer les appels `render()`**:
```bash
# Supprimer tous les Mimir.Htop.render() dans vos scripts
sed -i 's/htop\.render()/-- Mimir.Htop.render() -- NO-OP en v2.0/g' *.lua
```

**Option 2 - Garder pour compatibilité** (recommandé):
```lua
Mimir.Htop.render()  -- NO-OP en v2.0, conservé pour compatibilité
```

### 2. Visualizer - Rendu Asynchrone

#### ❌ Avant (v1.x)

```lua
Mimir.Viz.create(config)
Mimir.Viz.initialize()

while Mimir.Viz.is_open() do
    Mimir.Viz.process_events()  -- ⚠️ Bloque le thread
    
    -- Training
    Mimir.Viz.update_metrics(epoch, batch, loss, lr)
    Mimir.Viz.add_image(pixels, prompt)
    
    Mimir.Viz.update()  -- ⚠️ Bloque le thread
end
```

#### ✅ Après (v2.0)

```lua
Mimir.Viz.create(config)  -- Démarre un thread séparé
-- Mimir.Viz.initialize() n'est plus nécessaire (automatique)

for epoch = 1, 10 do
    for batch = 1, 100 do
        -- Training
        Mimir.Viz.update_metrics(epoch, batch, loss, lr)
        Mimir.Viz.add_image(pixels, prompt)
        
        -- ✓ PAS BESOIN de Mimir.Viz.process_events() ni Mimir.Viz.update()!
        -- C'est automatique dans le thread séparé
    end
end
```

#### Migration Automatique

```bash
# Remplacer la boucle while par une boucle for
sed -i 's/while viz\.is_open() do/for epoch = 1, total_epochs do/g' *.lua
sed -i 's/viz\.process_events()/-- Mimir.Viz.process_events() -- NO-OP en v2.0/g' *.lua
sed -i 's/viz\.update()/-- Mimir.Viz.update() -- NO-OP en v2.0/g' *.lua
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
Mimir.Allocator.configure({
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
sed -i 's/memory\.config(/Mimir.Allocator.configure(/g' *.lua
sed -i 's/memory\.print_stats()/Mimir.Allocator.print_stats()/g' *.lua
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
sed -i 's/htop\.render()/-- Mimir.Htop.render()/g' my_script.lua

# Visualizer
sed -i 's/viz\.process_events()/-- Mimir.Viz.process_events()/g' my_script.lua
sed -i 's/viz\.update()/-- Mimir.Viz.update()/g' my_script.lua

# Memory → Allocator
sed -i 's/memory\.config(/Mimir.Allocator.configure(/g' my_script.lua
sed -i 's/memory\.print_stats()/Mimir.Allocator.print_stats()/g' my_script.lua
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

- `Mimir.Htop.render()` → Rendu automatique dans le thread
- `Mimir.Viz.process_events()` → Traitement automatique
- `Mimir.Viz.update()` → Mise à jour automatique
- `Mimir.Viz.initialize()` → Initialisation automatique

**Vous pouvez garder ces appels** dans vos scripts, ils n'auront aucun impact négatif.

### Fonctions Dépréciées

Ces fonctions ont été renommées:

| v1.x | v2.0 | Migration |
|------|------|-----------|
| `memory.config()` | `Mimir.Allocator.configure()` | Renommer |
| `memory.print_stats()` | `Mimir.Allocator.print_stats()` | Renommer |
| `memory.get_stats()` | `Mimir.Allocator.get_stats()` | Renommer |

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
Mimir.Htop.create()

-- Visualizer
Mimir.Viz.create({enabled = true})
Mimir.Viz.initialize()

-- Training
model.create("encoder")
model.build()

while Mimir.Viz.is_open() do
    Mimir.Viz.process_events()
    
    for epoch = 1, 10 do
        for batch = 1, 100 do
            -- ... training ...
            
            Mimir.Htop.update(epoch, 10, batch, 100, loss, avg_loss, lr)
            Mimir.Htop.render()  -- BLOQUANT!
            
            Mimir.Viz.update_metrics(epoch, batch, loss, lr)
            Mimir.Viz.update()  -- BLOQUANT!
        end
    end
end

memory.print_stats()
```

### Après (v2.0)

```lua
#!/usr/bin/env lua5.3

-- Configuration mémoire (nouveau nom)
Mimir.Allocator.configure({
    max_ram_gb = 10.0,
    enable_compression = true,
    compression_threshold_mb = 100  -- Nouveau paramètre
})

-- Monitoring asynchrone
Mimir.Htop.create()

-- Visualizer asynchrone
Mimir.Viz.create({enabled = true})
-- Mimir.Viz.initialize() n'est plus nécessaire

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
        Mimir.Htop.update(epoch, 10, batch, 100, loss, avg_loss, lr)
        -- Pas besoin de Mimir.Htop.render()!
        
        Mimir.Viz.update_metrics(epoch, batch, loss, lr)
        -- Pas besoin de Mimir.Viz.update()!
    end
end

Mimir.Allocator.print_stats()  -- Nouveau nom
```

## Dépannage

### Erreur: "memory.config not found"

**Solution**: Remplacer par `Mimir.Allocator.configure()`

### Monitoring ne s'affiche pas

**Solution**: Vérifier que vous avez appelé `Mimir.Htop.create()` au début du script

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
- `memory.*` → `Mimir.Allocator.*`
- `Mimir.Htop.render()` → NO-OP (automatique)
- `Mimir.Viz.process_events()/update()` → NO-OP (automatique)

**Gains de Performance**:
- +20% sur training complet (threading)
- +100x potentiel sur GPU (layers > 10k params)
- -50% utilisation mémoire (compression LZ4)
