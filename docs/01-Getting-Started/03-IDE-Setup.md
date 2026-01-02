# Mímir Framework v2.0 - API Lua

## 🆕 Nouveautés v2.0

### Threading Asynchrone
- **HtopDisplay** et **Visualizer** s'exécutent dans des threads séparés
- Rendu automatique toutes les 100ms (non-bloquant)
- API simplifiée: plus besoin d'appeler `render()`, `process_events()`, etc.
- **Gain**: ~20% d'accélération sur le training complet

### Accélération GPU
- Détection automatique de **Vulkan Compute** au démarrage
- Dispatch intelligent CPU/GPU (seuil: 10k params par layer)
- Fallback automatique vers CPU si GPU indisponible
- **Gain potentiel**: ~100x sur layers éligibles (GPU RTX vs CPU AVX2)

### Gestion Mémoire Avancée
- Allocation dynamique avec compression **LZ4** réelle
- Éviction LRU automatique quand limite atteinte
- Lazy loading transparent (getData())

## Configuration de l'IDE

### Visual Studio Code

1. Installer l'extension **Lua Language Server** (sumneko.lua)
2. Le fichier `.luarc.json` configure automatiquement l'autocomplétion
3. Le fichier `mimir-api.lua` fournit les définitions de types

### Autres IDEs

Pour les autres IDEs supportant Lua Language Server :
- Copier `.luarc.json` à la racine du projet
- Configurer le LSP pour charger `mimir-api.lua`

## API Disponible

### Model (model.*)

Gestion des modèles de deep learning :

```lua
-- Créer et construire un modèle
model.create("encoder", {
    vocab_size = 100000,
    embed_dim = 256,
    num_layers = 4,
    num_heads = 8
})
model.build()
model.allocate_params()
model.init_weights("he")

-- Entraînement
model.train(sequences, 50, 0.0002)

-- Sauvegarde
model.save("checkpoints/my_model")
```

### Tokenizer (Mimir.Tokenizer.*)

Tokenisation et vocabulaire :

```lua
-- Créer et construire le vocabulaire
Mimir.Tokenizer.create(1000000)
for _, word in ipairs(words) do
    Mimir.Tokenizer.add_token(word)
end

-- Tokenisation
local ids = Mimir.Tokenizer.tokenize("Hello world")
local text = Mimir.Tokenizer.detokenize(ids)

-- Statistiques
Mimir.Tokenizer.print_stats()
```

### Dataset (Mimir.Dataset.*)

Chargement de données :

```lua
Mimir.Dataset.load("../data/train")
local sequences = Mimir.Dataset.prepare_sequences(tokenizer, 128)
```

### Allocator (Mimir.Allocator.*)

Gestion dynamique de la RAM avec compression LZ4 :

```lua
-- Configuration (limite stricte à 10 GB)
Mimir.Allocator.configure({
    max_ram_gb = 10.0,
    enable_compression = true
})

-- Monitoring
local stats = Mimir.Allocator.get_stats()
print("Tenseurs chargés: " .. stats.loaded_count .. "/" .. stats.tensor_count)
Mimir.Allocator.print_stats()
```

### HtopDisplay (Mimir.Htop.*)

Monitoring temps réel type htop :

```lua
-- Créer le moniteur
Mimir.Htop.create()

-- Dans la boucle d'entraînement
for epoch = 1, num_epochs do
    for batch = 1, num_batches do
        -- ... training ...
        
        Mimir.Htop.update(
            epoch, num_epochs,
            batch, num_batches,
            loss, avg_loss, lr,
            batch_time_ms, memory_mb, memory_freed,
            batches_per_sec, total_params, timestep,
            kl, wass, ent, mom, spat, temp, mse
        )
        Mimir.Htop.render()
    end
end
```

### Visualizer (Mimir.Viz.*)

Visualisation graphique SFML avec **rendu asynchrone non-bloquant** :

```lua
-- Créer et démarrer le visualizer asynchrone (lance un thread séparé)
Mimir.Viz.create({
    enabled = true,
    window_width = 1280,
    window_height = 720,
    show_loss_graph = true
})

-- Pas besoin de Mimir.Viz.initialize() - automatique dans le thread

-- Dans la boucle d'entraînement
for epoch = 1, total_epochs do
    for batch = 1, total_batches do
        -- ... training ...
        
        -- Mise à jour thread-safe (non-bloquante)
        Mimir.Viz.update_metrics(epoch, batch, loss, lr, mse, kl, wass, ent, mom, spat, temp)
        
        -- Ajouter images générées (thread-safe)
        if batch % 100 == 0 then
            Mimir.Viz.add_image(generated_pixels, "my prompt")
        end
        
        -- PAS BESOIN de Mimir.Viz.process_events() ni Mimir.Viz.update()!
        -- Le rendu et les événements sont gérés automatiquement
    end
end

-- Sauvegarder historique
Mimir.Viz.save_loss_history("loss_history.csv")
```

**Note**: `Mimir.Viz.process_events()` et `Mimir.Viz.update()` existent encore pour compatibilité mais sont des NO-OP.

### Guard & Memory (Mimir.Mimir.MemoryGuard.*, memory.*)

Gestion stricte de la mémoire :

```lua
-- Limite stricte (refuse allocations dépassant la limite)
Mimir.Mimir.MemoryGuard.set_limit(10.0)  -- 10 GB

-- Gestionnaire avec compression
memory.config({
    max_ram_gb = 10.0,
    enable_compression = true,
    enable_statistics = true
})

-- Statistiques
local guard_stats = Mimir.Mimir.MemoryGuard.get_stats()
print(string.format("RAM: %.1f%% (%.0f/%.0f MB)", 
    guard_stats.usage_percent, 
    guard_stats.current_mb, 
    guard_stats.limit_mb))
```

## Exemple Complet v2.0 (Asynchrone)

```lua
#!/usr/bin/env mimir --lua

log("=== Entraînement Transformer v2.0 (Asynchrone + GPU) ===")

-- Configuration RAM dynamique avec compression LZ4
Mimir.Allocator.configure({
    max_ram_gb = 10.0,
    enable_compression = true,
    compression_threshold_mb = 100
})

-- Monitoring asynchrone (non-bloquant)
Mimir.Htop.create()

-- Dataset et Tokenizer
Mimir.Dataset.load("../data/train")
Mimir.Tokenizer.create(100000)
-- ... build vocab ...

-- Modèle (détection GPU automatique au démarrage)
model.create("encoder", {
    vocab_size = Mimir.Tokenizer.vocab_size(),
    embed_dim = 256,
    num_layers = 4
})
model.build()
model.init_weights("he")

-- Vérifier l'accélération GPU
if model.has_vulkan_compute() then
    log("✓ Accélération GPU activée (Vulkan Compute)")
else
    log("⚠ Mode CPU uniquement")
end

-- Entraînement
local sequences = Mimir.Dataset.prepare_sequences(tokenizer, 128)
for epoch = 1, 50 do
    for batch = 1, #sequences do
        local start_time = os.clock()
        
        -- Forward/Backward (GPU automatique pour layers > 10k params)
        local output = model.forward(sequences[batch].input)
        local loss = model.compute_loss(output, sequences[batch].target)
        model.backward(loss_gradient)
        model.optimizer_step(optimizer, lr)
        
        local batch_time = (os.clock() - start_time) * 1000
        
        -- Mise à jour monitoring (thread-safe, non-bloquant)
        -- Le rendu se fait automatiquement dans le thread séparé
        Mimir.Htop.update(epoch, 50, batch, #sequences, 
                   loss, avg_loss, lr, batch_time, memory_mb)
        
        -- PAS BESOIN de Mimir.Htop.render() - c'est automatique!
    end
end

-- Sauvegarde
model.save("checkpoints/final")
Mimir.Allocator.print_stats()

log("✓ Entraînement terminé!")
```

## Performance v2.0

### Gains de Performance

**Threading Asynchrone**:
- ⚡ **~20% plus rapide**: Monitoring en parallèle du training
- 🔓 **Non-bloquant**: htop/viz n'impactent plus les performances
- 🧵 **Thread-safe**: Synchronisation automatique via mutex

**Accélération GPU** (layers > 10k params):
- ⚡ **~100x sur GPU**: 10 GFLOPS (CPU) → 1000 GFLOPS (GPU RTX 3060)
- 🔄 **Fallback automatique**: CPU si GPU indisponible
- 🎯 **Dispatch intelligent**: GPU seulement pour layers assez grands

### Exemple Benchmark

```
Avant v2.0 (synchrone):
Epoch 1, Batch 100: 250ms (200ms compute + 50ms monitoring)
                                            ^^^^^^^^^^^^^^ BLOQUANT!

Après v2.0 (asynchrone):
Epoch 1, Batch 100: 200ms (200ms compute)
                    Monitoring: 100ms en arrière-plan (thread séparé)
```

**Résultat**: 20% d'accélération sur l'entraînement complet

## Autocomplétion

Avec le Lua Language Server configuré, vous bénéficiez de :

- ✅ **Autocomplétion** : Suggestions intelligentes des fonctions et paramètres
- ✅ **Type checking** : Vérification des types d'arguments
- ✅ **Documentation inline** : Descriptions des fonctions en survolant
- ✅ **Diagnostics** : Détection d'erreurs avant exécution
- ✅ **Snippets** : Templates de code prêts à l'emploi

## Support IDE

| IDE | Support | Configuration |
|-----|---------|---------------|
| VS Code | ✅ Excellent | Extension Lua Language Server |
| Neovim | ✅ Excellent | lua-language-server via LSP |
| IntelliJ | ✅ Bon | Plugin EmmyLua |
| Sublime Text | ⚠️ Partiel | Plugin LSP + lua-language-server |

## Ressources

- Documentation complète : `docs/API_LUA.md`
- Exemples : `scripts/example_*.lua`
- Architecture C++ : `docs/ARCHITECTURE.md`
