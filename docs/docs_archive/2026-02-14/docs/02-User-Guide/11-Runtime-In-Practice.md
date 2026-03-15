# Runtime en pratique

Ce guide explique ce que fait le runtime de Mímir au démarrage, comment passer des arguments aux scripts Lua, et comment configurer la mémoire et les performances **sans modifier le C++**.

---

## Objectif

Quand tu exécutes un script via `./bin/mimir --lua ...`, le binaire :

- Initialise le runtime Lua et enregistre l’API (`Mimir.*` + quelques alias globaux).
- Configure et annonce les optimisations disponibles (OpenMP, AVX2/FMA/F16C/BMI2).
- Applique les garde-fous mémoire (tests `MemorySafety` au démarrage, `MemoryGuard` pendant l’exécution).

Le runtime est documenté en profondeur ici : [docs/04-Architecture-Internals/02-Runtime-Engine.md](../04-Architecture-Internals/02-Runtime-Engine.md).

---

## Démarrage : ce que tu dois voir

Au lancement, `main.cpp` affiche typiquement :

- Le nombre de threads OpenMP disponibles (si compilé avec OpenMP).
- Les capacités CPU détectées : AVX2 / FMA / F16C / BMI2.

Ensuite, le script est exécuté via l’API Lua.

---

## Exécuter un script Lua

Commande standard :

```bash
./bin/mimir --lua scripts/mon_script.lua
```

### Passer des arguments au script

Tout ce qui suit le chemin du script est injecté dans :

- `arg[0]` = chemin du script
- `arg[1..n]` = arguments
- `Mimir.Args[0..n]` = miroir de `arg` (exposé pour éviter les ambiguïtés)

Exemple :

```bash
./bin/mimir --lua scripts/train.lua -- --epochs 10 --lr 3e-4
```

Le séparateur `--` est conseillé pour éviter qu’un futur parseur CLI ne consomme des flags destinés au script.

---

## API Lua : namespace recommandé vs alias

Le runtime enregistre l’API sous `Mimir.*` et ajoute des alias globaux pour compatibilité.

Recommandé :

```lua
Mimir.Model.create("transformer", cfg)
Mimir.Model.allocate_params()
Mimir.Model.init_weights("xavier", 1234)
```

Alias disponibles (rétrocompatibilité / scripts historiques) :

- `model` = `Mimir.Model`
- `dataset` = `Mimir.Dataset`
- `tokenizer` = `Mimir.Tokenizer`
- `Allocator` = `Mimir.Allocator`
- `MemoryGuard` = `Mimir.MemoryGuard`

---

## Mémoire : les 2 réglages importants

### 1) Limite stricte (MemoryGuard)

`MemoryGuard` refuse les nouvelles allocations si la limite est dépassée.

```lua
-- Valeur <= 1000  -> interprétée comme des GB
Mimir.MemoryGuard.setLimit(8)      -- 8 GB

-- Valeur très grande -> interprétée comme des bytes
-- Mimir.MemoryGuard.setLimit(8 * 1024 * 1024 * 1024)
```

Pour diagnostiquer :

```lua
local stats = Mimir.MemoryGuard.getStats()
log(string.format("RAM: %.1f / %.1f MB (peak=%.1f MB)", stats.current_mb, stats.limit_mb, stats.peak_mb))
Mimir.MemoryGuard.printStats()
```

### 2) Allocateur dynamique (Mimir.Allocator)

`Mimir.Allocator` configure `DynamicTensorAllocator` (handles + lazy allocation + compression LZ4 en RAM).

```lua
local ok, err = Mimir.Allocator.configure({
  max_ram_gb = 8,
  enable_compression = true,
})
if not ok then error(err) end
```

Notes :

- Le mode lazy est activé par défaut côté C++ (les buffers réels sont alloués au premier `getData()`).
- En cas d’OOM, réduire la taille des batches/séquences, ou baisser les dimensions du modèle, est souvent plus efficace que d’augmenter la limite.

---

## Compute et performances

### OpenMP

Si le binaire est compilé avec OpenMP, une partie des kernels CPU utilise `#pragma omp`.

- Le runtime affiche le nombre de threads au démarrage.
- Le réglage le plus simple côté utilisateur est `OMP_NUM_THREADS` (variable d’environnement).

### SIMD (AVX2/FMA/F16C/BMI2)

Les hot paths utilisent des intrinsics SIMD (voir `src/SIMD_Ops.hpp` et `src/HardwareOpt.hpp`).

- Si AVX2/FMA sont absents, le runtime reste fonctionnel mais moins rapide.

### Vulkan Compute (optionnel)

Le modèle tente d’initialiser un backend Vulkan Compute au runtime.

- Si l’initialisation échoue, il retombe sur CPU (fallback).
- Ce chemin est **optionnel** et n’est pas requis pour utiliser le framework.

---

## Recettes rapides (copier-coller)

### Script minimal “runtime safe”

```lua
-- 1) Limite mémoire et allocateur
Mimir.MemoryGuard.setLimit(8)
local ok, err = Mimir.Allocator.configure({ max_ram_gb = 8, enable_compression = true })
if not ok then error(err) end

-- 2) Créer / allouer / initialiser
local cfg, cfg_err = Mimir.Architectures.default_config("transformer")
if not cfg then error(cfg_err) end

ok, err = Mimir.Model.create("transformer", cfg)
if not ok then error(err) end

ok = Mimir.Model.allocate_params()
if not ok then error("allocate_params failed") end
Mimir.Model.init_weights("xavier", 1234)

log("Runtime OK")
```

---

## Voir aussi

- Runtime (internals) : [docs/04-Architecture-Internals/02-Runtime-Engine.md](../04-Architecture-Internals/02-Runtime-Engine.md)
- Bonnes pratiques mémoire : [docs/02-User-Guide/10-Memory-Best-Practices.md](10-Memory-Best-Practices.md)
- Lua scripting : [docs/01-Getting-Started/07-Lua-Scripting.md](../01-Getting-Started/07-Lua-Scripting.md)
