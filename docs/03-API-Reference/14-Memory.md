# API : mémoire

Source : `src/LuaScripting.cpp`.

## `Mimir.Memory`

- `config(table)`
- `get_stats()` / `print_stats()`
- `clear()`
- `get_usage()`
- `set_limit(gb)`

## `Mimir.Guard` (strict)

- `set_limit(gb)`
- `get_stats()` / `print_stats()`
- `reset()`

## `Mimir.MemoryGuard` (alias moderne)

- `setLimit(gb)`
- `getCurrentUsage()`, `getPeakUsage()`, `getLimit()`
- `getStats()`, `printStats()`, `reset()`

## `Mimir.Allocator`

- `configure({max_ram_gb, enable_compression, swap_strategy, ...})`
- `get_stats()` / `print_stats()`

Recommandation : activer `MemoryGuard` + `Allocator` au début des scripts.
