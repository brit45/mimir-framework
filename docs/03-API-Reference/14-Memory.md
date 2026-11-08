# Mémoire

Trouver rapidement le contrat API réel et les paramètres utilisables.

**Public concerné :** Développeur et utilisateur intermédiaire/avancé.

> **Prérequis**
>
> Connaître les commandes de base de Mímir.


Source : `src/scriptings/Lua/luaScripting/LuaScriptingRuntimeAndViz.cpp`.

Les mesures ci-dessous concernent les allocations suivies par chaque gestionnaire.
Elles ne remplacent pas la mesure de toute la mémoire résidente du processus ni
la mémoire d'un device GPU.

## `Mimir.Memory`

- `config(table) -> bool` : options lues `max_ram_gb`, `enable_compression`,
  `enable_async_loading`, `enable_prediction`, `enable_statistics`,
  `preload_queue_size`, `worker_threads`.
- `get_stats() -> table` : `current_mb`, `peak_mb`, `usage_percent`.
- `print_stats() -> true`, `clear() -> true`.
- `get_usage() -> current_mb, peak_mb, usage_percent` : trois retours distincts.
- `set_limit(gb) -> true | (false, err)` : valeur strictement positive en Gio.

`config` et `set_limit` configurent `AdvancedRAMManager`, avec débordement disque
dans `.mimir-spill`. `set_limit` réapplique aussi ses réglages internes de
compression et de workers ; il ne modifie pas seulement un compteur de limite.
Les alias camelCase `getStats`, `printStats`, `getUsage` et `setLimit` existent.

## `Mimir.Guard` (strict)

- `set_limit(value) -> true | (false, err)` : valeur positive interprétée en Gio
  si `value <= 1000`, sinon en octets.
- `get_stats() -> table` : `current_mb`, `peak_mb`, `limit_mb`, `usage_percent`.
- `print_stats() -> true`, `reset() -> true`.

Les variantes `setLimit`, `getStats` et `printStats` sont des alias. Activer la
limite configure également le chemin d'éviction de `DynamicTensorAllocator`.

## `Mimir.MemoryGuard` (alias moderne)

Ce namespace utilise le même `MemoryGuard` que `Mimir.Guard` :

- `setLimit(value)` suit la même convention Gio/octets ;
- `getCurrentUsage()`, `getPeakUsage()`, `getLimit()` retournent des **octets** ;
- `getStats()` retourne les champs en Mio décrits ci-dessus ;
- `printStats()` et `reset()` sont les mêmes opérations que dans `Guard`.

## `Mimir.Allocator`

- `configure({max_ram_gb=10, enable_compression=true}) -> bool`.
- `get_stats() -> table` : `tensor_count`, `loaded_count`.
- `print_stats() -> true` ; alias `getStats`, `printStats`.

Le binding Lua ne lit pas `swap_strategy` : l'ajouter à la table n'impose aucune
politique d'éviction.

Configurez les limites au début du run, avant les allocations importantes. Les
arguments nommés `gb` et `mb` utilisent ici des multiples de 1024 (Gio et Mio).

## Étapes suivantes

- [Page précédente : API : `Mimir.Dataset`](13-Dataset.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : API : monitoring / visualisation](15-Viz-Htop.md)
