# Sérialisation (save/load) — résumé

Objectif : sauvegarder et restaurer un modèle + son tokenizer/encoder de manière fiable.

Pour la référence complète (formats, options, comportements), voir :

- `docs/03-API-Reference/16-Serialization.md`

## Formats

- SafeTensors : format “production” (interop avec écosystème HF)
- RawFolder : debug (dossiers + checksums)
- DebugJson : inspection (statistiques)

## API Lua (résumé)

Les fonctions sont exposées sous `Mimir.Serialization`.

| Fonction | Rôle | Exemple |
| --- | --- | --- |
| `Mimir.Serialization.save(path)` | sauvegarde checkpoint (poids + artefacts associés) | `Mimir.Serialization.save("checkpoint/run1.safetensors")` |
| `Mimir.Serialization.load(path)` | charge un checkpoint | `Mimir.Serialization.load("checkpoint/run1.safetensors")` |
| `Mimir.Serialization.detect_format(path)` | détecte le format | `Mimir.Serialization.detect_format("...")` |
| `Mimir.Serialization.save_enhanced_debug(path)` | écrit un debug JSON enrichi | `Mimir.Serialization.save_enhanced_debug("debug.json")` |

## Bonnes pratiques

- Toujours sauvegarder la config d’architecture (type, dims, seq_len, vocab_size).
- Fixer `cfg.dtype` si tu veux contrôler le dtype de stockage sur disque (ex: `"float16"`).
- Valider les checksums en chargement si dispo.
- Ne pas changer `seq_len` ou `vocab_size` après entraînement sans stratégie explicite (sinon shapes incompatibles).

## Exemple : init → save → load (workflow moderne)

```lua
local cfg, err = Mimir.Architectures.default_config("transformer")
assert(cfg, err)
cfg.seq_len = 64
cfg.vocab_size = 2000
cfg.dtype = "float16"  -- contrôle dtype de stockage (runtime reste float32-first)

-- Création (construire le réseau automatiquement)
assert(Mimir.Model.create("transformer", cfg))
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier", 42))

-- Sauvegarde
assert(Mimir.Serialization.save("/tmp/checkpoint.safetensors"))

-- Chargement (plus tard, ou dans un autre run)
assert(Mimir.Model.create("transformer", cfg))  -- recréer le réseau
assert(Mimir.Model.allocate_params())
local ok, err = Mimir.Serialization.load("/tmp/checkpoint.safetensors")
assert(ok, err)
```

## Côté Lua

Chercher des exemples dans `scripts/` et la définition dans `src/LuaScripting.cpp` (module `Mimir.Serialization`).
