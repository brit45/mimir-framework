# Sérialisation : sauvegarde et chargement

Trouver rapidement le contrat API réel et les paramètres utilisables.

**Public concerné :** Développeur et utilisateur intermédiaire/avancé.

> **Prérequis**
>
> Connaître les commandes de base de Mímir.


Objectif : sauvegarder et restaurer un modèle + son tokenizer/encoder de manière fiable.

Pour la référence complète (formats, options, comportements), voir :

- `docs/03-API-Reference/16-Serialization.md`

## Formats

- SafeTensors : format “production” (interop avec écosystème HF)
- RawFolder : checkpoint natif en dossiers, utilisé aussi pour les reprises
- DebugJson : inspection (statistiques + snapshot `framework_state`, format enrichi v1.4)

MPK n’est pas un format de checkpoint de `Mimir.Serialization` : il transporte
une architecture et sa configuration, sans poids. Cette voie reste expérimentale
et partielle pour la production. Voir
[MPK : packages d’architecture](../02-User-Guide/15-MPK.md).

## API Lua (résumé)

Les fonctions sont exposées sous `Mimir.Serialization`.

| Fonction | Rôle | Exemple |
| --- | --- | --- |
| `Mimir.Serialization.save(path)` | sauvegarde checkpoint (poids + artefacts associés) | `Mimir.Serialization.save("checkpoint/run1.safetensors")` |
| `Mimir.Serialization.load(path)` | charge un checkpoint | `Mimir.Serialization.load("checkpoint/run1.safetensors")` |
| `Mimir.Serialization.detect_format(path)` | détecte le format | `Mimir.Serialization.detect_format("...")` |
| `Mimir.Serialization.save_enhanced_debug(path)` | écrit un debug JSON enrichi (v1.3: inclut `framework_state`) | `Mimir.Serialization.save_enhanced_debug("debug.json")` |

Pour reprendre l'optimiseur, utilisez explicitement `save_optimizer=true` puis
`load_optimizer=true` dans les tables d'options ; leurs défauts sont `false`.
La topologie et le tokenizer doivent rester ceux du checkpoint. Les scripts
d'entraînement gèrent en plus les curseurs, schedules et états propres au modèle.

## Bonnes pratiques

- Toujours sauvegarder la config d’architecture (type, dims, seq_len, vocab_size).
- Fixer `cfg.dtype` si vous voulez contrôler le dtype de stockage sur disque (ex: `"float16"`).
- Valider les checksums en chargement si dispo.
- Ne pas changer `seq_len` ou `vocab_size` après entraînement sans stratégie explicite (sinon shapes incompatibles).

## Exemple : init → save → load (workflow moderne)

```lua
local cfg, err = Mimir.Architectures.default_config("transformer")
assert(cfg, err)
cfg.seq_len = 64
cfg.vocab_size = 2000
cfg.dtype = "float16"  -- dtype du modèle et de ses exports compatibles

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

Chercher des exemples dans `scripts/` et la définition dans `src/scriptings/Lua/luaScripting/LuaScripting.cpp` (module `Mimir.Serialization`).

## Étapes suivantes

- [Page précédente : Layers](01-Layers.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : API : `Mimir.Model`](10-Model.md)
