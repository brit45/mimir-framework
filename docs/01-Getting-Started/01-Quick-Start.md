# Quick start

Objectif : exécuter un script Lua, créer un modèle via le registre, initialiser les poids, faire un forward.

Si tu ne sais pas par où commencer, suis cette page de haut en bas sans improviser. Elle couvre le plus petit chemin utile pour vérifier que le runtime, l’API Lua et le registre d’architectures fonctionnent ensemble.

## Pré-requis

- Linux
- CMake + compilateur C++17

## 1) Build

```bash
cmake -S . -B build
cmake --build build -j
```

Si le build passe mais que tu veux un contrôle rapide avant de lire le reste de la doc :

```bash
./bin/mimir --help
```

Tu dois obtenir l’aide CLI et la détection matérielle au démarrage, sans crash immédiat.

## 2) Exécuter un script Lua

```bash
./bin/mimir --lua scripts/examples/vae_text_sample.lua --help
```

Ce test sert surtout à vérifier le câblage CLI → Lua. Pour une validation plus représentative du code actuellement maintenu, tu peux aussi lancer :

```bash
./bin/mimir --lua scripts/templates/template_new_model.lua
./bin/mimir --lua scripts/tests/test_serialization_smoke.lua
```

Le template vérifie le chemin “création de modèle”, le smoke test vérifie le chemin “save/load”. Ensemble, ils donnent un meilleur signal qu’un simple `--help`.

## 3) Créer un modèle (via le registre)

Exemple : Transformer simple.

```lua
local cfg, err = Mimir.Architectures.default_config("transformer")
if not cfg then error(err) end

cfg.vocab_size = 8000
cfg.seq_len = 128
cfg.d_model = 256
cfg.num_layers = 4
cfg.num_heads = 8
cfg.mlp_hidden = 1024
cfg.output_dim = 256

assert(Mimir.Model.create("transformer", cfg))
assert(Mimir.Model.build())
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier", 42))

-- forward avec ids int (Embedding) si l’archi le supporte,
-- sinon forward float: utilise __input__.
local ids = {}
for i = 1, cfg.seq_len do ids[i] = 1 end
local out = Mimir.Model.forward({ __input__ = ids }, false)
print("out_len=", #out)
```

Pourquoi cet exemple est structuré comme ça :

- `default_config()` évite d’oublier des champs attendus par le builder.
- `create()` fixe le type et la config dans le contexte runtime.
- `build()` matérialise la structure des layers.
- `allocate_params()` réserve les poids.
- `init_weights()` met le modèle dans un état exploitable avant le premier `forward()`.

Si tu remplaces l’entrée par une table map nommée, tu gardes un script compatible avec les architectures multi-input. C’est la convention recommandée même pour un seul tenseur d’entrée.

### Variante courte avec sauvegarde

```lua
local ok, save_err = Mimir.Serialization.save("checkpoint/quickstart_model", "safetensors")
assert(ok ~= false, save_err)
print("checkpoint écrit")
```

Cette étape est utile si tu veux valider tout de suite que la config, l’allocation et la sérialisation sont cohérentes dans ton environnement.

## 4) Où regarder ensuite

- `docs/02-User-Guide/02-Model-Lifecycle.md` pour comprendre le pipeline.
- `docs/03-API-Reference/02-Serialization.md` pour save/load.
- `docs/02-User-Guide/09-Memory.md` si ton premier vrai modèle dépasse vite la RAM disponible.
