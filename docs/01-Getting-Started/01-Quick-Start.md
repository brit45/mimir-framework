# Quick start

Objectif : exécuter un script Lua, créer un modèle via le registre, initialiser les poids, faire un forward.

## Pré-requis

- Linux
- CMake + compilateur C++17

## 1) Build

```bash
cmake -S . -B build
cmake --build build -j
```

## 2) Exécuter un script Lua

```bash
./bin/mimir --lua scripts/examples/vae_text_sample.lua --help
```

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

## 4) Où regarder ensuite

- `docs/02-User-Guide/02-Model-Lifecycle.md` pour comprendre le pipeline.
- `docs/03-API-Reference/02-Serialization.md` pour save/load.
