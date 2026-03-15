# Lua Scripting (Scripts Mímir)

Cette page explique **comment écrire** et **comment exécuter** des scripts Lua avec Mímir.

> Pour tout script qui utilise `Mimir.*`, lancez-le via le binaire `./bin/mimir`.

---

## Exécuter un script

### Exécution simple

```bash
./bin/mimir --lua scripts/examples/example_simple.lua
```

### Passer des arguments à un script

Le binaire passe tous les arguments situés **après** le chemin du script dans la table globale Lua `arg`.

Convention recommandée:

```bash
./bin/mimir --lua scripts/modules/api_ws_server.lua -- --host 127.0.0.1 --port 8088
```

Le séparateur `--` est optionnel, mais il aide à distinguer:
- les options du binaire (`--lua`, `--config`)
- des options du script.

---

## Parser des flags (`scripts/modules/args.lua`)

Un parseur de flags minimal est disponible: `scripts/modules/args.lua`.

Exemple d'usage:

```lua
local Args = dofile("scripts/modules/args.lua")
local opts, pos = Args.parse(arg)

local epochs = Args.get_int(opts, "epochs", 10)
local lr = Args.get_num(opts, "lr", 1e-4)
```

Flags supportés:
- `--k v`
- `--k=v`
- `--flag` (booléen)
- `--no-flag` (booléen)
- `--` (séparateur, ignoré)

---

## Squelette de script (recommandé)

### 1) Runtime: mémoire + hardware

```lua
Mimir.Allocator.configure({ max_ram_gb = 10.0, enable_compression = true })

local caps = Mimir.Model.hardware_caps()
if caps.avx2 or caps.fma then
  pcall(Mimir.Model.set_hardware, true)
end
```

### 2) Dataset

```lua
local ok, err = Mimir.Dataset.load("dataset")
if not ok then error(err) end

local ok2, err2 = Mimir.Dataset.prepare_sequences(256)
if not ok2 then error(err2) end
```

### 3) Modèle (registre)

```lua
local cfg, err = Mimir.Architectures.default_config("transformer")
if not cfg then error(err) end

local ok_create, create_err = Mimir.Model.create("transformer", cfg)
if not ok_create then error(create_err) end

-- build() reconstruit via registre (compat). Il ne remplace pas allocate/init.
local ok_build, build_err = Mimir.Model.build()
if not ok_build then error(build_err) end

local ok_alloc, alloc_info = Mimir.Model.allocate_params()
if not ok_alloc then error(alloc_info) end

local ok_init, init_err = Mimir.Model.init_weights("xavier", 42)
if ok_init == false then error(init_err) end
```

### 4) Entraînement + sauvegarde

```lua
local ok_train, train_err = Mimir.Model.train(10, 1e-4)
if ok_train == false then error(train_err) end

Mimir.Serialization.save("checkpoints/run.safetensors", "safetensors")
```

---

## Bonnes pratiques

- Gardez les scripts **déterministes**: passez un `seed` à `init_weights`.
- Vérifiez systématiquement `(ok, err)` quand une fonction peut échouer.
- Utilisez le registre `Mimir.Architectures.default_config(name)` pour partir d'une config valide.
- Préparez le dataset (`prepare_sequences(max_len)`) avant `Model.train(...)`.
