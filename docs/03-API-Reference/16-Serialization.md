# API : `Mimir.Serialization`

Source : `src/scriptings/Lua/luaScripting/LuaScripting.cpp` et `src/Serialization/*`.

## Formats

- `safetensors` (alias: `st`)
- `raw_folder` (alias: `raw`, `folder`)
- `debug_json` (alias: `debug`, `json`)

## `save(path: string, format?: string="safetensors", opts?: table) -> bool | (false, err)`

Options (principales) :

- `save_tokenizer` (bool)
- `save_encoder` (bool)
- `save_optimizer` (bool)
- `include_git_info` (bool)
- `include_checksums` (bool)

Options DebugJson “enhanced” :

- `include_gradients` (bool)
- `include_optimizer_state` (bool)
- `include_activations` (bool)
- `include_weight_deltas` (bool)
- `max_values_per_tensor` (int)

## DType (stockage sur disque)

Le framework est **runtime float32-first** (poids/activations en RAM). Le `dtype` du modèle sert surtout à contrôler le **dtype de sérialisation** des tenseurs float lors de `save()`.

- Source : `Model.default_dtype` (exposé via `Mimir.Model.dtype()`).
- Valeurs usuelles : `"float16"`, `"bfloat16"`, `"float32"`, `"float64"`.

Règles :

- `save()` (SafeTensors/RawFolder) convertit les tenseurs float du modèle en fonction de `Model.default_dtype`.
- `load()` convertit les tenseurs sur disque vers la représentation runtime (typiquement float32), mais **réapplique** `model_config.dtype` au modèle si présent dans les métadonnées (pour que les futurs `save()` restent cohérents).

Bon pattern :

- mettre `cfg.dtype` dans la config passée à `Mimir.Model.create(...)`.
- ou appeler `Mimir.Model.dtype("float16")` après `create()` si tu veux forcer un override.

## `load(path: string, format?: string|"auto", opts?: table) -> bool | (false, err)`

- Si `format` absent : auto-détection.

Options :

- `load_tokenizer`, `load_encoder`, `load_optimizer`
- `strict_mode`
- `validate_checksums`

## `detect_format(path: string) -> string | (nil, err)`

Retourne : `SAFETENSORS`, `RAWFOLDER`, `DEBUGJSON`.

## `save_enhanced_debug(path: string, opts?: table) -> bool | (false, err)`

Écrit un JSON d’inspection (stats + options avancées).
