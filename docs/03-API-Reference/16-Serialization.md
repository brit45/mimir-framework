# API : `Mimir.Serialization`

Source : `src/LuaScripting.cpp` et `src/Serialization/*`.

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
