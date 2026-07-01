# API Lua - vue d’ensemble

La source de vérité de l’API exposée est `src/scriptings/Lua/luaScripting/LuaScripting.cpp`.

Le contrat système commun (noms globaux + aliases, partagé entre langages de scripting) est défini dans `src/scriptings/ScriptingContext.hpp`.

## Modules principaux

- `Mimir.Model` : create/build/allocate/init/forward/backward + helpers d'entraînement + `dtype` + `create_from_config`
- `Mimir.Architectures` : liste et config par défaut des architectures (+ `info`, `dtypes`)
- `Mimir.Serialization` : save/load, formats, debug dumps
- `Mimir.Tokenizer` : tokenize/detokenize, BPE
- `Mimir.Dataset` : chargement et itération (selon scripts)
- `Mimir.IO` : lecture d'images (RGB u8)

Pages de référence associées :

- `Mimir.Model` : `docs/03-API-Reference/10-Model.md`
- `Mimir.Architectures` : `docs/03-API-Reference/11-Architectures.md`
- `Mimir.Tokenizer` : `docs/03-API-Reference/12-Tokenizer.md`
- `Mimir.Dataset` : `docs/03-API-Reference/13-Dataset.md`
- Mémoire : `docs/03-API-Reference/14-Memory.md`
- Viz/Htop : `docs/03-API-Reference/15-Viz-Htop.md`
- Sérialisation (détaillé) : `docs/03-API-Reference/16-Serialization.md`
- NeuroPulse : `docs/03-API-Reference/17-NeuroPulse.md`
- Ops layers : `docs/03-API-Reference/18-Layers-Module.md`
- Globals/aliases : `docs/03-API-Reference/19-Globals.md`
- Mapping Lua ↔ C++ (sommaire) : `docs/03-API-Reference/20-Lua-API-Cpp-Mapping.md`
- `Mimir.IO` (I/O images) : `docs/03-API-Reference/21-IO.md`

## Convention d’arguments

La plupart des API retournent soit :

- `true/false` + message d’erreur
- ou une table Lua + erreur

Les exemples de scripts dans `scripts/` restent les meilleurs “tests vivants” de l’API.
