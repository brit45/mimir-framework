# Organisation du dépôt

## Pour qui

Débutant qui découvre le dépôt.

## Objectif

Trouver vite où modifier selon le besoin (scripts, modèle, runtime, docs).

## Avant de commencer

Aucun prérequis technique fort.

## Résultat attendu

Tu navigues le projet sans te perdre.


Repères utiles pour naviguer vite.

## Répertoires

- `src/` : runtime C++ (Model, layers, backends, sérialisation, bindings Lua).
- `scripts/` : scripts Lua (training, demos, examples, modules partagés).
- `docs/` : documentation (réécrite).
- `docs_archive/` : snapshots de l’ancienne documentation.
- `checkpoint/` / `checkpoints/` : checkpoints produits localement.
- `build/` : build CMake (généré).
- `shaders/` : shaders (si backends GPU/compute activés).

## Fichiers clefs (source de vérité)

- API Lua : `src/scriptings/Lua/luaScripting/LuaScripting.cpp`
- Runtime modèle / exécution layers : `src/Model.cpp`
- Définition layers : `src/Layers.hpp`, `src/LayerTypes.hpp`
- Registre des architectures : `src/Models/Registry/ModelArchitectures.cpp`
- Sérialisation : `src/Serialization/*`
- CLI : `src/main.cpp`

## Scripts importants

- Exemples : `scripts/examples/`
- Entraînement : `scripts/training/`
- Modules partagés : `scripts/modules/` (args, tokenizer base, resume checkpoint, etc.)
