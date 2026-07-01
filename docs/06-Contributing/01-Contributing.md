# Contribuer

Compléments développeurs :

- Ajouter une architecture + registry + script Lua + outils : [docs/06-Contributing/02-New-Architecture-And-Tools.md](02-New-Architecture-And-Tools.md)

## Philosophie

- Changements petits, testables.
- Préférer corriger la cause racine.

## Conventions

- Ajouter une doc courte quand on ajoute une feature.
- Mettre un script smoke test si possible.

## Où modifier

- Nouveau layer : `src/Layers.hpp` + exécution/backward dans `src/Model.cpp`
- Nouvelle architecture : `src/Models/Registry/ModelArchitectures.cpp`
- API Lua : `src/LuaScripting.cpp`
