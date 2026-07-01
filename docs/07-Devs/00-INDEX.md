# Devs - Index

Cette section est le point d'entree principal pour les developpeurs du framework.

Objectif : expliquer comment le framework fonctionne de l'interieur et comment ajouter/modifier des briques sans casser la coherence globale.

Chaque chapitre contient des demos orientees contraintes metier : reproductibilite, stabilite de contrat, fallback runtime, et tracabilite config.

## Parcours recommande

1. Lire le fonctionnement general du framework.
2. Lire la construction des modeles (`model.push(...)`, I/O des layers).
3. Lire la composition de config et l'enregistrement au registre.
4. Lire la section runtime (modification et ajout).
5. Lire le contrat API scripting (stabilite inter-langages).
6. Executer les demos de chaque chapitre avant merge.

## Chapitres

- Fonctionnement du framework : [docs/07-Devs/01-How-The-Framework-Works.md](01-How-The-Framework-Works.md)
- Construire un modele (model.push + set_layer_io) : [docs/07-Devs/02-Building-Models-And-Layers.md](02-Building-Models-And-Layers.md)
- Config et registre : [docs/07-Devs/03-Config-And-Registry.md](03-Config-And-Registry.md)
- Runtime (modifier / ajouter) : [docs/07-Devs/04-Runtime-Development.md](04-Runtime-Development.md)
- Contrat API scripting : [docs/07-Devs/05-Scripting-System-Contract.md](05-Scripting-System-Contract.md)

## Sources code de reference

- Moteur modele : `src/Model.hpp`, `src/Model.cpp`
- Registre archis : `src/Models/Registry/ModelArchitectures.hpp`, `src/Models/Registry/ModelArchitectures.cpp`
- Runtime abstrait : `src/runtimes/AbstractRuntime.hpp`
- Scripting (contrat commun) : `src/scriptings/ScriptingContext.hpp`, `src/scriptings/ScriptingRuntime.hpp`
- Bridge Lua actuel : `src/scriptings/Lua/luaScripting/LuaScripting.hpp`, `src/scriptings/Lua/luaScripting/LuaScripting.cpp`
