```markdown
# Internals : registre d’architectures (`ModelArchitectures`) et builders

Cette page documente le registre d’architectures (config JSON → instance de `Model`) et la manière dont les topologies sont construites.

Source de vérité :

- API : `src/Models/Registry/ModelArchitectures.hpp`
- Implémentation : `src/Models/Registry/ModelArchitectures.cpp`
- Builders : `src/Models/**` (ex: MLP/Transformer/Diffusion)
- `Model::push` + routing : `src/Model.hpp`, `src/Model.cpp`

## 1) Concepts

- Une **architecture** = un nom + une config par défaut + une fonction `create(cfg)`.
- La config est un JSON (`nlohmann::json`).
- `create` instancie généralement une classe dérivée (ou configure un `Model`) et construit la liste de `Layer`.

## 2) API publique

Dans `ModelArchitectures.hpp` :

- `available()` : liste des archis.
- `defaultConfig(name)` : config par défaut.
- `create(name, cfg)` : instancie un modèle.

Sous le capot :

- `Registry::ensureBuiltinsRegistered()` enregistre les entrées (lazy via `std::once_flag`).

## 3) Invariants de build

- Le build doit définir `params_count` correctement pour chaque layer.
- Le wiring doit être cohérent (`Layer.inputs/output`).
- Après build :
  - `allocateParams()` doit pouvoir allouer les blocs.
  - `initializeWeights()` doit pouvoir initialiser selon la méthode.

## 4) Où le registre est utilisé

- CLI : `src/main.cpp` (`--config`)
- Lua : `LuaScripting` (`Mimir.Architectures.*` et `Mimir.Model.create`)

```