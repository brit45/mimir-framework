```markdown
# Internals : CLI (binaire `mimir`) et points d’entrée

Cette page documente le binaire CLI : parsing des arguments, chemins d’exécution, et conventions.

Source de vérité :

- Entry point : `src/main.cpp`
- Scripting Lua : `src/LuaScripting.hpp/.cpp`
- Config → modèle : `src/Models/Registry/ModelArchitectures.hpp/.cpp`
- Sécurité mémoire : `src/MemorySafety.hpp` + `src/MemoryGuard.hpp`

## 1) Options supportées

Dans `src/main.cpp`, la CLI expose :

- `--lua <script.lua>` : exécute un script Lua.
- `--config <config.json>` : crée un modèle depuis un JSON (via le registre) puis init (training “à implémenter”).
- `--help` : usage.

Les arguments après `--lua script.lua` sont passés au script (injectés comme `arg`).

## 2) Séquence de boot

Au démarrage :

- bannière de version,
- `MemorySafety::validateLegacyDisabled()` + `runMemoryIntegrityTest()`,
- affichage capacités CPU (AVX2/FMA/F16C/BMI2),
- puis dispatch selon les arguments.

## 3) Path `--lua`

- vérifie l’existence du fichier,
- instancie `LuaScripting`, injecte args, charge et exécute.

## 4) Path `--config`

- lit JSON,
- choisit `architecture` (default `t2i_autoencoder`),
- part de `ModelArchitectures::defaultConfig(name)` puis applique les overrides:
  - section `model` si présente,
  - section `config[arch_type]` si présente,
- `ModelArchitectures::create`, puis `allocateParams` + `initializeWeights("he")`.

```