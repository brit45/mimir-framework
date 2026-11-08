# Variables globales et alias

Trouver rapidement le contrat API réel et les paramètres utilisables.

**Public concerné :** Développeur et utilisateur intermédiaire/avancé.

> **Prérequis**
>
> Connaître les commandes de base de Mímir.


Cette page documente les globals disponibles dans un script lancé avec
`./run_mimir.sh --lua ...`.

Objectif :

- vous permettre d’écrire des scripts courts (aliases globaux pratiques)
- rester compatible avec des scripts historiques
- éviter les ambiguïtés autour de `arg` et les collisions de noms

Source de vérité : `src/scriptings/Lua/luaScripting/LuaScripting.cpp`.

## 1) Globals injectés

### Table `arg`

Le binaire remplit une table globale `arg` “à la façon de Lua” :

- `arg[0]` = chemin du script
- `arg[1..n]` = arguments

Pour analyser ces valeurs, chargez `scripts/modules/args.lua` puis appelez
`Args.parse(arg)`. Le runtime n'injecte pas de table `Mimir.Args`.

### Configuration et racine de travail

- `CONF` : table de configuration injectée pour le script ;
- `CONF_PATH`, `CONF_DIR` : chemin et dossier de cette configuration, chaînes
  vides en l'absence de fichier ;
- `ROOTWORK` : chemin absolu normalisé issu de l'environnement `ROOTWORK`, ou
  du répertoire courant par défaut. Le lanceur le positionne sur le dépôt.

`CONF` est la copie remise à la VM. Modifier cette table ne modifie pas les
réglages d'un modèle déjà créé. Les entrées racine `env` de la configuration
agissent sur l'environnement du processus et peuvent donc concerner les scripts
suivants ; elles ne sont pas des variables locales au fichier Lua.

### Table `Mimir`

Le runtime publie la table `Mimir` avec les sous-modules (`Mimir.Model`, `Mimir.Serialization`, etc.).

### Fonctions utilitaires globales

Ces fonctions existent pour éviter d’importer un module juste pour des tâches basiques.

| Nom | Signature | Retour | À utiliser quand | Notes |
| --- | --- | --- | --- | --- |
| `log` | `log(msg: string)` | rien | logs côté runtime Mímir | alias d’un `print` runtime |
| `read_json` | `read_json(path: string)` | `table` ou `(nil, err)` | charger une config/metadata | parse JSON -> table Lua |
| `write_json` | `write_json(path: string, table)` | `true` ou `(false, err)` | écrire un état/rapport | sérialise table Lua -> JSON |

## 2) Aliases globaux (rétrocompat + confort)

En plus de `Mimir.*`, le runtime publie des **aliases globaux** qui pointent vers les sous-tables `Mimir.*`.

| Alias global | Pointe vers | Pourquoi ça existe | Recommandation |
| --- | --- | --- | --- |
| `Mimir.model` | `Mimir.Model` | alias lowercase pour scripts compacts (v3.0) | OK pour scripts jetables, préfère `Mimir.Model` en projet long |
| `model` | `Mimir.Model` | scripts courts, legacy | OK pour scripts jetables, préfère `Mimir.Model` en projet long |
| `architectures` | `Mimir.Architectures` | idem | idem |
| `tokenizer` | `Mimir.Tokenizer` | idem | idem |
| `dataset` | `Mimir.Dataset` | idem | idem |
| `Memory` | `Mimir.Memory` | legacy + quick debug | préfère `Mimir.Memory` |
| `MemoryGuard` | `Mimir.MemoryGuard` | limite stricte mémoire | recommandé (sécurité) |
| `Allocator` | `Mimir.Allocator` | allocateur dynamique | recommandé (stabilité) |
| `htop` | `Mimir.Htop` | monitoring terminal | optionnel |
| `viz` | `Mimir.Viz` | visualisation avec le backend compilé | optionnel |

## 3) Conseils (éviter les pièges)

- Ne redéfinissez pas ces noms globalement (`model = ...`, `MemoryGuard = ...`).
  Pour une variable de script, utilisez `local model = ...`.
- Pour éviter les collisions, vous pouvez adopter ce pattern :

```lua
local M = Mimir
local Model = Mimir.Model
local Ser = Mimir.Serialization
```

- Pour vérifier les noms publics, consultez `registerAPI()` dans
  `src/scriptings/Lua/luaScripting/LuaScripting.cpp`, puis le binding concerné dans
  les fichiers `LuaScripting*.cpp` voisins pour ses arguments et retours.

## Étapes suivantes

- [Page précédente : API : `Mimir.Layers` (ops)](18-Layers-Module.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Mapping API Lua → C/C++ (bindings) → sous-systèmes](20-Lua-API-Cpp-Mapping.md)
