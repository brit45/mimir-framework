# Développeurs : Ajouter une Architecture et Utiliser les Outils

Cette page explique, de manière opérationnelle, comment :

- ajouter une nouvelle architecture de modèle dans le framework,
- l'enregistrer dans le registre C++,
- écrire un script Lua de training/inférence pour l'utiliser,
- utiliser les scripts de `scripts/tools/`.

## 1. Ajouter une nouvelle architecture C++

### 1.1 Créer la classe modèle

Créer un modèle dans `src/Models/...` (par exemple `src/Models/MyDomain/MyNewModel.hpp` et `src/Models/MyDomain/MyNewModel.cpp`).

Bonnes pratiques minimales :

- définir une `Config` interne avec des valeurs par défaut raisonnables,
- construire explicitement la topologie dans `build()` (ou équivalent),
- garantir que le modèle est compatible avec :
  - allocation des paramètres,
  - initialisation des poids,
  - forward,
  - sérialisation.

Exemple de squelette :

```cpp
class MyNewModel : public Model {
public:
    struct Config {
        int d_model = 256;
        int num_layers = 4;
        int output_dim = 1024;
    };

    explicit MyNewModel(const Config& cfg);
    void build();

private:
    Config cfg_;
};
```

## 2. Enregistrer l'architecture dans le registre

Le registre central est :

- `src/Models/Registry/ModelArchitectures.hpp`
- `src/Models/Registry/ModelArchitectures.cpp`

### 2.1 Inclure la nouvelle classe

Dans `src/Models/Registry/ModelArchitectures.cpp`, ajouter l'include de la nouvelle classe.

### 2.2 Mapper JSON -> Config

Ajouter une fonction helper du style `myNewCfgFromJson(const json& cfg)` qui lit les champs de config avec fallback.

### 2.3 Définir la config par défaut

Ajouter une fonction `myNewDefaultConfigJson()` retournant les champs JSON attendus par le modèle.

### 2.4 Ajouter l'entrée du registre

Dans l'enregistrement des builtins, ajouter une entrée `Entry` avec :

- `name` : nom CLI/Lua de l'architecture,
- `description` : description courte,
- `default_config` : config JSON par défaut,
- `create` : lambda/fonction qui instancie le modèle.

Exemple conceptuel :

```cpp
registry.registerArchitecture({
    "my_new_model",
    "Mon nouveau modèle",
    myNewDefaultConfigJson(),
    [](const json& cfg) {
        auto c = myNewCfgFromJson(cfg);
        return std::make_shared<MyNewModel>(c);
    }
});
```

### 2.5 Vérifier la visibilité API

Valider que l'architecture est visible côté runtime :

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -a
```

## 3. Écrire un script Lua pour utiliser l'architecture

Partir de :

- `scripts/templates/template_new_model.lua` (API directe bas niveau)
- `scripts/templates/template_pipeline_only.lua` (pipeline minimal)
- `scripts/templates/template_pipeline_args.lua` (pipeline + args + registry-first)

### 3.1 Mode pipeline + registry-first (recommandé)

C'est le flux moderne via `scripts/modules/pipeline.lua`.

Pattern simple :

```lua
local P = dofile("scripts/modules/pipeline.lua")

-- Créer un pipeline générique basé sur le registre.
local pipe, err = P.FromRegistry("my_new_model")
if not pipe then error(err) end

-- Charger la config par défaut du registre.
local ok, cfg = pipe:loadDefaultConfig("my_new_model")
if not ok then error(cfg) end

-- Appliquer des patches/overrides locaux.
local ok, _ = pipe:patchConfig({ d_model = 256, num_layers = 4 })
if not ok then error(_) end

-- Build, train, save.
pipe:build()
pipe:train("dataset.bin", 10, 0.0003)
pipe:save("checkpoint/my_new_model.safetensors")
```

Avantages :

- No hardcoding d'architecture dans le script.
- Réutilisable pour n'importe quelle archi du registre.
- Config chargée depuis le C++ = garantie de cohérence.

### 3.2 Avec parseur d'arguments (template complet)

See [scripts/templates/template_pipeline_args.lua](../../../scripts/templates/template_pipeline_args.lua) pour un exemple complet avec `--from-registry` et `--arch`.

Usage :

```bash
./bin/mimir --lua scripts/templates/template_pipeline_args.lua -- \
  --from-registry --arch transformer \
  --d-model 256 --layers 4 --heads 8 --seq-len 128 \
  --override mlp_hidden=1024 \
  --dataset dataset.bin --epochs 10 --lr 0.0003 \
  --save checkpoint/run.safetensors
```

## 4. Checklist d'intégration

Avant de merger une nouvelle architecture :

- vérifier l'entrée dans `inspect_architectures.lua`,
- vérifier le cycle create -> allocate -> init -> forward,
- vérifier un run de training court,
- vérifier save/load du checkpoint,
- documenter les paramètres principaux dans la doc API,
- ajouter au moins un script smoke test.

## 5. Scripts outils (`scripts/tools/*.lua`)

Ces scripts sont pensés pour le debug, la conversion d'artefacts et l'inspection rapide.

### 5.1 inspect_architectures.lua

Rôle : lister les architectures disponibles, dtypes, paramètres de config, couches.

Usage typique :

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -a
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -l vae_conv -p --layers
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- --json
```

Options clés : `-a`, `-l`, `-p`, `--layers`, `--stats`, `-d`, `--json`.

### 5.2 analyze_model.lua

Rôle : analyser un checkpoint/model (RawFolder, SafeTensors, Debug JSON), afficher synthèse utile (composants, tensors, tailles, etc.).

Usage typique :

```bash
./bin/mimir --lua scripts/tools/analyze_model.lua -- --in checkpoint/vae_conv_run
./bin/mimir --lua scripts/tools/analyze_model.lua -- --in checkpoint/model.safetensors
```

### 5.3 build_tags_vocab.lua

Rôle : construire un vocabulaire de tags à partir des fichiers `.txt` d'un dataset.

Usage typique :

```bash
./bin/mimir --lua scripts/tools/build_tags_vocab.lua -- \
  --dataset-root dataset_2 \
  --out checkpoint/tags_vocab.txt \
  --min-freq 2 --top-k 5000
```

Options utiles : `--lowercase`, `--min-freq`, `--top-k`, `--max-files`.

### 5.4 convert_checkpoint2safetensor.lua

Rôle : convertir un checkpoint RawFolder vers SafeTensors.

Usage typique :

```bash
./bin/mimir --lua scripts/tools/convert_checkpoint2safetensor.lua -- \
  --checkpoint checkpoint/vae_conv_run \
  --out checkpoint/vae_conv_run.safetensors
```

Note : le script reconstruit un modèle depuis l'architecture du checkpoint avant de charger puis de sauvegarder.

### 5.5 convert_safetensors2raw_folder.lua

Rôle : convertir un checkpoint SafeTensors vers RawFolder.

Usage typique :

```bash
./bin/mimir --lua scripts/tools/convert_safetensors2raw_folder.lua -- \
  --in checkpoint/model.safetensors \
  --out checkpoint/model_raw
```

### 5.6 show-graph.lua

Rôle : générer un rapport HTML interactif (Chart.js) à partir des CSV d'entraînement.

Usage typique :

```bash
./bin/mimir --lua scripts/tools/show-graph.lua -- checkpoints/loss_history.csv
./bin/mimir --lua scripts/tools/show-graph.lua -- --csv-dir checkpoints --watch --out graph_report.html
```

Options clés : `--csv`, `--csv-dir`, `--model`, `--algo`, `--checkpoint-dir`, `--out`, `--watch`, `--watch-interval`.

## 6. Commandes rapides pour un dev

Lister les archis + vérifier la nouvelle :

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -a
```

Tester conversion checkpoint :

```bash
./bin/mimir --lua scripts/tools/convert_checkpoint2safetensor.lua -- --checkpoint checkpoint/my_run --out checkpoint/my_run.safetensors
./bin/mimir --lua scripts/tools/convert_safetensors2raw_folder.lua -- --in checkpoint/my_run.safetensors --out checkpoint/my_run_raw
```

Analyser un artefact :

```bash
./bin/mimir --lua scripts/tools/analyze_model.lua -- --in checkpoint/my_run
```
