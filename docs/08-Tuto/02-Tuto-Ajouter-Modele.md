# Tuto - Ajouter un modele

## Pour qui

Debutant/intermediaire qui veut ajouter une nouvelle architecture dans Mimir.

## Objectif

Ajouter un modele C++, l'enregistrer dans le registre, puis le lancer avec un script.

## Avant de commencer

1. Build fonctionnel de Mimir.
2. Lecture conseillee: [docs/06-Contributing/02-New-Architecture-And-Tools.md](../06-Contributing/02-New-Architecture-And-Tools.md).
3. Connaissances C++ de base (classe, struct, include).

## Résultat attendu

Ton architecture apparait dans la liste du registre et peut etre instanciee depuis Lua.

## Etape 1 - Creer la classe du modele

1. Creer les fichiers dans un dossier logique, par exemple:
- `src/Models/MyDomain/MyNewModel.hpp`
- `src/Models/MyDomain/MyNewModel.cpp`

2. Ajouter une config simple dans la classe:
- dimensions principales,
- nombre de couches,
- valeurs par defaut raisonnables.

3. Implementer la construction du graphe dans `buildFromConfig(...)` (ou helper equivalent appele par le registre).

A retenir: commencer petit. Une version minimale qui compile et forward vaut mieux qu'un gros modele incomplet.

## Etape 2 - Enregistrer dans le registre d'architectures

Fichiers de reference:
- `src/Models/Registry/ModelArchitectures.hpp`
- `src/Models/Registry/ModelArchitectures.cpp`

Checklist:
1. Ajouter l'include de ta classe.
2. Mapper config JSON vers ta `Config` C++.
3. Definir une config par defaut JSON.
4. Ajouter l'entree registre (`name`, `description`, `default_config`, `create`).

## Etape 3 - Verifier que le modele est visible

Commande simple:

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -a
```

Tu dois voir le nom de ton architecture.

## Etape 4 - Lancer un script minimal

Point de depart recommande:
- `scripts/templates/template_new_model.lua`

Objectif du premier run:
1. create
2. allocate params
3. init weights
4. forward

## Exemple pratique - Squelette complet

### Contexte

Voici un squelette fonctionnel minimal que tu peux compiler et utiliser immediatement. Remplace les noms et ajuste les dimensions selon tes besoins.

### Header (`src/Models/MyDomain/MySimpleModel.hpp`)

```cpp
#pragma once

#include "../Model.hpp"

class MySimpleModel : public Model {
public:
    struct Config {
        // Dimensionnalité de l'entrée
        int input_dim = 256;
        // Dimensionnalité de la couche cachee
        int hidden_dim = 512;
        // Dimensionnalité de la sortie
        int output_dim = 128;
    };

    MySimpleModel();
    ~MySimpleModel() override = default;

    void buildFromConfig(const Config& cfg);

private:
    Config cfg_;
};
```

### Code (`src/Models/MyDomain/MySimpleModel.cpp`)

```cpp
#include "MySimpleModel.hpp"

MySimpleModel::MySimpleModel() {
    setModelName("MySimpleModel");
    setHasEncoder(false);
}

void MySimpleModel::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    
    // Efface les couches precedentes
    getMutableLayers().clear();
    
    // Fixe la config du modele
    setModelName("MySimpleModel");
    modelConfig["type"] = "my_simple_model";
    modelConfig["input_dim"] = cfg_.input_dim;
    modelConfig["hidden_dim"] = cfg_.hidden_dim;
    modelConfig["output_dim"] = cfg_.output_dim;
    
    // Couche 1: lineaire entree -> cachee
    // push(name, type, params_count)
    // params_count = in * out + out (weights + biases)
    size_t linear1_params = cfg_.input_dim * cfg_.hidden_dim + cfg_.hidden_dim;
    push("simple/fc1", "Linear", linear1_params);
    if (auto* L = getLayerByName("simple/fc1")) {
        L->inputs = {"__input__"};  // lit de l'entree
        L->output = "simple/h1";    // produit "simple/h1"
        L->in_features = cfg_.input_dim;
        L->out_features = cfg_.hidden_dim;
        L->use_bias = true;
    }
    
    // Couche 2: activation ReLU
    push("simple/relu1", "ReLU", 0);  // pas de params pour ReLU
    if (auto* L = getLayerByName("simple/relu1")) {
        L->inputs = {"simple/h1"};  // lit de h1
        L->output = "simple/a1";    // produit "simple/a1"
    }
    
    // Couche 3: lineaire cachee -> sortie
    size_t linear2_params = cfg_.hidden_dim * cfg_.output_dim + cfg_.output_dim;
    push("simple/fc2", "Linear", linear2_params);
    if (auto* L = getLayerByName("simple/fc2")) {
        L->inputs = {"simple/a1"};  // lit de a1
        L->output = "x";             // produit la sortie finale
        L->in_features = cfg_.hidden_dim;
        L->out_features = cfg_.output_dim;
        L->use_bias = true;
    }
}
```

### Enregistrement (dans `src/Models/Registry/ModelArchitectures.cpp`)

Ajoute d'abord les helpers pour parser la config JSON (au début du fichier, avant `ensureBuiltinsRegistered()`):

```cpp
// Helper robuste pour extraire les valeurs JSON
template <typename T>
static T jget(const json& j, const char* key, T def) {
    if (!j.is_object()) return def;
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return def;
    try {
        return it->get<T>();
    } catch (...) {
        return def;
    }
}

// Config par defaut JSON
static json mySimpleModelDefaultConfigJson() {
    MySimpleModel::Config d;  // instance par defaut
    return json{
        {"input_dim", d.input_dim},
        {"hidden_dim", d.hidden_dim},
        {"output_dim", d.output_dim},
    };
}

// Mapper JSON -> Config C++
static MySimpleModel::Config mySimpleModelCfgFromJson(const json& cfg) {
    MySimpleModel::Config out;
    out.input_dim = jget<int>(cfg, "input_dim", out.input_dim);
    out.hidden_dim = jget<int>(cfg, "hidden_dim", out.hidden_dim);
    out.output_dim = jget<int>(cfg, "output_dim", out.output_dim);
    return out;
}
```

Puis dans `ensureBuiltinsRegistered()`, ajoute:

```cpp
entries_.emplace(
    "my_simple_model",
    Entry{
        "my_simple_model",
        "Simple linear model template",
        mySimpleModelDefaultConfigJson(),
        [](const json& cfg) -> std::shared_ptr<Model> {
            auto m = std::make_shared<MySimpleModel>();
            m->buildFromConfig(mySimpleModelCfgFromJson(cfg));
            return m;
        },
    }
);
```

### Explication

1. **Header** : classe héritant de `Model`, Config struct avec les hyperparam.
2. **Constructeur** : appelle `setModelName()` et `setHasEncoder()`.
3. **buildFromConfig()** : 
   - efface les couches avec `getMutableLayers().clear()`
   - stocke la config dans `modelConfig` (JSON interne pour save/load)
   - ajoute chaque couche avec `push(name, type, params_count)`
   - récupère la couche avec `getLayerByName()` et configure:
     - `inputs`: liste des tenseurs d'entrée (ou `{"__input__"}` pour l'entrée du modèle)
     - `output`: nom du tenseur de sortie
     - `in_features`, `out_features`: dimensions (pour Linear)
     - `use_bias`: si les biais sont présents
4. **Registre** : helpers config + lambda qui crée le modèle puis appelle `buildFromConfig()`.

### Test rapide

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -a
```

Verification attendue: le nom `my_simple_model` apparait dans la liste.

## Etape 5 - Validation rapide

Tu peux valider si:
1. la compilation passe,
2. l'architecture est visible dans `inspect_architectures.lua`,
3. un forward simple passe sans crash,
4. save/load checkpoint fonctionne sur un mini test.

## Erreurs frequentes

1. Nom d'architecture incoherent entre registre et script.
2. Parametre de config absent ou mal type.
3. Build du modele incomplet (couches non branchees).
4. Formes de tenseurs incompatibles dans le forward.

## Suite

- Guide complet contribution: [docs/06-Contributing/02-New-Architecture-And-Tools.md](../06-Contributing/02-New-Architecture-And-Tools.md)
- Lifecycle modele: [docs/02-User-Guide/02-Model-Lifecycle.md](../02-User-Guide/02-Model-Lifecycle.md)
