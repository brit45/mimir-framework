# Ajouter un modèle

Ce tutoriel décrit le chemin réellement utilisé par `basic_mlp` : définir une
classe dérivée de `Model`, construire son graphe, convertir sa configuration
JSON et l'enregistrer dans `ModelArchitectures`. Le graphe DL est défini
exclusivement en C/C++ dans cette classe dérivée ; Lua et JSON configurent et
pilotent le modèle. L'autre voie, MPK, reste expérimentale et n'est pas encore
entièrement exploitable en production. Voir le
[contrat des architectures](../07-Devs/08-Architecture-Contract.md).

> **Prérequis**
>
> Le projet doit compiler et `./bin/mimir --lua
> scripts/tools/inspect_architectures.lua -- -a` doit fonctionner.

## Choisir son niveau

| Niveau | Travail demandé | Résultat attendu |
| --- | --- | --- |
| Débutant | Lire `BasicMLPModel` et identifier classe, config et factory | Expliquer qui définit le graphe et qui lance le run |
| Intermédiaire | Ajouter la classe et ses trois couches ci-dessous | Passe avant avec dimensions cohérentes |
| Avancé | Relier la configuration JSON, le registre et les checkpoints | Même topologie avant et après sauvegarde/relecture |
| Expert | Ajouter objectifs, gradients et tests négatifs propres au modèle | Erreurs contrôlées et validation numérique ciblée |

## Sources de vérité

- `src/Models/MLP/BasicMLPModel.hpp`
- `src/Models/MLP/BasicMLPModel.cpp`
- `src/Models/Registry/ModelArchitectures.cpp`
- `src/Model.hpp`
- `src/Layers.hpp`
- `Tests/test_registry_cfg_from_config.cpp`
- `Tests/test_registry_create_from_config.cpp`
- `Tests/test_registry_aliases.cpp`

Les noms utilisés ci-dessous sont illustratifs. Les fichiers `MySimpleModel.*`
ne sont pas fournis : vous devez les créer.

## Étape 1 — Définir la configuration

Créez `src/Models/MyDomain/MySimpleModel.hpp` sur le modèle de
`BasicMLPModel.hpp` :

```cpp
#pragma once

#include "Model.hpp"

class MySimpleModel : public Model {
public:
    struct Config {
        int input_dim = 64;
        int hidden_dim = 128;
        int output_dim = 16;
    };

    MySimpleModel();
    void buildFromConfig(const Config& cfg);
    static void buildInto(Model& model, const Config& cfg);

private:
    Config cfg_;
};
```

`buildInto` est un helper statique appartenant à la classe dérivée. Il centralise
la description du graphe et est appelé par `buildFromConfig` sur cette instance.
Il ne dispense pas d'hériter de `Model`. C'est le pattern employé par
`BasicMLPModel`.

## Étape 2 — Construire le graphe

Dans `src/Models/MyDomain/MySimpleModel.cpp`, reproduisez le contrat réel de
`Model::push` et des champs de `Layer` :

```cpp
#include "MySimpleModel.hpp"
#include <stdexcept>

MySimpleModel::MySimpleModel() {
    setModelName("MySimpleModel");
    setHasEncoder(false);
}

void MySimpleModel::buildFromConfig(const Config& cfg) {
    if (cfg.input_dim <= 0 || cfg.hidden_dim <= 0 || cfg.output_dim <= 0) {
        throw std::invalid_argument("MySimpleModel: dimensions must be positive");
    }
    cfg_ = cfg;
    buildInto(*this, cfg_);
}

void MySimpleModel::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("MySimpleModel");
    model.modelConfig["type"] = "my_simple_model";
    model.modelConfig["input_dim"] = cfg.input_dim;
    model.modelConfig["hidden_dim"] = cfg.hidden_dim;
    model.modelConfig["output_dim"] = cfg.output_dim;

    model.push(
        "my_simple/fc1",
        "Linear",
        static_cast<size_t>(cfg.input_dim) * cfg.hidden_dim + cfg.hidden_dim
    );
    if (auto* layer = model.getLayerByName("my_simple/fc1")) {
        layer->inputs = {"__input__"};
        layer->output = "my_simple/hidden";
        layer->in_features = cfg.input_dim;
        layer->out_features = cfg.hidden_dim;
        layer->use_bias = true;
    }

    model.push("my_simple/gelu", "GELU", 0);
    if (auto* layer = model.getLayerByName("my_simple/gelu")) {
        layer->inputs = {"my_simple/hidden"};
        layer->output = "my_simple/activated";
    }

    model.push(
        "my_simple/out",
        "Linear",
        static_cast<size_t>(cfg.hidden_dim) * cfg.output_dim + cfg.output_dim
    );
    if (auto* layer = model.getLayerByName("my_simple/out")) {
        layer->inputs = {"my_simple/activated"};
        layer->output = "x";
        layer->in_features = cfg.hidden_dim;
        layer->out_features = cfg.output_dim;
        layer->use_bias = true;
    }
}
```

Les points importants sont factuels :

- `push(name, type, params_count)` attend un nom de type reconnu par
  `LayerRegistry` dans `src/LayerTypes.hpp` ;
- les entrées et sorties sont des noms de tenseurs ;
- `__input__` est l'entrée conventionnelle et `x` la sortie principale ;
- pour `Linear`, le nombre de paramètres avec biais vaut
  `in_features * out_features + out_features`.

**Vérification intermédiaire :** pour `input_dim=64`, `hidden_dim=128` et
`output_dim=16`, les formes successives sont `[64]`, `[128]`, `[128]`, `[16]`.
Les deux blocs de poids contiennent respectivement `64*128+128` et
`128*16+16` paramètres. Les calculs de `Linear` et `GELU` sont exécutés par les
runtimes ; la classe décrit leur composition.

## Étape 3 — Ajouter les sources à CMake

Ajouter les `.cpp` au dépôt ne suffit pas. Incluez
`src/Models/MyDomain/MySimpleModel.cpp` dans la liste de sources qui construit
`mimir_core` dans `CMakeLists.txt`, à côté des autres modèles.

Une erreur de lien sur le constructeur ou `buildFromConfig` indique
généralement que le `.cpp` n'est pas compilé dans la cible.

## Étape 4 — Convertir la configuration JSON

`ModelArchitectures.cpp` possède déjà le helper interne `jget`. Ne redéfinissez
pas un second template du même nom dans ce fichier. Ajoutez seulement les deux
fonctions propres au modèle :

```cpp
static MySimpleModel::Config mySimpleCfgFromJson(const json& cfg) {
    MySimpleModel::Config out;
    out.input_dim = jget<int>(cfg, "input_dim", out.input_dim);
    out.hidden_dim = jget<int>(cfg, "hidden_dim", out.hidden_dim);
    out.output_dim = jget<int>(cfg, "output_dim", out.output_dim);
    return out;
}

static json mySimpleDefaultConfigJson() {
    MySimpleModel::Config defaults;
    return json{
        {"input_dim", defaults.input_dim},
        {"hidden_dim", defaults.hidden_dim},
        {"output_dim", defaults.output_dim},
    };
}
```

Ajoutez également l'include de `MySimpleModel.hpp`.

## Étape 5 — Enregistrer l'architecture

Dans `Registry::ensureBuiltinsRegistered()`, ajoutez :

```cpp
entries_.emplace(
    "my_simple_model",
    Entry{
        "my_simple_model",
        "Minimal two-layer MLP",
        mySimpleDefaultConfigJson(),
        [](const json& cfg) -> std::shared_ptr<Model> {
            auto model = std::make_shared<MySimpleModel>();
            model->buildFromConfig(mySimpleCfgFromJson(cfg));
            return model;
        },
    }
);
```

N'affectez pas `modelConfig` une seconde fois dans la lambda. Après la factory,
`Registry::create` attache la configuration JSON fusionnée au modèle et
propage aussi `dtype`.

## Étape 6 — Compiler et inspecter

```bash
cmake --build build -j2

./bin/mimir --lua scripts/tools/inspect_architectures.lua -- \
  --list my_simple_model --params --layers --stats
```

L'inspection doit montrer trois layers et deux blocs de paramètres `Linear`.

## Étape 7 — Ajouter les tests

Les tests du registre sont répartis entre
`Tests/test_registry_cfg_from_config.cpp`,
`Tests/test_registry_create_from_config.cpp` et
`Tests/test_registry_aliases.cpp`. Ils sont déclarés par
`Tests/CMakeLists.txt`. Vérifiez au minimum :

1. la présence de `my_simple_model` dans `Registry::available()` ;
2. les valeurs de la configuration par défaut ;
3. la création avec overrides ;
4. le nombre et le câblage des layers ;
5. un cycle allocation, initialisation et passe avant ;
6. une sauvegarde/relecture si le modèle doit être sérialisable.

```bash
ctest --test-dir build --output-on-failure -R RegistryTest
```

## Exercices après intégration

- **Intermédiaire :** changez les dimensions via la configuration autorisée,
  sans ajouter ni recâbler de couches dans le script Lua. Comparez le nombre
  de paramètres attendu et celui du modèle.
- **Avancé :** entraînez sur une petite fixture synthétique, sauvegardez puis
  rechargez un checkpoint avec la même topologie. Comparez une sortie connue.
- **Expert :** ajoutez un cas de dimension invalide, une vérification des
  gradients et un test de sauvegarde/relecture. Modifiez la topologie dans la
  classe C++, puis vérifiez qu'une reprise incompatible est rejetée.

Ces exercices supposent que vous avez effectivement ajouté et compilé
`MySimpleModel` ; les fichiers pédagogiques ne sont pas installés par cette page.

## Erreurs fréquentes

> **Attention**
> `Model.create()` construit déjà le graphe via la factory du registre.
> `Model.build()` est conservé comme no-op de compatibilité dans le chemin
> moderne ; ne construisez pas le graphe une seconde fois depuis Lua.

- Graphe défini dans un script Lua plutôt que dans la classe dérivée.
- MPK utilisé comme garantie de compatibilité ou de production : son chargement
  reste expérimental et doit être validé sur la configuration concernée.
- Type de layer absent de `LayerRegistry`.
- Mauvais `params_count` pour un layer entraînable.
- Sortie d'un layer différente de l'entrée attendue par le suivant.
- Source `.cpp` absente de la cible CMake.
- Config par défaut et parser JSON utilisant des noms différents.

## Étapes suivantes

- [Coder un script Lua](03-Tuto-Coder-Script.md)
- [Ajouter une opération](05-Tuto-Ajouter-Op.md)
- [Guide du registre](../07-Devs/03-Config-And-Registry.md)
