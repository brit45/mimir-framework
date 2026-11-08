# Construire une architecture dérivée de `Model`

Ce guide explique où déclarer un graphe de deep learning, comment relier ses
couches et comment le rendre accessible aux scripts. Il s'adresse aux
contributeurs qui savent lire du C++ et raisonner sur les dimensions.

## Sur cette page

- [Contrat de définition des architectures](#contrat-de-définition-des-architectures)
- [Débutant : reconnaître les responsabilités](#débutant--reconnaître-les-responsabilités)
- [Intermédiaire : une classe et un graphe valides](#intermédiaire--une-classe-et-un-graphe-valides)
- [Avancé : configuration et registre](#avancé--configuration-et-registre)
- [Piloter une architecture existante depuis Lua](#piloter-une-architecture-existante-depuis-lua)
- [MPK et API internes de construction](#mpk-et-api-internes-de-construction)
- [Expert : intégrer une opération et son runtime](#expert--intégrer-une-opération-et-son-runtime)
- [Vérifier et diagnostiquer](#vérifier-et-diagnostiquer)
- [Étapes suivantes](#étapes-suivantes)

## Contrat de définition des architectures

Le [contrat des architectures](08-Architecture-Contract.md) impose que les
architectures, ou graphes DL, soient écrites dans le code C/C++ du framework,
**dans une classe C++ héritant de `Model`**. Les couches, leurs
connexions, les branches et les paramètres structurels appartiennent à cette
classe. Le constructeur et les méthodes de construction peuvent appeler un
helper statique `buildInto(Model&, ...)` défini dans la même classe.

L'autre moyen de décrire une architecture est **MPK, encore expérimental et
pas encore entièrement exploitable en production**. Un package lisible ou
compilable ne prouve pas que son graphe et ses opérations soient exécutables.
Pour un modèle à intégrer au framework, le chemin de référence reste une classe
C++ dérivée de `Model` et enregistrée dans le registre.

Lua, JSON et les autres bridges configurent et exécutent ces architectures.
Ils ne constituent pas une troisième voie de définition d'un graphe DL.

## Débutant : reconnaître les responsabilités

| Élément | Responsabilité |
| --- | --- |
| Classe `MonModele : public Model` | Topologie, noms des tenseurs, couches et dimensions |
| `Config` de cette classe | Paramètres structurels autorisés et valeurs par défaut |
| `ModelArchitectures` | Nom public, conversion JSON et création de la classe |
| Script Lua / fichier JSON | Choix d'une architecture, données, entraînement, inférence et sauvegarde |
| Runtime | Calcul des opérations forward/backward, sélection du backend et contrôles numériques |
| MPK expérimental | Transport ou exploration d'une configuration et d'une description de graphe |

Commencez par `src/Models/MLP/BasicMLPModel.hpp` et `.cpp` : la classe hérite
explicitement de `Model`; `buildFromConfig` appelle son helper `buildInto`.
Repérez ensuite la factory `basic_mlp` dans
`src/Models/Registry/ModelArchitectures.cpp`.

**Exercice :** retrouvez le même chemin pour `VAEConvModel`, puis expliquez où
sont décidés les layers et où sont lus les arguments CLI.

## Intermédiaire : une classe et un graphe valides

Voici un exemple C++ complet de déclaration, à placer dans un header de votre
modèle. Il utilise les primitives Primer et ne possède aucun poids appris.
Le graphe est fixé dans la classe ; l'entrée est un vecteur de six valeurs
interprété comme une matrice `[2,3]`.

```cpp
#pragma once
#include "Model.hpp"

class PrimerReductionModel final : public Model {
public:
    PrimerReductionModel() {
        setModelName("PrimerReductionModel");
        setHasEncoder(false);
        buildInto(*this);
    }

    static void buildInto(Model& model) {
        model.getMutableLayers().clear();
        model.modelConfig["type"] = "primer_reduction_tutorial";

        model.push("primer/squared", "SquaredReLU", 0);
        auto* squared = model.getLayerByName("primer/squared");
        squared->inputs = {"__input__"};
        squared->output = "primer/activated";

        model.push("primer/mean", "ReduceMean", 0);
        auto* mean = model.getLayerByName("primer/mean");
        mean->inputs = {"primer/activated"};
        mean->output = "x";
        mean->shape = {2, 3};
        mean->axis = -1;
    }
};
```

`Model::push(name, type, params_count)` crée une couche du registre des types.
`Layer.inputs` et `Layer.output` relient les tenseurs nommés. Ici, la sortie de
`SquaredReLU` alimente `ReduceMean`, qui conserve la dimension réduite à 1 :
la sortie contient deux valeurs de forme `[2,1]`.

Pour l'entrée `{-1,2,3,4,-5,6}`, les sorties attendues sont `13/3` et `52/3`.
Ce calcul doit passer par les runtimes ; il n'est pas réimplémenté dans la classe.
Construisez le graphe avant l'allocation. Ne reconstruisez pas une instance
qui possède déjà des poids et un état optimiseur actifs.

Le nom `primer_reduction_tutorial` ci-dessus n'est pas une architecture livrée
dans le registre. Pour l'exposer à Lua, effectuez l'intégration décrite dans le
[tutoriel d'ajout d'un modèle](../08-Tuto/02-Tuto-Ajouter-Modele.md).

**Exercice :** remplacez, dans la classe C++, la réduction moyenne par une somme.
Calculez les résultats attendus avant d'exécuter un test.

## Avancé : configuration et registre

Un modèle entraînable suit généralement ce chemin :

1. Définir `class MonModele : public Model` et sa structure `Config` dans un header.
2. Valider les dimensions, puis construire le graphe dans la classe.
3. Ajouter son `.cpp` aux sources de `mimir_core` dans `CMakeLists.txt`.
4. Ajouter un parser JSON et une configuration par défaut au registre.
5. Faire créer la classe dérivée par la factory du registre.
6. Tester création, allocation, forward/backward et checkpoint.

Le helper `buildInto` n'est pas une architecture autonome : il centralise la
construction au sein de la classe dérivée, comme dans `BasicMLPModel` et
`VAEConvModel`. Le registre instancie cette classe avant de construire son graphe.

Pour un layer `Linear` avec biais, le nombre de paramètres vaut
`in_features * out_features + out_features`. Les layers sans poids, par exemple
`SquaredReLU`, `Add` et les réductions Primer, utilisent `params_count=0`.
Chaque branche doit avoir des noms d'entrées et sorties non ambigus.

**Exercice :** ajoutez une dimension dans `Config`, son parser JSON et sa valeur
par défaut. Vérifiez que la même valeur pilote le nombre de paramètres, les
métadonnées des couches et le chargement d'un checkpoint compatible.

## Piloter une architecture existante depuis Lua

Le script demande au registre une architecture déjà écrite en C++ :

```lua
local cfg = Mimir.Architectures.default_config("basic_mlp")
cfg.input_dim = 2
cfg.hidden_dim = 4
cfg.output_dim = 1
cfg.hidden_layers = 1
cfg.dropout = 0
assert(Mimir.Model.create("basic_mlp", cfg))
assert(Mimir.Model.set_hardware(false))
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier", 42))
local prediction = assert(Mimir.Model.forward({0.25, -0.5}, false))
assert(#prediction == 1)
```

Les options `Config` restent celles autorisées par la classe. Le script ne pousse
aucune couche et ne recâble aucun tenseur. `Mimir.Model.create` construit déjà
le graphe via sa factory ; `Model.build()` est un no-op de compatibilité dans
ce chemin et ne doit pas déclencher une seconde construction.

## MPK et API internes de construction

Les bindings `create_empty`, `push_layer` et `set_layer_io` existent dans le
moteur pour ses tests et certains mécanismes internes, notamment le chargement
de graphes MPK. Leur existence ne fait pas de la construction directe d'un
réseau en Lua une méthode d'architecture recommandée ou autorisée par ce contrat.
Les tutoriels d'architecture utilisent une classe C++ dérivée de `Model`.

Les outils Lua qui génèrent un MPK produisent une spécification de package ;
l'architecture alternative reste le MPK expérimental, pas un script de calcul
Lua autonome. Consultez ses [limites et formats](../02-User-Guide/15-MPK.md)
avant de tenter une reconstruction dynamique.

## Expert : intégrer une opération et son runtime

Une architecture compose des opérations disponibles. Si une primitive manque :

1. Enregistrer son type canonique et ses alias dans `src/LayerTypes.hpp`, sans
   modifier les identifiants existants.
2. Définir sa géométrie et ses contrôles sous `src/runtimes/`.
3. Écrire ses calculs forward/backward dans les kernels du backend, pas dans
   `Layers.hpp` ou dans la description de l'architecture.
4. Brancher le dispatcher et les capacités déclarées ; `RuntimeRouter` choisit
   le backend effectif ou un repli explicite.
5. Préserver métadonnées, planification et sérialisation.
6. Tester valeurs, gradients, domaines et refus sans publication partielle.

Pour un exemple réel, lisez `src/runtimes/cpu/PrimerKernels.hpp`,
`src/runtimes/cpu/RuntimeLayerDispatch.hpp` et les
[contrats des primitives Primer](../04-Architecture-Internals/14-Layers-And-Ops.md#primitives-issues-de-létude-primer).
Un test GPU doit vérifier le backend réellement choisi ; une capacité déclarée
ne prouve pas une exécution native.

## Vérifier et diagnostiquer

Les tests suivants existent déjà ; ils vérifient les mécanismes du framework,
pas la classe pédagogique non intégrée de cette page :

```bash
ctest --test-dir build --output-on-failure \
  -R '^(RegistryTest\.(CfgFromConfig|CreateFromConfig|Aliases)|ModelTest\.PrimerOps)$'
```

| Symptôme | Vérification |
| --- | --- |
| Type invalide dans `push` | Nom canonique et mapping dans `LayerTypes.hpp` |
| Couche non exécutée | Noms `inputs`/`output` et sortie principale `x` |
| Mauvais nombre de paramètres | Formule et dimensions effectives du layer |
| Erreur de lien de la nouvelle classe | Présence du `.cpp` dans les sources CMake |
| Reprise incohérente | Topologie, configuration et tokenizer du checkpoint |
| Graphe MPK exporté mais inexécutable | Couverture des types, champs exportés et limites expérimentales |

## Étapes suivantes

- [Ajouter un modèle dérivé de Model](../08-Tuto/02-Tuto-Ajouter-Modele.md)
- [Config et registre](03-Config-And-Registry.md)
- [Ajouter une opération](../08-Tuto/05-Tuto-Ajouter-Op.md)
- [MPK expérimental](../08-Tuto/09-Tuto-Creer-MPK.md)
