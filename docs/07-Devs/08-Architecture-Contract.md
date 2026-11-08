# Contrat de définition des architectures

Cette règle s’applique à tous les tutoriels, scripts et exemples du framework.
Une architecture de deep learning définit les couches, leurs connexions et leur
configuration structurelle. Elle est écrite en C/C++ dans une classe C++ qui
hérite de `Model`. L’autre voie est le format MPK, encore expérimental et pas
entièrement exploitable en production.

## Deux voies de définition

| Voie | Définition du graphe | Usage |
| --- | --- | --- |
| Architecture native | Classe `MyModel : public Model`, sous `src/Models/` | Voie de référence, compilée et enregistrée dans le registre. |
| MPK | Description du graphe et de ses métadonnées dans un package MPK | Expérimentation et échanges de descriptions ; prise en charge encore partielle. |

Lua, JavaScript, Rust, C# et JSON pilotent les modèles exposés par le moteur.
Ils sélectionnent une architecture, fournissent sa configuration, préparent les
données et déclenchent entraînement, validation, inférence et sauvegarde.
Ils ne constituent pas des langages de définition du graphe DL.

Une option telle que `num_layers`, `conv_mode` ou `attention` peut modifier le
graphe construit : son interprétation et les couches résultantes restent
définies dans la classe C++. Le fichier JSON fournit les valeurs, pas un
nouveau programme de construction de couches.

## Classe native et registre

Le modèle de référence est `BasicMLPModel`, déclaré dans
[`BasicMLPModel.hpp`](../../src/Models/MLP/BasicMLPModel.hpp). Son organisation est :

```text
class MyModel : public Model
  → Config typée
  → buildFromConfig(Config)
  → couches et connexions définies en C++
  → enregistrement dans ModelArchitectures
  → sélection et exécution depuis Lua ou JSON
```

Une méthode statique `buildInto(Model&, const Config&)` peut partager la
construction avec d’autres composants natifs. Elle appartient à la classe du
modèle ; elle ne remplace pas l’héritage de `Model` par une topologie écrite dans
un script. Les calculs des opérations restent dans les runtimes et passent par
le dispatcher et le routeur.

Voici une classe minimale illustrant deux primitives sans paramètres. Elle
définit un graphe `[2,3] → SquaredReLU → ReduceMean(axis=-1) → [2,1]` :

```cpp
#include "Model.hpp"

class PrimerReductionModel final : public Model {
public:
    PrimerReductionModel() {
        setModelName("primer_reduction");
        setHasEncoder(false);

        push("primer/squared", "SquaredReLU", 0);
        auto* squared = getLayerByName("primer/squared");
        squared->inputs = {"__input__"};
        squared->output = "activated";

        push("primer/mean", "ReduceMean", 0);
        auto* mean = getLayerByName("primer/mean");
        mean->inputs = {"activated"};
        mean->output = "x";
        mean->shape = {2, 3};
        mean->axis = -1;
    }
};
```

Cette définition C++ ne comporte pas encore l’entrée de registre ni sa cible de
build. Le [tutoriel d’ajout d’un modèle](../08-Tuto/02-Tuto-Ajouter-Modele.md)
montre ces étapes. Après intégration, le script sélectionne le nom enregistré
avec `Mimir.Model.create(...)`, puis alloue et exécute le modèle.

## API bas niveau et checkpoints

Les bindings `create_empty`, `push_layer`, `clear_layers` et `set_layer_io`
existent dans le moteur pour les tests, la compatibilité et les mécanismes
d’import MPK. Leur présence ne définit pas une troisième voie de conception
d’architectures dans les tutoriels utilisateur.

De même, reconstruire un graphe depuis les métadonnées d’un checkpoint restaure
une architecture déjà définie ; ce n’est pas une nouvelle méthode d’écriture
du modèle. Une reprise doit préserver sa topologie, ses poids et, lorsque
nécessaire, son tokenizer et l’état de l’optimiseur.

## Statut de MPK

MPK est l’alternative expérimentale à la définition native. La validité du
package, son chargement et son apparition dans le registre ne démontrent pas
que tout son graphe s’exécute correctement en entraînement et en inférence.
Le format ne remplace pas un checkpoint de poids.

Le [tutoriel MPK](../08-Tuto/09-Tuto-Creer-MPK.md) explique comment explorer son
format et inspecter ses limites. Il ne constitue pas une procédure de
déploiement validée pour la production.

## Vérifier une contribution

- La topologie native appartient à une classe dérivée de `Model`.
- Lua et JSON configurent et exécutent cette classe via le registre.
- Chaque nouvelle opération possède un contrat de formes et de gradients,
  implémenté dans les runtimes concernés.
- Les tests couvrent construction, forward, backward et sauvegarde/rechargement.
- Les exemples MPK portent leur statut expérimental et leurs limites.

## Étapes suivantes

- [Parcours de tutoriels par niveau](../08-Tuto/00-INDEX.md)
- [Construire un modèle natif](02-Building-Models-And-Layers.md)
- [Ajouter une opération aux runtimes](../08-Tuto/05-Tuto-Ajouter-Op.md)
