# Parcours par niveau de compétence

Ces parcours accompagnent la découverte, l’utilisation autonome et le
développement du framework. Le [catalogue des tutoriels](../08-Tuto/00-INDEX.md)
permet aussi de choisir un aspect précis.

Les architectures DL sont définies en C/C++ dans une classe dérivée de `Model`.
Lua et JSON pilotent leur configuration et leur exécution. MPK est l’alternative
expérimentale, pas encore entièrement exploitable en production. Consultez le
[contrat des architectures](../07-Devs/08-Architecture-Contract.md).

## Choisir un parcours

| Niveau | Vous savez déjà | Votre prochaine réalisation |
| --- | --- | --- |
| 1 — Débutant | Utiliser un terminal | Lancer et inspecter un modèle du registre sans dataset. |
| 2 — Pratiquant | Modifier une configuration | Préparer un corpus et observer un entraînement validé. |
| 3 — Autonome | Entraîner et reprendre | Reproduire, mesurer et comparer des runs. |
| 4 — Développeur modèle | Lire C++ et les formes de tenseurs | Ajouter une classe dérivée de `Model` et ses tests. |
| 5 — Développeur moteur | Écrire un backward ou un kernel | Étendre le dispatch avec une preuve numérique et matérielle. |

## Niveau 1 — Démarrer sans dataset

Suivez [installation et configuration](../08-Tuto/10-Tuto-Installation-Configuration.md),
puis le [framework en trois étapes](../08-Tuto/01-Cours-Framework-3-Etapes.md).
Depuis la racine du dépôt, avec le binaire compilé :

```bash
./run_mimir.sh --help
./run_mimir.sh --lua scripts/tools/inspect_architectures.lua -- --show-archs
```

Choisissez ensuite [du registre au checkpoint](../08-Tuto/08-Tuto-Registre-Pipeline-Checkpoint.md).
Vous devez pouvoir expliquer la différence entre architecture, configuration,
script et checkpoint. Aucun benchmark volumineux n’est nécessaire.

Exercice : inspectez `basic_mlp`, changez une dimension dans sa configuration et
retrouvez l’effet sur ses paramètres. Le builder C++ reste la source du graphe.

## Niveau 2 — Réaliser un cycle d’apprentissage

1. Préparez les données avec [l’atelier tokenizer et modalités](../08-Tuto/11-Tuto-Donnees-Tokenizer.md).
2. Effectuez [entraînement, validation, reprise et inférence](../08-Tuto/12-Tuto-Entrainement-Validation-Reprise-Inference.md).
3. Lisez les métriques avec [Viz et diagnostic](../08-Tuto/14-Tuto-Viz-Metriques-Diagnostic.md).
4. Comparez les besoins des [familles de modèles](../08-Tuto/13-Tuto-Familles-De-Modeles.md).

Critère de passage : retrouver entrées, tokenizer, courbes train et validation
et checkpoint d’un même run. Une reprise conserve paramètres et topologie, puis
avance le compteur d’optimisation.

Exercice : comparez deux petits runs avec le même seed et un seul changement de
learning rate. Distinguez le fonctionnement technique de la qualité du texte
produit sur un corpus d’exercice.

## Niveau 3 — Maîtriser les ressources et la reproductibilité

Travaillez [mémoire, performances et routage](../08-Tuto/15-Tuto-Memoire-Performances-Runtimes.md),
puis [compilation, AMP et DDP](../08-Tuto/16-Tuto-Compilation-Portabilite-AMP-DDP.md).
Terminez par [scripts, exports et vérifications](../08-Tuto/17-Tuto-Scripting-Exports-Tests-Contribution.md).

Critère de passage : reproduire un résultat avec ses fichiers, mesurer
forward/backward/optimiseur et montrer le backend choisi. Un gain de temps doit
être comparé à configuration et précision équivalentes.

Exercice : observez un petit modèle sur CPU puis, si disponible, sur un GPU.
Expliquez le coût des transferts et tout repli CPU. L’absence de GPU ne bloque
pas le parcours CPU ; indiquez la limite de votre preuve.

## Niveau 4 — Développer une architecture native

1. Lisez le [contrat d’architecture](../07-Devs/08-Architecture-Contract.md).
2. Suivez [l’ajout d’un modèle](../08-Tuto/02-Tuto-Ajouter-Modele.md).
3. Étudiez [layers et connexions C++](../07-Devs/02-Building-Models-And-Layers.md).
4. Vérifiez [les opérations et leurs gradients](../08-Tuto/05-Tuto-Ajouter-Op.md).

Votre livraison comprend une classe `MyModel : public Model`, sa configuration,
son entrée de registre, ses scripts d’orchestration et ses tests. Un script Lua
qui empile des couches n’est pas une livraison conforme à ce contrat.

Exercice : remplacez une activation dans une classe native d’exercice, puis
vérifiez formes, gradients et sauvegarde/rechargement. Explorez ensuite
[MPK](../08-Tuto/09-Tuto-Creer-MPK.md) pour comprendre son format, en gardant son
statut expérimental distinct du chemin natif.

## Niveau 5 — Contribuer au moteur

Suivez [le tutoriel runtime](../08-Tuto/04-Tuto-Modifier-Ou-Ajouter-Runtime.md) et
[l’ajout d’une opération](../08-Tuto/05-Tuto-Ajouter-Op.md), puis étudiez
calibration, budgets mémoire et fusion dans
[l’atelier performances](../08-Tuto/15-Tuto-Memoire-Performances-Runtimes.md).

Critère de passage : dispatcher et routeur branchés, entrées invalides refusées
sans publication partielle, tests de sorties, gradients et paramètres après
mise à jour. Chaque backend annoncé doit disposer de sa preuve d’exécution.

Exercice : implémentez une référence CPU, puis une variante GPU. Mesurez les
deux et justifiez la tolérance numérique. Conservez un repli explicite pour les
formes ou la précision non prises en charge.

## Où trouver la référence

| Question | Source |
| --- | --- |
| Quelles architectures existent ? | `src/Models/Registry/ModelArchitectures.cpp` |
| Qui définit le graphe ? | La classe dérivée sous `src/Models/` |
| Quelle API de script est disponible ? | `src/scriptings/Lua/luaScripting/` et `mimir-api.lua` |
| Comment sont lus les arguments ? | `scripts/modules/args.lua` et global standard `arg` |
| Quel runtime exécute une couche ? | `src/runtimes/RuntimeRouter.cpp` et les traces du run |
| Où sont les calculs CPU ? | `src/runtimes/cpu/RuntimeLayerDispatch.hpp` et ses kernels |
| Comment sont restaurés les poids ? | `src/Serialization/` et le résolveur du script |

## Étapes suivantes

- [Catalogue et couverture par aspect](../08-Tuto/00-INDEX.md)
- [Smoketest](05-Smoketest.md)
- [Index général](../00-INDEX.md)
