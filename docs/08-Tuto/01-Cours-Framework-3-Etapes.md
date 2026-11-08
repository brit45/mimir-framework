# Apprendre le framework en trois étapes

Ce parcours mène du premier modèle exécuté à la contribution au moteur.
Choisissez votre point de départ selon les gestes que vous savez déjà réaliser,
pas selon votre âge ou votre formation. Aucun dataset externe n'est nécessaire
pour les exercices de base de cette page.

## Contrat à retenir

Une architecture DL s'écrit en C/C++ **dans une classe héritant de `Model`**.
Les scripts Lua et les fichiers JSON configurent et pilotent ces architectures.
L'autre voie de description est MPK, encore expérimental et pas encore
entièrement exploitable en production. Le
[contrat des architectures](../07-Devs/08-Architecture-Contract.md) précise cette
séparation et le rôle des API internes.

## Avant de commencer

Placez-vous à la racine d'un checkout compilé. Les exercices CTest demandent un
build configuré avec les tests. Pour l'installation, suivez
[Installation et configuration](10-Tuto-Installation-Configuration.md).

```bash
./bin/mimir --help
```

Vous devez pouvoir afficher l'aide avant de tenter un entraînement. Un backend
GPU et une interface graphique ne sont pas nécessaires pour les premiers pas.

## Étape 1 — Débutant : lancer et reconnaître les composants

L'objectif est d'instancier un modèle déjà présent dans le registre et de
comprendre la différence entre son architecture et sa configuration de run.

### 1. Examiner le registre

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -a
```

Retrouvez `basic_mlp`. Son nom correspond à une factory C++ qui crée
`BasicMLPModel`, une classe dérivée de `Model`. Les autres entrées peuvent être
partielles ou expérimentales : l'inventaire n'est pas une garantie de maturité.

### 2. Configurer une instance sans dataset

```bash
./bin/mimir --lua scripts/templates/template_pipeline_args.lua -- \
  --from-registry --arch basic_mlp --no-train \
  --override input_dim=4 --override hidden_dim=8 --override output_dim=2
```

Attendez le message `build() OK`. Avec ces overrides et les deux couches
cachées par défaut de `basic_mlp`, le modèle compte six couches et
130 paramètres. Le template a sélectionné une architecture native, appliqué
une configuration, alloué et initialisé les poids. Avec `--no-train`, cette commande ne prouve ni un
apprentissage ni une passe avant.

### 3. Expliquer ce que vous avez changé

Réduisez `hidden_dim`, puis relancez la commande. Le nombre de paramètres doit
changer. Vous avez modifié un champ autorisé par la classe C++, sans écrire
une nouvelle topologie dans le script.

**Étape validée si :** vous distinguez nom d'architecture, paramètres de
configuration et poids ; vous savez retrouver l'erreur initiale dans un log.

Poursuivez avec [Scripting Lua](03-Tuto-Coder-Script.md) et
[Données et tokenizer](11-Tuto-Donnees-Tokenizer.md).

## Étape 2 — Autonome : sauvegarder, évaluer et reproduire

L'objectif est de préserver un état numérique et de vérifier ce qu'un
entraînement réel demande en plus de la simple construction du modèle.

### 1. Sauvegarder un petit modèle natif

```bash
./bin/mimir --lua scripts/templates/template_pipeline_args.lua -- \
  --from-registry --arch basic_mlp --no-train \
  --override input_dim=4 --override hidden_dim=8 --override output_dim=2 \
  --seed 42 --save /tmp/mimir_course_basic_mlp.safetensors

./bin/mimir --lua scripts/tools/analyze_model.lua -- \
  --in /tmp/mimir_course_basic_mlp.safetensors
```

L'analyse doit lire les métadonnées et les tenseurs du checkpoint : le type
`basic_mlp`, les six couches et les 130 paramètres doivent correspondre au
modèle construit. Les poids sont initialisés ; les sauvegarder ne les entraîne pas. Un MPK, à la différence
de ce fichier, ne conserve pas les poids appris ni l'état de l'optimiseur.

### 2. Choisir un objectif et des données compatibles

Lisez l'aide du script adapté au modèle, puis suivez
[Entraînement, validation, reprise et inférence](12-Tuto-Entrainement-Validation-Reprise-Inference.md).
Notez pour un run court : données utilisées, seed, commande, loss, métrique de
validation et chemin du checkpoint. Utilisez une fixture séparée des données
réelles avant un entraînement long.

### 3. Vérifier une reprise

Rechargez un checkpoint avec sa configuration structurelle et son tokenizer.
Contrôlez la progression de l'optimiseur et une prédiction connue. Une baisse
de loss d'entraînement ne prouve pas la qualité sur des données tenues à l'écart.

**Étape validée si :** vous distinguez état initial et état entraîné, vous savez
évaluer puis reprendre le même modèle sans reconstruire une autre topologie.

Poursuivez avec [les familles de modèles](13-Tuto-Familles-De-Modeles.md) et
[la lecture des métriques dans Viz/Htop](14-Tuto-Viz-Metriques-Diagnostic.md).

## Étape 3 — Contributeur : étendre et vérifier

Cette étape comporte deux degrés : ajouter un modèle natif, puis modifier les
calculs du moteur. Il faut savoir lire du C++ pour le premier et vérifier des
gradients pour le second.

### Avancé : une architecture dérivée de `Model`

Suivez [Ajouter un modèle](02-Tuto-Ajouter-Modele.md). Le résultat attendu est
une classe `MonModele : public Model`, sa configuration, une factory du registre
et un test ciblé. Le script Lua utilise ensuite `Model.create`, sans pousser
ni recâbler les couches lui-même.

Les mécanismes existants se vérifient avec :

```bash
ctest --test-dir build --output-on-failure \
  -R '^RegistryTest\.(CfgFromConfig|CreateFromConfig|Aliases)$'
```

Ces tests vérifient le registre livré ; ils ne remplacent pas un test de la
nouvelle classe que vous ajoutez.

### Expert : une opération ou un runtime

Lisez [Ajouter une opération](05-Tuto-Ajouter-Op.md) et
[Modifier un runtime](04-Tuto-Modifier-Ou-Ajouter-Runtime.md). Commencez par un
contrat de formes et de domaines, écrivez forward/backward dans les runtimes,
puis comparez valeurs, gradients et paramètres après un pas optimiseur.

Un backend annoncé comme natif doit aussi être celui effectivement exécuté.
Séparez les mesures CPU, transferts, forward, backward et optimiseur. Le
[tutoriel mémoire et performance](15-Tuto-Memoire-Performances-Runtimes.md)
propose la démarche progressive ; [AMP et DDP](16-Tuto-Compilation-Portabilite-AMP-DDP.md)
demandent leurs propres prérequis de build et d'exécution.

**Étape validée si :** vous produisez un résultat vérifiable, ses limites et une
preuve de non-régression. Une compilation seule ne prouve pas une exécution GPU.

## Conseils de progression

- Conservez commande, configuration et observations dans un journal de run.
- Changez un seul paramètre avant de comparer deux expériences.
- Si un problème apparaît, réduisez sa fixture plutôt que d'ajouter des options.
- Explorez [MPK](09-Tuto-Creer-MPK.md) comme un parcours expérimental distinct,
  après avoir compris la construction native et les checkpoints.

## Étapes suivantes

- [Catalogue des tutoriels et niveaux](00-INDEX.md)
- [Parcours complet du framework](06-Parcours-Complet-Framework.md)
- [Scripting, exports, tests et contribution](17-Tuto-Scripting-Exports-Tests-Contribution.md)
