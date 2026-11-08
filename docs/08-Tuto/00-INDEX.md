# Tutoriels et parcours de maîtrise

Les tutoriels couvrent l’utilisation du framework et l’extension de son moteur.
Choisissez un niveau et un objectif, puis suivez les étapes, leurs vérifications
et les exercices. Chaque atelier précise ses besoins en données, checkpoints,
compilation et matériel.

## Règle de définition des architectures

Un graphe DL est écrit en C/C++ dans une classe dérivée de `Model`. Lua et JSON
sélectionnent, configurent et exécutent cette architecture ; ils ne définissent
pas sa topologie. MPK est l’autre voie de description, encore expérimentale et
pas entièrement exploitable en production. Le
[contrat des architectures](../07-Devs/08-Architecture-Contract.md) s’applique à
tous les parcours.

## Choisir son niveau

| Niveau | Point de départ | Résultat visé |
| --- | --- | --- |
| 1 — Découverte | Vous savez ouvrir un terminal. | Lancer et inspecter un modèle enregistré sans dataset. |
| 2 — Pratique | Vous savez modifier une option. | Préparer des données, entraîner, valider, reprendre et inférer. |
| 3 — Autonomie | Vous savez gérer un checkpoint et lire une loss. | Reproduire et comparer les runs, leurs ressources et leurs métriques. |
| 4 — Développement de modèles | Vous connaissez C++ et les formes de tenseurs. | Ajouter une classe dérivée de `Model` et vérifier ses gradients. |
| 5 — Expertise moteur | Vous savez développer un backward ou un kernel. | Étendre les runtimes et prouver leur conformité numérique. |

Votre niveau peut varier selon le sujet. Les
[parcours détaillés](../01-Getting-Started/06-Learning-Paths.md) proposent des
critères de passage et des exercices.

## Parcours guidés

### Niveau 1 — Du terminal au checkpoint

1. [Installer et configurer Mímir](10-Tuto-Installation-Configuration.md).
2. [Comprendre le framework en trois étapes](01-Cours-Framework-3-Etapes.md).
3. [Écrire un script d’orchestration Lua](03-Tuto-Coder-Script.md).
4. [Passer du registre au checkpoint sans dataset](08-Tuto-Registre-Pipeline-Checkpoint.md).

Vous avez terminé lorsque vous distinguez architecture C++, configuration,
script de run et poids sauvegardés.

### Niveau 2 — Un apprentissage observable

1. [Préparer les données et le tokenizer](11-Tuto-Donnees-Tokenizer.md).
2. [Entraîner, valider, reprendre et inférer](12-Tuto-Entrainement-Validation-Reprise-Inference.md).
3. [Observer et diagnostiquer un run](14-Tuto-Viz-Metriques-Diagnostic.md).
4. [Choisir une famille de modèles](13-Tuto-Familles-De-Modeles.md).

Vous avez terminé lorsque vous retrouvez les entrées, le tokenizer, les
métriques train/validation et le checkpoint d’un même run.

### Niveau 3 — Reproduire et comparer

1. [Rejouer le cycle d’apprentissage](12-Tuto-Entrainement-Validation-Reprise-Inference.md)
   avec un seul changement de configuration.
2. [Mesurer mémoire, performances et routage](15-Tuto-Memoire-Performances-Runtimes.md).
3. [Étudier les profils de compilation, AMP et DDP](16-Tuto-Compilation-Portabilite-AMP-DDP.md).
4. [Automatiser scripts, exports et vérifications](17-Tuto-Scripting-Exports-Tests-Contribution.md).

Vous avez terminé lorsque vous distinguez backend compilé, initialisé, annoncé
comme natif et réellement utilisé par une opération.

### Niveau 4 — Ajouter une architecture native

1. Lire le [contrat de définition](../07-Devs/08-Architecture-Contract.md).
2. [Ajouter une classe dérivée de Model](02-Tuto-Ajouter-Modele.md).
3. Étudier [couches, formes et connexions](../07-Devs/02-Building-Models-And-Layers.md).
4. [Suivre le contrat d’une opération et de son backward](05-Tuto-Ajouter-Op.md).
5. [Valider un VAEConv et son prior sans dataset](07-Tuto-VAEConv-Sans-Dataset.md).

Vous avez terminé lorsque votre classe est créée par le registre, que les
gradients atteignent ses paramètres et qu’un checkpoint recharge ses résultats.

### Niveau 5 — Étendre le moteur

1. [Modifier ou ajouter un runtime](04-Tuto-Modifier-Ou-Ajouter-Runtime.md).
2. [Implémenter et tester une opération](05-Tuto-Ajouter-Op.md).
3. [Mesurer planner, mémoire et kernels](15-Tuto-Memoire-Performances-Runtimes.md).
4. [Vérifier les configurations matérielles et distribuées](16-Tuto-Compilation-Portabilite-AMP-DDP.md).
5. [Préparer les tests et la contribution](17-Tuto-Scripting-Exports-Tests-Contribution.md).

Un résultat CPU ne valide pas un GPU. Une compilation AMP/DDP ne prouve pas que
le run les utilise. Les tutoriels séparent les preuves propres à chaque
environnement.

## Trouver un atelier par aspect

| Aspect | Atelier | Référence |
| --- | --- | --- |
| Installation, CLI, configuration JSON et `env` | [10 — Installation](10-Tuto-Installation-Configuration.md) | [CLI](../01-Getting-Started/03-CLI.md) |
| Modèle, registre et cycle de vie | [08 — Registre au checkpoint](08-Tuto-Registre-Pipeline-Checkpoint.md) | [02 — Classe native](02-Tuto-Ajouter-Modele.md) |
| Lua, arguments et scripts | [03 — Script Lua](03-Tuto-Coder-Script.md) | [17 — Automatisation](17-Tuto-Scripting-Exports-Tests-Contribution.md) |
| Texte, images, audio, vidéo, annotations | [11 — Données](11-Tuto-Donnees-Tokenizer.md) | [Données et formats](../02-User-Guide/03-Data.md) |
| BPE, vocabulaire et contexte | [11 — Tokenizer](11-Tuto-Donnees-Tokenizer.md) | [Tokenizer et encodeur](../02-User-Guide/07-Tokenizer-Encoder.md) |
| Loss, gradients, optimiseur et validation | [12 — Apprentissage](12-Tuto-Entrainement-Validation-Reprise-Inference.md) | [Autograd](../04-Architecture-Internals/13-Autograd-Gradients.md) |
| Reprise, checkpoints et inférence | [12 — Apprentissage](12-Tuto-Entrainement-Validation-Reprise-Inference.md) | [Checkpoints](../02-User-Guide/08-Checkpoints.md) |
| Causal LM, Transformer, T5 et VAE texte | [13 — Familles](13-Tuto-Familles-De-Modeles.md) | [Guide causal](../02-User-Guide/12-Transformer-GPT.md) |
| Vision, VAEConv, convolutions 2D/3D/4D | [07 — VAEConv](07-Tuto-VAEConv-Sans-Dataset.md) | [13 — Familles](13-Tuto-Familles-De-Modeles.md) |
| Diffusion, Lumen et conditionnement | [13 — Familles](13-Tuto-Familles-De-Modeles.md) | [Diffusion](../02-User-Guide/13-Diffusion.md) |
| Viz, Htop, tips, rapports et projection 3D | [14 — Observation](14-Tuto-Viz-Metriques-Diagnostic.md) | [Projection 3D](../02-User-Guide/18-Projection-3D.md) |
| RAM, allocator et scratchpads | [15 — Mémoire](15-Tuto-Memoire-Performances-Runtimes.md) | [Gestion mémoire](../02-User-Guide/09-Memory.md) |
| Planner, calibration, fusion, benchmarks | [15 — Performances](15-Tuto-Memoire-Performances-Runtimes.md) | [Planner](../04-Architecture-Internals/22-Planning.md) |
| CPU/SIMD, Vulkan, OpenCL, CUDA et ROCm | [15 — Runtimes](15-Tuto-Memoire-Performances-Runtimes.md) | [16 — Compilation](16-Tuto-Compilation-Portabilite-AMP-DDP.md) |
| AMP, dtypes et DDP/MPI | [16 — AMP et DDP](16-Tuto-Compilation-Portabilite-AMP-DDP.md) | [Variables d’environnement](../03-API-Reference/22-Environment-Variables.md) |
| Layers et parité numérique | [05 — Opération](05-Tuto-Ajouter-Op.md) | [Contrats des ops](../04-Architecture-Internals/14-Layers-And-Ops.md) |
| Nouveaux backends et dispatcher | [04 — Runtime](04-Tuto-Modifier-Ou-Ajouter-Runtime.md) | [Développement runtime](../07-Devs/04-Runtime-Development.md) |
| SafeTensors, RawFolder, DebugJSON et bridges | [17 — Scripting et exports](17-Tuto-Scripting-Exports-Tests-Contribution.md) | [Scripting](../03-API-Reference/00-Scripting-Contract.md) |
| MPK expérimental | [09 — Explorer MPK](09-Tuto-Creer-MPK.md) | [Format et limites](../02-User-Guide/15-MPK.md) |
| Tests, documentation, CI et contribution | [17 — Vérifier et contribuer](17-Tuto-Scripting-Exports-Tests-Contribution.md) | [Contribution](../06-Contributing/01-Contributing.md) |

Les fonctions partielles, dont certains chemins TPU/FPGA, les bridges optionnels
et les architectures placeholders, sont abordées avec leur statut réel dans la
[carte de maturité](../00-PROJECT-STATUS.md).

## Utiliser chaque atelier

1. Vérifier les prérequis et la disponibilité des données ou du matériel.
2. Lancer le cas minimal et conserver configuration et sorties.
3. Comparer au résultat attendu, puis réaliser les exercices.
4. Modifier un paramètre à la fois et expliquer l’effet observé.
5. Consulter la référence liée pour adapter l’atelier à son propre projet.

Les cas sans dataset valident le fonctionnement, pas la qualité d’un modèle
entraîné. Les recettes d’installation et les longues sessions d’entraînement
doivent être choisies pour votre environnement.

## Étapes suivantes

- [Installation et première configuration](10-Tuto-Installation-Configuration.md)
- [Parcours détaillés par niveau](../01-Getting-Started/06-Learning-Paths.md)
- [Index général](../00-INDEX.md)
