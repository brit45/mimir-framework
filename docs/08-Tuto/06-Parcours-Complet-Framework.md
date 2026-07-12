# Tuto - Parcours complet du framework

## Pour qui

- Debutants qui veulent une vue d'ensemble concrete.
- Utilisateurs intermediaires qui veulent passer d'un test simple a un workflow complet.
- Contributeurs qui veulent comprendre ou se situe chaque piece du framework.

## Objectif

Apprendre le framework de bout en bout avec un chemin guide qui couvre la compilation, les donnees, les modeles, l'entrainement, l'inference, les checkpoints, le scripting, le runtime et l'extension du code.

## Avant de commencer

1. Compiler Mimir.
2. Verifier le smoketest.
3. Ouvrir les pages de reference si besoin:
- [docs/00-Framework-Philosophy.md](../00-Framework-Philosophy.md)
- [docs/01-Getting-Started/00-GET-STARTED.md](../01-Getting-Started/00-GET-STARTED.md)
- [docs/02-User-Guide/01-Core-Concepts.md](../02-User-Guide/01-Core-Concepts.md)

## Resultat attendu

Tu sais decrire le role de chaque sous-systeme du framework et tu sais vers quelle page aller selon le probleme a resoudre.

## Vue d'ensemble

![Vue d'ensemble du framework](../graphs/00_framework_overview.svg)

![Architecture en couches](../graphs/01_layered_architecture.svg)

## Parcours recommande

### 1. Demarrage et philosophie

Commence par poser le cadre technique.

1. Lire [docs/00-Framework-Philosophy.md](../00-Framework-Philosophy.md).
2. Lire [docs/01-Getting-Started/00-GET-STARTED.md](../01-Getting-Started/00-GET-STARTED.md).
3. Suivre [docs/01-Getting-Started/01-Quick-Start.md](../01-Getting-Started/01-Quick-Start.md).
4. Lancer [docs/01-Getting-Started/05-Smoketest.md](../01-Getting-Started/05-Smoketest.md).

### 2. Donnees et modalites

Apprends comment Mimir voit un dataset et comment il relie texte, image, audio et video.

1. Lire [docs/02-User-Guide/03-Data.md](../02-User-Guide/03-Data.md).
2. Lire [docs/03-API-Reference/13-Dataset.md](../03-API-Reference/13-Dataset.md).
3. Lire [docs/02-User-Guide/09-Memory.md](../02-User-Guide/09-Memory.md) pour le comportement memo et evictions.
4. Tester un dataset minimal avec images et texte.

### 3. Cycle de vie d'un modele

Comprends le chemin create/build/allocate/init/forward/backward.

1. Lire [docs/02-User-Guide/02-Model-Lifecycle.md](../02-User-Guide/02-Model-Lifecycle.md).
2. Lire [docs/03-API-Reference/01-Layers.md](../03-API-Reference/01-Layers.md).
3. Lire [docs/03-API-Reference/11-Architectures.md](../03-API-Reference/11-Architectures.md).
4. Lire [docs/03-API-Reference/20-Lua-API-Cpp-Mapping.md](../03-API-Reference/20-Lua-API-Cpp-Mapping.md).

### 4. Entrainement et inference

Passe du build au run reproductible.

1. Lire [docs/02-User-Guide/04-Training.md](../02-User-Guide/04-Training.md).
2. Lire [docs/02-User-Guide/05-Inference.md](../02-User-Guide/05-Inference.md).
3. Lire [docs/02-User-Guide/08-Checkpoints.md](../02-User-Guide/08-Checkpoints.md).
4. Faire un mini run train puis reload checkpoint.

### 5. Scripting et automatisation

Utilise Lua pour piloter le framework sans toucher au C++.

1. Lire [docs/02-User-Guide/06-Lua-Scripting.md](../02-User-Guide/06-Lua-Scripting.md).
2. Lire [docs/02-User-Guide/08-Config-Driven-Scripting.md](../02-User-Guide/08-Config-Driven-Scripting.md).
3. Tester un template dans [docs/08-Tuto/03-Tuto-Coder-Script.md](03-Tuto-Coder-Script.md).

### 6. Runtime, memoire et performance

Apprends ou le framework peut echouer ou ralentir.

1. Lire [docs/04-Architecture-Internals/01-Engine-Overview.md](../04-Architecture-Internals/01-Engine-Overview.md).
2. Lire [docs/04-Architecture-Internals/02-Memory.md](../04-Architecture-Internals/02-Memory.md).
3. Lire [docs/04-Architecture-Internals/03-Hardware-Backends.md](../04-Architecture-Internals/03-Hardware-Backends.md).
4. Lire [docs/05-Advanced/01-Performance.md](../05-Advanced/01-Performance.md) et [docs/05-Advanced/02-Debugging.md](../05-Advanced/02-Debugging.md).

### 7. Extension du framework

Quand il manque une piece, tu dois savoir la brancher proprement.

1. Ajouter un modele avec [docs/08-Tuto/02-Tuto-Ajouter-Modele.md](02-Tuto-Ajouter-Modele.md).
2. Modifier un runtime avec [docs/08-Tuto/04-Tuto-Modifier-Ou-Ajouter-Runtime.md](04-Tuto-Modifier-Ou-Ajouter-Runtime.md).
3. Ajouter une OP avec [docs/08-Tuto/05-Tuto-Ajouter-Op.md](05-Tuto-Ajouter-Op.md).
4. Lire [docs/06-Contributing/03-Extending-Models-Runtimes-And-Features.md](../06-Contributing/03-Extending-Models-Runtimes-And-Features.md).

## Atelier pratique

Fais cette sequence simple pour couvrir tous les blocs au moins une fois:

1. Compile le projet.
2. Lance le smoketest.
3. Ouvre un script Lua minimal.
4. Charge un dataset.
5. Lance un mini entrainement.
6. Sauvegarde un checkpoint.
7. Recharge le checkpoint et fais une inference.
8. Lis la section runtime si la memoire ou la performance ne correspond pas aux attentes.

## Mini repere visuel

![Carte des composants](../graphs/02_component_map.svg)

## Quand tu bloques

- Si le probleme est conceptuel, reviens a la philosophie et au cycle de vie.
- Si le probleme est runtime, regarde la memoire, les backends et le debugging.
- Si le probleme est d'extension, passe par la couche contrib et les tutos pratiques.

## Suite logique

- Parcours numerote: [docs/08-Tuto/01-Cours-Framework-3-Etapes.md](01-Cours-Framework-3-Etapes.md)
- Tuto modele: [docs/08-Tuto/02-Tuto-Ajouter-Modele.md](02-Tuto-Ajouter-Modele.md)
- Tuto scripting: [docs/08-Tuto/03-Tuto-Coder-Script.md](03-Tuto-Coder-Script.md)
- Tuto runtime: [docs/08-Tuto/04-Tuto-Modifier-Ou-Ajouter-Runtime.md](04-Tuto-Modifier-Ou-Ajouter-Runtime.md)
- Tuto op: [docs/08-Tuto/05-Tuto-Ajouter-Op.md](05-Tuto-Ajouter-Op.md)