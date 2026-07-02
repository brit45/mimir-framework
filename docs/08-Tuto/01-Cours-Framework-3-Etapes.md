# Tuto - Cours framework en 3 etapes

## Pour qui

- Debutants (college, lycee)
- Etudiants (ingenierie, prof, autodidactes techniques)
- Niveau avance (fac, scientifique, dev senior)

## Objectif

Apprendre a utiliser Mimir de facon progressive, du premier modele jusqu'a l'optimisation et l'extension du framework.

## Avant de commencer

1. Compiler Mimir (voir [docs/01-Getting-Started/02-Installation.md](../01-Getting-Started/02-Installation.md)).
2. Verifier l'environnement avec le smoketest (voir [docs/01-Getting-Started/05-Smoketest.md](../01-Getting-Started/05-Smoketest.md)).
3. Savoir lancer une commande terminal simple.

## Résultat attendu

A la fin de ce cours, tu peux:

- creer un script de base,
- lancer entrainement et inference,
- diagnostiquer un probleme de performance ou de memoire,
- choisir la documentation adaptee a ton niveau.

## Parcours numerote 1 a 5

1. Installer et verifier l'environnement
2. Lancer un premier modele qui fonctionne
3. Faire un mini entrainement puis une inference
4. Sauvegarder, recharger et comparer les resultats
5. Mesurer, optimiser, puis valider la non-regression

Ce parcours est compatible avec les 3 niveaux ci-dessous: debutant, etudiant, avance.

---

## Etape 1 - Debutants (college, lycee)

### Mission

Comprendre ce qu'est un modele, lancer un script qui marche, et lire une sortie.

### Notions a retenir

- Un modele prend une entree et produit une sortie.
- Dans Mimir, l'entree par defaut est souvent `__input__` et la sortie principale `x`.
- On suit un cycle: creation -> allocation -> execution.

### Parcours recommande (45 a 90 min)

1. Lire [docs/01-Getting-Started/00-GET-STARTED.md](../01-Getting-Started/00-GET-STARTED.md).
2. Faire le quick start [docs/01-Getting-Started/01-Quick-Start.md](../01-Getting-Started/01-Quick-Start.md).
3. Comprendre les concepts de base [docs/02-User-Guide/01-Core-Concepts.md](../02-User-Guide/01-Core-Concepts.md).

### Exercices

1. Lancer un template modele:

```bash
./bin/mimir --lua scripts/templates/template_new_model.lua
```

2. Identifier dans les logs:
- si le modele est cree,
- si l'allocation memoire est faite,
- si un forward passe sans erreur.

3. Modifier un parametre simple (taille, batch, ou seed) et relancer.

### Validation de fin d'etape

Tu valides l'etape si:

- tu lances un script sans crash,
- tu expliques en une phrase le role de `forward`,
- tu sais ou regarder en cas d'erreur de lancement.

---

## Etape 2 - Etudiants (ingenierie, prof, etc.)

### Mission

Construire une petite experience ML complete: donnees, entrainement, checkpoint, inference.

### Notions a retenir

- Le cycle de vie complet d'un modele (`create/build/allocate/init/forward/backward`).
- Le role du dataset, des hyperparametres et des checkpoints.
- La difference entre entrainement et inference.

### Parcours recommande (1 a 2 jours)

1. Lifecycle modele: [docs/02-User-Guide/02-Model-Lifecycle.md](../02-User-Guide/02-Model-Lifecycle.md).
2. Donnees: [docs/02-User-Guide/03-Data.md](../02-User-Guide/03-Data.md).
3. Entrainement: [docs/02-User-Guide/04-Training.md](../02-User-Guide/04-Training.md).
4. Inference: [docs/02-User-Guide/05-Inference.md](../02-User-Guide/05-Inference.md).
5. Checkpoints: [docs/02-User-Guide/08-Checkpoints.md](../02-User-Guide/08-Checkpoints.md).

### Exercices

1. Lancer un script d'entrainement existant:

```bash
./bin/mimir --lua scripts/training/ponyxl_ddpm_train.lua -- --help
```

2. Faire un run court (peu d'iterations), sauvegarder un checkpoint.
3. Recharger le checkpoint et comparer une metrique simple (loss ou temps).
4. Documenter 3 hypotheses expliquant une baisse ou une hausse de performance.

### Validation de fin d'etape

Tu valides l'etape si:

- tu reproduis un mini-run de train + reload checkpoint,
- tu sais expliquer la difference train/inference,
- tu peux partager un protocole d'experience simple et reproductible.

---

## Etape 3 - Avance (fac, scientifique, dev senior)

### Mission

Analyser le framework en profondeur, optimiser les performances, et etendre les composants.

### Notions a retenir

- API et contrat de scripting inter-langages.
- Architecture interne du moteur d'execution.
- Gestion memoire, runtime allocator, et stabilite numerique.

### Parcours recommande (2 a 5 jours)

1. API overview: [docs/03-API-Reference/00-API-Overview.md](../03-API-Reference/00-API-Overview.md).
2. Memory API: [docs/03-API-Reference/14-Memory.md](../03-API-Reference/14-Memory.md).
3. Internals engine: [docs/04-Architecture-Internals/01-Engine-Overview.md](../04-Architecture-Internals/01-Engine-Overview.md).
4. RuntimeAllocator/scratchpads: [docs/04-Architecture-Internals/18-RuntimeAllocator-And-Scratchpads.md](../04-Architecture-Internals/18-RuntimeAllocator-And-Scratchpads.md).
5. Debug/perf: [docs/05-Advanced/01-Performance.md](../05-Advanced/01-Performance.md) et [docs/05-Advanced/02-Debugging.md](../05-Advanced/02-Debugging.md).
6. Dev guide: [docs/07-Devs/00-INDEX.md](../07-Devs/00-INDEX.md).

### Exercices

1. Mesurer un benchmark de reference:

```bash
./bin/mimir --lua scripts/benchmarks/benchmark_official.lua -- --safe --iters 1
```

2. Proposer une optimisation (memoire, layout, batch, precision dtype).
3. Implementer un changement minimal et mesurer avant/apres.
4. Ecrire une note technique: objectif, methode, resultats, limites.

### Validation de fin d'etape

Tu valides l'etape si:

- tu produis une mesure avant/apres argumentee,
- tu relies ton resultat a des choix runtime/memoire,
- tu identifies un risque de regression et un test de non-regression.

---

## Conseils de progression

1. Ne saute pas les etapes: la base accelere vraiment le niveau avance.
2. Garde un journal d'experiences (commande, config, resultat).
3. Utilise les checkpoints pour comparer proprement les changements.
4. Si un run echoue, reduis l'experience (taille plus petite) et isole le probleme.

## Suite logique

- Approfondir VAEText: [docs/02-User-Guide/11-VAEText.md](../02-User-Guide/11-VAEText.md)
- Approfondir Transformer/GPT: [docs/02-User-Guide/12-Transformer-GPT.md](../02-User-Guide/12-Transformer-GPT.md)
- Approfondir diffusion: [docs/02-User-Guide/13-Diffusion.md](../02-User-Guide/13-Diffusion.md)
