# Benchmarks Lua natifs

Depuis la racine du dépôt :

```sh
cmake --build build -j2 --target mimir
OMP_NUM_THREADS=2 ./bin/mimir --lua scripts/benchmarks/run_suite.lua -- --suite all --quick
./bin/mimir --conf configs/benchmarks-config.json --run smoke
```

La suite par défaut est `core`. `all` ajoute les contrôles API/spill et le stress
long ; `--quick` réduit aussi le stress. Chaque script s'exécute dans un processus
séparé. `--keep-going` poursuit après un échec, puis renvoie une erreur globale.
`ROOTWORK` désigne le projet ; `MIMIR_BIN` permet de choisir le binaire enfant.
Les variables `env` du fichier de configuration sont appliquées et héritées.
Aucun dataset n'est utilisé.

| Script | Travail mesuré ou vérifié |
|---|---|
| `benchmark.lua` | Tokenizer, Transformer causal/non causal, sauvegarde/rechargement SafeTensors |
| `benchmark_attention.lua` | Forward VAEConv avec attention spatiale et Transformer sur tokens entiers |
| `benchmark_conv_train.lua` | Vraie couche Conv2d 3×3, backward, AdamW, baisse de MSE |
| `benchmark_complet.lua` | Transformer, ViT, UNet, VAE, ResNet, diffusion, Add multi-entrée, SafeTensors |
| `benchmark_official.lua` | Construction du graphe, allocation et initialisation séparées |
| `benchmark_stress.lua` | Entraînement MLP croissant et créations répétées |
| `benchmark_nms.lua` | Suppression de boîtes synthétiques et validité des indices |
| `dtype_api_smoke.lua` | Setter/getter et rejet d'un dtype invalide |
| `spill_cleanup_smoke.lua` | Spill réellement observé, puis dossier vide après sortie de l'enfant |

Les scripts utilisant `common.lua` acceptent `--ram` (GiB), `--dtype`,
`--no-compress` et `--no-accel` via leur configuration commune, sauf les contrôles
spécialisés dtype/spill. Les forwards acceptent `--iters` et `--warmup`.
Les paramètres propres à chaque script figurent dans ses commentaires de tête.
Les variables `MIMIR_BENCH_RAM_GB`, `MIMIR_BENCH_ITERS` et `MIMIR_DTYPE`
sont utilisées lorsque l'option CLI correspondante est absente.
Pour régler précisément un benchmark, lancer son script directement :

```sh
./bin/mimir --lua scripts/benchmarks/benchmark_attention.lua -- --warmup 3 --iters 20
./bin/mimir --lua scripts/benchmarks/benchmark_conv_train.lua -- --steps 100
./bin/mimir --lua scripts/benchmarks/benchmark_official.lua -- --safe --iters 2
```

Les durées `cpu_ms` utilisent `os.clock` : temps CPU cumulé du processus,
**pas une latence murale**, particulièrement avec OpenMP. Les mesures de forward
incluent le binding, le transfert vers les tables Lua et leur validation.
NMS exprime son débit par seconde CPU. Fixer le nombre de threads, le binaire,
les dimensions et les options pour comparer deux exécutions.

`set_hardware(bool)` contrôle l'accélération des kernels ; il ne sélectionne
pas un backend nommé et ne prouve pas qu'un calcul a été exécuté sur GPU.
Le dtype affiché est la préférence du modèle, pas une preuve de calcul natif
en basse précision. Les compteurs MemoryGuard/Allocator ne sont pas le RSS.
Le pic mémoire reste cumulatif : aucun `MemoryGuard.reset()` n'est exécuté
alors qu'un modèle est vivant. Les répétitions ne prouvent pas l'absence de fuite.

Les profils `safe` et `quick` de construction utilisent des dimensions modestes ;
`--full` et `--extreme` restent explicites. La limite de l'allocateur n'est pas une
limite globale du processus. Le test spill utilise un budget spécifique de 4 MiB,
sans compression, dans un dossier temporaire isolé avec un lanceur POSIX.
Les fichiers temporaires sont nettoyés après succès ; ceux d'un test spill en
échec sont conservés au chemin indiqué. Aucun checkpoint utilisateur n'est écrasé.
