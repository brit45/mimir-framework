# Choisir un build portable, puis activer AMP et DDP

Ce tutoriel sépare les options de compilation, la disponibilité matérielle et
les options d'un run. La première étape consulte les capacités sans installer
de dépendances. Les recettes de compilation et DDP exigent un environnement
préparé ; elles ne sont pas des commandes de smoketest universelles.

| Niveau | Objectif |
| --- | --- |
| Débutant | Construire ou reconnaître un profil CPU sans interface. |
| Intermédiaire | Choisir un backend Viz et vérifier AMP sans changer le graphe. |
| Expert | Lancer plusieurs rangs MPI et vérifier la synchronisation effective. |

Les architectures restent des classes C++ dérivées de `Model`, selon le
[contrat d'architecture](../07-Devs/08-Architecture-Contract.md). La compilation
active des chemins d'exécution, pas une seconde définition du graphe. Lua et
JSON configurent les modèles natifs ; MPK est expérimental et n'est pas encore
pleinement exploitable en production.

## 1. Lire les options disponibles

Depuis un shell Bash à la racine du dépôt :

```bash
./config.sh --help
./run_mimir.sh --launcher-help
```

`config.sh` configure CMake et peut installer des dépendances. `run_mimir.sh`
lance un binaire existant et prépare son environnement. Les options MPI du
lanceur doivent apparaître avant le séparateur `--` destiné au script Lua.

Vérifier la plateforme et le gestionnaire de paquets détectés plutôt que de
copier une commande `apt` sur un autre système :

| Environnement | Chemin prévu par le script | Limite à connaître |
| --- | --- | --- |
| Linux | apt, dnf/yum, pacman, zypper, apk et plusieurs autres gestionnaires détectés. | Les noms de paquets et les SDK GPU varient. |
| macOS | Homebrew si présent ; dépendances manuelles sinon. | Pas de promesse de support des GPU ou hôtes Linux. |
| FreeBSD / OpenBSD / NetBSD | Gestionnaires BSD détectés selon le système. | Détection ne signifie pas compilation validée sur chaque version. |
| Windows avec MSYS2 | Préfixe MSYS2/Mingw reconnu et paquets du préfixe. | Bash requis pour ces scripts ; le toolchain doit rester cohérent. |
| Autre environnement | Dépendances installées manuellement. | Employer CMake directement ou `--skip-system-deps`. |

L'installation universelle de tous les SDK GPU n'est pas garantie. La
[page d'installation](../01-Getting-Started/02-Installation.md) fournit les
prérequis ; les matrices de validation doivent rester propres à l'OS testé.

## 2. Préparer un profil headless isolé

Recette à exécuter seulement si compilateur C++, CMake et dépendances sont
déjà installés :

```bash
./config.sh --gui NONE --build-dir build-tutoriel \
  --non-interactive --skip-system-deps \
  --amp --no-ddp --enable MIMIR_ENABLE_TESTS \
  --disable ENABLE_CUDA --disable ENABLE_ROCM \
  --disable ENABLE_VULKAN --disable ENABLE_OPENCL
cmake --build build-tutoriel --target mimir --parallel 2
```

Le dossier distinct conserve le cache du build habituel. Par défaut ce profil
écrit son binaire dans `build-tutoriel/bin/mimir`, tandis que `build/` alimente
le `bin/` du dépôt. `MIMIR_RUNTIME_OUTPUT_DIRECTORY` permet de changer cette
destination. `--skip-system-deps`
ne fournit aucune bibliothèque manquante. Examiner le résumé CMake et les
erreurs de dépendance avant de poursuivre.

Les options CMake correspondantes sont `ENABLE_AMP` et `ENABLE_MPI`. AMP est
activé par défaut dans le CMake actuel ; MPI est optionnel. Compiler avec MPI
nécessite une implémentation et ses en-têtes C++.

Les commandes CMake sont aussi utilisables dans PowerShell après préparation
d'un toolchain natif. Exemple de configuration CPU :

```powershell
cmake -S . -B build-tutoriel -DMIMIR_VIZ_BACKEND=NONE -DENABLE_AMP=ON -DENABLE_MPI=OFF -DENABLE_CUDA=OFF -DENABLE_ROCM=OFF -DENABLE_VULKAN=OFF -DENABLE_OPENCL=OFF
cmake --build build-tutoriel --target mimir --config Release --parallel 2
```

Retrouver le chemin de l'exécutable dans la sortie du build ; un générateur
multi-configuration peut ajouter un sous-dossier `Release`. La variante
PowerShell ne lance pas `config.sh` et suppose les dépendances déjà disponibles.

## 3. Choisir l'interface séparément du calcul

| Backend Viz | Usage | Disponibilité du code actuel |
| --- | --- | --- |
| `NONE` | Serveur ou terminal sans fenêtre. | Profil de départ le moins dépendant. |
| `SFML` | Fenêtre native avec SFML 3. | Selon les plateformes supportées par le build. |
| `QT`, `GTK` | Hôte desktop séparé. | Linux actuellement. |
| `WEB` | Navigateur local et Canvas2D. | Hôte Linux, écoute locale. |

`MIMIR_VIZ_BACKEND` ne sélectionne ni Vulkan ni OpenCL. Une interface Web peut
observer un entraînement CPU ; un build sans interface peut calculer sur GPU.
Voir [le choix de Viz](../02-User-Guide/17-Viz-Backends.md).

## 4. Vérifier AMP sur une expérience existante

Commencer par le petit run causal du [tutoriel entraînement](12-Tuto-Entrainement-Validation-Reprise-Inference.md),
avec un dossier de sortie propre. Ajouter ensuite :

```text
--amp --amp-dtype bfloat16
```

Ces options sont acceptées par `train_causal_lm.lua`. Le script appelle
`Mimir.Model.set_autocast(...)`. Un binaire compilé sans `ENABLE_AMP` refuse
l'autocast réduit au lieu de l'activer silencieusement. `--no-amp` désactive le
chemin demandé par ce script.

Ne pas confondre `--dtype`, qui sélectionne la préférence de stockage du modèle,
et l'autocast de calcul. Les accumulations et certaines opérations peuvent
rester FP32. Le libellé `bfloat16` ne prouve ni instructions natives BF16 ni gain
de vitesse. Comparer la finitude de la loss, les gradients, les métriques de
validation et le temps complet au run FP32 de référence.

Le script de vérification du setter de dtype peut se lancer sans dataset :

```bash
./run_mimir.sh --lua scripts/benchmarks/dtype_api_smoke.lua
```

Il vérifie l'API de dtype, pas une preuve d'exécution AMP native.

## 5. Préparer DDP sans confondre threads et rangs

DDP nécessite un build `ENABLE_MPI=ON`, un lanceur MPI, le corpus de l'atelier
causal et un tokenizer partagé compatible. Les rangs sont des processus ;
OpenMP utilise des threads dans chaque processus.

Recette de compilation conditionnelle, distincte du profil précédent :

```bash
./config.sh --gui NONE --build-dir build-ddp \
  --non-interactive --skip-system-deps --ddp --amp
cmake --build build-ddp --target mimir --parallel 2
```

Le lanceur utilise normalement `bin/mimir`. Le build secondaire ci-dessus
produit `build-ddp/bin/mimir` ; la commande suivante le sélectionne explicitement
avec `MIMIR_EXECUTABLE`.
Après création du corpus du tutoriel 11, la recette suivante utilise un nouveau
dossier de sortie :

```bash
MIMIR_EXECUTABLE="$PWD/build-ddp/bin/mimir" \
./run_mimir.sh --mpi-ranks 2 --lua scripts/training/train_causal_lm.lua -- \
  --corpus /tmp/mimir_tutoriels/corpus.txt \
  --tokenizer /tmp/mimir_tutoriels/causal_ddp/tokenizer.json \
  --checkpoint-dir /tmp/mimir_tutoriels/causal_ddp \
  --vocab-size 512 --bpe-merges 40 --seq-len 32 \
  --d-model 16 --layers 1 --heads 2 --kv-heads 1 \
  --mlp-hidden 32 --latent-dim 8 --steps 4 --lr 0.0003 \
  --optimizer adamw --seed 1337 --log-every 1 --save-every 2 \
  --ddp --no-amp --no-viz --no-htop
```

Cette recette nécessite MPI ; elle ne fait pas partie du smoketest CPU sans
MPI. Réutiliser un dossier existant exige une reprise explicite, pas un run neuf
sur ses fichiers. `--mpi-ranks 2` lance `mpirun -n 2`.
`--mpi-exec` choisit un lanceur alternatif. Les alias `--mpi` et
`--ddp-ranks` existent également. Le flag `--ddp` du script reste nécessaire :
deux processus indépendants ne constituent pas à eux seuls un entraînement DDP.

Le lanceur refuse les rangs inférieurs à deux et un MPI imbriqué détecté. Il
répartit les threads OpenMP par défaut entre les rangs ; un réglage explicite
d'`OMP_NUM_THREADS` reste possible. Le modèle doit synchroniser ses gradients
via les collectives MPI. Les interfaces et écritures partagées suivent le rang
principal, plutôt que d'ouvrir plusieurs fenêtres pour le même run.

## 6. Exercices et limites matérielles

| Niveau | Exercice | Vérification |
| --- | --- | --- |
| Débutant | Comparer le résumé CMake et l'aide du binaire. | Une option compilée est distinguée d'un device disponible. |
| Intermédiaire | Refaire le même petit run avec et sans AMP. | Les valeurs non finies et le coût global sont contrôlés. |
| Expert | Comparer un et deux rangs. | Les étapes, gradients synchronisés, débits et fichiers de sortie sont cohérents. |

Les runtimes CUDA, ROCm, Vulkan, OpenCL et TPU ont leurs SDK, pilotes, budgets
et limites propres. Une compilation CUDA ne valide pas une carte NVIDIA ;
une syntaxe OpenCL correcte ne valide pas son exécution. Le runtime FPGA reste
un sujet expérimental ; ne pas le présenter comme un GPU généraliste.

## Étapes suivantes

- [Runtimes et preuves matérielles](../04-Architecture-Internals/21-GPU-Runtimes.md)
- [Étude FPGA](../05-Advanced/06-FPGA-iCESugar-Pro-Runtime-Study.md)
- [Exports, tests et contribution](17-Tuto-Scripting-Exports-Tests-Contribution.md)
- [Index des tutoriels](00-INDEX.md)
