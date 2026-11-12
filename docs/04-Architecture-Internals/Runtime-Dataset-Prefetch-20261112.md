# Préparation anticipée du dataset — 12 novembre 2026

Le lecteur lazy prépare un nombre fixe d'échantillons d'avance sur un thread
natif : dix via le lanceur, huit avec le défaut natif sans variable. La
lecture, le décodage, le redimensionnement et les conversions
déterministes se déroulent pendant le calcul du modèle. Un emplacement
supplémentaire contient l'échantillon consommé ; ses buffers redeviennent
disponibles après utilisation. Les contenus sont déchargés sur le producteur.

Le [rapport JSON](Runtime-Dataset-Prefetch-20261112.json) contient les mesures,
les empreintes des sources, des exécutables et des tenseurs comparés. Les
journaux, fixtures et résultats bruts restent dans
`outputs/benchmarks/dataset-prefetch-20261112/`.

## Contrat et intégration

`DatasetPrefetcher` possède un instantané des métadonnées et des emplacements
stables. Un échantillon emprunté reste valide pendant les lectures suivantes,
une validation imbriquée, ou l'arrêt du lecteur. Une permutation explicite
conserve les doublons et l'ordre restauré depuis un checkpoint. Les demandes
du consommateur ont priorité sur les lectures spéculatives.

Le producteur ne touche ni au modèle, ni au RNG d'entraînement, ni aux états
Lua. Il prépare RGB, HWC ou CHW selon le consommateur. Les conversions des
pixels conservent les expressions et les arrondis précédents, y compris la
division FP32 propre au détecteur médical. VAEText utilise un instantané du
tokenizer et prépare ses tokens et son padding ; le choix des fenêtres
aléatoires reste sur le thread d'entraînement. Les buffers normalisés sont
échangés avec ceux de l'entraînement, sans copie complète par batch.

Les boucles natives VAEConv, VAEText, VGG16Feat, VGG16, VGG19 et du détecteur
médical utilisent ce lecteur. `Dataset.get` anticipe les accès séquentiels ;
`Dataset.prefetch(order)` permet aux scripts de fournir un ordre mélangé.
Lumen fournit son ordre après mélange ou restauration. Les bridges
JavaScript/Rust et les helpers IDE exposent aussi cette API.

`MIMIR_DATASET_PREFETCH` fixe le nombre d'éléments d'avance, avec un défaut de
8 dans le lecteur natif et un maximum de 256. Le lanceur `run_mimir.sh`
exporte 10 si la variable est absente ; les mesures ci-dessous imposent
explicitement 8. `MIMIR_DATASET_PREFETCH_BYTES` fixe le budget cible des
contenus et capacités réutilisables, par défaut 64 Mio. Mettre l'une à zéro
conserve le chemin synchrone. L'index de métadonnées et les temporaires des
décodeurs sont distincts de ce budget. Un élément demandé trop volumineux peut
être préparé seul, toujours sous les contrôles RAM du dataset. Sous pression,
les contenus spéculatifs sont évincés avant les données encore utilisées.

Les appels, exemples et compteurs sont décrits dans
l'[API Dataset](../03-API-Reference/13-Dataset.md#préchargement-en-arrière-plan).
`buffer_growths` compte les agrandissements des buffers de conversion, sans
prétendre compter toutes les allocations des décodeurs.

## Mesures de débit CPU

Harnais natif `Model::trainStep` : Linear et SiLU, FP32, MSE et AdamW. Les
images PPM sources mesurent deux fois la résolution cible. Chaque campagne
compare le même calcul avec zéro puis huit éléments d'avance, sur trois
répétitions ; l'ordre est inversé lors de la deuxième répétition. Le lecteur
et le modèle utilisent respectivement un thread et un worker CPU, sur Ryzen
5 5500U avec AVX2. Les backends GPU sont désactivés et la calibration est
désactivée. Toutes les exécutions passent par `./run_mimir.sh`.

Les temps totaux incluent le premier échantillon, les attentes de lecture et
conversion, forward/loss/backward/optimiseur, puis la libération synchrone ou
la restitution au producteur. La génération des fichiers, la construction du
modèle, l'instantané des métadonnées, l'arrêt final du lecteur et les dumps
de contrôle sont hors chronométrage. Le temps total indiqué est la médiane
des trois exécutions ; la colonne d'attente est la médiane des trois médianes
par échantillon, premier élément compris.

| Résolution cible | Largeur Linear | Étapes | Total synchrone → anticipation | Débit | Attente médiane synchrone → anticipation | Pic RSS médian |
|---|---:|---:|---:|---:|---:|---:|
| 128 × 128 | 64 | 32 | 281,849 → 215,752 ms | ×1,31 | 5,688 → 0,0014 ms | 35,19 → 37,14 Mio |
| 128 × 128 | 512 | 24 | 483,663 → 340,611 ms | ×1,42 | 5,787 → 0,0016 ms | 38,74 → 41,07 Mio |
| 256 × 256 | 512 | 24 | 1 780,213 → 1 193,526 ms | ×1,49 | 22,747 → 0,0015 ms | 44,79 → 54,82 Mio |

Les neuf paires conservent exactement les prédictions et poids finaux. Le
pool atteint respectivement 2,11 et 8,44 Mio pour les résolutions 128 et 256,
avec neuf agrandissements de buffers lors du remplissage initial, puis leur
réutilisation. Ces mesures synthétiques ne garantissent pas le même gain sur
un dataset réel. La fréquence CPU, l'affinité, le cache du système de fichiers
et la charge extérieure ne sont pas isolés ; le producteur dispose d'un
thread CPU supplémentaire.

La petite projection de largeur 64 laisse encore 10 à 13 demandes en attente
sur 32 : la préparation ne suit pas toujours le calcul. Les deux profils de
largeur 512 attendent uniquement le premier élément sur ces mesures. La
médiane proche de zéro ne signifie donc pas une absence d'attente pour tous
les accès.

## Validation de la précision et du cycle de vie

- Six entraînements natifs complets comparent le lecteur synchrone, trois
  éléments d'avance et l'exécutable conservé avant cette modification.
  Les 244 fichiers de tenseurs sont identiques bit à bit : paramètres,
  gradients et états d'optimiseur présents dans les checkpoints.
- Un VAEConv vidéo compare 24 autres fichiers identiques, avec une couverture
  différente des frames. Le lecteur prépare le texte et le masque de cette
  entrée ; le modèle continue à consommer les vraies frames en streaming.
- Les reprises Lumen en fin et au milieu d'une époque restent identiques à
  l'entraînement continu, avec anticipation activée et désactivée. Les indices
  entiers restaurés comme nombres flottants Lua sont acceptés, tandis que les
  fractions, chaînes et valeurs hors limites sont refusées.
- Sept CTest ciblés passent avec le noyau natif réel : anticipation, RGB,
  symlinks, annotations, Lua anticipé/synchrone et reprise. Les tests Lua sont
  repassés après la correction des indices restaurés. JavaScript et Rust
  exécutent réellement les nouvelles fonctions ; leurs sockets locaux ont
  nécessité une exécution hors sandbox des outils.
- Les tests du producteur couvrent RGB/HWC/CHW, les arrondis float/double,
  masques, texte, tokens et padding, permutations et doublons, fichiers
  absents/corrompus, récupération sous budget RAM, buffers réutilisés,
  validation imbriquée, arrêt pendant une lecture et leases encore actifs.
  Une FIFO bloquée vérifie que les statistiques et le changement d'ordre
  restent accessibles pendant une lecture.
- ASan/UBSan et TSan passent, ainsi que vingt répétitions du cycle de vie.
  Le harnais sanitizer utilise le vrai lecteur, Helpers, Tokenizer et stb,
  avec staging FPGA neutre ; TSan est exécuté sans OpenMP et la détection des
  fuites ASan est désactivée. Le compteur RAM du dataset revient au niveau
  initial. Le test natif complet utilise le staging réel sans matériel FPGA.
- La compilation Release, la configuration CMake des tests, la synchronisation
  des helpers IDE, les vérifications docs/API et `git diff --check` passent.

## Limites pratiques

Le premier élément et les accès imprévus peuvent attendre. Un producteur plus
lent que l'entraînement finit par vider la file ; une profondeur supérieure
n'accélère pas le décodage lui-même. La fermeture attend la lecture déjà
engagée et abandonne le reste de la permutation.

Les tables Lua/JSON restent matérialisées sur le thread du langage. Les
lectures directes via `Mimir.IO.readImageRGBU8`, notamment les deux jeux
source/style du script GAN, ne passent pas par ce lecteur Dataset. Les
vidéos conservent leur décodage de frames en streaming. Le chargement eager
explicite conserve son contrat. Cette campagne mesure le chemin hôte CPU ;
elle ne mesure aucun gain sur GPU ou TPU physique.

## Reproduire

Dans un build déjà configuré avec `MIMIR_ENABLE_TESTS=ON` :

```sh
cmake --build build --target mimir mimir_bench_dataset_prefetch FrameworkTest.DatasetPrefetch -j2
ctest --test-dir build -R 'FrameworkTest.Dataset(Prefetch|RGB|Symlinks|Annotations)|LuaTest.Dataset(Prefetch|TrainingPrefetch)' --output-on-failure
python3 scripts/tests/test_dataset_training_prefetch.py --binary build/bin/mimir
MIMIR_DATASET_PREFETCH=3 MIMIR_DISABLE_ROCM=1 OMP_NUM_THREADS=1 \
  MIMIR_TEST_BINARY="$PWD/build/bin/mimir" python3 scripts/tests/test_lumen_checkpoint_resume.py
```

Pour la mesure, répéter les deux appels et inverser leur ordre entre séries :

```sh
MIMIR_EXECUTABLE="$PWD/build/Tests/mimir_bench_dataset_prefetch" \
  MIMIR_DISABLE_CUDA=1 MIMIR_DISABLE_ROCM=1 MIMIR_DISABLE_OPENCL=1 MIMIR_DISABLE_VULKAN=1 \
  MIMIR_RUNTIME_AUTOTUNE=0 MIMIR_PLANNER_DUMP=0 OMP_NUM_THREADS=1 MIMIR_CPU_ISA=avx2 \
  ./run_mimir.sh 0 24 256 512
MIMIR_EXECUTABLE="$PWD/build/Tests/mimir_bench_dataset_prefetch" \
  MIMIR_DISABLE_CUDA=1 MIMIR_DISABLE_ROCM=1 MIMIR_DISABLE_OPENCL=1 MIMIR_DISABLE_VULKAN=1 \
  MIMIR_RUNTIME_AUTOTUNE=0 MIMIR_PLANNER_DUMP=0 OMP_NUM_THREADS=1 MIMIR_CPU_ISA=avx2 \
  ./run_mimir.sh 8 24 256 512
```

Les arguments sont : éléments d'avance, étapes, côté de l'image cible,
largeur Linear. `MIMIR_BENCH_VALUES` active le dump binaire de prédiction et
poids pour comparer les deux chemins. Le test Python accepte aussi
`--reference /chemin/mimir-avant` et `--models video_vae_conv` pour les contrôles
complémentaires ; le second nécessite FFmpeg dans le build et sur le système.
