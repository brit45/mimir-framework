# Latence des layers et des runtimes — 12 novembre 2026

La [suite sur l'attention par blocs](Runtime-Attention-Tiling-20261112.md)
optimise ensuite les longues séquences. Les mesures ci-dessous décrivent
l'étape précédente, conservée comme référence.

La campagne compare le moteur au commit `350ad97dbeb5` sur un Ryzen 5 5500U,
avec `run_mimir.sh`, AVX2 et le même ordre de calcul FP32. Les données et les
limites ci-dessous concernent cette machine. Une autre tâche d'entraînement
était active : les petits écarts de temps peuvent venir de la charge système.
Le cœur est compilé en C++17/O3 portable, les sondes de layers en O2, avec
`-ffp-contract=off` pour le dispatcher SIMD, dans les deux versions comparées.

Les médianes, empreintes des valeurs, tailles et étapes isolées figurent dans
[la synthèse JSON](Runtime-Dimensions-20261112.json). Les temps bruts, journaux
de validation et comparaisons initiales restent dans
`outputs/benchmarks/layer-dimensions-20261112/`.

## Changements et inspection des six runtimes

| Surface | Constat et action | Validation locale |
|---|---|---|
| CPU | Les projections relisaient les matrices avec des strides qui pénalisaient les grandes dimensions. GEMM ordonné par blocs, transposition de petits panneaux par mouvements SSE2, accès stridés pour les gradients. Chemin direct conservé pour les petits produits et les faibles réductions contiguës. | Comparaisons binaires des sorties/gradients ; scalaire, SSE2, AVX, AVX2 ; AVX-512 indisponible. |
| SelfAttention / CrossAttention | Projections forward et gradients dirigés vers les GEMM compatibles avec leur layout. Le backward SelfAttention emprunte une arène réutilisable ; dP/dS sont réemployés entre les têtes. | 68 comparaisons de layers couvrant 512/1024/2048, batches 1/16/64, 1 et 4 workers explicites, séquences 512/2048 à largeur 64. |
| OpenCL | Deux uploads initiaux non bloquants dans la file ordonnée du dispositif portable ; lecture finale bloquante. Drainage de la file à chaque échec pour protéger la durée de vie des tableaux hôtes. Les arguments temporaires des passes d'optimiseur gardent leur écriture bloquante. | GPU AMD/Rusticl : 207 cas natifs, quatre erreurs injectées et quatre reprises. Aucun gain de débit GPU revendiqué. |
| Vulkan | Pipelines et buffers déjà conservés ; barrières, fences et lecture de staging nécessaires à la cohérence conservées. | GPU AMD/RADV : 207 cas natifs. Aucun gain de débit GPU revendiqué. |
| CUDA | Revue des GEMM batchés cuBLAS et des scratchpads persistants Linear/pointwise. Les transferts des paramètres à chaque appel protègent les modifications en place. | Revue du code ; aucun nvcc ni périphérique NVIDIA disponible. |
| ROCm | Les activations et opérations binaires réutilisent les buffers du runtime sous mutex. Suppression du `hipDeviceSynchronize` redondant avant la copie D2H bloquante. Paramètres/entrées retransférés à chaque appel. | Compilation HIP pour gfx90c réussie ; exécution ROCm désactivée. |
| TPU | Suppression du tableau de bruit de la taille de l'entrée pour les forwards déterministes. Tirages, clé RNG et masque Dropout conservés. Processus persistant et cache JIT existants conservés. | Référence JAX sur CPU : 133 cas, 74 types ; protocole persistant, Dropout et Reparameterize en entraînement et inférence. Aucun TPU physique disponible. |
| Layers.hpp | `getWeights()` emprunte le stockage existant ; `getWeightsSize()` utilise la taille déclarée du bloc. Accès constants et mutations publiques préservés. | Revue et tests des paramètres dans les chemins natifs. |
| Model.cpp / Model.hpp | Snapshots typés déjà paresseux ; conservation des valeurs nécessaires au backward et de la propriété des résultats. API et layout des paramètres préservés. | API publique `trainStep`, inférence, tests de snapshots et du planner. |

## Calculs et exactitude

Chaque produit scalaire réduit les termes dans l'ordre croissant de la dimension
réduite. Les chemins avec FMA conservent FMA ; les chemins séparant produit et
addition conservent cette séparation. Un gradient préexistant reçoit le produit
scalaire terminé une seule fois. Les biais QKV de SelfAttention restent la
valeur initiale de la réduction. Les blocs intermédiaires reprennent
l'accumulateur FP32 ; ils n'additionnent pas des sommes partielles réordonnées.

Les panneaux temporaires occupent au plus 30 Kio par worker GEMM et 24 Kio pour
la transposition. L'arène SelfAttention utilise le pool soumis à MemoryGuard,
avec des slices disjointes, sans référence conservée dans les résultats. Le
nombre d'allocations C++ du backward passe de 17 à 1 dans la sonde chaude
(width 32, quatre têtes, séquences 1/7/13) ; la dernière allocation porte le
gradient retourné. La comparaison binaire de cette sonde couvre 312 440 octets.

Les tests par différences finies couvrent les entrées et paramètres de Linear,
SelfAttention sans biais et CrossAttention sans biais, dont les longueurs
query/key différentes et le masque causal. **Limite préexistante :** les
backwards des attentions conservent leur contrat historique concernant les
biais, notamment l'absence de gradient des biais. Cette optimisation ne corrige
pas ce contrat ; les comparaisons de parité incluent aussi ses variantes avec
biais. L'identité bit à bit vérifie la conservation du comportement existant.

## Layers CPU, batch de 64 lignes

Trois répétitions après deux chauffes, un worker explicite. Temps en ms.
Pour Conv2d, les entrées font 8×8 et les canaux valent largeur/8, soit
64/128/256 canaux ; la colonne largeur n'est pas un nombre de canaux Conv2d.

| Layer | Largeur | Forward avant → après | Backward avant → après |
|---|---:|---:|---:|
| Linear | 512 | 3.280 → 1.234 | 20.270 → 3.047 |
| SelfAttention | 512 | 53.573 → 6.077 | 496.077 → 37.525 |
| CrossAttention | 512 | 21.058 → 6.828 | 556.738 → 45.196 |
| Conv2d | 512 | 0.487 → 0.306 | 0.542 → 0.436 |
| Linear | 1024 | 28.667 → 8.270 | 98.450 → 20.634 |
| SelfAttention | 1024 | 225.062 → 24.888 | 3856.503 → 182.679 |
| CrossAttention | 1024 | 194.149 → 44.173 | 3442.898 → 196.970 |
| Conv2d | 1024 | 1.462 → 0.718 | 1.526 → 1.417 |
| Linear | 2048 | 125.892 → 35.853 | 516.595 → 88.035 |
| SelfAttention | 2048 | 976.756 → 98.876 | 15910.795 → 545.263 |
| CrossAttention | 2048 | 981.637 → 181.089 | 15913.677 → 553.286 |
| Conv2d | 2048 | 6.571 → 3.385 | 8.040 → 6.632 |

Le gain concerne surtout les projections de grande largeur. À largeur 64 et
séquence 2048, les boucles quadratiques de l'attention dominent encore : le
backward passe de 6 932 à 6 655 ms, environ 4 %. Les normalisations gardent leur
implémentation et servent aussi de contrôle de la variabilité de la machine.

## API complète Model

MLP de contrôle : deux paires Linear/SiLU, float32, MSE, AdamW, batch 64.
Dix étapes de chauffe puis cinq répétitions. Le tableau utilise un thread
OpenMP, les GPU désactivés et la calibration désactivée pour fixer le routage
CPU sériel. Les temps comprennent le chemin public complet ; chaque étape
isolée et l'optimiseur figurent séparément dans le JSON.

| Largeur | trainStep avant → après | Inférence avant → après | Gain trainStep |
|---|---:|---:|---:|
| 512 | 25.755 → 12.050 ms | 7.256 → 3.830 ms | ×2.14 |
| 1024 | 195.258 → 58.822 ms | 54.015 → 13.526 ms | ×3.32 |
| 2048 | 1136.661 → 254.321 ms | 215.797 → 51.087 ms | ×4.47 |

Avec la calibration active et `OMP_NUM_THREADS=4`, le profil largeur 1024 / batch
64 passe de 71.644 à 65.831 ms par `trainStep`, soit environ
8.1 % de temps en moins. L'inférence passe de 15.036 à
12.019 ms ; la première étape incluant la calibration passe de
1545.3 à 630.5 ms. Le choix déjà parallèle de l'ancien moteur réduit
le gain relatif : les facteurs du tableau sériel ne s'appliquent pas tels quels
à une exécution calibrée. `OMP_NUM_THREADS` limite les candidats ; le planner
sélectionne ses workers par opération.

Les huit profils Model comparent les prédictions et les poids finaux bit à bit.
Les 68 comparaisons de layers comparent les sorties et gradients bit à bit.
Les variations de temps de l'optimiseur isolé sont documentées ; son algorithme
n'a pas été changé dans cette intervention. Le test utilise un MLP synthétique,
sans dataset ni fenêtre Viz active.

## Reproduire les sondes

Dans une compilation avec `MIMIR_ENABLE_TESTS=ON` et `BUILD_TESTING=ON` :

```sh
cmake --build build --target mimir_bench_layer_dimensions mimir_bench_whole_model -j2
MIMIR_EXECUTABLE="$PWD/build/Tests/mimir_bench_layer_dimensions" \
  MIMIR_DISABLE_ROCM=1 OMP_NUM_THREADS=1 \
  ./run_mimir.sh 2048 64 5 1 all
MIMIR_EXECUTABLE="$PWD/build/Tests/mimir_bench_whole_model" \
  MIMIR_DISABLE_CUDA=1 MIMIR_DISABLE_ROCM=1 \
  MIMIR_DISABLE_OPENCL=1 MIMIR_DISABLE_VULKAN=1 \
  MIMIR_RUNTIME_AUTOTUNE=0 OMP_NUM_THREADS=1 \
  ./run_mimir.sh float32 5 2 2048 64 2
```

La sonde de layers accepte `largeur lignes répétitions workers [type]` et écrit
les temps bruts en JSON. `MIMIR_LAYER_VALUES=/tmp/values.bin` active le dump
binaire des sorties/gradients ; `MIMIR_BENCH_VALUES` active celui des prédictions
et paramètres du MLP. Comparer deux builds distincts avec le même ISA, les mêmes
entrées, paramètres, affinités et réglages de routage. Pour reproduire la
calibration, régler `MIMIR_RUNTIME_AUTOTUNE=1` et une limite OpenMP identique.
Pour la référence, compiler la même sonde avec les headers et la bibliothèque
du commit `350ad97dbeb5` ; la sonde de dimensions doit être copiée dans ce build.

Validation locale : trois CTest indépendants (SIMD, rejet d'ISA invalide,
scratchpads), six tests natifs ciblés (dérivées, Linear, attention, convolution,
snapshots, planner), deux suites GPU de 207 cas, test de durée de vie OpenCL,
référence CPU/JAX et compilation HIP. La configuration CMake avec tests et
OpenCL/Vulkan activés enregistre aussi le nouveau test d'injection d'erreurs.
