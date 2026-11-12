# Attention par blocs — suite du 12 novembre 2026

Cette suite optimise le backward CPU de SelfAttention et CrossAttention pour
les longues séquences. La référence est **le moteur déjà optimisé de la
[première campagne](Runtime-Dimensions-20261112.md)**, avant cette continuation.
Les facteurs ci-dessous mesurent ce gain supplémentaire.

Les [mesures JSON](Runtime-Attention-Tiling-20261112.json) conservent les
médianes, étapes Model, tailles, empreintes des valeurs et sources. Les temps
bruts et journaux sont dans `outputs/benchmarks/attention-tiling-20261112/`.
Une autre tâche d'entraînement était active sur la machine ; les petits écarts
et les contrôles inchangés varient avec la charge.

## Calcul et mémoire

`AttentionBackward.cpp` calcule les probabilités par blocs de 64 requêtes.
Deux panneaux par worker contiennent les probabilités et leur dérivée ; le
second est réemployé pour la dérivée des scores. Les têtes réutilisent ces
panneaux. La dimension clé reçoit 16 floats de padding, exclus des calculs,
pour éviter les grands strides en puissance de deux lors des réductions
transposées.

Les gradients de K et V continuent leur accumulateur FP32 **terme par terme**
d'un bloc au suivant, via `RuntimeSimd::matmulStridedContinue`. Additionner des
GEMM partiels changerait l'ordre des arrondis ; cette implémentation ne le
fait pas. La mise à l'échelle de dK intervient après la réduction complète.
Les probabilités, le masque causal fini `-1e9`, le maximum initial `-1e30` et
l'epsilon `1e-9` restent identiques au backward précédent. Les produits et
additions restent séparés, sans approximation des exponentielles.

Les têtes peuvent s'exécuter sur les workers autorisés par `CpuScope`. Chaque
thread possède son slot, avec une allocation avant la région OpenMP et un
retour après la barrière. La boucle des têtes n'utilise pas `omp simd`, car les
têtes successives d'un thread partagent son slot. Les appels déjà imbriqués
dans une région OpenMP restent sériels.

Pour R = min(64, nombre de requêtes), K clés et D dimensions par tête, un slot
contient `2*(R*(K+16)+K*D)` floats. Le stockage des scores ne croît plus comme
`heads*queries*keys`. La complexité arithmétique de l'attention reste
quadratique en longueur. Les très petites attentions (au plus 16 requêtes,
16 clés et 16 dimensions par tête) utilisent deux panneaux de 256 floats sur
la pile pour éviter le coût de préparation des GEMM.

Les arènes restent soumises à MemoryGuard et sont empruntées de manière
exclusive. Le chemin scalaire demandé par la désactivation de l'accélération
matérielle est conservé. La voie GQA/RoPE dédiée n'est pas remplacée. Le contrat
historique des biais d'attention est conservé, avec la limite concernant leurs
gradients déjà décrite dans le rapport précédent.

## Layers complets, largeur 64

Mesures CPU AVX2 sur Ryzen 5 5500U. Trois répétitions après deux chauffes.
Les comparaisons binaires couvrent sortie, gradients d'entrées et paramètres.
Les workers sont imposés explicitement par la sonde native ; un nombre OpenMP
maximal ne garantit pas à lui seul le même choix dans le planner du modèle.

| Layer | Tokens | Workers | Backward avant → après | Gain | Pic RSS avant → après |
|---|---:|---:|---:|---:|---:|
| SelfAttention | 512 | 1 | 244.820 → 39.723 ms | ×6.16 | 18.1 → 8.6 Mio |
| CrossAttention | 512 | 1 | 175.031 → 41.463 ms | ×4.22 | 18.2 → 8.6 Mio |
| SelfAttention | 1024 | 1 | 1890.116 → 155.974 ms | ×12.12 | 49.9 → 10.5 Mio |
| CrossAttention | 1024 | 1 | 858.707 → 168.075 ms | ×5.11 | 50.4 → 10.8 Mio |
| SelfAttention | 2048 | 1 | 8014.944 → 610.112 ms | ×13.14 | 173.9 → 15.2 Mio |
| CrossAttention | 2048 | 1 | 4035.632 → 676.397 ms | ×5.97 | 174.5 → 15.0 Mio |
| SelfAttention | 1024 | 4 | 1888.433 → 39.381 ms | ×47.95 | 49.9 → 12.3 Mio |
| CrossAttention | 1024 | 4 | 837.074 → 42.967 ms | ×19.48 | 50.2 → 12.5 Mio |
| SelfAttention | 2048 | 4 | 8087.811 → 161.102 ms | ×50.20 | 174.0 → 18.7 Mio |
| CrossAttention | 2048 | 4 | 3956.107 → 178.673 ms | ×22.14 | 174.5 → 18.5 Mio |

Les sondes couvrent également les largeurs 512/1024/2048 avec 64 lignes, les
petites séquences 1/7/13, une largeur irrégulière 15 et deux contrôles inchangés
(Linear et LayerNorm). Les 26 profils conservent les valeurs bit à bit. Le
chemin direct des petites attentions supprime la régression de préparation
observée dans le premier prototype ; les temps de quelques microsecondes
restent sensibles au bruit de mesure.

## Model::trainStep

Modèle natif de contrôle : une paire SelfAttention/SiLU, largeur 64, huit têtes,
float32, MSE et AdamW, sans biais. CPU avec un worker et calibration désactivée.
Dix étapes de chauffe puis trois répétitions. Le temps d'entraînement inclut
forward, loss, backward, mise à jour de l'optimiseur et orchestration Model.

| Tokens | trainStep avant → après | Gain | Inférence avant → après |
|---|---:|---:|---:|
| 512 | 291.602 → 67.035 ms | ×4.35 | 21.367 → 22.871 ms |
| 1024 | 2067.779 → 246.810 ms | ×8.38 | 88.577 → 89.184 ms |

Les prédictions et les poids finaux sont identiques bit à bit après les étapes
d'entraînement et les mises à jour répétées. L'inférence conserve son calcul ;
cette suite porte sur le backward. Le pic RSS de la sonde Model reste ici à
environ 85 Mio : la réduction du scratch d'un layer ne se traduit pas toujours
par la même baisse du pic RSS de l'application. Il s'agit d'un modèle de
contrôle synthétique, sans dataset ni fenêtre Viz.

## Validation

- Quatre CTest : SIMD, attention par blocs, rejet d'ISA invalide et scratchpads.
- Référence scalaire indépendante : têtes, longueurs query/key différentes,
  fins de panneaux, valeurs extrêmes du masque causal, workers, réutilisation,
  imbrication OpenMP et refus de MemoryGuard avant modification des résultats.
- Layers intégrés : 64 configurations, deux appels successifs, causalité,
  biais, workers 1/3, gradients déjà non nuls et mutations en place.
- Dérivées numériques des entrées et paramètres de Linear et attentions sans
  biais ; conservation des contraintes documentées pour les variantes biaisées.
- GCC avec et sans OpenMP, Clang sans OpenMP ; ASan/UBSan réussis (détection des
  fuites désactivée).
- GPU AMD : 207 cas OpenCL/Rusticl et 207 Vulkan/RADV, sorties et gradients
  comparés au nouveau CPU avec la tolérance native des tests GPU.
- Référence CPU/JAX du worker TPU : 133 cas et 74 types, protocole compris.
  Cette preuve n'est pas une exécution sur TPU physique.
- Compilation HIP pour gfx90c ; aucune exécution ROCm. CUDA n'a pas été compilé
  ni exécuté faute de toolchain/périphérique NVIDIA.

## Reproduire

Dans un build configuré avec les tests activés :

```sh
cmake --build build --target RuntimeTest.AttentionBackward RuntimeTest.AttentionLayers \
  mimir_bench_layer_dimensions mimir_bench_whole_model -j2
MIMIR_EXECUTABLE="$PWD/build/Tests/mimir_bench_layer_dimensions" \
  MIMIR_DISABLE_ROCM=1 OMP_NUM_THREADS=1 \
  ./run_mimir.sh 64 2048 3 1 SelfAttention
MIMIR_EXECUTABLE="$PWD/build/Tests/mimir_bench_whole_model" \
  MIMIR_DISABLE_CUDA=1 MIMIR_DISABLE_ROCM=1 \
  MIMIR_DISABLE_VULKAN=1 MIMIR_DISABLE_OPENCL=1 \
  MIMIR_RUNTIME_AUTOTUNE=0 OMP_NUM_THREADS=1 \
  ./run_mimir.sh float32 3 2 64 1024 1 SelfAttention
```

Le dernier argument du benchmark Model choisit désormais `Linear` (défaut
conservé) ou `SelfAttention`. `MIMIR_LAYER_VALUES` et `MIMIR_BENCH_VALUES`
activent les dumps binaires correspondants. Comparer avec le même harnais et
la même configuration dans un build de référence ; les empreintes des sources
avant cette continuation figurent dans le JSON.
