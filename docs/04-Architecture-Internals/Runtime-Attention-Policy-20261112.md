# Sélection mesurée de l'attention — 12 novembre 2026

Cette suite active la calibration de SelfAttention et CrossAttention dans le
routeur utilisé par Model. Elle s'appuie sur le moteur déjà optimisé par
[l'attention par blocs](Runtime-Attention-Tiling-20261112.md) : les gains
ci-dessous sont supplémentaires. Elle accélère aussi les contrôles NaN/Inf
sur les buffers hôte, sans changer les formules ni les APIs publiques.

Le [rapport JSON](Runtime-Attention-Policy-20261112.json) conserve les dix
profils, les étapes forward/backward/optimiseur, les allocations C++, les
choix réellement publiés, les empreintes binaires et les validations. Les
journaux et temps bruts restent dans
`outputs/benchmarks/attention-policy-20261112/final/`.

## Routage et validation

Les deux attentions natives actuelles sont déterministes, sans dropout ni cache
KV persistant. Elles peuvent donc comparer les workers CPU et les backends
natifs disponibles. Les clés incluent maintenant num_heads, num_kv_heads,
head_dim, causal et rope_theta, en plus des formes et modes existants. Un
changement de géométrie provoque une nouvelle décision.

Chaque candidat conserve une préparation puis cinq mesures et le cache garde
son rafraîchissement à 1024 usages. Les variantes CPU doivent conserver les
sorties et gradients de paramètres exactement ; les GPU conservent la
tolérance existante `4e-4*(1+abs(reference))`. Les opérations stochastiques
restent exclues. Un refus du budget conserve la référence CPU sérielle.

Les copies jetables du layer isolent les gradients des sondes et de l'exécution
calibrée. Une écriture partielle ou non finie d'un backend ne doit jamais être
validée dans le modèle. Ces copies restent nécessaires ici ; elles augmentent
les allocations par batch. Leur préparation se trouve hors de la fenêtre des
échantillons de calibration, mais dans les mesures complètes de trainStep et
de démarrage ci-dessous. Aucun cache persistant de snapshots n'est ajouté.

`FiniteCheckDispatch.cpp` choisit au runtime les classifications SSE2, AVX2 ou
AVX512F selon CPUID et les registres OS validés par RuntimeSimd ; la demande AVX
utilise SSE2 pour ce contrôle entier. Aucun flag global `-mavx2` n'est requis.
Les masques d'exposant IEEE refusent NaN/Inf, y compris les NaN signalants, sans
modifier les valeurs. Les petites vues et les builds sans SIMD gardent une voie
scalaire ; les formats non IEEE utilisent std::isfinite.

## Model::trainStep après chauffe

Sonde native : une paire SelfAttention/SiLU, huit têtes, float32, MSE, AdamW,
sans biais. Dix étapes de chauffe, puis trois à cinq répétitions pour les
formes ci-dessous. Largeur désigne la dimension des embeddings, lignes le
nombre de tokens du tenseur. Le trainStep inclut forward, loss, backward,
optimiseur et orchestration Model.

CPU Ryzen 5 5500U, AVX2, GCC 15.2, quatre threads OpenMP autorisés. Ce nombre
est un budget de candidats ; les profils du JSON donnent le choix effectif.
Le lanceur fixe OMP_PROC_BIND=false et OMP_PLACES=cores, et remplace donc les
places demandées par le script. Aucune affinité stricte ni fréquence fixe n'est
revendiquée. Un autre entraînement était actif sur la machine.

| Largeur × lignes | trainStep avant → après | Gain | Inférence avant → après |
|---|---:|---:|---:|
| 64 × 512 | 97.980 → 29.777 ms | ×3.29 | 26.519 → 11.930 ms |
| 64 × 1024 | 362.798 → 156.086 ms | ×2.32 | 129.575 → 38.784 ms |
| 512 × 64 | 47.817 → 23.504 ms | ×2.03 | 9.594 → 11.000 ms |
| 1024 × 64 | 228.306 → 126.942 ms | ×1.80 | 40.112 → 30.957 ms |
| 2048 × 64 | 769.720 → 483.060 ms | ×1.59 | 175.674 → 83.430 ms |

Les cinq formes ci-dessus et les cinq contrôles supplémentaires conservent
les prédictions et poids finaux identiques bit à bit. Le contrôle Linear,
le mode à un seul worker et le mode sans calibration figurent dans le JSON.
Leurs variations importantes montrent l'effet de la charge partagée : les
valeurs sont des mesures locales, sans promesse d'un facteur universel.

Étapes chronométrées séparément, en millisecondes :

| Largeur × lignes | Forward avant → après | Backward avant → après | Optimiseur avant → après |
|---|---:|---:|---:|
| 64 × 512 | 31.924 → 11.934 | 48.609 → 19.724 | 0.013 → 0.012 |
| 64 × 1024 | 127.032 → 66.052 | 197.871 → 104.360 | 0.009 → 6.003 |
| 512 × 64 | 9.859 → 4.158 | 28.361 → 16.495 | 1.825 → 1.087 |
| 1024 × 64 | 40.764 → 35.353 | 179.956 → 132.585 | 4.641 → 4.893 |
| 2048 × 64 | 212.728 → 86.583 | 617.638 → 303.114 | 20.031 → 19.431 |

Ces étapes sont des séries distinctes ; leur somme n'est pas la mesure de
trainStep. Les coûts de l'optimiseur restent soumis à sa propre calibration.

## Coût de la première utilisation

| Largeur × lignes | Premier trainStep avant → après (ms) | Première inférence avant → après (ms) |
|---|---:|---:|
| 64 × 512 | 91.3 → 1542.6 | 60.7 → 366.2 |
| 64 × 1024 | 397.3 → 4095.1 | 146.5 → 1507.5 |
| 512 × 64 | 473.7 → 652.1 | 58.8 → 477.9 |
| 1024 × 64 | 1135.4 → 3445.8 | 68.1 → 620.3 |
| 2048 × 64 | 3143.3 → 13477.9 | 357.8 → 4308.9 |

Le coût initial augmente car les variantes sont réellement exécutées et
contrôlées. En dimension 2048, le premier trainStep passe de 3,14 à 13,48 s,
puis la médiane après chauffe de 770 à 483 ms. Le cache amortit ce coût sur les
appels suivants. L'inférence et l'entraînement ont des signatures distinctes.
`MIMIR_RUNTIME_AUTOTUNE=0` conserve le contrôle déjà disponible pour désactiver
la calibration ; les formules du calcul restent les mêmes.

Les petits batches ne gagnent pas systématiquement. Trois comparaisons de
200 répétitions donnent, pour largeur 64 et une ligne, respectivement
0,1202 → 0,1111 ; 0,1262 → 0,1252 ; 0,1068 → 0,1077 ms. Avec seize lignes :
0,3863 → 0,4714 ; 0,5447 → 0,4584 ; 0,3698 → 0,4815 ms. L'ordre avant/après
est inversé à la deuxième comparaison. Les empreintes restent identiques.
Le surcoût peut atteindre environ 0,11 ms sur ce cas court. La copie de
protection ajoute notamment environ 64 Kio de demandes new par appel à largeur
64, et environ 64 Mio à largeur 2048. Ces compteurs ne sont ni le RSS ni les
allocations de tous les threads. Cette suite vise surtout les calculs plus
longs et ne supprime pas ce coût de protection.

## Contrôles NaN/Inf

Sonde isolée en compilation portable, onze séries alternées par variante,
mutation finie d'un élément entre les appels pour empêcher l'élimination du
calcul. L'ancienne fonction reste scalaire dans ce build sans -mavx2. Le
nouveau dispatch utilise AVX2. Médianes par appel :

| Nombre de floats | Avant | Après |
|---:|---:|---:|
| 16 | 0.030 µs | 0.024 µs |
| 64 | 0.075 µs | 0.023 µs |
| 16384 | 15.882 µs | 1.564 µs |
| 1048576 | 1064.190 µs | 206.118 µs |

Ce gain concerne seulement la classification, pas une accélération équivalente
de l'entraînement complet. Le source et le JSON de cette sonde sont conservés
avec les journaux locaux.

## Validation de cette continuation

- CTest RuntimeTest.FiniteDispatch : signes, subnormaux, exposants, infinis,
  NaN calmes/signalants, vues désalignées, fins de vecteurs et sentinelles
  invalides hors de la vue. Scalar, SSE2, AVX et AVX2 exécutés sur cet hôte ;
  AVX512F compilé mais non exécuté faute de support matériel.
- Même test sans SIMD, sous Clang portable et sous ASan/UBSan sans OpenMP ;
  détection des fuites désactivée pour le contrôle sanitizer.
- AttentionCalibration : rejet d'un écart d'un ULP, accumulation une seule
  fois, cache, causalité/têtes, budget nul, exclusion stochastique, repli après
  écriture partielle et après gradient NaN, portée des workers restaurée.
- Quatre attentions CPU réelles : Self/Cross, causal ou non, longueurs 17/20.
- GPU AMD : quatre cas OpenCL/Rusticl et quatre Vulkan/RADV. Les sondes GPU
  forward et backward sont exécutées et validées ; le CPU gagne la décision
  finale sur ces petites formes. Aucun gain de vitesse GPU n'est revendiqué.
- ExecutionCalibration, ExecutionPolicy et Fusion réussis ; la fusion couvre
  notamment les entrées non finies, F32/F16, des branches et VAEConv.
- Application bin/mimir reconstruite ; git diff --check réussi.

CUDA n'a été ni compilé ni exécuté dans cette continuation. ROCm reste désactivé
à l'exécution. Aucun test de TPU physique n'a été effectué. Les validations HIP
et CPU/JAX précédentes restent décrites dans le rapport d'attention par blocs.
Les limites historiques des gradients de biais d'attention y restent également
documentées ; cette modification de routage ne les corrige pas. Ces sondes ne
remplacent pas un entraînement sur dataset réel ni une validation de Viz.

## Reproduire

Dans un build configuré avec les tests activés :

```sh
cmake --build build --target FrameworkTest.AttentionCalibration RuntimeTest.FiniteDispatch \
  mimir_bench_whole_model -j2
ctest --test-dir build --output-on-failure \
  -R '^(FrameworkTest.AttentionCalibration|RuntimeTest.FiniteDispatch)$'
MIMIR_EXECUTABLE="$PWD/build/Tests/mimir_bench_whole_model" \
  MIMIR_DISABLE_CUDA=1 MIMIR_DISABLE_ROCM=1 \
  MIMIR_DISABLE_VULKAN=1 MIMIR_DISABLE_OPENCL=1 \
  OMP_NUM_THREADS=4 MIMIR_RUNTIME_AUTOTUNE=1 \
  ./run_mimir.sh float32 5 2 64 1024 1 SelfAttention
ctest --test-dir build --output-on-failure \
  -R '^RuntimeTest.(OPENCL|VULKAN)AttentionCalibration$'
```

Les tests GPU sont enregistrés uniquement pour les backends compilés et rendent
77 si le matériel manque. Le JSON du benchmark distingue maintenant première
inférence, médianes après chauffe, activation de l'autotune et profils de workers.
Comparer deux builds avec ce même harnais et les mêmes nombres d'étapes ;
`MIMIR_BENCH_VALUES` active le dump des prédictions et poids finaux.
