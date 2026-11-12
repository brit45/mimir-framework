# Benchmarks Colab : CPU, CUDA, Vulkan, OpenCL et TPU

Le [notebook de comparaison](../../Colab-bench-multi-framework.ipynb) contient
un microbenchmark multi-framework et un benchmark MNIST complet séparé.
Le [notebook de validation](../../Colab-test.ipynb) compile et vérifie les
runtimes, sans produire le même classement. Les deux embarquent les sources
locales, y compris les nouveaux headers et tests, avec un manifeste SHA256.

## Démarrer une session cohérente

1. Charger la version actuelle du notebook dans Colab et exécuter les cellules
   depuis la configuration, l’extraction des sources et la compilation.
2. Choisir le matériel et les runtimes Mímir dans la configuration. Sur une VM
   GPU, `GPU_CUDA`, `GPU_VULKAN` et `GPU_OPENCL` sont indépendants ; les concurrents
   PyTorch, TensorFlow et JAX utilisent CUDA pour cette catégorie de matériel.
3. Exécuter la validation avant les mesures. Un SDK présent ne prouve pas qu’un
   pilote ou un device est utilisable. Un runtime indisponible est consigné ;
   `GPU_REQUIRE_ALL=True` rend son absence bloquante.
4. Conserver le dossier de résultats : manifeste, fixtures, métriques brutes,
   versions, erreurs, CSV, HTML et archive exportée.

`SOURCE_MODE="embedded"` extrait l’instantané dans un dossier dédié. Le mode
`checkout` est une option explicite ; il peut pointer sur une branche distante
plus ancienne que les sources embarquées. Le générateur de maintenance est
`scripts/colab/embed_sources.py`. Il synchronise les archives et le helper du
microbenchmark, sans régénérer les kernels du core.

## Choisir le profil du microbenchmark

| Profil | Cas | Usage |
| --- | ---: | --- |
| `portable` | 7 | Formes bornées pour les kernels de référence Vulkan/OpenCL ; choix initial si l’un de ces runtimes est demandé sur GPU. |
| `quick` | 7 | Premier contrôle des chemins CPU/CUDA/TPU et des nouvelles opérations. |
| `standard` | 11 | Matrices plus grandes et attention à 2 048 tokens ; les formes peuvent être refusées par un runtime. |

Les sept familles de base sont Linear, Conv2d, LayerNorm, Linear+SGD,
BatchMatMul forward, BatchMatMul backward et attention causale. Standard ajoute
deux anciens cas larges, un BatchMatMul large et une attention longue.
Les dimensions d’un profil sont identiques pour tous les concurrents : le
notebook ne réduit pas une forme uniquement pour améliorer le score d’un runtime.

Le backward matriciel mesure uniquement dA et dB à partir de A, B et du gradient
de sortie fourni. Tous les adaptateurs calculent ces deux produits, sans forward
ni mise à jour des poids dans ce cas. Linear+SGD suit un autre contrat, avec
forward, perte, backward et mise à jour vérifiés séparément.

## Identités des runtimes et classement

Sur GPU, chaque runtime Mímir activé et validé produit son concurrent :
`mimir-cuda`, `mimir-vulkan`, `mimir-opencl`. Les autres frameworks ne sont pas
dupliqués. Le résultat est accepté seulement si le runtime demandé correspond
au champ retourné et à la sélection native du dispatcher. Un repli CPU n’est
pas une mesure Vulkan/OpenCL valide.

Les kernels portables conservent leurs limites de taille et de travail.
Un gros cas refusé reste incomplet ; il n’est ni remplacé par un plus petit cas
ni compté comme une latence nulle. La note exige une couverture complète et au
moins deux concurrents complets. Elle utilise la moyenne géométrique des ratios
entre meilleure médiane et médiane du concurrent, avec le même poids par cas.

Le protocole microbenchmark est `mimir-colab-micro-v3`. Comparer les notes entre
runs du même protocole, profil, matériel, précision et frontière de mesure.
Le score mesure ces opérations ; il n’évalue pas la qualité générale d’un framework.

## Temps, mémoire et provenance

Les microbenchmarks utilisent FP32 et incluent les entrées/poids depuis l’hôte,
les résultats matérialisés sur l’hôte et leur synchronisation. Imports et
initialisation sont distingués du chemin chaud ; chauffe et essais sont explicites.
Médiane, p95, IQR, débit et erreurs numériques sont conservés.

Le RSS du microbenchmark est obtenu par `/usr/bin/time -f %M`, pour éviter un
pic hérité du processus notebook. Il inclut les imports et allocations hôte du
processus mesuré. Il ne mesure pas la VRAM/HBM et ne participe pas à la note.
La mesure remontée par le worker est aussi conservée pour diagnostic.

Le benchmark MNIST utilise les API de modèle complètes, des poids persistants,
une référence indépendante de la première étape, un jeu de test séparé et un
seuil de qualité. Ses limites de temps et RSS+swap concernent l’arbre des processus
et sont contrôlées par `scripts/benchmarks/colab_real_suite.py`. Les métriques,
le score et les profils MNIST restent séparés de ceux des microbenchmarks.

## Éviter les cellules et sources décalées

Une trace contenant `spec_from_file_location('mimir_colab_micro_suite', ...)`
puis un fichier absent sous `/content/mimir-source-...` indique une ancienne
cellule de chargement. La cellule actuelle embarque `MICRO_SUITE_SOURCE`, écrit
le helper dans le dossier de résultats et le charge sans dépendre d’un fichier
dans REPO. Elle fonctionne même avec un ancien checkout. Cela ne met pas à jour
le binaire : après changement des sources ou du pilote, reprendre l’extraction,
la compilation, puis la validation.

### Shader Vulkan manquant après les tests de pooling

Un échec `GPU forward failed` immédiatement après `GlobalAvgPool2d` correspond
au premier cas `Conv2d`. Les anciens instantanés embarqués construisaient les
clients du benchmark sans leur cible `mimir_shaders` ; le chemin historique ne
cherchait pas non plus dans `build-colab-cuda/shaders`. Les opérations précédentes
utilisent les shaders embarqués de `PortableRuntime` et pouvaient donc passer.

La cellule de validation prépare maintenant les shaders externes et les copie
dans `REPO/bin/shaders` pour les binaires issus de ces anciens instantanés. Pour
réparer une session déjà compilée, exécuter avant de reprendre cette cellule :

```python
run(['cmake','--build',BUILD,'-j',JOBS,'--target','mimir_shaders'],timeout=600)
run(['cmake','-E','copy_directory',BUILD/'shaders',REPO/'bin/shaders'])
RUNTIME_ENV['MIMIR_VULKAN_SHADER_DIR']=str(BUILD/'shaders')
```

Le moteur actuel génère ces fichiers avec les clients natifs et les retrouve
depuis l'exécutable sous Linux. Les nouveaux tests indiquent aussi la couche et
sa géométrie en cas d'échec ; le notebook active `MIMIR_ACCEL_VERBOSE=1` pour la
validation portable. Les échecs numériques ou de forward restent bloquants ;
seule une indisponibilité explicite du runtime peut donner le code `77`.

### Assertion NMS backward must select GPU sous CUDA

Si les calculs CUDA passent puis que le test échoue sur
`NMS backward must select GPU`, vérifier la politique de placement du test.
La calibration automatique peut retenir le CPU pour ce stop-gradient très
court, alors que le test de parité exige explicitement le backend CUDA.
Le mode `MIMIR_PLANNER_MODE=static` seul ne désactive pas cette calibration.

Les tests natifs CUDA, Vulkan et OpenCL utilisent maintenant la même
préparation avec `MIMIR_RUNTIME_AUTOTUNE=0`. Le notebook applique aussi ce
réglage dans le seul environnement de validation CUDA, pour les binaires déjà
compilés. Les calculs, tolérances et assertions de sélection GPU sont conservés ;
l'environnement des mesures garde ses réglages de calibration.

Dans une session existante, remplacer la construction de l'environnement
dans la cellule de validation puis relancer cette cellule :

```python
cuda_validation_env = dict(
    REFERENCE_ENV, MIMIR_DISABLE_CUDA='0',
    MIMIR_RUNTIME_AUTOTUNE='0', MIMIR_PLANNER_MODE='static')
```

Le test de routage utilise des runtimes simulés pour reproduire le choix du CPU
avec calibration, puis le choix CUDA forcé avec le même résultat stop-gradient. Une exécution CUDA
sur le GPU NVIDIA de la session reste nécessaire pour confirmer le passage
complet du validateur matériel.

### Blocage après la dernière ligne CrossAttention

`GPU_NATIVE_PARITY_OK CrossAttention` ne marque pas la fin du processus. Le
test vérifie ensuite l'attention avec des dimensions K/V différentes, le
filtrage NMS des valeurs non finies, la reconfiguration et la fermeture du
runtime. Les nouveaux repères `GPU_NATIVE_STAGE` sont vidés immédiatement
vers le journal avant chacune de ces étapes. `GPU_NATIVE_PARITY_COMPLETE`
n'est écrit qu'après la fermeture et le contrôle de libération du staging.

Le cache de staging ne détient plus le périphérique jusqu'à la destruction
du stockage local au thread : les runtimes et calculs actifs gèrent sa durée
de vie. Cette correction évite d'appeler le pilote depuis une destruction
de cache trop tardive. Elle est validée localement sur AMD ; la résolution
du blocage observé sur NVIDIA doit être confirmée dans la session concernée.

Interrompre la cellule bloquée, charger le notebook mis à jour, puis reprendre
l'extraction des sources, la compilation et la validation. Une session dont
`REPO` pointe encore sur `mimir-source-a2e029f80298b42a` conserve l'ancien moteur.
La seule relance de sa cellule de validation n'applique pas la correction.
Les instantanés sont régénérés sans les exécutables des tests ou les produits
CMake et sans effacer les sorties locales des notebooks.

Après compilation des nouvelles sources, le contrôle final peut être isolé :

```python
probe_env = dict(REFERENCE_ENV, MIMIR_DISABLE_VULKAN='0', MIMIR_ACCEL_VERBOSE='1')
run([BUILD/'Tests/PortableNativeOps', 'VULKAN', '--final-checks'],
    cwd=REPO, env=probe_env, timeout=180)
```

Le lanceur du benchmark affiche un état après 15 secondes sans sortie.
La parité portable a un délai maximal de 180 secondes ; un dépassement
interrompt le processus et rapporte la dernière étape, sans classer le runtime
comme simplement indisponible ni autoriser les mesures GPU.
L'attente après `SIGKILL` est elle aussi bornée à cinq secondes, y compris lors
d'une interruption manuelle. Si le pilote retient encore le processus, le
lanceur signale qu'il faut redémarrer l'environnement avant de reprendre.

### Reprendre après une alerte de provenance de la matrice

Une ancienne cellule enregistrait `colab_config_matrix.py` dans
`provenance['sources']`, puis comparait cet inventaire aux seuls fichiers
`colab_real_*` et `colab_architecture_*`. L'assertion pouvait donc échouer
après les essais sans modification réelle des sources. Le contrôle actuel
inclut aussi les helpers enregistrés par les cellules suivantes et indique
les fichiers manquants, ajoutés ou modifiés.

Si les essais sont terminés et que `REAL_RECORDS` est encore présent dans la
session, vérifier les empreintes dans une cellule séparée, sans relancer
l'entraînement :

```python
paths = {p.name: p for pattern in ('colab_real_*', 'colab_architecture_*')
         for p in real_source.glob(pattern) if p.is_file()}
paths.update({name: real_source / name for name in provenance['sources']})
actual = {name: REAL_SUITE.sha256(path) for name, path in paths.items() if path.is_file()}
assert provenance['sources'] == actual, 'Sources réellement modifiées'
assert REAL_DATA_MANIFEST['files'] == {
    p.name: REAL_SUITE.sha256(p) for p in REAL_DATA.iterdir()
    if p.suffix in {'.u8', '.f32'}
}, 'Données réellement modifiées'
```

Si ces contrôles passent, reprendre la cellule de comparaison d'usage réel
et d'export du rapport. Les résultats et les fichiers de `REAL_DIR` sont
conservés ; la vérification ne modifie pas les empreintes de référence.

## Validation effectuée et limites

Le 18 octobre 2026, les onze cas standard ont passé la référence NumPy sur CPU
dans Mímir, PyTorch, TensorFlow et JAX : 44 comparaisons numériques. Les sept
cas portables ont passé la même validation sur Vulkan/RADV et OpenCL/Rusticl
avec le GPU AMD Radeon Graphics : 14 combinaisons natives. Les cellules de mesure,
classement et export ont également été exécutées pour les deux runtimes séparés.
Les tests de parité du core passent 165 cas par runtime ; les 19 tests Python
Colab passent, dont le chargement du helper sans checkout récent.

Ces essais courts valident le fonctionnement et la précision, pas un classement
publiable. Ils ne prouvent pas l’exécution sur le GPU NVIDIA d’une session Colab,
ni sur TPU. Les traces et tests matériels de la session cible restent nécessaires.

Le 12 novembre 2026, la correction des ressources Vulkan passe les quatre CTest
ciblés (`PortableWorkBudget`, `OPENCLNativeOps`, `VULKANNativeOps`,
`VulkanShaderPaths`), avec 207 cas de parité par backend sur le GPU AMD local.
La compilation du seul client `PortableNativeOps` régénère bien un shader
périmé. Les dix tests Python ciblés de disponibilité GPU, d'environnement et de
configuration Vulkan NVIDIA passent ; ces derniers simulent la configuration
du pilote et ne constituent pas une exécution sur GPU NVIDIA.

Le 13 novembre 2026, la correction de fermeture passe quatorze CTest ciblés sur AMD :
parité native, produits matriciels et fusion pointwise Vulkan/OpenCL,
packing, budget de staging, optimiseurs et pertes CPU/Vulkan/OpenCL.
Les contrôles finaux atteignent la fermeture du
runtime et le marqueur de fin ; cette preuve locale ne vaut pas validation
de la fermeture sur le pilote NVIDIA de Colab.

## Vérifier les helpers sans accélérateur

Depuis la racine, avec NumPy et pandas dans l’interpréteur choisi :

```bash
python3 -m unittest discover -s scripts/tests -p 'test_colab_*.py'
```

Cette suite a été exécutée localement avec ces dépendances. Sans elles, certains
tests numériques sont ignorés ; un résultat avec skips n’équivaut pas à cette
validation complète. Les tests Python n’exécutent pas les GPU Colab.

- [Performance et limites des gains](01-Performance.md)
- [Runtimes GPU](../04-Architecture-Internals/21-GPU-Runtimes.md)
- [Index](../00-INDEX.md)
