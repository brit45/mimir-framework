# Planificateur d’exécution

Le planner prépare le graphe, puis choisit les chemins d’exécution selon les
capacités natives et les latences mesurées. La construction du graphe, le
placement des opérations et la sélection d’un kernel fusionné sont des étapes
distinctes.

**Public concerné :** développeurs du moteur C++ et utilisateurs qui analysent
les performances d’un modèle.

Sources de vérité :

- `src/Planning/Planner.hpp` : graphe, durées de vie, groupes de fusion, placement et scratchpads ;
- `src/Planning/ExecutionCalibration.cpp` : classement des runtimes par opération ;
- `src/Planning/FusionCalibration.cpp` : comparaison des chaînes ordinaires et fusionnées ;
- `src/Planning/ExecutionPolicy.hpp` et `.cpp` : choix CPU et budgets de protection ;
- `src/runtimes/RuntimeRouter.cpp` : dispatch et repli après refus ou échec ;
- `src/Model.cpp`, `src/Model.hpp` : intégration au modèle et cache `static_plan_`.

## Sur cette page

- [Lecture rapide](#lecture-rapide)
- [1) Durées de vie et buffers](#1-durées-de-vie-et-buffers)
- [2) Fusion de graphe et fusion de kernels](#2-fusion-de-graphe-et-fusion-de-kernels)
- [3) Calibration des runtimes](#3-calibration-des-runtimes)
- [4) Cache et exécution à chaud](#4-cache-et-exécution-à-chaud)
- [5) Configuration et lancement](#5-configuration-et-lancement)
- [Déboguer efficacement le planner](#déboguer-efficacement-le-planner)
- [Validation](#validation)

## Lecture rapide

1. Le premier forward construit le plan si son cache est absent ou invalide.
2. Le plan décrit les tenseurs, leurs utilisations, les groupes de fusion et les buffers.
3. Les fusions peuvent fonctionner en inférence **et en entraînement** ; les
   valeurs intermédiaires nécessaires au backward sont conservées.
4. Les runtimes natifs et les configurations CPU sont comparés sur des
   opérations rejouables, avec contrôle numérique avant sélection.
5. Une fusion de kernels n’est sélectionnée que si elle est bit à bit conforme
   à la chaîne ordinaire mesurée et suffisamment plus rapide.
6. Une décision en cache reste soumise aux changements de configuration,
   aux échecs d’exécution et à la recalibration périodique.

## 1) Durées de vie et buffers

`analyze_tensor_lifetimes()` fournit la vue historique par nom de tenseur.
Le plan global, construit par `build_execution_plan_global()`, représente aussi
les versions des valeurs : réécrire `x` ne signifie pas que toutes ses versions
ont simultanément plusieurs consommateurs.

Le plan contient notamment `layers`, `tensors`, `fusion_groups`, `transfers` et
`regions`, ainsi que les tableaux de compatibilité `skip_layer` et
`fuse_chain_next`. `plan_host_buffer_reuse()` prépare la réutilisation des
buffers ; leur emploi effectif reste soumis aux conditions du chemin runtime.
La présence d’un emplacement réutilisable dans le plan ne prouve donc pas que
chaque forward ou backward l’emploie.

`plan_conv2d_fastpath_scratch()` et `apply_global_scratch_plan()` préparent les
besoins de scratch Conv2d. Les choix CPU et les limites des kernels portables
sont centralisés dans `ExecutionPolicy`. Les budgets de protection évitent un
travail ou un staging excessif ; ils ne constituent pas des seuils de vitesse
qui favoriseraient automatiquement un accélérateur.

## 2) Fusion de graphe et fusion de kernels

### Regroupement logique

`FusionKind` couvre `CONV2D_RELU`, les activations, les transformations de forme,
`Split`/`Chunk` et `GENERIC_LAYER`. Les groupes sont composés à partir des
liaisons du graphe et de leurs consommateurs. Les extrémités de branches
historiques forment des frontières ; les chaînes indépendantes autour de ces
branches peuvent rester fusionnables.

Un consommateur marqué `skip_layer` est exécuté dans le parcours de son
producteur. Ce regroupement peut encore appeler plusieurs opérations runtime :
il ne prouve pas qu’un seul kernel device a été lancé.

L’appel bas niveau `build_execution_plan_static(layers, training)` conserve un
défaut conservateur, `allow_training_fusion=false`. Le plan global utilisé par
`Model` autorise les fusions d’entraînement lorsque les variables de fusion du
modèle le permettent.

### Chaînes pointwise natives

Les kernels de fusion prennent en charge les chaînes d’activations suivantes :

`Identity`, `ReLU`, `LeakyReLU`, `GELU`, `SiLU`, `Tanh`, `Sigmoid`, `Softplus`,
`Mish`, `HardSigmoid` et `HardSwish`.

Les opérations doivent être sans poids et unaires. Les réductions telles que
`Softmax`, les changements de forme et les opérations stochastiques ne font pas
partie de ce programme pointwise ; elles peuvent conserver leur regroupement
logique ou leur dispatch ordinaire.

| Backend | Exécution de la fusion |
| --- | --- |
| CPU | Traitement par tuiles SIMD ; deux activations adjacentes sont abaissées dans une même boucle, en forward et backward. |
| Vulkan | Un dispatch compute parcourt toute la chaîne, à partir du SPIR-V généré. |
| OpenCL | Un lancement de kernel parcourt toute la chaîne, à partir du source généré. |
| CUDA | Kernel natif partagé, compilé par le compilateur CUDA. |
| ROCm | Même source de kernel, compilée en HIP. |

Avec `retain=true`, chaque sortie intermédiaire est restituée. Le backward
consomme les snapshots d’entrée de chaque étape et propage le gradient dans
l’ordre inverse. En inférence, les noms intermédiaires observables restent
préservés ; lorsque les sorties utilisent le même nom et que les conditions du
modèle le permettent, seule la sortie finale est restituée.

Ces programmes utilisent les hooks `queryPointwiseFusionCapability()`,
`pointwiseFusionForward()` et `pointwiseFusionBackward()` d’`AbstractRuntime`.
Les détails device et les limites de validation figurent dans
[Runtimes GPU](21-GPU-Runtimes.md#fusion-native-des-chaînes-pointwise).

### Sélection d’une fusion

`FusionCalibration` compare la chaîne ordinaire routée aux candidats natifs :

1. Une exécution de préparation précède cinq mesures ; la médiane est retenue.
2. Les résultats doivent être finis, de même forme et **bit à bit identiques**
   à ceux de la chaîne ordinaire de référence.
3. Le meilleur candidat fusionné doit satisfaire
   `fused_seconds < unfused_seconds * 0.98`.
4. Si la chaîne ordinaire est plus rapide, si la parité échoue ou si la mesure
   est refusée, le parcours ordinaire reste disponible.

La référence de fusion est la chaîne routée, qui peut elle-même sélectionner
plusieurs backends. Une comparaison numérique tolérante avec le CPU dans un
test GPU direct ne remplace pas ce contrôle exact de sélection.

## 3) Calibration des runtimes

`ExecutionCalibration::rank()` classe les runtimes dont la capacité pour
l’opération réelle est `Native` ou `NativeOptimized`. `HostFallback` ne prouve
pas une exécution GPU et n’est pas un candidat natif.

La comparaison peut porter sur plusieurs backends, mais aussi sur plusieurs
configurations d’un seul CPU : nombres de workers OpenMP et, lorsque disponible,
algorithme de convolution pointwise. Un candidat unique sans choix d’exécution
supplémentaire n’est pas rejoué pour une mesure inutile. Sans choix mesuré, le
chemin CPU de référence est sériel ; le scope CPU sélectionné ne fuit pas vers
l’opération suivante et les équipes OpenMP imbriquées sont évitées.

Chaque probe reçoit une copie jetable du layer pour isoler les gradients du
modèle. Une préparation précède cinq mesures. La durée mesurée inclut le calcul
et les transferts/synchronisations nécessaires au retour hôte. Les choix CPU
sont comparés exactement ; les autres backends utilisent le contrôle numérique
prévu par cette calibration. NMS conserve une comparaison exacte de ses indices.
Les opérations aléatoires ou à état non rejouable ne sont pas mesurées de cette
manière.

Le budget provient de la mémoire disponible et de `MemoryGuard`, éventuellement
plafonné par `MIMIR_PLANNER_CALIBRATION_MAX_BYTES`. Il n’existe plus de plafond
fixe implicite de 64 Mio. Si la mémoire disponible ne peut pas être établie,
un budget explicite ou un `MemoryGuard` est nécessaire. Une mesure refusée
donne priorité à la référence CPU sérielle si elle est éligible ; ce choix
n’est pas présenté comme un résultat de performance mesuré.

La calibration runtime est indépendante du mode du graphe :
`MIMIR_ENABLE_PLANNER=0`, `MIMIR_PLANNER_MODE=legacy` ou
`MIMIR_PLANNER_COST_MODEL=0` ne désactivent pas les mesures de dispatch.
`MIMIR_RUNTIME_AUTOTUNE=0` est le réglage prévu pour les désactiver.

## 4) Cache et exécution à chaud

Les clés tiennent compte des formes, modes forward/backward et
entraînement/inférence, dtype, paramètres pertinents de l’opération,
génération de routage, runtimes/configurations disponibles et contexte OpenMP.
Les clés de fusion incluent aussi la rétention des intermédiaires et les
configurations de workers candidates. Le cache de calibration est local au
thread, borné à 2 048 entrées, avec un intervalle de rafraîchissement de
1 024 utilisations par décision.

Sur un hit de fusion, `FusionCalibration::lookup()` utilise les métadonnées des
candidats sans construire leurs callbacks de mesure. Les diagnostics inchangés
ne sont pas recopiés à chaque hit. Les routes classées sont utilisées directement
pendant le dispatch et le contrôle des entrées finies emploie le chemin SIMD
lorsqu’il est disponible.

Un échec de kernel invalide sa décision. La recalibration ou l’invalidation
rafraîchit également les diagnostics des autres layers partageant la même clé.
Vider les profils d’affichage ne supprime pas les décisions réutilisables : les
profils sont republiés au prochain hit.

Ces optimisations réduisent le coût des appels déjà calibrés. Elles ne
suppriment pas le coût du premier appel ni ne garantissent un gain global sur
tous les modèles, tailles ou backends.

## 5) Configuration et lancement

Le moteur choisit `cost` si `MIMIR_PLANNER_MODE` est absent et si le modèle de
coût est actif. `run_mimir.sh` fournit par défaut `static`, active la fusion et
la réutilisation de buffers, laisse l’autotuning actif et fixe
`MIMIR_PLANNER_COST_MODEL=0`. Il faut donc conserver le lanceur et son
environnement lors des comparaisons.

`MIMIR_PLANNER_FUSION` est le nom prioritaire ; `MIMIR_ENABLE_FUSION` reste son
alias de compatibilité. `MIMIR_ENABLE_FUSION_TRAIN=0` désactive la fusion pendant
l’entraînement. Les valeurs détaillées du moteur et du lanceur figurent dans
[Variables d’environnement](../03-API-Reference/22-Environment-Variables.md#4-planner-et-fusion).

## Déboguer efficacement le planner

- Utiliser `MIMIR_ACCEL_VERBOSE=1` pour les décisions et
  `MIMIR_RUNTIME_TRACE=1` pour le backend et le chemin d’exécution réels.
- Lire `graph_fused` et `kernel_fused` séparément. Dans le snapshot Viz,
  `fusion_forward_decision` et `fusion_backward_decision` exposent le runtime,
  la raison, les alternatives mesurées et leurs contrôles `bit_exact`.
- Distinguer `cost_basis=measured_host_roundtrip` d’un coût estimé.
- Comparer fusion active/inactive avec `MIMIR_PLANNER_FUSION`, en gardant
  `MIMIR_RUNTIME_AUTOTUNE` et les autres réglages identiques.
- Utiliser `MIMIR_PLANNER_STDOUT=1` pour afficher le dump dans le terminal ;
  `MIMIR_PLANNER_JSON` désigne le fichier JSON écrit lors d’un dump.

## Validation

Les commandes suivantes exécutent les tests ciblés dans un build existant
configuré avec `MIMIR_ENABLE_TESTS=ON` et `BUILD_TESTING=ON` :

```bash
cmake --build build --target RuntimeTest.Fusion PlannerTest.FusionCalibration FrameworkTest.ExecutionCalibration PlannerTest.ExecutionPolicy -j4
ctest --test-dir build --output-on-failure -R '^(RuntimeTest.Fusion|RuntimeTest.VAEConvTrainingFusion|PlannerTest.FusionCalibration|FrameworkTest.ExecutionCalibration|PlannerTest.ExecutionPolicy)$'
```

Ces tests passent dans le checkout validé le 28 octobre 2026. Ils couvrent les
121 couples d’activations CPU, les chaînes de longueur 1 à 7, les sorties et
gradients, les branches VAEConv, l’isolation des probes, la parité stricte,
les clés de cache, les profils partagés et la recalibration périodique.
Les validations GPU sont détaillées dans [Runtimes GPU](21-GPU-Runtimes.md#validation-de-la-fusion-au-28-octobre-2026).

## Étapes suivantes

- [Runtimes GPU](21-GPU-Runtimes.md)
- [Développer un runtime](../07-Devs/04-Runtime-Development.md)
- [Index de la documentation](../00-INDEX.md)
