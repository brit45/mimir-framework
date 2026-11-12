# `RuntimeAllocator` et scratchpads

Comprendre le fonctionnement interne exact des composants runtime.

**Public concerné :** Développeur avancé qui modifie le moteur C/C++.

> **Prérequis**
>
> Connaître les bases C++ et la structure du dépôt.

Cette page documente le gestionnaire mémoire runtime utilisé dans les hot-path (forward/backward) pour éviter les allocations sauvages et respecter les limites `MemoryGuard`.

Source de vérité :

- Runtime allocator : `src/runtimes/cpu/RuntimeAllocator.hpp`
- Memory guard : `src/MemoryGuard.hpp`
- Dynamic allocator : `src/runtimes/cpu/DynamicTensorAllocator.hpp` (autre stratégie)
- Utilisation dans Model : `src/Model.cpp` (création `RuntimeAllocator allocator(guard, cap_mb)`)

## 1) Objectif

- Les allocations effectuées via cet allocator sont **comptabilisées** ; des vecteurs de kernels et des bibliothèques externes peuvent allouer en dehors de ces compteurs.
- Réduire la fragmentation et les `new/delete` répétés via des scratchpads réutilisables.

## 2) Deux types de handles RAII

### A) `TensorHandle`

- encapsule un `std::vector<float>` + un `TensorDescriptor`.
- au destructeur : `MemoryGuard::releaseAllocation(size_bytes)`.

### B) `BufferHandle`

- encapsule un `float*` alloué avec `new[]`.
- au destructeur : `delete[]` + `releaseAllocation(size_bytes)`.

Invariant : ces handles sont non-copyable et movables.

## 3) Cap local vs limite globale

`RuntimeAllocator` prend :

- une référence vers `MemoryGuard` (limite globale),
- un `max_ram_mb` optionnel (cap local).

Le cap local **ne modifie pas** la limite globale : il sert à garder une marge de sécurité dans un codepath.

## 4) Scratchpad pool

- `get_scratchpad(min_bytes, tag)` :
  - réutilise un buffer existant (par `tag`) si sa taille suffit,
  - sinon alloue un nouveau buffer.

- `return_scratchpad(BufferHandle&&)` : remet un buffer dans le pool.

`allocate_buffer(bytes, tag)` initialise les octets demandés à zéro lors d’une
nouvelle allocation. `allocate_buffer_uninitialized(bytes, tag)` réserve sans
initialiser : le kernel doit écrire chaque élément avant toute lecture.
`get_scratchpad_uninitialized(min_bytes, tag)` applique le même contrat à une
réservation ou à un buffer réutilisé. Un scratchpad déjà présent dans le pool
n’est pas remis à zéro automatiquement, y compris avec `get_scratchpad`.

Les demandes d’octets non multiples de `sizeof(float)` disposent d’un stockage
physique arrondi à l’élément supérieur. Le handle et MemoryGuard comptabilisent
les octets demandés ; le remplissage d’alignement physique n’ajoute donc pas une
mesure de RSS à ces compteurs. Une exception d’allocation libère la réservation
MemoryGuard. Les déplacements des handles transfèrent la propriété.

⚠️ Attention : le pool retient de la mémoire (comptabilisée) tant qu’il vit. Le caller doit décider quand nettoyer via `clear_scratchpad_pool()`.

## 5) Buffers et arènes réutilisables du moteur CPU

`src/runtimes/cpu/ReusableScratch.hpp` fournit les espaces de travail des kernels :

- `FloatBuffer` loue un tableau brut de `float` via `RuntimeAllocator`, avec
  restitution RAII dans un pool propre au thread. Une location active ne peut
  pas être réattribuée, même lors d'un appel imbriqué.
- `FloatArena` loue un seul buffer pour plusieurs tableaux temporaires.
  Ses `FloatSpan` sont des vues disjointes ; seize floats séparent les régions
  pour éviter des alias de cache aux tailles puissances de deux. La place de
  ces séparateurs est comptabilisée dans MemoryGuard.
- `ConstFloatView` emprunte des poids en lecture seule. L'attention ne recopie
  donc plus les matrices Q/K/V et de projection avant de les lire.

Les kernels de convolution réutilisent leur im2col/dCol ; l'attention classique
réutilise ses projections, scores, contextes et gradients ; les RNN/GRU/LSTM
réutilisent leurs états et dérivées temporaires dans les chemins retenus. Les transpositions et facteurs
intermédiaires du backward natif commun aux backends passent aussi par ces
buffers. Concat/Stack écrivent dans le stockage de sortie disponible après
validation, avec une voie temporaire distincte si une entrée est cette sortie.
L'attention causale/GQA et le forward LSTM gardent
leurs temporaires précédents : les conversions mesurées ralentissaient les cas
contrôlés. La baisse du nombre d'allocations seule ne suffit pas à sélectionner
une implémentation plus rapide.

Le pool conserve au plus **32 buffers inactifs** et au plus
`min(64 MiB, limite MemoryGuard / 16)` **par thread**. Les locations actives
restent soumises à MemoryGuard, sans ce plafond de rétention. Un buffer plus
grand n'est pas conservé après usage. Les anciens buffers trop petits sont
libérés avant l'allocation d'un nouvel espace de travail. Les limites sont
réévaluées au prochain emprunt/retour ; `RuntimeScratch::Pool::local().clear()`
libère explicitement les réserves inactives du thread. Le compteur MemoryGuard
inclut cette mémoire conservée, qui n'est pas une fuite ni une mesure du RSS.

`FloatBuffer(count)` et `assign(count,value)` initialisent les éléments, même
sur un buffer recyclé. `resize()` préserve le préfixe et initialise à zéro la
nouvelle partie. `resizeUninitialized()` et `FloatArena::take()` exigent que
chaque élément soit écrit avant sa lecture. Les accumulateurs sont remis à
zéro aux mêmes étapes qu'avant, notamment à chaque tête ou pas récurrent.
Les dépassements de taille et les débordements arithmétiques sont refusés.

Ces espaces de travail ne doivent jamais s'échapper du kernel ni être gardés
par une tâche asynchrone. Les sorties, gradients persistants, activations
nécessaires au backward et interfaces publiques restent des conteneurs
propriétaires. Remplacer leur type sans adapter leur durée de vie créerait des
alias ou des références invalides. Un `std::vector` contigu n'a pas en soi un
accès lent : l'optimisation porte sur les allocations et copies répétées.
Les types numériques, les formules et l'ordre des réductions ne changent pas.

### Vérification

`RuntimeTest.ReusableScratch` vérifie réutilisation, initialisation, limites,
débordements, appels imbriqués, isolation des threads et Concat avec alias.
`RuntimeTest.ConvolutionScratch` vérifie les résultats et l'accumulation des
gradients, puis l'absence de nouvelle allocation de scratch après chauffe.
Les tests mathématiques et de dérivation existants couvrent les chemins de calcul.

Le target `RuntimeBench.Buffers` produit un CSV des allocations C++ et du temps
par appel de kernel, ainsi qu'un flux binaire des sorties et gradients :

```bash
cmake --build build --target RuntimeBench.Buffers
MIMIR_EXECUTABLE="$PWD/build/Tests/RuntimeBench.Buffers" OMP_NUM_THREADS=1 \
  MIMIR_RUNTIME_AUTOTUNE=0 MIMIR_DISABLE_ROCM=1 \
  ./run_mimir.sh /tmp/mimir-buffer-values.bin
```

Pour comparer deux révisions, construire chaque sonde avec les mêmes options,
conserver les deux flux binaires et les comparer octet par octet. Alterner
plusieurs exécutions avant/après au repos pour les timings. Cette sonde mesure
des kernels CPU ; elle n'établit ni une accélération globale de l'entraînement,
ni une baisse de consommation électrique, ni une validation GPU. Les allocations
C++ comptées ne couvrent pas tous les `malloc` des bibliothèques et ne sont pas le RSS.

### Contrôle local de la modification

La sonde avant/après a conservé **77 916 valeurs FP32 identiques bit à bit**
(sorties et gradients, formes de séquence 1, 7 et 13, convolutions avec stride
et dilation variables). Un entraînement natif Conv2d/AdamW de 30 pas, lancé
avec `run_mimir.sh`, a également conservé son flux de prédictions, pertes et
gradients identique octet par octet. Ces contrôles portent sur le CPU.

Exemples d'allocations C++ par appel après chauffe dans la sonde ; les sorties
propriétaires sont incluses dans ces nombres :

| Chemin et forme | Avant | Après |
|---|---:|---:|
| SelfAttention forward, 13 tokens, dimension 32, 4 têtes | 12 | 1 |
| CrossAttention forward, 13 requêtes, 16 clés, dimension 32 | 13 | 1 |
| CrossAttention backward, mêmes dimensions | 19 | 2 |
| RNN backward, 13 pas, entrée 16, état 24 | 17 | 1 |
| GRU backward, mêmes dimensions | 72 | 1 |
| LSTM backward, mêmes dimensions | 37 | 1 |
| Conv2d forward/backward, 7×9, 2→3 canaux, noyau 3 | 1 / 1 | 0 / 0 |

Les gains de temps varient selon le kernel et la forme. Les chemins mesurés
plus lents ont gardé leur implémentation précédente ; ces compteurs ne sont
pas une promesse d'accélération uniforme de tous les modèles.

## 6) `RUNTIME_CHECK` / mode strict

Le header définit :

- `MIMIR_STRICT_MODE` (par défaut `1`).
- `RUNTIME_ERROR_STRICT(msg)` : throw en strict, sinon log en permissif.
- `RUNTIME_CHECK(cond, msg)`.

Important : `MIMIR_STRICT_MODE` est une macro de compilation (`#define`), pas une variable d'environnement lue via `getenv`.

Ce mécanisme est utilisé dans le runtime forward/backward pour éviter des corruptions silencieuses.

## 7) Audit des allocations du moteur et mesures complètes

Le benchmark `scripts/benchmarks/benchmark_whole_model.cpp` mesure le chemin
public `Model::trainStep` (forward, loss, backward et optimiseur), puis
l'inférence et les étapes isolées. Il rapporte les allocations C++ du thread
appelant, leurs octets, les temps bruts, le démarrage à froid et le pic RSS.
Les allocations des workers et celles du pilote GPU ne sont pas incluses dans
le compteur C++ ; les buffers OpenCL disposent de leur propre compteur.

Les changements du moteur sont les suivants :

- Les snapshots flottants `TypedTensor` sont construits à la première lecture
  par `getTypedTensor`, avec le dtype enregistré au stockage. Les conversions
  entières restent immédiates. `getTensorMutable` fige le snapshot avant de
  permettre une modification du vecteur source ; les lectures const concurrentes
  du cache sont protégées. Le champ `typed_tensor_store` est désormais un cache :
  utiliser les accesseurs pour demander une représentation typée.
- Les routes de sélection des backends utilisent six emplacements internes,
  avec débordement dynamique pour les listes plus longues. L'ordre, la
  calibration et l'invalidation des candidats restent identiques.
- Les opérations natives OpenCL partagent quatre buffers extensibles protégés
  par mutex. Les entrées sont transférées à chaque appel, même si leur adresse
  est identique. La rétention au repos est limitée à 64 Mio ; un échec
  d'allocation évince le cache puis réessaie une fois. Les références des
  arguments déjà empruntés restent valides, et `cleanup` libère le cache.
  Les synchronisations nécessaires à la lecture des résultats sont conservées.
- Adam et AdamW peuvent sélectionner AVX2 à l'exécution dans une compilation
  portable x86-64 GCC/Clang. Les machines incompatibles gardent le chemin
  scalaire. Les opérations conservent leur ordre, sans FMA ni approximation
  de division ou de racine carrée ; les fins de tableaux restent scalaires.
- Le moniteur réutilise ses files de métriques et écrit les lignes CSV par
  lots contigus de même destination : une ouverture et un flush par lot.
  L'ordre, les anciens en-têtes et le drainage explicite restent préservés.

Campagne CPU du 11 novembre 2026 : cinq répétitions avant/après alternées,
100 appels par phase, un thread, launcher `run_mimir.sh`, GPU désactivés,
planner désactivé. Neuf configurations de **MLP Linear/SiLU** couvrent SGD,
Adam, AdamW, Lion, plusieurs tailles et les dtypes de stockage F32/F16/BF16.
Les calculs flottants de compatibilité existants sont conservés ; ces dtypes
ne signifient pas que tous les calculs sont natifs F16/BF16.

| Mesure | Avant | Après |
|---|---:|---:|
| MLP AdamW, entraînement complet, médiane | 0,448812 ms | 0,323788 ms |
| Même cas, allocations C++ par entraînement | 441 | 333 |
| Même cas, octets alloués par entraînement | 298 929 | 222 353 |
| Même cas, inférence, médiane | 0,087173 ms | 0,084980 ms |
| Même cas, allocations C++ par inférence | 153 | 77 |
| Même cas, optimiseur isolé, médiane | 0,142357 ms | 0,021110 ms |

Sur ces neuf configurations, le temps d'entraînement diminue d'environ
1 à 28 % et le temps d'inférence de 2 à 11 %. Les sorties et poids finaux
comparés sont identiques bit à bit. Une comparaison Lua Conv2d/AdamW sur
30 étapes confirme également l'identité des prédictions, pertes et gradients.
Cela ne constitue pas une mesure de toutes les architectures ni des datasets.

Sur OpenCL AMD Radeon Graphics (radeonsi, renoir), la sonde de six primitives
sur plusieurs formes vérifie zéro nouvelle allocation de buffer après chauffe
et 33 516 octets de résultats avant/après identiques. Cinq répétitions alternées
montrent des variations de temps de -22 % à +10 % suivant le cas ; la réduction
des allocations ne garantit pas à elle seule une accélération de chaque kernel.
Le test d'injection d'échecs couvre éviction, réessai, arguments vivants,
sortie inchangée lors d'un échec et nettoyage.

Les validations incluent 15 tests CPU ciblés et, séparément, 207 comparaisons
natives OpenCL et 207 Vulkan. CUDA et ROCm ne sont pas mesurés ici ; ROCm est
resté désactivé. Les mécanismes d'éviction, de spill et de suivi de mémoire
restent actifs. Les données brutes, dumps binaires et synthèses de cette
campagne sont conservés localement dans
`outputs/benchmarks/engine-stack-20261111-194320/`.

## 8) Latence des grandes projections

La campagne du 12 novembre 2026 couvre les largeurs 512/1024/2048, les
projections de Linear/SelfAttention/CrossAttention et le chemin public
`Model::trainStep`. Les produits par blocs conservent l'ordre des réductions,
les politiques FMA et les biais ; le backward SelfAttention emprunte une arène
réutilisable. Les transferts OpenCL, les scratchpads pointwise ROCm et le bruit
du worker TPU ont aussi été examinés et optimisés.

Le [rapport détaillé](Runtime-Dimensions-20261112.md) distingue les mesures
CPU, les vérifications GPU réellement exécutées, la compilation HIP et la
référence CPU/JAX. Il fournit les temps avant/après, les limites du contrat
mathématique existant et les commandes des sondes reproductibles.

La [suite sur le backward d'attention](Runtime-Attention-Tiling-20261112.md)
remplace les grandes matrices de probabilités par des blocs réutilisables,
avec poursuite des accumulateurs FP32 entre blocs et vérification bit à bit.

La [suite sur la sélection mesurée de l'attention](Runtime-Attention-Policy-20261112.md)
active le choix de workers pour SelfAttention/CrossAttention dans le routeur
Model. Elle accélère aussi les contrôles NaN/Inf avec un dispatch SIMD portable
et distingue le gain après chauffe du coût de calibration au démarrage.

## 9) Préparation anticipée des données

La [préparation anticipée du dataset](Runtime-Dataset-Prefetch-20261112.md)
utilise une file bornée et des buffers de conversion réutilisés pour recouvrir
les lectures et conversions avec le calcul des batches. Le rapport conserve
les mesures de débit, les budgets et les contrôles de précision.

## Étapes suivantes

- [Page précédente : Internals : bindings Lua (`LuaScripting` / `LuaContext`)](17-Lua-Bindings-Internals.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Internals : registre d’architectures (`ModelArchitectures`) et builders](19-Models-Registry-And-Builders.md)
