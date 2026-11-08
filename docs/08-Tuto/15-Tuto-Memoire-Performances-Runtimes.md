# Mesurer la mémoire et les performances des runtimes

Ce tutoriel part d'un MLP natif minuscule pour apprendre à mesurer sans
confondre préférence de dtype, mémoire de tenseurs et exécution GPU. Aucun
dataset ni accélérateur n'est nécessaire au premier atelier.

| Niveau | Compétence visée |
| --- | --- |
| Débutant | Reconnaître paramètres, activations, gradients et états d'optimiseur. |
| Intermédiaire | Produire une mesure répétable et comparer deux configurations. |
| Expert | Expliquer une décision du planner et vérifier une optimisation numérique. |

## 1. Poser un budget avant l'allocation

Un poids FP32 occupe quatre octets, mais les poids ne représentent qu'une
partie du coût. Il faut aussi compter les gradients, les états d'optimiseur,
les activations conservées pour le backward et les buffers temporaires. Les
copies hôte/device et les structures du processus s'ajoutent à ces tenseurs.

`MemoryGuard` comptabilise et limite les allocations qu'il connaît.
L'allocateur dynamique gère les tenseurs qui passent par lui, avec éviction et
compression si configurées. Aucun de ces deux compteurs n'est le RSS total du
processus. Un scratchpad est un espace temporaire de calcul, pas un paramètre
appris ni un checkpoint.

Enregistrer ce script dans `/tmp/mimir_mesurer.lua` sur un système POSIX :

```lua
assert(Mimir.MemoryGuard.setLimit(1))
assert(Mimir.Allocator.configure({
  max_ram_gb = 1, enable_compression = true,
}))
assert(Mimir.Model.create("basic_mlp", {
  input_dim = 4, hidden_dim = 8, output_dim = 3,
  hidden_layers = 1, dropout = 0, dtype = "float32",
}))
assert(Mimir.Model.set_hardware(false))
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier", 51))
local input = {0.2, -0.1, 0.7, 0.0}
local reference = assert(Mimir.Model.forward(input, false))
for i = 1, 5 do assert(Mimir.Model.forward(input, false)) end

local samples = {}
for i = 1, 31 do
  local started = Mimir.Benchmark.now()
  local output = assert(Mimir.Model.forward(input, false))
  samples[i] = (Mimir.Benchmark.now() - started) * 1000
  for j, expected in ipairs(reference) do
    assert(math.abs(output[j] - expected) <= 1e-6)
  end
end
table.sort(samples)
print("MEASURE_OK", "median_ms", samples[16], "p95_ms", samples[30])
print("parameters", Mimir.Model.total_params())
local guard = Mimir.MemoryGuard.getStats()
local allocator = Mimir.Allocator.get_stats()
print("guard_mb", guard.current_mb, "peak_mb", guard.peak_mb,
      "limit_mb", guard.limit_mb)
print("dynamic_tensors", allocator.tensor_count, "loaded", allocator.loaded_count)
```

La topologie vient de `BasicMLPModel`, une classe C++ dérivée de `Model`, selon
le [contrat d'architecture](../07-Devs/08-Architecture-Contract.md).
Lua configure et appelle cette architecture ; aucun graphe n'y est assemblé.
Le budget de 1 Gio est celui de cet atelier minuscule, pas une valeur universelle.

## 2. Construire une référence reproductible

Depuis la racine du dépôt :

```bash
OMP_NUM_THREADS=1 MIMIR_DISABLE_CUDA=1 MIMIR_DISABLE_ROCM=1 \
MIMIR_DISABLE_VULKAN=1 MIMIR_DISABLE_OPENCL=1 \
MIMIR_RUNTIME_AUTOTUNE=0 \
./run_mimir.sh --lua /tmp/mimir_mesurer.lua
```

Résultat attendu : `MEASURE_OK`, deux durées et les statistiques mémoire.
`Mimir.Benchmark.now()` est une horloge murale monotone. Les durées incluent le
binding Lua et la matérialisation des sorties, pas uniquement le noyau CPU.
Pour une si petite charge, le coût de ce binding peut dominer.

Les cinq forwards de préparation sont séparés des 31 mesures. La médiane et le
p95 évitent de résumer une expérience par son meilleur échantillon. Le test de
valeurs n'est pas inclus dans l'intervalle chronométré.

Répéter en conservant dimensions, seed, dtype et binaire. Une première étape
beaucoup plus lente peut inclure la construction du plan, la compilation d'un
shader ou une calibration ; ce coût doit être publié séparément.

## 3. Comprendre ce que choisit le routeur

Le routeur distingue quatre niveaux :

| Niveau | Ce qu'il affirme |
| --- | --- |
| `Unsupported` | Le backend refuse ce type ou cette opération. |
| `HostFallback` | Le backend passe par le CPU ; ce n'est pas une preuve GPU. |
| `Native` | Le backend possède un chemin natif. |
| `NativeOptimized` | Il déclare un chemin natif spécialisé. |

Les capacités par type ne suffisent pas : formes, paramètres, valeurs,
configuration et budgets sont vérifiés sur l'opération réelle. Une opération
refusée peut être essayée ailleurs. `set_hardware(true)` autorise
l'accélération, mais ne sélectionne pas une carte par son nom.

Pour examiner les déclarations du binaire sans les confondre avec une trace :

```bash
./run_mimir.sh --lua scripts/tools/inspect_architectures.lua -- --ops --runtime
```

L'ordre de base est ROCm, CUDA, Vulkan, OpenCL, TPU si sa session est fournie,
puis CPU. En mode
de coût, la calibration peut privilégier un candidat natif plus rapide pour
l'opération courante. Elle ne transforme pas un `HostFallback` en calcul GPU.
Voir les [runtimes GPU](../04-Architecture-Internals/21-GPU-Runtimes.md).

## 4. Changer une variable à la fois

| Expérience | Conserver | Comparer |
| --- | --- | --- |
| Threads CPU | Graphe, données, ISA | 1 worker puis plusieurs workers. |
| Planner | Seed, dtype, entrées | Mode statique puis mode de coût. |
| Backend | Mêmes valeurs et formes | Temps global, transferts compris. |
| Fusion | Graphe et optimiseur | Sorties, gradients et poids après un pas. |

Les variables `MIMIR_PLANNER_MODE=static|cost`, `MIMIR_RUNTIME_AUTOTUNE`,
`MIMIR_PLANNER_FUSION`, `MIMIR_ENABLE_FUSION_TRAIN` et
`MIMIR_PLANNER_BUFFER_REUSE` correspondent à des mécanismes différents.
Désactiver une fusion n'est pas désactiver tout le planner. Une réutilisation
prévue dans le plan ne prouve pas son emploi dans chaque chemin runtime.

Pour le CPU, `MIMIR_CPU_ISA=auto|scalar|sse2|avx|avx256|avx2|avx512`
contrôle les variantes disponibles ; `avx256` est un alias d'AVX. Le dispatch
vérifie le CPU et le support de l'OS. Une ISA demandée et indisponible est
refusée ; ne l'imposez pas simplement parce qu'elle existe sur une autre machine.

Le [guide du planner](../04-Architecture-Internals/22-Planning.md) décrit les
mesures sur copies jetables, les règles de parité et les décisions en cache.
Les [couches et SIMD](../04-Architecture-Internals/14-Layers-And-Ops.md) précisent
les variantes et les limites des réductions.

## 5. Vérifier une optimisation

Une expérience d'entraînement doit publier des mesures séparées pour forward,
backward et mise à jour de l'optimiseur, en plus du pas complet. Pour une
optimisation censée conserver le résultat, comparer aussi les gradients et les
poids après l'étape ; la seule baisse de loss ne suffit pas.

Les tests ciblés, lorsque leurs cibles ont été construites, comprennent :

```bash
ctest --test-dir build --output-on-failure \
  -R '^RuntimeTest\.(SimdDispatch|FiniteCheck|ConvolutionScratch|AllocatorScratch)$'
```

Choisir ensuite les tests de la famille modifiée : attention, convolutions ND,
optimiseurs, fusion ou primitives Primer. Ne pas lancer un benchmark nommé
`safe` sans lire ses dimensions ; certains profils de construction restent gros.
Les [benchmarks disponibles](../../scripts/benchmarks/README.md) indiquent leurs
charges et limites.

## 6. Exercices et erreurs fréquentes

| Niveau | Travail | Critère de réussite |
| --- | --- | --- |
| Débutant | Expliquer les trois compteurs mémoire du run. | Paramètres, mémoire suivie et RSS ne sont pas confondus. |
| Intermédiaire | Comparer deux nombres de workers. | La même entrée et les mêmes valeurs attendues sont utilisées. |
| Expert | Expliquer une fusion refusée. | Capacité, parité et coût sont distingués dans le plan ou la trace. |

Une limite mémoire atteinte se traite d'abord par des dimensions adaptées et un
diagnostic des buffers ; augmenter le budget ne corrige pas une forme erronée.
Des répétitions sans croissance visible ne prouvent pas l'absence de fuite.
Un microbenchmark plus rapide ne garantit pas un entraînement complet plus rapide.

## Étapes suivantes

- [Compilation portable, AMP et DDP](16-Tuto-Compilation-Portabilite-AMP-DDP.md)
- [Index des tutoriels](00-INDEX.md)
