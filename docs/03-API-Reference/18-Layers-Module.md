# `Mimir.Layers` — inspection en lecture seule

`Mimir.Layers` retourne des listes des couches du modèle courant. Ces appels ne
calculent pas de tenseurs et n'allouent pas les poids du modèle.

```lua
assert(Mimir.Model.create("basic_mlp"))
for _, layer in ipairs(Mimir.Layers.linear()) do
  print(layer.index, layer.name, layer.type, layer.param_count)
end

-- Chaque nom canonique possède également son alias entièrement minuscule.
local convs = Mimir.Layers.Conv2d() -- identique à Mimir.Layers.conv2d()
local attention = Mimir.Layers.crossattention()
```

| Appel | Retour |
|---|---|
| `Mimir.Layers.available()` | Les noms canoniques fournis par `LayerRegistry::get_all_supported_types()` |
| `Mimir.Layers.<Type>()` / `Mimir.Layers.<type_minuscule>()` | Toutes les couches du type exact, dans l'ordre du graphe |
| `Mimir.Layers.by_type("GroupNorm")` | Le même filtre, avec un nom passé en argument |
| `Mimir.Layers.activation()` | Le groupe historique des activations, défini par une liste fixe |
| `Mimir.Layers.batchnorm()` | Le groupe BatchNorm2d / InstanceNorm2d |
| `Mimir.Layers.attention()` | Le groupe des attentions |

Chaque type ajouté au registre reçoit automatiquement son filtre canonique et
son alias en minuscules. Les 18 primitives Primer sont donc inspectables ainsi :

| Famille | Filtres exacts disponibles |
| --- | --- |
| Fonctions élémentaires | `SquaredReLU`, `Square`, `SqrtAbs`, `Exp`, `LogAbs`, `Abs`, `ReciprocalNoNan`, `Sign`, `Sin`, `Cos` |
| Comparaisons | `Maximum`, `Minimum` |
| Réductions | `ReduceSum`, `ReduceMean`, `ReduceMin`, `ReduceMax`, `ReduceProd` |
| Masque | `LowerTriangular` |

```lua
local squared = Mimir.Layers.SquaredReLU()
local means = Mimir.Layers.reducemean()
local masks = Mimir.Layers.by_type("LowerTriangular")
```

Ces appels inspectent uniquement des couches déjà présentes. Les architectures
sont définies en C/C++ dans une classe dérivée de `Model`, puis sélectionnées
par le registre et exécutées depuis les scripts. Le [contrat des architectures](../07-Devs/08-Architecture-Contract.md)
décrit cette règle et l’alternative MPK expérimentale, encore partielle pour
la production. L’[exemple Primer](01-Layers.md#primitives-primer) utilise cette
définition native.
Les alias de construction tels que `ReLU2` se passent à `by_type("ReLU2")` ; ils
ne créent pas une fonction `Mimir.Layers.ReLU2()`.

Les points d'entrée historiques (`activation`, `batchnorm`, `layernorm`,
`attention`, `maxpool2d`, `avgpool2d`) restent des filtres de lecture ; ils ne
lancent pas un calcul standalone. Le groupe `activation()` ne contient pas
actuellement `SquaredReLU` ou `SwiGLU` : employer leurs filtres exacts.

Les listes sont des snapshots détachés du modèle. Les affectations ordinaires
aux listes, aux fiches et aux tableaux `inputs` lèvent une erreur. `#`, `ipairs`
et `pairs` permettent leur lecture. Sans modèle ou sans couche correspondante,
le résultat est une liste vide.

Chaque fiche contient `index` (base 1), `name`, `type`, `param_count`, `output`,
`inputs` et les dimensions de convolution, projection, embedding ou attention.
`param_count` décrit les paramètres du graphe même avant leur allocation.
Il vaut zéro pour chacune des 18 primitives Primer. La taille de la liste
`Mimir.Layers.available()` est indépendante du nombre de couches du modèle actif.

## Ops disponibles par runtime

`Mimir.Model.runtime_ops()` retourne une liste de backends CPU, CUDA, ROCM,
VULKAN, OPENCL et TPU. Chaque entrée contient `name`, `compiled`, `scope` et `ops`.
Chaque op contient `type`, `forward` et `backward`.

Les niveaux sont `NativeOptimized`, `Native`, `HostFallback` (repli CPU) et
`Unsupported`. Un backend absent du build indique `compiled=false` et des
ops `Unsupported`. La requête décrit les capacités déclarées par type sans
initialiser les devices ; les formes de tenseurs, la configuration et le
matériel peuvent ensuite restreindre l'exécution effective.

TPU possède une entrée dans cette matrice, mais `compiled=true` n'atteste pas
qu'un device JAX/XLA/PJRT est accessible ; consultez `hardware_caps().tpu` pour
l'état de session. FPGA n'est pas une entrée de cette matrice d'opérations.

```sh
./run_mimir.sh --lua scripts/tools/inspect_architectures.lua -- --ops --runtime
./run_mimir.sh --lua scripts/tools/inspect_architectures.lua -- --all
./run_mimir.sh --lua scripts/tools/inspect_architectures.lua -- --all --list vae_conv
```

Pour inspecter un fichier externe, `--extern-model` sélectionne
`external_safetensors_base` et transmet le chemin à sa configuration native :

```sh
./run_mimir.sh --lua scripts/tools/inspect_architectures.lua -- --extern-model model.safetensors
./run_mimir.sh --lua scripts/tools/inspect_architectures.lua -- --extern-model "chemin avec espaces/model.safetensors" --all
```

Sans autre sélection de rapport, les paramètres, layers et statistiques du
fichier sont affichés. `--layers`, `--stats`, `--ops`, `--json` et `--export`
restent utilisables. `--list` peut désigner `external_safetensors_base`, mais une
autre architecture est incompatible avec `--extern-model`.

L'inspection lit l'en-tête SafeTensors sans allouer les poids. Le modèle externe
représente les clés de tenseurs par des couches `Identity` ; il ne reconstruit pas
un graphe exécutable à partir des seuls poids. En cas d'export explicite, les
poids du fichier source sont chargés avant la sauvegarde.

`--all` active les rapports, y compris les vues détaillée et compacte des ops.
Sans `--list`, les paramètres, layers et statistiques de toutes les architectures
sont affichés. Aucun export n'est déclenché implicitement ; `--json` et
`--export <path>` restent explicites.

- [Index de la documentation](../00-INDEX.md)
- [Globals & aliases](19-Globals.md)
