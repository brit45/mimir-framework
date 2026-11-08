# Classe `Model` en C++

Comprendre le fonctionnement interne exact des composants runtime.

**Public concerné :** Développeur avancé qui modifie le moteur C/C++.

> **Prérequis**
>
> Connaître les bases C++ et la structure du dépôt.


Cette page documente la classe C++ `Model` : ce qu’elle stocke, quelles méthodes font quoi, et comment ça correspond à l’API Lua.

Source de vérité :

- Déclaration : `src/Model.hpp`
- Implémentation : `src/Model.cpp`
- Layers : `src/Layers.hpp`, `src/LayerTypes.hpp`
- Gradients : `src/Autograd.hpp`

## Sur cette page

- [Vue rapide (Lua → C++)](#vue-rapide-lua-c)
- [Les grandes responsabilités de Model](#les-grandes-responsabilités-de-model)
- [API C++ principale (méthodes clés)](#api-c-principale-méthodes-clés)
- [Créer un modèle via le framework (héritage de Model)](#créer-un-modèle-via-le-framework-héritage-de-model)
- [AMP et entraînement distribué](#amp-et-entraînement-distribué)
- [Construire un graphe manuellement avec push()](#construire-un-graphe-manuellement-avec-push)
- [Routing (wiring) : inputs / output](#routing-wiring-inputs-output)
- [Paramétrer les paramètres internes des layers (C++)](#paramétrer-les-paramètres-internes-des-layers-c)
- [Exemple (C++) — squelette minimal](#exemple-c-squelette-minimal)
- [Points d’attention](#points-dattention)
- [Étapes suivantes](#étapes-suivantes)

## Vue rapide (Lua → C++)

|Appel Lua|Binding C++|Cible principale|Effet|
|---|---|---|---|
|`Mimir.Model.allocate_params()`|`LuaScripting::lua_allocateParams`|`Model::allocateParams`|Alloue les buffers de paramètres (poids) du modèle.|
|`Mimir.Model.init_weights(method, seed)`|`LuaScripting::lua_initWeights`|`Model::initializeWeights`|Initialise les poids (Xavier/He/…).|
|`Mimir.Model.forward(input, training)`|`LuaScripting::lua_forwardPass`|`Model::forwardPass*`|Forward pass (mono-input ou multi-input nommé).|
|`Mimir.Model.backward(loss_grad)`|`LuaScripting::lua_backwardPass`|`Model::backwardPass` (best-effort)|Backprop (quand supportée par l’archi/layers).|
|`Mimir.Model.optimizer_step(...)`|`LuaScripting::lua_optimizerStep`|`Model::optimizerStep` / `Optimizer`|Met à jour les paramètres avec l’optimizer.|

> Remarque : le runtime Lua orchestre aussi un contexte (`LuaContext`) qui contient tokenizer/encoder/dataset/etc. La classe `Model` est le cœur calcul.

## Les grandes responsabilités de `Model`

|Zone|Où|Rôle|
|---|---|---|
|Topologie / layers|`Model` + `Layer`|Stocke la liste ordonnée de `Layer` et leur routing (`inputs`/`output`).|
|Paramètres|`Layer::weight_block` + buffers legacy|Réserve/organise les poids + gradients + états (BatchNorm, etc.).|
|Execution forward|`Model::forwardPass*`|Résout les entrées et exécute le plan via `RuntimeRouter` ; les kernels appartiennent aux runtimes.|
|Execution backward|`Model::backwardPass` (et variantes)|Best-effort selon layers.|
|Optimisation|`Optimizer`|Applique SGD, Adam, AdamW, Lion, Adafactor, RAdam, NAdam, RMSprop ou LAMB + stratégies de LR decay.|
|Sécurité mémoire|`MemoryGuard`|Bloque/limite certaines allocations si strict mode.|

## API C++ principale (méthodes clés)

|Méthode|Effet|Notes|
|---|---|---|
|`build()`|Workflow générique historique C++|Appelle les hooks de topologie et alloue/initialise les poids. Le registre utilise les builders natifs ; le binding Lua `Mimir.Model.build()` est un no-op de compatibilité.|
|`allocateParams()`|Alloue les paramètres|À faire avant `initializeWeights` et avant l’entraînement.|
|`initializeWeights(method, seed)`|Init des poids|Peut être bloqué si paramètres gelés (freeze).|
|`totalParamCount()`|Compte les paramètres|Utile pour métriques/logs.|
|`forwardPass(input, training)`|Forward (float)|Retourne `std::vector<float>` (copie).|
|`forwardPassView(input, training)`|Forward (float) sans copie|Retourne une vue sur un buffer interne (valide jusqu’au prochain forward).|
|`forwardPass(input_ids, training)`|Forward (ids int)|Pour archis qui consomment des tokens (Embedding).|
|`forwardPassNamed(float_inputs, int_inputs, training)`|Forward multi-entrées nommées|Chemin recommandé pour combiner latent + ids texte, etc.|
|`setTokenizer(t)` / `setEncoder(e)`|Injecte tokenizer/encoder|En pratique, le runtime synchronise modèle ↔ contexte.|
|`freezeParameters(true/false)`|Gèle/dégèle les paramètres|Bloque les opérations qui mutent poids/gradients.|

## AMP et entraînement distribué

`setAutocastDType` accepte `float32`, `float16` et `bfloat16`. Les deux
précisions réduites requièrent `ENABLE_AMP=ON` (défaut CMake). Le routeur
interroge `supportsAutocast` pour chaque opération : seul le runtime CPU
déclare actuellement `Linear` F16/BF16. Les autres opérations gardent leur
chemin ordinaire ; activer AMP ne convertit pas tout le graphe en précision
réduite. Un `Linear` demandé en AMP sans backend compatible est refusé.
Le backward utilise les activations arrondies du forward pour ces projections.
Le dtype de stockage (`setDefaultDType`) est un réglage distinct.

`DistributedRuntime`, sous `src/runtimes/`, utilise MPI avec
`ENABLE_MPI=ON` (défaut CMake `OFF`). Il initialise le groupe avec
`MPI_THREAD_FUNNELED`, vérifie la topologie identique entre rangs, diffuse les
poids initiaux et moyenne les gradients avant l'update. Une compilation sans
MPI renvoie une erreur explicite si DDP est demandé.

Le lancement multiprocessus et l'activation du mode d'entraînement sont deux
étapes distinctes : `run_mimir.sh --mpi-ranks 2` invoque le lanceur MPI,
alors que le script causal attend aussi `--ddp` ou `training.ddp=true`.
Le script initialise DDP avant les threads de monitoring et réserve les
interfaces au rang principal. Deux rangs MPI ne signifient pas deux GPU ;
le choix du périphérique et la couverture des opérations restent séparés.

Les raccourcis de compilation sont `config.sh --amp` et `config.sh --ddp` ;
ils règlent respectivement `ENABLE_AMP` et `ENABLE_MPI`. Ils ne démarrent pas
eux-mêmes un entraînement.

## Créer un modèle via le framework (héritage de `Model`)

Une architecture native est définie en C/C++ dans une classe dérivée de `Model`.
Deux modes d’instanciation de cette même classe sont possibles :

- **Approche A (recommandée)** : passer par le registre `ModelArchitectures` (usiné pour CLI/Lua/config JSON).
- **Approche B** : instancier directement la classe dérivée et appeler sa routine
  de construction, par exemple `buildFromConfig`.

Ces deux modes ne changent pas le langage de définition du graphe. Les scripts
et JSON pilotent les instances. MPK est l’autre voie de description, encore
expérimentale et pas entièrement exploitable en production ; voir le
[contrat des architectures](../07-Devs/08-Architecture-Contract.md).

### Approche A — Registre `ModelArchitectures` (config → modèle)

Source : `src/Models/Registry/ModelArchitectures.hpp` + `src/Models/Registry/ModelArchitectures.cpp`.

Le registre expose 3 opérations simples :

- `ModelArchitectures::available()` : liste des archis.
- `ModelArchitectures::defaultConfig(name)` : retourne la config par défaut.
- `ModelArchitectures::create(name, cfg)` : instancie **une classe dérivée** et appelle sa routine de build “from config”.

Ensuite, vous faites (en général) :

1. Construire la topologie (via `create`)
2. Allouer les poids (`allocateParams()`)
3. Initialiser les poids (`initializeWeights(...)`) ou charger un checkpoint

Exemple C++ minimal (même pattern que `src/main.cpp`) :

```cpp
#include "Models/Registry/ModelArchitectures.hpp"

int main() {
    using ModelArchitectures::json;

    const std::string arch = "basic_mlp";
    json cfg = ModelArchitectures::defaultConfig(arch);
    cfg["input_dim"] = 128;
    cfg["hidden_dim"] = 64;
    cfg["output_dim"] = 10;
    cfg["hidden_layers"] = 2;
    cfg["dropout"] = 0.1;

    auto model = ModelArchitectures::create(arch, cfg);

    // Important: la plupart des modèles “buildInto” posent la topologie,
    // mais n’allouent pas forcément les poids automatiquement.
    model->allocateParams();
    model->initializeWeights("he", 42);

    return 0;
}
```

#### Comment une architecture est définie (existant)

Dans `ModelArchitectures.cpp`, chaque entrée ressemble à :

- `Entry{name, description, default_config, create(cfg)}`

Le `create(cfg)` instancie une classe dérivée (ex: `BasicMLPModel`, `TransformerModel`, …) et appelle une méthode du style :

- `m->buildFromConfig(parsed_cfg)`

Un exemple concret à lire :

- `src/Models/MLP/BasicMLPModel.cpp` → `BasicMLPModel::buildInto(Model&, Config)`
- `src/Models/NLP/TransformerModel.cpp` → `TransformerModel::buildInto(Model&, Config)`

Ces “builders” utilisent `Model::push(...)` **et** le routing par noms (`Layer.inputs` / `Layer.output`) pour construire le graphe.

#### Ajouter votre propre architecture au framework

1. Crée une classe dérivée `MyModel : public Model` (dans `src/Models/...`).
2. Ajoute une `Config` + `buildFromConfig(const Config&)`.
3. Implémente un `static void buildInto(Model&, const Config&)` qui :
   - vide les layers existants,
   - appelle `push(...)` pour ajouter vos layers,
   - configure `Layer.inputs`/`Layer.output` + champs de dimensions,
   - calcule `params_count` correctement.
4. Enregistre l’entrée dans `ModelArchitectures::Registry` (typiquement dans `ensureBuiltinsRegistered()` de `ModelArchitectures.cpp`).

### Approche B — Instancier la classe native directement

L'usage actuel consiste à appeler `buildFromConfig(...)` sur la classe dérivée,
puis à allouer et initialiser ou charger ses poids. Cette construction est la
même que celle utilisée par sa fabrique dans le registre.

### Construction générique historique C++

La classe `Model` définit :

- `build()` : une construction générique (actuellement un exemple backbone U-Net via `buildBackboneUNet(...)`) et une allocation/init automatiques si `totalParamCount()>0`.
- `autoBuildFromDataset(dir)` : détecte les modalités, appelle `buildBackboneUNet(...)`, puis `buildTextBranch/buildImageBranch/...` et `injectMagicToken(...)`.

Les hooks à surcharger (dans une classe dérivée) sont déclarés `virtual` dans `src/Model.hpp` :

- `buildBackboneUNet(...)`
- `buildTextBranch(...)`, `buildImageBranch(...)`, `buildAudioBranch(...)`, `buildVideoBranch(...)`
- `injectMagicToken(...)`

`Model::build()` et `autoBuildFromDataset()` ne sont pas déclarées `virtual`.
Seuls les hooks ci-dessus sont surchargeables polymorphiquement. Pour une
nouvelle architecture, préférez une routine explicite `buildFromConfig` et
son enregistrement ; ne comptez pas sur un override de `build()` via `Model*`.

## Construire un graphe manuellement avec `push()`

Quand vous voulez construire un modèle *sans* passer par le registre d’architectures, la méthode centrale est :

- `Model::push(name, type, params_count)`

Ce que fait `push()` (implémentation actuelle dans `src/Model.cpp`) :

- Normalise `type` (alias) via `normalize_type()` / `LayerRegistry::normalize_type()`.
- Construit un `Layer(name, normalized_type, params_count)`.
- Rejette le layer si le type n’est pas supporté (`LayerType::UNKNOWN` → exception).
- Applique quelques paramètres depuis `modelConfig` **si présents** (ex: `in_channels`, `out_channels`, `height`, `width`, `kernel`, `stride`, `padding`).
- Calcule `output_height/output_width` pour `Conv2d` / `ConvTranspose2d` si possible.
- Ajoute le layer à `Model::layers`.

Ce que `push()` **ne fait pas** :

- Il ne devine pas le bon `params_count` (vous devez le calculer).
- Il ne configure pas automatiquement `in_features/out_features`, `seq_len`, `embed_dim`, etc.
- Il ne route pas des tenseurs multi-entrées à votre place (ça passe par `Layer.inputs` / `Layer.output`).

### Règle d’or : figer la topologie avant `allocateParams()`

`allocateParams()` alloue **un bloc de poids par layer** (un `tensor` de taille `params_count`) et le relie via `Layer::weight_block`.

Conséquence pratique :

- Si vous changez `params_count` après `allocateParams()`, vous devez réallouer (rebuild + `allocateParams()` à nouveau), sinon les lectures via `Layer::getWeights()` deviennent incohérentes.

## Routing (wiring) : `inputs` / `output`

Le forward (dans `Model::forwardPassView`) utilise un `TensorStore` interne adressé par nom :

- Si `layer.inputs` est vide → entrée par défaut `{"x"}`.
- Si `layer.output` est vide → sortie par défaut `"x"`.

Donc, pour construire un graphe non-linéaire (skip connections, concat, add…), vous réglez :

- `layer.inputs = {"a", "b", ...}`
- `layer.output = "c"`

Notes importantes :

- Le modèle injecte aussi un alias immuable `"__input__"` au début du forward (utile si `"x"` est réutilisé par la suite avec une taille différente).
- En mode multi-entrées (`forwardPassNamed`), vous pouvez injecter des tenseurs float et des tenseurs int dans le store (et certains alias sont ajoutés côté runtime).

## Paramétrer les paramètres internes des layers (C++)

Les paramètres “internes” d’un layer sont essentiellement les champs de `struct Layer` (`src/Layers.hpp`).
Ils sont validés et lus par les capacités, les dispatchers et les kernels des
runtimes. `Model.cpp` orchestre le graphe ; il ne doit pas devenir le lieu
d'implémentation des calculs d'une nouvelle couche.

### Pattern général

1. `m.push("nom", "Type", params_count)`
2. `Layer* L = m.getLayerByName("nom")` puis configurer `L->...`
3. Quand tout est prêt : `m.allocateParams(); m.initializeWeights(...)`

### Cheat-sheet : champs minimum + `params_count` attendu

Les formules ci-dessous correspondent à ce que les forwards lisent **actuellement** (donc c’est la meilleure source pour définir `params_count`).

|Type (`LayerType`)|Champs à configurer (minimum utile)|Layout des poids (dans `weight_block`)|`params_count` typique|
|---|---|---|---|
|`Linear`|`in_features`, `out_features`, `use_bias` (optionnel), `seq_len` (optionnel)|`W` puis `bias` si activé. `W` est indexé comme `out_features` blocs de `in_features`.|$in\_f\cdot out\_f + (use\_bias?out\_f:0)$|
|`Conv2d`|`in_channels`, `out_channels`, `input_height`, `input_width`, `kernel_size`, `stride`, `padding`, `use_bias`|Filtres avec l'index `((oc*in_c+ic)*k+kh)*k+kw`, puis `out_channels` biais si activés.|$out\_c\cdot in\_c\cdot k\cdot k + (use\_bias?out\_c:0)$|
|`LayerNorm`|`in_features` (taille normalisée), `eps`, `affine`, `use_bias`|Si `affine`: `gamma[normalized]` puis `beta[normalized]` si `use_bias`.|Si `affine`: $n + (use\_bias?n:0)$ sinon `0`|
|`GroupNorm`|`in_channels`, `input_height`, `input_width`, `num_groups`, `eps`, `affine`, `use_bias`|Si `affine`: `gamma[channels]` puis `beta[channels]` si `use_bias`.|Si `affine`: $c + (use\_bias?c:0)$ sinon `0`|
|`Embedding`|`vocab_size`, `embed_dim`, `padding_idx` (optionnel)|Table `[vocab_size * embed_dim]` (pas de bias).|$vocab\_size\cdot embed\_dim$|
|`SelfAttention` / `MultiHeadAttention`|`seq_len`, `embed_dim`, `num_heads`, `causal`|`Wqkv` puis `Wout`. `Wqkv` a taille `3*E*E`, `Wout` a taille `E*E`.|$4\cdot E\cdot E$|
|`CrossAttention`|`embed_dim` (ou `head_dim*num_heads`), `num_heads`, `causal`|`Wq[E*E]`, `Wkv[2*E*E]`, `Wout[E*E]`.|$4\cdot E\cdot E$|
|`PatchEmbed`|`embed_dim` (d_model), `patch_dim`, `num_patches`, `seq_text`|`W[patch_dim*d_model]` puis `b[d_model]`.|$patch\_dim\cdot d\_model + d\_model$|
|`Bilinear`|`in_features` (in1), `out_features` (in2), `embed_dim` (out), `use_bias`|`W[out*in1*in2]` puis `bias[out]` si activé.|$out\cdot in1\cdot in2 + (use\_bias?out:0)$|
|Ops sans poids (`ReLU`, `GELU`, `Add`, `Concat`, `Split`, `MatMul`, …)|Selon l’opération (souvent rien)|Aucun poids|`0`|

### Exemple C++ : mini MLP (Linear → ReLU → Linear)

Cet exemple montre le principe : `push()` + configuration des champs + allocation/initialisation.

```cpp
#include "Model.hpp"

class MiniMLPModel final : public Model {
public:
    MiniMLPModel() {
        setModelName("mini_mlp");
        push("fc1", "Linear", 128 * 64 + 64);
        auto* fc1 = getLayerByName("fc1");
        fc1->inputs = {"__input__"};
        fc1->output = "hidden";
        fc1->in_features = 128;
        fc1->out_features = 64;
        fc1->use_bias = true;

        push("relu1", "ReLU", 0);
        auto* relu = getLayerByName("relu1");
        relu->inputs = {"hidden"};
        relu->output = "activated";

        push("fc2", "Linear", 64 * 10 + 10);
        auto* fc2 = getLayerByName("fc2");
        fc2->inputs = {"activated"};
        fc2->output = "x";
        fc2->in_features = 64;
        fc2->out_features = 10;
        fc2->use_bias = true;
    }
};

int main() {
    MiniMLPModel model;
    model.allocateParams();
    model.initializeWeights("xavier", 42);
    const auto output = model.forwardPass(std::vector<float>(128, 0.25f), false);
    return output.size() == 10 ? 0 : 1;
}
```

### Exemple : wiring multi-entrées (Add)

Ici, on produit deux tenseurs nommés puis on les additionne.

```cpp
// Hypothèse: vous avez déjà créé deux branches qui écrivent dans "a" et "b".
m.push("add", "Add", 0);
Layer* add = m.getLayerByName("add");
add->inputs = {"a", "b"};
add->output = "x";
```

## Exemple (C++) — squelette minimal

```cpp
#include "Models/MLP/BasicMLPModel.hpp"

int main() {
    // BasicMLPModel hérite de Model ; son graphe est défini dans sa classe C++.
    BasicMLPModel model;
    BasicMLPModel::Config config;
    config.input_dim = 4;
    config.hidden_dim = 8;
    config.output_dim = 2;
    config.hidden_layers = 1;
    model.buildFromConfig(config);
    model.allocateParams();
    model.initializeWeights("xavier", 42);
    const auto output = model.forwardPass(
        std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f}, false);
    return output.size() == 2 ? 0 : 1;
}
```

## Points d’attention

- `forwardPassView` / `forwardPassNamedView` renvoient une référence vers un buffer interne : si vous appelez un autre forward ensuite, la référence peut changer.
- Le support backward/optimizer dépend de l’architecture et des layers réellement utilisés : c’est volontairement “best-effort”.
- La config (dimensions, `seq_len`, `embed_dim`, etc.) est majoritairement définie par le registre `ModelArchitectures`.

## Étapes suivantes

- [Page précédente : Internals: AdvancedRAMManager (cache RAM / compression / spill disque)](05-AdvancedRAMManager.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Internals : `Helpers.hpp` (helpers C++)](11-Helpers.md)
