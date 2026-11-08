# Layers disponibles

Trouver rapidement le contrat API réel et les paramètres utilisables.

**Public concerné :** Développeur et utilisateur intermédiaire/avancé.

> **Prérequis**
>
> Connaître les commandes de base de Mímir.

## Sur cette page

- [Lecture rapide](#lecture-rapide)
- [Où sont les layers](#où-sont-les-layers)
- [Statut](#statut)
- [Paramètres courants](#paramètres-courants)
- [Exemple guidé](#exemple-guidé)
- [Vérifier ce qui est réellement utilisé](#vérifier-ce-qui-est-réellement-utilisé)
- [Catalogue des types (LayerType)](#catalogue-des-types-layertype)
- [Primitives Primer](#primitives-primer)
- [Contrat de NMS](#contrat-de-nms)
- [Reparameterize](#reparameterize)
- [Constant et paramètres sans entrée](#constant-et-paramètres-sans-entrée)
- [Dépannage rapide](#dépannage-rapide)
- [Étapes suivantes](#étapes-suivantes)

## Lecture rapide

Parcours conseillé pour éviter les erreurs :

1. Créer un modèle via le registre (`Mimir.Architectures` + `Mimir.Model.create`).
2. Vérifier `allocate_params`, `init_weights`, puis `forward`.
3. Ajuster ensuite les paramètres layer par layer.

Le runtime exécute une liste de layers. Chaque layer lit des tenseurs nommés et produit un tenseur de sortie.

Cette page parle des layers du graphe modèle (ceux qui vivent dans `Model` et sont exécutés par `model.forward()` / `Mimir.Model.forward()`).

Le module Lua `Mimir.Layers` inspecte les couches du modèle courant en lecture seule. Voir [Mimir.Layers](18-Layers-Module.md).

## Où sont les layers

- Définition: `src/Layers.hpp`
- Orchestration du forward/backward : `src/Model.cpp`
- Dispatch CPU partagé : `src/runtimes/cpu/RuntimeLayerDispatch.hpp`
- Primitives runtime : `src/runtimes/LayerOps.cpp`, `src/runtimes/cpu/LayerOps.hpp` et `src/runtimes/cpu/LayerOpsExt.hpp`

Le catalogue des types est défini dans `src/LayerTypes.hpp`.

## Statut

Tous les types de layers ne sont pas au même niveau.

- Stable (souvent): `Linear`, activations courantes, `LayerNorm`, `GroupNorm`, `Conv2d`, `Add`, `Softmax`, `LogSoftmax`.
- Expérimental ou non optimisé: attention (`Self/MultiHead/Cross`) et certaines ops de shape/routing.

Règle de pouce: si vous voulez un workflow reproductible, passe par `Mimir.Architectures` + `Mimir.Model.create(...)` plutôt que du build manuel layer par layer.

## Paramètres courants

Ces champs existent sur la structure `Layer` et sont généralement remplis depuis la config d'architecture.

| Champ (config) | Utilisé par | Effet |
| --- | --- | --- |
| `in_features`, `out_features` | `Linear`, `Bilinear` | Dimensions du layer dense. |
| `in_channels`, `out_channels` | `Conv2d`, `ConvTranspose2d`, `BatchNorm2d`, `GroupNorm` | Dimensions canaux. |
| `input_spatial_shape`, `kernel_shape`, `stride_shape`, `padding_shape`, `dilation_shape` | `Conv3d`, `Conv4d` | Paramètres ND explicites, dans l'ordre D/H/W ou T/D/H/W ; prioritaires sur les champs scalaires. |
| `groups` | Convolutions groupées | Groupes de canaux ; divisent les canaux d'entrée et de sortie. |
| `kernel_size` (ou `kernel_h`, `kernel_w`) | Conv/Pool | Taille du noyau. |
| `stride` (ou `stride_h`, `stride_w`) | Conv/Pool | Stride. |
| `padding` (ou `pad_h`, `pad_w`) | Conv/Pool/Pad | Padding. |
| `eps` | `LayerNorm`, `GroupNorm`, `RMSNorm` | Stabilité numérique. |
| `num_groups` | `GroupNorm` | Nombre de groupes. |
| `dropout_p` | `Dropout`, `Dropout2d`, attention | Probabilité de dropout. |
| `vocab_size`, `embed_dim` | `Embedding`, `EmbeddingBag` | Dimensions vocab/embedding. |
| `axis` | `Softmax`, `LogSoftmax` | Axe de réduction. |
| `shape`, `axis` | `ReduceSum`, `ReduceMean`, `ReduceMin`, `ReduceMax`, `ReduceProd` | Forme d'entrée et axe indexé depuis zéro ; la dimension réduite reste présente, de taille 1. |
| `shape` | `LowerTriangular` | Forme explicite de rang au moins 2 ; masque les deux derniers axes. |
| `target_shape` | `Reshape`, `View` | Shape cible. |
| `permute_dims` | `Permute` | Ordre des dimensions. |
| `concat_axis` | `Concat` | Axe de concaténation. |
| `split_axis`, `num_splits`, `split_sizes` | `Split` | Paramètres de split. |
| `num_heads`, `head_dim`, `seq_len`, `causal` | Attention | Paramètres d'attention. |
| `nms_iou_threshold`, `nms_score_threshold` | `NMS` | Seuils IoU et score. |
| `nms_max_detections`, `nms_class_agnostic` | `NMS` | Limite de sorties (`0` = illimitée) et mode inter-classes. |

## Exemple guidé

Objectif: obtenir un premier forward fiable avec une architecture du registre.

```lua
local cfg, err = Mimir.Architectures.default_config("transformer")
assert(cfg, err)

cfg.vocab_size = 128
cfg.seq_len = 8
cfg.d_model = 32
cfg.num_layers = 2
cfg.num_heads = 4
cfg.mlp_hidden = 64
cfg.output_dim = 16

assert(Mimir.Model.create("transformer", cfg))
local ok_alloc, nparams_or_err = Mimir.Model.allocate_params()
assert(ok_alloc, nparams_or_err)
assert(Mimir.Model.init_weights("xavier", 0))

local ids, zeros = {}, {}
for i = 1, cfg.seq_len do ids[i] = 1 end
for i = 1, cfg.d_model do zeros[i] = 0.0 end
local out, ferr = Mimir.Model.forward({
    __input__ = ids, mag = zeros, mod = zeros,
}, false)
assert(out, ferr)
assert(#out == cfg.output_dim)
print("out_len=", #out)
```

Pourquoi ce flux aide :

1. Le registre fournit les champs essentiels.
2. Les erreurs apparaissent tôt (paramètres, shape, I/O).
3. Le débogage devient plus local et lisible.

## Vérifier ce qui est réellement utilisé

- Lire l'architecture concernée dans `src/Models/Registry/ModelArchitectures.cpp`.
- Lancer un smoke test et inspecter les logs runtime/planner.

## Catalogue des types (`LayerType`)

![Familles de couches natives, convolutions ND et primitives Primer](../graphs/13_layer_registry.svg)

Le schéma regroupe des familles et exemples canoniques. La présence d’un type
ne garantit pas son exécution native sur chaque backend ; le routeur vérifie
la capacité de l’opération réelle.

Les `type` des configs sont normalisés via `LayerRegistry::normalize_type`.

| Type (canonique) | Alias normalisés (exemples) | Effet |
| --- | --- | --- |
| `Conv2d` | - | Convolution 2D. |
| `ConvTranspose2d` | `ConvTranspose`, `Deconv2d` | Convolution transposée 2D. |
| `Conv1d` | - | Convolution 1D. |
| `Conv3d`, `Conv4d` | `Conv3D`, `Conv4D`, `Convolution3D`, `Convolution4D` | Convolutions à trois ou quatre axes spatiaux. |
| `DepthwiseConv2d` | - | Convolution depthwise 2D. |
| `CausalDepthwiseConv1d` | `causal_depthwise_conv1d` | Convolution causale par canal sur une séquence token-major `[T,C]`. |
| `Linear` | - | Layer dense (GEMM). |
| `Bilinear` | - | Transformation bilinéaire. |
| `Embedding` | - | Embedding (ids int vers vecteurs). |
| `EmbeddingBag` | - | Embedding agrégé. |
| `BatchNorm2d` | `BatchNorm`, `BN`, `BN2d` | BatchNorm 2D. |
| `BatchNorm1d` | `BN1d` | BatchNorm 1D. |
| `LayerNorm` | `LN` | LayerNorm. |
| `GroupNorm` | `GN` | GroupNorm. |
| `InstanceNorm2d` | `IN` | InstanceNorm 2D. |
| `RMSNorm` | - | RMSNorm. |
| `ReLU` | `ReLu`, `Relu`, `RELU` | Activation ReLU. |
| `LeakyReLU` | - | Activation LeakyReLU. |
| `GELU` | `Gelu` | Activation GELU. |
| `GEGLU` | `Geglu` | Activation gated GELU. |
| `SwiGLU` | `SwiGlu`, `swiglu` | Deux entrées de mêmes tailles : `SiLU(gate)*value`. |
| `SiLU` | `Swish`, `silu`, `swish`, `Silu` | Activation SiLU/Swish. |
| `Tanh` | - | Activation tanh. |
| `Sigmoid` | - | Activation sigmoid. |
| `Softmax` | - | Softmax. |
| `LogSoftmax` | - | Log-Softmax. |
| `Softplus` | - | Softplus. |
| `Mish` | - | Mish. |
| `HardSigmoid` | - | Hard-sigmoid. |
| `HardSwish` | - | Hard-swish. |
| `MaxPool2d` | `MaxPool` | MaxPool 2D. |
| `AvgPool2d` | `AvgPool` | AvgPool 2D. |
| `AdaptiveAvgPool2d` | `AdaptiveAvgPool` | Adaptive AvgPool 2D. |
| `GlobalAvgPool2d` | `GlobalAvgPool` | Global AvgPool 2D. |
| `MaxPool1d` | - | MaxPool 1D. |
| `AvgPool1d` | - | AvgPool 1D. |
| `TokenMeanPool` | - | Mean pooling sur tokens. |
| `Dropout` | - | Dropout. |
| `Dropout2d` | - | Dropout 2D. |
| `AlphaDropout` | - | AlphaDropout. |
| `Flatten` | - | Flatten. |
| `Reshape` | - | Reshape. |
| `Transpose` | - | Transpose. |
| `Permute` | - | Permute dimensions. |
| `Squeeze` | - | Squeeze. |
| `Unsqueeze` | - | Ajoute une dimension. |
| `View` | - | Alias reshape/view. |
| `Add` | - | Addition élément par élément. |
| `Subtract` | - | Soustraction élément par élément. |
| `Multiply` | - | Multiplication élément par élément. |
| `Divide` | - | Division élément par élément. |
| `Reparameterize` | - | Réparamétrisation VAE. |
| `Concat` | `Concatenate`, `Cat` | Concaténation. |
| `Split` | - | Split. |
| `Chunk` | - | Chunk. |
| `Stack` | - | Stack. |
| `MatMul` | - | MatMul. |
| `BatchMatMul` | - | Batch MatMul. |
| `NMS` | `NonMaxSuppression` | Supprime les boîtes redondantes selon leur IoU et leur score. |
| `SelfAttention` | - | Self-attention. |
| `MultiHeadAttention` | - | Multi-head attention. |
| `CrossAttention` | - | Cross-attention. |
| `UpsampleNearest` | - | Upsample nearest. |
| `UpsampleBilinear` | - | Upsample bilinear. |
| `UpsampleBicubic` | - | Upsample bicubic. |
| `PixelShuffle` | - | Pixel shuffle. |
| `LSTM` | - | LSTM. |
| `GRU` | - | GRU. |
| `RNN` | - | RNN. |
| `ZeroPad2d` | - | Padding zero 2D. |
| `ReflectionPad2d` | - | Padding reflection 2D. |
| `ReplicationPad2d` | - | Padding replication 2D. |
| `Identity` | - | Identity. |
| `Constant` | - | Tenseur sans entrée, fixe par défaut ou paramètre appris si explicitement marqué. |
| `Lambda` | - | Lambda custom. |
| `PatchEmbed` | `PatchProjection` | Patch embedding/projection. |

## Primitives Primer

Ces 18 types complètent le catalogue. Ils sont sans poids : utiliser
`params_count=0` lors de leur ajout. Les tenseurs sont aplatis dans l'API Lua ;
`shape` décrit leur géométrie lorsque l'opération en a besoin.

| Types canoniques | Calcul et paramètres |
| --- | --- |
| `SquaredReLU`, `Square` | `max(0,x)^2` et `x*x`, élément par élément. Alias de `SquaredReLU` : `ReLU2`, `ReLUSquared`. |
| `SqrtAbs`, `LogAbs` | `sqrt(abs(x))` et `log(abs(x))`. Alias `AbsRoot` et `log_abs`, respectivement. |
| `Exp`, `Abs`, `ReciprocalNoNan`, `Sign`, `Sin`, `Cos` | Exponentielle, valeur absolue, inverse avec zéro conservé, signe, sinus et cosinus. |
| `Maximum`, `Minimum` | Deux entrées de mêmes tailles, ou une entrée scalaire diffusée sur l'autre. Pas de broadcasting de rang supérieur. |
| `ReduceSum`, `ReduceMean`, `ReduceMin`, `ReduceMax`, `ReduceProd` | Un axe de `shape`, désigné par `axis` ; indices négatifs acceptés. Sans `shape`, réduction du vecteur entier avec `axis=0` ou `-1`. |
| `LowerTriangular` | `shape` obligatoire de rang au moins 2 ; conserve `col<=row` sur les deux derniers axes et met le reste à zéro. |

Les réductions conservent la dimension réduite à 1. `Maximum`/`Minimum` partagent
le gradient à égalité ; les réductions min/max le partagent entre tous les
extrema égaux. `ReduceProd` calcule son gradient sans diviser par le produit
total, afin de prendre en compte les facteurs nuls.

`LogAbs` refuse zéro. `SqrtAbs` accepte zéro en forward mais refuse son gradient
singulier ; aucun epsilon n'est ajouté implicitement. Les gradients de `Abs`,
`Sign` et `ReciprocalNoNan` en zéro sont définis nuls. Les tenseurs vides ou
non finis et les résultats non finis sont refusés.

`LowerTriangular` ne remplace pas le masque adapté aux logits d'une attention
causale : un logit mis à zéro peut encore recevoir une probabilité après softmax.

La topologie est définie en C/C++ dans une classe dérivée de `Model`.
La classe `PrimerReductionModel` du [contrat des architectures](../07-Devs/08-Architecture-Contract.md)
illustre `SquaredReLU` puis `ReduceMean`. Après sa définition, cet extrait C++
exécute le graphe et son backward :

```cpp
PrimerReductionModel model;
model.allocateParams();
model.zeroGradients();
const auto output = model.forwardPass(
    std::vector<float>{-1.f, 2.f, 3.f, 4.f, -5.f, 6.f}, true);
model.backwardPass({1.f, 1.f});
// Deux valeurs de sortie, de forme [2,1] : 13/3 et 52/3.
```

Les contrats runtime, gradients et limites numériques sont détaillés dans
[Layers et opérations](../04-Architecture-Internals/14-Layers-And-Ops.md#primitives-issues-de-létude-primer).
L'accès aux nouvelles primitives ne modifie pas automatiquement les architectures
existantes. `scripts/tests/test_primer_ops.lua` exerce les bindings internes
utilisés par les tests et l'import MPK ; il ne constitue pas un modèle de
définition d'architecture depuis Lua. MPK reste expérimental et partiel pour
la production.

## Contrat de `NMS`

`NMS` possède un forward de sélection discrète et un backward explicite de
stop-gradient. Elle reçoit :

1. `boxes` : un vecteur aplati de `N * 4` coordonnées au format
   `x1, y1, x2, y2` ;
2. `scores` : `N` scores ;
3. `classes` : `N` identifiants de classe optionnels.

La sortie contient les indices d'origine conservés, triés par score décroissant
et encodés en `float` comme les autres tenseurs du runtime. À score égal,
l'indice le plus petit est prioritaire. Les candidats non finis sont ignorés.
Avec l'entrée `classes`, la suppression est intra-classe par défaut ; activez
`nms_class_agnostic` pour supprimer aussi entre classes.

Le forward et le backward sont natifs sur CPU, CUDA, Vulkan et OpenCL quand
le backend est compilé et disponible. ROCm conserve sa couverture actuelle ;
le routeur peut sélectionner un autre backend, dont le CPU.

Le backward reçoit un seul gradient amont, dont la longueur doit correspondre
au nombre d'indices conservés. Il renvoie un tenseur de zéros de même taille que
chaque entrée : `4 * N` pour les boîtes, `N` pour les scores et `N` pour les
classes lorsqu'elles sont fournies. Les entrées vides sont acceptées ; une forme
invalide ou un gradient amont de mauvaise longueur fait retourner `false`.

Ces zéros définissent un **stop-gradient** : ils ne constituent pas une dérivée
sur les frontières de sélection, ni une approximation différentiable de NMS.
Une loss placée uniquement après ces indices ne fournit donc pas de gradient
pour entraîner les boîtes ou les scores. Les losses de détection doivent porter
sur les prédictions avant NMS.

Le test `RuntimeTest.NMS` couvre le contrat CPU et le routage ; les tests
`RuntimeTest.OPENCLNativeOps` et `RuntimeTest.VULKANNativeOps` comparent les
gradients au CPU et vérifient la sélection effective du backend GPU.

## `Reparameterize`

Entrées :

```text
inputs[0] = mu
inputs[1] = logvar
```

En entraînement stochastique :

```text
z = mu + exp(0.5 * clamp(logvar, -20, 20)) * epsilon
epsilon ~ N(0, I)
```

En inférence, ou si `modelConfig.stochastic_latent=false`, la sortie est `z=mu`.

Le backward principal conserve la sortie `z` du forward afin de reconstruire `epsilon`. Il propage :

```text
dz/dmu     = 1
dz/dlogvar = 0.5 * epsilon * exp(0.5 * logvar)
```

La dérivée par rapport à `logvar` est nulle lorsque la valeur brute est hors des bornes du clamp.

## `Constant` et paramètres sans entrée

Une couche `Constant` lit directement son bloc de poids et n’a pas d’entrée.

Par défaut :

```cpp
layer.trainable_parameter = false;
```

Elle reste alors fixe et son backward ne crée aucun gradient.

Pour représenter un paramètre appris sans entrée :

```cpp
model.push("latent_bias", "Constant", latent_dim);
if (auto* layer = model.getLayerByName("latent_bias")) {
    layer->inputs = {};
    layer->output = "latent_bias_out";
    layer->trainable_parameter = true;
}
```

Le gradient amont est accumulé directement dans `grad_weights`, puis l’optimiseur met à jour le bloc. `vae_conv/z_prior_bias` utilise ce mécanisme.

N’active pas ce marqueur sur une constante structurelle — masque, zéro de remplissage ou valeur de référence — sauf si elle doit réellement être apprise.

## Dépannage rapide

Si vous voyez une erreur de shape :

1. Vérifier les dimensions du layer source (`in_*`, `out_*`).
2. Vérifier les paramètres structurants (`kernel`, `stride`, `padding`).
3. Vérifier `inputs` et `output` si le graphe est multi-branches.

Si un type semble ignoré :

1. Vérifier son nom normalisé.
2. Vérifier les capacités via `Model.runtime_ops()` puis le dispatcher du runtime.
3. Vérifier s'il est stable ou expérimental.

## Étapes suivantes

- [Page précédente : Scripting Contract](00-Scripting-Contract.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Sérialisation (save/load) — résumé](02-Serialization.md)
