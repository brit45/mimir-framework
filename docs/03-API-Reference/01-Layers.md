# Layers

Le runtime exécute une liste de layers. Chaque layer lit des tenseurs nommés et produit un tenseur de sortie.

Cette page parle des **layers du graphe du modèle** (ceux qui vivent dans `Model` et sont exécutés par `model.forward()` / `Mimir.Model.forward()`).

À ne pas confondre avec le module Lua `Mimir.Layers` (ops standalone) : dans la version actuelle, ces ops sont des **stubs**. Voir `docs/03-API-Reference/18-Layers-Module.md`.

## Où sont les layers ?

- Définition: `src/Layers.hpp`
- Exécution (forward): `src/Model.cpp` (switch sur LayerType)
- Backward: `src/Model.cpp` (best-effort selon layer)

Le catalogue des types est défini dans `src/LayerTypes.hpp`.

## Statut (important)

Tous les types de layers ne sont pas au même niveau.

- Stable (souvent): Linear, activations, LayerNorm/GroupNorm, Conv2d (préservé), Add, Softmax/LogSoftmax.
- Expérimental / non-optimisé: Attention (Self/MultiHead/Cross) fonctionne mais peut être coûteux; certaines ops “shape” et variantes peuvent être best-effort.

Règle de pouce : si tu veux un workflow reproductible, passe par `Mimir.Architectures` + `Mimir.Model.create(...)` plutôt que de “builder” layer par layer.

## Paramètres

Les paramètres sont configurés par les architectures du registre (recommandé). Certains layers exigent des champs (ex: `in_features`, `out_features`, `seq_len`, `embed_dim`).

### Paramètres courants (les plus utiles)

Ces champs existent sur la structure `Layer` et sont typiquement renseignés via la config d’architecture.

|Champ (config)|Utilisé par|Effet|
|---|---|---|
|`in_features`, `out_features`|`Linear`, `Bilinear`|Dimensions du layer dense.|
|`in_channels`, `out_channels`|`Conv2d`, `ConvTranspose2d`, `BatchNorm2d`, `GroupNorm`|Dimensions canaux.|
|`kernel_size` (ou `kernel_h`, `kernel_w`)|Conv/Pool|Taille du noyau (carré ou rectangulaire).|
|`stride` (ou `stride_h`, `stride_w`)|Conv/Pool|Stride.|
|`padding` (ou `pad_h`, `pad_w`)|Conv/Pool/Pad|Padding.|
|`eps`|Norm (`LayerNorm`, `GroupNorm`, `RMSNorm`, …)|Stabilité numérique.|
|`num_groups`|`GroupNorm`|Nombre de groupes.|
|`dropout_p`|`Dropout`, `Dropout2d`, attention|Probabilité de dropout.|
|`vocab_size`, `embed_dim`|`Embedding`, `EmbeddingBag`|Dimensions de vocab/embedding.|
|`axis`|`Softmax`, `LogSoftmax`|Axe.|
|`target_shape`|`Reshape`, `View`|Shape cible.|
|`permute_dims`|`Permute`|Ordre des dimensions.|
|`concat_axis`|`Concat`|Axe concat.|
|`split_axis`, `num_splits`, `split_sizes`|`Split`|Paramètres de split.|
|`num_heads`, `head_dim`, `seq_len`, `causal`|Attention|Paramètres d’attention.|

## Vérifier ce qui est utilisé

Le plus fiable est :

- regarder l’architecture dans `src/Models/Registry/ModelArchitectures.cpp`
- exécuter un script smoke test et inspecter les logs d’allocation des layers

## Exemple (script) — utiliser les layers via une architecture

```lua
local cfg, err = Mimir.Architectures.default_config("transformer")
assert(cfg, err)

assert(Mimir.Model.create("transformer", cfg))
local ok_alloc, nparams_or_err = Mimir.Model.allocate_params()
assert(ok_alloc, nparams_or_err)
assert(Mimir.Model.init_weights("xavier", 0))

local out, ferr = Mimir.Model.forward({ __input__ = { 1, 2, 3, 4 } }, false)
assert(out, ferr)
log("out_len:", #out)
```

## Catalogue des types (`LayerType`)

Les `type` que tu vois dans les configs d’architecture (et/ou dans `push_layer`) sont normalisés via `LayerRegistry::normalize_type`.

|Type (canonique)|Alias normalisés (exemples)|Effet|
|---|---|---|
|`Conv2d`|—|Convolution 2D.|
|`ConvTranspose2d`|`ConvTranspose`, `Deconv2d`|Convolution transposée 2D (upsampling).|
|`Conv1d`|—|Convolution 1D.|
|`DepthwiseConv2d`|—|Convolution depthwise 2D.|
|`Linear`|—|Layer dense (GEMM).|
|`Bilinear`|—|Transformation bilinéaire.|
|`Embedding`|—|Embedding (ids int → vecteurs).|
|`EmbeddingBag`|—|Embedding agrégé (bag).|
|`BatchNorm2d`|`BatchNorm`, `BN`, `BN2d`|BatchNorm 2D.|
|`BatchNorm1d`|`BN1d`|BatchNorm 1D.|
|`LayerNorm`|`LN`|LayerNorm.|
|`GroupNorm`|`GN`|GroupNorm.|
|`InstanceNorm2d`|`IN`|InstanceNorm 2D.|
|`RMSNorm`|—|RMSNorm.|
|`ReLU`|`ReLu`, `Relu`, `RELU`|Activation ReLU.|
|`LeakyReLU`|—|Activation LeakyReLU.|
|`GELU`|`Gelu`|Activation GELU.|
|`SiLU`|`Swish`, `silu`, `swish`, `Silu`|Activation SiLU/Swish.|
|`Tanh`|—|Activation tanh.|
|`Sigmoid`|—|Activation sigmoid.|
|`Softmax`|—|Softmax.|
|`LogSoftmax`|—|Log-Softmax.|
|`Softplus`|—|Softplus.|
|`Mish`|—|Mish.|
|`HardSigmoid`|—|Hard-sigmoid.|
|`HardSwish`|—|Hard-swish.|
|`MaxPool2d`|`MaxPool`|MaxPool 2D.|
|`AvgPool2d`|`AvgPool`|AvgPool 2D.|
|`AdaptiveAvgPool2d`|`AdaptiveAvgPool`|Adaptive AvgPool 2D.|
|`GlobalAvgPool2d`|`GlobalAvgPool`|Global AvgPool 2D.|
|`MaxPool1d`|—|MaxPool 1D.|
|`AvgPool1d`|—|AvgPool 1D.|
|`TokenMeanPool`|—|Pooling “mean” sur tokens.|
|`Dropout`|—|Dropout.|
|`Dropout2d`|—|Dropout 2D.|
|`AlphaDropout`|—|AlphaDropout.|
|`Flatten`|—|Flatten.|
|`Reshape`|—|Reshape.|
|`Transpose`|—|Transpose.|
|`Permute`|—|Permute dimensions.|
|`Squeeze`|—|Squeeze dimension(s).|
|`Unsqueeze`|—|Ajoute une dimension.|
|`View`|—|Alias reshape/view.|
|`Add`|—|Addition élément-par-élément (souvent résiduel).|
|`Subtract`|—|Soustraction élément-par-élément.|
|`Multiply`|—|Multiplication élément-par-élément.|
|`Divide`|—|Division élément-par-élément.|
|`Reparameterize`|—|Réparamétrisation (VAE).|
|`Concat`|`Concatenate`, `Cat`|Concaténation.|
|`Split`|—|Split.|
|`Chunk`|—|Chunk (split en N chunks).|
|`Stack`|—|Stack.|
|`MatMul`|—|MatMul.|
|`BatchMatMul`|—|Batch MatMul.|
|`SelfAttention`|—|Self-attention.|
|`MultiHeadAttention`|—|Multi-head attention.|
|`CrossAttention`|—|Cross-attention.|
|`UpsampleNearest`|—|Upsample nearest.|
|`UpsampleBilinear`|—|Upsample bilinear.|
|`UpsampleBicubic`|—|Upsample bicubic.|
|`PixelShuffle`|—|Pixel shuffle.|
|`LSTM`|—|LSTM.|
|`GRU`|—|GRU.|
|`RNN`|—|RNN.|
|`ZeroPad2d`|—|Padding zero 2D.|
|`ReflectionPad2d`|—|Padding reflection 2D.|
|`ReplicationPad2d`|—|Padding replication 2D.|
|`Identity`|—|Identity.|
|`Lambda`|—|Lambda (custom).|
|`PatchEmbed`|`PatchProjection`|Patch embedding/projection.|
