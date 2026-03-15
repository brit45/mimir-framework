# Workflow modèle (lifecycle)

Cette page explique **l’ordre des appels** et surtout **pourquoi** cet ordre existe.

Si tu suis ce lifecycle, tu évites 90% des “ça crash / ça renvoie vide / ça explose en mémoire”.

## Vue d’ensemble (ordre recommandé)

| Étape | Appel | Ce que ça fait | Quand |
| ---: | --- | --- | --- |
| 1 | `Architectures.default_config` | récupère une config canonique | toujours |
| 2 | `Model.create` | fixe type+cfg dans le contexte runtime | toujours |
| 3 | `Model.build` | construit la structure/layers | toujours |
| 4 | `Model.allocate_params` | alloue les poids | avant init/load |
| 5a | `Model.init_weights` | init aléatoire | entraînement from scratch |
| 5b | `Serialization.load` | charge un checkpoint | reprise/inférence |
| 6 | `Model.forward` | exécute | entraînement/inférence |

## 1) Récupérer une config (et la surcharger)

```lua
local cfg, err = Mimir.Architectures.default_config("transformer")
assert(cfg, err)

cfg.seq_len = 128
cfg.vocab_size = 8000
```

Conseil : évite de fabriquer une config “from scratch” : tu risques d’oublier des champs attendus.

## 2) Create + build (structure)

```lua
assert(Mimir.Model.create("transformer", cfg))
assert(Mimir.Model.build())
```

Important : `build()` reconstruit la structure à partir de `ctx.modelType` + `ctx.modelConfig`.

Règle simple : décide de ta config, puis appelle `create()` et seulement ensuite `build()`.

## Types de layers gérés (table rapide)

Le runtime exécute une liste de `Layer` (C++) dont le champ `type` est normalisé via `LayerRegistry::normalize_type` (ex: `Relu` → `ReLU`, `ConvTranspose` → `ConvTranspose2d`).

Cette table liste les types **les plus fréquents**. Pour la liste complète (avec alias), voir:

- [docs/03-API-Reference/01-Layers.md](../03-API-Reference/01-Layers.md)
- `src/LayerTypes.hpp`

| Type (canonique) | Catégorie | Rôle (résumé) | Champs/config typiques |
| --- | --- | --- | --- |
| `Conv2d` | Conv | Convolution 2D (vision / U-Net / feature extractor). | `in_channels`, `out_channels`, `kernel_size` (ou `kernel_h/kernel_w`), `stride`, `padding`, `dilation`, `groups` |
| `ConvTranspose2d` | Conv | Convolution transposée 2D (upsampling). | idem `Conv2d` |
| `Linear` | Dense | Projection dense (GEMM). | `in_features`, `out_features`, `use_bias` |
| `Embedding` | NLP | IDs int → vecteurs (chemin “int path”). | `vocab_size`, `embed_dim`, `padding_idx` |
| `LayerNorm` | Norm | Normalisation sur la dernière dimension (Transformer). | `eps` |
| `GroupNorm` | Norm | Normalisation groupée (vision). | `num_groups`, `eps` |
| `RMSNorm` | Norm | Variante de LayerNorm (RMS-only). | `eps` |
| `ReLU` | Activation | Non-linéarité ReLU. | — |
| `GELU` | Activation | Activation GELU (Transformer). | — |
| `SiLU` | Activation | SiLU/Swish (vision/diffusion). | — |
| `Softmax` | Activation | Softmax (souvent sur logits). | `axis` |
| `LogSoftmax` | Activation | Log-Softmax (souvent pour NLL). | `axis` |
| `Dropout` | Reg | Dropout (entraînement). | `dropout_p` |
| `Add` | Elemwise | Addition élément-par-élément (résiduel). | multi-input via `inputs`, `output` |
| `MatMul` | Tensor op | Matmul (produit matriciel). | dépend des shapes |
| `Concat` | Tensor op | Concaténation le long d’un axe. | `concat_axis` |
| `Reshape` / `View` | Shape | Refaçonne les dimensions sans changer le contenu. | `target_shape` |
| `SelfAttention` | Attention | Self-attention (expérimental/couteux selon build). | `seq_len`, `num_heads`, `head_dim`, `causal` |
| `MultiHeadAttention` | Attention | Variante multi-têtes. | idem |
| `CrossAttention` | Attention | Attention query↔context (diffusion/encoder-decoder). | idem |
| `TokenMeanPool` | Pooling | Moyenne sur tokens (résumé séquence). | `axis` (selon implémentation) |
| `Identity` | Special | No-op (debug / placeholder). | — |
| `PatchEmbed` | Custom | Projection learnable de patches (vision/diffusion). | `patch_dim`, `num_patches`, etc. |

### Table complète (tous les `LayerType`)

Notes:

- Cette liste reflète l’enum `LayerType` (C++). Le runtime peut normaliser des alias (colonne “Alias”).
- Le fait qu’un type existe ne garantit pas que son backward soit complet (certaines ops sont best-effort). Pour le statut et les champs fréquents, voir `docs/03-API-Reference/01-Layers.md`.

| Type (canonique) | Alias normalisés (si présents) | Catégorie | Description rapide |
| --- | --- | --- | --- |
| `Conv2d` | — | Conv | Convolution 2D. |
| `ConvTranspose2d` | `ConvTranspose`, `Deconv2d` | Conv | Convolution transposée 2D (upsampling). |
| `Conv1d` | — | Conv | Convolution 1D. |
| `DepthwiseConv2d` | — | Conv | Convolution depthwise 2D. |
| `Linear` | — | Dense | Projection dense (GEMM). |
| `Bilinear` | — | Dense | Transformation bilinéaire. |
| `Embedding` | — | Embedding | IDs int → vecteurs. |
| `EmbeddingBag` | — | Embedding | Embedding agrégé (bag). |
| `BatchNorm2d` | `BatchNorm`, `BN`, `BN2d` | Norm | BatchNorm 2D. |
| `BatchNorm1d` | `BN1d` | Norm | BatchNorm 1D. |
| `LayerNorm` | `LN` | Norm | LayerNorm. |
| `GroupNorm` | `GN` | Norm | GroupNorm. |
| `InstanceNorm2d` | `IN` | Norm | InstanceNorm 2D. |
| `RMSNorm` | — | Norm | RMSNorm. |
| `ReLU` | `ReLu`, `Relu`, `RELU` | Activation | Activation ReLU. |
| `LeakyReLU` | — | Activation | Activation LeakyReLU. |
| `GELU` | `Gelu` | Activation | Activation GELU. |
| `SiLU` | `Swish`, `Silu`, `silu`, `swish` | Activation | Activation SiLU/Swish. |
| `Tanh` | — | Activation | Activation tanh. |
| `Sigmoid` | — | Activation | Activation sigmoid. |
| `Softmax` | — | Activation | Softmax. |
| `LogSoftmax` | — | Activation | Log-Softmax. |
| `Softplus` | — | Activation | Softplus. |
| `Mish` | — | Activation | Mish. |
| `HardSigmoid` | — | Activation | Hard-sigmoid. |
| `HardSwish` | — | Activation | Hard-swish. |
| `MaxPool2d` | `MaxPool` | Pooling | MaxPool 2D. |
| `AvgPool2d` | `AvgPool` | Pooling | AvgPool 2D. |
| `AdaptiveAvgPool2d` | `AdaptiveAvgPool` | Pooling | Adaptive AvgPool 2D. |
| `GlobalAvgPool2d` | `GlobalAvgPool` | Pooling | Global AvgPool 2D. |
| `MaxPool1d` | — | Pooling | MaxPool 1D. |
| `AvgPool1d` | — | Pooling | AvgPool 1D. |
| `TokenMeanPool` | — | Pooling | Moyenne sur tokens (résumé séquence). |
| `Dropout` | — | Regularization | Dropout. |
| `Dropout2d` | — | Regularization | Dropout 2D. |
| `AlphaDropout` | — | Regularization | AlphaDropout. |
| `Flatten` | — | Shape | Flatten. |
| `Reshape` | — | Shape | Reshape. |
| `Transpose` | — | Shape | Transpose. |
| `Permute` | — | Shape | Permute dimensions. |
| `Squeeze` | — | Shape | Squeeze dimension(s). |
| `Unsqueeze` | — | Shape | Ajoute une dimension. |
| `View` | — | Shape | Alias reshape/view. |
| `Add` | — | Elemwise | Addition élément-par-élément. |
| `Subtract` | — | Elemwise | Soustraction élément-par-élément. |
| `Multiply` | — | Elemwise | Multiplication élément-par-élément. |
| `Divide` | — | Elemwise | Division élément-par-élément. |
| `Reparameterize` | — | VAE | Réparamétrisation (VAE). |
| `Concat` | `Concatenate`, `Cat` | Tensor op | Concaténation. |
| `Split` | — | Tensor op | Split. |
| `Chunk` | — | Tensor op | Chunk (split en N chunks). |
| `Stack` | — | Tensor op | Stack. |
| `MatMul` | — | Tensor op | MatMul. |
| `BatchMatMul` | — | Tensor op | Batch MatMul. |
| `SelfAttention` | — | Attention | Self-attention. |
| `MultiHeadAttention` | — | Attention | Multi-head attention. |
| `CrossAttention` | — | Attention | Cross-attention. |
| `UpsampleNearest` | — | Upsample | Upsample nearest. |
| `UpsampleBilinear` | — | Upsample | Upsample bilinear. |
| `UpsampleBicubic` | — | Upsample | Upsample bicubic. |
| `PixelShuffle` | — | Upsample | Pixel shuffle. |
| `LSTM` | — | Recurrent | LSTM. |
| `GRU` | — | Recurrent | GRU. |
| `RNN` | — | Recurrent | RNN. |
| `ZeroPad2d` | — | Padding | Padding zero 2D. |
| `ReflectionPad2d` | — | Padding | Padding reflection 2D. |
| `ReplicationPad2d` | — | Padding | Padding replication 2D. |
| `Identity` | — | Special | Identity (no-op). |
| `Lambda` | — | Special | Layer custom/placeholder (best-effort). |
| `PatchEmbed` | `PatchProjection` | Custom | Patch embedding / projection learnable. |

## 3) Allouer les paramètres (poids)

```lua
local ok, total_or_err = Mimir.Model.allocate_params()
assert(ok ~= false, total_or_err)
```

Pourquoi c’est séparé : tu peux construire la structure et choisir ensuite comment/si tu alloues (utile pour debug / estimation mémoire).

## 4) Initialiser OU charger un checkpoint

### Option A : init from scratch

```lua
assert(Mimir.Model.init_weights("xavier", 42))
```

### Option B : charger

```lua
local ok, err = Mimir.Serialization.load("checkpoint/my_model.safetensors")
assert(ok ~= false, err)
```

Conseil : la config doit matcher (ex: `seq_len`, `vocab_size`, dims). Sinon tu auras un mismatch de shapes.

## 5) Forward (inputs)

Recommandation : utilise une table map, même pour un seul input :

```lua
local out = Mimir.Model.forward({ __input__ = {1,1,1,1} }, false)
assert(out)
```

Pourquoi : ça rend les scripts compatibles multi-input (ex: texte+image) sans refactor.

## 6) Backward / optimizer (niveau bas)

Si tu construis ta boucle à la main :

- `Mimir.Model.zero_grads()`
- `Mimir.Model.forward(...)`
- construire `grad_out` (gradient par rapport à la sortie)
- `Mimir.Model.backward(grad_out)`
- `Mimir.Model.optimizer_step()`

Note : selon les layers, tous les backward ne sont pas au même niveau de maturité (donc commence par des tests unitaires courts).
