# VAEConv : architecture, configuration et entraînement

Comprendre le graphe réel de `vae_conv`, choisir une configuration cohérente et éviter les erreurs de dimensions ou de contrat latent.

**Public concerné :** Utilisateur qui veut construire, entraîner, inspecter ou réutiliser le VAE convolutionnel de Mímir.

> **Prérequis**
>
> Avoir compilé Mímir et connaître le cycle `create → allocate_params → init_weights ou load`.

## Sur cette page

- [Sources de vérité](#sources-de-vérité)
- [1. Graphe réel](#1-graphe-réel)
- [2. Réparamétrisation et mode déterministe](#2-réparamétrisation-et-mode-déterministe)
- [3. Le prior appris](#3-le-prior-appris)
- [4. Contraintes de formes](#4-contraintes-de-formes)
- [5. Paramètres importants](#5-paramètres-importants)
- [6. Exemple : inspecter la configuration sans dataset](#6-exemple-inspecter-la-configuration-sans-dataset)
- [7. Exemple : décoder un latent](#7-exemple-décoder-un-latent)
- [8. Entraîner avec le script officiel](#8-entraîner-avec-le-script-officiel)
- [9. Vérifications après modification](#9-vérifications-après-modification)
- [10. Diagnostic rapide](#10-diagnostic-rapide)
- [Étapes suivantes](#étapes-suivantes)

## Sources de vérité

- configuration publique : `src/Models/Vision/VAEConvModel.hpp`;
- construction du graphe : `src/Models/Vision/VAEConvModel.cpp`;
- configuration du registre : `src/Models/Registry/ModelArchitectures.cpp`;
- loss VAE et backward : `src/Model.cpp`;
- boucle Lua haut niveau : `src/scriptings/Lua/luaScripting/LuaScriptingModelAndRegistry.cpp`;
- script fourni : `scripts/training/train_vae_conv.lua`.

## 1. Graphe réel

Pour une image `H × W × C` et un latent `LH × LW × LC` :

```text
image HWC
  → Reshape/Permute vers CHW
  → Conv2d C→base
  → [ResBlock et SelfAttention optionnels]
  → N convolutions stride 2
  → Conv2d 1×1 vers mu     (LC×LH×LW)
  → Conv2d 1×1 vers logvar (LC×LH×LW)
  → Reparameterize : z = mu + exp(0.5×logvar)×epsilon
  → Add(z, z_prior_bias) si use_encoder_prior=true
  → Conv2d LC→base
  → N upsamplings
  → Conv2d base→C
  → Tanh
  → Permute vers HWC
```

La sortie principale `x` est un vecteur concaténé :

```text
reconstruction[image_dim] || mu[latent_dim] || logvar[latent_dim]
```

Si `text_cond=true`, deux projections sont ajoutées :

```text
... || img_proj[proj_dim] || txt_proj[proj_dim]
```

Le latent opérationnel `z` sert au décodeur. Il n’est pas placé dans la zone `mu` de la sortie : la loss KL doit être calculée sur les paramètres de la distribution, pas sur un échantillon.

## 2. Réparamétrisation et mode déterministe

En entraînement, si `stochastic_latent=true` :

\[
z = \mu + \exp(0.5\,\mathrm{logvar})\,\epsilon,\qquad
\epsilon \sim \mathcal{N}(0, I)
\]

Le runtime borne `logvar` dans `[-20, 20]` pour éviter les exponentielles non finies.

Si `stochastic_latent=false`, ou pendant un forward d’inférence, `z=mu`. Ce mode est utile pour produire un encodage reproductible, mais il transforme le chemin de reconstruction en autoencodeur déterministe. Si une KL non nulle est utilisée pendant l’entraînement, le script officiel active par défaut le mode stochastique pour les nouveaux runs.

La KL utilisée par `trainStepVAE` est :

\[
\mathrm{KL}(q(z|x)\,\|\,\mathcal{N}(0,I))
= \frac{1}{2D}\sum_i
\left(\mu_i^2+\exp(\mathrm{logvar}_i)-1-\mathrm{logvar}_i\right)
\]

où `D = latent_dim`.

## 3. Le prior appris

Avec `use_encoder_prior=true`, le modèle crée `vae_conv/z_prior_bias`, de taille `latent_dim`, puis calcule :

\[
z_{\mathrm{decoder}} = z + b_{\mathrm{prior}}
\]

Cette couche est une `Constant` marquée `trainable_parameter=true`. Elle :

- ne possède aucune entrée ;
- reçoit le gradient produit par les features du décodeur ;
- est mise à jour par SGD, Adam ou AdamW comme les autres blocs de poids ;
- est sérialisée dans le checkpoint.

Les autres couches `Constant` restent fixes par défaut. Cette distinction évite notamment d’entraîner les tenseurs zéro utilisés par le décodeur autonome pour remplacer des skips absents.

Le prior est un biais latent global appris. Ce n’est pas un prior conditionnel dépendant de l’image ou du texte.

## 4. Contraintes de formes

Le builder accepte uniquement des réductions identiques sur les deux axes, obtenues par divisions successives par deux.

Valide :

```text
64×64 → 16×16   (deux downsamples)
512×512 → 64×64 (trois downsamples)
```

Non valide :

```text
64×32 → 16×16   (facteurs 4 et 2 différents)
60×60 → 16×16   (16 n’est pas atteint par /2)
```

Vérifie également la taille du latent :

```text
image_dim  = image_w × image_h × image_c
latent_dim = latent_w × latent_h × latent_c
```

Un latent peut être spatialement plus petit tout en contenant davantage de scalaires que l’image si `latent_c` est élevé. Ce n’est pas une erreur de shape, mais ce n’est alors pas un bottleneck de compression.

## 5. Paramètres importants

| Paramètre | Défaut C++ | Effet réel |
| --- | ---: | --- |
| `image_w`, `image_h`, `image_c` | `64`, `64`, `3` | Forme HWC de l’image. |
| `latent_w`, `latent_h`, `latent_c` | `16`, `16`, `256` | Forme CHW du latent. |
| `base_channels` | `64` | Nombre de canaux dans le corps encodeur/décodeur. |
| `stochastic_latent` | `false` | Bruit de réparamétrisation pendant l’entraînement. |
| `use_attention` | `true` | Nom historique : active les ResBlocks, pas la SelfAttention. |
| `use_attn` | `false` | Active la SelfAttention spatiale. |
| `resnet_max_tokens` | `0` | Limite `H×W` des ResBlocks ; `0` signifie sans limite. |
| `attn_max_tokens` | `0` | Limite `H×W` de l’attention ; `0` signifie sans limite. |
| `enc_norm`, `dec_norm` | `groupnorm` | `none`, `groupnorm`/`gn`, `layernorm`/`ln`. |
| `decoder_upsample` | `conv_transpose` | `conv_transpose` ou `nearest_conv`. |
| `use_skip_connections` | `false` | Skips encodeur-décodeur par concaténation puis Conv 1×1. |
| `use_encoder_prior` | `false` | Ajoute le biais latent global appris. |
| `text_cond` | `false` | Ajoute la branche texte et les deux projections. |

Attention : le coût mémoire de la SelfAttention croît quadratiquement avec le nombre de tokens `H×W`. Pour une carte `64×64`, elle traite 4096 tokens. Utilise `attn_max_tokens` pour la limiter au bottleneck.

## 6. Exemple : inspecter la configuration sans dataset

Le dépôt fournit `scripts/examples/inspect_vae_conv.lua`. Sa structure essentielle est :

```lua
local cfg, err = Mimir.Architectures.default_config("vae_conv")
assert(cfg, err)

cfg.image_w = 64
cfg.image_h = 64
cfg.latent_w = 16
cfg.latent_h = 16
cfg.latent_c = 32
cfg.base_channels = 32
cfg.stochastic_latent = true
cfg.use_attention = true   -- ResBlocks, nom historique
cfg.use_attn = false
cfg.use_encoder_prior = true

assert(Mimir.Model.create("vae_conv", cfg))
local ok, count = Mimir.Model.allocate_params()
assert(ok, count)
assert(Mimir.Model.init_weights("xavier", 1234))

print("parameters:", Mimir.Model.total_params())
for _, layer in ipairs(Mimir.Model.get_layers()) do
  print(layer.index, layer.name, layer.type, layer.param_count)
end
```

Puis exécute :

```bash
./bin/mimir --lua scripts/examples/inspect_vae_conv.lua
```

Ce test construit et alloue le modèle, mais ne charge aucune donnée.

## 7. Exemple : décoder un latent

`vae_conv_decode` reconstruit uniquement le décodeur. Son entrée est un vecteur CHW aplati de taille `latent_dim`, sa sortie une image HWC aplatie de taille `image_dim`.

```lua
local cfg = assert(Mimir.Architectures.default_config("vae_conv_decode"))
cfg.image_w, cfg.image_h, cfg.image_c = 64, 64, 3
cfg.latent_w, cfg.latent_h, cfg.latent_c = 16, 16, 32
cfg.enc_norm, cfg.dec_norm = "groupnorm", "groupnorm"

assert(Mimir.Model.create("vae_conv_decode", cfg))
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier", 1234))

local latent = {}
for i = 1, cfg.latent_w * cfg.latent_h * cfg.latent_c do
  latent[i] = 0.0
end

local rgb, err = Mimir.Model.forward({__input__ = latent}, false)
assert(rgb, err)
assert(#rgb == cfg.image_w * cfg.image_h * cfg.image_c)
```

Un décodeur initialisé aléatoirement ne produit pas une image utile. Pour une reconstruction réelle, charge les poids du décodeur provenant du checkpoint VAE compatible.

## 8. Entraîner avec le script officiel

Exemple CPU volontairement réduit :

```bash
OMP_NUM_THREADS=8 ./bin/mimir \
  --lua scripts/training/train_vae_conv.lua -- \
  --dataset-root /chemin/vers/images \
  --out-dir checkpoint/vae_conv_demo \
  --image-w 64 --image-h 64 \
  --latent-w 16 --latent-h 16 --latent-c 16 \
  --base-channels 32 \
  --epochs 2 --lr 3e-5 \
  --stochastic-latent true \
  --encoder-prior true \
  --cpu-only true \
  --attn false
```

Les images doivent être normalisées selon le chemin de chargement du script ; la dernière activation est `Tanh`, donc la reconstruction modèle est dans `[-1,1]`.

Pour reprendre un checkpoint, conserve la même topologie :

```bash
./bin/mimir --lua scripts/training/train_vae_conv.lua -- \
  --dataset-root /chemin/vers/images \
  --out-dir checkpoint/vae_conv_demo \
  --resume true
```

Ne change pas entre deux reprises les options qui ajoutent ou retirent des couches : normalisation, skips, prior, ResBlocks, attention, dimensions ou mode d’upsampling.

## 9. Vérifications après modification

Le test de contrat dédié est :

```bash
cmake --build build --target ModelTest.VAEConvContract
ctest --test-dir build --output-on-failure -R ModelTest.VAEConvContract
```

Il vérifie :

- que la sortie packée contient `mu` ;
- qu’une `Constant` fixe ne devient pas implicitement apprenable ;
- que `z_prior_bias` reçoit un gradient via les features du décodeur ;
- qu’un optimizer step modifie réellement un paramètre apprenable.

Pour les briques mathématiques adjacentes :

```bash
ctest --test-dir build --output-on-failure \
  -R 'AutogradTest.Numerical|RuntimeTest.MathConv2d|RuntimeTest.MathNorms|RuntimeTest.MathAttention'
```

## 10. Diagnostic rapide

| Symptôme | Vérification |
| --- | --- |
| Erreur « cannot reach latent » | Les deux ratios image/latent doivent être la même puissance de deux. |
| OOM avec attention | Réduire `attn_max_tokens`, `latent_h/w`, `base_channels` ou désactiver `use_attn`. |
| KL instable | Vérifier `stochastic_latent`, `kl_beta`, le warmup KL et les bornes `logvar`. |
| Checkpoint incompatible | Comparer dimensions, normes, skips, prior, ResBlocks, attention et upsampling. |
| Prior inchangé | Vérifier que le checkpoint contient `vae_conv/z_prior_bias` et que le layer porte `trainable_parameter=true`. |
| Reconstruction trop lisse | Examiner capacité du latent, poids KL, normalisation et losses additionnelles avant d’augmenter le modèle. |

## Étapes suivantes

- [Page précédente : Tutoriel : diffusion (SD3.5 / autoencoder)](13-Diffusion.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : MPK : packages d’architecture](15-MPK.md)
