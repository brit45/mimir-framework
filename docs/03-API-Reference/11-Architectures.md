# API : `Mimir.Architectures`

Le registre d’architectures est la manière recommandée de créer des modèles.

Source : `src/Models/Registry/ModelArchitectures.cpp` et bindings `src/LuaScripting.cpp`.

Ce module fournit :

- une liste des architectures disponibles
- des configs par défaut (faciles à surcharger)
- une normalisation de noms (alias/rétrocompat)
- une construction “safe” de modèles (avec métadonnées standardisées)

## `available() -> table<string> | (nil, err)`

Retourne la liste triée des architectures connues.

Notes :

- La liste provient du registre C++ (`ModelArchitectures::available()`), qui construit la liste à partir des entrées enregistrées.
- Les noms retournés sont les **noms canoniques** (après normalisation), triés.

## `default_config(name: string) -> table | (nil, err)`

Retourne la config par défaut (JSON -> table Lua).

Comportement :

- `name` passe par une phase de **canonicalisation** (voir section Alias).
- si l’architecture est inconnue : erreur.
- la config retournée sert de base : on peut ensuite fournir des overrides à la création.

## `info(name?: string) -> ArchitectureInfo[] | ArchitectureInfo | (nil, err)`

Lit **toutes les infos** du registry pour une (ou toutes les) architecture(s).

- Sans argument : renvoie la **liste complète** des entrées (`ArchitectureInfo[]`).
- Avec un `name` : renvoie l’entrée correspondante (`ArchitectureInfo`), ou `(nil, err)` si inconnue.

Chaque entrée est une table :

| Champ | Type | Description |
| --- | --- | --- |
| `name` | `string` | Nom canonique (clé du registry) |
| `description` | `string` | Description courte (peut être vide) |
| `config` | `table` | Config par défaut complète (peut contenir un champ `dtype`) |

Notes :

- C’est le seul accesseur qui expose le champ `description` du registry C++.
- `config` est identique à ce que renvoie `default_config(name)`.

Exemple :

```lua
for _, entry in ipairs(Mimir.Architectures.info()) do
  print(entry.name, entry.description)
end
```

## `dtypes() -> DTypeInfo[]`

Liste les **dtypes pris en charge** par le framework (source : `src/DType.hpp`).

Chaque entrée est une table :

| Champ | Type | Description |
| --- | --- | --- |
| `name` | `string` | Nom canonique (ex: `float32`, `bfloat16`) |
| `aliases` | `string` | Alias acceptés, séparés par des virgules (ex: `f32, float32`) |
| `bytes` | `integer` | Taille en octets d’un élément |
| `kind` | `string` | Famille : `float` / `int` / `uint` / `bool` |

Le **dtype par défaut** d’une architecture est le champ `config.dtype` s’il est présent,
sinon `float32` (défaut global du modèle). On peut le changer via `Mimir.model.dtype(name)`
ou via un override `dtype=...` à la création.

Exemple :

```lua
for _, dt in ipairs(Mimir.Architectures.dtypes()) do
  print(dt.name, dt.bytes, dt.kind, dt.aliases)
end
```

> Astuce : le script `scripts/tools/inspect_architectures.lua` affiche tout cela
> sous forme de tableaux colorés (`-a` pour archis + dtypes, `-l <arch> -p` pour
> les paramètres d’une archi, `-d` pour les dtypes seuls).

## Création de modèles

> **IMPORTANT:** `Mimir.Architectures.create()` n'existe PAS. Utilisez `Mimir.Model.create(name, cfg)` à la place.

## Alias / rétrocompat

Certains noms historiques sont normalisés (ex: PonyXL). Voir `canonicalArchName` dans le code du registre.

Canonicalisation observée dans le registre C++ :

- anciens noms PonyXL (ex: `ponyxl_ddpm`, `t2i_autoencoder`, `ponyxl_sdxl_stub`, `ponyxl_sdxl_unet2d`) -> `ponyxl_sdxl`
- variantes conviviales `SD3.5` / `sd3.5` / `SD3_5` -> `sd3_5`

Recommandation : utilisez les noms canoniques pour les configs et la sérialisation (ils sont stables), et ne comptez sur les alias que pour charger des anciens scripts.

---

## Architectures disponibles

### Modèles MLP

**`basic_mlp`** — MLP simple

Config par défaut :
```json
{
  "input_dim": 256,
  "hidden_dim": 256,
  "output_dim": 256,
  "hidden_layers": 2,
  "dropout": 0.0
}
```

### Modèles NLP (Transformers)

**`transformer`** — Transformer encodeur/décodeur

```json
{
  "seq_len": 64, "d_model": 128, "vocab_size": 4096,
  "num_layers": 4, "num_heads": 4, "mlp_hidden": 256,
  "output_dim": 256, "causal": false
}
```

**`vae_text`** — VAE texte latent

```json
{
  "vocab_size": 32000, "seq_len": 256,
  "d_model": 256, "num_layers": 4,
  "latent_tokens": 32, "stochastic_latent": true
}
```
Sortie : `[logits(seq*vocab) || mu(latent_dim) || logvar(latent_dim) || img_proj || text_proj]`

### Modèles Vision

**`unet`** — U-Net image-vers-image

```json
{ "image_w": 64, "image_h": 64, "image_c": 3, "base_channels": 32, "depth": 3 }
```

**`vae`** — VAE standard (MLP latent)

```json
{ "image_w": 64, "image_h": 64, "image_c": 3, "latent_dim": 128, "hidden_dim": 1024 }
```

**`vae_conv`** — VAE convolutionnel (recommandé pour les images)

```json
{
  "image_w": 64, "image_h": 64, "image_c": 3,
  "latent_h": 16, "latent_w": 16, "latent_c": 256,
  "base_channels": 64, "stochastic_latent": false,
  "use_attention": true, "enc_norm": "groupnorm",
  "enc_gn_groups": 32, "attn_heads": 4,
  "text_cond": false
}
```

**`vit`** — Vision Transformer (ViT)

```json
{
  "num_tokens": 197, "d_model": 128, "num_layers": 4,
  "num_heads": 4, "mlp_hidden": 256, "output_dim": 1000
}
```

**`resnet`** — ResNet-18 like  
**`vgg16`** / **`vgg19`** — VGG classique  
**`mobilenet`** — MobileNet v1 like

### Discriminateurs

**`gan_latent`** — Discriminateur MLP latent  
**`patch_discriminator`** — PatchGAN discriminator

### Modèles Diffusion

**`diffusion`** — eps-predictor basique

```json
{
  "image_w": 64, "image_h": 64, "image_c": 3,
  "time_dim": 128, "hidden_dim": 2048, "dropout": 0.0
}
```

**`cond_diffusion`** — Diffusion conditionnée (prompt embedding)

```json
{
  "prompt_dim": 128, "latent_w": 32, "latent_h": 32, "latent_c": 4,
  "time_dim": 128, "hidden_dim": 2048
}
```

**`ponyxl_sdxl`** / `ponyxl_ddpm` (alias) — SDXL-like complet 🔥

Config principale (extraits) :
```json
{
  "d_model": 256, "max_vocab": 32000, "text_ctx_len": 1300,
  "latent_seq_len": 4096, "latent_in_dim": 64,
  "num_heads": 8, "unet_layers": 16, "text_layers": 8,
  "image_w": 64, "image_h": 64, "image_c": 3,
  "ddpm_steps": 1000, "ddpm_beta_start": 1e-4, "ddpm_beta_end": 0.02,
  "peltier_noise": true, "peltier_mix": 0.65,
  "caption_structured_enable": true,
  "vae_arch": "vae_conv", "vae_scale": 1.0,
  "cfg_dropout_prob": 0.10
}
```

Méthodes spécifiques :
```lua
-- Entraînement
local stats = model:train_step_sdxl_latent_diffusion(prompt, rgb, w, h, opt, lr)
-- stats: loss, timestep, kl_divergence, wasserstein, entropy_diff, ...

-- Validation
local val = model:validate_step_sdxl_latent_diffusion(prompt, wrong_prompt, rgb, w, h)
-- val: eps_mse, x0_mse, img_mse, eps_mse_wrong, assoc_margin

-- Génération
local img = model:text2img_sdxl_latent_diffusion(prompt, seed, steps, guidance_scale)
```

**`sd3_5`** — Placeholder SD3.5 (stub)

```json
{ "stub_only": false, "q_len": 32, "kv_len": 32, "d_model": 64, "num_heads": 4, "num_layers": 2 }
```

### Modèles HuggingFace / SDXL (checkpoints externes)

Ces architectures sont conçues pour charger/refléter des checkpoints HuggingFace/PyTorch
(SDXL notamment). Utilisez-les avec un mapping JSON et `Mimir.Serialization.load(...)`.
Voir les scripts d'exemple dans `scripts/inferences/`.

**`external_safetensors_base`** — Base non-exécutable qui reflète exactement les clés d'un
checkpoint safetensors externe (utile pour inspecter/mapper un fichier source).

```json
{ "source_safetensors": "", "max_tensors": 0 }
```

Champs optionnels supplémentaires : `include_prefixes` / `exclude_prefixes`
(listes de préfixes pour filtrer les tenseurs créés).

**`hf_clip_text_encoder_1`** — Encodeur texte CLIP/SDXL exécutable
(`conditioner.embedders.0`).

```json
{
  "vocab_size": 49408, "padding_idx": 0, "seq_len": 77, "d_model": 768,
  "num_layers": 12, "num_heads": 12, "mlp_hidden": 3072, "causal": true
}
```

**`hf_clip_text_encoder_2`** — Encodeur texte OpenCLIP/SDXL exécutable
(`conditioner.embedders.1`), avec projection et logit scale optionnels.

```json
{
  "vocab_size": 49408, "padding_idx": 0, "seq_len": 77, "d_model": 1280,
  "num_layers": 32, "num_heads": 20, "mlp_hidden": 5120,
  "proj_dim": 1280, "causal": true, "include_logit_scale": true
}
```

**`hf_sdxl_transformer_block`** — Bloc transformer SDXL exécutable
(SelfAttention + CrossAttention + FeedForward).

```json
{
  "q_len": 64, "kv_len": 77, "d_model": 640, "context_dim": 2048,
  "num_heads": 10, "ff_hidden": 2560,
  "self_attn_qkv_bias": false, "self_attn_out_bias": true, "cross_attn_out_bias": true
}
```

**`hf_vae_decoder`** — Décodeur VAE SDXL exécutable
(`first_stage_model.decoder`).

```json
{
  "image_w": 512, "image_h": 512, "image_c": 3,
  "latent_w": 64, "latent_h": 64, "latent_c": 4,
  "num_heads": 1, "norm_groups": 32
}
```

> Les valeurs ci-dessus sont indicatives : utilisez
> `Mimir.Architectures.default_config(name)` (ou `info(name)`) pour obtenir la config
> exacte à jour côté runtime.

