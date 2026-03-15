# Architectures Prédéfinies

**Version:** 2.3.0

Cette page décrit les architectures **built-in** réellement disponibles via le registre C++.

## ✅ Point clé (v2.3+)

- `Mimir.Architectures` **ne** fournit pas de builders `Mimir.Architectures.<name>(cfg)`.
- Il expose uniquement :
    - `Mimir.Architectures.available()`
    - `Mimir.Architectures.default_config(name)`
- La création se fait via `Mimir.Model.create(name, cfg)`.

## 📋 Liste des architectures disponibles

Source de vérité : registre C++ (`ModelArchitectures`).

- `basic_mlp`
- `t2i_autoencoder`
- `transformer`
- `vit`
- `vae`
- `resnet`
- `unet`
- `mobilenet`
- `vgg16`
- `vgg19`
- `diffusion`

Vérification côté Lua :

```lua
local names, err = Mimir.Architectures.available()
if not names then error(err) end
for _, n in ipairs(names) do
    print("arch:", n)
end
```

## 🧩 Pattern recommandé

```lua
-- 1) Récupérer une config par défaut
local cfg, err = Mimir.Architectures.default_config("transformer")
if not cfg then error(err) end

-- 2) Ajuster (optionnel)
cfg.vocab_size = 10000
cfg.d_model = 512
cfg.num_layers = 6
cfg.num_heads = 8

-- 3) Créer le modèle
local ok, create_err = Mimir.Model.create("transformer", cfg)
if not ok then error(create_err) end

-- 4) Allouer + init
local ok_alloc, params = Mimir.Model.allocate_params()
if not ok_alloc then error("allocation failed") end
Mimir.Model.init_weights("xavier", 42)
print("params:", params)
```

## 🔍 Inspecter la configuration exacte

Pour éviter toute divergence entre doc et code, l’approche la plus fiable consiste à exporter la config par défaut :

```lua
local cfg = assert(Mimir.Architectures.default_config("unet"))
write_json("unet_default_config.json", cfg)
```

## 📚 Exemples (scripts du repo)

Les scripts suivants sont alignés avec l’implémentation (à utiliser comme référence) :

> Astuce: plusieurs scripts font `require("arch")` (fichier `scripts/modules/arch.lua`).
> Si vous voyez `module 'arch' not found`, lancez avec :
> `LUA_PATH='scripts/modules/?.lua;;' ./bin/mimir --lua <script.lua>`

- `transformer` : `scripts/examples/example_gpt.lua`, `scripts/examples/example_simple.lua`
- `unet` : `scripts/demos/demo_unet.lua`
- `vae` : `scripts/demos/demo_vae.lua`
- `vit` : `scripts/demos/demo_vit.lua`
- `resnet` : `scripts/demos/demo_resnet.lua`
- `mobilenet` : `scripts/demos/demo_mobilenet.lua`
- `diffusion` : `scripts/demos/demo_diffusion.lua`
- `basic_mlp` : `scripts/examples/train_basic_mlp.lua`
- `t2i_autoencoder` : `scripts/examples/t2i_autoencoder_generate.lua`

<!--
LEGACY (obsolète) : l'ancien contenu ci-dessous est conservé pour historique.
Il utilise des APIs qui ne correspondent plus à l'implémentation v2.3.

# Architectures Prédéfinies

Guide des architectures **built-in** disponibles dans le registre C++ de Mímir (v2.3.0).

> En v2.3, `Mimir.Architectures` ne construit pas directement des modèles.
> Il expose uniquement le registre : `available()` et `default_config(name)`.
> La création se fait via `Mimir.Model.create(name, cfg)`.

---

## 📋 Table des Matières

- [Vue d'Ensemble](#vue-densemble)
- [UNet](#unet)
- [VAE (Variational Autoencoder)](#vae-variational-autoencoder)
- [ViT (Vision Transformer)](#vit-vision-transformer)
- [GAN (Generative Adversarial Network)](#gan-generative-adversarial-network)
- [Diffusion Model](#diffusion-model)
- [Transformer](#transformer)
- [ResNet](#resnet)
- [MobileNet](#mobilenet)
- [Comparaison](#comparaison)

---

## 🎯 Vue d'Ensemble

Mímir propose **11 architectures prédéfinies** accessibles via le registre :

- `basic_mlp`
- `t2i_autoencoder`
- `transformer`
- `vit`
- `vae`
- `resnet`
- `unet`
- `mobilenet`
- `vgg16`
- `vgg19`
- `diffusion`

### Pattern recommandé (source de vérité)

```lua
-- 1) Découvrir la liste exacte
local names, err = Mimir.Architectures.available()
if not names then error(err) end

-- 2) Partir d'une config par défaut, puis ajuster
local cfg, cfg_err = Mimir.Architectures.default_config("transformer")
if not cfg then error(cfg_err) end
cfg.vocab_size = 10000
cfg.d_model = 512
cfg.num_layers = 6
cfg.num_heads = 8

-- 3) Créer le modèle
local ok, create_err = Mimir.Model.create("transformer", cfg)
if not ok then error(create_err) end

-- 4) Allouer + init
local ok_alloc = Mimir.Model.allocate_params()
if not ok_alloc then error("allocation failed") end
Mimir.Model.init_weights("xavier", 42)
```

**Avantages** :
- ✅ Configuration rapide
- ✅ Hyperparamètres validés
- ✅ Best practices intégrées
- ✅ Prêt pour entraînement

---

## 🔵 UNet

### Description

**UNet** : Architecture encoder-decoder avec skip connections pour segmentation d'images.

```
Input → [Encoder] → Bottleneck → [Decoder + Skip] → Output
```

### Utilisation

```lua
local cfg, err = Mimir.Architectures.default_config("unet")
if not cfg then error(err) end

cfg.in_channels = 3
cfg.out_channels = 1

local ok, create_err = Mimir.Model.create("unet", cfg)
if not ok then error(create_err) end
assert(Mimir.Model.allocate_params())
Mimir.Model.init_weights("he", 42)
```

### Paramètres

Les clés exactes et leurs valeurs par défaut dépendent du code C++.
Pour éviter toute divergence, inspectez la config retournée par :

```lua
local cfg = assert(Mimir.Architectures.default_config("unet"))
write_json("unet_default_config.json", cfg)
```

### Architecture Résultante

```
Input: [B, 3, H, W]
  ↓ Conv(3→64) + ReLU + Conv(64→64) + ReLU
  ↓ MaxPool(2×2)
  ↓ Conv(64→128) + ReLU + Conv(128→128) + ReLU
  ↓ MaxPool(2×2)
  ↓ Conv(128→256) + ReLU + Conv(256→256) + ReLU [Bottleneck]
  ↓ UpSample(2×2) + Concat(Skip128)
  ↓ Conv(256+128→128) + ReLU + Conv(128→128) + ReLU
  ↓ UpSample(2×2) + Concat(Skip64)
  ↓ Conv(128+64→64) + ReLU + Conv(64→64) + ReLU
  ↓ Conv(64→out_channels)
Output: [B, out_channels, H, W]
```

### Cas d'Usage

- Segmentation d'images médicales
- Segmentation sémantique
- Super-résolution
- Denoising

### Exemple Complet

```lua
-- Créer UNet pour segmentation médicale
local unet = Mimir.Architectures.unet({
    in_channels = 1,      -- Images grayscale
    out_channels = 2,     -- Background + Tumor
    base_channels = 32,
    depth = 3
})

-- Configurer
model.configure(unet, {
    learning_rate = 0.001,
    optimizer = "adam",
    loss = "cross_entropy"
})

-- Charger dataset
local dataset = Mimir.Dataset.loadFromJson("medical_scans.json")

-- Entraîner
Mimir.Model.train(unet, dataset, 50)

-- Sauvegarder
Mimir.Model.save(unet, "unet_medical.json")
```

---

## 🟣 VAE (Variational Autoencoder)

### Description

**VAE** : Autoencoder probabiliste pour génération et interpolation.

```
Input → Encoder → [μ, σ] → Latent(z) → Decoder → Output
```

### Utilisation

```lua
local ok, err = Mimir.Model.create("vae", {
    input_dim = 784,                -- 28×28 images aplaties
    latent_dim = 20,                -- Dimension latente
    encoder_hidden = {256, 128},    -- MLP encoder
    decoder_hidden = {128, 256},    -- MLP decoder
    kl_beta = 1.0,                  -- Poids KL (β-VAE)
    activation = "relu",
    use_mean_in_infer = true
})
assert(ok, err)

ok, err = Mimir.Model.build()
assert(ok, err)

-- Note: `Mimir.Architectures.vae(config)` existe, mais ne lit actuellement
-- que `input_dim` et `latent_dim` (le reste reste aux valeurs par défaut).
```

### Paramètres

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `input_dim` | int | 256 | Dimension d'entrée (ex: embeddings) |
| `embed_dim` | int | — | Alias de `input_dim` (si vous réutilisez un config Transformer) |
| `latent_dim` | int | 64 | Dimension de l'espace latent |
| `encoder_hidden` | table | {512, 256} | Dimensions des couches cachées (encoder) |
| `decoder_hidden` | table | {256, 512} | Dimensions des couches cachées (decoder) |
| `kl_beta` | float | 1.0 | Poids KL divergence (β-VAE) |
| `activation` | string | "relu" | relu / gelu / tanh / sigmoid / swish(silu) |
| `use_mean_in_infer` | bool | true | Si vrai: z = μ en inférence (sinon sampling) |
| `seed` | int | 0xC0FFEE | Seed par défaut pour le sampling |

### Architecture Résultante

```
Encoder:
  Input[784] → Linear(784→256) → ReLU
             → Linear(256→128) → ReLU
             → Linear(128→latent_dim×2) [μ, log(σ²)]

Reparameterization:
  z = μ + σ × ε, où ε ~ N(0,1)

Decoder:
  z[latent_dim] → Linear(latent_dim→128) → ReLU
                → Linear(128→256) → ReLU
                → Linear(256→784) → Sigmoid
```

### Loss Function

```
L = Reconstruction_Loss + β × KL_Divergence
  = MSE(x, x̂) + β × KL(q(z|x) || p(z))
```

### Cas d'Usage

- Génération d'images
- Compression avec reconstruction
- Interpolation dans l'espace latent
- Anomaly detection

### Exemple Complet

```lua
-- VAE pour MNIST
local ok, err = Mimir.Model.create("vae", {
    input_dim = 784,
    latent_dim = 10,
    encoder_hidden = {512, 256},
    decoder_hidden = {256, 512},
    kl_beta = 1.0
})
assert(ok, err)
Mimir.Model.build()

-- Entraîner (le dataset doit être préparé via Mimir.Dataset.*)
Mimir.Model.train(100, 0.001)
```

---

## 🟢 ViT (Vision Transformer)

### Description

**ViT** : Transformers appliqués à la vision via patch embedding.

```
Image → Patches → Embedding → Transformer Blocks → Classification
```

### Utilisation

```lua
local vit = Mimir.Architectures.vit({
    image_size = 224,     -- Taille image
    patch_size = 16,      -- Taille patch (16×16)
    num_classes = 1000,   -- Classes de sortie
    d_model = 768,        -- Dimension modèle
    num_layers = 12,      -- Blocs Transformer
    num_heads = 12        -- Têtes d'attention
})
```

### Paramètres

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `image_size` | int | 224 | Taille image (carré) |
| `patch_size` | int | 16 | Taille patch (16×16 pixels) |
| `num_classes` | int | 1000 | Nombre de classes |
| `d_model` | int | 768 | Dimension embeddings |
| `num_layers` | int | 12 | Nombre de blocs Transformer |
| `num_heads` | int | 12 | Têtes d'attention par bloc |
| `mlp_ratio` | float | 4.0 | Ratio FFN (d_model × mlp_ratio) |
| `dropout` | float | 0.1 | Dropout rate |

### Architecture Résultante

```
Input: [B, 3, 224, 224]
  ↓ Patch Embedding: (224/16)² = 196 patches
  ↓ Linear Projection: [B, 196, 768]
  ↓ + Positional Encoding
  ↓ + [CLS] Token
  ↓
  ↓ Transformer Block ×12:
  │   LayerNorm → Multi-Head Attention → Residual
  │   LayerNorm → FFN (768→3072→768) → Residual
  ↓
  ↓ Extract [CLS] Token
  ↓ Linear: 768 → num_classes
Output: [B, num_classes]
```

### Variants

```lua
-- ViT-Base
local vit_base = Mimir.Architectures.vit({
    d_model = 768, num_layers = 12, num_heads = 12
})

-- ViT-Large
local vit_large = Mimir.Architectures.vit({
    d_model = 1024, num_layers = 24, num_heads = 16
})

-- ViT-Tiny (pour CPU)
local vit_tiny = Mimir.Architectures.vit({
    d_model = 192, num_layers = 12, num_heads = 3
})
```

### Cas d'Usage

- Classification d'images
- Feature extraction
- Transfer learning

### Exemple Complet

```lua
-- ViT pour ImageNet-like dataset
local vit = Mimir.Architectures.vit({
    image_size = 224,
    patch_size = 16,
    num_classes = 100,  -- Custom dataset
    d_model = 384,      -- Reduced for CPU
    num_layers = 6,
    num_heads = 6
})

-- Configuration
model.configure(vit, {
    learning_rate = 0.0003,
    optimizer = "adam",
    loss = "cross_entropy",
    batch_size = 16,
    gradient_clip = 1.0
})

-- Entraîner
Mimir.Model.train(vit, dataset, 50)
```

---

## 🔴 GAN (Generative Adversarial Network)

### Description

**GAN** : Deux réseaux adversaires (Generator + Discriminator) pour génération.

```
Noise(z) → Generator → Fake Image
Real Image / Fake Image → Discriminator → [Real/Fake]
```

### Utilisation

```lua
local gan = Mimir.Architectures.gan({
    latent_dim = 100,     -- Dimension bruit z
    image_channels = 1,   -- Grayscale
    image_size = 28,      -- 28×28
    generator_dims = {256, 512},
    discriminator_dims = {512, 256}
})
```

### Paramètres

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `latent_dim` | int | 100 | Dimension vecteur bruit z |
| `image_channels` | int | 1 | Canaux image (1=grayscale, 3=RGB) |
| `image_size` | int | 28 | Taille image (carré) |
| `generator_dims` | table | {256, 512} | Couches cachées Generator |
| `discriminator_dims` | table | {512, 256} | Couches cachées Discriminator |

### Architecture Résultante

**Generator** :
```
z[100] → Linear(100→256) → LeakyReLU → BatchNorm
       → Linear(256→512) → LeakyReLU → BatchNorm
       → Linear(512→784) → Tanh
Output: [B, 1, 28, 28]
```

**Discriminator** :
```
Image[784] → Linear(784→512) → LeakyReLU → Dropout
           → Linear(512→256) → LeakyReLU → Dropout
           → Linear(256→1) → Sigmoid
Output: [B, 1] (probabilité image réelle)
```

### Training Loop

```lua
-- GAN nécessite training custom
for epoch = 1, epochs do
    for batch in dataset do
        -- 1. Train Discriminator
        local real_images = batch.images
        local z = sample_noise(batch_size, latent_dim)
        local fake_images = Mimir.Model.forward(generator, z)
        
        local d_loss_real = discriminator_loss(discriminator, real_images, 1)
        local d_loss_fake = discriminator_loss(discriminator, fake_images, 0)
        local d_loss = d_loss_real + d_loss_fake
        
        update_discriminator(d_loss)
        
        -- 2. Train Generator
        local z = sample_noise(batch_size, latent_dim)
        local fake_images = Mimir.Model.forward(generator, z)
        local g_loss = generator_loss(discriminator, fake_images, 1)  -- Fool D
        
        update_generator(g_loss)
    end
end
```

### Cas d'Usage

- Génération d'images
- Data augmentation
- Style transfer
- Super-résolution

---

## 🟡 Diffusion Model

### Description

**Diffusion** : Modèle génératif via processus de débruitage itératif.

```
Noise → Denoise(t=1000) → ... → Denoise(t=1) → Image
```

### Utilisation

```lua
local diffusion = Mimir.Architectures.diffusion({
    image_channels = 3,
    image_size = 64,
    model_channels = 128,
    num_timesteps = 1000,
    beta_schedule = "linear"
})
```

### Paramètres

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `image_channels` | int | 3 | Canaux image |
| `image_size` | int | 64 | Taille image |
| `model_channels` | int | 128 | Channels UNet interne |
| `num_timesteps` | int | 1000 | Steps de diffusion |
| `beta_schedule` | string | "linear" | Schedule bruit (linear/cosine) |

### Process

**Forward Diffusion** (ajout bruit) :
```
x_t = √(1-β_t) × x_{t-1} + √β_t × ε, où ε ~ N(0,1)
```

**Reverse Diffusion** (débruitage) :
```
x_{t-1} = 1/√α_t × (x_t - (1-α_t)/√(1-ᾱ_t) × ε_θ(x_t, t))
```

### Cas d'Usage

- Génération d'images haute qualité
- Inpainting
- Text-to-image (avec conditioning)
- Image editing

---

## 🟠 Transformer

### Description

**Transformer** : Architecture attention-based pour séquences (NLP).

```
Input Tokens → Embedding → Transformer Blocks → Output Logits
```

### Utilisation

```lua
local transformer = Mimir.Architectures.transformer({
    vocab_size = 10000,
    d_model = 512,
    num_layers = 6,
    num_heads = 8,
    d_ff = 2048,
    max_seq_length = 512,
    dropout = 0.1
})
```

### Paramètres

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `vocab_size` | int | 10000 | Taille vocabulaire |
| `d_model` | int | 512 | Dimension embeddings |
| `num_layers` | int | 6 | Nombre de blocs |
| `num_heads` | int | 8 | Têtes d'attention |
| `d_ff` | int | 2048 | Dimension FFN |
| `max_seq_length` | int | 512 | Longueur max séquence |
| `dropout` | float | 0.1 | Dropout rate |

### Architecture Résultante

```
Input: Token IDs [B, seq_len]
  ↓ Embedding: [B, seq_len, d_model]
  ↓ + Positional Encoding
  ↓
  ↓ Transformer Block ×6:
  │   LayerNorm
  │   Multi-Head Self-Attention (8 heads)
  │   Residual Connection
  │   LayerNorm
  │   Feed-Forward (512→2048→512)
  │   Residual Connection
  ↓
  ↓ LayerNorm
  ↓ Linear: d_model → vocab_size
Output: Logits [B, seq_len, vocab_size]
```

### Variants

```lua
-- GPT-like (decoder-only)
local gpt = Mimir.Architectures.transformer({
    vocab_size = 50000,
    d_model = 768,
    num_layers = 12,
    num_heads = 12,
    causal = true  -- Masque causal
})

-- BERT-like (encoder-only)
local bert = Mimir.Architectures.transformer({
    vocab_size = 30000,
    d_model = 768,
    num_layers = 12,
    num_heads = 12,
    causal = false
})

-- Tiny Transformer (CPU-friendly)
local tiny = Mimir.Architectures.transformer({
    vocab_size = 10000,
    d_model = 256,
    num_layers = 4,
    num_heads = 4
})
```

### Cas d'Usage

- Language modeling (GPT)
- Text classification (BERT)
- Machine translation (encoder-decoder)
- Text generation

### Exemple Complet

```lua
-- Transformer pour génération de texte
local tokenizer = Mimir.Tokenizer.create()
Mimir.Tokenizer.loadVocab(tokenizer, "vocab.json")

local vocab_size = Mimir.Tokenizer.getVocabSize(tokenizer)

local transformer = Mimir.Architectures.transformer({
    vocab_size = vocab_size,
    d_model = 512,
    num_layers = 6,
    num_heads = 8,
    max_seq_length = 256
})

model.configure(transformer, {
    learning_rate = 0.0003,
    optimizer = "adam",
    loss = "cross_entropy",
    gradient_clip = 1.0
})

-- Dataset texte
local dataset = Mimir.Dataset.loadText("corpus.txt", tokenizer)

-- Entraîner
Mimir.Model.train(transformer, dataset, 50)

-- Générer
local prompt = "Once upon a time"
local prompt_ids = Mimir.Tokenizer.encode(tokenizer, prompt)
local generated = model.generate(transformer, prompt_ids, 100)
local text = Mimir.Tokenizer.decode(tokenizer, generated)
print(text)
```

---

## 🔵 ResNet

### Description

**ResNet** : CNN avec residual connections pour vision deep.

```
x → Conv Block → Residual Block ×N → Global Pool → FC
```

### Utilisation

```lua
local resnet = Mimir.Architectures.resnet({
    num_classes = 1000,
    depth = 50,  -- ResNet-50
    in_channels = 3
})
```

### Paramètres

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `num_classes` | int | 1000 | Nombre de classes |
| `depth` | int | 50 | Profondeur (18, 34, 50, 101, 152) |
| `in_channels` | int | 3 | Canaux d'entrée |

### Variants

```lua
-- ResNet-18 (léger)
local resnet18 = Mimir.Architectures.resnet({depth = 18})

-- ResNet-50 (standard)
local resnet50 = Mimir.Architectures.resnet({depth = 50})

-- ResNet-101 (deep)
local resnet101 = Mimir.Architectures.resnet({depth = 101})
```

### Architecture (ResNet-50)

```
Input: [B, 3, 224, 224]
  ↓ Conv(7×7, 64) + BN + ReLU + MaxPool
  ↓ ResBlock ×3 (64 channels)
  ↓ ResBlock ×4 (128 channels, stride=2)
  ↓ ResBlock ×6 (256 channels, stride=2)
  ↓ ResBlock ×3 (512 channels, stride=2)
  ↓ Global Average Pooling
  ↓ Linear(512 → num_classes)
Output: [B, num_classes]
```

**Residual Block** :
```
x → Conv(1×1) → BN → ReLU → Conv(3×3) → BN → ReLU → Conv(1×1) → BN
  ↓                                                              ↓
  └─────────────────────── Shortcut ────────────────────────────┘
                                    ↓ Add
                                  ReLU
```

### Cas d'Usage

- Classification d'images
- Feature extraction
- Transfer learning

---

## 🟢 MobileNet

### Description

**MobileNet** : CNN léger avec depthwise separable convolutions.

```
Input → Depthwise Conv → Pointwise Conv → ... → Classification
```

### Utilisation

```lua
local mobilenet = Mimir.Architectures.mobilenet({
    num_classes = 1000,
    width_multiplier = 1.0,
    input_size = 224
})
```

### Paramètres

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `num_classes` | int | 1000 | Nombre de classes |
| `width_multiplier` | float | 1.0 | Facteur largeur (0.25, 0.5, 0.75, 1.0) |
| `input_size` | int | 224 | Taille image |

### Depthwise Separable Convolution

```
Standard Conv 3×3 (C_in → C_out):
  Params = 3 × 3 × C_in × C_out
  
Depthwise Separable:
  1. Depthwise Conv 3×3 (C_in → C_in, groups=C_in)
     Params = 3 × 3 × C_in
  2. Pointwise Conv 1×1 (C_in → C_out)
     Params = 1 × 1 × C_in × C_out
  
Total = 3×3×C_in + C_in×C_out ≈ 8-9× fewer params!
```

### Variants

```lua
-- MobileNet-v1 (1.0)
local mobilenet_10 = Mimir.Architectures.mobilenet({
    width_multiplier = 1.0
})

-- MobileNet-v1 (0.5) - Ultra-light
local mobilenet_05 = Mimir.Architectures.mobilenet({
    width_multiplier = 0.5
})

-- MobileNet-v1 (0.25) - Extreme-light
local mobilenet_025 = Mimir.Architectures.mobilenet({
    width_multiplier = 0.25
})
```

### Cas d'Usage

- Mobile/edge deployment
- Real-time inference
- Resource-constrained environments

---

## 📊 Comparaison

| Architecture | Params (approx) | Cas d'Usage Principal | Difficulté |
|--------------|-----------------|----------------------|------------|
| **MobileNet** | 4M | Mobile, edge, real-time | ⭐ |
| **ResNet-18** | 11M | Classification rapide | ⭐⭐ |
| **UNet** | 30M | Segmentation | ⭐⭐ |
| **VAE** | 5-50M | Génération, compression | ⭐⭐ |
| **ResNet-50** | 25M | Classification robuste | ⭐⭐⭐ |
| **ViT-Base** | 86M | Vision de pointe | ⭐⭐⭐ |
| **Transformer** | 60-100M | NLP, génération texte | ⭐⭐⭐⭐ |
| **GAN** | 10-50M | Génération adversariale | ⭐⭐⭐⭐ |
| **Diffusion** | 50-500M | Génération haute qualité | ⭐⭐⭐⭐⭐ |

### Recommandations CPU

**Entraînement faisable** :
- MobileNet, ResNet-18, UNet (small), VAE, Tiny Transformer

**Entraînement lent** :
- ResNet-50, ViT (small), Transformer (6 layers)

**Inference uniquement** :
- ResNet-101, ViT-Large, Transformer (>12 layers), Diffusion

---

## 🎯 Prochaines Étapes

- [Training](06-Training.md) - Entraîner ces architectures
- [Model Creation](02-Model-Creation.md) - Créer architectures custom
- [Advanced](../05-Advanced/) - Optimisation et déploiement

---

**Questions ?** Consultez [INDEX](../00-INDEX.md) ou les [exemples](../../scripts/).

-->
