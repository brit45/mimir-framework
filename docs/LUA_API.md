# API Lua - Mímir Framework

Documentation complète de l'API Lua pour piloter le Mímir Framework.

## Table des Matières

- [model API](#model-api)
- [architectures API](#architectures-api)
- [layers API](#layers-api)
- [tokenizer API](#tokenizer-api)
- [dataset API](#dataset-api)
- [Utilitaires](#utilitaires)

---

## model API

Gestion des modèles et opérations de base.

### `model.create(type)`

Crée un nouveau modèle.

```lua
local success, err = model.create("transformer")
```

**Paramètres:**
- `type` (string): Type de modèle (libre, pour information)

**Retour:**
- `success` (boolean): true si succès
- `err` (string, optionnel): Message d'erreur

### `model.allocate_params()`

Alloue la mémoire pour les paramètres du modèle.

```lua
local success, count = model.allocate_params()
print(string.format("Paramètres alloués: %d", count))
```

**Retour:**
- `success` (boolean): true si succès
- `count` (number): Nombre de paramètres alloués

### `model.init_weights(method, seed)`

Initialise les poids du modèle.

```lua
local success = model.init_weights("he", 42)
```

**Paramètres:**
- `method` (string): Méthode d'initialisation
  - `"he"` / `"kaiming"`: He initialization (optimal pour ReLU/GELU)
  - `"xavier"` / `"glorot"`: Xavier initialization
  - `"normal"`: Distribution normale simple
- `seed` (number, optionnel): Seed aléatoire (0 = aléatoire)

**Retour:**
- `success` (boolean): true si succès

### `model.total_params()`

Retourne le nombre total de paramètres.

```lua
local count = model.total_params()
```

**Retour:**
- `count` (number): Nombre total de paramètres

### `model.push_layer(name, type, params_count)`

Ajoute un layer manuellement au modèle.

```lua
model.push_layer("conv1", "Conv2d", 64 * 3 * 3 * 3 + 64)
```

**Paramètres:**
- `name` (string): Nom du layer
- `type` (string): Type de layer
- `params_count` (number): Nombre de paramètres

### `model.forward(input)`

Effectue un forward pass.

```lua
local input = {0.1, 0.2, 0.3, ...}
local output = model.forward(input)
```

**Paramètres:**
- `input` (table): Tableau de floats

**Retour:**
- `output` (table): Tableau de floats

### `model.backward(loss_gradient)`

Effectue un backward pass.

```lua
local grad = {0.01, 0.02, ...}
local success = model.backward(grad)
```

**Paramètres:**
- `loss_gradient` (table): Gradient de la loss

**Retour:**
- `success` (boolean): true si succès

### `model.optimizer_step(lr, optimizer_type)`

Effectue un pas d'optimisation.

```lua
model.optimizer_step(0.0001, "adamw")
```

**Paramètres:**
- `lr` (number): Learning rate
- `optimizer_type` (string, optionnel): Type d'optimiseur
  - `"sgd"`: Stochastic Gradient Descent
  - `"adam"`: Adam
  - `"adamw"`: Adam with weight decay (défaut)

### `model.set_hardware(enable)`

Active/désactive l'accélération hardware.

```lua
model.set_hardware(true)  -- Activer AVX2/FMA
```

**Paramètres:**
- `enable` (boolean): true pour activer

### `model.hardware_caps()`

Retourne les capacités hardware disponibles.

```lua
local hw = model.hardware_caps()
print("AVX2: " .. tostring(hw.avx2))
print("FMA:  " .. tostring(hw.fma))
print("F16C: " .. tostring(hw.f16c))
print("BMI2: " .. tostring(hw.bmi2))
```

**Retour:**
- `caps` (table): Table avec les capacités
  - `avx2` (boolean)
  - `fma` (boolean)
  - `f16c` (boolean)
  - `bmi2` (boolean)

### `model.save(filepath)`

Sauvegarde le modèle.

```lua
local success = model.save("checkpoints/model.safetensors")
```

### `model.load(filepath)`

Charge un modèle.

```lua
local success = model.load("checkpoints/model.safetensors")
```

---

## architectures API

Construction d'architectures modernes prêtes à l'emploi.

### `architectures.unet(config)`

Construit une architecture UNet.

```lua
local config = {
    input_channels = 3,
    output_channels = 1,
    base_channels = 64,
    num_levels = 4,           -- Nombre de niveaux de downsampling
    blocks_per_level = 2,     -- Blocs par niveau
    use_attention = true,     -- Attention aux niveaux profonds
    use_residual = true,      -- Skip connections
    use_batchnorm = true,
    dropout = 0.0
}

local success = architectures.unet(config)
```

**Paramètres:**
- `config` (table): Configuration UNet

**Cas d'usage:** Segmentation d'images, génération, super-résolution

### `architectures.vae(config)`

Construit un Variational Autoencoder.

```lua
local config = {
    input_dim = 784,           -- 28*28 pour MNIST
    latent_dim = 128,
    encoder_hidden = {512, 256},
    decoder_hidden = {256, 512},
    use_batchnorm = false
}

local success = architectures.vae(config)
```

**Cas d'usage:** Génération d'images, compression, représentation latente

### `architectures.vit(config)`

Construit un Vision Transformer.

```lua
local config = {
    image_size = 224,
    patch_size = 16,
    num_classes = 1000,
    d_model = 768,
    num_heads = 12,
    num_layers = 12,
    mlp_ratio = 4,
    dropout = 0.1,
    use_cls_token = true
}

local success = architectures.vit(config)
```

**Cas d'usage:** Classification d'images avec transformers

### `architectures.gan(type, config)`

Construit un Generator ou Discriminator pour GAN.

```lua
local config = {
    latent_dim = 100,
    image_size = 64,
    image_channels = 3,
    g_base_channels = 64,
    d_base_channels = 64,
    self_attention = true
}

-- Generator
architectures.gan("generator", config)

-- Discriminator
architectures.gan("discriminator", config)
```

**Paramètres:**
- `type` (string): "generator" ou "discriminator"
- `config` (table): Configuration GAN

**Cas d'usage:** Génération d'images réalistes

### `architectures.diffusion(config)`

Construit un modèle de diffusion (DDPM).

```lua
local config = {
    image_size = 32,
    image_channels = 3,
    base_channels = 128,
    num_res_blocks = 2,
    channel_multipliers = {1, 2, 2, 2},
    attention_levels = {1, 2, 3},
    time_embed_dim = 512
}

local success = architectures.diffusion(config)
```

**Cas d'usage:** Génération d'images haute qualité (Stable Diffusion, DALL-E)

### `architectures.transformer(config)`

Construit un Transformer (GPT-style).

```lua
local config = {
    vocab_size = 50000,
    max_seq_len = 2048,
    d_model = 768,
    num_heads = 12,
    num_layers = 12,
    d_ff = 3072,
    dropout = 0.1,
    causal = true            -- true=GPT, false=BERT
}

local success = architectures.transformer(config)
```

**Cas d'usage:** Génération de texte, traduction, completion

### `architectures.resnet(config)`

Construit un ResNet-50.

```lua
local config = {
    num_classes = 1000,
    layers = {3, 4, 6, 3},   -- ResNet-50
    base_channels = 64,
    use_bottleneck = true
}

local success = architectures.resnet(config)
```

**Cas d'usage:** Classification d'images (backbone classique)

### `architectures.mobilenet(config)`

Construit un MobileNetV2.

```lua
local config = {
    num_classes = 1000,
    width_multiplier = 1.0,
    resolution = 224
}

local success = architectures.mobilenet(config)
```

**Cas d'usage:** Classification d'images sur mobile/embarqué

---

## layers API

Opérations de layers bas niveau (stubs - utilisez `model.forward()` à la place).

### `layers.conv2d(...)`

Convolution 2D (stub).

### `layers.linear(...)`

Layer linéaire/dense (stub).

### `layers.maxpool2d(...)`

Max pooling 2D (stub).

### `layers.avgpool2d(...)`

Average pooling 2D (stub).

### `layers.activation(...)`

Fonctions d'activation (stub).

### `layers.batchnorm(...)`

Batch normalization (stub).

### `layers.layernorm(...)`

Layer normalization (stub).

### `layers.attention(...)`

Multi-head attention (stub).

---

## tokenizer API

Gestion de la tokenization.

### `tokenizer.create(vocab_size)`

Crée un nouveau tokenizer.

```lua
local success = tokenizer.create(10000)
```

**Paramètres:**
- `vocab_size` (number): Taille du vocabulaire

### `tokenizer.tokenize(text)`

Tokenize un texte.

```lua
local tokens = tokenizer.tokenize("Bonjour le monde")
-- tokens = {1234, 5678, 9012}
```

**Paramètres:**
- `text` (string): Texte à tokenizer

**Retour:**
- `tokens` (table): Liste d'IDs

### `tokenizer.detokenize(tokens)`

Reconstruit le texte depuis les tokens.

```lua
local text = tokenizer.detokenize({1234, 5678, 9012})
-- text = "Bonjour le monde"
```

**Paramètres:**
- `tokens` (table): Liste d'IDs

**Retour:**
- `text` (string): Texte reconstruit

### `tokenizer.vocab_size()`

Retourne la taille du vocabulaire.

```lua
local size = tokenizer.vocab_size()
```

### `tokenizer.save(filepath)`

Sauvegarde le tokenizer.

```lua
tokenizer.save("tokenizer.json")
```

### `tokenizer.load(filepath)`

Charge un tokenizer.

```lua
tokenizer.load("tokenizer.json")
```

---

## dataset API

Chargement et préparation de datasets.

### `dataset.load(directory)`

Charge un dataset depuis un répertoire.

```lua
local success = dataset.load("data/train")
```

### `dataset.prepare_sequences()`

Prépare les séquences pour l'entraînement.

```lua
local success = dataset.prepare_sequences()
```

---

## Utilitaires

### `log(message)`

Affiche un message de log.

```lua
log("Début de l'entraînement")
```

### `read_json(filepath)`

Lit un fichier JSON.

```lua
local data = read_json("config.json")
print(data.learning_rate)
```

### `write_json(filepath, data)`

Écrit un fichier JSON.

```lua
local config = {
    learning_rate = 0.001,
    batch_size = 32
}
write_json("config.json", config)
```

---

## Exemples Complets

### Exemple 1: UNet pour segmentation

```lua
-- Créer et configurer le modèle
model.create("unet_segmentation")

local config = {
    input_channels = 3,
    output_channels = 1,
    base_channels = 64,
    num_levels = 4
}

architectures.unet(config)
model.allocate_params()
model.init_weights("he", 42)

print(string.format("Modèle prêt: %d paramètres", model.total_params()))

-- Entraînement (simplifié)
for epoch = 1, 100 do
    -- Forward, backward, optimizer step
    model.optimizer_step(0.0001, "adamw")
    
    if epoch % 10 == 0 then
        log(string.format("Epoch %d/100", epoch))
    end
end

-- Sauvegarder
model.save("checkpoints/unet_epoch100.safetensors")
```

### Exemple 2: Transformer pour génération de texte

```lua
-- Tokenizer
tokenizer.create(50000)
tokenizer.load("tokenizer.json")

-- Modèle
model.create("gpt")

local config = {
    vocab_size = 50000,
    d_model = 512,
    num_layers = 6,
    num_heads = 8,
    max_seq_len = 1024
}

architectures.transformer(config)
model.allocate_params()
model.init_weights("xavier", 123)

-- Forward pass
local text = "Il était une fois"
local tokens = tokenizer.tokenize(text)
local output = model.forward(tokens)

-- Générer
local next_token = output[#output]  -- Dernier token
```

### Exemple 3: GAN pour génération d'images

```lua
-- Generator
model.create("gan_gen")
local gan_config = {
    latent_dim = 100,
    image_size = 64
}
architectures.gan("generator", gan_config)
model.allocate_params()
model.init_weights("he")

local generator = model  -- Sauvegarder la référence

-- Discriminator
model.create("gan_disc")
architectures.gan("discriminator", gan_config)
model.allocate_params()
model.init_weights("he")

-- Entraînement GAN (simplifié)
for step = 1, 10000 do
    -- 1. Train discriminator
    -- ...
    
    -- 2. Train generator
    -- ...
    
    if step % 100 == 0 then
        log(string.format("Step %d/10000", step))
    end
end
```

---

## Bonnes Pratiques

1. **Toujours vérifier les capacités hardware:**
   ```lua
   local hw = model.hardware_caps()
   if hw.avx2 and hw.fma then
       model.set_hardware(true)
   end
   ```

2. **Initialiser les poids selon l'architecture:**
   - `"he"` pour ReLU, GELU, ELU
   - `"xavier"` pour Tanh, Sigmoid

3. **Sauvegarder régulièrement:**
   ```lua
   if epoch % 10 == 0 then
       model.save(string.format("checkpoint_epoch%d.safetensors", epoch))
   end
   ```

4. **Utiliser l'optimiseur adapté:**
   - `"adamw"` pour la plupart des cas (weight decay découplé)
   - `"adam"` pour fine-tuning
   - `"sgd"` pour certains cas spécifiques

5. **Monitorer la taille du modèle:**
   ```lua
   local params = model.total_params()
   local memory_mb = params * 2 / 1024 / 1024  -- FP16
   print(string.format("Mémoire: %.1f MB", memory_mb))
   ```

---

## Limitations Actuelles

- Les layers API (`layers.conv2d`, etc.) sont des stubs
- Pas de gradient clipping direct (à implémenter dans C++)
- Pas de mixed precision training (FP16/FP32)
- Dataset API minimale (à étendre)

---

## Référence Rapide

| API | Description |
|-----|-------------|
| `model.create()` | Crée un modèle |
| `model.allocate_params()` | Alloue la mémoire |
| `model.init_weights()` | Initialise les poids |
| `model.forward()` | Forward pass |
| `model.backward()` | Backward pass |
| `model.optimizer_step()` | Pas d'optimisation |
| `architectures.unet()` | UNet |
| `architectures.vae()` | VAE |
| `architectures.vit()` | Vision Transformer |
| `architectures.gan()` | GAN |
| `architectures.diffusion()` | Diffusion Model |
| `architectures.transformer()` | Transformer (GPT) |
| `architectures.resnet()` | ResNet |
| `architectures.mobilenet()` | MobileNet |
| `tokenizer.create()` | Crée un tokenizer |
| `tokenizer.tokenize()` | Tokenize |
| `tokenizer.detokenize()` | Détokenize |

---

## Support

Pour plus d'informations, consultez:
- `docs/API_LUA.md` - Cette documentation
- `docs/MODEL_ARCHITECTURES.md` - Détails des architectures
- `docs/LAYER_OPERATIONS.md` - Opérations bas niveau
- `examples/` - Exemples de code
