# Création de Modèles

Guide complet pour créer et configurer des modèles dans Mímir Framework.

---

## 📋 Table des Matières

- [Introduction](#introduction)
- [Approche Déclarative](#approche-déclarative)
- [Architectures Prédéfinies](#architectures-prédéfinies)
- [Configuration de Modèle](#configuration-de-modèle)
- [Construction et Initialisation](#construction-et-initialisation)
- [Exemples Complets](#exemples-complets)
- [Bonnes Pratiques](#bonnes-pratiques)

---

## 🎯 Introduction

Dans Mímir, créer un modèle suit ce workflow **déclaratif** :

```lua
1. Définir une configuration (table Lua)
2. Créer le modèle avec Mimir.Model.create(type, config)
   OU utiliser un builder Mimir.Architectures.xxx(config)
3. Construire: Mimir.Model.build()
4. Initialiser poids: Mimir.Model.init_weights()
5. Entraîner: Mimir.Model.train()
```

**Mímir utilise une approche déclarative**, pas impérative. Vous ne construisez pas le modèle couche par couche.

---

## 🏗️ Approche Déclarative

### Créer avec Configuration

```lua
-- 1. Définir configuration
local config = {
    vocab_size = 5000,
    embed_dim = 256,
    num_layers = 4,
    num_heads = 8,
    d_ff = 1024,
    max_seq_len = 128,
    dropout = 0.1
}

-- 2. Créer modèle
Mimir.Model.create("transformer", config)

-- 3. Construire (allouer mémoire)
local ok, num_params = Mimir.Model.build()
if ok then
    log(string.format("✓ Modèle construit: %d paramètres", num_params))
end

-- 4. Initialiser poids
Mimir.Model.init_weights("xavier")

-- 5. Prêt pour l'entraînement
Mimir.Model.train(epochs, learning_rate)
```

### Types de Modèles Supportés

```lua
-- Transformers
Mimir.Model.create("transformer", config)  -- Transformer complet
Mimir.Model.create("encoder", config)      -- Encoder seulement (BERT-like)
Mimir.Model.create("decoder", config)      -- Decoder seulement (GPT-like)

-- Vision
Mimir.Model.create("unet", config)         -- U-Net segmentation
Mimir.Model.create("vit", config)          -- Vision Transformer
Mimir.Model.create("resnet", config)       -- ResNet
Mimir.Model.create("mobilenet", config)    -- MobileNet

-- Génératif
Mimir.Model.create("vae", config)          -- Variational Autoencoder
Mimir.Model.create("gan", config)          -- GAN
Mimir.Model.create("diffusion", config)    -- Diffusion models
```

---

## 🎨 Architectures Prédéfinies

### Utiliser les Builders

Les builders `Mimir.Architectures.*` sont des helpers qui créent des modèles avec configurations prédéfinies.

#### Transformer

```lua
local config = {
    vocab_size = 50000,
    embed_dim = 768,
    num_layers = 12,
    num_heads = 12,
    d_ff = 3072,
    max_seq_len = 512,
    dropout = 0.1
}

-- Utiliser le builder
local transformer = Mimir.Architectures.transformer(config)

-- Puis construire
local ok, params = Mimir.Model.build()
```

#### U-Net

```lua
local config = {
    in_channels = 3,
    out_channels = 1,
    base_channels = 64,
    num_levels = 4,
    blocks_per_level = 2,
    use_attention = true,
    use_residual = true,
    dropout = 0.1
}

local unet = Mimir.Architectures.unet(config)
Mimir.Model.build()
Mimir.Model.init_weights("he")
```

#### Vision Transformer (ViT)

```lua
local config = {
    image_size = 224,
    patch_size = 16,
    num_classes = 1000,
    embed_dim = 768,
    num_layers = 12,
    num_heads = 12,
    d_ff = 3072,
    dropout = 0.1
}

local vit = Mimir.Architectures.vit(config)
Mimir.Model.build()
```

#### VAE

```lua
local config = {
    input_dim = 784,
    latent_dim = 128,
    encoder_hidden = {512, 256},
    decoder_hidden = {256, 512},
    kl_beta = 1.0
}

Mimir.Model.create("vae", config)
Mimir.Model.build()
```

#### GAN

```lua
local config = {
    latent_dim = 100,
    image_channels = 3,
    image_size = 64,
    gen_channels = {64, 128, 256},
    disc_channels = {64, 128, 256}
}

local gan = Mimir.Architectures.gan(config)
Mimir.Model.build()
```

#### ResNet

```lua
local config = {
    num_classes = 1000,
    layers = {3, 4, 6, 3},  -- ResNet-50
    base_channels = 64,
    use_bottleneck = true
}

local resnet = Mimir.Architectures.resnet(config)
Mimir.Model.build()
```

#### MobileNet

```lua
local config = {
    num_classes = 1000,
    width_mult = 1.0,
    resolution = 224
}

local mobilenet = Mimir.Architectures.mobilenet(config)
Mimir.Model.build()
```

---

## ⚙️ Configuration de Modèle

### Configuration Transformer Complète

```lua
local transformer_config = {
    -- Vocabulaire
    vocab_size = 50000,
    
    -- Architecture
    embed_dim = 768,        -- Dimension des embeddings
    num_layers = 12,        -- Nombre de couches
    num_heads = 12,         -- Têtes d'attention
    d_ff = 3072,           -- Dimension feed-forward
    
    -- Séquences
    max_seq_len = 512,      -- Longueur max séquence
    
    -- Régularisation
    dropout = 0.1,
    
    -- Options
    use_prenorm = true,     -- Pre-LayerNorm (GPT-3 style)
    use_causal_mask = false -- Masque causal (pour decoder)
}
```

### Configuration U-Net Complète

```lua
local unet_config = {
    -- Entrée/Sortie
    in_channels = 3,        -- RGB
    out_channels = 21,      -- Classes
    
    -- Architecture
    base_channels = 64,
    num_levels = 4,
    blocks_per_level = 2,
    
    -- Mécanismes
    use_attention = true,
    attention_levels = {2, 3},
    use_residual = true,
    
    -- Régularisation
    dropout = 0.1
}
```

### Configuration ViT Complète

```lua
local vit_config = {
    -- Image
    image_size = 224,
    patch_size = 16,
    in_channels = 3,
    
    -- Classification
    num_classes = 1000,
    
    -- Architecture Transformer
    embed_dim = 768,
    num_layers = 12,
    num_heads = 12,
    d_ff = 3072,
    
    -- Options
    use_class_token = true,
    dropout = 0.1
}
```

---

## 🔨 Construction et Initialisation

### Étape 1: Build

```lua
-- Construire modèle (allouer paramètres)
local ok, num_params, err = Mimir.Model.build()

if ok then
    log(string.format("✓ Modèle construit avec %d paramètres", num_params))
    log(string.format("  Mémoire: %.2f MB", num_params * 4 / 1024 / 1024))
else
    log("✗ Erreur construction: " .. (err or "inconnue"))
end
```

### Étape 2: Initialisation des Poids

```lua
-- Méthodes d'initialisation disponibles
Mimir.Model.init_weights("xavier")    -- Xavier/Glorot (défaut)
Mimir.Model.init_weights("he")        -- He initialization (pour ReLU)
Mimir.Model.init_weights("normal")    -- Distribution normale
Mimir.Model.init_weights("uniform")   -- Distribution uniforme

-- Avec seed pour reproductibilité
Mimir.Model.init_weights("xavier", 42)
```

### Étape 3: Allocation Explicite (optionnel)

```lua
-- Parfois utile pour debugging
Mimir.Model.allocate_params()
```

### Vérifier Paramètres

```lua
-- Compter paramètres total
local total = Mimir.Model.total_params()
log(string.format("Paramètres: %d (%.2f M)", total, total / 1e6))
```

---

## 📝 Exemples Complets

### Exemple 1: Transformer GPT-style

```lua
-- 1. Configuration
local gpt_config = {
    vocab_size = 50000,
    embed_dim = 768,
    num_layers = 12,
    num_heads = 12,
    d_ff = 3072,
    max_seq_len = 1024,
    dropout = 0.1,
    use_causal_mask = true  -- Auto-régressif
}

-- 2. Créer
Mimir.Model.create("decoder", gpt_config)

-- 3. Build
local ok, params = Mimir.Model.build()
if ok then
    log(string.format("GPT créé: %d paramètres", params))
end

-- 4. Initialiser
Mimir.Model.init_weights("xavier", 42)

-- 5. Dataset
Mimir.Dataset.load("data/corpus")
Mimir.Dataset.prepare_sequences(1024)

-- 6. Entraîner
Mimir.Model.train(50, 0.0003)

-- 7. Sauvegarder
Mimir.Model.save("checkpoints/gpt_model")
```

### Exemple 2: BERT Classifier

```lua
-- 1. Configuration encoder
local bert_config = {
    vocab_size = 30000,
    embed_dim = 768,
    num_layers = 12,
    num_heads = 12,
    d_ff = 3072,
    max_seq_len = 512,
    dropout = 0.1,
    use_prenorm = true,
    pooling = "cls"  -- Utiliser token [CLS]
}

-- 2. Créer
Mimir.Model.create("encoder", bert_config)

-- 3. Build
Mimir.Model.build()
Mimir.Model.init_weights("xavier")

-- 4. Dataset classification
Mimir.Dataset.load("data/sentiment")
Mimir.Dataset.prepare_sequences(512)

-- 5. Entraîner
Mimir.Model.train(20, 0.0001)

-- 6. Sauvegarder
Mimir.Model.save("checkpoints/bert_classifier")
```

### Exemple 3: U-Net Segmentation

```lua
-- 1. Configuration
local unet_config = {
    in_channels = 3,      -- RGB
    out_channels = 21,    -- 21 classes
    base_channels = 64,
    num_levels = 4,
    use_attention = true,
    attention_levels = {2, 3}
}

-- 2. Créer avec builder
local unet = Mimir.Architectures.unet(unet_config)

-- 3. Build
Mimir.Model.build()
Mimir.Model.init_weights("he")  -- He pour ReLU

-- 4. Dataset images
Mimir.Dataset.load("data/images")

-- 5. Entraîner
Mimir.Model.train(100, 0.0002)

-- 6. Sauvegarder
Mimir.Model.save("checkpoints/unet_segmentation")
```

### Exemple 4: Vision Transformer

```lua
-- 1. Configuration
local vit_config = {
    image_size = 224,
    patch_size = 16,
    in_channels = 3,
    num_classes = 1000,  -- ImageNet
    embed_dim = 768,
    num_layers = 12,
    num_heads = 12,
    d_ff = 3072,
    dropout = 0.1,
    use_class_token = true
}

-- 2. Créer
Mimir.Model.create("vit", vit_config)

-- 3. Build
Mimir.Model.build()
Mimir.Model.init_weights("xavier")

-- 4. Dataset
Mimir.Dataset.load("data/imagenet")

-- 5. Entraîner (long sur CPU!)
Mimir.Model.train(300, 0.0003)

-- 6. Sauvegarder
Mimir.Model.save("checkpoints/vit_imagenet")
```

---

## ✅ Bonnes Pratiques

### 1. Toujours Vérifier Build

```lua
local ok, params, err = Mimir.Model.build()
if not ok then
    error("Erreur build: " .. (err or "inconnue"))
end
```

### 2. Initialisation Appropriée

```lua
-- Pour ReLU: He initialization
Mimir.Model.init_weights("he")

-- Pour Tanh/Sigmoid: Xavier
Mimir.Model.init_weights("xavier")
```

### 3. Reproductibilité

```lua
-- Utiliser seed fixe
Mimir.Model.init_weights("xavier", 42)
```

### 4. Vérifier Mémoire

```lua
local params = Mimir.Model.total_params()
local memory_mb = params * 4 / 1024 / 1024
log(string.format("Mémoire requise: %.2f MB", memory_mb))

-- Vérifier avant de commencer
if memory_mb > 8000 then  -- 8 GB
    error("Modèle trop gros pour RAM disponible")
end
```

### 5. Configurations Testées

```lua
-- Commencer petit pour tester
local tiny_config = {
    vocab_size = 1000,
    embed_dim = 128,
    num_layers = 2,
    num_heads = 4,
    d_ff = 512,
    max_seq_len = 64,
    dropout = 0.1
}

-- Puis scaler vers production
local production_config = {
    vocab_size = 50000,
    embed_dim = 768,
    num_layers = 12,
    num_heads = 12,
    d_ff = 3072,
    max_seq_len = 512,
    dropout = 0.1
}
```

### 6. Documentation des Configs

```lua
-- Toujours documenter les choix
local config = {
    -- Vocabulaire: BPE trained on Wikipedia + BookCorpus
    vocab_size = 50000,
    
    -- Architecture: BERT-base configuration
    embed_dim = 768,
    num_layers = 12,
    num_heads = 12,
    
    -- Hyperparamètres: optimisés via grid search
    dropout = 0.1,
    max_seq_len = 512
}
```

---

## 📚 Ressources

- **Scripts exemples:** `scripts/example_simple.lua`, `scripts/example_gpt.lua`
- **API complète:** [API Reference](../03-API-Reference/00-API-Complete.md)
- **Architectures:** [Model Architectures](../05-Advanced/05-Model-Architectures.md)
- **Pipeline API:** [Pipeline API](../05-Advanced/01-Pipeline-API.md)

---

## ⚠️ Ce qui N'EXISTE PAS

Les fonctions suivantes **n'existent pas** dans Mímir:

- ❌ `layers.addLinear()`
- ❌ `layers.addActivation()`
- ❌ `layers.addDropout()`
- ❌ `layers.addLayerNorm()`
- ❌ `layers.addBatchNorm()`
- ❌ `layers.addConv2D()`
- ❌ `layers.addEmbedding()`
- ❌ `model.configure()`
- ❌ `model.setMode()`

**Le module `layers` contient des fonctions de CALCUL** (conv2d, linear, maxpool2d), **PAS de construction de modèle**.

Utiliser à la place:
- ✅ `Mimir.Model.create(type, config)`
- ✅ `Mimir.Architectures.xxx(config)`
- ✅ `Mimir.Model.build()`
- ✅ `Mimir.Model.init_weights()`

---

**Guide créé le 22 décembre 2025**  
**Validé contre:** `src/LuaScripting.cpp`, `scripts/example_simple.lua`, `scripts/example_gpt.lua`
