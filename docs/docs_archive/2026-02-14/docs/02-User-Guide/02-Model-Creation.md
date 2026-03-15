# Création de Modèles

Guide complet pour créer et configurer des modèles dans Mímir Framework.

---

## 📋 Table des Matières

- [Introduction](#introduction)
- [Approche Déclarative](#approche-déclarative)
- [Architectures Prédéfinies](#architectures-prédéfinies)
- [Configuration de Modèle](#configuration-de-modèle)
- [Construction et Initialisation](#construction-et-initialisation)
- [Bonnes Pratiques](#bonnes-pratiques)

---

## Introduction

Dans Mímir, créer un modèle suit ce workflow **déclaratif** :

```lua
1. Définir une configuration (table Lua)
2. (Optionnel) Partir d'une config par défaut via Mimir.Architectures.default_config(name)
3. Créer le modèle: Mimir.Model.create(name, config)
4. Allouer les paramètres: Mimir.Model.allocate_params()
5. Initialiser les poids: Mimir.Model.init_weights()
6. Entraîner: Mimir.Model.train(epochs, learning_rate)

> Note: `Mimir.Model.build()` existe pour compatibilité/rebuild (reconstruit le modèle via le registre),
> mais ne remplace pas `allocate_params()` ni `init_weights()`.
```

**Mímir utilise une approche déclarative**, pas impérative. Vous ne construisez pas le modèle couche par couche.

---

## Approche Déclarative

### Créer avec Configuration

```lua
-- 1. Partir d'une config par défaut (recommandé)
local config, err = Mimir.Architectures.default_config("transformer")
if not config then error(err) end

-- 2. Ajuster quelques paramètres
config.vocab_size = 5000
config.d_model = 256
config.num_layers = 4
config.num_heads = 8
config.max_seq_len = 128

-- 3. Créer modèle
local ok, create_err = Mimir.Model.create("transformer", config)
if not ok then error(create_err) end

-- (Optionnel) Rebuild compat
local ok_build, build_err = Mimir.Model.build()
if not ok_build then error(build_err) end

-- 4. Allouer mémoire
local success, num_params = Mimir.Model.allocate_params()
if not success then error("allocation failed") end
log(string.format("✓ Paramètres: %d", num_params))

-- 4. Initialiser poids
Mimir.Model.init_weights("xavier")

-- 5. Prêt pour l'entraînement
Mimir.Model.train(epochs, learning_rate)
```

### Types de Modèles Supportés

```lua
-- Liste exacte (v2.3.0) depuis le registre C++
-- Astuce: local names = Mimir.Architectures.available()
Mimir.Model.create("basic_mlp", config)
Mimir.Model.create("t2i_autoencoder", config)
Mimir.Model.create("transformer", config)
Mimir.Model.create("vit", config)
Mimir.Model.create("vae", config)
Mimir.Model.create("resnet", config)
Mimir.Model.create("unet", config)
Mimir.Model.create("mobilenet", config)
Mimir.Model.create("vgg16", config)
Mimir.Model.create("vgg19", config)
Mimir.Model.create("diffusion", config)
```

---

## Architectures Prédéfinies

### Utiliser le Registre d'Architectures (v2.3+)

> Les builders `Mimir.Architectures.<name>(cfg)` ne sont pas exposés en v2.3.
> Le module `Mimir.Architectures` fournit uniquement des helpers de registre : `available()` et `default_config(name)`.

#### Transformer

```lua
local cfg, err = Mimir.Architectures.default_config("transformer")
if not cfg then error(err) end

cfg.vocab_size = 50000
cfg.d_model = 768
cfg.num_layers = 12
cfg.num_heads = 12
cfg.max_seq_len = 512

local ok, create_err = Mimir.Model.create("transformer", cfg)
if not ok then error(create_err) end

local ok_alloc, params = Mimir.Model.allocate_params()
if not ok_alloc then error("allocation failed") end
```

#### U-Net

```lua
local cfg, err = Mimir.Architectures.default_config("unet")
if not cfg then error(err) end

cfg.in_channels = 3
cfg.out_channels = 1

local ok, create_err = Mimir.Model.create("unet", cfg)
if not ok then error(create_err) end

local ok_alloc = Mimir.Model.allocate_params()
if not ok_alloc then error("allocation failed") end
Mimir.Model.init_weights("he")
```

#### Vision Transformer (ViT)

```lua
local cfg, err = Mimir.Architectures.default_config("vit")
if not cfg then error(err) end

cfg.image_size = 224
cfg.patch_size = 16
cfg.num_classes = 1000

local ok, create_err = Mimir.Model.create("vit", cfg)
if not ok then error(create_err) end
local ok_alloc = Mimir.Model.allocate_params()
if not ok_alloc then error("allocation failed") end
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
Mimir.Model.allocate_params()
```

#### GAN

```lua
-- Note: l'architecture "gan" n'est pas une builtin du registre v2.3.0.
-- Utilisez Mimir.Architectures.available() pour la liste exacte.
```

#### ResNet

```lua
local cfg, err = Mimir.Architectures.default_config("resnet")
if not cfg then error(err) end

cfg.num_classes = 1000

local ok, create_err = Mimir.Model.create("resnet", cfg)
if not ok then error(create_err) end
local ok_alloc = Mimir.Model.allocate_params()
if not ok_alloc then error("allocation failed") end
```

#### MobileNet

```lua
local cfg, err = Mimir.Architectures.default_config("mobilenet")
if not cfg then error(err) end

cfg.num_classes = 1000

local ok, create_err = Mimir.Model.create("mobilenet", cfg)
if not ok then error(create_err) end
local ok_alloc = Mimir.Model.allocate_params()
if not ok_alloc then error("allocation failed") end
```

---

## Configuration de Modèle

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

## Construction et Initialisation

### Étape 1: Allocation des Paramètres (v2.3)

```lua
-- Allouer les paramètres du modèle courant
local ok, num_params = Mimir.Model.allocate_params()
if not ok then error("allocation failed") end

log(string.format("✓ Paramètres: %d", num_params))
log(string.format("  Mémoire (poids fp32): %.2f MB", num_params * 4 / 1024 / 1024))
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

### Exemple 1: Transformer (texte)

> Note v2.3 : les types `encoder`/`decoder` ne font pas partie des architectures built-in du registre.
> Utilisez `transformer` (avec une config adaptée) et référez-vous aux scripts `scripts/examples/` pour des pipelines complets.

```lua
-- 1. Partir de la config par défaut
local cfg = assert(Mimir.Architectures.default_config("transformer"))

-- 2. Ajuster
cfg.vocab_size = 50000
cfg.max_seq_len = 1024
cfg.d_model = 768
cfg.num_layers = 12
cfg.num_heads = 12
cfg.use_causal_mask = true  -- si supporté par l'implémentation

-- 3. Créer + allouer
assert(Mimir.Model.create("transformer", cfg))
local ok, params = Mimir.Model.allocate_params()
if not ok then error("allocation failed") end
log(string.format("Transformer créé: %d paramètres", params))

-- 4. Initialiser
Mimir.Model.init_weights("xavier", 42)
```

### Exemple 3: U-Net Segmentation

```lua
-- 1. Partir d'une config par défaut (recommandé)
local cfg = assert(Mimir.Architectures.default_config("unet"))
cfg.in_channels = 3
cfg.out_channels = 21

-- 2. Créer
assert(Mimir.Model.create("unet", cfg))

-- 3. Allouer
assert(Mimir.Model.allocate_params())
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

-- 2. Créer (config exacte: préférez default_config("vit"))
assert(Mimir.Model.create("vit", vit_config))

-- 3. Allouer
assert(Mimir.Model.allocate_params())
Mimir.Model.init_weights("xavier")

-- 4. Dataset
Mimir.Dataset.load("data/imagenet")

-- 5. Entraîner (long sur CPU!)
Mimir.Model.train(300, 0.0003)

-- 6. Sauvegarder
Mimir.Model.save("checkpoints/vit_imagenet")
```

---

## Bonnes Pratiques

### 1. Toujours Vérifier l'Allocation

```lua
local ok, params = Mimir.Model.allocate_params()
if not ok then error("allocation failed") end
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

## Ressources

- **Scripts exemples:** `scripts/example_simple.lua`, `scripts/example_gpt.lua`
- **API complète:** [API Reference](../03-API-Reference/00-API-Complete.md)
- **Architectures:** [Model Architectures](../05-Advanced/05-Model-Architectures.md)
- **Pipeline API:** [Pipeline API](../05-Advanced/01-Pipeline-API.md)

---

## Ce qui N'EXISTE PAS

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
- ✅ `Mimir.Architectures.default_config(name)`
- ✅ `Mimir.Model.allocate_params()`
- ✅ `Mimir.Model.init_weights()`

---

**Guide créé le 22 décembre 2025**  
**Validé contre:** `src/LuaScripting.cpp`, `scripts/examples/example_simple.lua`, `scripts/examples/example_gpt.lua`
