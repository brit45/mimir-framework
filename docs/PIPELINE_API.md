# 🎯 Pipeline API - Documentation Complète

## Vue d'ensemble

Le **Pipeline API** de Mímir Framework offre une interface Lua unifiée pour piloter **8 architectures de deep learning** avec une approche CPU-only optimisée.

## 🏗️ Architectures Disponibles

### 1. **Transformer** - Natural Language Processing
```lua
local transformer = Pipeline.Transformer({
    vocab_size = 5000,
    embed_dim = 256,
    num_layers = 4,
    num_heads = 8,
    d_ff = 1024,
    max_seq_len = 128,
    dropout = 0.1,
    model_type = "transformer"  -- ou "encoder", "decoder"
})

transformer:build()
transformer:train(dataset_path, epochs, lr)
local output = transformer:generate(prompt, max_length)
```

**Cas d'usage**: GPT, BERT, génération de texte, traduction, Q&A

---

### 2. **UNet** - Segmentation d'Images
```lua
local unet = Pipeline.UNet({
    input_channels = 3,      -- RGB
    output_channels = 1,     -- Masque
    base_channels = 64,
    num_levels = 4,
    blocks_per_level = 2,
    use_attention = true,
    use_residual = true,
    dropout = 0.1
})

unet:build()
unet:train(dataset_path, epochs, lr)
local mask = unet:segment(image_path)
```

**Cas d'usage**: Segmentation médicale, détection d'objets, inpainting

---

### 3. **VAE** - Variational Autoencoder
```lua
local vae = Pipeline.VAE({
    input_dim = 784,         -- 28x28 images
    latent_dim = 128,
    hidden_dims = {512, 256}
})

vae:build()
vae:train(dataset_path, epochs, lr)

-- Modes d'utilisation
local latent = vae:encode(image)
local reconstructed = vae:decode(latent)
local samples = vae:generate(num_samples)
```

**Cas d'usage**: Génération d'images, compression latente, détection d'anomalies

---

### 4. **ViT** - Vision Transformer
```lua
local vit = Pipeline.ViT({
    image_size = 224,
    patch_size = 16,
    num_classes = 1000,
    embed_dim = 768,
    num_layers = 12,
    num_heads = 12,
    dropout = 0.1
})

vit:build()
vit:train(dataset_path, epochs, lr)
local class = vit:classify(image_path)
```

**Cas d'usage**: Classification ImageNet, transfer learning, feature extraction

---

### 5. **GAN** - Generative Adversarial Network
```lua
local gan = Pipeline.GAN({
    latent_dim = 100,
    image_channels = 3,
    image_size = 64,
    gen_channels = 64,
    disc_channels = 64
})

gan:build()  -- Construit Generator + Discriminator
gan:train(dataset_path, epochs, lr)
local images = gan:generate(num_samples)
```

**Cas d'usage**: Génération d'images réalistes, data augmentation, style transfer

---

### 6. **Diffusion** - Denoising Diffusion Models
```lua
local diffusion = Pipeline.Diffusion({
    image_channels = 3,
    image_size = 256,
    timesteps = 1000,
    model_channels = 128,
    num_res_blocks = 2
})

diffusion:build()
diffusion:train(dataset_path, epochs, lr)
local image = diffusion:generate(prompt, num_steps)
```

**Cas d'usage**: Text-to-image (Stable Diffusion style), image editing, super-résolution

---

### 7. **ResNet** - Residual Network
```lua
local resnet = Pipeline.ResNet({
    num_classes = 1000,
    layers = {3, 4, 6, 3},  -- ResNet-50
    base_channels = 64,
    use_bottleneck = true
})

resnet:build()
resnet:train(dataset_path, epochs, lr)
local class = resnet:classify(image_path)
```

**Cas d'usage**: Classification, object detection (backbone), transfer learning

**Variantes**:
- ResNet-18: `[2,2,2,2]` → ~11M params
- ResNet-34: `[3,4,6,3]` → ~21M params
- ResNet-50: `[3,4,6,3]` + bottleneck → ~25M params
- ResNet-101: `[3,4,23,3]` → ~44M params

---

### 8. **MobileNet** - Mobile/Edge Efficient
```lua
local mobilenet = Pipeline.MobileNet({
    num_classes = 1000,
    width_mult = 1.0,        -- 0.5, 0.75, 1.0, 1.4
    input_size = 224
})

mobilenet:build()
mobilenet:train(dataset_path, epochs, lr)
local class = mobilenet:classify(image_path)
```

**Cas d'usage**: Classification mobile, edge devices, real-time inference, IoT

**Optimisé CPU** ✓ : Dépthwise separable convolutions → 8-9× moins de calculs

---

## 🎯 Pipeline Manager

Gérer plusieurs modèles simultanément :

```lua
local manager = Pipeline.PipelineManager:new()

-- Ajouter des pipelines
manager:add("transformer", Pipeline.Transformer(config))
manager:add("mobilenet", Pipeline.MobileNet(config))
manager:add("resnet", Pipeline.ResNet(config))

-- Lister
manager:list()

-- Récupérer
local model = manager:get("transformer")

-- Sauvegarder tous
manager:save_all("./checkpoints")
```

---

## 📊 Comparatif des Architectures

### Par Taille (Paramètres)

| Architecture | Params | Mémoire | Vitesse CPU |
|-------------|--------|---------|-------------|
| **MobileNet** | 3.4M | ~13MB | ⚡⚡⚡ Excellent |
| **Transformer (small)** | 5.8M | ~22MB | ⚡⚡ Bon |
| **ResNet-50** | 25M | ~100MB | ⚡ Moyen |
| **ViT-Base** | 86M | ~340MB | 🐌 Lent |
| **Diffusion** | 100M+ | ~400MB+ | 🐌🐌 Très lent |

### Par Domaine

| Domaine | Architectures | Best CPU |
|---------|--------------|----------|
| **NLP** | Transformer | ✅ |
| **Vision - Classification** | ViT, ResNet, MobileNet | MobileNet ⚡ |
| **Vision - Segmentation** | UNet | ✅ |
| **Génération** | VAE, GAN, Diffusion | VAE ⚡ |

---

## 💡 Recommandations CPU-Only

### 🏆 Top 3 pour Mímir

1. **MobileNet** (3.4M params)
   - ✓ Meilleur rapport vitesse/précision
   - ✓ Depthwise separable (CPU-friendly)
   - ✓ Scalable (width_mult: 0.5 → 1.4)

2. **Transformer** (5-25M params)
   - ✓ Excellent pour NLP
   - ✓ Architecture mature
   - ✓ Nombreux use cases

3. **ResNet** (11-60M params)
   - ✓ Performance éprouvée
   - ✓ Transfer learning efficace
   - ✓ Backbone polyvalent

### ⚠️ À éviter sur CPU

- **ViT** : Nécessite beaucoup de calcul d'attention
- **Diffusion** : 1000 timesteps → très lent
- **GAN** : Training adversarial instable

---

## 🔧 API Unifiée

Toutes les architectures partagent la même interface :

```lua
-- 1. Créer
local model = Pipeline.XXX(config)

-- 2. Construire
local ok, params = model:build()

-- 3. Entraîner
model:train(dataset_path, epochs, learning_rate)

-- 4. Inférer
local output = model:infer(input)
-- ou model:generate(), model:classify(), model:segment()

-- 5. Sauvegarder
model:save(checkpoint_path)
```

---

## 📂 Scripts de Démonstration

Chaque architecture a son script de demo :

```bash
./bin/mimir --lua scripts/demo_mobilenet.lua
./bin/mimir --lua scripts/demo_transformer.lua
./bin/mimir --lua scripts/demo_resnet.lua
./bin/mimir --lua scripts/demo_unet.lua
./bin/mimir --lua scripts/demo_vae.lua
./bin/mimir --lua scripts/demo_vit.lua
./bin/mimir --lua scripts/demo_gan.lua
./bin/mimir --lua scripts/demo_diffusion.lua
```

Test complet :
```bash
./bin/mimir --lua scripts/test_all_pipelines.lua
```

---

## 🎓 Exemple Complet

```lua
local Pipeline = dofile("./scripts/pipeline_api.lua")

-- Configuration
local config = {
    num_classes = 10,
    width_mult = 1.0,
    input_size = 224
}

-- Créer et construire
local model = Pipeline.MobileNet(config)
local ok, params = model:build()

if ok then
    log("✓ Modèle: " .. params .. " paramètres")
    
    -- Entraîner
    model:train("./data/cifar10", 100, 0.01)
    
    -- Classifier
    local class = model:classify("./test.jpg")
    log("Classe prédite: " .. class)
end
```

---

## 🚀 Prochaines Étapes

1. ✅ **8 architectures intégrées**
2. ✅ **Pipeline API unifié**
3. ✅ **Scripts de démonstration**
4. ⏳ **Entraînement réel sur datasets**
5. ⏳ **Inférence optimisée CPU**
6. ⏳ **Benchmarks comparatifs**

---

## 📖 Documentation Complète

- [Architecture du Framework](../docs/ARCHITECTURE.md)
- [API C++](../docs/API_CPP.md)
- [API Lua](../docs/API_LUA.md)
- [CPU-Only Philosophy](../docs/WHY_CPU_ONLY.md)
- [Quickstart](../docs/QUICKSTART.md)

---

**Mímir Framework v2.0** - Deep Learning CPU-Only 🖥️✨
