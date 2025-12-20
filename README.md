# Mímir Framework v2.0

<div align="center">

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![C++17](https://img.shields.io/badge/C++-17-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![OpenMP](https://img.shields.io/badge/OpenMP-enabled-red)
![Vulkan](https://img.shields.io/badge/Vulkan-Compute-purple)

**Framework de Deep Learning haute performance en C++17 avec optimisations hardware avancées, API Lua complète et monitoring asynchrone**

[Documentation](#-documentation) • [Installation](#-installation) • [Quickstart](#-quickstart) • [v2.0 Features](#-nouveautés-v20)

</div>

---

## 🆕 Nouveautés v2.0

### 🧵 Threading Asynchrone
- **HtopDisplay** et **Visualizer** s'exécutent dans des threads séparés
- Rendu automatique non-bloquant (100ms d'intervalle)
- API simplifiée: plus besoin d'appeler `render()`, `process_events()`
- **Gain**: ~20% d'accélération sur le training complet

### ⚡ Accélération GPU (Vulkan Compute)
- Détection automatique de Vulkan au démarrage
- Dispatch intelligent CPU/GPU (seuil: 10k params/layer)
- Fallback automatique vers CPU si GPU indisponible
- **Gain potentiel**: ~100x sur layers éligibles

### 💾 Gestion Mémoire Avancée
- Allocation dynamique avec compression **LZ4** réelle (~50% économie)
- Éviction LRU automatique quand limite atteinte
- Lazy loading transparent via `getData()`
- Thread-safe avec protection mutex

## 🎯 Vue d'ensemble

Mímir est un framework de deep learning **CPU-only** moderne écrit en C++17, conçu pour rendre l'IA accessible et abordable sans nécessiter de GPU coûteux.

**Philosophy**: Entraîner et déployer des modèles d'IA performants sur n'importe quel CPU moderne.

### Pourquoi CPU-only?

- 💰 **Accessible**: Pas besoin de GPU à 1000€+, un CPU suffit
- 🌍 **Universel**: Fonctionne sur laptop, desktop, serveur, cloud
- 🔧 **Simple**: Pas de drivers CUDA/ROCm, juste GCC et OpenMP
- ⚡ **Optimisé**: AVX2, FMA, F16C exploitent le CPU au maximum
- 🎯 **Pratique**: Prototypage rapide, entraînement local, inférence edge

### Points Forts

- **Performance CPU maximale**: Optimisations AVX2, FMA, F16C, BMI2, HugePages
- **Flexibilité**: API Lua complète pour créer des modèles sans recompilation
- **Modernité**: 8 architectures state-of-the-art prêtes à l'emploi
- **Simplicité**: Interface intuitive pour le prototypage rapide
- **Coût zéro**: Aucun investissement hardware supplémentaire requis

## ✨ Fonctionnalités principales

### 🏗️ Architectures modernes pré-implémentées

| Architecture | Description | Use Cases |
|--------------|-------------|-----------|
| **UNet** | Encoder-Decoder avec skip connections | Segmentation, super-résolution, génération |
| **VAE** | Variational Autoencoder | Génération, compression, représentation latente |
| **ViT** | Vision Transformer (patch-based) | Classification d'images avec attention |
| **GAN** | Generator + Discriminator (StyleGAN) | Génération d'images réalistes |
| **Diffusion** | DDPM (Denoising Diffusion) | Génération d'images haute qualité |
| **Transformer** | GPT-style (causal masking) | Génération de texte, LLMs |
| **ResNet** | ResNet-50 avec bottleneck blocks | Classification d'images (backbone) |
| **MobileNet** | MobileNetV2 (inverted residuals) | Classification sur mobile/embarqué |

### ⚡ Optimisations CPU avancées

**Mímir exploite au maximum les capacités des CPU modernes** pour rivaliser avec les performances GPU dans de nombreux cas d'usage.

| Optimisation | Gain | Description |
|--------------|------|-------------|
| **AVX2** | 4-8× | Vectorisation SIMD 256-bit |
| **FMA** | 2-3× | Fused Multiply-Add (3 accumulateurs) |
| **F16C** | 50% mémoire | Conversion hardware FP32↔FP16 |
| **BMI2** | 3-5× quant | Bit manipulation pour quantization |
| **HugePages** | 10-30% | Pages 2MB (réduction TLB misses) |
| **OpenMP** | N× | Parallélisation multi-thread |

**Speedup global observé**: **2.5-4× sur training complet CPU**

> 💡 **Note**: Mímir est **volontairement CPU-only**. Pas de CUDA, pas de ROCm. L'objectif est de démocratiser l'IA en permettant à quiconque avec un CPU moderne de créer et entraîner des modèles sans investir dans du matériel GPU coûteux.

### 🎨 API Lua complète

```lua
-- Créer un Transformer en 10 lignes
model.create("gpt")
architectures.transformer({
    vocab_size = 50000,
    d_model = 768,
    num_layers = 12,
    num_heads = 12
})
model.allocate_params()
model.init_weights("he", 42)
-- ~117M paramètres prêts!
```

## 📦 Installation

### Prérequis

```bash
# Ubuntu/Debian
sudo apt-get install -y g++ make liblua5.3-dev

# ArchLinux
sudo pacman -S gcc make lua53

# macOS
brew install gcc lua
```

**Minimum requis**:
- GCC 7+ ou Clang 6+ (support C++17)
- CPU moderne avec AVX2 (Intel Haswell 2013+, AMD Excavator 2015+)
- 4GB RAM minimum
- **Aucun GPU requis** - Fonctionne sur n'importe quel ordinateur moderne

### Compilation

```bash
git clone https://github.com/brit45/mimir-framework.git
cd mimir-framework
make -j$(nproc)
```

**Résultat**: `bin/mimir` (1.4MB optimisé)

**Options de compilation**:
```bash
make clean              # Nettoyer
make DEBUG=1            # Build debug (symboles + asserts)
make VERBOSE=1          # Afficher les commandes complètes
```

## 🚀 Quickstart

### Option 1: Script Lua v2.0 (Asynchrone)

```lua
-- my_first_model_v2.lua
print("🚀 Mon premier modèle Transformer v2.0")

-- Configuration mémoire avec compression LZ4
allocator.configure({
    max_ram_gb = 10.0,
    enable_compression = true,
    compression_threshold_mb = 100
})

-- Monitoring asynchrone (non-bloquant)
htop.create()

-- Vérifier l'accélération GPU
local hw = model.hardware_caps()
if model.has_vulkan_compute() then
    print("✓ Accélération GPU activée (Vulkan Compute)")
else
    print("⚠ Mode CPU uniquement")
end

-- Créer le modèle
model.create("my_transformer")
architectures.transformer({
    vocab_size = 10000,
    d_model = 512,
    num_layers = 6,
    num_heads = 8
})

-- Allouer et initialiser
local success, params = model.allocate_params()
print(string.format("Paramètres: %d (%.1f MB)", 
    params, params * 2 / 1024 / 1024))

model.init_weights("he", 42)

-- Entraînement avec monitoring asynchrone
for epoch = 1, 10 do
    for batch = 1, 100 do
        -- Forward/Backward (GPU automatique si disponible)
        local output = model.forward(input)
        local loss = model.compute_loss(output, target)
        model.backward(loss_grad)
        model.optimizer_step(optimizer, lr)
        
        -- Mise à jour monitoring (thread-safe, non-bloquant)
        htop.update(epoch, 10, batch, 100, loss, avg_loss, lr)
        -- PAS BESOIN de htop.render() - c'est automatique!
    end
end

-- Statistiques finales
allocator.print_stats()
model.save("checkpoints/my_model")
```

**Exécution**:
```bash
bin/mimir --lua my_first_model_v2.lua
```

### Option 2: Script Lua v1.x (Compatibilité)

```lua
-- my_first_model.lua (ancien style, toujours supporté)
print("🚀 Mon premier modèle Transformer")

-- Créer le modèle
model.create("my_transformer")
architectures.transformer({
    vocab_size = 10000,
    d_model = 512,
    num_layers = 6,
    num_heads = 8
})

-- Allouer et initialiser
local success, params = model.allocate_params()
print(string.format("Paramètres: %d (%.1f MB)", 
    params, params * 4 / 1024 / 1024))

model.init_weights("he", 42)

-- Tokenizer
tokenizer.create(10000)
local tokens = tokenizer.tokenize("Hello world")
print("Tokens:", table.concat(tokens, ", "))

-- Forward pass
local output = model.forward(tokens)
print(string.format("Output shape: %d", #output))

-- Sauvegarder
model.save("checkpoints/my_model.safetensors")
print("✅ Modèle sauvegardé!")
```

**Exécuter**:
```bash
./bin/mimir --lua my_first_model.lua
```

### Option 2: Démo C++

```bash
# Tester une architecture
./bin/mimir --demo transformer
./bin/mimir --demo unet
./bin/mimir --demo vit

# Afficher l'aide
./bin/mimir --help
```

### Option 3: Configuration JSON

```json
{
    "architecture": "transformer",
    "transformer": {
        "vocab_size": 50000,
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "max_seq_len": 2048
    }
}
```

```bash
./bin/mimir --config config.json
```

## 🏗️ Architectures

### UNet (Segmentation / Génération)

```lua
architectures.unet({
    input_channels = 3,      -- RGB
    output_channels = 1,     -- Mask binaire
    base_channels = 64,      -- Filtres de base
    num_levels = 4,          -- Profondeur (16× downsampling)
    blocks_per_level = 2,    -- Blocs conv par niveau
    use_attention = true,    -- Self-attention aux niveaux profonds
    use_residual = true,     -- Skip connections
    use_batchnorm = true,    -- Batch normalization
    dropout = 0.1
})
-- ~15.6M paramètres
```

**Cas d'usage**: Segmentation médicale, super-résolution, inpainting

### VAE (Génération / Compression)

```lua
architectures.vae({
    input_dim = 784,           -- 28×28 MNIST
    latent_dim = 128,          -- Espace latent
    encoder_hidden = {512, 256},
    decoder_hidden = {256, 512},
    use_batchnorm = false
})
-- ~1.3M paramètres
```

**Cas d'usage**: Génération d'images, anomaly detection, compression

### Vision Transformer (Classification)

```lua
architectures.vit({
    image_size = 224,
    patch_size = 16,           -- 14×14 patches
    num_classes = 1000,        -- ImageNet
    d_model = 768,
    num_heads = 12,
    num_layers = 12,
    mlp_ratio = 4,
    dropout = 0.1,
    use_cls_token = true
})
-- ~86M paramètres (ViT-Base)
```

**Cas d'usage**: Classification d'images, feature extraction

### GAN (Génération d'images)

```lua
-- Generator
architectures.gan("generator", {
    latent_dim = 100,
    image_size = 64,
    image_channels = 3,
    g_base_channels = 64,
    self_attention = true
})
-- ~3.5M paramètres

-- Discriminator
architectures.gan("discriminator", {
    latent_dim = 100,
    image_size = 64,
    d_base_channels = 64,
    self_attention = true
})
-- ~2.8M paramètres
```

**Cas d'usage**: Génération d'images, data augmentation

### Diffusion (Génération haute qualité)

```lua
architectures.diffusion({
    image_size = 32,
    image_channels = 3,
    base_channels = 128,
    num_res_blocks = 2,
    channel_multipliers = {1, 2, 2, 2},
    attention_levels = {1, 2, 3},
    time_embed_dim = 512
})
-- ~35M paramètres
```

**Cas d'usage**: Génération d'images (Stable Diffusion, DALL-E)

### Transformer (LLM / Génération de texte)

```lua
architectures.transformer({
    vocab_size = 50000,
    max_seq_len = 2048,
    d_model = 768,
    num_heads = 12,
    num_layers = 12,
    d_ff = 3072,
    dropout = 0.1,
    causal = true              -- GPT-style
})
-- ~117M paramètres (GPT-2 small)
```

**Cas d'usage**: Génération de texte, chatbots, completion

### ResNet (Classification backbone)

```lua
architectures.resnet({
    num_classes = 1000,
    layers = {3, 4, 6, 3},     -- ResNet-50
    base_channels = 64,
    use_bottleneck = true
})
-- ~25M paramètres
```

**Cas d'usage**: Classification d'images, feature extraction

### MobileNet (Mobile / Embarqué)

```lua
architectures.mobilenet({
    num_classes = 1000,
    width_multiplier = 1.0,
    resolution = 224
})
-- ~3.5M paramètres
```

**Cas d'usage**: Classification sur mobile, edge computing

## 📚 Documentation

### 🆕 Guides v2.0

| Document | Description |
|----------|-------------|
| **[MIGRATION_V2.md](docs/MIGRATION_V2.md)** | 📖 Guide de migration v1.x → v2.0 |
| **[THREADING_AND_COMPUTE.md](docs/THREADING_AND_COMPUTE.md)** | 🧵 Threading asynchrone et accélération GPU |
| **[LUA_IDE_SETUP.md](docs/LUA_IDE_SETUP.md)** | 💡 Configuration IDE et autocomplétion Lua |

### Guides principaux

- **[Quickstart](docs/QUICKSTART.md)** - Premier modèle en 5 minutes
- **[Installation](docs/INSTALLATION.md)** - Guide d'installation détaillé
- **[API Lua](docs/LUA_API.md)** - Référence complète de l'API Lua
- **[Model Architectures](docs/MODEL_ARCHITECTURES.md)** - Détails des architectures

### Références techniques

- **[Architecture](docs/ARCHITECTURE.md)** - Architecture interne du framework
- **[Hardware Optimizations](docs/HARDWARE_OPTIMIZATIONS.md)** - Optimisations AVX2/FMA/etc.
- **[Layer Operations](docs/LAYER_OPERATIONS.md)** - Opérations de layers optimisées

### Exemples de Code v2.0

| Fichier | Description |
|---------|-------------|
| `examples/async_monitoring_demo.lua` | 🧵 Monitoring asynchrone non-bloquant |
| `scripts/example_training.lua` | ⚡ Entraînement complet avec GPU |
| `scripts/example_gpt.lua` | 🤖 Transformer GPT-style |

### Fichiers d'Aide IDE

| Fichier | Utilisation |
|---------|-------------|
| `mimir-api.lua` | Définitions LSP pour autocomplétion IDE |
| `.luarc.json` | Configuration Lua Language Server |

### Documentation API

- **[API Lua Complète](docs/LUA_API.md)** - 60+ fonctions exposées
  - `model.*` - Gestion du modèle (18 fonctions)
  - `architectures.*` - Construction d'architectures (8 fonctions)
  - `layers.*` - Opérations de layers (8 fonctions)
  - `tokenizer.*` - Tokenization (6 fonctions)
  - `dataset.*` - Chargement de données (2 fonctions)

## 🔧 API Lua - Référence rapide

### Model API

```lua
model.create(name)                           -- Créer un modèle
model.allocate_params()                      -- Allouer mémoire
model.init_weights(method, seed)             -- Initialiser poids (he/xavier/normal)
model.total_params()                         -- Nombre de paramètres
model.push_layer(name, type, params)         -- Ajouter un layer manuellement
model.forward(input)                         -- Forward pass
model.backward(loss_gradient)                -- Backward pass
model.optimizer_step(lr, type)               -- Optimisation (sgd/adam/adamw)
model.set_hardware(enable)                   -- Activer/désactiver hardware
model.hardware_caps()                        -- Capacités hardware
model.save(filepath)                         -- Sauvegarder
model.load(filepath)                         -- Charger
```

### Architectures API

```lua
architectures.unet(config)                   -- Construire UNet
architectures.vae(config)                    -- Construire VAE
architectures.vit(config)                    -- Construire Vision Transformer
architectures.gan(type, config)              -- Construire GAN (generator/discriminator)
architectures.diffusion(config)              -- Construire Diffusion Model
architectures.transformer(config)            -- Construire Transformer (GPT)
architectures.resnet(config)                 -- Construire ResNet
architectures.mobilenet(config)              -- Construire MobileNet
```

### Tokenizer API

```lua
tokenizer.create(vocab_size)                 -- Créer tokenizer
tokenizer.tokenize(text)                     -- Tokenizer texte
tokenizer.detokenize(tokens)                 -- Détokenizer
tokenizer.vocab_size()                       -- Taille vocabulaire
tokenizer.save(filepath)                     -- Sauvegarder
tokenizer.load(filepath)                     -- Charger
```

### Utilitaires

```lua
log(message)                                 -- Logger
read_json(filepath)                          -- Lire JSON
write_json(filepath, data)                   -- Écrire JSON
```

## 🎓 Exemples

### Exemple 1: Classification avec ResNet

```lua
-- Créer ResNet-50
model.create("resnet50")
architectures.resnet({num_classes = 1000})
model.allocate_params()
model.init_weights("he")

-- Forward pass (batch de 1 image 224×224×3)
local image = {} -- 224*224*3 = 150528 valeurs
for i = 1, 150528 do
    image[i] = math.random() -- Données aléatoires
end

local logits = model.forward(image)
print("Prédictions:", #logits) -- 1000 classes
```

### Exemple 2: Génération de texte avec Transformer

```lua
-- Créer GPT-style model
model.create("gpt")
architectures.transformer({
    vocab_size = 50000,
    d_model = 768,
    num_layers = 12,
    num_heads = 12,
    causal = true
})
model.allocate_params()
model.init_weights("xavier")

-- Tokenizer
tokenizer.create(50000)
local prompt = "Il était une fois"
local tokens = tokenizer.tokenize(prompt)

-- Générer 50 tokens
for i = 1, 50 do
    local output = model.forward(tokens)
    local next_token = output[#output] -- Dernier token
    table.insert(tokens, math.floor(next_token))
end

local generated = tokenizer.detokenize(tokens)
print("Texte généré:", generated)
```

### Exemple 3: Segmentation avec UNet

```lua
-- Créer UNet pour segmentation médicale
model.create("medical_unet")
architectures.unet({
    input_channels = 1,    -- Grayscale
    output_channels = 1,   -- Mask binaire
    base_channels = 32,
    num_levels = 4
})
model.allocate_params()
model.init_weights("he")

-- Training loop (pseudo-code)
for epoch = 1, 100 do
    -- Forward
    local prediction = model.forward(image)
    
    -- Compute loss (BCE)
    local loss_grad = {} -- Calculer gradient
    
    -- Backward
    model.backward(loss_grad)
    
    -- Optimizer step
    model.optimizer_step(0.0001, "adamw")
    
    if epoch % 10 == 0 then
        model.save(string.format("checkpoints/epoch_%d.safetensors", epoch))
    end
end
```

## 🔬 Performance

### Benchmarks (training loop complet)

| Optimisation | Temps (ms/iter) | Speedup |
|--------------|-----------------|---------|
| Baseline CPU | 245 ms | 1.0× |
| + OpenMP | 98 ms | 2.5× |
| + AVX2 | 72 ms | 3.4× |
| + FMA | 61 ms | 4.0× |
| + HugePages | 55 ms | 4.5× |

**Configuration**: ResNet-50, batch=32, 224×224, Intel i7-9700K

### Comparaison avec PyTorch

| Framework | Conv2D (ms) | MatMul (ms) | Attention (ms) |
|-----------|-------------|-------------|----------------|
| Mímir (AVX2+FMA) | 12.3 | 8.7 | 15.2 |
| PyTorch CPU | 18.5 | 11.2 | 22.1 |
| **Speedup** | **1.5×** | **1.3×** | **1.5×** |

## 🏛️ Architecture du code

```
mimir-framework/
├── src/
│   ├── main.cpp                    # Point d'entrée
│   ├── Model.hpp/cpp               # Classe Model principale
│   ├── LuaScripting.hpp/cpp        # Bindings Lua
│   ├── tensors.hpp/cpp             # Système de tenseurs
│   ├── Autograd.hpp                # Backpropagation
│   ├── Layers.hpp                  # Définitions de layers
│   ├── SIMD_Ops.hpp                # Optimisations SIMD
│   ├── Tokenizer.hpp/cpp           # BPE Tokenizer
│   ├── Encoder.hpp/cpp             # Encoder utilities
│   ├── Visualizer.hpp/cpp          # Visualisation SFML
│   ├── Models/
│   │   └── ModelArchitectures.hpp  # 8 architectures modernes
│   └── include/
│       └── json.hpp                # nlohmann/json
├── scripts/
│   ├── test_lua_api.lua            # Tests complets API
│   ├── example_simple.lua          # Exemple simple
│   ├── example_gpt.lua             # Exemple GPT
│   └── example_training.lua        # Exemple training
├── docs/
│   ├── README.md                   # Index documentation
│   ├── QUICKSTART.md               # Guide rapide
│   ├── LUA_API.md                  # API Lua complète
│   ├── ARCHITECTURE.md             # Architecture framework
│   ├── HARDWARE_OPTIMIZATIONS.md   # Optimisations hardware
│   └── MODEL_ARCHITECTURES.md      # Détails architectures
├── checkpoints/                    # Modèles sauvegardés
├── bin/
│   └── mimir                       # Binary compilé
├── Makefile                        # Build system
└── README.md                       # Ce fichier
```

## 🤝 Contribution

Contributions bienvenues! Consultez [CONTRIBUTING.md](CONTRIBUTING.md).

### Roadmap

#### Features Planifiées

- [ ] Mixed precision training (FP16/FP32)
- [ ] Gradient clipping et accumulation
- [ ] Learning rate schedulers
- [ ] Plus d'architectures (CLIP, Whisper, Stable Diffusion)
- [ ] Bindings Python (pybind11)
- [ ] Distributed training CPU (MPI)
- [ ] Optimisations ARM Neon (Raspberry Pi, Apple Silicon)

#### Non-Goals

- ❌ **Support GPU (CUDA/ROCm)** - Mímir reste volontairement CPU-only
- ❌ **Cloud inference payant** - Tout reste local et gratuit
- ❌ **Dépendances complexes** - Maximum de simplicité

> 🎯 **Mission**: Garder l'IA accessible à tous, sans barrière financière ou technique.

## 📄 License

MIT License - voir [LICENSE](LICENSE)

## 🙏 Remerciements

- **nlohmann/json** - JSON parsing
- **Lua 5.3** - Scripting engine
- Architecture inspirée de PyTorch et JAX

## 📞 Contact

- GitHub: [@brit45](https://github.com/brit45)
- Issues: [github.com/brit45/mimir-framework/issues](https://github.com/brit45/mimir-framework/issues)

---

<div align="center">

**Fait avec ❤️ en C++17**

[⬆ Retour en haut](#mímir-framework-v20)

</div>
