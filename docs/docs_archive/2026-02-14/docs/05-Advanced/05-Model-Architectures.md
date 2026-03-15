# Mímir Framework v2.0 - Model Architectures

## Vue d'ensemble

Le framework Mímir propose maintenant **8 architectures modernes prêtes à l'emploi**, avec optimisations hardware (AVX2, FMA, F16C) et support multi-threading (OpenMP).

## 🏗️ Architectures Disponibles

### 1. **UNet** - Segmentation d'images
- **Usage**: Segmentation, image-to-image, médical imaging
- **Architecture**: Encoder-decoder avec skip connections
- **Paramètres**: ~15M (config par défaut)
- **Caractéristiques**:
  - 4 niveaux de downsampling/upsampling
  - Attention multi-head aux niveaux profonds
  - Residual connections
  - Batch normalization

```cpp
#include "Models/Registry/ModelArchitectures.hpp"
using namespace ModelArchitectures;

Model model;
UNetConfig config;
config.input_channels = 3;
config.output_channels = 1;
config.base_channels = 64;
config.num_levels = 4;
config.use_attention = true;

buildUNet(model, config);
model.allocateParams();
model.initializeWeights("he");
```

### 2. **VAE** - Variational Autoencoder
- **Usage**: Génération, compression, apprentissage de représentations
- **Architecture**: Encoder → Latent (μ, σ) → Decoder
- **Paramètres**: ~1.2M (MNIST config)
- **Caractéristiques**:
  - Reparameterization trick
  - KL divergence loss
  - Configurable hidden layers

```cpp
VAEConfig config;
config.input_dim = 784;      // 28x28 MNIST
config.latent_dim = 128;
config.encoder_hidden = {512, 256};
config.decoder_hidden = {256, 512};

buildVAE(model, config);
```

### 3. **Vision Transformer (ViT)** - Classification
- **Usage**: Classification d'images, feature extraction
- **Architecture**: Patch embedding + Transformer encoder
- **Paramètres**: ~86M (ViT-Base)
- **Caractéristiques**:
  - Patch size 16x16
  - 12 layers, 12 heads
  - CLS token pour classification
  - Position embeddings

```cpp
ViTConfig config;
config.image_size = 224;
config.patch_size = 16;
config.num_classes = 1000;
config.d_model = 768;
config.num_heads = 12;
config.num_layers = 12;

buildViT(model, config);
```

### 4. **GAN** - Generative Adversarial Network
- **Usage**: Génération d'images réalistes
- **Architecture**: Generator + Discriminator
- **Paramètres**: ~3.6M (Generator), ~3M (Discriminator)
- **Caractéristiques**:
  - Self-attention aux résolutions moyennes
  - Spectral normalization (optionnel)
  - Progressive upsampling (Generator)
  - Progressive downsampling (Discriminator)

```cpp
GANConfig config;
config.latent_dim = 100;
config.image_size = 64;
config.self_attention = true;

Model generator, discriminator;
buildGenerator(generator, config);
buildDiscriminator(discriminator, config);
```

### 5. **Diffusion Model (DDPM)** - Génération haute qualité
- **Usage**: Génération d'images, super-resolution, inpainting
- **Architecture**: U-Net avec time conditioning
- **Paramètres**: ~40M
- **Caractéristiques**:
  - Time embedding MLP
  - ResBlocks avec scale/shift normalization
  - Multi-level attention
  - Symmetric encoder-decoder

```cpp
DiffusionConfig config;
config.image_size = 32;
config.base_channels = 128;
config.num_res_blocks = 2;
config.channel_multipliers = {1, 2, 2, 2};
config.attention_levels = {1, 2, 3};

buildDiffusion(model, config);
```

### 6. **Transformer (GPT-style)** - Génération de texte
- **Usage**: Language modeling, text generation, completion
- **Architecture**: Decoder-only transformer avec causal masking
- **Paramètres**: ~163M (config small)
- **Caractéristiques**:
  - Causal attention (autoregressive)
  - Position embeddings
  - Layer normalization
  - Feed-forward networks

```cpp
TransformerConfig config;
config.vocab_size = 50000;
config.max_seq_len = 1024;
config.d_model = 768;
config.num_heads = 12;
config.num_layers = 12;
config.causal = true;  // GPT-style

buildTransformer(model, config);
```

### 7. **ResNet** - Classification robuste
- **Usage**: Classification ImageNet, transfer learning
- **Architecture**: Deep residual network avec bottlenecks
- **Paramètres**: ~2M (ResNet-50)
- **Caractéristiques**:
  - Skip connections
  - Bottleneck blocks (1x1 → 3x3 → 1x1)
  - Batch normalization
  - Architecture [3, 4, 6, 3]

```cpp
ResNetConfig config;
config.num_classes = 1000;
config.layers = {3, 4, 6, 3};  // ResNet-50
config.use_bottleneck = true;

buildResNet(model, config);
```

### 8. **MobileNetV2** - Efficient mobile
- **Usage**: Mobile/embedded deployment, edge devices
- **Architecture**: Inverted residuals + Depthwise separable convolutions
- **Paramètres**: ~3.5M
- **Caractéristiques**:
  - Width multiplier pour scaling
  - Depthwise separable convolutions
  - Inverted residual blocks
  - Linear bottlenecks

```cpp
MobileNetConfig config;
config.num_classes = 1000;
config.width_multiplier = 1.0f;
config.resolution = 224;

buildMobileNetV2(model, config);
```

## 🚀 Utilisation

### Méthode 1: Ligne de commande

```bash
# Démonstration d'une architecture
./bin/mimir --demo unet
./bin/mimir --demo transformer
./bin/mimir --demo gan

# Depuis un fichier de config
./bin/mimir --config config.json
```

### Méthode 2: Programme complet

```bash
# Voir toutes les architectures avec statistiques
./bin/model_architectures_demo
```

### Méthode 3: API C++

```cpp
#include "Models/Registry/ModelArchitectures.hpp"

using namespace ModelArchitectures;

// Créer un modèle
Model model;
UNetConfig config;
// ... configurer ...
buildUNet(model, config);

// Allouer et initialiser
model.allocateParams();
model.initializeWeights("he");

// Forward pass
std::vector<float> input(3 * 224 * 224);
std::vector<float> output = model.forwardPass(input);

// Backward pass
Gradients grads = model.backwardPass(loss_gradient);

// Optimizer step
Optimizer optimizer;
model.optimizerStep(optimizer, learning_rate);
```

## ⚡ Optimisations Hardware

Toutes les architectures bénéficient automatiquement de:

- **AVX2**: Vectorisation 256-bit (8 floats simultanés)
- **FMA**: Fused Multiply-Add saturé (3 ops/cycle)
- **F16C**: Conversion hardware FP16↔FP32
- **BMI2**: Bit manipulation avancée
- **OpenMP**: Parallélisation multi-thread

### Activation/Désactivation

```cpp
// Désactiver globalement
Model::setHardwareAcceleration(false);

// Par opération
Model::computeConv2D(input, output, params, h, w, ic, oc, false);  // Force CPU
```

### Détection des capacités

```cpp
if (Model::hasAVX2()) {
    std::cout << "AVX2 disponible\n";
}
if (Model::hasFMA()) {
    std::cout << "FMA disponible\n";
}
```

## 📊 Benchmarks

Architecture | Params | FP32 Size | FP16 Size | Forward (ms) | Backward (ms)
------------|--------|-----------|-----------|--------------|---------------
UNet        | 15.6M  | 59.4 MB   | 29.7 MB   | 12.3         | 24.8
VAE         | 1.2M   | 4.5 MB    | 2.2 MB    | 0.8          | 1.6
ViT         | 86.6M  | 330 MB    | 165 MB    | 45.2         | 91.5
GAN (Gen)   | 3.6M   | 13.8 MB   | 6.9 MB    | 5.4          | 10.9
Diffusion   | 39.9M  | 152 MB    | 76 MB     | 28.6         | 57.4
Transformer | 163M   | 621 MB    | 310 MB    | 89.3         | 178.9
ResNet-50   | 2.0M   | 7.7 MB    | 3.8 MB    | 3.2          | 6.5
MobileNetV2 | 3.5M   | 13.4 MB   | 6.7 MB    | 2.1          | 4.3

*CPU: AMD Ryzen, 6 threads, AVX2+FMA*

## 🎯 Cas d'Usage

### Segmentation médicale
```cpp
UNetConfig config;
config.input_channels = 1;    // Grayscale
config.output_channels = 3;   // 3 classes
config.base_channels = 32;    // Smaller for 3D
buildUNet(model, config);
```

### Generation de visages (GAN)
```cpp
GANConfig config;
config.latent_dim = 512;
config.image_size = 256;
config.g_base_channels = 64;
config.d_base_channels = 64;
config.self_attention = true;
config.spectral_norm = true;
```

### Language model (GPT-2 style)
```cpp
TransformerConfig config;
config.vocab_size = 50257;
config.max_seq_len = 2048;
config.d_model = 768;
config.num_heads = 12;
config.num_layers = 12;
config.d_ff = 3072;
config.causal = true;
```

### Mobile classification
```cpp
MobileNetConfig config;
config.num_classes = 1000;
config.width_multiplier = 0.75f;  // Faster
config.resolution = 192;          // Lower res
```

## 📝 Configuration JSON

```json
{
  "architecture": "unet",
  "unet": {
    "input_channels": 3,
    "output_channels": 1,
    "base_channels": 64,
    "num_levels": 4,
    "blocks_per_level": 2,
    "use_attention": true
  },
  "training": {
    "learning_rate": 1e-4,
    "batch_size": 4,
    "optimizer": "adamw",
    "weight_decay": 0.01
  }
}
```

## 🔧 Compilation

```bash
# Tout compiler
make clean && make

# Uniquement le framework
make bin/mimir

# Uniquement l'exemple
make bin/model_architectures_demo
```

## 📚 Documentation Complète

- `docs/LAYER_OPERATIONS.md` - Opérations de layers détaillées
- `docs/ARCHITECTURE.md` - Architecture du framework
- `docs/API_CPP.md` - API C++ complète

## 🎓 Exemples Avancés

Voir le dossier `examples/` pour:
- `model_architectures_demo.cpp` - Démonstration complète
- Scripts Lua pour configuration et entraînement

## 🤝 Contribution

Les architectures sont modulaires et extensibles. Pour ajouter une nouvelle architecture:

1. Définir la config structure dans `ModelArchitectures.hpp`
2. Implémenter `buildXXX(Model&, Config&)` 
3. Utiliser `model.push(name, type, params_count)` pour chaque layer
4. Tester avec `--demo xxx`

## 📄 License

MIT License - Voir LICENSE pour détails
