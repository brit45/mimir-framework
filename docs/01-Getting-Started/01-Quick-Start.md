# Guide de Démarrage Rapide - Mímir

## 🚀 Installation en 5 Minutes

### Prérequis

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    g++ \
    make \
    libopencl-dev \
    libsfml-dev \
    liblua5.3-dev

# macOS
brew install gcc make opencl-headers sfml lua
```

### Compilation

```bash
cd /path/to/mimir
make clean
make
```

**Résultat** : Binaire `bin/unet` (1.5 MB)

---

## 🎯 Premier Modèle en 2 Minutes

### 1. Script Lua Minimal

Créez `my_first_model.lua` :

```lua
-- Configuration du modèle
log("🚀 Création de mon premier modèle avec Mímir!")

-- Créer un tokenizer
tokenizer.create(32000)

-- Configuration du modèle Encoder (BERT-like)
local config = {
    num_layers = 6,
    d_model = 512,
    num_heads = 8,
    vocab_size = 32000,
    max_seq_len = 512,
    dropout = 0.1
}

-- Créer et construire le modèle
model.create("encoder", config)
local ok, num_params = model.build()

if ok then
    log(string.format("✓ Modèle créé avec %d paramètres", num_params))
else
    log("✗ Erreur lors de la création du modèle")
end

-- Charger un dataset
dataset.load("datasets/text")
dataset.prepare_sequences(512)

-- Entraîner le modèle
log("📚 Début de l'entraînement...")
model.train(10, 0.0001)  -- 10 epochs, LR = 0.0001

-- Sauvegarder
model.save("checkpoints/my_first_model")
log("✓ Modèle sauvegardé!")
```

### 2. Exécution

```bash
./bin/unet --script my_first_model.lua
```

---

## 📚 Exemples Rapides

### Encoder (BERT-like) pour Classification

```lua
-- Encoder pour classification de texte
tokenizer.create(30000)

local encoder_config = {
    num_layers = 12,
    num_heads = 12,
    d_model = 768,
    d_ff = 3072,
    max_seq_len = 512,
    vocab_size = 30000,
    dropout = 0.1,
    use_prenorm = true,
    pooling = "cls"  -- Utiliser le token [CLS]
}

model.create("encoder", encoder_config)
model.build()

dataset.load("datasets/text")
model.train(20, 0.0001)
model.save("checkpoints/bert_classifier")
```

### Decoder (GPT-like) pour Génération

```lua
-- Decoder pour génération de texte
tokenizer.create(50000)

local decoder_config = {
    num_layers = 12,
    num_heads = 12,
    d_model = 768,
    d_ff = 3072,
    max_seq_len = 1024,
    vocab_size = 50000,
    use_causal_mask = true,  -- Auto-régressif
    dropout = 0.1
}

model.create("decoder", decoder_config)
model.build()

dataset.load("datasets/text")
model.train(50, 0.0003)

-- Générer du texte
local prompt = "Once upon a time"
local generated = model.infer(prompt)
log("Texte généré: " .. generated)
```

### U-Net pour Segmentation d'Images

```lua
-- U-Net pour segmentation
local unet_config = {
    in_channels = 3,      -- RGB
    out_channels = 21,    -- 21 classes de segmentation
    base_channels = 64,
    num_levels = 4,
    use_attention = true,
    attention_levels = {2, 3}
}

model.create("unet", unet_config)
model.build()

dataset.load("datasets/images")
model.train(100, 0.0002)
model.save("checkpoints/segmentation_model")
```

### Vision Transformer (ViT) pour Classification

```lua
-- ViT pour classification d'images
local vit_config = {
    image_size = 224,
    patch_size = 16,
    in_channels = 3,
    num_classes = 1000,  -- ImageNet
    d_model = 768,
    num_layers = 12,
    num_heads = 12,
    d_ff = 3072,
    dropout = 0.1,
    use_class_token = true
}

model.create("vit", vit_config)
model.build()

dataset.load("datasets/imagenet")
model.train(300, 0.0003)
model.save("checkpoints/vit_imagenet")
```

---

## 🛠️ Configuration via JSON

Créez `config.json` :

```json
{
  "model": {
    "type": "encoder",
    "num_layers": 6,
    "d_model": 512,
    "num_heads": 8,
    "vocab_size": 32000
  },
  "training": {
    "num_epochs": 50,
    "learning_rate": 0.0001,
    "batch_size": 32,
    "optimizer": "adamW",
    "weight_decay": 0.01,
    "warmup_steps": 4000,
    "lr_decay": {
      "enabled": true,
      "strategy": "cosine",
      "min_lr_ratio": 0.1
    }
  },
  "dataset": {
    "dir": "datasets/text",
    "cache_enabled": true,
    "lazy_loading": true
  }
}
```

Utilisation en Lua :

```lua
local config = read_json("config.json")
model.create(config.model.type, config.model)
model.build()

-- Utiliser les paramètres de training
local num_epochs = config.training.num_epochs
local lr = config.training.learning_rate
model.train(num_epochs, lr)
```

---

## 📊 Utilisation en C++

### Exemple Simple

```cpp
#include "Model.hpp"
#include "Model.hpp"

int main() {
    // Configuration
    EncoderModel::Config config;
    config.vocab_size = 30000;
    config.embed_dim = 768;
    config.num_layers = 12;
    config.num_heads = 12;
    
    // Créer modèle
    EncoderModel model(config);
    model.buildArchitecture();
    model.allocateParams();
    
    std::cout << "Paramètres: " << model.totalParamCount() << std::endl;
    
    // Forward pass
    std::vector<int> tokens = {1, 2, 3, 4, 5};
    auto output = model.encode(tokens);
    
    // Pooling
    auto pooled = model.pool(output, 
        EncoderModel::PoolingStrategy::CLS_TOKEN);
    
    return 0;
}
```

### Compilation

```bash
g++ -std=c++17 -O3 -march=native -mavx2 -fopenmp \
    my_program.cpp \
    src/Model.cpp \
    src/tensors.cpp \
    src/Encoder.cpp \
    src/Tokenizer.cpp \
    -I./src \
    -o my_program \
    -lOpenCL -lsfml-graphics -lsfml-window -lsfml-system
```

---

## 🔍 Vérification de l'Installation

### Test 1 : Compilation

```bash
make clean && make
# Devrait produire bin/unet sans erreurs
```

### Test 2 : Exécution Simple

```bash
cat > test.lua << 'EOF'
log("Mímir est installé correctement!")
tokenizer.create(1000)
log("Tokenizer créé avec succès")
EOF

./bin/unet --script test.lua
```

### Test 3 : Optimisations SIMD

```bash
# Vérifier support AVX2
grep -q avx2 /proc/cpuinfo && echo "AVX2: ✓" || echo "AVX2: ✗"

# Vérifier OpenCL
clinfo | head -20
```

---

## 🐛 Dépannage

### Erreur : `lua.h: No such file`

```bash
# Ubuntu/Debian
sudo apt-get install liblua5.3-dev

# macOS
brew install lua
```

### Erreur : `CL/cl.h: No such file`

```bash
# Ubuntu/Debian
sudo apt-get install opencl-headers ocl-icd-opencl-dev

# macOS
brew install opencl-headers
```

### Erreur : SFML non trouvé

```bash
# Ubuntu/Debian
sudo apt-get install libsfml-dev

# macOS
brew install sfml
```

### Segmentation Fault

Vérifiez :
1. Le dataset existe
2. La configuration JSON est valide
3. Les chemins sont corrects
4. Suffisamment de RAM disponible

---

## 📖 Prochaines Étapes

1. **Tutoriels** : Consultez les autres guides dans [Getting Started](../01-Getting-Started/)
2. **API Lua** : Voir [API Reference](../03-API-Reference/00-API-Complete.md) pour la référence complète
3. **Exemples** : Explorez `scripts/` pour plus d'exemples
4. **Architecture** : Lisez [System Architecture](../04-Architecture-Internals/01-System-Architecture.md) pour comprendre le design

---

## 🎯 Ressources

- **Documentation** : [INDEX](../00-INDEX.md)
- **Scripts Exemples** : `scripts/example_*.lua`
- **Config Exemples** : `config.json`
- **Code Exemples** : `examples/*.cpp`

---

**Félicitations !** 🎉 Vous êtes prêt à utiliser Mímir pour vos projets de deep learning.
