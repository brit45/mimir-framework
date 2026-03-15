# Guide de Démarrage Rapide - Mímir

## 🚀 Installation en 5 Minutes

### Prérequis

```bash
# Ubuntu/Debian (minimum)
sudo apt-get update
sudo apt-get install -y \
    cmake \
    g++ \
    make \
    liblua5.3-dev \
    liblz4-dev

# macOS (minimum)
brew install cmake gcc make lua lz4
```

### Compilation (C++)

```bash
cd /path/to/mimir

# Build via Makefile (wrapper CMake)
make

# Ou directement via CMake
# cmake -S . -B build && cmake --build build -j
```

**Résultat** : Binaire `bin/mimir`

---

## 🎯 Premier Modèle en 2 Minutes

### 1. Script Lua Minimal

Créez `my_first_model.lua` :

```lua
log("🚀 Création de mon premier modèle avec Mímir!")

-- (Recommandé) Configuration allocator RAM
Mimir.Allocator.configure({
    max_ram_gb = 10.0,
    enable_compression = true
})

-- (Optionnel) Activer l'accélération matérielle si dispo
local caps = Mimir.Model.hardware_caps()
if caps.avx2 or caps.fma then
    pcall(Mimir.Model.set_hardware, true)
end

-- 1) Config modèle via registre (source de vérité côté C++)
local cfg, err = Mimir.Architectures.default_config("transformer")
if not cfg then error(err) end

cfg.vocab_size = 32000
cfg.d_model = 256
cfg.num_layers = 4
cfg.num_heads = 8
cfg.max_seq_len = 256

-- 2) Créer (instancie l'archi)
local ok_create, create_err = Mimir.Model.create("transformer", cfg)
if not ok_create then error(create_err) end

-- 3) (Optionnel) rebuild compat
local ok_build, params_or_err = Mimir.Model.build()
if not ok_build then error(params_or_err) end

-- 4) Allouer + init
local ok_alloc, params_or_err2 = Mimir.Model.allocate_params()
if not ok_alloc then error(params_or_err2) end
local ok_init, init_err = Mimir.Model.init_weights("xavier", 42)
if ok_init == false then error(init_err) end

log(string.format("✓ Modèle prêt (%d paramètres)", tonumber(params_or_err2) or 0))

-- 5) Dataset (texte)
local ok_ds, ds_err = Mimir.Dataset.load("dataset")
if not ok_ds then error(ds_err) end
local ok_prep, prep_err = Mimir.Dataset.prepare_sequences(cfg.max_seq_len)
if not ok_prep then error(prep_err) end

-- Entraîner le modèle
log("📚 Début de l'entraînement...")
Mimir.Model.train(10, 1e-4)  -- 10 epochs, LR = 1e-4

-- Sauvegarder
-- Recommandé v2.3+: sérialisation unifiée
Mimir.Serialization.save("checkpoints/my_first_model.safetensors", "safetensors")
log("✓ Modèle sauvegardé!")
```

### 2. Exécution

```bash
./bin/mimir --lua my_first_model.lua

# Passer des arguments au script (optionnel):
# ./bin/mimir --lua my_first_model.lua -- --epochs 10 --lr 1e-4
```

---

## 📚 Exemples Rapides

### Encoder (BERT-like) pour Classification

```lua
-- Encoder pour classification de texte
Mimir.Tokenizer.create(30000)

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

Mimir.Model.create("encoder", encoder_config)
Mimir.Model.build()

Mimir.Dataset.load("datasets/text")
model.train(20, 0.0001)
model.save("checkpoints/bert_classifier")
```

### Decoder (GPT-like) pour Génération

```lua
-- Decoder pour génération de texte
Mimir.Tokenizer.create(50000)

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

Mimir.Model.create("decoder", decoder_config)
Mimir.Model.build()

Mimir.Dataset.load("datasets/text")
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

Mimir.Model.create("unet", unet_config)
Mimir.Model.build()

Mimir.Dataset.load("datasets/images")
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

Mimir.Model.create("vit", vit_config)
Mimir.Model.build()

Mimir.Dataset.load("datasets/imagenet")
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
Mimir.Model.create(config.model.type, config.model)
Mimir.Model.build()

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
# Devrait produire bin/mimir sans erreurs
```

### Test 2 : Exécution Simple

```bash
cat > test.lua << 'EOF'
log("Mímir est installé correctement!")
Mimir.Tokenizer.create(1000)
log("Tokenizer créé avec succès")
EOF

./bin/mimir --lua test.lua
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
