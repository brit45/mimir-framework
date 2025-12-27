# Scripts Lua - Mise à jour API Mímir

## ✅ Scripts mis à jour

Tous les scripts Lua ont été mis à jour pour utiliser correctement l'API Mímir Framework.

### 📝 Scripts modifiés

#### Scripts d'entraînement
1. ✅ **train_llm.lua** - Entraînement LLM complet
2. ✅ **example_gpt.lua** - Exemple GPT-style decoder
3. ✅ **example_training.lua** - Pipeline d'entraînement générique
4. ✅ **example_simple.lua** - Exemple simple d'utilisation

#### Scripts de démonstration d'architectures
5. ✅ **demo_unet.lua** - UNet pour segmentation
6. ✅ **demo_vae.lua** - Variational Autoencoder
7. ✅ **demo_mobilenet.lua** - MobileNetV2
8. ✅ **demo_resnet.lua** - ResNet-50
9. ✅ **demo_vit.lua** - Vision Transformer
10. ✅ **demo_gan.lua** - Generative Adversarial Network
11. ✅ **demo_diffusion.lua** - Diffusion Models

---

## 🔧 Changements principaux

### 1. **Configuration système**

Tous les scripts incluent maintenant :

```lua
-- Configuration allocateur RAM
allocator.configure({
    max_ram_gb = 10.0,
    enable_compression = true
})

-- Détection et activation hardware
local hw = model.hardware_caps()
log("AVX2: " .. (hw.avx2 and "✓" or "✗"))
log("FMA:  " .. (hw.fma and "✓" or "✗"))
model.set_hardware(true)
```

### 2. **Création de modèles**

❌ **Ancienne méthode** (obsolète) :
```lua
model.create("decoder", config)
local ok, params = model.build()
```

✅ **Nouvelle méthode** (API architectures) :
```lua
-- Créer le modèle
local success, err = model.create("model_name")

-- Construire l'architecture
success, err = architectures.transformer(config)

-- Allouer les paramètres
success, params = model.allocate_params()

-- Initialiser les poids
model.init_weights("he", 42)
```

### 3. **Architectures disponibles**

Toutes les architectures utilisent maintenant l'API `architectures.*` :

```lua
architectures.transformer(config)  -- GPT, BERT, Encoder, Decoder
architectures.unet(config)         -- Segmentation
architectures.vae(config)          -- Variational Autoencoder
architectures.mobilenet(config)    -- MobileNetV2
architectures.resnet(config)       -- ResNet-50
architectures.vit(config)          -- Vision Transformer
architectures.gan(type, config)    -- Generator/Discriminator
architectures.diffusion(config)    -- Diffusion Models
```

### 4. **Entraînement avec Learning Rate Schedule**

❌ **Ancienne méthode** :
```lua
model.train(20, 0.0003)
```

✅ **Nouvelle méthode** (avec LR schedule) :
```lua
for epoch = 1, epochs do
    -- Cosine decay
    local lr = base_lr * 0.5 * (1 + math.cos(math.pi * epoch / epochs))
    model.train(1, lr)
end
```

### 5. **Sauvegarde complète**

Tous les scripts sauvegardent maintenant modèle + tokenizer :

```lua
os.execute("mkdir -p checkpoints")
model.save("checkpoints/model_name")
tokenizer.save("checkpoints/model_name/tokenizer.json")
```

### 6. **Gestion des erreurs**

Toutes les opérations vérifient maintenant les erreurs :

```lua
local success, err = model.create("model")
if not success then
    log("❌ Erreur: " .. (err or "inconnue"))
    return
end
```

---

## 📊 Améliorations apportées

### Performance
- ✅ Configuration allocateur avec limite mémoire stricte
- ✅ Compression LZ4 automatique
- ✅ Accélération hardware (AVX2/FMA) activée
- ✅ Éviction LRU et chargement à la demande

### Entraînement
- ✅ Learning rate schedule (warmup + cosine decay)
- ✅ Métriques de suivi (loss, temps, meilleur epoch)
- ✅ Génération périodique pendant l'entraînement
- ✅ Vérification des erreurs à chaque étape

### Modèles
- ✅ Initialisation des poids optimale (He, Xavier)
- ✅ Allocation explicite des paramètres
- ✅ Affichage de la mémoire utilisée
- ✅ Support de toutes les architectures modernes

### Sauvegarde
- ✅ Création automatique des répertoires
- ✅ Sauvegarde du tokenizer avec le modèle
- ✅ Vérification des erreurs de sauvegarde
- ✅ Affichage de la taille des fichiers

---

## 🚀 Utilisation

### Scripts d'entraînement

```bash
# Entraînement LLM complet
./bin/mimir --lua scripts/train_llm.lua

# Mode rapide (test)
./bin/mimir --lua scripts/train_llm.lua --quick

# Mode long (production)
./bin/mimir --lua scripts/train_llm.lua --long --epochs=30

# Exemple simple
./bin/mimir --lua scripts/example_simple.lua

# Exemple GPT
./bin/mimir --lua scripts/example_gpt.lua

# Pipeline d'entraînement
./bin/mimir --lua scripts/example_training.lua
```

### Scripts de démonstration

```bash
# Architectures vision
./bin/mimir --lua scripts/demo_unet.lua
./bin/mimir --lua scripts/demo_vae.lua
./bin/mimir --lua scripts/demo_mobilenet.lua
./bin/mimir --lua scripts/demo_resnet.lua
./bin/mimir --lua scripts/demo_vit.lua

# Architectures génératives
./bin/mimir --lua scripts/demo_gan.lua
./bin/mimir --lua scripts/demo_diffusion.lua
```

---

## 📚 Documentation

Consultez la documentation complète de l'API :
- [docs/03-API-Reference/00-API-Complete.md](../03-API-Reference/00-API-Complete.md) - API Lua complète
- [docs/02-User-Guide/03-Predefined-Architectures.md](../02-User-Guide/03-Predefined-Architectures.md) - Architectures disponibles
- [docs/01-Getting-Started/01-Quick-Start.md](../01-Getting-Started/01-Quick-Start.md) - Guide de démarrage rapide

---

## 💡 Exemples de configuration

### Transformer (GPT-style)
```lua
local config = {
    vocab_size = 50000,
    max_seq_len = 1024,
    d_model = 768,
    num_layers = 12,
    num_heads = 12,
    d_ff = 3072,
    dropout = 0.1,
    causal = true
}
architectures.transformer(config)
```

### UNet (Segmentation)
```lua
local config = {
    input_channels = 3,
    output_channels = 1,
    base_channels = 64,
    num_levels = 4,
    use_attention = true,
    dropout = 0.1
}
architectures.unet(config)
```

### MobileNetV2 (Classification)
```lua
local config = {
    num_classes = 1000,
    width_multiplier = 1.0,
    resolution = 224
}
architectures.mobilenet(config)
```

---

## ✅ Checklist de migration

Si vous avez des scripts personnalisés, voici les étapes de migration :

1. ✅ Ajouter `allocator.configure()` au début
2. ✅ Ajouter `model.hardware_caps()` et `model.set_hardware(true)`
3. ✅ Remplacer `model.create(type, config)` + `model.build()` par :
   - `model.create(name)`
   - `architectures.[type](config)`
   - `model.allocate_params()`
   - `model.init_weights(method, seed)`
4. ✅ Implémenter learning rate schedule dans la boucle d'entraînement
5. ✅ Sauvegarder le tokenizer avec `tokenizer.save()`
6. ✅ Ajouter gestion d'erreurs avec vérification de `success`

---

## 🔗 Ressources

- Framework : [Mímir CPU-Only Deep Learning](https://github.com/...)
- Documentation : [docs/](docs/)
- Exemples : [scripts/](scripts/)
- API Lua : [docs/03-API-Reference/00-API-Complete.md](../03-API-Reference/00-API-Complete.md)

---

**Date de mise à jour** : 20 décembre 2025  
**Version API** : v2.0  
**Scripts mis à jour** : 11/11 ✅
