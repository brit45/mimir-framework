# Scripts Lua - Mise à jour API Mímir

## ✅ Scripts mis à jour

Tous les scripts Lua ont été mis à jour pour utiliser correctement l'API Mímir Framework.

### 📝 Scripts modifiés

#### Scripts d'entraînement (répertoire `scripts/training/`)
1. ✅ **train_llm.lua** - Entraînement LLM complet
2. ✅ **train_t2i_autoencoder.lua** - Entraînement autoencoder T2I (débruitage conditionné)
3. ✅ **train_t5_encoder.lua** - Entraînement encoder
4. ✅ **train_tokenizer.lua** - Entraînement tokenizer

#### Scripts de démonstration (répertoire `scripts/demos/`)
5. ✅ **demo_unet.lua** - UNet
6. ✅ **demo_vae.lua** - Variational Autoencoder
7. ✅ **demo_mobilenet.lua** - MobileNet
8. ✅ **demo_resnet.lua** - ResNet
9. ✅ **demo_vit.lua** - Vision Transformer
10. ✅ **demo_diffusion.lua** - Diffusion

---

## 🔧 Changements principaux

### 1. **Configuration système**

Tous les scripts incluent maintenant :

```lua
-- Configuration allocateur RAM
Mimir.Allocator.configure({
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

✅ **Méthode actuelle** (registre d'architectures v2.3+) :
```lua
-- Partir d'une config par défaut (recommandé)
local cfg, cfg_err = Mimir.Architectures.default_config("transformer")
if not cfg then error(cfg_err) end

-- Ajuster
cfg.vocab_size = 10000
cfg.d_model = 256
cfg.num_layers = 4
cfg.num_heads = 8

-- Créer
local success, err = model.create("transformer", cfg)
if not success then error(err) end

-- (Optionnel) build() est un rebuild compat via registre; il n'alloue plus/ n'init plus automatiquement.
model.build()

-- Allouer + init
local ok_alloc, params_or_err = model.allocate_params()
if not ok_alloc then error(params_or_err or "allocation failed") end
local ok_init, init_err = model.init_weights("he", 42)
if not ok_init then error(init_err or "init failed") end
```

> Note: selon versions, `Mimir.Architectures.default_config()` peut retourner `(cfg)` ou `(cfg, err)`.

### 3. **Architectures disponibles**

La liste exacte vient du registre C++ :

```lua
local names = assert(Mimir.Architectures.available())
for _, n in ipairs(names) do
    print(n)
end
```

Création : `model.create(name, cfg)` (avec `cfg` optionnelle), puis `model.allocate_params()` et `model.init_weights()`.

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

Tous les scripts modernes privilégient la sérialisation unifiée + tokenizer :

```lua
os.execute("mkdir -p checkpoints")
Mimir.Serialization.save("checkpoints/model_name.safetensors", "safetensors")
Mimir.Tokenizer.save("checkpoints/model_name/Mimir.Tokenizer.json")
```

> Compat: `model.save/load` existe encore, mais `Mimir.Serialization.save/load` est recommandé (v2.3+).

---

## 🌐 Serveur REST / WebSocket (scripts)

Un serveur HTTP minimal est disponible pour piloter Mímir via JSON:

- Script: `scripts/modules/api_ws_server.lua`
- Auth: `Authorization: Bearer <token>` / `X-API-Key: <key>`
- TLS optionnel via LuaSec (`ssl`) avec `--tls --tls-cert --tls-key`

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
local cfg = assert(Mimir.Architectures.default_config("transformer"))
cfg.vocab_size = 50000
cfg.max_seq_len = 1024
cfg.d_model = 768
cfg.num_layers = 12
cfg.num_heads = 12

assert(model.create("transformer", cfg))
assert(model.allocate_params())
model.init_weights("xavier", 42)
```

### UNet (Segmentation)
```lua
local cfg = assert(Mimir.Architectures.default_config("unet"))
cfg.in_channels = 3
cfg.out_channels = 1

assert(model.create("unet", cfg))
assert(model.allocate_params())
model.init_weights("he", 42)
```

### MobileNetV2 (Classification)
```lua
local cfg = assert(Mimir.Architectures.default_config("mobilenet"))
cfg.num_classes = 1000

assert(model.create("mobilenet", cfg))
assert(model.allocate_params())
model.init_weights("he", 42)
```

---

## ✅ Checklist de migration

Si vous avez des scripts personnalisés, voici les étapes de migration :

1. ✅ Ajouter `Mimir.Allocator.configure()` au début
2. ✅ Ajouter `model.hardware_caps()` et `model.set_hardware(true)`
3. ✅ Remplacer les anciens builders `Mimir.Architectures.<name>(cfg)` par :
    - `local cfg = Mimir.Architectures.default_config(name)`
    - `model.create(name, cfg)`
    - `model.allocate_params()` puis `model.init_weights()`
   - `Mimir.Architectures.[type](config)`
   - `model.allocate_params()`
   - `model.init_weights(method, seed)`
4. ✅ Implémenter learning rate schedule dans la boucle d'entraînement
5. ✅ Sauvegarder le tokenizer avec `Mimir.Tokenizer.save()`
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
