# API Lua - Mímir

Guide de référence complet de l'API Lua pour contrôler Mímir sans recompilation.

---

## 📋 Table des Matières

- [Model API](#model-api)
- [Tokenizer API](#tokenizer-api)
- [Dataset API](#dataset-api)
- [Utilitaires](#utilitaires)
- [Exemples Complets](#exemples-complets)

---

## Model API

### `model.create(type, config)`

Crée un nouveau modèle du type spécifié.

**Paramètres :**
- `type` (string) : Type de modèle
  - `"encoder"` : EncoderModel (BERT-like)
  - `"decoder"` : DecoderModel (GPT-like)
  - `"encoder_decoder"` : EncoderDecoderModel (T5-like)
  - `"autoencoder"` : AutoencoderModel (VAE)
  - `"unet"` : UNetModel
  - `"vit"` : VisionTransformerModel
  - `"multimodal"` : MultiModalModel
- `config` (table) : Configuration du modèle (spécifique au type)

**Retour :**
- `success` (boolean) : `true` si succès, `false` sinon
- `error` (string, optionnel) : Message d'erreur si échec

**Exemple :**

```lua
local config = {
    num_layers = 12,
    d_model = 768,
    num_heads = 12,
    vocab_size = 30000,
    max_seq_len = 512,
    dropout = 0.1
}

local ok, err = model.create("encoder", config)
if not ok then
    log("Erreur: " .. err)
    error("Arrêt")
end
```

---

### `model.build()`

Construit l'architecture du modèle et alloue les paramètres.

**Retour :**
- `success` (boolean) : `true` si succès
- `num_params` (number) : Nombre total de paramètres

**Exemple :**

```lua
local ok, params = model.build()
if ok then
    log(string.format("Modèle construit: %d paramètres", params))
else
    log("Échec de la construction")
end
```

---

### `model.train(epochs, learning_rate)`

⚠️ **TODO** : En cours d'implémentation

Entraîne le modèle sur le dataset chargé.

**Paramètres :**
- `epochs` (number) : Nombre d'epochs
- `learning_rate` (number) : Taux d'apprentissage initial

**Retour :**
- `success` (boolean)
- `final_loss` (number) : Loss finale
- `metrics` (table) : Métriques d'entraînement

**Exemple :**

```lua
local ok, loss, metrics = model.train(50, 0.0001)
log(string.format("Loss finale: %.6f", loss))
log(string.format("Accuracy: %.2f%%", metrics.accuracy * 100))
```

---

### `model.infer(input)`

⚠️ **TODO** : En cours d'implémentation

Effectue une inférence sur l'entrée donnée.

**Paramètres :**
- `input` (string|table) : Entrée (texte ou tenseur)

**Retour :**
- `output` (string|table) : Sortie du modèle

**Exemple :**

```lua
-- Pour un modèle de texte
local output = model.infer("Once upon a time")
log("Génération: " .. output)

-- Pour un modèle d'image
local image_tensor = {0.5, 0.3, ...}  -- Pixels normalisés
local prediction = model.infer(image_tensor)
log("Classe prédite: " .. prediction.class)
```

---

### `model.save(path)`

⚠️ **TODO** : En cours d'implémentation

Sauvegarde le modèle au format SafeTensors.

**Paramètres :**
- `path` (string) : Chemin de sauvegarde

**Retour :**
- `success` (boolean)

**Exemple :**

```lua
local saved = model.save("checkpoints/my_model")
if saved then
    log("✓ Modèle sauvegardé")
else
    log("✗ Échec de la sauvegarde")
end
```

---

### `model.load(path)`

⚠️ **TODO** : En cours d'implémentation

Charge un modèle depuis un fichier SafeTensors.

**Paramètres :**
- `path` (string) : Chemin du checkpoint

**Retour :**
- `success` (boolean)
- `config` (table) : Configuration du modèle chargé

**Exemple :**

```lua
local ok, config = model.load("checkpoints/my_model")
if ok then
    log("Modèle chargé")
    log("Layers: " .. config.num_layers)
end
```

---

## Tokenizer API

### `tokenizer.create(vocab_size)`

Crée un nouveau tokenizer BPE.

**Paramètres :**
- `vocab_size` (number) : Taille du vocabulaire (max 100,000)

**Retour :**
- `success` (boolean)

**Exemple :**

```lua
tokenizer.create(32000)
log("Tokenizer créé avec vocab_size=32000")
```

---

### `tokenizer.tokenize(text)`

Tokenise un texte en IDs de tokens.

**Paramètres :**
- `text` (string) : Texte à tokeniser

**Retour :**
- `tokens` (table) : Liste d'IDs de tokens

**Exemple :**

```lua
local tokens = tokenizer.tokenize("Hello world!")
for i, token_id in ipairs(tokens) do
    log(string.format("Token %d: %d", i, token_id))
end
```

---

### `tokenizer.detokenize(tokens)`

Convertit des IDs de tokens en texte.

**Paramètres :**
- `tokens` (table) : Liste d'IDs de tokens

**Retour :**
- `text` (string) : Texte reconstruit

**Exemple :**

```lua
local tokens = {150, 1045, 2088}
local text = tokenizer.detokenize(tokens)
log("Texte: " .. text)
```

---

## Dataset API

### `dataset.load(path)`

⚠️ **TODO** : En cours d'implémentation

Charge un dataset depuis un répertoire.

**Paramètres :**
- `path` (string) : Chemin du dataset

**Retour :**
- `success` (boolean)
- `num_samples` (number) : Nombre d'échantillons chargés

**Exemple :**

```lua
local ok, count = dataset.load("datasets/text")
if ok then
    log(string.format("%d échantillons chargés", count))
end
```

---

### `dataset.prepare_sequences(max_length)`

⚠️ **TODO** : En cours d'implémentation

Prépare les séquences pour l'entraînement.

**Paramètres :**
- `max_length` (number) : Longueur maximale des séquences

**Retour :**
- `success` (boolean)
- `num_sequences` (number) : Nombre de séquences préparées

**Exemple :**

```lua
local ok, count = dataset.prepare_sequences(512)
log(string.format("%d séquences préparées", count))
```

---

## Utilitaires

### `log(message)`

Affiche un message dans la console avec préfixe `[LUA]`.

**Paramètres :**
- `message` (string) : Message à afficher

**Exemple :**

```lua
log("Début de l'entraînement")
log(string.format("Epoch %d/%d", 5, 50))
```

---

### `read_json(filepath)`

Lit un fichier JSON et retourne une table Lua.

**Paramètres :**
- `filepath` (string) : Chemin du fichier JSON

**Retour :**
- `data` (table) : Données JSON parsées

**Exemple :**

```lua
local config = read_json("config.json")
log("Learning rate: " .. config.training.learning_rate)
log("Batch size: " .. config.training.batch_size)

-- Accéder aux paramètres
local lr = config.training.learning_rate
local epochs = config.training.num_epochs
```

---

### `write_json(filepath, data)`

Écrit une table Lua dans un fichier JSON.

**Paramètres :**
- `filepath` (string) : Chemin de sortie
- `data` (table) : Données à sauvegarder

**Retour :**
- `success` (boolean)

**Exemple :**

```lua
local results = {
    model = "encoder",
    epochs = 50,
    final_loss = 0.234,
    metrics = {
        accuracy = 0.92,
        f1_score = 0.89
    }
}

write_json("results.json", results)
log("Résultats sauvegardés")
```

---

## Exemples Complets

### Workflow Complet d'Entraînement

```lua
-- ========================================
-- Script d'entraînement complet
-- ========================================

log("=== Début du workflow ===")

-- 1. Configuration
local config = read_json("configs/encoder_base.json")

-- 2. Créer tokenizer
tokenizer.create(config.tokenizer.max_vocab)
log("✓ Tokenizer créé")

-- 3. Créer modèle
model.create("encoder", config.model)
local ok, params = model.build()
log(string.format("✓ Modèle construit: %d paramètres", params))

-- 4. Charger dataset
dataset.load(config.dataset.dir)
dataset.prepare_sequences(config.model.max_seq_len)
log("✓ Dataset préparé")

-- 5. Entraînement
local num_epochs = config.training.num_epochs
local lr = config.training.learning_rate

for epoch = 1, num_epochs do
    log(string.format("=== Epoch %d/%d ===", epoch, num_epochs))
    
    local ok, loss = model.train(1, lr)
    log(string.format("Loss: %.6f", loss))
    
    -- Checkpoint tous les 10 epochs
    if epoch % 10 == 0 then
        local ckpt_path = string.format("checkpoints/epoch_%d", epoch)
        model.save(ckpt_path)
        log("✓ Checkpoint sauvegardé: " .. ckpt_path)
    end
    
    -- Learning rate decay
    if config.training.lr_decay.enabled then
        lr = lr * config.training.lr_decay.decay_rate
    end
end

-- 6. Sauvegarder modèle final
model.save("checkpoints/final_model")

-- 7. Export résultats
local results = {
    model_type = "encoder",
    total_epochs = num_epochs,
    final_lr = lr,
    total_params = params
}
write_json("training_results.json", results)

log("=== Workflow terminé ===")
```

---

### Génération de Texte (GPT-like)

```lua
-- Configuration GPT
local gpt_config = {
    num_layers = 12,
    num_heads = 12,
    d_model = 768,
    d_ff = 3072,
    max_seq_len = 1024,
    vocab_size = 50000,
    use_causal_mask = true,
    dropout = 0.1
}

-- Créer modèle
tokenizer.create(gpt_config.vocab_size)
model.create("decoder", gpt_config)
model.build()

-- Entraîner
dataset.load("datasets/text_generation")
model.train(100, 0.0003)

-- Générer
local prompts = {
    "Once upon a time",
    "In the future",
    "The secret to happiness is"
}

for _, prompt in ipairs(prompts) do
    log("Prompt: " .. prompt)
    local generated = model.infer(prompt)
    log("Génération: " .. generated)
    log("---")
end
```

---

### Classification d'Images (ViT)

```lua
-- Configuration ViT
local vit_config = {
    image_size = 224,
    patch_size = 16,
    in_channels = 3,
    num_classes = 1000,
    d_model = 768,
    num_layers = 12,
    num_heads = 12,
    d_ff = 3072,
    dropout = 0.1,
    use_class_token = true
}

-- Créer modèle
model.create("vit", vit_config)
model.build()

-- Charger ImageNet
dataset.load("datasets/imagenet")

-- Entraînement avec monitoring
for epoch = 1, 300 do
    local ok, loss, metrics = model.train(1, 0.0003)
    
    log(string.format("Epoch %d | Loss: %.4f | Acc: %.2f%%",
        epoch, loss, metrics.accuracy * 100))
    
    -- Early stopping
    if metrics.accuracy > 0.95 then
        log("Accuracy target atteint!")
        break
    end
end

model.save("checkpoints/vit_imagenet")
```

---

### Segmentation (U-Net)

```lua
-- Configuration U-Net
local unet_config = {
    in_channels = 3,
    out_channels = 21,  -- 21 classes
    base_channels = 64,
    num_levels = 4,
    use_batch_norm = true,
    use_attention = true,
    attention_levels = {2, 3}
}

-- Créer modèle
model.create("unet", unet_config)
local ok, params = model.build()
log(string.format("U-Net: %d paramètres", params))

-- Entraîner
dataset.load("datasets/segmentation")
model.train(200, 0.0002)

-- Test sur nouvelles images
local test_images = {"img1.jpg", "img2.jpg", "img3.jpg"}
for _, img_path in ipairs(test_images) do
    local segmentation = model.infer(img_path)
    log("Segmentation: " .. img_path)
    -- Sauvegarder résultat
    local out_path = img_path:gsub(".jpg", "_seg.png")
    -- save_segmentation(out_path, segmentation)
end
```

---

### Multi-Modal (Texte + Image)

```lua
-- Configuration Multi-Modal
local mm_config = {
    text_vocab_size = 30000,
    text_embed_dim = 512,
    text_num_layers = 6,
    
    image_size = 224,
    vision_embed_dim = 512,
    vision_num_layers = 6,
    
    fusion_dim = 512,
    num_fusion_layers = 3,
    output_dim = 512,
    
    use_audio = false
}

-- Créer modèle
tokenizer.create(mm_config.text_vocab_size)
model.create("multimodal", mm_config)
model.build()

-- Entraîner sur dataset multi-modal
dataset.load("datasets/multimodal")
model.train(100, 0.0001)

-- Inférence
local text = "A cat sitting on a couch"
local image = "cat_on_couch.jpg"
local embedding = model.infer({text = text, image = image})

log("Embedding multi-modal calculé")
```

---

## 🔧 Configuration Avancée

### Learning Rate Scheduler

```lua
-- Cosine annealing
function cosine_lr(epoch, total_epochs, initial_lr, min_lr)
    local progress = epoch / total_epochs
    local cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (initial_lr - min_lr) * cosine
end

-- Utilisation
local initial_lr = 0.001
local min_lr = 0.00001
local total_epochs = 100

for epoch = 1, total_epochs do
    local lr = cosine_lr(epoch, total_epochs, initial_lr, min_lr)
    model.train(1, lr)
    log(string.format("Epoch %d | LR: %.6f", epoch, lr))
end
```

---

### Callbacks Personnalisés

```lua
-- Callback après chaque epoch
function on_epoch_end(epoch, metrics)
    log(string.format("Epoch %d terminée", epoch))
    
    -- Sauvegarder si meilleure accuracy
    if metrics.accuracy > best_accuracy then
        best_accuracy = metrics.accuracy
        model.save("checkpoints/best_model")
        log("✓ Nouveau meilleur modèle sauvegardé")
    end
    
    -- Export metrics
    table.insert(history, metrics)
    write_json("history.json", history)
end

-- Utilisation
local best_accuracy = 0
local history = {}

for epoch = 1, 50 do
    local ok, loss, metrics = model.train(1, 0.0001)
    on_epoch_end(epoch, metrics)
end
```

---

## 📚 Ressources

- [Exemples complets](../scripts/)
- [Guide d'architecture](ARCHITECTURE.md)
- [API C++](API_CPP.md)
- [Tutoriels](tutorials/)
