# 🎯 API Mimir v2.3.0 - Structure Complète Hiérarchique

**Date:** 12 janvier 2026  
**Version:** 2.3.0  
**Type:** Architecture hiérarchique complète

---

## 📦 Vue d'Ensemble

Toute l'API Mimir est désormais organisée sous un **namespace unique** `Mimir.*` avec des sous-namespaces pour chaque domaine fonctionnel :

```lua
Mimir
├── Model.*              -- Gestion et manipulation de modèles
├── Architectures.*      -- Helpers de registre (available/default_config)
├── Layers.*             -- Opérations de layers bas niveau
├── Tokenizer.*          -- Tokenisation et vocabulaire
├── Dataset.*            -- Chargement et préparation de données
├── Serialization.*      -- Sauvegarde/chargement (SafeTensors, RawFolder, DebugJson)
├── Allocator.*          -- Allocation dynamique de tenseurs
├── Memory.*             -- Gestion mémoire RAM
├── Guard.*              -- Enforcement mémoire strict
├── MemoryGuard.*        -- API moderne MemoryGuard
├── Htop.*               -- Monitoring système en temps réel
├── Viz.*                -- Visualisation SFML
└── Checkpoint.*         -- API checkpoint legacy (deprecated)
```

---

## 🏗️ Mimir.Model

Gestion et manipulation de modèles.

### Fonctions Principales

```lua
-- Création et construction
Mimir.Model.create(name, config?)
Mimir.Model.build()
Mimir.Model.allocate_params()
Mimir.Model.init_weights(method, seed?)

-- Forward/Backward
Mimir.Model.forward(input, training?)
Mimir.Model.backward(loss_gradient)
Mimir.Model.optimizer_step(learning_rate)
Mimir.Model.zero_grads()
Mimir.Model.get_gradients()

-- Manipulation layers
Mimir.Model.push_layer(name, type, num_params)
Mimir.Model.set_layer_io(layer_name, input_layers)

-- Informations
Mimir.Model.total_params()
Mimir.Model.hardware_caps()
Mimir.Model.set_hardware(enabled)

-- Training/Inference
Mimir.Model.train(epochs, learning_rate)
Mimir.Model.infer(input)

-- Legacy save/load (deprecated, utilisez Mimir.Serialization.*)
Mimir.Model.save(dir)  -- @deprecated
Mimir.Model.load(dir)  -- @deprecated
```

### Exemple

```lua
-- Création d'un modèle simple
Mimir.Model.create("my_classifier", {
    input_size = 784,
    num_classes = 10
})

-- Ajout de layers
Mimir.Model.push_layer("fc1", "Linear", 784 * 256)
Mimir.Model.push_layer("relu1", "ReLU", 0)
Mimir.Model.push_layer("fc2", "Linear", 256 * 10)

-- Allocation et initialisation
Mimir.Model.allocate_params()
Mimir.Model.init_weights("xavier", 42)

-- Forward pass
local output = Mimir.Model.forward(input_tensor, true)

-- Backward pass
Mimir.Model.backward(loss_gradient)
Mimir.Model.optimizer_step(0.001)
```

---

## 🏛️ Mimir.Architectures

Helpers de registre d'architectures (C++).

### Fonctions

```lua
Mimir.Architectures.available()
Mimir.Architectures.default_config(name)
```

### Exemple

```lua
-- Charger config par défaut puis override quelques champs
local cfg = Mimir.Architectures.default_config("transformer")
cfg.vocab_size = 50000
cfg.d_model = 512
cfg.num_layers = 6
cfg.num_heads = 8

-- Créer le modèle via le registre
local ok, err = Mimir.Model.create("transformer", cfg)
assert(ok, err)

-- Allocation (si nécessaire)
local ok2, params = Mimir.Model.allocate_params()
print(string.format("Modèle créé: %d paramètres", params))
```

---

> Note: Les modules `Mimir.Flux` / `Mimir.FluxModel` ne sont pas présents dans cette version du code source.

---

## 🔧 Mimir.Layers

Opérations de layers bas niveau.

```lua
Mimir.Layers.conv2d(input, weight, bias, config)
Mimir.Layers.linear(input, weight, bias)
Mimir.Layers.maxpool2d(input, kernel_size, stride)
Mimir.Layers.avgpool2d(input, kernel_size, stride)
Mimir.Layers.activation(input, type)  -- relu, gelu, silu, tanh
Mimir.Layers.batchnorm(input, weight, bias, running_mean, running_var)
Mimir.Layers.layernorm(input, weight, bias)
Mimir.Layers.attention(q, k, v, mask?)
```

---

## 📝 Mimir.Tokenizer

Tokenisation et gestion du vocabulaire.

### Fonctions Principales

```lua
-- Création et chargement
Mimir.Tokenizer.create(vocab_size)
Mimir.Tokenizer.save(path)
Mimir.Tokenizer.load(path)

-- Tokenisation
Mimir.Tokenizer.tokenize(text)
Mimir.Tokenizer.detokenize(tokens)
Mimir.Tokenizer.tokenize_ensure(text)  -- Ajoute tokens manquants

-- Vocabulaire
Mimir.Tokenizer.add_token(token)
Mimir.Tokenizer.ensure_vocab_from_text(text)
Mimir.Tokenizer.vocab_size()
Mimir.Tokenizer.get_token_by_id(id)

-- BPE
Mimir.Tokenizer.learn_bpe(corpus, num_merges)
Mimir.Tokenizer.tokenize_bpe(text)

-- Padding et batching
Mimir.Tokenizer.set_max_length(length)
Mimir.Tokenizer.pad_sequence(tokens)
Mimir.Tokenizer.batch_tokenize(texts)

-- Tokens spéciaux
Mimir.Tokenizer.pad_id()
Mimir.Tokenizer.unk_id()
Mimir.Tokenizer.seq_id()
Mimir.Tokenizer.mod_id()
Mimir.Tokenizer.mag_id()

-- Analyse
Mimir.Tokenizer.print_stats()
Mimir.Tokenizer.get_frequencies()
Mimir.Tokenizer.analyze_text(text)
Mimir.Tokenizer.extract_keywords(text, top_k)
```

### Exemple

```lua
-- Créer un tokenizer
Mimir.Tokenizer.create(50000)

-- Apprendre BPE sur un corpus
Mimir.Tokenizer.learn_bpe("path/to/corpus.txt", 10000)

-- Tokeniser du texte
local tokens = Mimir.Tokenizer.tokenize("Hello world!")
print("Tokens:", table.concat(tokens, ", "))

-- Sauvegarder
Mimir.Tokenizer.save("Mimir.Tokenizer.json")
```

---

## 📊 Mimir.Dataset

Chargement et préparation de données.

```lua
Mimir.Dataset.load(path)
Mimir.Dataset.get(index)
Mimir.Dataset.prepare_sequences(max_length)
```

### Exemple

```lua
-- Charger un dataset
Mimir.Dataset.load("data/train/")

-- Préparer des séquences
Mimir.Dataset.prepare_sequences(512)

-- Récupérer un batch
local batch = Mimir.Dataset.get(0)
```

---

## 💾 Mimir.Serialization

Sauvegarde et chargement de modèles (3 formats supportés).

### Fonctions

```lua
Mimir.Serialization.save(path, format, options?)
Mimir.Serialization.load(path, format?, options?)
Mimir.Serialization.detect_format(path)
Mimir.Serialization.save_enhanced_debug(path, options?)
```

### Formats

- `"safetensors"` : Production (HuggingFace compatible)
- `"raw_folder"` : Debug avec checksums SHA256
- `"debug_json"` : Inspection avec Enhanced Debug JSON v1.1.0

### Exemple

```lua
-- SafeTensors (production)
Mimir.Serialization.save("model.safetensors", "safetensors", {
    save_optimizer = true,
    include_git_info = true
})

-- Debug JSON Enhanced
Mimir.Serialization.save("debug.json", "debug_json", {
    include_gradients = true,
    include_optimizer_state = true,
    include_checksums = true,
    max_values_per_tensor = 20
})

-- Chargement avec auto-détection
Mimir.Serialization.load("model.safetensors")
```

---

## 🎛️ Mimir.Allocator

Allocation dynamique de tenseurs avec gestion intelligente de la mémoire.

```lua
Mimir.Allocator.configure(config)
Mimir.Allocator.print_stats()
Mimir.Allocator.get_stats()
```

### Configuration

```lua
Mimir.Allocator.configure({
    max_ram_gb = 8.0,
    enable_compression = true,
    enable_hugepages = false
})
```

---

## 💾 Mimir.Memory

Gestion de la mémoire RAM globale.

```lua
Mimir.Memory.config(max_gb)
Mimir.Memory.get_stats()
Mimir.Memory.print_stats()
Mimir.Memory.clear()
Mimir.Memory.get_usage()
Mimir.Memory.set_limit(gb)
```

---

## 🛡️ Mimir.Guard & Mimir.MemoryGuard

Enforcement strict de la mémoire.

### Mimir.Guard (API simple)

```lua
Mimir.Guard.set_limit(gb)
Mimir.Guard.get_stats()
Mimir.Guard.print_stats()
Mimir.Guard.reset()
```

### Mimir.MemoryGuard (API moderne)

```lua
Mimir.MemoryGuard.setLimit(gb)
Mimir.MemoryGuard.getCurrentUsage()
Mimir.MemoryGuard.getPeakUsage()
Mimir.MemoryGuard.getLimit()
Mimir.MemoryGuard.getStats()
Mimir.MemoryGuard.printStats()
Mimir.MemoryGuard.reset()
```

---

## 📊 Mimir.Htop

Monitoring système en temps réel.

```lua
Mimir.Htop.create()
Mimir.Htop.update()
Mimir.Htop.render()
Mimir.Htop.clear()
Mimir.Htop.enable(enabled)
```

---

## 📈 Mimir.Viz

Visualisation avec SFML.

```lua
-- Création et initialisation
Mimir.Viz.create(width, height, title)
Mimir.Viz.initialize()
Mimir.Viz.is_open()
Mimir.Viz.process_events()

-- Mise à jour
Mimir.Viz.update()
Mimir.Viz.update_metrics(loss, accuracy, lr)
Mimir.Viz.add_image(path, x, y)
Mimir.Viz.add_loss_point(epoch, loss)

-- Contrôle
Mimir.Viz.clear()
Mimir.Viz.set_enabled(enabled)
Mimir.Viz.save_loss_history(path)
```

---

## 📦 Mimir.Checkpoint

**⚠️ DEPRECATED** - Utilisez `Mimir.Serialization.*` à la place.

```lua
-- API actuelle
Mimir.Serialization.save(path, format?, options?)
Mimir.Serialization.load(path, format?, options?)
Mimir.Serialization.detect_format(path)
```

---

## 🎯 Exemple Complet

```lua
-- Configuration mémoire
Mimir.Allocator.configure({
    max_ram_gb = 4.0,
    enable_compression = false
})

-- Créer un modèle
local cfg, err = Mimir.Architectures.default_config("transformer")
assert(cfg, err)
cfg.vocab_size = 50000
cfg.d_model = 768
cfg.num_layers = 12
cfg.num_heads = 12
cfg.seq_len = 1024

assert(Mimir.Model.create("transformer", cfg))

-- Allocation et init
local ok, params = Mimir.Model.allocate_params()
print(string.format("Modèle: %d paramètres", params))
Mimir.Model.init_weights("xavier", 42)

-- Tokenizer
Mimir.Tokenizer.create(50000)
Mimir.Tokenizer.learn_bpe("corpus.txt", 10000)
Mimir.Tokenizer.save("Mimir.Tokenizer.json")

-- Dataset
Mimir.Dataset.load("data/train/")
Mimir.Dataset.prepare_sequences(1024)

-- Training loop
for epoch = 1, 10 do
    -- Forward
    local output = Mimir.Model.forward(input, true)
    
    -- Backward
    Mimir.Model.backward(loss_grad)
    
    -- Update
    Mimir.Model.optimizer_step(0.0001)
    Mimir.Model.zero_grads()
    
    -- Monitoring
    Mimir.Viz.add_loss_point(epoch, loss)
end

-- Sauvegarde
Mimir.Serialization.save("gpt2_small.safetensors", "safetensors", {
    save_optimizer = true,
    save_tokenizer = true
})

-- Debug snapshot
Mimir.Serialization.save("debug.json", "debug_json", {
    include_gradients = true,
    include_optimizer_state = true,
    include_checksums = true
})
```

---

## 🔄 Migration depuis l'API Legacy

### Avant (API globale)

```lua
Mimir.Model.create("my_model")
-- (ancienne doc) Mimir.Architectures.transformer({...})
Mimir.Allocator.configure({...})
Mimir.Tokenizer.create(50000)
Mimir.Serialization.save(...)
```

### Après (API hiérarchique)

```lua
local cfg = Mimir.Architectures.default_config("transformer")
-- override cfg ici...
assert(Mimir.Model.create("transformer", cfg))
Mimir.Allocator.configure({...})
Mimir.Tokenizer.create(50000)
Mimir.Serialization.save(...)
```

**Remplacement simple** : Préfixer tous les appels API avec `Mimir.`

---

## ✅ Avantages de la Structure Hiérarchique

### 1. Organisation Claire
Toutes les fonctionnalités sont logiquement groupées sous `Mimir.*`

### 2. Namespace Propre
Évite la pollution du namespace global Lua

### 3. Auto-complétion IDE
Les éditeurs peuvent suggérer toutes les API disponibles

### 4. Extensibilité
Facile d'ajouter de nouveaux modules sans conflits

### 5. Documentation
Structure claire facilite la documentation et l'apprentissage

---

## 📚 Références

- [API Structure](API_STRUCTURE.md) - Structure détaillée Serialization
- [Enhanced Debug JSON v1.1.0](ENHANCED_DEBUG_JSON.md) - Format diagnostic
- [Migration Guide](UNIFIED_API_MIGRATION.md) - Guide de migration

---

## 📝 Notes de Version

**v2.3.0** - 28 décembre 2025
- ✅ Architecture hiérarchique complète `Mimir.*`
- ✅ 15 namespaces fonctionnels
- ✅ 117+ fonctions API
- ✅ Enhanced Debug JSON v1.1.0 intégré
- ✅ Rétro-compatibilité assurée
