# Mímir Framework - Référence Rapide API Lua

**Version:** 2.3.0  
**Dernière mise à jour:** 12 Janvier 2026  
**Synchronisé avec:** mimir-api.lua (EmmyLua stub)  

Guide de référence rapide pour l'API Lua du framework Mímir.

> ⚠️ **Important:** Toujours appeler `Mimir.Allocator.configure()` avant d'utiliser le framework.
> 💡 **Syntaxe Recommandée:** Utiliser `Mimir.Module.*` pour bénéficier de l'autocompletion IDE (annotations EmmyLua dans `mimir-api.lua`).

## 🆕 Nouveautés v2.3.0

### Multi-Input / Branch Support
- ✅ **TensorStore System** - Routage de tensors nommés pour architectures complexes
- ✅ **model.set_layer_io()** - Configuration des entrées/sorties des layers
- ✅ **Residual connections** - Support skip connections (ResNet, DenseNet, U-Net)
- ✅ **Operations multi-input** - Add, Multiply, Concat, MatMul, Split fonctionnelles

---

## Modules Disponibles (13 modules + 3 globales)

| Module | Fonctions | Description |
|--------|-----------|-------------|
| `Mimir.Model` | 18 🆕 | Gestion du modèle + multi-input support |
| `Mimir.Architectures` | 2 | Registry helpers (`available`, `default_config`) |
| `Mimir.Layers` | 8 | Opérations de couches (placeholders) |
| `Mimir.Tokenizer` | 24 | Tokenization, BPE, analyse vocabulaire |
| `Mimir.Dataset` | 3 | Chargement et préparation données |
| `Mimir.Memory` | 6 | Gestion avancée RAM (AdvancedRAMManager) |
| `Mimir.Guard` | 4 | Enforcement strict limites (API legacy) |
| `Mimir.MemoryGuard` | 7 | Enforcement strict limites (API moderne) |
| `Mimir.Allocator` | 3 | Allocation dynamique tenseurs avec compression |
| `Mimir.Serialization` | 4 | Sauvegarde/chargement (SafeTensors, RawFolder, DebugJson) |
| `Mimir.Htop` | 5 | Monitoring temps réel style htop (terminal) |
| `Mimir.Viz` | 11 | Visualisation graphique SFML (images, métriques, loss) |
| `Mimir.Checkpoint` | 2 | API checkpoint (legacy) |
| **Globales** | 3 | `log()`, `read_json()`, `write_json()` |

> **Note:** Les globales (`model.*`, `Mimir.Memory.*`, etc.) restent disponibles pour compatibilité, mais **utilisez `Mimir.*` pour l'autocompletion**.

---

## Workflow Standard

```lua
-- 1. Setup mémoire
Mimir.Memory.set_limit(8000)
Mimir.Allocator.configure({max_tensors = 1000, enable_compression = true})

-- 2. Tokenizer
Mimir.Tokenizer.create(50000)
Mimir.Tokenizer.ensure_vocab_from_text(corpus_text)

-- 3. Dataset
Mimir.Dataset.load("data/corpus")
Mimir.Dataset.prepare_sequences(512)

-- 4. Modèle
local cfg = Mimir.Architectures.default_config("transformer")
cfg.vocab_size = 50000
cfg.d_model = 512
cfg.num_layers = 6
Mimir.Model.create("transformer", cfg)
-- Note: build() reconstruit le modèle (compat/rebuild) mais n'alloue plus/ n'init plus automatiquement.
Mimir.Model.build()
Mimir.Model.allocate_params()
Mimir.Model.init_weights("xavier")

-- 5. Monitoring
Mimir.Htop.create()
Mimir.Htop.enable(true)

-- 6. Entraînement
Mimir.Model.train(10, 3e-4)

-- 7. Sauvegarde
Mimir.Serialization.save("checkpoints/run1.safetensors", "safetensors")
Mimir.Tokenizer.save("checkpoints/Mimir.Tokenizer.json")
```

---

## Fonctions Essentielles

### Model

```lua
Mimir.Model.create(type, config)          -- Créer modèle
Mimir.Model.build()                       -- Compat: reconstruire via registre (ne fait plus allocate/init auto)
Mimir.Model.allocate_params()             -- Allouer les paramètres
Mimir.Model.train(epochs, lr)             -- Entraîner
Mimir.Model.infer(prompt)                 -- Génération
Mimir.Model.total_params()                -- Nombre paramètres
Mimir.Model.init_weights(method, seed)    -- Initialiser poids
Mimir.Model.hardware_caps()               -- Capacités CPU

-- 🆕 Multi-input support
Mimir.Model.set_layer_io(name, inputs, output)  -- Config entrées/sorties layer

-- Serialization (remplace model.save/load)
Mimir.Serialization.save(path, "safetensors") -- Sauvegarder (SafeTensors)
Mimir.Serialization.load(path)                -- Charger (auto-détection)
```

---

## Serveur REST / WebSocket (Lua)

Le framework fournit un serveur HTTP minimal côté scripts pour piloter Mímir via JSON.

- Script: `scripts/modules/api_ws_server.lua`
- HTTP: routes JSON (`/health`, `/model/create`, `/model/build`, ...)
- WebSocket: `/ws` (messages JSON `{method, path, body}`)

### Sécurité (auth + TLS)

Le serveur supporte:

- Auth: `Authorization: Bearer <token>` et/ou `X-API-Key: <key>`
- TLS optionnel via LuaSec (`ssl`) + `--tls-cert/--tls-key`

**Exemple (auth token, pas de TLS):**

```bash
./bin/mimir --lua scripts/modules/api_ws_server.lua -- \
  --host 127.0.0.1 --port 8088 \
  --auth token --auth-token-file secrets/api_token.txt
```

**Exemple (TLS + API key):**

```bash
./bin/mimir --lua scripts/modules/api_ws_server.lua -- \
  --host 0.0.0.0 --port 8443 \
  --auth api_key --api-key-file secrets/api_key.txt \
  --tls --tls-cert certs/server.crt --tls-key certs/server.key
```

> Note: `Mimir.Model.set_hardware(...)` dans l'API Lua actuelle est un booléen (enable/disable) et ne sélectionne pas un backend nommé.

**Exemple Multi-Input (Residual Connection):**
```lua
-- Créer skip connection (x + conv_output)
Mimir.Model.push_layer("conv1", "Conv2d", 64*64*3*3)
Mimir.Model.set_layer_io("conv1", {"x"}, "skip")  -- Sauver output comme "skip"

model.push_layer("add", "Add", 0)
model.set_layer_io("add", {"x", "skip"}, "x")  -- Add(x, skip) → x
```

### Architectures

```lua
-- v2.3+ : les builders `Mimir.Architectures.<name>(cfg)` ne sont plus exposés.
-- Utiliser le registre :
local names = Mimir.Architectures.available()
local cfg = Mimir.Architectures.default_config("transformer")
local ok, err = Mimir.Model.create("transformer", cfg)
```

> Note: `Mimir.Flux` / `Mimir.FluxModel` ne sont pas présents dans cette version du code source.

### Tokenizer (24 fonctions)

```lua
Mimir.Tokenizer.create(max_vocab)                  -- Créer
Mimir.Tokenizer.tokenize(text)                     -- Tokeniser
Mimir.Tokenizer.detokenize(tokens)                 -- Dé-tokeniser
Mimir.Tokenizer.vocab_size()                       -- Taille vocab
Mimir.Tokenizer.add_token(token)                   -- Ajouter token
Mimir.Tokenizer.ensure_vocab_from_text(text)       -- Étendre vocab
Mimir.Tokenizer.tokenize_ensure(text)              -- Tokeniser + étendre
Mimir.Tokenizer.learn_bpe(corpus_path, size)       -- Apprendre BPE
Mimir.Tokenizer.tokenize_bpe(text)                 -- Tokeniser BPE
Mimir.Tokenizer.set_max_length(max_len)            -- Longueur max
Mimir.Tokenizer.pad_sequence(tokens, len, pad)     -- Padder séquence
Mimir.Tokenizer.batch_tokenize(texts)              -- Tokeniser batch
Mimir.Tokenizer.pad_id()                           -- ID padding
Mimir.Tokenizer.unk_id()                           -- ID unknown
Mimir.Tokenizer.seq_id()                           -- ID sequence
Mimir.Tokenizer.mod_id()                           -- ID modifier
Mimir.Tokenizer.mag_id()                           -- ID magnitude
Mimir.Tokenizer.get_token_by_id(id)                -- Récupérer token
Mimir.Tokenizer.print_stats()                      -- Afficher stats
Mimir.Tokenizer.get_frequencies()                  -- Fréquences tokens
Mimir.Tokenizer.analyze_text(text)                 -- Analyser texte
Mimir.Tokenizer.extract_keywords(text, count)      -- Extraire mots-clés
Mimir.Tokenizer.pad_sequence(tokens, len)          -- Padding
Mimir.Tokenizer.batch_tokenize(texts)              -- Batch tokenization
Mimir.Tokenizer.extract_keywords(text, top_k)      -- Extraction keywords
Mimir.Tokenizer.save(path)                         -- Sauvegarder
Mimir.Tokenizer.load(path)                         -- Charger
```

**Tokens spéciaux:**
```lua
Mimir.Tokenizer.pad_id()   -- Padding
Mimir.Tokenizer.unk_id()   -- Unknown
Mimir.Tokenizer.seq_id()   -- Sequence
Mimir.Tokenizer.mod_id()   -- Modulation
Mimir.Tokenizer.mag_id()   -- Magnitude
```

### Dataset

```lua
Mimir.Dataset.load(dir)                  -- Charger dataset
Mimir.Dataset.prepare_sequences(max_len) -- Préparer séquences
```

### Memory

```lua
Mimir.Memory.config(cfg)         -- Configurer
Mimir.Memory.get_stats()         -- Stats (current_mb, peak_mb, usage_percent)
Mimir.Memory.print_stats()       -- Afficher stats
Mimir.Memory.clear()             -- Libérer caches
Mimir.Memory.get_usage()         -- Usage actuel (MB)
Mimir.Memory.set_limit(limit_mb) -- Définir limite
```

### Guard (enforcement strict)

```lua
-- API moderne (camelCase)
Mimir.MemoryGuard.setLimit(10)         -- 10 GB (auto)
local stats = Mimir.MemoryGuard.getStats()
Mimir.MemoryGuard.printStats()
Mimir.MemoryGuard.reset()
```

### Allocator

```lua
Mimir.Allocator.configure({
  max_tensors = 1000,
  offload_threshold_mb = 6000,
  enable_compression = true,
  max_ram_gb = 8
})
Mimir.Allocator.print_stats()    -- Afficher stats
Mimir.Allocator.get_stats()      -- Stats (tensor_count, loaded_count)
```

### Htop (monitoring)

```lua
Mimir.Htop.create(config)        -- Créer monitor
Mimir.Htop.enable(enabled)       -- Activer/désactiver
Mimir.Htop.update({              -- Mettre à jour métriques
  epoch = 1,
  total_epochs = 10,
  batch = 10,
  total_batches = 100,
  loss = 2.5,
  avg_loss = 2.6,
  lr = 3e-4,
  memory_mb = 5000
})
Mimir.Htop.render()              -- Forcer render
Mimir.Htop.clear()               -- Clear affichage
```

### Viz (visualisation)

```lua
Mimir.Viz.create(title, width, height)   -- Créer fenêtre
Mimir.Viz.initialize()                   -- Initialiser
Mimir.Viz.is_open()                      -- Fenêtre ouverte?
Mimir.Viz.process_events()               -- Traiter événements
Mimir.Viz.update()                       -- Update render
Mimir.Viz.add_image(pixels, w, h, ch)    -- Afficher image
Mimir.Viz.update_metrics(metrics)        -- Update métriques
Mimir.Viz.add_loss_point(loss)           -- Ajouter point loss
Mimir.Viz.clear()                        -- Clear viz
Mimir.Viz.set_enabled(enabled)           -- Activer/désactiver
Mimir.Viz.save_loss_history(path)        -- Sauvegarder historique
```

### Fonctions Globales

```lua
log(message)                 -- Logger message
read_json(filepath)          -- Lire JSON
write_json(filepath, data)   -- Écrire JSON
```

---

## Types de Modèles

| Type | Description |
|------|-------------|
| `"encoder"` | Encoder transformer |
| `"decoder"` | Decoder transformer |
| `"transformer"` | Transformer complet |
| `"unet"` | U-Net |
| `"vae"` | Variational Autoencoder |
| `"vit"` | Vision Transformer |
| `"gan"` | GAN |
| `"diffusion"` | Modèle de diffusion |
| `"resnet"` | ResNet |
| `"mobilenet"` | MobileNet |

---

## Configuration Transformer

```lua
{
  vocab_size = 50000,      -- Taille vocabulaire
  embed_dim = 512,         -- Dimension embeddings
  num_layers = 6,          -- Nombre de couches
  num_heads = 8,           -- Têtes d'attention
  d_ff = 2048,            -- Dimension FFN
  max_seq_len = 512,      -- Longueur max séquence
  dropout = 0.1           -- Dropout rate
}
```

---

## Configuration UNet

```lua
{
  input_channels = 3,
  output_channels = 1,
  base_channels = 64,
  num_levels = 4,
  blocks_per_level = 2,
  use_attention = true,
  use_residual = true,
  dropout = 0.1
}
```

---

## Configuration VAE

```lua
{
  input_dim = 784,
  latent_dim = 20,
  encoder_hidden = 400,
  decoder_hidden = 400
}
```

---

## Configuration ViT

```lua
{
  image_size = 224,
  patch_size = 16,
  embed_dim = 768,
  num_layers = 12,
  num_heads = 12,
  mlp_ratio = 4.0,
  num_classes = 1000
}
```

---

## Configuration Flux

> ⚠️ Historique : `Mimir.Flux` / `Mimir.FluxModel` ne sont pas exportés dans le binaire v2.3. Cette configuration est conservée uniquement comme référence documentaire.

```lua
{
  image_resolution = 256,          -- Résolution images
  latent_channels = 4,             -- Canaux espace latent
  latent_resolution = 32,          -- Résolution latent
  vae_base_channels = 128,         -- Canaux base VAE
  vae_channel_mult = {1,2,4,4},    -- Multiplicateurs canaux
  num_res_blocks = 2,              -- Blocs résiduels
  vocab_size = 50000,              -- Vocabulaire
  text_max_length = 77,            -- Longueur max texte
  text_embed_dim = 768,            -- Dimension embeddings
  transformer_dim = 768,           -- Dimension transformer
  num_transformer_blocks = 12,     -- Blocs transformer
  num_attention_heads = 12,        -- Têtes attention
  mlp_ratio = 4.0,                 -- Ratio MLP
  timestep_embed_dim = 256,        -- Dimension timestep
  num_diffusion_steps = 1000       -- Steps diffusion
}
```

---

## Méthodes d'Initialisation Poids

- `"xavier"` : Xavier/Glorot initialization
- `"he"` : He initialization
- `"normal"` : Distribution normale
- `"uniform"` : Distribution uniforme
- `"zeros"` : Initialisation à zéro

---

## Backends Hardware

- `"cpu"` : CPU uniquement (défaut)
- `"opencl"` : OpenCL (si compilé)
- `"vulkan"` : Vulkan Compute (si compilé)
- `"auto"` : Détection automatique

---

## Structures de Données

### HtopMetrics

```lua
{
  epoch = 1,             -- Epoch courante
  total_epochs = 10,     -- Total epochs
  batch = 10,            -- Batch courant
  total_batches = 100,   -- Total batches
  loss = 2.5,           -- Loss courante
  avg_loss = 2.6,       -- Loss moyenne
  lr = 3e-4,            -- Learning rate
  batch_time_ms = 150,  -- Temps batch (ms)
  memory_mb = 5000,     -- Mémoire (MB)
  memory_freed = 100,   -- Mémoire libérée
  bps = 0.66,          -- Batches/sec
  params = 125000000,   -- Nombre paramètres
  timestep = 0.5,      -- Timestep (diffusion)
  kl = 0.1,            -- KL divergence
  wass = 0.2,          -- Wasserstein
  ent = 0.3,           -- Entropy
  mom = 0.9,           -- Momentum
  spat = 0.5,          -- Spatial metric
  temp = 1.0,          -- Temperature
  mse = 0.01           -- MSE
}
```

### VizMetrics

```lua
{
  epoch = 1,
  total_epochs = 10,
  loss = 2.5,
  avg_loss = 2.6,
  lr = 3e-4,
  memory_mb = 5000,
  tokens_per_sec = 1500,
  custom = {            -- Métriques custom
    accuracy = 0.85,
    perplexity = 12.3
  }
}
```

### MemoryStats

```lua
{
  current_mb = 5000,     -- Utilisation courante
  peak_mb = 6500,        -- Pic d'utilisation
  usage_percent = 62.5   -- Pourcentage
}
```

### GuardStats

```lua
{
  current_mb = 5000,
  peak_mb = 6500,
  limit_mb = 8000,
  usage_percent = 62.5
}
```

### AllocatorStats

```lua
{
  tensor_count = 1000,   -- Nombre total tenseurs
  loaded_count = 850     -- Tenseurs chargés en mémoire
}
```

### HardwareCaps

```lua
{
  avx2 = true,   -- Support AVX2
  fma = true,    -- Support FMA
  f16c = true,   -- Support F16C
  bmi2 = true    -- Support BMI2
}
```

---

## Exemples Rapides

### Entraînement Simple

```lua
Mimir.Memory.set_limit(8000)
Mimir.Tokenizer.create(50000)
Mimir.Tokenizer.ensure_vocab_from_text(corpus)
Mimir.Dataset.load("data/train")
Mimir.Dataset.prepare_sequences(512)
Mimir.Model.create("transformer", {vocab_size = 50000, embed_dim = 512})
Mimir.Model.build()
Mimir.Model.train(10, 3e-4)
Mimir.Model.save("checkpoints/final")
```

### Inférence

```lua
Mimir.Tokenizer.load("checkpoints/Mimir.Tokenizer.json")
Mimir.Model.load("checkpoints/final")
local output = Mimir.Model.infer("Once upon a time")
print(output)
```

### Monitoring Complet

```lua
Mimir.Htop.create()
Mimir.Htop.enable(true)
Mimir.Viz.create("Training", 1280, 720)
Mimir.Viz.initialize()

-- Dans boucle d'entraînement:
Mimir.Htop.update({epoch = e, batch = b, loss = l, ...})
Mimir.Viz.update_metrics({epoch = e, loss = l})
Mimir.Viz.add_loss_point(l)
if Mimir.Viz.is_open() then
  Mimir.Viz.process_events()
  Mimir.Viz.update()
end
```

---

## Debugging et Logs

```lua
-- Afficher infos mémoire
Mimir.Memory.print_stats()
Mimir.MemoryGuard.printStats()
Mimir.Allocator.print_stats()

-- Afficher infos tokenizer
Mimir.Tokenizer.print_stats()

-- Logger custom
log("Mon message de debug")

-- Hardware capabilities
local caps = model.hardware_caps()
log("AVX2: " .. tostring(caps.avx2))
log("FMA: " .. tostring(caps.fma))
```

---

## Gestion Erreurs

Toutes les fonctions retournant `(ok, err)` peuvent être gérées ainsi:

```lua
local ok, err = Mimir.Model.train(10, 3e-4)
if not ok then
  log("Erreur entraînement: " .. (err or "unknown"))
  os.exit(1)
end
```

Pour les fonctions retournant directement une valeur:

```lua
local output = Mimir.Model.infer(prompt)
if not output then
  log("Erreur inférence")
end
```

---

## Best Practices

1. **Toujours définir une limite mémoire** : `Mimir.Memory.set_limit()` ou `Mimir.MemoryGuard.setLimit()`
2. **Configurer l'allocateur** : Active compression et offload pour grands modèles
3. **Sauvegarder régulièrement** : Checkpoints toutes les N epochs
4. **Monitoring** : Utilise htop ou viz pour suivre progression
5. **Clear périodique** : `Mimir.Memory.clear()` entre epochs si nécessaire
6. **Hardware detection** : Vérifie `model.hardware_caps()` pour optimisations

---

## Documentation Complète

Pour plus de détails, voir:
- [API Complete](00-API-Complete.md) - Documentation exhaustive
- [Pipeline API](../05-Advanced/01-Pipeline-API.md) - API de pipeline haut niveau
- [Quick Start](../01-Getting-Started/01-Quick-Start.md) - Guide de démarrage
- `mimir-api.lua` - Stub IDE avec annotations EmmyLua

---

**© 2025 Mímir Framework - bri45**
