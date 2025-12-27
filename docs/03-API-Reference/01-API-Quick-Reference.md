# Mímir Framework - Référence Rapide API Lua

**Version:** 2.1.0  
**Dernière mise à jour:** Décembre 2025  
**Synchronisé avec:** src/LuaScripting.cpp  

Guide de référence rapide pour l'API Lua du framework Mímir.

> ⚠️ **Important:** Toujours appeler `allocator.configure()` avant d'utiliser le framework.

---

## Modules Disponibles (13 modules, 114 fonctions)

| Module | Fonctions | Description |
|--------|-----------|-------------|
| `model` | 17 | Gestion du modèle (create, build, train, infer, save/load) |
| `architectures` | 9 | Builders pour architectures pré-définies |
| `flux` | 5 | API Flux pour génération d'images guidée par texte |
| `FluxModel` | 12 | Classe FluxModel (orientée objet) |
| `layers` | 8 | Opérations de couches (placeholders) |
| `tokenizer` | 24 | Tokenization, BPE, analyse vocabulaire |
| `dataset` | 3 | Chargement et préparation données |
| `memory` | 6 | Gestion avancée RAM (AdvancedRAMManager) |
| `guard` | 4 | Enforcement strict limites (API ancienne) |
| `MemoryGuard` | 7 | Enforcement strict limites (API moderne) |
| `allocator` | 3 | Allocation dynamique tenseurs avec compression |
| `htop` | 5 | Monitoring temps réel style htop (terminal) |
| `viz` | 11 | Visualisation graphique SFML (images, métriques, loss) |
| **Globales** | 3 | `log()`, `read_json()`, `write_json()` |

---

## Workflow Standard

```lua
-- 1. Setup mémoire
memory.set_limit(8000)
allocator.configure({max_tensors = 1000, enable_compression = true})

-- 2. Tokenizer
tokenizer.create(50000)
tokenizer.ensure_vocab_from_text(corpus_text)

-- 3. Dataset
dataset.load("data/corpus")
dataset.prepare_sequences(512)

-- 4. Modèle
model.create("transformer", {vocab_size = 50000, embed_dim = 512, num_layers = 6})
model.build()
model.init_weights("xavier")

-- 5. Monitoring
htop.create()
htop.enable(true)

-- 6. Entraînement
model.train(10, 3e-4)

-- 7. Sauvegarde
model.save("checkpoints/run1")
tokenizer.save("checkpoints/tokenizer.json")
```

---

## Fonctions Essentielles

### Model

```lua
model.create(type, config)          -- Créer modèle
model.build()                       -- Construire (allouer mémoire)
model.train(epochs, lr)             -- Entraîner
model.infer(prompt)                 -- Génération
model.save(dir)                     -- Sauvegarder
model.load(dir)                     -- Charger
model.total_params()                -- Nombre paramètres
model.init_weights(method, seed)    -- Initialiser poids
model.hardware_caps()               -- Capacités CPU
```

### Architectures

```lua
architectures.transformer(config)   -- Transformer GPT-like
architectures.unet(config)          -- U-Net
architectures.vae(config)           -- VAE
architectures.vit(config)           -- Vision Transformer
architectures.gan(config)           -- GAN
architectures.diffusion(config)     -- Diffusion model
architectures.resnet(config)        -- ResNet
architectures.mobilenet(config)     -- MobileNet
```

### Flux (API Fonctionnelle)

```lua
flux.generate(prompt, steps)           -- Générer image depuis prompt
flux.encode_image(img_path)            -- Encoder image → latent
flux.decode_latent(latent)             -- Décoder latent → image
flux.encode_text(text)                 -- Encoder texte → embeddings
flux.set_tokenizer(tok_path)           -- Définir tokenizer
```

### FluxModel (API Orientée Objet)

```lua
local flux_model = FluxModel.new(config)     -- Créer instance
flux_model.train()                            -- Mode entraînement
flux_model.eval()                             -- Mode évaluation
flux_model.isTraining()                       -- Vérifier mode
flux_model.encodeImage(path)                  -- Encoder image
flux_model.decodeLatent(latent)               -- Décoder latent
flux_model.tokenizePrompt(text)               -- Tokeniser prompt
flux_model.encodeText(tokens)                 -- Encoder tokens
flux_model.predictNoise(latent, embed, t)     -- Prédire bruit
flux_model.generate(prompt, steps)            -- Générer image
flux_model.computeDiffusionLoss(img, txt, t) -- Calculer loss
flux_model.setPromptTokenizer(tok_path)       -- Définir tokenizer
```

### Tokenizer (24 fonctions)

```lua
tokenizer.create(max_vocab)                  -- Créer
tokenizer.tokenize(text)                     -- Tokeniser
tokenizer.detokenize(tokens)                 -- Dé-tokeniser
tokenizer.vocab_size()                       -- Taille vocab
tokenizer.add_token(token)                   -- Ajouter token
tokenizer.ensure_vocab_from_text(text)       -- Étendre vocab
tokenizer.tokenize_ensure(text)              -- Tokeniser + étendre
tokenizer.learn_bpe(corpus_path, size)       -- Apprendre BPE
tokenizer.tokenize_bpe(text)                 -- Tokeniser BPE
tokenizer.set_max_length(max_len)            -- Longueur max
tokenizer.pad_sequence(tokens, len, pad)     -- Padder séquence
tokenizer.batch_tokenize(texts)              -- Tokeniser batch
tokenizer.pad_id()                           -- ID padding
tokenizer.unk_id()                           -- ID unknown
tokenizer.seq_id()                           -- ID sequence
tokenizer.mod_id()                           -- ID modifier
tokenizer.mag_id()                           -- ID magnitude
tokenizer.get_token_by_id(id)                -- Récupérer token
tokenizer.print_stats()                      -- Afficher stats
tokenizer.get_frequencies()                  -- Fréquences tokens
tokenizer.analyze_text(text)                 -- Analyser texte
tokenizer.extract_keywords(text, count)      -- Extraire mots-clés
tokenizer.pad_sequence(tokens, len)          -- Padding
tokenizer.batch_tokenize(texts)              -- Batch tokenization
tokenizer.extract_keywords(text, top_k)      -- Extraction keywords
tokenizer.save(path)                         -- Sauvegarder
tokenizer.load(path)                         -- Charger
```

**Tokens spéciaux:**
```lua
tokenizer.pad_id()   -- Padding
tokenizer.unk_id()   -- Unknown
tokenizer.seq_id()   -- Sequence
tokenizer.mod_id()   -- Modulation
tokenizer.mag_id()   -- Magnitude
```

### Dataset

```lua
dataset.load(dir)                  -- Charger dataset
dataset.prepare_sequences(max_len) -- Préparer séquences
```

### Memory

```lua
memory.config(cfg)         -- Configurer
memory.get_stats()         -- Stats (current_mb, peak_mb, usage_percent)
memory.print_stats()       -- Afficher stats
memory.clear()             -- Libérer caches
memory.get_usage()         -- Usage actuel (MB)
memory.set_limit(limit_mb) -- Définir limite
```

### Guard (enforcement strict)

```lua
guard.set_limit(limit_mb)  -- Limite stricte
guard.get_stats()          -- Stats (current_mb, peak_mb, limit_mb, usage_percent)
guard.print_stats()        -- Afficher stats
guard.reset()              -- Reset compteurs
```

### Allocator

```lua
allocator.configure({
  max_tensors = 1000,
  offload_threshold_mb = 6000,
  enable_compression = true,
  max_ram_gb = 8
})
allocator.print_stats()    -- Afficher stats
allocator.get_stats()      -- Stats (tensor_count, loaded_count)
```

### Htop (monitoring)

```lua
htop.create(config)        -- Créer monitor
htop.enable(enabled)       -- Activer/désactiver
htop.update({              -- Mettre à jour métriques
  epoch = 1,
  total_epochs = 10,
  batch = 10,
  total_batches = 100,
  loss = 2.5,
  avg_loss = 2.6,
  lr = 3e-4,
  memory_mb = 5000
})
htop.render()              -- Forcer render
htop.clear()               -- Clear affichage
```

### Viz (visualisation)

```lua
viz.create(title, width, height)   -- Créer fenêtre
viz.initialize()                   -- Initialiser
viz.is_open()                      -- Fenêtre ouverte?
viz.process_events()               -- Traiter événements
viz.update()                       -- Update render
viz.add_image(pixels, w, h, ch)    -- Afficher image
viz.update_metrics(metrics)        -- Update métriques
viz.add_loss_point(loss)           -- Ajouter point loss
viz.clear()                        -- Clear viz
viz.set_enabled(enabled)           -- Activer/désactiver
viz.save_loss_history(path)        -- Sauvegarder historique
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
memory.set_limit(8000)
tokenizer.create(50000)
tokenizer.ensure_vocab_from_text(corpus)
dataset.load("data/train")
dataset.prepare_sequences(512)
model.create("transformer", {vocab_size = 50000, embed_dim = 512})
model.build()
model.train(10, 3e-4)
model.save("checkpoints/final")
```

### Inférence

```lua
tokenizer.load("checkpoints/tokenizer.json")
model.load("checkpoints/final")
local output = model.infer("Once upon a time")
print(output)
```

### Monitoring Complet

```lua
htop.create()
htop.enable(true)
viz.create("Training", 1280, 720)
viz.initialize()

-- Dans boucle d'entraînement:
htop.update({epoch = e, batch = b, loss = l, ...})
viz.update_metrics({epoch = e, loss = l})
viz.add_loss_point(l)
if viz.is_open() then
  viz.process_events()
  viz.update()
end
```

---

## Debugging et Logs

```lua
-- Afficher infos mémoire
memory.print_stats()
guard.print_stats()
allocator.print_stats()

-- Afficher infos tokenizer
tokenizer.print_stats()

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
local ok, err = model.train(10, 3e-4)
if not ok then
  log("Erreur entraînement: " .. (err or "unknown"))
  os.exit(1)
end
```

Pour les fonctions retournant directement une valeur:

```lua
local output = model.infer(prompt)
if not output then
  log("Erreur inférence")
end
```

---

## Best Practices

1. **Toujours définir une limite mémoire** : `memory.set_limit()` ou `guard.set_limit()`
2. **Configurer l'allocateur** : Active compression et offload pour grands modèles
3. **Sauvegarder régulièrement** : Checkpoints toutes les N epochs
4. **Monitoring** : Utilise htop ou viz pour suivre progression
5. **Clear périodique** : `memory.clear()` entre epochs si nécessaire
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
