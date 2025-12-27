# Mímir Framework - API Lua Complète

**Version:** 2.1.0  
**Date:** Décembre 2025  
**Auteur:** bri45

Ce document décrit l'API Lua complète du framework Mímir, exposant toutes les fonctionnalités du moteur C++ via des bindings Lua.

---

## Table des matières

1. [Architecture Générale](#architecture-générale)
2. [Module `model`](#module-model)
3. [Module `architectures`](#module-architectures)
4. [Module `flux`](#module-flux)
5. [Module `FluxModel`](#module-fluxmodel)
6. [Module `layers`](#module-layers)
7. [Module `tokenizer`](#module-tokenizer)
8. [Module `dataset`](#module-dataset)
9. [Module `memory`](#module-memory)
10. [Module `guard`](#module-guard)
11. [Module `MemoryGuard`](#module-memoryguard)
12. [Module `allocator`](#module-allocator)
13. [Module `htop`](#module-htop)
14. [Module `viz`](#module-viz)
15. [Fonctions Globales](#fonctions-globales)
16. [Exemples Complets](#exemples-complets)

---

## Architecture Générale

Le framework Mímir utilise Lua comme langage de scripting pour piloter l'entraînement et l'inférence de modèles de deep learning. L'architecture est organisée en modules, chacun exposant une API spécifique :

- **`model`** : Gestion du cycle de vie du modèle (création, build, train, infer, save/load)
- **`architectures`** : Builders pour architectures pré-définies (UNet, VAE, ViT, GAN, Flux, etc.)
- **`flux`** : API fonctionnelle pour génération d'images guidée par texte
- **`FluxModel`** : API orientée objet pour le modèle Flux
- **`layers`** : Opérations de couches bas niveau (placeholders)
- **`tokenizer`** : Tokenization (word-level, BPE), gestion vocabulaire
- **`dataset`** : Chargement et préparation des données
- **`memory`** : Gestion avancée de la RAM (AdvancedRAMManager)
- **`guard`** : Enforcement strict des limites mémoire (API ancienne, deprecated)
- **`MemoryGuard`** : Enforcement strict des limites mémoire (API moderne, recommandée)
- **`allocator`** : Allocation dynamique de tenseurs avec offload/compression
- **`htop`** : Monitoring temps réel style htop en terminal
- **`viz`** : Visualisation graphique (SFML) - images, métriques, loss curves

### Flux typique

```lua
-- 1. Configuration mémoire
memory.set_limit(8000) -- 8 GB
allocator.configure({max_tensors = 1000, enable_compression = true})

-- 2. Tokenizer
tokenizer.create(50000)
tokenizer.ensure_vocab_from_text(corpus_text)

-- 3. Dataset
dataset.load("data/my_corpus")
dataset.prepare_sequences(512)

-- 4. Modèle
model.create("transformer", {vocab_size = 50000, embed_dim = 512, num_layers = 6})
model.build()
model.init_weights("xavier")

-- 5. Monitoring
htop.create({enable_viz = false})
htop.enable(true)

-- 6. Entraînement
model.train(10, 3e-4)

-- 7. Sauvegarde
model.save("checkpoints/run1/final")
tokenizer.save("checkpoints/run1/tokenizer.json")
```

---

## Module `model`

### `model.create(model_type, config)`

Crée un modèle avec le type et la configuration spécifiés. Le modèle n'est pas encore construit (allocation mémoire) à ce stade.

**Paramètres:**
- `model_type` (string) : Type de modèle
  - `"encoder"` : Encoder transformer
  - `"decoder"` : Decoder transformer
  - `"transformer"` : Transformer complet
  - `"unet"` : U-Net pour segmentation/diffusion
  - `"vae"` : Variational Autoencoder
  - `"vit"` : Vision Transformer
  - `"gan"` : Generative Adversarial Network
  - `"diffusion"` : Modèle de diffusion (DDPM-like)
  - `"resnet"` : ResNet
  - `"mobilenet"` : MobileNet
- `config` (table) : Configuration spécifique au modèle

**Configuration commune:**
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

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

**Exemple:**
```lua
local ok, err = model.create("transformer", {
  vocab_size = 50000,
  embed_dim = 768,
  num_layers = 12,
  num_heads = 12,
  d_ff = 3072,
  max_seq_len = 1024,
  dropout = 0.1
})
if not ok then
  print("Erreur:", err)
end
```

---

### `model.build()`

Construit le modèle (alloue la mémoire, initialise les paramètres). Doit être appelé après `model.create()`.

**Retour:**
- `ok` (boolean) : Succès
- `params` (integer, optionnel) : Nombre total de paramètres scalaires
- `err` (string, optionnel) : Message d'erreur

**Exemple:**
```lua
local ok, params, err = model.build()
if ok then
  print(string.format("Modèle construit: %d paramètres", params))
else
  print("Erreur:", err)
end
```

---

### `model.train(epochs, learning_rate)`

Entraîne le modèle sur le dataset chargé.

**Prérequis:**
- Dataset chargé via `dataset.load()`
- Séquences préparées via `dataset.prepare_sequences()`
- Modèle construit via `model.build()`

**Paramètres:**
- `epochs` (integer) : Nombre d'epochs
- `learning_rate` (number) : Taux d'apprentissage (ex: 3e-4)

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

**Exemple:**
```lua
local ok, err = model.train(10, 3e-4)
if not ok then
  print("Erreur entraînement:", err)
end
```

---

### `model.infer(input)`

Effectue une inférence sur un prompt texte ou une séquence de tokens.

**Paramètres:**
- `input` (string ou table) : Prompt texte ou séquence de token IDs

**Retour:**
- `output` (string ou nil) : Texte généré, ou nil en cas d'erreur

**Exemple:**
```lua
local output = model.infer("Once upon a time")
if output then
  print("Génération:", output)
else
  print("Erreur inférence")
end
```

---

### `model.save(dir)`

Sauvegarde le modèle (checkpoint) dans un répertoire.

**Paramètres:**
- `dir` (string) : Chemin du répertoire de sauvegarde

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

**Exemple:**
```lua
local ok, err = model.save("checkpoints/run1/epoch_10")
```

---

### `model.load(dir)`

Charge un modèle depuis un checkpoint.

**Paramètres:**
- `dir` (string) : Chemin du répertoire de chargement

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `model.allocate_params()`

Alloue explicitement les paramètres du modèle (si non fait par `build()`).

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `model.init_weights(init_method, seed)`

Initialise les poids du modèle selon une méthode.

**Paramètres:**
- `init_method` (string, optionnel) : Méthode d'initialisation
  - `"xavier"` : Xavier/Glorot
  - `"he"` : He initialization
  - `"normal"` : Distribution normale
  - `"uniform"` : Distribution uniforme
  - `"zeros"` : Zéros
- `seed` (integer, optionnel) : Seed aléatoire

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

**Exemple:**
```lua
model.init_weights("xavier", 42)
```

---

### `model.total_params()`

Retourne le nombre total de paramètres du modèle.

**Retour:**
- `params` (integer) : Nombre de paramètres

---

### `model.forward(input)`

Effectue une passe forward manuelle (bas niveau).

**Paramètres:**
- `input` : Entrée (dépend du modèle)

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `model.backward()`

Effectue une passe backward manuelle (calcul des gradients).

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `model.optimizer_step(learning_rate)`

Effectue un pas d'optimisation manuel.

**Paramètres:**
- `learning_rate` (number) : Taux d'apprentissage

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `model.set_hardware(backend)`

Définit le backend d'accélération matérielle (expérimental).

**Paramètres:**
- `backend` (string) : Backend
  - `"cpu"` : CPU uniquement
  - `"opencl"` : OpenCL (si compilé)
  - `"vulkan"` : Vulkan Compute (si compilé)
  - `"auto"` : Détection automatique

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `model.hardware_caps()`

Retourne les capacités hardware détectées.

**Retour:**
- `caps` (table) : Capacités
  - `avx2` (boolean) : Support AVX2
  - `fma` (boolean) : Support FMA
  - `f16c` (boolean) : Support F16C
  - `bmi2` (boolean) : Support BMI2

**Exemple:**
```lua
local caps = model.hardware_caps()
print("AVX2:", caps.avx2)
print("FMA:", caps.fma)
```

---

## Module `architectures`

Ce module fournit des builders pour construire rapidement des architectures pré-définies.

### `architectures.unet(config)`

Construit un modèle U-Net.

**Configuration:**
```lua
{
  input_channels = 3,      -- Canaux d'entrée
  output_channels = 1,     -- Canaux de sortie
  base_channels = 64,      -- Canaux de base
  num_levels = 4,          -- Nombre de niveaux (encoder/decoder)
  blocks_per_level = 2,    -- Blocs par niveau
  use_attention = true,    -- Utiliser attention
  use_residual = true,     -- Connexions résiduelles
  dropout = 0.1
}
```

---

### `architectures.vae(config)`

Construit un Variational Autoencoder.

**Configuration:**
```lua
{
  input_dim = 784,         -- Dimension d'entrée
  latent_dim = 20,         -- Dimension latente
  encoder_hidden = 400,    -- Dimension cachée encoder
  decoder_hidden = 400     -- Dimension cachée decoder
}
```

---

### `architectures.vit(config)`

Construit un Vision Transformer.

**Configuration:**
```lua
{
  image_size = 224,        -- Résolution image
  patch_size = 16,         -- Taille des patches
  embed_dim = 768,         -- Dimension embeddings
  num_layers = 12,         -- Nombre de layers
  num_heads = 12,          -- Têtes d'attention
  mlp_ratio = 4.0,         -- Ratio MLP
  num_classes = 1000       -- Nombre de classes
}
```

---

### `architectures.gan(config)`

Construit un GAN (Generator + Discriminator).

**Configuration:**
```lua
{
  latent_dim = 100,        -- Dimension latente
  image_channels = 3,      -- Canaux image
  resolution = 64,         -- Résolution image
  gen_channels = 64,       -- Canaux générateur
  disc_channels = 64       -- Canaux discriminateur
}
```

---

### `architectures.diffusion(config)`

Construit un modèle de diffusion (DDPM-like).

**Configuration:**
```lua
{
  image_channels = 3,
  resolution = 64,
  model_channels = 128,
  num_res_blocks = 2,
  use_attention = true,
  dropout = 0.1,
  use_bottleneck = true
}
```

---

### `architectures.transformer(config)`

Construit un transformer complet (encoder-decoder ou decoder-only).

**Configuration:**
```lua
{
  vocab_size = 50000,
  embed_dim = 512,
  num_layers = 6,
  num_heads = 8,
  d_ff = 2048,
  max_seq_len = 512,
  dropout = 0.1
}
```

---

### `architectures.resnet(config)`

Construit un ResNet.

**Configuration:**
```lua
{
  num_classes = 1000,
  input_channels = 3
}
```

---

### `architectures.mobilenet(config)`

Construit un MobileNet.

**Configuration:**
```lua
{
  num_classes = 1000,
  width_mult = 1.0  -- Multiplicateur de largeur
}
```

---

## Module `tokenizer`

### `tokenizer.create(max_vocab)`

Crée un tokenizer avec une taille de vocabulaire maximale.

**Paramètres:**
- `max_vocab` (integer) : Taille max du vocabulaire

**Retour:**
- `ok` (boolean) : Succès

---

### `tokenizer.tokenize(text)`

Tokenise un texte en séquence de token IDs.

**Paramètres:**
- `text` (string) : Texte à tokeniser

**Retour:**
- `tokens` (table) : Liste de token IDs (indices Lua 1..N)

**Exemple:**
```lua
local tokens = tokenizer.tokenize("Hello world")
-- tokens = {1523, 2891}
```

---

### `tokenizer.detokenize(tokens)`

Convertit une séquence de tokens en texte.

**Paramètres:**
- `tokens` (table) : Liste de token IDs

**Retour:**
- `text` (string) : Texte résultant

---

### `tokenizer.vocab_size()`

Retourne la taille actuelle du vocabulaire.

**Retour:**
- `size` (integer) : Taille du vocabulaire

---

### `tokenizer.add_token(token)`

Ajoute un token au vocabulaire.

**Paramètres:**
- `token` (string) : Token à ajouter

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `tokenizer.ensure_vocab_from_text(text)`

Étend le vocabulaire en analysant un texte (ajoute les mots manquants).

**Paramètres:**
- `text` (string) : Texte à analyser

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `tokenizer.tokenize_ensure(text)`

Tokenise un texte et étend le vocabulaire si nécessaire.

**Paramètres:**
- `text` (string) : Texte à tokeniser

**Retour:**
- `tokens` (table) : Liste de token IDs

---

### Tokens Spéciaux

- `tokenizer.pad_id()` : ID du token de padding
- `tokenizer.unk_id()` : ID du token unknown
- `tokenizer.seq_id()` : ID du token de séquence
- `tokenizer.mod_id()` : ID du token de modulation
- `tokenizer.mag_id()` : ID du token de magnitude

---

### `tokenizer.get_token_by_id(id)`

Récupère un token string depuis son ID.

**Paramètres:**
- `id` (integer) : ID du token

**Retour:**
- `token` (string) : Token correspondant

---

### BPE (Byte-Pair Encoding)

#### `tokenizer.learn_bpe(corpus_path, vocab_target)`

Apprend un vocabulaire BPE depuis un corpus.

**Paramètres:**
- `corpus_path` (string) : Chemin du fichier corpus
- `vocab_target` (integer, optionnel) : Taille cible du vocabulaire

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

#### `tokenizer.tokenize_bpe(text)`

Tokenise un texte via BPE.

**Paramètres:**
- `text` (string) : Texte à tokeniser

**Retour:**
- `tokens` (table) : Liste de token IDs

---

### `tokenizer.set_max_length(max_length)`

Définit la longueur maximale des séquences.

**Paramètres:**
- `max_length` (integer) : Longueur maximale

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `tokenizer.pad_sequence(tokens, max_length, pad_id)`

Pad une séquence à une longueur donnée.

**Paramètres:**
- `tokens` (table) : Séquence à padder
- `max_length` (integer) : Longueur cible
- `pad_id` (integer, optionnel) : ID du token de padding (défaut: tokenizer.pad_id())

**Retour:**
- `padded` (table) : Séquence paddée

---

### `tokenizer.batch_tokenize(texts)`

Tokenise une liste de textes.

**Paramètres:**
- `texts` (table) : Liste de strings

**Retour:**
- `batches` (table) : Liste de listes de token IDs

---

### `tokenizer.print_stats()`

Affiche les statistiques du vocabulaire.

---

### `tokenizer.get_frequencies()`

Retourne les fréquences des tokens.

**Retour:**
- `freqs` (table) : Map {token: count}

---

### `tokenizer.analyze_text(text)`

Analyse un texte (statistiques: mots, chars, densité, etc.).

**Paramètres:**
- `text` (string) : Texte à analyser

**Retour:**
- `results` (table) : Statistiques diverses

---

### `tokenizer.extract_keywords(text, top_k)`

Extrait les mots-clés d'un texte (heuristique: fréquence, longueur).

**Paramètres:**
- `text` (string) : Texte source
- `top_k` (integer, optionnel) : Nombre de keywords (défaut: 10)

**Retour:**
- `keywords` (table) : Liste de mots-clés

---

### Sauvegarde/Chargement

- `tokenizer.save(path)` : Sauvegarde le vocabulaire en JSON
- `tokenizer.load(path)` : Charge le vocabulaire depuis JSON

---

## Module `dataset`

### `dataset.load(dir)`

Charge un dataset depuis un répertoire.

**Paramètres:**
- `dir` (string) : Chemin du répertoire contenant les fichiers texte

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `dataset.prepare_sequences(max_length)`

Prépare les séquences tokenisées pour l'entraînement. Les séquences sont stockées dans le contexte interne et utilisées par `model.train()`.

**Paramètres:**
- `max_length` (integer) : Longueur maximale des séquences

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

## Module `memory`

Gestion avancée de la RAM via le `AdvancedRAMManager`.

### `memory.config(cfg)`

Configure le gestionnaire mémoire.

**Paramètres:**
- `cfg` (table) : Configuration

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `memory.get_stats()`

Récupère les statistiques mémoire.

**Retour:**
- `stats` (table) : Statistiques
  - `current_mb` (number) : Utilisation courante (MB)
  - `peak_mb` (number) : Pic d'utilisation (MB)
  - `usage_percent` (number) : Pourcentage d'utilisation

---

### `memory.print_stats()`

Affiche les statistiques mémoire.

---

### `memory.clear()`

Libère les caches mémoire.

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `memory.get_usage()`

Retourne l'utilisation mémoire actuelle.

**Retour:**
- `usage_mb` (number) : Utilisation en MB

---

### `memory.set_limit(limit_mb)`

Définit une limite mémoire.

**Paramètres:**
- `limit_mb` (number) : Limite en MB

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

## Module `guard`

Enforcement strict des limites mémoire via `MemoryGuard`. Arrête l'exécution si la limite est dépassée.

### `guard.set_limit(limit_mb)`

Définit une limite stricte de mémoire.

**Paramètres:**
- `limit_mb` (number) : Limite en MB

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `guard.get_stats()`

Récupère les statistiques du guard.

**Retour:**
- `stats` (table) : Statistiques
  - `current_mb` (number) : Utilisation courante
  - `peak_mb` (number) : Pic
  - `limit_mb` (number) : Limite
  - `usage_percent` (number) : Pourcentage

---

### `guard.print_stats()`

Affiche les statistiques du guard.

---

### `guard.reset()`

Réinitialise les compteurs du guard.

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

## Module `allocator`

Gestion dynamique des tenseurs avec offload et compression via `DynamicTensorAllocator`.

### `allocator.configure(config)`

Configure l'allocateur.

**Paramètres:**
- `config` (table) : Configuration
  - `max_tensors` (integer) : Nombre max de tenseurs en mémoire
  - `offload_threshold_mb` (number) : Seuil de RAM pour offload
  - `swap_strategy` (string) : Stratégie de swap ("lru", "fifo", etc.)
  - `max_ram_gb` (number) : Limite RAM globale
  - `enable_compression` (boolean) : Activer compression LZ4

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

**Exemple:**
```lua
allocator.configure({
  max_tensors = 1000,
  offload_threshold_mb = 6000,
  enable_compression = true,
  max_ram_gb = 8
})
```

---

### `allocator.print_stats()`

Affiche les statistiques de l'allocateur.

---

### `allocator.get_stats()`

Récupère les statistiques.

**Retour:**
- `stats` (table) : Statistiques
  - `tensor_count` (integer) : Nombre de tenseurs
  - `loaded_count` (integer) : Tenseurs chargés en mémoire

---

## Module `htop`

Monitoring temps réel style htop dans le terminal via `HtopDisplay` et `AsyncMonitor`.

### `htop.create(config)`

Crée le monitor avec configuration optionnelle.

**Paramètres:**
- `config` (table, optionnel) : Configuration
  - `enable_viz` (boolean) : Activer visualisation graphique
  - `refresh_rate` (integer) : Taux de rafraîchissement (ms)

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `htop.enable(enabled)`

Active/désactive l'affichage htop.

**Paramètres:**
- `enabled` (boolean) : État

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `htop.update(metrics)`

Met à jour les métriques affichées.

**Paramètres:**
- `metrics` (table) : Structure HtopMetrics
  - `epoch` (integer) : Epoch courante
  - `total_epochs` (integer) : Total epochs
  - `batch` (integer) : Batch courant
  - `total_batches` (integer) : Total batches
  - `loss` (number) : Loss courante
  - `avg_loss` (number) : Loss moyenne
  - `lr` (number) : Learning rate
  - `batch_time_ms` (integer, optionnel) : Temps batch (ms)
  - `memory_mb` (integer, optionnel) : Mémoire utilisée (MB)
  - `memory_freed` (integer, optionnel) : Mémoire libérée
  - `bps` (number, optionnel) : Batches per second
  - `params` (integer, optionnel) : Nombre de paramètres
  - `timestep` (number, optionnel) : Timestep (diffusion)
  - `kl` (number, optionnel) : KL divergence
  - `wass` (number, optionnel) : Wasserstein distance
  - `ent` (number, optionnel) : Entropy
  - `mom` (number, optionnel) : Momentum
  - `spat` (number, optionnel) : Spatial metric
  - `temp` (number, optionnel) : Temperature
  - `mse` (number, optionnel) : MSE

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

**Exemple:**
```lua
htop.update({
  epoch = 5,
  total_epochs = 10,
  batch = 42,
  total_batches = 100,
  loss = 2.345,
  avg_loss = 2.450,
  lr = 3e-4,
  memory_mb = 5120
})
```

---

### `htop.render()`

Force un rendu de l'affichage.

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `htop.clear()`

Efface l'affichage htop.

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

## Module `viz`

Visualisation graphique via SFML (fenêtre, images, métriques, courbes).

### `viz.create(title, width, height)`

Crée une fenêtre de visualisation.

**Paramètres:**
- `title` (string, optionnel) : Titre de la fenêtre (défaut: "Mimir Visualizer")
- `width` (integer, optionnel) : Largeur (pixels, défaut: 1280)
- `height` (integer, optionnel) : Hauteur (pixels, défaut: 720)

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `viz.initialize()`

Initialise et ouvre la fenêtre.

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `viz.is_open()`

Vérifie si la fenêtre est ouverte.

**Retour:**
- `is_open` (boolean) : État de la fenêtre

---

### `viz.process_events()`

Traite les événements fenêtre (fermeture, clavier, souris).

---

### `viz.update()`

Met à jour et affiche le rendu.

---

### `viz.add_image(pixels, width, height, channels)`

Affiche une image dans le visualiseur.

**Paramètres:**
- `pixels` (table) : Tableau de pixels (valeurs 0-255, row-major order)
- `width` (integer) : Largeur de l'image
- `height` (integer) : Hauteur de l'image
- `channels` (integer) : Nombre de canaux (3=RGB, 4=RGBA)

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

**Exemple:**
```lua
-- Image 64x64 RGB
local pixels = {}
for i = 1, 64*64*3 do
  pixels[i] = math.random(0, 255)
end
viz.add_image(pixels, 64, 64, 3)
```

---

### `viz.update_metrics(metrics)`

Met à jour les métriques d'entraînement affichées.

**Paramètres:**
- `metrics` (table) : Métriques VizMetrics
  - `epoch` (integer, optionnel) : Epoch
  - `total_epochs` (integer, optionnel) : Total epochs
  - `loss` (number, optionnel) : Loss
  - `avg_loss` (number, optionnel) : Loss moyenne
  - `lr` (number, optionnel) : Learning rate
  - `memory_mb` (number, optionnel) : Mémoire (MB)
  - `tokens_per_sec` (number, optionnel) : Tokens/sec
  - `custom` (table, optionnel) : Métriques custom {name: value}

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `viz.add_loss_point(loss)`

Ajoute un point à l'historique de loss (pour courbe).

**Paramètres:**
- `loss` (number) : Valeur de loss

---

### `viz.clear()`

Efface le visualiseur.

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `viz.set_enabled(enabled)`

Active/désactive le visualiseur.

**Paramètres:**
- `enabled` (boolean) : État

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `viz.save_loss_history(filepath)`

Sauvegarde l'historique de loss dans un fichier.

**Paramètres:**
- `filepath` (string) : Chemin du fichier de sortie

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

## Fonctions Globales

### `log(message)`

Affiche un message dans la console (binding C++ vers std::cout).

**Paramètres:**
- `message` (string) : Message à logger

---

### `read_json(filepath)`

Lit un fichier JSON et retourne une table Lua.

**Paramètres:**
- `filepath` (string) : Chemin du fichier JSON

**Retour:**
- `data` (table ou nil) : Données parsées, ou nil en cas d'erreur

**Exemple:**
```lua
local config = read_json("config.json")
if config then
  print("Vocab size:", config.vocab_size)
end
```

---

### `write_json(filepath, data)`

Écrit une table Lua dans un fichier JSON.

**Paramètres:**
- `filepath` (string) : Chemin du fichier de sortie
- `data` (table) : Données à sérialiser

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

**Exemple:**
```lua
local ok, err = write_json("output.json", {
  vocab_size = 50000,
  embed_dim = 768
})
```

---

## Exemples Complets

### Exemple 1: Entraînement Transformer Simple

```lua
-- Configuration mémoire
memory.set_limit(8000) -- 8 GB
allocator.configure({max_tensors = 1000, enable_compression = true})

-- Tokenizer
tokenizer.create(50000)
local corpus = io.open("data/corpus.txt"):read("*all")
tokenizer.ensure_vocab_from_text(corpus)
tokenizer.save("tokenizer.json")

-- Dataset
dataset.load("data/train")
dataset.prepare_sequences(512)

-- Modèle
model.create("transformer", {
  vocab_size = tokenizer.vocab_size(),
  embed_dim = 512,
  num_layers = 6,
  num_heads = 8,
  d_ff = 2048,
  max_seq_len = 512,
  dropout = 0.1
})
local ok, params = model.build()
print(string.format("Modèle construit: %d paramètres", params))
model.init_weights("xavier", 42)

-- Monitoring
htop.create()
htop.enable(true)

-- Entraînement
model.train(10, 3e-4)

-- Sauvegarde
model.save("checkpoints/final")
tokenizer.save("checkpoints/tokenizer.json")
```

---

### Exemple 2: Génération avec Modèle Chargé

```lua
-- Charger tokenizer et modèle
tokenizer.load("checkpoints/tokenizer.json")
model.load("checkpoints/final")

-- Inférence
local prompt = "Once upon a time"
local output = model.infer(prompt)
print("Génération:", output)
```

---

### Exemple 3: Monitoring Avancé avec Visualisation

```lua
-- Configuration
memory.set_limit(12000) -- 12 GB
htop.create({enable_viz = false})
htop.enable(true)

-- Visualiseur
viz.create("Training Monitor", 1920, 1080)
viz.initialize()

-- Tokenizer, dataset, modèle...
-- (voir exemple 1)

-- Boucle d'entraînement custom
for epoch = 1, 10 do
  for batch = 1, 100 do
    -- Forward, backward, optimizer_step...
    
    -- Mise à jour monitoring
    htop.update({
      epoch = epoch,
      total_epochs = 10,
      batch = batch,
      total_batches = 100,
      loss = current_loss,
      avg_loss = avg_loss,
      lr = 3e-4,
      memory_mb = memory.get_usage()
    })
    
    viz.update_metrics({
      epoch = epoch,
      loss = current_loss,
      lr = 3e-4
    })
    viz.add_loss_point(current_loss)
    
    if viz.is_open() then
      viz.process_events()
      viz.update()
    end
  end
end

-- Sauvegarde historique
viz.save_loss_history("loss_history.json")
```

---

### Exemple 4: VAE avec Visualisation Images

```lua
-- Créer VAE
architectures.vae({
  input_dim = 784,  -- 28x28 MNIST
  latent_dim = 20,
  encoder_hidden = 400,
  decoder_hidden = 400
})
model.build()

-- Visualiseur
viz.create("VAE Training", 800, 600)
viz.initialize()

-- Entraînement...

-- Visualiser une reconstruction
local reconstructed_pixels = {} -- obtenir depuis le modèle
viz.add_image(reconstructed_pixels, 28, 28, 1) -- grayscale
viz.update()
```

---

## Notes de Performance

### Optimisations Mémoire

1. **Limites strictes** : Utilisez `guard.set_limit()` pour éviter les OOM
2. **Compression** : Activez `allocator.configure({enable_compression = true})`
3. **Offload** : Configurez `offload_threshold_mb` pour swap sur disque
4. **Clear périodique** : Appelez `memory.clear()` entre epochs

### Optimisations CPU

1. **Hardware detection** : Vérifiez `model.hardware_caps()` pour AVX2/FMA
2. **Backend** : Utilisez `model.set_hardware("cpu")` explicitement
3. **Batching** : Utilisez des batch sizes adaptés à votre RAM

### Monitoring

1. **htop** : Léger, idéal pour serveurs sans GUI
2. **viz** : Plus lourd (SFML), idéal pour desktop avec visualisation

---

## Changelog

**Version 2.0.0** (Décembre 2025)
- API complète documentée
- Ajout modules `memory`, `guard`, `allocator`, `htop`, `viz`
- Tokenizer étendu (BPE, analyse, keywords)
- Architectures pré-définies (UNet, VAE, ViT, GAN, Diffusion, ResNet, MobileNet)
- Hardware detection et backends
- Fonctions globales `log`, `read_json`, `write_json`

---

## Références

- **Code source** : `/src/LuaScripting.cpp`, `/src/LuaScripting.hpp`
- **Stub IDE** : `mimir-api.lua`
- **Exemples** : `/scripts/*.lua`
- **Documentation** : `/docs/LUA_API.md`, `/docs/PIPELINE_API.md`

---

**© 2025 Mímir Framework - bri45**
