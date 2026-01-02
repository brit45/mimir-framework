# Mímir Framework - API Lua Complète

**Version:** 2.3.0  
**Date:** 28 Décembre 2025  
**Auteur:** bri45  
**Stub:** mimir-api.lua (EmmyLua annotations)

Ce document décrit l'API Lua complète du framework Mímir, exposant toutes les fonctionnalités du moteur C++ via des bindings Lua.

> ⚠️ **Syntaxe Recommandée:** Utiliser `Mimir.Module.*` pour bénéficier de l'autocompletion IDE et du type checking (annotations EmmyLua dans `mimir-api.lua`).

---

## Table des matières

1. [Architecture Générale](#architecture-générale)
2. [Module `Mimir.Model`](#module-mimirmodel)
3. [Module `Mimir.Architectures`](#module-mimirarchitectures)
4. [Module `Mimir.Flux`](#module-mimirflux)
5. [Module `Mimir.FluxModel`](#module-mimirfluxmodel)
6. [Module `Mimir.Layers`](#module-mimirlayers)
7. [Module `Mimir.Tokenizer`](#module-mimirtokenizer)
8. [Module `Mimir.Dataset`](#module-mimirdataset)
9. [Module `Mimir.Memory`](#module-mimirmemory)
10. [Module `Mimir.MemoryGuard`](#module-mimirmemoryguard)
11. [Module `Mimir.Allocator`](#module-mimirallocator)
12. [Module `Mimir.Serialization`](#module-mimirserialization)
13. [Module `Mimir.Htop`](#module-mimirhtop)
14. [Module `Mimir.Viz`](#module-mimirviz)
15. [Fonctions Globales](#fonctions-globales)
16. [Exemples Complets](#exemples-complets)

---

## Architecture Générale

Le framework Mímir utilise Lua comme langage de scripting pour piloter l'entraînement et l'inférence de modèles de deep learning. L'architecture est organisée en modules sous le namespace `Mimir.*`, chacun exposant une API spécifique :

- **`Mimir.Model`** : Gestion du cycle de vie du modèle (création, build, train, infer, save/load)
- **`Mimir.Architectures`** : Builders pour architectures pré-définies (UNet, VAE, ViT, GAN, Flux, etc.)
- **`Mimir.Flux`** : API fonctionnelle pour génération d'images guidée par texte
- **`Mimir.FluxModel`** : API orientée objet pour le modèle Flux
- **`Mimir.Layers`** : Opérations de couches bas niveau (placeholders)
- **`Mimir.Tokenizer`** : Tokenization (word-level, BPE), gestion vocabulaire
- **`Mimir.Dataset`** : Chargement et préparation des données
- **`Mimir.Memory`** : Gestion avancée de la RAM (AdvancedRAMManager)
- **`Mimir.MemoryGuard`** : Enforcement strict des limites mémoire (API moderne, recommandée)
- **`Mimir.Allocator`** : Allocation dynamique de tenseurs avec offload/compression
- **`Mimir.Serialization`** : Sauvegarde/chargement (SafeTensors, RawFolder, DebugJson)
- **`Mimir.Htop`** : Monitoring temps réel style htop en terminal
- **`Mimir.Viz`** : Visualisation graphique (SFML) - images, métriques, loss curves

> **Note:** Les globales (`model.*`, `Mimir.Memory.*`, etc.) restent disponibles pour compatibilité ascendante, mais la syntaxe `Mimir.*` est **fortement recommandée**.

### Flux typique

```lua
-- 1. Configuration mémoire
Mimir.Mimir.Memory.set_limit(8000) -- 8 GB
Mimir.Mimir.Allocator.configure({max_tensors = 1000, enable_compression = true})

-- 2. Tokenizer
Mimir.Mimir.Tokenizer.create(50000)
Mimir.Mimir.Tokenizer.ensure_vocab_from_text(corpus_text)

-- 3. Dataset
Mimir.Mimir.Dataset.load("data/my_corpus")
Mimir.Mimir.Dataset.prepare_sequences(512)

-- 4. Modèle
Mimir.Model.create("transformer", {vocab_size = 50000, embed_dim = 512, num_layers = 6})
Mimir.Model.build()
Mimir.Model.init_weights("xavier")

-- 5. Monitoring
Mimir.Mimir.Htop.create({enable_viz = false})
Mimir.Mimir.Htop.enable(true)

-- 6. Entraînement
Mimir.Model.train(10, 3e-4)

-- 7. Sauvegarde
Mimir.Serialization.save_safetensors("checkpoints/run1/final.safetensors")
Mimir.Mimir.Tokenizer.save("checkpoints/run1/Mimir.Tokenizer.json")
```

---

## Module `Mimir.Model`

### `Mimir.Model.create(model_type, config)`

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
local ok, err = Mimir.Model.create("transformer", {
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

### `Mimir.Model.build()`

Construit le modèle (alloue la mémoire, initialise les paramètres). Doit être appelé après `Mimir.Model.create()`.

**Retour:**
- `ok` (boolean) : Succès
- `params` (integer, optionnel) : Nombre total de paramètres scalaires
- `err` (string, optionnel) : Message d'erreur

**Exemple:**
```lua
local ok, params, err = Mimir.Model.build()
if ok then
  print(string.format("Modèle construit: %d paramètres", params))
else
  print("Erreur:", err)
end
```

---

### `Mimir.Model.train(epochs, learning_rate)`

Entraîne le modèle sur le dataset chargé.

**Prérequis:**
- Dataset chargé via `Mimir.Mimir.Dataset.load()`
- Séquences préparées via `Mimir.Mimir.Dataset.prepare_sequences()`
- Modèle construit via `Mimir.Model.build()`

**Paramètres:**
- `epochs` (integer) : Nombre d'epochs
- `learning_rate` (number) : Taux d'apprentissage (ex: 3e-4)

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

**Exemple:**
```lua
local ok, err = Mimir.Model.train(10, 3e-4)
if not ok then
  print("Erreur entraînement:", err)
end
```

---

### `Mimir.Model.infer(input)`

Effectue une inférence sur un prompt texte ou une séquence de tokens.

**Paramètres:**
- `input` (string ou table) : Prompt texte ou séquence de token IDs

**Retour:**
- `output` (string ou nil) : Texte généré, ou nil en cas d'erreur

**Exemple:**
```lua
local output = Mimir.Model.infer("Once upon a time")
if output then
  print("Génération:", output)
else
  print("Erreur inférence")
end
```

---

### `Mimir.Model.save(dir)`

Sauvegarde le modèle (checkpoint) dans un répertoire.

**Paramètres:**
- `dir` (string) : Chemin du répertoire de sauvegarde

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

**Exemple:**
```lua
local ok, err = Mimir.Model.save("checkpoints/run1/epoch_10")
```

---

### `Mimir.Model.load(dir)`

Charge un modèle depuis un Mimir.Checkpoint.

**Paramètres:**
- `dir` (string) : Chemin du répertoire de chargement

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `Mimir.Model.allocate_params()`

Alloue explicitement les paramètres du modèle (si non fait par `build()`).

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `Mimir.Model.init_weights(init_method, seed)`

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
Mimir.Model.init_weights("xavier", 42)
```

---

### `Mimir.Model.total_params()`

Retourne le nombre total de paramètres du modèle.

**Retour:**
- `params` (integer) : Nombre de paramètres

---

### `Mimir.Model.forward(input)`

Effectue une passe forward manuelle (bas niveau).

**Paramètres:**
- `input` : Entrée (dépend du modèle)

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `Mimir.Model.backward()`

Effectue une passe backward manuelle (calcul des gradients).

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `Mimir.Model.optimizer_step(learning_rate)`

Effectue un pas d'optimisation manuel.

**Paramètres:**
- `learning_rate` (number) : Taux d'apprentissage

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `Mimir.Model.set_hardware(backend)`

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

### `Mimir.Model.hardware_caps()`

Retourne les capacités hardware détectées.

**Retour:**
- `caps` (table) : Capacités
  - `avx2` (boolean) : Support AVX2
  - `fma` (boolean) : Support FMA
  - `f16c` (boolean) : Support F16C
  - `bmi2` (boolean) : Support BMI2

**Exemple:**
```lua
local caps = Mimir.Model.hardware_caps()
print("AVX2:", caps.avx2)
print("FMA:", caps.fma)
```

---

## Module `architectures`

Ce module fournit des builders pour construire rapidement des architectures pré-définies.

### `Mimir.Mimir.Architectures.unet(config)`

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

### `Mimir.Mimir.Architectures.vae(config)`

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

### `Mimir.Mimir.Architectures.vit(config)`

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

### `Mimir.Mimir.Architectures.gan(config)`

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

### `Mimir.Architectures.diffusion(config)`

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

### `Mimir.Mimir.Architectures.transformer(config)`

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

### `Mimir.Architectures.resnet(config)`

Construit un ResNet.

**Configuration:**
```lua
{
  num_classes = 1000,
  input_channels = 3
}
```

---

### `Mimir.Architectures.mobilenet(config)`

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

### `Mimir.Mimir.Tokenizer.create(max_vocab)`

Crée un tokenizer avec une taille de vocabulaire maximale.

**Paramètres:**
- `max_vocab` (integer) : Taille max du vocabulaire

**Retour:**
- `ok` (boolean) : Succès

---

### `Mimir.Tokenizer.tokenize(text)`

Tokenise un texte en séquence de token IDs.

**Paramètres:**
- `text` (string) : Texte à tokeniser

**Retour:**
- `tokens` (table) : Liste de token IDs (indices Lua 1..N)

**Exemple:**
```lua
local tokens = Mimir.Tokenizer.tokenize("Hello world")
-- tokens = {1523, 2891}
```

---

### `Mimir.Tokenizer.detokenize(tokens)`

Convertit une séquence de tokens en texte.

**Paramètres:**
- `tokens` (table) : Liste de token IDs

**Retour:**
- `text` (string) : Texte résultant

---

### `Mimir.Tokenizer.vocab_size()`

Retourne la taille actuelle du vocabulaire.

**Retour:**
- `size` (integer) : Taille du vocabulaire

---

### `Mimir.Tokenizer.add_token(token)`

Ajoute un token au vocabulaire.

**Paramètres:**
- `token` (string) : Token à ajouter

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `Mimir.Mimir.Tokenizer.ensure_vocab_from_text(text)`

Étend le vocabulaire en analysant un texte (ajoute les mots manquants).

**Paramètres:**
- `text` (string) : Texte à analyser

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `Mimir.Tokenizer.tokenize_ensure(text)`

Tokenise un texte et étend le vocabulaire si nécessaire.

**Paramètres:**
- `text` (string) : Texte à tokeniser

**Retour:**
- `tokens` (table) : Liste de token IDs

---

### Tokens Spéciaux

- `Mimir.Tokenizer.pad_id()` : ID du token de padding
- `Mimir.Tokenizer.unk_id()` : ID du token unknown
- `Mimir.Tokenizer.seq_id()` : ID du token de séquence
- `Mimir.Tokenizer.mod_id()` : ID du token de modulation
- `Mimir.Tokenizer.mag_id()` : ID du token de magnitude

---

### `Mimir.Tokenizer.get_token_by_id(id)`

Récupère un token string depuis son ID.

**Paramètres:**
- `id` (integer) : ID du token

**Retour:**
- `token` (string) : Token correspondant

---

### BPE (Byte-Pair Encoding)

#### `Mimir.Tokenizer.learn_bpe(corpus_path, vocab_target)`

Apprend un vocabulaire BPE depuis un corpus.

**Paramètres:**
- `corpus_path` (string) : Chemin du fichier corpus
- `vocab_target` (integer, optionnel) : Taille cible du vocabulaire

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

#### `Mimir.Tokenizer.tokenize_bpe(text)`

Tokenise un texte via BPE.

**Paramètres:**
- `text` (string) : Texte à tokeniser

**Retour:**
- `tokens` (table) : Liste de token IDs

---

### `Mimir.Tokenizer.set_max_length(max_length)`

Définit la longueur maximale des séquences.

**Paramètres:**
- `max_length` (integer) : Longueur maximale

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `Mimir.Tokenizer.pad_sequence(tokens, max_length, pad_id)`

Pad une séquence à une longueur donnée.

**Paramètres:**
- `tokens` (table) : Séquence à padder
- `max_length` (integer) : Longueur cible
- `pad_id` (integer, optionnel) : ID du token de padding (défaut: Mimir.Tokenizer.pad_id())

**Retour:**
- `padded` (table) : Séquence paddée

---

### `Mimir.Tokenizer.batch_tokenize(texts)`

Tokenise une liste de textes.

**Paramètres:**
- `texts` (table) : Liste de strings

**Retour:**
- `batches` (table) : Liste de listes de token IDs

---

### `Mimir.Tokenizer.print_stats()`

Affiche les statistiques du vocabulaire.

---

### `Mimir.Tokenizer.get_frequencies()`

Retourne les fréquences des tokens.

**Retour:**
- `freqs` (table) : Map {token: count}

---

### `Mimir.Tokenizer.analyze_text(text)`

Analyse un texte (statistiques: mots, chars, densité, etc.).

**Paramètres:**
- `text` (string) : Texte à analyser

**Retour:**
- `results` (table) : Statistiques diverses

---

### `Mimir.Tokenizer.extract_keywords(text, top_k)`

Extrait les mots-clés d'un texte (heuristique: fréquence, longueur).

**Paramètres:**
- `text` (string) : Texte source
- `top_k` (integer, optionnel) : Nombre de keywords (défaut: 10)

**Retour:**
- `keywords` (table) : Liste de mots-clés

---

### Sauvegarde/Chargement

- `Mimir.Mimir.Tokenizer.save(path)` : Sauvegarde le vocabulaire en JSON
- `Mimir.Mimir.Tokenizer.load(path)` : Charge le vocabulaire depuis JSON

---

## Module `dataset`

### `Mimir.Mimir.Dataset.load(dir)`

Charge un dataset depuis un répertoire.

**Paramètres:**
- `dir` (string) : Chemin du répertoire contenant les fichiers texte

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `Mimir.Mimir.Dataset.prepare_sequences(max_length)`

Prépare les séquences tokenisées pour l'entraînement. Les séquences sont stockées dans le contexte interne et utilisées par `Mimir.Model.train()`.

**Paramètres:**
- `max_length` (integer) : Longueur maximale des séquences

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

## Module `memory`

Gestion avancée de la RAM via le `AdvancedRAMManager`.

### `Mimir.Memory.config(cfg)`

Configure le gestionnaire mémoire.

**Paramètres:**
- `cfg` (table) : Configuration

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `Mimir.Mimir.Memory.get_stats()`

Récupère les statistiques mémoire.

**Retour:**
- `stats` (table) : Statistiques
  - `current_mb` (number) : Utilisation courante (MB)
  - `peak_mb` (number) : Pic d'utilisation (MB)
  - `usage_percent` (number) : Pourcentage d'utilisation

---

### `Mimir.Memory.print_stats()`

Affiche les statistiques mémoire.

---

### `Mimir.Memory.clear()`

Libère les caches mémoire.

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `Mimir.Mimir.Memory.get_usage()`

Retourne l'utilisation mémoire actuelle.

**Retour:**
- `usage_mb` (number) : Utilisation en MB

---

### `Mimir.Mimir.Memory.set_limit(limit_mb)`

Définit une limite mémoire.

**Paramètres:**
- `limit_mb` (number) : Limite en MB

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

## Module `guard`

Enforcement strict des limites mémoire via `MemoryGuard`. Arrête l'exécution si la limite est dépassée.

### `Mimir.Guard.set_limit(limit_mb)`

Définit une limite stricte de mémoire.

**Paramètres:**
- `limit_mb` (number) : Limite en MB

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `Mimir.Guard.get_stats()`

Récupère les statistiques du Mimir.Mimir.MemoryGuard.

**Retour:**
- `stats` (table) : Statistiques
  - `current_mb` (number) : Utilisation courante
  - `peak_mb` (number) : Pic
  - `limit_mb` (number) : Limite
  - `usage_percent` (number) : Pourcentage

---

### `Mimir.Mimir.MemoryGuard.print_stats()`

Affiche les statistiques du Mimir.Mimir.MemoryGuard.

---

### `Mimir.Guard.reset()`

Réinitialise les compteurs du Mimir.Mimir.MemoryGuard.

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

## Module `allocator`

Gestion dynamique des tenseurs avec offload et compression via `DynamicTensorAllocator`.

### `Mimir.Mimir.Allocator.configure(config)`

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
Mimir.Mimir.Allocator.configure({
  max_tensors = 1000,
  offload_threshold_mb = 6000,
  enable_compression = true,
  max_ram_gb = 8
})
```

---

### `Mimir.Allocator.print_stats()`

Affiche les statistiques de l'allocateur.

---

### `Mimir.Mimir.Allocator.get_stats()`

Récupère les statistiques.

**Retour:**
- `stats` (table) : Statistiques
  - `tensor_count` (integer) : Nombre de tenseurs
  - `loaded_count` (integer) : Tenseurs chargés en mémoire

---

## Module `htop`

Monitoring temps réel style htop dans le terminal via `HtopDisplay` et `AsyncMonitor`.

### `Mimir.Mimir.Htop.create(config)`

Crée le monitor avec configuration optionnelle.

**Paramètres:**
- `config` (table, optionnel) : Configuration
  - `enable_viz` (boolean) : Activer visualisation graphique
  - `refresh_rate` (integer) : Taux de rafraîchissement (ms)

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `Mimir.Mimir.Htop.enable(enabled)`

Active/désactive l'affichage Mimir.Htop.

**Paramètres:**
- `enabled` (boolean) : État

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `Mimir.Htop.update(metrics)`

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
Mimir.Htop.update({
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

### `Mimir.Htop.render()`

Force un rendu de l'affichage.

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `Mimir.Htop.clear()`

Efface l'affichage Mimir.Htop.

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

## Module `viz`

Visualisation graphique via SFML (fenêtre, images, métriques, courbes).

### `Mimir.Mimir.Viz.create(title, width, height)`

Crée une fenêtre de visualisation.

**Paramètres:**
- `title` (string, optionnel) : Titre de la fenêtre (défaut: "Mimir Visualizer")
- `width` (integer, optionnel) : Largeur (pixels, défaut: 1280)
- `height` (integer, optionnel) : Hauteur (pixels, défaut: 720)

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `Mimir.Mimir.Viz.initialize()`

Initialise et ouvre la fenêtre.

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `Mimir.Viz.is_open()`

Vérifie si la fenêtre est ouverte.

**Retour:**
- `is_open` (boolean) : État de la fenêtre

---

### `Mimir.Viz.process_events()`

Traite les événements fenêtre (fermeture, clavier, souris).

---

### `Mimir.Viz.update()`

Met à jour et affiche le rendu.

---

### `Mimir.Mimir.Viz.add_image(pixels, width, height, channels)`

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
Mimir.Mimir.Viz.add_image(pixels, 64, 64, 3)
```

---

### `Mimir.Viz.update_metrics(metrics)`

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

### `Mimir.Viz.add_loss_point(loss)`

Ajoute un point à l'historique de loss (pour courbe).

**Paramètres:**
- `loss` (number) : Valeur de loss

---

### `Mimir.Viz.clear()`

Efface le visualiseur.

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `Mimir.Viz.set_enabled(enabled)`

Active/désactive le visualiseur.

**Paramètres:**
- `enabled` (boolean) : État

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `Mimir.Mimir.Viz.save_loss_history(filepath)`

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
Mimir.Mimir.Memory.set_limit(8000) -- 8 GB
Mimir.Mimir.Allocator.configure({max_tensors = 1000, enable_compression = true})

-- Tokenizer
Mimir.Mimir.Tokenizer.create(50000)
local corpus = io.open("data/corpus.txt"):read("*all")
Mimir.Mimir.Tokenizer.ensure_vocab_from_text(corpus)
Mimir.Mimir.Tokenizer.save("Mimir.Tokenizer.json")

-- Dataset
Mimir.Mimir.Dataset.load("data/train")
Mimir.Mimir.Dataset.prepare_sequences(512)

-- Modèle
Mimir.Model.create("transformer", {
  vocab_size = Mimir.Tokenizer.vocab_size(),
  embed_dim = 512,
  num_layers = 6,
  num_heads = 8,
  d_ff = 2048,
  max_seq_len = 512,
  dropout = 0.1
})
local ok, params = Mimir.Model.build()
print(string.format("Modèle construit: %d paramètres", params))
Mimir.Model.init_weights("xavier", 42)

-- Monitoring
Mimir.Mimir.Htop.create()
Mimir.Mimir.Htop.enable(true)

-- Entraînement
Mimir.Model.train(10, 3e-4)

-- Sauvegarde
Mimir.Model.save("checkpoints/final")
Mimir.Mimir.Tokenizer.save("checkpoints/Mimir.Tokenizer.json")
```

---

### Exemple 2: Génération avec Modèle Chargé

```lua
-- Charger tokenizer et modèle
Mimir.Mimir.Tokenizer.load("checkpoints/Mimir.Tokenizer.json")
Mimir.Model.load("checkpoints/final")

-- Inférence
local prompt = "Once upon a time"
local output = Mimir.Model.infer(prompt)
print("Génération:", output)
```

---

### Exemple 3: Monitoring Avancé avec Visualisation

```lua
-- Configuration
Mimir.Mimir.Memory.set_limit(12000) -- 12 GB
Mimir.Mimir.Htop.create({enable_viz = false})
Mimir.Mimir.Htop.enable(true)

-- Visualiseur
Mimir.Mimir.Viz.create("Training Monitor", 1920, 1080)
Mimir.Mimir.Viz.initialize()

-- Tokenizer, dataset, modèle...
-- (voir exemple 1)

-- Boucle d'entraînement custom
for epoch = 1, 10 do
  for batch = 1, 100 do
    -- Forward, backward, optimizer_step...
    
    -- Mise à jour monitoring
    Mimir.Htop.update({
      epoch = epoch,
      total_epochs = 10,
      batch = batch,
      total_batches = 100,
      loss = current_loss,
      avg_loss = avg_loss,
      lr = 3e-4,
      memory_mb = Mimir.Mimir.Memory.get_usage()
    })
    
    Mimir.Viz.update_metrics({
      epoch = epoch,
      loss = current_loss,
      lr = 3e-4
    })
    Mimir.Viz.add_loss_point(current_loss)
    
    if Mimir.Viz.is_open() then
      Mimir.Viz.process_events()
      Mimir.Viz.update()
    end
  end
end

-- Sauvegarde historique
Mimir.Mimir.Viz.save_loss_history("loss_history.json")
```

---

### Exemple 4: VAE avec Visualisation Images

```lua
-- Créer VAE
Mimir.Mimir.Architectures.vae({
  input_dim = 784,  -- 28x28 MNIST
  latent_dim = 20,
  encoder_hidden = 400,
  decoder_hidden = 400
})
Mimir.Model.build()

-- Visualiseur
Mimir.Mimir.Viz.create("VAE Training", 800, 600)
Mimir.Mimir.Viz.initialize()

-- Entraînement...

-- Visualiser une reconstruction
local reconstructed_pixels = {} -- obtenir depuis le modèle
Mimir.Mimir.Viz.add_image(reconstructed_pixels, 28, 28, 1) -- grayscale
Mimir.Viz.update()
```

---

## Notes de Performance

### Optimisations Mémoire

1. **Limites strictes** : Utilisez `Mimir.Guard.set_limit()` pour éviter les OOM
2. **Compression** : Activez `Mimir.Mimir.Allocator.configure({enable_compression = true})`
3. **Offload** : Configurez `offload_threshold_mb` pour swap sur disque
4. **Clear périodique** : Appelez `Mimir.Memory.clear()` entre epochs

### Optimisations CPU

1. **Hardware detection** : Vérifiez `Mimir.Model.hardware_caps()` pour AVX2/FMA
2. **Backend** : Utilisez `Mimir.Model.set_hardware("cpu")` explicitement
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
