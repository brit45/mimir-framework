# Mímir Framework - API Lua Complète

**Version:** 2.3.0  
**Date:** 12 Janvier 2026  
**Auteur:** bri45  
**Stub:** mimir-api.lua (EmmyLua annotations)

Ce document décrit l'API Lua complète du framework Mímir, exposant toutes les fonctionnalités du moteur C++ via des bindings Lua.

> ⚠️ **Syntaxe Recommandée:** Utiliser `Mimir.Module.*` pour bénéficier de l'autocompletion IDE et du type checking (annotations EmmyLua dans `mimir-api.lua`).

---

## Table des matières

1. [Architecture Générale](#architecture-générale)
2. [Module `Mimir.Model`](#module-mimirmodel)
3. [Module `Mimir.Architectures`](#module-mimirarchitectures)
4. [Module `Mimir.Tokenizer`](#module-mimirtokenizer)
5. [Module `Mimir.Dataset`](#module-mimirdataset)
6. [Module `Mimir.Memory`](#module-mimirmemory)
7. [Module `Mimir.Guard`](#module-mimirguard)
8. [Module `Mimir.MemoryGuard`](#module-mimirmemoryguard)
9. [Module `Mimir.Allocator`](#module-mimirallocator)
10. [Module `Mimir.Serialization`](#module-mimirserialization)
11. [Module `Mimir.Htop`](#module-mimirhtop)
12. [Module `Mimir.Viz`](#module-mimirviz)
13. [Fonctions Globales](#fonctions-globales)
14. [Exemples Complets](#exemples-complets)

---

## Architecture Générale

Le framework Mímir utilise Lua comme langage de scripting pour piloter l'entraînement et l'inférence de modèles de deep learning. L'architecture est organisée en modules sous le namespace `Mimir.*`, chacun exposant une API spécifique :

- **`Mimir.Model`** : Gestion du cycle de vie du modèle (création, build, train, infer, save/load)
- **`Mimir.Architectures`** : Helpers de registre (`available`, `default_config`)
- **`Mimir.Layers`** : Opérations de couches bas niveau (placeholders)
- **`Mimir.Tokenizer`** : Tokenization (word-level, BPE), gestion vocabulaire
- **`Mimir.Dataset`** : Chargement et préparation des données
- **`Mimir.Memory`** : Gestion avancée de la RAM (AdvancedRAMManager)
- **`Mimir.Guard`** : Enforcement strict (API legacy)
- **`Mimir.MemoryGuard`** : Enforcement strict (API moderne, recommandée)
- **`Mimir.Allocator`** : Allocation dynamique de tenseurs avec offload/compression
- **`Mimir.Serialization`** : Sauvegarde/chargement (SafeTensors, RawFolder, DebugJson)
- **`Mimir.Htop`** : Monitoring temps réel style htop en terminal
- **`Mimir.Viz`** : Visualisation graphique (SFML) - images, métriques, loss curves

> **Note:** Les globales (`model.*`, `Mimir.Memory.*`, etc.) restent disponibles pour compatibilité ascendante, mais la syntaxe `Mimir.*` est **fortement recommandée**.

### Workflow typique

```lua
-- 1. Configuration mémoire
Mimir.Memory.set_limit(8000) -- 8 GB
Mimir.Allocator.configure({max_tensors = 1000, enable_compression = true})

-- 2. Tokenizer
Mimir.Tokenizer.create(50000)
Mimir.Tokenizer.ensure_vocab_from_text(corpus_text)

-- 3. Dataset
Mimir.Dataset.load("data/my_corpus")
Mimir.Dataset.prepare_sequences(512)

-- 4. Modèle
local cfg = Mimir.Architectures.default_config("transformer")
cfg.vocab_size = 50000
cfg.d_model = 512
cfg.num_layers = 6
Mimir.Model.create("transformer", cfg)
Mimir.Model.build() -- compat
Mimir.Model.init_weights("xavier")

-- 5. Monitoring
Mimir.Htop.create({enable_viz = false})
Mimir.Htop.enable(true)

-- 6. Entraînement
Mimir.Model.train(10, 3e-4)

-- 7. Sauvegarde
Mimir.Serialization.save("checkpoints/run1/final.safetensors", "safetensors")
Mimir.Tokenizer.save("checkpoints/run1/MIMIR_Tokenizer.json")
```

> Source de vérité: le stub IDE [mimir-api.lua](../../../mimir-api.lua) est synchronisé avec l’implémentation C++ ([src/LuaScripting.cpp](../../../src/LuaScripting.cpp)).

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
- Dataset chargé via `Mimir.Dataset.load()`
- Séquences préparées via `Mimir.Dataset.prepare_sequences()`
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

> **Mise à jour v2.3 (Jan 2026)** : les builders `Mimir.Architectures.<name>(cfg)` ne sont **pas** exposés.
> Le module `Mimir.Architectures` fournit uniquement des helpers pour le registre C++.

### `Mimir.Architectures.available()`

Liste les architectures disponibles dans le registre.

**Retour:**
- `names` (table|nil) : Liste de strings
- `err` (string|nil)

**Exemple:**
```lua
local names, err = Mimir.Architectures.available()
if not names then error(err) end
for _, n in ipairs(names) do
  print("arch:", n)
end
```

---

### `Mimir.Architectures.default_config(name)`

Retourne une configuration par défaut pour une architecture.

**Paramètres:**
- `name` (string)

**Retour:**
- `config` (table|nil)
- `err` (string|nil)

**Exemple:**
```lua
local cfg, err = Mimir.Architectures.default_config("transformer")
if not cfg then error(err) end
cfg.vocab_size = 32000
cfg.d_model = 512
cfg.num_layers = 6
cfg.num_heads = 8
assert(Mimir.Model.create("transformer", cfg))
```

---

## Module `tokenizer`

### `Mimir.Tokenizer.create(max_vocab)`

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

### `Mimir.Tokenizer.ensure_vocab_from_text(text)`

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

- `Mimir.Tokenizer.save(path)` : Sauvegarde le vocabulaire en JSON
- `Mimir.Tokenizer.load(path)` : Charge le vocabulaire depuis JSON

---

## Module `dataset`

### `Mimir.Dataset.load(dir)`

Charge un dataset depuis un répertoire.

**Paramètres:**
- `dir` (string) : Chemin du répertoire contenant les fichiers texte

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `Mimir.Dataset.prepare_sequences(max_length)`

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

### `Mimir.Memory.get_stats()`

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

### `Mimir.Memory.get_usage()`

Retourne l'utilisation mémoire actuelle.

**Retour:**
- `usage_mb` (number) : Utilisation en MB

---

### `Mimir.Memory.set_limit(limit_mb)`

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

Récupère les statistiques du guard (API legacy).

**Retour:**
- `stats` (table) : Statistiques
  - `current_mb` (number) : Utilisation courante
  - `peak_mb` (number) : Pic
  - `limit_mb` (number) : Limite
  - `usage_percent` (number) : Pourcentage

---

### `Mimir.Guard.print_stats()`

Affiche les statistiques du guard.

---

### `Mimir.Guard.reset()`

Réinitialise les compteurs du guard.

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

## Module `Mimir.MemoryGuard`

API moderne recommandée (fonctions en camelCase). Cette API retourne des valeurs en **bytes** pour l’usage courant/peak/limit, et une table en MB/% via `getStats()`.

### Fonctions

```lua
Mimir.MemoryGuard.setLimit(limit)         -- limit = bytes (grand nombre) OU GB (si <= 1000)
Mimir.MemoryGuard.getLimit()              -- bytes
Mimir.MemoryGuard.getCurrentUsage()       -- bytes
Mimir.MemoryGuard.getPeakUsage()          -- bytes
Mimir.MemoryGuard.getStats()              -- {current_mb, peak_mb, limit_mb, usage_percent}
Mimir.MemoryGuard.printStats()
Mimir.MemoryGuard.reset()
```

---

## Module `allocator`

Gestion dynamique des tenseurs avec offload et compression via `DynamicTensorAllocator`.

### `Mimir.Allocator.configure(config)`

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
Mimir.Allocator.configure({
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

### `Mimir.Allocator.get_stats()`

Récupère les statistiques.

**Retour:**
- `stats` (table) : Statistiques
  - `tensor_count` (integer) : Nombre de tenseurs
  - `loaded_count` (integer) : Tenseurs chargés en mémoire

---

## Module `Mimir.Serialization`

API de sérialisation v2.3 (recommandée). Permet de sauvegarder/charger des checkpoints et de détecter automatiquement le format.

### `Mimir.Serialization.save(path, format?)`

**Paramètres:**
- `path` (string) : Fichier (SafeTensors / DebugJson) ou dossier (RawFolder)
- `format` (string, optionnel) : "safetensors" | "raw_folder" | "debug_json" (si omis, auto-détecté via l’extension ou le chemin)

**Retour:**
- `ok` (boolean)
- `err` (string|nil)

**Exemple:**
```lua
assert(Mimir.Serialization.save("checkpoints/run1/model.safetensors", "safetensors"))
assert(Mimir.Serialization.save("checkpoints/run1/", "raw_folder"))
assert(Mimir.Serialization.save("checkpoints/run1/debug.json", "debug_json"))
```

---

### `Mimir.Serialization.load(path, format?)`

Charge un checkpoint (auto-détection par défaut).

---

### `Mimir.Serialization.detect_format(path)`

Retourne un string de format (ou nil + err).

---

### `Mimir.Serialization.save_enhanced_debug(path)`

Sauvegarde un DebugJson enrichi.

---

## Module `htop`

Monitoring temps réel style htop dans le terminal via `HtopDisplay` et `AsyncMonitor`.

### `Mimir.Htop.create(config)`

Crée le monitor avec configuration optionnelle.

**Paramètres:**
- `config` (table, optionnel) : Configuration
  - `enable_viz` (boolean) : Activer visualisation graphique
  - `refresh_rate` (integer) : Taux de rafraîchissement (ms)

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `Mimir.Htop.enable(enabled)`

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

### `Mimir.Viz.create(title, width, height)`

Crée une fenêtre de visualisation.

**Paramètres:**
- `title` (string, optionnel) : Titre de la fenêtre (défaut: "Mimir Visualizer")
- `width` (integer, optionnel) : Largeur (pixels, défaut: 1280)
- `height` (integer, optionnel) : Hauteur (pixels, défaut: 720)

**Retour:**
- `ok` (boolean) : Succès
- `err` (string, optionnel) : Message d'erreur

---

### `Mimir.Viz.initialize()`

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

### `Mimir.Viz.add_image(pixels, width, height, channels)`

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
Mimir.Viz.add_image(pixels, 64, 64, 3)
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

### `Mimir.Viz.save_loss_history(filepath)`

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
Mimir.Memory.set_limit(8000) -- 8 GB
Mimir.Allocator.configure({max_tensors = 1000, enable_compression = true})

-- Tokenizer
Mimir.Tokenizer.create(50000)
local corpus = io.open("data/corpus.txt"):read("*all")
Mimir.Tokenizer.ensure_vocab_from_text(corpus)
Mimir.Tokenizer.save("Mimir.Tokenizer.json")

-- Dataset
Mimir.Dataset.load("data/train")
Mimir.Dataset.prepare_sequences(512)

-- Modèle
local cfg, err = Mimir.Architectures.default_config("transformer")
if not cfg then error(err) end
cfg.vocab_size = Mimir.Tokenizer.vocab_size()
cfg.d_model = 512
cfg.num_layers = 6
cfg.num_heads = 8
cfg.seq_len = 512

assert(Mimir.Model.create("transformer", cfg))
local ok, params = Mimir.Model.build()
print(string.format("Modèle construit: %d paramètres", params))
Mimir.Model.init_weights("xavier", 42)

-- Monitoring
Mimir.Htop.create()
Mimir.Htop.enable(true)

-- Entraînement
Mimir.Model.train(10, 3e-4)

-- Sauvegarde
Mimir.Serialization.save("checkpoints/final.safetensors", "safetensors")
Mimir.Tokenizer.save("checkpoints/Mimir.Tokenizer.json")
```

---

### Exemple 2: Génération avec Modèle Chargé

```lua
-- Charger tokenizer et modèle
Mimir.Tokenizer.load("checkpoints/Mimir.Tokenizer.json")
Mimir.Serialization.load("checkpoints/final.safetensors")

-- Inférence
local prompt = "Once upon a time"
local output = Mimir.Model.infer(prompt)
print("Génération:", output)
```

---

### Exemple 3: Monitoring Avancé avec Visualisation

```lua
-- Configuration
Mimir.Memory.set_limit(12000) -- 12 GB
Mimir.Htop.create({enable_viz = false})
Mimir.Htop.enable(true)

-- Visualiseur
Mimir.Viz.create("Training Monitor", 1920, 1080)
Mimir.Viz.initialize()

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
      memory_mb = Mimir.Memory.get_usage()
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
Mimir.Viz.save_loss_history("loss_history.json")
```

---

### Exemple 4: VAE avec Visualisation Images

```lua
-- Créer VAE
local cfg, err = Mimir.Architectures.default_config("vae")
if not cfg then error(err) end
cfg.input_dim = 784  -- 28x28 MNIST
cfg.latent_dim = 20
cfg.encoder_hidden = 400
cfg.decoder_hidden = 400

assert(Mimir.Model.create("vae", cfg))
Mimir.Model.build()

-- Visualiseur
Mimir.Viz.create("VAE Training", 800, 600)
Mimir.Viz.initialize()

-- Entraînement...

-- Visualiser une reconstruction
local reconstructed_pixels = {} -- obtenir depuis le modèle
Mimir.Viz.add_image(reconstructed_pixels, 28, 28, 1) -- grayscale
Mimir.Viz.update()
```

---

## Notes de Performance

### Optimisations Mémoire

1. **Limites strictes** : Utilisez `Mimir.Guard.set_limit()` pour éviter les OOM
2. **Compression** : Activez `Mimir.Allocator.configure({enable_compression = true})`
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
