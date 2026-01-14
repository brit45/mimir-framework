---@meta
---@version 2.3.0
---@author <bri45> for "Mímir Framework"
---@date 12 janvier 2026 (dernière sync)
---@diagnostic disable: missing-return, unused-local, unused-vararg, duplicate-doc-field, redundant-parameter

--=============================================================================
-- Mímir Framework v2.3 — IDE Stub (EmmyLua)
--=============================================================================
-- Ce fichier est un "stub" destiné aux IDE (LuaLS / EmmyLua / IntelliJ, etc.).
-- Il documente l'API globale exposée par le binaire `mimir` (bindings C/C++).
--
-- ⚠️  IMPORTANT: Ce fichier est synchronisé avec src/LuaScripting.cpp
--    Toute modification de l'API C++ doit être reflétée ici.
--    Dernière synchronisation: 12 janvier 2026 - 16 modules, 122+ fonctions
--
-- Objectifs :
--  • Autocomplétion IDE, signatures, types, docstrings
--  • Stable : les scripts Lua sont l'API publique
--  • Documentation de référence pour les utilisateurs
--
-- 🆕 Nouveautés v2.3.0 (28 décembre 2025) :
--  • Multi-Input / Branch Support - TensorStore system
--  • model.set_layer_io() - Configuration des entrées/sorties des layers
--  • Operations multi-input complètes (Add, Multiply, Concat, MatMul, Split)
--  • Support residual connections et skip connections
--  • Documentation complète (MULTI_INPUT_SUPPORT.md)
--  • Mode Strict activé (0 pass-through, RUNTIME_CHECK/RUNTIME_ERROR_STRICT)
--
-- Nouveautés v2.1.0 :
--  • Scripts réorganisés en catégories (demos, examples, tests, etc.)
--  • Documentation corrigée (33 fixes)
--  • Synchronisation API validée (114 fonctions)
--
-- Historique v2.0.0 :
--  • MemoryGuard API (limite stricte 10 GB par défaut)
--  • Mimir.Allocator.configure() pour configuration mémoire (OBLIGATOIRE)
--  • Modes train()/eval() pour tous les modèles
--  • API Mimir.Htop et Mimir.Viz pour monitoring temps réel
--
-- Remarques :
--  • Les fonctions retournent souvent (ok:boolean, value) ou (ok:boolean, err:string)
--  • ⚠️  Toujours appeler Mimir.Allocator.configure() au début des scripts!
--  • Les tables de stats sont des structs (tables Lua avec champs nommés)
--  • Les "Mimir.Layers.*" sont des opérations bas niveau (préférer Mimir.Model.forward())
--  • Mimir.MemoryGuard: API moderne recommandée (Mimir.Guard: API legacy compatible)
--=============================================================================

--=============================================================================
-- Namespace Mimir
--=============================================================================

---@class Mimir
Mimir = {}

---@class Mimir
---@field Model MimirModelAPI
---@field Architectures MimirArchitecturesAPI
---@field Layers MimirLayersAPI
---@field Tokenizer MimirTokenizerAPI
---@field Dataset MimirDatasetAPI
---@field Memory MimirMemoryAPI
---@field Guard MimirGuardAPI
---@field MemoryGuard MimirMemoryGuardAPI
---@field Allocator MimirAllocatorAPI
---@field Htop MimirHtopAPI
---@field Viz MimirVizAPI
---@field Serialization MimirSerializationAPI

--=============================================================================
-- Aliases / Types de base
--=============================================================================

---@alias int integer
---@alias float number
---@alias bool boolean
---@alias bytes integer

---@alias TokenId integer
---@alias TokenIds TokenId[]     -- liste d'IDs token (indices Lua 1..N)
---@alias TokenText string

---@alias ModelType
---| "ponyxl_ddpm"
---| "basic_mlp"
---| "transformer"
---| "vit"
---| "vae"
---| "resnet"
---| "unet"
---| "mobilenet"
---| "vgg16"
---| "vgg19"
---| "diffusion"

---@alias ActivationType
---| "relu"
---| "gelu"
---| "silu"
---| "swish"
---| "tanh"
---| "sigmoid"
---| "softmax"

---@alias WeightInit
---| "xavier"
---| "he"
---| "normal"
---| "uniform"
---| "zeros"

---@alias HardwareAccelMode boolean

---@alias KeywordList string[]

--=============================================================================
-- Configs de modèles
--=============================================================================

---@class ModelConfig
---@field type? ModelType @Injecté côté C++ lors de `Model.create()` (metadata)
---@field dropout? float @Dropout générique (si supporté par l'architecture)
---@field optimizer? string @"sgd"|"adam"|"adamw" (utilisé par `Model.train()`)
---@field beta1? float
---@field beta2? float
---@field epsilon? float
---@field weight_decay? float
---@field min_lr? float
---@field decay_rate? float
---@field decay_steps? int
---@field warmup_steps? int
---@field decay_strategy? string @"none"|"cosine"|"step"|"exponential"|"linear"

---@class BasicMLPConfig: ModelConfig
---@field input_dim? int
---@field hidden_dim? int
---@field output_dim? int
---@field hidden_layers? int

---@class TransformerConfig: ModelConfig
---@field seq_len? int
---@field d_model? int
---@field vocab_size? int
---@field padding_idx? int
---@field num_layers? int
---@field num_heads? int
---@field mlp_hidden? int
---@field output_dim? int
---@field causal? bool

---@class ViTConfig: ModelConfig
---@field num_tokens? int @Nombre de tokens/patches (entrée = num_tokens*d_model)
---@field d_model? int
---@field num_layers? int
---@field num_heads? int
---@field mlp_hidden? int
---@field output_dim? int
---@field causal? bool

---@class VAEConfig: ModelConfig
---@field image_w? int
---@field image_h? int
---@field image_c? int
---@field latent_dim? int
---@field hidden_dim? int

---@class ResNetConfig: ModelConfig
---@field image_w? int
---@field image_h? int
---@field image_c? int
---@field base_channels? int
---@field num_classes? int
---@field blocks1? int
---@field blocks2? int
---@field blocks3? int
---@field blocks4? int

---@class UNetConfig: ModelConfig
---@field image_w? int
---@field image_h? int
---@field image_c? int
---@field base_channels? int
---@field depth? int

---@class MobileNetConfig: ModelConfig
---@field image_w? int
---@field image_h? int
---@field image_c? int
---@field base_channels? int
---@field num_classes? int

---@class VGG16Config: ModelConfig
---@field image_w? int
---@field image_h? int
---@field image_c? int
---@field base_channels? int
---@field num_classes? int
---@field fc_hidden? int

---@class VGG19Config: ModelConfig
---@field image_w? int
---@field image_h? int
---@field image_c? int
---@field base_channels? int
---@field num_classes? int
---@field fc_hidden? int

---@class DiffusionConfig: ModelConfig
---@field image_w? int
---@field image_h? int
---@field image_c? int
---@field time_dim? int
---@field hidden_dim? int

---@class PonyXLDDPMConfig: ModelConfig
---@field d_model? int
---@field seq_len? int
---@field max_vocab? int
---@field image_w? int
---@field image_h? int
---@field image_c? int
---@field hidden_dim? int
---@field latent_dim? int
---@field blur_levels? int
---@field cfg_dropout_prob? float
---@field cond_image_dropout_prob? float
---@field cond_image_dropout_lr_scale? float
---@field cond_image_dropout_noise_std? float

-- NOTE: les schémas ci-dessus reflètent le registre C++ (ModelArchitectures::defaultConfig).
-- Pour obtenir la config exacte à jour côté runtime:
--   local cfg = Mimir.Architectures.default_config("transformer")
--   local ok = Mimir.Model.create("transformer", cfg)

--=============================================================================
-- Stats / Structs
--=============================================================================

---@class HardwareCaps
---@field avx2 bool @Support AVX2 SIMD instructions
---@field fma bool @Support FMA (Fused Multiply-Add)
---@field f16c bool @Support Half-precision floats
---@field bmi2 bool @Support BMI2 instructions

---@class MemoryStats
---@field current_mb float
---@field peak_mb float
---@field usage_percent float

---@class GuardStats
---@field current_mb float
---@field peak_mb float
---@field limit_mb float
---@field usage_percent float

---@class AllocatorStats
---@field tensor_count int @Nombre de tensors alloués
---@field loaded_count int @Nombre de tensors chargés en RAM

---@class HtopMetrics
---@field epoch int
---@field total_epochs int
---@field batch int
---@field total_batches int
---@field loss float
---@field avg_loss float
---@field lr float
---@field batch_time_ms int
---@field memory_mb int
---@field memory_freed int
---@field bps float
---@field params int
---@field timestep float
---@field kl float
---@field wass float
---@field ent float
---@field mom float
---@field spat float
---@field temp float
---@field mse float
---@field grad_norm? float
---@field grad_max? float
---@field opt_type? int
---@field opt_step? int
---@field opt_beta1? float
---@field opt_beta2? float
---@field opt_eps? float
---@field opt_weight_decay? float

---@class VizMetrics
---@field epoch? int
---@field total_epochs? int
---@field loss? float
---@field avg_loss? float
---@field lr? float
---@field memory_mb? float
---@field tokens_per_sec? float
---@field custom? table<string, number>

--=============================================================================
-- Module: Mimir.Model
--=============================================================================

---@class MimirModelAPI
Mimir.Model = {}

---Créer un modèle via le registre d'architectures (C++).
---Le modèle est construit via le registre C++ immédiatement.
---Note: `Mimir.Model.create()` ne fait plus allocate/init automatiquement.
---Utilisez ensuite `Mimir.Model.allocate_params()` et `Mimir.Model.init_weights()` si nécessaire.
---@param model_type ModelType
---@param config? ModelConfig|BasicMLPConfig|TransformerConfig|ViTConfig|VAEConfig|ResNetConfig|UNetConfig|MobileNetConfig|VGG16Config|VGG19Config|DiffusionConfig|PonyXLDDPMConfig|table
---@return boolean ok
---@return string? err
function Mimir.Model.create(model_type, config) end

---[COMPAT] Reconstruit le modèle courant via le registre.
---Préférez `Mimir.Model.create(type, cfg)`.
---Retour: ok + nombre de paramètres (scalars).
---@return boolean ok
---@return integer? params
---@return string? err
function Mimir.Model.build() end

---Entraîner le modèle courant.
---Nécessite un dataset chargé et des séquences préparées via `Mimir.Dataset.prepare_sequences(max_seq_len)`.
---@param epochs integer @Nombre d'epochs
---@param learning_rate number @LR (ex: 3e-4)
---@return boolean ok
---@return string? err
function Mimir.Model.train(epochs, learning_rate) end

---Inférence sur un prompt (string) ou une séquence de tokens.
---Retour: string (texte) ou nil si échec.
---@param input string|TokenIds
---@return string|nil output
function Mimir.Model.infer(input) end

---[DÉPRÉCIÉ] Sauvegarder le modèle (ancienne API).
---⚠️  Utilisez Mimir.Serialization.save() pour la nouvelle API v2.3
---@param dir string @Ex: "checkpoints/run1/epoch_10"
---@return boolean ok
---@return string? err
---@deprecated Utilisez Mimir.Serialization.save()
function Mimir.Model.save(dir) end

---[DÉPRÉCIÉ] Charger un modèle depuis un répertoire (ancienne API).
---⚠️  Utilisez Mimir.Serialization.load() pour la nouvelle API v2.3
---@param dir string
---@return boolean ok
---@return string? err
---@deprecated Utilisez Mimir.Serialization.load()
function Mimir.Model.load(dir) end

---Allouer explicitement les paramètres (si supporté).
---@return boolean ok
---@return integer? params @Nombre total de paramètres alloués (si ok)
---@return string? err
function Mimir.Model.allocate_params() end

---Initialiser les poids du modèle (si supporté).
---@param init? WeightInit
---@param seed? integer
---@return boolean ok
---@return string? err
function Mimir.Model.init_weights(init, seed) end

---Nombre total de paramètres (si disponible).
---@return integer params
function Mimir.Model.total_params() end

---Pousser une couche (API bas niveau / description) dans le modèle.
---Note: cette fonction est surtout utilisée par les builders d'architectures.
---@param name string
---@param layer_type string
---@param params_count integer
---@return boolean ok
---@return string? err
function Mimir.Model.push_layer(name, layer_type, params_count) end

---Configure les entrées et sortie d'un layer pour le support multi-input/branch.
---Permet de créer des architectures avec skip connections, concat, split, etc.
---@param layer_name string @Nom du layer à configurer
---@param inputs table @Table des noms de tensors en entrée, ex: {"x", "skip"}
---@param output? string @Nom du tensor de sortie (optionnel, défaut: "x")
---@return boolean ok @true si succès, false si layer non trouvé
---@return string? err @Message d'erreur éventuel
---
---Exemples:
---```lua
--- -- Residual connection (skip connection)
--- Mimir.Model.set_layer_io("conv1", {"x"}, "skip")
--- Mimir.Model.set_layer_io("add", {"x", "skip"}, "x")
---
--- -- Concat multiple branches
--- Mimir.Model.set_layer_io("concat", {"branch1", "branch2", "branch3"}, "fused")
---
--- -- MatMul (A × B)
--- Mimir.Model.set_layer_io("matmul", {"A", "B"}, "result")
---
--- -- Split (1 input → N outputs nommés output_0, output_1, ...)
--- Mimir.Model.set_layer_io("split", {"x"}, "branches")
---```
function Mimir.Model.set_layer_io(layer_name, inputs, output) end

---Forward pass (si exposé par l'implémentation).
---Mode training activé par défaut pour permettre le backward pass.
---@param input TokenIds|float[] @Données d'entrée (table d'entiers -> chemin token ids, table de floats -> chemin float)
---@param training? bool @Mode training (défaut: true) pour calculer les gradients
---@return float[]|nil @Sortie du modèle
---@return string? err
function Mimir.Model.forward(input, training) end

---Backward pass pour calculer les gradients.
---@param loss_gradient float[] @Gradient de la loss
---@return boolean ok
---@return string? err
function Mimir.Model.backward(loss_gradient) end

---Réinitialise tous les gradients à zéro.
---Important: à appeler avant chaque itération d'entraînement.
---@return boolean ok
---@return string? err
function Mimir.Model.zero_grads() end

---Récupère les gradients actuels de tous les paramètres.
---@return float[]|nil @Vecteur de tous les gradients
---@return string? err
function Mimir.Model.get_gradients() end

---Step optimiseur (si exposé). Le LR peut être transmis.
---@param learning_rate number
---@param opt_type? string @"sgd"|"adam"|"adamw" (défaut: "adamw")
---@return boolean ok
---@return string? err
function Mimir.Model.optimizer_step(learning_rate, opt_type) end

---Choisir backend hardware (CPU/OpenCL/Vulkan/Auto) si supporté.
---@param enable HardwareAccelMode @true pour activer l'accélération (si dispo), false pour forcer CPU
---@return boolean ok
---@return string? err
function Mimir.Model.set_hardware(enable) end

---Retourne les capacités détectées (AVX2/FMA/F16C/BMI2).
---@return HardwareCaps caps
function Mimir.Model.hardware_caps() end

--=============================================================================
-- Module: Mimir.Architectures
--=============================================================================

---@class MimirArchitecturesAPI
Mimir.Architectures = {}

---Lister les architectures disponibles (côté C++ registry).
---@return string[]|nil names
---@return string? err
function Mimir.Architectures.available() end

---Retourner la config par défaut d'une architecture.
---@param name string
---@return table|nil config
---@return string? err
function Mimir.Architectures.default_config(name) end

--=============================================================================
-- Module: Mimir.Layers (placeholder / low-level)
--=============================================================================

---@class MimirLayersAPI
Mimir.Layers = {}

---Conv2D low-level (placeholder). Utiliser `Mimir.Model.forward()` à la place.
---@return boolean ok
---@return string err
function Mimir.Layers.conv2d(...) end

---Linear low-level (placeholder). Utiliser `Mimir.Model.forward()` à la place.
---@return boolean ok
---@return string err
function Mimir.Layers.linear(...) end

---MaxPool2D low-level (placeholder).
---@return boolean ok
---@return string err
function Mimir.Layers.maxpool2d(...) end

---AvgPool2D low-level (placeholder).
---@return boolean ok
---@return string err
function Mimir.Layers.avgpool2d(...) end

---Activation low-level (placeholder).
---@return boolean ok
---@return string err
function Mimir.Layers.activation(...) end

---BatchNorm low-level (placeholder).
---@return boolean ok
---@return string err
function Mimir.Layers.batchnorm(...) end

---LayerNorm low-level (placeholder).
---@return boolean ok
---@return string err
function Mimir.Layers.layernorm(...) end

---Attention low-level (placeholder).
---@return boolean ok
---@return string err
function Mimir.Layers.attention(...) end

--=============================================================================
-- Module: Mimir.Tokenizer
--=============================================================================

---@class MimirTokenizerAPI
Mimir.Tokenizer = {}

---Créer un tokenizer avec vocab max.
---@param max_vocab integer
---@return boolean ok
function Mimir.Tokenizer.create(max_vocab) end

---Tokeniser un texte (word/BPE selon impl).
---@param text string
---@return TokenIds tokens
function Mimir.Tokenizer.tokenize(text) end

---Dé-tokeniser une séquence (ids -> texte).
---@param tokens TokenIds
---@return string text
function Mimir.Tokenizer.detokenize(tokens) end

---Retourne la taille actuelle du vocab.
---@return integer size
function Mimir.Tokenizer.vocab_size() end

---Sauvegarder le tokenizer dans un fichier.
---@param path string @Ex: "checkpoints/run1/tokenizer.json"
---@return boolean ok
---@return string? err
function Mimir.Tokenizer.save(path) end

---Charger le tokenizer depuis un fichier.
---@param path string
---@return boolean ok
---@return string? err
function Mimir.Tokenizer.load(path) end

---Ajouter un token au vocab.
---@param token string
---@return boolean ok
---@return string? err
function Mimir.Tokenizer.add_token(token) end

---Étendre le vocab à partir d'un texte (analyse corpus simple).
---@param text string
---@return boolean ok
---@return string? err
function Mimir.Tokenizer.ensure_vocab_from_text(text) end

---Tokenize et étend le vocab si nécessaire.
---@param text string
---@return TokenIds tokens
function Mimir.Tokenizer.tokenize_ensure(text) end

---IDs spéciaux.
---@return integer id
function Mimir.Tokenizer.pad_id() end
---@return integer id
function Mimir.Tokenizer.unk_id() end
---@return integer id
function Mimir.Tokenizer.seq_id() end
---@return integer id
function Mimir.Tokenizer.mod_id() end
---@return integer id
function Mimir.Tokenizer.mag_id() end

---Récupérer un token string depuis un id.
---@param id integer
---@return string token
function Mimir.Tokenizer.get_token_by_id(id) end

---Apprendre un BPE depuis un corpus (chemin ou texte selon impl).
---@param corpus_path string
---@param vocab_target? integer
---@return boolean ok
---@return string? err
function Mimir.Tokenizer.learn_bpe(corpus_path, vocab_target) end

---Tokeniser via BPE (si appris/chargé).
---@param text string
---@return TokenIds tokens
function Mimir.Tokenizer.tokenize_bpe(text) end

---Définir la longueur max des séquences côté tokenizer.
---@param max_length integer
---@return boolean ok
---@return string? err
function Mimir.Tokenizer.set_max_length(max_length) end

---Pad une séquence à une longueur cible.
---@param tokens TokenIds
---@param max_length integer
---@param pad_id? integer
---@return TokenIds padded
function Mimir.Tokenizer.pad_sequence(tokens, max_length, pad_id) end

---Tokeniser une liste de textes.
---@param texts string[]
---@return TokenIds[] batches
function Mimir.Tokenizer.batch_tokenize(texts) end

---Afficher des stats (stdout/log).
function Mimir.Tokenizer.print_stats() end

---Fréquences tokens (si supporté).
---@return table<string, integer> freqs
function Mimir.Tokenizer.get_frequencies() end

---Analyse texte (mots, chars, densité, etc. selon impl).
---Retourne une table avec des statistiques et analyses sémantiques.
---@param text string @Texte à analyser
---@return TextAnalysisResult results @Résultats de l'analyse
---
---**Exemple:**
---```lua
--- local analysis = Mimir.Tokenizer.analyze_text("A majestic dragon flying over mountains")
--- print("Complexité:", analysis.complexity)
--- print("Sujet principal:", analysis.main_subject)
--- if analysis.entities then
---   for _, entity in ipairs(analysis.entities) do
---     print("  Entité:", entity)
---   end
--- end
---```
function Mimir.Tokenizer.analyze_text(text) end

---Extraction keywords (heuristique).
---@param text string
---@param top_k? integer
---@return KeywordList keywords
function Mimir.Tokenizer.extract_keywords(text, top_k) end

--=============================================================================
-- Module: Mimir.Dataset
--=============================================================================

---@class DatasetItem
---@field text_file? string Chemin du fichier texte
---@field image_file? string Chemin du fichier image
---@field audio_file? string Chemin du fichier audio
---@field video_file? string Chemin du fichier vidéo
---@field text? string Contenu texte (si chargé)
---@field width? int Largeur de l'image
---@field height? int Hauteur de l'image
---@field size? int Taille du fichier en bytes

---@class TextAnalysisResult
---@field complexity? float Complexité du texte (métrique interne)
---@field main_subject? string Sujet principal détecté
---@field entities? string[] Entités nommées extraites
---@field actions? string[] Actions/verbes principaux
---@field modifiers? string[] Modificateurs/adjectifs importants
---@field context? string Contexte général du texte
---@field resolution? int Résolution estimée pour génération d'image

---@class MimirDatasetAPI
Mimir.Dataset = {}

---Charger un dataset depuis un répertoire.
---Attendu: répertoire contenant des fichiers textes / structure interne.
---@param dir string
---@param target_w? integer Largeur image cible (si images)
---@param target_h? integer Hauteur image cible (si images)
---@param min_modalities? integer Seuil de modalités pour valider les linkables (ex: 2 => texte+image)
---@return boolean ok
---@return integer|string? count_or_err Nombre d'items si ok, sinon message d'erreur
function Mimir.Dataset.load(dir, target_w, target_h, min_modalities) end

---Récupérer un item du dataset par son index (1-based).
---Retourne une table avec les chemins et métadonnées de l'item.
---@param index integer Index de l'item (commence à 1)
---@return DatasetItem|nil item Item du dataset
---@return string? err Message d'erreur si échec
function Mimir.Dataset.get(index) end

---Préparer les séquences (stockées dans le contexte interne).
---La séquence length est utilisée ensuite par Mimir.Model.train().
---@param max_length integer
---@return boolean ok
---@return string? err
function Mimir.Dataset.prepare_sequences(max_length) end

--=============================================================================
-- Module: Mimir.Memory (AdvancedRAMManager)
--=============================================================================

---@class MimirMemoryAPI
Mimir.Memory = {}

---Configurer le gestionnaire RAM avancé (implémentation dépendante).
---@param cfg table
---@return boolean ok
---@return string? err
function Mimir.Memory.config(cfg) end

---Récupérer stats RAM.
---@return MemoryStats stats
function Mimir.Memory.get_stats() end

---Imprimer stats RAM.
function Mimir.Memory.print_stats() end

---Purger / clear caches mémoire.
---@return boolean ok
---@return string? err
function Mimir.Memory.clear() end

---Usage actuel en MB (alias utilitaire).
---@return number mb
function Mimir.Memory.get_usage() end

---Définir une limite RAM (en MB).
---@param limit_mb number
---@return boolean ok
---@return string? err
function Mimir.Memory.set_limit(limit_mb) end

--=============================================================================
-- Module: Mimir.Guard (MemoryGuard) - API Ancienne
--=============================================================================

---@class MimirGuardAPI
Mimir.Guard = {}

---Définir la limite de mémoire (MB).
---@param limit_mb number
---@return boolean ok
---@return string? err
function Mimir.Guard.set_limit(limit_mb) end

---Stats de la limite stricte.
---@return GuardStats stats
function Mimir.Guard.get_stats() end

---Imprimer stats guard.
function Mimir.Guard.print_stats() end

---Reset stats (peak, compteurs).
function Mimir.Guard.reset() end

--=============================================================================
-- Module: Mimir.MemoryGuard (API Moderne - Recommandée)
--=============================================================================

---@class MemoryGuardStats
---@field current_mb float @Utilisation RAM courante en MB
---@field peak_mb float @Pic d'utilisation en MB
---@field limit_mb float @Limite configurée en MB
---@field usage_percent float @Pourcentage d'utilisation

---@class MimirMemoryGuardAPI
Mimir.MemoryGuard = {}

---Définir la limite de mémoire RAM stricte.
---Accepte des valeurs en bytes (grands nombres) ou en GB (si <= 1000).
---@param limit number @Limite en bytes ou en GB (si valeur <= 1000)
---@return boolean ok @true si succès
---
---**Exemples:**
---```lua
--- -- Définir limite à 10 Go
--- Mimir.MemoryGuard.setLimit(10 * 1024 * 1024 * 1024)  -- en bytes
--- Mimir.MemoryGuard.setLimit(10)  -- en GB (auto-détecté car < 1000)
---```
function Mimir.MemoryGuard.setLimit(limit) end

---Obtenir la limite de mémoire configurée.
---@return integer bytes @Limite en bytes
---
---**Exemple:**
---```lua
--- local limit = Mimir.MemoryGuard.getLimit()
--- print(string.format("Limite: %.2f GB", limit / 1024 / 1024 / 1024))
---```
function Mimir.MemoryGuard.getLimit() end

---Obtenir l'utilisation RAM courante.
---@return integer bytes @RAM utilisée actuellement en bytes
---
---**Exemple:**
---```lua
--- local current = Mimir.MemoryGuard.getCurrentUsage()
--- local limit = Mimir.MemoryGuard.getLimit()
--- local percent = (current / limit) * 100
--- print(string.format("RAM: %.2f%%", percent))
---```
function Mimir.MemoryGuard.getCurrentUsage() end

---Obtenir le pic d'utilisation RAM.
---@return integer bytes @Pic d'utilisation en bytes depuis le démarrage
---
---**Exemple:**
---```lua
--- local peak = Mimir.MemoryGuard.getPeakUsage()
--- print(string.format("Pic RAM: %.2f GB", peak / 1024 / 1024 / 1024))
---```
function Mimir.MemoryGuard.getPeakUsage() end

---Récupérer toutes les statistiques en une seule fois.
---@return MemoryGuardStats stats @Structure contenant toutes les stats
---
---**Exemple:**
---```lua
--- local stats = Mimir.MemoryGuard.getStats()
--- print("RAM courante: " .. stats.current_mb .. " MB")
--- print("Pic: " .. stats.peak_mb .. " MB")
--- print("Limite: " .. stats.limit_mb .. " MB")
--- print("Utilisation: " .. stats.usage_percent .. "%")
---```
function Mimir.MemoryGuard.getStats() end

---Afficher les statistiques formatées dans la console.
---Affiche un tableau détaillé avec toutes les métriques.
---
---**Format de sortie:**
---```
---╔═══════════════════════════════════════════════════════╗
---║           MEMORY GUARD - STATISTIQUES                ║
---╠═══════════════════════════════════════════════════════╣
---║ Limite:          10240 MB                         ║
---║ Actuel:           2456 MB                         ║
---║ Pic:              3892 MB                         ║
---║ Utilisation:     38.0 %                          ║
---║ Allocations:      1523                            ║
---║ Libérations:       892                            ║
---╠═══════════════════════════════════════════════════════╣
---║ État:       🔓 ACTIF                             ║
---╚═══════════════════════════════════════════════════════╝
---```
---
---**Exemple:**
---```lua
--- Mimir.MemoryGuard.setLimit(10 * 1024 * 1024 * 1024)
--- -- ... opérations ...
--- Mimir.MemoryGuard.printStats()  -- Affiche le rapport complet
---```
function Mimir.MemoryGuard.printStats() end

---Réinitialiser les compteurs de statistiques.
---Remet à zéro le pic et les compteurs d'allocations/libérations.
---La limite configurée est préservée.
---
---**Exemple:**
---```lua
--- Mimir.MemoryGuard.reset()
--- print("✓ Statistiques réinitialisées")
---```
function Mimir.MemoryGuard.reset() end

--=============================================================================
-- Module: Mimir.Allocator (DynamicTensorAllocator)
--=============================================================================

---@class AllocatorConfig
---@field max_tensors? integer @Nombre max de tenseurs en mémoire
---@field offload_threshold_mb? float @Seuil de RAM pour offload
---@field swap_strategy? string @Stratégie de swap (lru, fifo, etc.)
---@field max_ram_gb? number @Limite RAM globale
---@field enable_compression? boolean @Activer compression LZ4

---@class MimirAllocatorAPI
Mimir.Allocator = {}

---Configurer l'allocator dynamique (tenseurs, offload, compression).
---@param cfg AllocatorConfig|table
---@return boolean ok
---@return string? err
function Mimir.Allocator.configure(cfg) end

---Imprimer stats allocator (stdout/log).
function Mimir.Allocator.print_stats() end

---Récupérer stats allocator.
---@return AllocatorStats stats
function Mimir.Allocator.get_stats() end

--=============================================================================
-- Module: Mimir.Htop (AsyncMonitor / Terminal UI)
--=============================================================================

---@class MimirHtopAPI
Mimir.Htop = {} --avec configuration optionnelle.
---@param config? table @Configuration: {enable_viz: bool, refresh_rate: int, ...}
---@return boolean ok
---@return string? err
function Mimir.Htop.create(config) end

---Activer/désactiver l'affichage htop.
---@param enabled boolean
---@return boolean ok
---@return string? err
function Mimir.Htop.enable(enabled) end

---Mettre à jour les métriques affichées dans htop.
---Les paramètres peuvent être passés via une table HtopMetrics ou individuellement.
---@param metrics? HtopMetrics|table @Structure de métriques (optionnel)
---@param ... any @Paramètres legacy (optionnels)
---@return boolean ok
---@return string? err
function Mimir.Htop.update(metrics, ...) end

---Forcer un render (si supporté).
---@return boolean ok
---@return string? err
function Mimir.Htop.render() end

---Clear écran / reset UI.
---@return boolean ok
---@return string? err
function Mimir.Htop.clear() end

--=============================================================================
-- Module: Mimir.Viz (SFML Visualizer)
--=============================================================================

---@class MimirVizAPI
Mimir.Viz = {}

---Créer la fenêtre visualiseur SFML avec titre et dimensions optionnels.
---@param title? string @Titre de la fenêtre (défaut: "Mimir Visualizer")
---@param width? integer @Largeur en pixels (défaut: 1280)
---@param height? integer @Hauteur en pixels (défaut: 720)
---@return boolean ok
---@return string? err
function Mimir.Viz.create(title, width, height) end

---Initialiser le visualiseur (ouvre la fenêtre SFML).
---@return boolean ok
---@return string? err
function Mimir.Viz.initialize() end

---Vérifier si la fenêtre est ouverte.
---@return boolean open
function Mimir.Viz.is_open() end

---Traiter les événements fenêtre (fermeture, clavier, souris).
function Mimir.Viz.process_events() end

---Mettre à jour et afficher le rendu de la fenêtre.
function Mimir.Viz.update() end

---Ajouter/afficher une image dans le visualiseur.
---Les pixels sont fournis sous forme de tableau plat (row-major).
---@param pixels number[] @Tableau de valeurs pixel [0-255], RGBA ou RGB
---@param width integer @Largeur de l'image
---@param height integer @Hauteur de l'image
---@param channels integer @Nombre de canaux (3=RGB, 4=RGBA)
---@return boolean ok
---@return string? err
function Mimir.Viz.add_image(pixels, width, height, channels) end

---Mettre à jour les métriques d'entraînement affichées.
---@param metrics VizMetrics|table @Métriques: epoch, loss, lr, memory_mb, etc.
---@return boolean ok
---@return string? err
function Mimir.Viz.update_metrics(metrics) end

---Ajouter un point à l'historique de loss (pour graphe).
---@param loss number @Valeur de loss
function Mimir.Viz.add_loss_point(loss) end

---Clear viz.
---@return boolean ok
---@return string? err
function Mimir.Viz.clear() end

---Activer/désactiver la viz (NO-OP si non compilée).
---@param enabled boolean
---@return boolean ok
---@return string? err
function Mimir.Viz.set_enabled(enabled) end

---Sauvegarder l'historique de loss.
---@param path string
---@return boolean ok
---@return string? err
function Mimir.Viz.save_loss_history(path) end

--=============================================================================
-- Mimir.Serialization API (v2.3.0)
--=============================================================================
-- Nouveau système de sérialisation avec 3 formats :
--  • SafeTensors : Format production compatible HuggingFace
--  • RawFolder   : Format debug avec checksums SHA256
--  • DebugJson   : Format inspection avec statistiques

---@class MimirSerializationAPI
Mimir.Serialization = {}

---@alias SerializationFormat
---| "safetensors"  # Format production (défaut)
---| "raw_folder"    # Format debug avec checksums
---| "debug_json"    # Format inspection avec stats

---@class SaveOptions
---@field format? SerializationFormat @Format de sauvegarde (défaut: SAFETENSORS)
---@field save_tokenizer? boolean @Sauvegarder le tokenizer (défaut: true)
---@field save_encoder? boolean @Sauvegarder l'encoder (défaut: true)
---@field save_optimizer? boolean @Sauvegarder l'état optimizer (défaut: false)
---@field debug_max_values? integer @Nombre max de valeurs debug (legacy, défaut: 100)
---@field include_git_info? boolean @Inclure info git (défaut: true)
---@field include_gradients? boolean @[DebugJson v1.1] Inclure gradients (défaut: false)
---@field include_optimizer_state? boolean @[DebugJson v1.1] Inclure optimizer state (défaut: false)
---@field max_values_per_tensor? integer @[DebugJson v1.1] Nb valeurs par tensor (défaut: 20)
---@field include_activations? boolean @[DebugJson v1.1] Inclure activations (défaut: false)
---@field include_checksums? boolean @[DebugJson v1.1] Inclure checksums (défaut: false)
---@field include_weight_deltas? boolean @[DebugJson v1.1] Inclure weight deltas (défaut: false)

---@class LoadOptions
---@field load_tokenizer? boolean @Charger le tokenizer (défaut: true)
---@field load_encoder? boolean @Charger l'encoder (défaut: true)
---@field load_optimizer? boolean @Charger l'état optimizer (défaut: false)
---@field verify_checksums? boolean @Vérifier checksums SHA256 (défaut: true, RawFolder uniquement)

---Sauvegarder un checkpoint avec le nouveau système de sérialisation v2.3.
---
---Formats supportés :
---  • SAFETENSORS : Format production compatible HuggingFace/PyTorch
---  • RAWFOLDER   : Format debug avec checksums SHA256
---  • DEBUGJSON   : Format inspection avec statistiques
---
---@param path string @Chemin du fichier/dossier (ex: "model.safetensors", "checkpoint/")
---@param format? SerializationFormat @Format (défaut: "SAFETENSORS")
---@param options? SaveOptions @Options de sauvegarde
---@return boolean ok @true si succès
---@return string? err @Message d'erreur si échec
---
---Exemples :
---```lua
--- -- SafeTensors (production)
--- Mimir.Serialization.save("model.safetensors")
--- 
--- -- RawFolder (debug)
--- Mimir.Serialization.save("checkpoint/", "raw_folder")
--- 
--- -- DebugJson (inspection)
--- Mimir.Serialization.save("debug.json", "debug_json", {debug_max_values = 20})
--- 
--- -- DebugJson Enhanced v1.1 (diagnostic entraînement)
--- Mimir.Serialization.save("debug.json", "debug_json", {
---     include_gradients = true,
---     include_optimizer_state = true,
---     include_weight_deltas = true,
---     include_checksums = true,
---     max_values_per_tensor = 20
--- })
--- 
--- -- Avec options
--- Mimir.Serialization.save("model.safetensors", "safetensors", {
---     save_optimizer = true,
---     include_git_info = true
--- })
---```
function Mimir.Serialization.save(path, format, options) end

---Charger un checkpoint avec détection automatique du format.
---
---La fonction détecte automatiquement le format :
---  • Fichier .safetensors → SafeTensors
---  • Dossier avec manifest.json → RawFolder
---
---@param path string @Chemin du fichier/dossier
---@param format? SerializationFormat @Format (optionnel, auto-détecté)
---@param options? LoadOptions @Options de chargement
---@return boolean ok @true si succès
---@return string? err @Message d'erreur si échec
---
---Exemples :
---```lua
--- -- Auto-détection du format
--- Mimir.Serialization.load("model.safetensors")
--- Mimir.Serialization.load("checkpoint/")
--- 
--- -- Format explicite
--- Mimir.Serialization.load("model.safetensors", "safetensors")
--- 
--- -- Avec vérification checksums
--- Mimir.Serialization.load("checkpoint/", "raw_folder", {
---     verify_checksums = true
--- })
---```
function Mimir.Serialization.load(path, format, options) end

---Détecter automatiquement le format d'un checkpoint.
---@param path string @Chemin du fichier/dossier
---@return SerializationFormat? format @Format détecté ou nil si inconnu
---@return string? err @Message d'erreur si échec
function Mimir.Serialization.detect_format(path) end

---Sauvegarder un checkpoint avec Enhanced Debug JSON v1.1.0.
---Fonction spécialisée pour le diagnostic d'entraînement avec toutes les options v1.1.
---@param path string @Chemin du fichier JSON
---@param options? SaveOptions @Options Enhanced Debug JSON v1.1
---@return boolean ok @true si succès
---@return string? err @Message d'erreur si échec
function Mimir.Serialization.save_enhanced_debug(path, options) end

--=============================================================================
-- Fonctions globales utilitaires
--=============================================================================

---Logger (côté C++). Equivalent console + buffer.
---@param msg string
function log(msg) end

---Lire un JSON depuis un fichier.
---@param path string
---@return table|nil obj
---@return string? err
function read_json(path) end

---Écrire un JSON dans un fichier.
---@param path string
---@param obj table
---@return boolean ok
---@return string? err
function write_json(path, obj) end

---Print (redirigé) — peut être surchargé par le runtime.
---@param ... any
function print(...) end

--=============================================================================
-- Pipeline API (optionnel)
--=============================================================================
-- Si vous chargez pipeline_api.lua dans votre script, l'IDE bénéficiera aussi
-- de ces signatures. Ces fonctions sont en Lua pur, mais on les déclare ici
-- pour l'autocomplétion (sans dépendre du require dans l'IDE).

---@class PipelineConfig
---@field dataset_dir string
---@field out_dir string
---@field model_type ModelType
---@field model_config table
---@field tokenizer_vocab int
---@field max_seq_len int
---@field epochs int
---@field lr float
---@field min_lr? float
---@field warmup_epochs? int
---@field save_every? int
---@field gen_every? int
---@field prompt? string
---@field max_ram_gb? float
---@field enable_compression? bool
---@field enable_htop? bool
---@field enable_viz? bool


--=============================================================================
-- Module: Pipeline
--=============================================================================

---@class Pipeline
Pipeline = {}

---@param cfg PipelineConfig|table
---@return boolean ok
---@return string? err
function Pipeline.setup(cfg) end

---Exécuter un pipeline training complet (si présent).
---@return boolean ok
---@return string? err
function Pipeline.run() end

---Sauvegarder un checkpoint pipeline (si présent).
---@param tag string
---@return boolean ok
---@return string? err
function Pipeline.save(tag) end

---Tenter une reprise depuis un checkpoint (si présent).
---@param dir string
---@return boolean ok
---@return string? err
function Pipeline.resume(dir) end

--=============================================================================
-- Exports globaux (pour l'IDE)
--=============================================================================
---@type MimirModelAPI
Mimir.Model = Mimir.Model
---@type MimirArchitecturesAPI
Mimir.Architectures = Mimir.Architectures
---@type MimirLayersAPI
Mimir.Layers = Mimir.Layers
---@type MimirTokenizerAPI
Mimir.Tokenizer = Mimir.Tokenizer
---@type MimirDatasetAPI
Mimir.Dataset = Mimir.Dataset
---@type MimirMemoryAPI
Mimir.Memory = Mimir.Memory
---@type MimirGuardAPI
Mimir.Guard = Mimir.Guard
---@type MimirMemoryGuardAPI
Mimir.MemoryGuard = Mimir.MemoryGuard
---@type MimirAllocatorAPI
Mimir.Allocator = Mimir.Allocator
---@type MimirHtopAPI
Mimir.Htop = Mimir.Htop
---@type MimirVizAPI
Mimir.Viz = Mimir.Viz
