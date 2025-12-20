---@meta
---@version 2.0.0
---@author <bri45> for "Mímir Framework"
---@diagnostic disable: missing-return, unused-local, unused-vararg, duplicate-doc-field, redundant-parameter

--=============================================================================
-- Mímir Framework — IDE Stub (EmmyLua)
--=============================================================================
-- Ce fichier est un "stub" destiné aux IDE (LuaLS / EmmyLua / IntelliJ, etc.).
-- Il documente l'API globale exposée par le binaire `mimir` (bindings C/C++).
--
-- Objectifs :
--  • Autocomplétion, signatures, types, docstrings
--  • Stable : les scripts Lua sont l'API publique ; l'implémentation C/C++ peut évoluer
--
-- Remarques :
--  • Les fonctions retournent souvent (ok:boolean) ou (ok:boolean, err:string)
--  • Les tables de stats sont des "structs" (tables Lua avec champs nommés)
--  • Les "layers.*" sont actuellement des placeholders (utiliser model.forward())
--=============================================================================

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
---| "encoder"
---| "decoder"
---| "transformer"
---| "unet"
---| "vae"
---| "vit"
---| "gan"
---| "generator"
---| "discriminator"
---| "diffusion"
---| "resnet"
---| "mobilenet"

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

---@alias HardwareBackend
---| "cpu"
---| "opencl"
---| "vulkan"
---| "auto"

---@alias KeywordList string[]

--=============================================================================
-- Configs de modèles
--=============================================================================

---@class ModelConfig
---@field vocab_size? int         @Taille vocabulaire (encoder/decoder/transformer)
---@field embed_dim? int          @Dimension embeddings
---@field num_layers? int         @Nombre de couches transformer
---@field num_heads? int          @Nombre de têtes attention
---@field d_ff? int               @Dimension FFN
---@field max_seq_len? int        @Longueur max de séquence
---@field dropout? float          @Dropout (0..1)
---@field input_dim? int          @Dimension entrée (certains modèles)
---@field num_classes? int        @Nombre de classes (classif)
---@field image_channels? int     @Canaux image (CV)
---@field image_size? int         @Résolution image (CV)

---@class UNetConfig
---@field input_channels? int
---@field output_channels? int
---@field base_channels? int
---@field num_levels? int
---@field blocks_per_level? int
---@field use_attention? bool
---@field use_residual? bool
---@field dropout? float

---@class VAEConfig
---@field input_dim? int
---@field latent_dim? int
---@field encoder_hidden? int
---@field decoder_hidden? int

---@class ViTConfig
---@field image_size? int
---@field patch_size? int
---@field embed_dim? int
---@field num_layers? int
---@field num_heads? int
---@field mlp_ratio? float
---@field num_classes? int

---@class GANConfig
---@field latent_dim? int
---@field image_channels? int
---@field resolution? int
---@field gen_channels? int
---@field disc_channels? int

---@class DiffusionConfig
---@field image_channels? int
---@field resolution? int
---@field model_channels? int
---@field num_res_blocks? int
---@field use_attention? bool
---@field dropout? float
---@field use_bottleneck? bool

---@class TransformerConfig
---@field vocab_size? int
---@field embed_dim? int
---@field num_layers? int
---@field num_heads? int
---@field d_ff? int
---@field max_seq_len? int
---@field dropout? float

---@class ResNetConfig
---@field num_classes? int
---@field input_channels? int

---@class MobileNetConfig
---@field num_classes? int
---@field width_mult? float

--=============================================================================
-- Stats / Structs
--=============================================================================

---@class HardwareCaps
---@field avx2 bool
---@field fma bool
---@field f16c bool
---@field bmi2 bool

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
---@field tensor_count int
---@field loaded_count int

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
-- Module: model
--=============================================================================

---@class MimirModelAPI
local model = {}

---Créer un modèle (définit le type + stocke une config pour build()).
---Le modèle réel est construit par `model.build()`.
---@param model_type ModelType
---@param config? ModelConfig|UNetConfig|VAEConfig|ViTConfig|GANConfig|DiffusionConfig|TransformerConfig|ResNetConfig|MobileNetConfig|table
---@return boolean ok
---@return string? err
function model.create(model_type, config) end

---Construire le modèle courant (à partir de model_type + config passés à create()).
---Retour: ok + nombre de paramètres (scalars).
---@return boolean ok
---@return integer? params
---@return string? err
function model.build() end

---Entraîner le modèle courant.
---Nécessite un dataset chargé et des séquences préparées via `dataset.prepare_sequences(max_seq_len)`.
---@param epochs integer @Nombre d'epochs
---@param learning_rate number @LR (ex: 3e-4)
---@return boolean ok
---@return string? err
function model.train(epochs, learning_rate) end

---Inférence sur un prompt (string) ou une séquence de tokens.
---Retour: string (texte) ou nil si échec.
---@param input string|TokenIds
---@return string|nil output
function model.infer(input) end

---Sauvegarder le modèle (checkpoint) dans un répertoire.
---@param dir string @Ex: "checkpoints/run1/epoch_10"
---@return boolean ok
---@return string? err
function model.save(dir) end

---Charger un modèle depuis un répertoire (checkpoint).
---@param dir string
---@return boolean ok
---@return string? err
function model.load(dir) end

---Allouer explicitement les paramètres (si supporté).
---@return boolean ok
---@return string? err
function model.allocate_params() end

---Initialiser les poids du modèle (si supporté).
---@param init? WeightInit
---@param seed? integer
---@return boolean ok
---@return string? err
function model.init_weights(init, seed) end

---Nombre total de paramètres (si disponible).
---@return integer params
function model.total_params() end

---Pousser une couche (API bas niveau / description) dans le modèle.
---Note: cette fonction est surtout utilisée par les builders d'architectures.
---@param name string
---@param layer_type string
---@param params_count integer
---@return boolean ok
---@return string? err
function model.push_layer(name, layer_type, params_count) end

---Forward pass (si exposé par l'implémentation).
---@param input TokenIds|string|table
---@return boolean ok
---@return string? err
function model.forward(input) end

---Backward pass (si exposé par l'implémentation).
---@return boolean ok
---@return string? err
function model.backward() end

---Step optimiseur (si exposé). Le LR peut être transmis.
---@param learning_rate number
---@return boolean ok
---@return string? err
function model.optimizer_step(learning_rate) end

---Choisir backend hardware (CPU/OpenCL/Vulkan/Auto) si supporté.
---@param backend HardwareBackend
---@return boolean ok
---@return string? err
function model.set_hardware(backend) end

---Retourne les capacités détectées (AVX2/FMA/F16C/BMI2).
---@return HardwareCaps caps
function model.hardware_caps() end

--=============================================================================
-- Module: architectures (builders)
--=============================================================================

---@class MimirArchitecturesAPI
local architectures = {}

---Construire un UNet dans le modèle courant.
---@param config? UNetConfig|table
---@return boolean ok
---@return string? err
function architectures.unet(config) end

---Construire un VAE dans le modèle courant.
---@param config? VAEConfig|table
---@return boolean ok
---@return string? err
function architectures.vae(config) end

---Construire un Vision Transformer dans le modèle courant.
---@param config? ViTConfig|table
---@return boolean ok
---@return string? err
function architectures.vit(config) end

---Construire un GAN (generator + discriminator ou global selon impl).
---@param config? GANConfig|table
---@return boolean ok
---@return string? err
function architectures.gan(config) end

---Construire un modèle de diffusion (DDPM-like).
---@param config? DiffusionConfig|table
---@return boolean ok
---@return string? err
function architectures.diffusion(config) end

---Construire un transformer GPT-style (ou transformer générique).
---@param config? TransformerConfig|table
---@return boolean ok
---@return string? err
function architectures.transformer(config) end

---Construire un ResNet (variante interne).
---@param config? ResNetConfig|table
---@return boolean ok
---@return string? err
function architectures.resnet(config) end

---Construire un MobileNet (variante interne).
---@param config? MobileNetConfig|table
---@return boolean ok
---@return string? err
function architectures.mobilenet(config) end

--=============================================================================
-- Module: layers (placeholder / low-level)
--=============================================================================

---@class MimirLayersAPI
local layers = {}

---Conv2D low-level (placeholder). Utiliser `model.forward()` à la place.
---@return boolean ok
---@return string err
function layers.conv2d(...) end

---Linear low-level (placeholder). Utiliser `model.forward()` à la place.
---@return boolean ok
---@return string err
function layers.linear(...) end

---MaxPool2D low-level (placeholder).
---@return boolean ok
---@return string err
function layers.maxpool2d(...) end

---AvgPool2D low-level (placeholder).
---@return boolean ok
---@return string err
function layers.avgpool2d(...) end

---Activation low-level (placeholder).
---@return boolean ok
---@return string err
function layers.activation(...) end

---BatchNorm low-level (placeholder).
---@return boolean ok
---@return string err
function layers.batchnorm(...) end

---LayerNorm low-level (placeholder).
---@return boolean ok
---@return string err
function layers.layernorm(...) end

---Attention low-level (placeholder).
---@return boolean ok
---@return string err
function layers.attention(...) end

--=============================================================================
-- Module: tokenizer
--=============================================================================

---@class MimirTokenizerAPI
local tokenizer = {}

---Créer un tokenizer avec vocab max.
---@param max_vocab integer
---@return boolean ok
function tokenizer.create(max_vocab) end

---Tokeniser un texte (word/BPE selon impl).
---@param text string
---@return TokenIds tokens
function tokenizer.tokenize(text) end

---Dé-tokeniser une séquence (ids -> texte).
---@param tokens TokenIds
---@return string text
function tokenizer.detokenize(tokens) end

---Retourne la taille actuelle du vocab.
---@return integer size
function tokenizer.vocab_size() end

---Sauvegarder le tokenizer dans un fichier.
---@param path string @Ex: "checkpoints/run1/tokenizer.json"
---@return boolean ok
---@return string? err
function tokenizer.save(path) end

---Charger le tokenizer depuis un fichier.
---@param path string
---@return boolean ok
---@return string? err
function tokenizer.load(path) end

---Ajouter un token au vocab.
---@param token string
---@return boolean ok
---@return string? err
function tokenizer.add_token(token) end

---Étendre le vocab à partir d'un texte (analyse corpus simple).
---@param text string
---@return boolean ok
---@return string? err
function tokenizer.ensure_vocab_from_text(text) end

---Tokenize et étend le vocab si nécessaire.
---@param text string
---@return TokenIds tokens
function tokenizer.tokenize_ensure(text) end

---IDs spéciaux.
---@return integer id
function tokenizer.pad_id() end
---@return integer id
function tokenizer.unk_id() end
---@return integer id
function tokenizer.seq_id() end
---@return integer id
function tokenizer.mod_id() end
---@return integer id
function tokenizer.mag_id() end

---Récupérer un token string depuis un id.
---@param id integer
---@return string token
function tokenizer.get_token_by_id(id) end

---Apprendre un BPE depuis un corpus (chemin ou texte selon impl).
---@param corpus_path string
---@param vocab_target? integer
---@return boolean ok
---@return string? err
function tokenizer.learn_bpe(corpus_path, vocab_target) end

---Tokeniser via BPE (si appris/chargé).
---@param text string
---@return TokenIds tokens
function tokenizer.tokenize_bpe(text) end

---Définir la longueur max des séquences côté tokenizer.
---@param max_length integer
---@return boolean ok
---@return string? err
function tokenizer.set_max_length(max_length) end

---Pad une séquence à une longueur cible.
---@param tokens TokenIds
---@param max_length integer
---@param pad_id? integer
---@return TokenIds padded
function tokenizer.pad_sequence(tokens, max_length, pad_id) end

---Tokeniser une liste de textes.
---@param texts string[]
---@return TokenIds[] batches
function tokenizer.batch_tokenize(texts) end

---Afficher des stats (stdout/log).
function tokenizer.print_stats() end

---Fréquences tokens (si supporté).
---@return table<string, integer> freqs
function tokenizer.get_frequencies() end

---Analyse texte (mots, chars, densité, etc. selon impl).
---@param text string
---@return table results
function tokenizer.analyze_text(text) end

---Extraction keywords (heuristique).
---@param text string
---@param top_k? integer
---@return KeywordList keywords
function tokenizer.extract_keywords(text, top_k) end

--=============================================================================
-- Module: dataset
--=============================================================================

---@class MimirDatasetAPI
local dataset = {}

---Charger un dataset depuis un répertoire.
---Attendu: répertoire contenant des fichiers textes / structure interne.
---@param dir string
---@return boolean ok
---@return string? err
function dataset.load(dir) end

---Préparer les séquences (stockées dans le contexte interne).
---La séquence length est utilisée ensuite par model.train().
---@param max_length integer
---@return boolean ok
---@return string? err
function dataset.prepare_sequences(max_length) end

--=============================================================================
-- Module: memory (AdvancedRAMManager)
--=============================================================================

---@class MimirMemoryAPI
local memory = {}

---Configurer le gestionnaire RAM avancé (implémentation dépendante).
---@param cfg table
---@return boolean ok
---@return string? err
function memory.config(cfg) end

---Récupérer stats RAM.
---@return MemoryStats stats
function memory.get_stats() end

---Imprimer stats RAM.
function memory.print_stats() end

---Purger / clear caches.
function memory.clear() end

---Usage en MB (alias utilitaire).
---@return number mb
function memory.get_usage() end

---Définir une limite RAM (MB ou bytes selon impl).
---@param limit_mb number
---@return boolean ok
---@return string? err
function memory.set_limit(limit_mb) end

--=============================================================================
-- Module: guard (MemoryGuard)
--=============================================================================

---@class MimirGuardAPI
local guard = {}

---Définir la limite de mémoire (MB).
---@param limit_mb number
---@return boolean ok
---@return string? err
function guard.set_limit(limit_mb) end

---Stats de la limite stricte.
---@return GuardStats stats
function guard.get_stats() end

---Imprimer stats guard.
function guard.print_stats() end

---Reset stats (peak, compteurs).
function guard.reset() end

--=============================================================================
-- Module: allocator (DynamicTensorAllocator)
--=============================================================================

---@class AllocatorConfig
---@field max_ram_gb number
---@field enable_compression boolean

---@class MimirAllocatorAPI
local allocator = {}

---Configurer l'allocator (RAM max + compression LZ4).
---@param cfg AllocatorConfig|table
---@return boolean ok
---@return string? err
function allocator.configure(cfg) end

---Imprimer stats allocator (stdout/log).
function allocator.print_stats() end

---Récupérer stats allocator.
---@return AllocatorStats stats
function allocator.get_stats() end

--=============================================================================
-- Module: htop (AsyncMonitor / Terminal UI)
--=============================================================================

---@class MimirHtopAPI
local htop = {}

---Créer le monitor async (et option viz).
---@param enable_viz? boolean
---@return boolean ok
---@return string? err
function htop.create(enable_viz) end

---Activer/désactiver l'affichage.
---@param enabled boolean
---@return boolean ok
---@return string? err
function htop.enable(enabled) end

---Mettre à jour les métriques affichées.
---Signature actuelle: (epoch, total_epochs, batch, total_batches, loss, avg_loss, lr,
---                    batch_time_ms?, memory_mb?, memory_freed?, bps?, params?, timestep?,
---                    kl?, wass?, ent?, mom?, spat?, temp?, mse?)
---@param epoch integer
---@param total_epochs integer
---@param batch integer
---@param total_batches integer
---@param loss number
---@param avg_loss number
---@param lr number
---@param batch_time_ms? integer
---@param memory_mb? integer
---@param memory_freed? integer
---@param bps? number
---@param params? integer
---@param timestep? number
---@param kl? number
---@param wass? number
---@param ent? number
---@param mom? number
---@param spat? number
---@param temp? number
---@param mse? number
---@return boolean ok
---@return string? err
function htop.update(epoch, total_epochs, batch, total_batches, loss, avg_loss, lr,
                     batch_time_ms, memory_mb, memory_freed, bps, params, timestep,
                     kl, wass, ent, mom, spat, temp, mse) end

---Forcer un render (si supporté).
---@return boolean ok
---@return string? err
function htop.render() end

---Clear écran / reset UI.
---@return boolean ok
---@return string? err
function htop.clear() end

--=============================================================================
-- Module: viz (SFML Visualizer)
--=============================================================================

---@class MimirVizAPI
local viz = {}

---Créer la fenêtre / contexte viz (optionnel).
---@return boolean ok
---@return string? err
function viz.create() end

---Initialiser (si séparé de create()).
---@return boolean ok
---@return string? err
function viz.initialize() end

---Fenêtre ouverte ?
---@return boolean open
function viz.is_open() end

---Traitement events (clavier/souris/close).
---@return boolean ok
---@return string? err
function viz.process_events() end

---Update interne (tick).
---@return boolean ok
---@return string? err
function viz.update() end

---Ajouter une image à afficher (chemin vers fichier image).
---@param label string
---@param image_path string
---@return boolean ok
---@return string? err
function viz.add_image(label, image_path) end

---Mettre à jour les métriques affichées dans la viz.
---@param metrics VizMetrics|table
---@return boolean ok
---@return string? err
function viz.update_metrics(metrics) end

---Ajouter un point (loss history).
---@param step integer
---@param loss number
---@return boolean ok
---@return string? err
function viz.add_loss_point(step, loss) end

---Clear viz.
---@return boolean ok
---@return string? err
function viz.clear() end

---Activer/désactiver la viz (NO-OP si non compilée).
---@param enabled boolean
---@return boolean ok
---@return string? err
function viz.set_enabled(enabled) end

---Sauvegarder l'historique de loss.
---@param path string
---@return boolean ok
---@return string? err
function viz.save_loss_history(path) end

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

---@class Pipeline
Pipeline = {}

---Setup config globale pipeline (si présent).
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
model = model
---@type MimirTokenizerAPI
tokenizer = tokenizer
---@type MimirDatasetAPI
dataset = dataset
---@type MimirMemoryAPI
memory = memory
---@type MimirGuardAPI
guard = guard
---@type MimirAllocatorAPI
allocator = allocator
---@type MimirHtopAPI
htop = htop
---@type MimirVizAPI
viz = viz
---@type MimirArchitecturesAPI
architectures = architectures
---@type MimirLayersAPI
layers = layers
