---@meta
---@version 2.1.0
---@author <bri45> for "Mímir Framework"
---@date 27 décembre 2025
---@diagnostic disable: missing-return, unused-local, unused-vararg, duplicate-doc-field, redundant-parameter

--=============================================================================
-- Mímir Framework v2.1 — IDE Stub (EmmyLua)
--=============================================================================
-- Ce fichier est un "stub" destiné aux IDE (LuaLS / EmmyLua / IntelliJ, etc.).
-- Il documente l'API globale exposée par le binaire `mimir` (bindings C/C++).
--
-- ⚠️  IMPORTANT: Ce fichier est synchronisé avec src/LuaScripting.cpp
--    Toute modification de l'API C++ doit être reflétée ici.
--
-- Objectifs :
--  • Autocomplétion IDE, signatures, types, docstrings
--  • Stable : les scripts Lua sont l'API publique
--  • Documentation de référence pour les utilisateurs
--
-- Nouveautés v2.1.0 :
--  • Scripts réorganisés en catégories (demos, examples, tests, etc.)
--  • Documentation corrigée (33 fixes)
--  • Synchronisation API validée (114 fonctions)
--
-- Historique v2.0.0 :
--  • MemoryGuard API (limite stricte 10 GB par défaut)
--  • allocator.configure() pour configuration mémoire (OBLIGATOIRE)
--  • FluxModel API complète (diffusion text-to-image avec VAE)
--  • Modes train()/eval() pour tous les modèles
--  • Support complet 8 architectures (UNet, VAE, ViT, GAN, Diffusion, Transformer, ResNet, MobileNet)
--  • API htop et viz pour monitoring temps réel
--
-- Remarques :
--  • Les fonctions retournent souvent (ok:boolean, value) ou (ok:boolean, err:string)
--  • ⚠️  Toujours appeler allocator.configure() au début des scripts!
--  • Les tables de stats sont des structs (tables Lua avec champs nommés)
--  • Les "layers.*" sont des opérations bas niveau (préférer model.forward())
--  • MemoryGuard: API moderne recommandée (guard: API legacy compatible)
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

---@class FluxConfig
---@field image_resolution? int @Résolution des images (défaut: 256)
---@field latent_channels? int @Canaux de l'espace latent (défaut: 4)
---@field latent_resolution? int @Résolution de l'espace latent (défaut: 32)
---@field vae_base_channels? int @Canaux de base VAE (défaut: 128)
---@field vae_channel_mult? int[] @Multiplicateurs de canaux VAE (ex: {1,2,4,4})
---@field num_res_blocks? int @Nombre de blocs résiduels (défaut: 2)
---@field vocab_size? int @Taille du vocabulaire (défaut: 50000)
---@field text_max_length? int @Longueur max du texte (défaut: 77)
---@field text_embed_dim? int @Dimension embeddings texte (défaut: 768)
---@field transformer_dim? int @Dimension transformer (défaut: 768)
---@field num_transformer_blocks? int @Nombre de blocs transformer (défaut: 12)
---@field num_attention_heads? int @Nombre de têtes attention (défaut: 12)
---@field mlp_ratio? float @Ratio MLP (défaut: 4.0)
---@field timestep_embed_dim? int @Dimension embedding timestep (défaut: 256)
---@field num_diffusion_steps? int @Nombre de steps diffusion (défaut: 1000)

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
---Mode training activé par défaut pour permettre le backward pass.
---@param input TokenIds|string|table|float[] @Données d'entrée
---@param training? bool @Mode training (défaut: true) pour calculer les gradients
---@return float[]|nil @Sortie du modèle
---@return string? err
function model.forward(input, training) end

---Backward pass pour calculer les gradients.
---@param loss_gradient float[] @Gradient de la loss
---@return boolean ok
---@return string? err
function model.backward(loss_gradient) end

---Réinitialise tous les gradients à zéro.
---Important: à appeler avant chaque itération d'entraînement.
---@return boolean ok
---@return string? err
function model.zero_grads() end

---Récupère les gradients actuels de tous les paramètres.
---@return float[]|nil @Vecteur de tous les gradients
---@return string? err
function model.get_gradients() end

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

---Construire un modèle Flux (Diffusion avec VAE et text conditioning).
---@param config? FluxConfig|table
---@return boolean ok
---@return string? err
function architectures.flux(config) end

--=============================================================================
-- Module: flux (API Fonctionnelle Flux)
--=============================================================================

---@class FluxAPI
local flux = {}

---Générer une image depuis un prompt texte (API simplifiée).
---Utilise le modèle Flux global configuré.
---@param prompt string @Prompt texte décrivant l'image à générer
---@param num_steps? integer @Nombre de steps de diffusion (défaut: 50)
---@return table? image_data @Pixels de l'image générée (ou nil si erreur)
---@return string? err @Message d'erreur
---
---**Exemple:**
---```lua
--- local img, err = flux.generate("A beautiful sunset over mountains", 50)
--- if img then
---   print("Image générée avec succès")
--- else
---   print("Erreur:", err)
--- end
---```
function flux.generate(prompt, num_steps) end

---Encoder une image en espace latent via le VAE.
---@param image_path string @Chemin vers l'image à encoder
---@return table? latent @Représentation latente (ou nil si erreur)
---@return string? err @Message d'erreur
---
---**Exemple:**
---```lua
--- local latent, err = flux.encode_image("input.png")
--- if latent then
---   print("Image encodée:", #latent, "éléments")
--- end
---```
function flux.encode_image(image_path) end

---Décoder un vecteur latent en image via le VAE.
---@param latent table @Vecteur latent à décoder
---@return table? pixels @Pixels de l'image (RGBA, row-major)
---@return string? err @Message d'erreur
---
---**Exemple:**
---```lua
--- local pixels, err = flux.decode_latent(latent)
--- if pixels then
---   viz.add_image(pixels, width, height, 4)
--- end
---```
function flux.decode_latent(latent) end

---Encoder un prompt texte en embeddings via le text encoder.
---@param text string @Texte à encoder
---@return table? embeddings @Embeddings textuels (ou nil si erreur)
---@return string? err @Message d'erreur
---
---**Exemple:**
---```lua
--- local embed, err = flux.encode_text("A cat sitting on a chair")
--- if embed then
---   print("Texte encodé:", #embed, "dimensions")
--- end
---```
function flux.encode_text(text) end

---Définir le tokenizer pour le text encoder.
---@param tokenizer_path string @Chemin vers le fichier tokenizer.json
---@return boolean ok @true si succès
---@return string? err @Message d'erreur
---
---**Exemple:**
---```lua
--- local ok, err = flux.set_tokenizer("checkpoints/tokenizer.json")
--- if not ok then
---   print("Erreur tokenizer:", err)
--- end
---```
function flux.set_tokenizer(tokenizer_path) end

--=============================================================================
-- Module: FluxModel (API Orientée Objet Flux)
--=============================================================================

---@class FluxModelAPI
local FluxModel = {}

---Créer une nouvelle instance de FluxModel.
---@param config? FluxConfig|table @Configuration du modèle
---@return table flux_model @Instance FluxModel
---@return string? err @Message d'erreur
---
---**Exemple:**
---```lua
--- local flux_model, err = FluxModel.new({
---   image_resolution = 256,
---   latent_channels = 4,
---   vocab_size = 50000,
---   text_embed_dim = 768
--- })
--- if not flux_model then
---   print("Erreur:", err)
--- end
---```
function FluxModel.new(config) end

---Activer le mode training (pour entraînement).
---En mode training: dropout activé, reparametrization trick VAE appliqué.
---
---**Exemple:**
---```lua
--- FluxModel.train()
--- -- Entraînement du modèle...
---```
function FluxModel.train() end

---Activer le mode evaluation/inference.
---En mode eval: dropout désactivé, VAE déterministe (pas de bruit).
---
---**Exemple:**
---```lua
--- FluxModel.eval()
--- local image = FluxModel.generate("beautiful sunset", 50)
---```
function FluxModel.eval() end

---Vérifier si le modèle est en mode training.
---@return boolean is_training @true si en mode training, false si en eval
---
---**Exemple:**
---```lua
--- if FluxModel.isTraining() then
---     print("Mode training activé")
--- else
---     print("Mode inference activé")
--- end
---```
function FluxModel.isTraining() end

---Encoder une image vers l'espace latent VAE.
---@param image number[] @Image RGB aplatie (H×W×3 valeurs entre -1 et 1)
---@return number[] latent @Vecteur latent compressé (latent_resolution²×latent_channels)
---
---**Exemple:**
---```lua
--- local image_size = 3 * 256 * 256
--- local image = {}
--- for i = 1, image_size do
---     image[i] = math.random() * 2 - 1  -- [-1, 1]
--- end
--- local latent = FluxModel.encodeImage(image)
--- print("Latent size: " .. #latent)
---```
function FluxModel.encodeImage(image) end

---Décoder un vecteur latent vers une image RGB.
---@param latent number[] @Vecteur latent
---@return number[] image @Image RGB aplatie (H×W×3 valeurs entre -1 et 1)
---
---**Exemple:**
---```lua
--- local reconstructed = FluxModel.decodeLatent(latent)
--- print("Image reconstruite: " .. #reconstructed .. " pixels")
---```
function FluxModel.decodeLatent(latent) end

---Tokeniser un prompt texte.
---@param prompt string @Texte à tokeniser
---@return integer[] tokens @Séquence de tokens (max text_max_length)
---
---**Exemple:**
---```lua
--- local tokens = FluxModel.tokenizePrompt("a beautiful mountain landscape")
--- print("Nombre de tokens: " .. #tokens)
---```
function FluxModel.tokenizePrompt(prompt) end

---Encoder un prompt texte en embeddings.
---@param tokens integer[] @Séquence de tokens
---@return number[] embeddings @Embeddings texte (text_max_length × text_embed_dim)
---
---**Exemple:**
---```lua
--- local tokens = FluxModel.tokenizePrompt("cyberpunk city")
--- local text_emb = FluxModel.encodeText(tokens)
--- print("Text embedding size: " .. #text_emb)
---```
function FluxModel.encodeText(tokens) end

---Prédire le bruit dans un latent bruité (step de diffusion).
---@param noisy_latent number[] @Latent avec bruit
---@param text_embedding number[] @Embeddings texte pour conditioning
---@param timestep integer @Step de diffusion (0 à num_diffusion_steps)
---@return number[] predicted_noise @Bruit prédit par le modèle
---
---**Exemple:**
---```lua
--- local noise = FluxModel.predictNoise(noisy_latent, text_emb, 500)
--- print("Bruit prédit: " .. #noise .. " valeurs")
---```
function FluxModel.predictNoise(noisy_latent, text_embedding, timestep) end

---Générer une image depuis un prompt texte (pipeline complet).
---@param prompt string @Description textuelle de l'image
---@param num_steps? integer @Nombre de steps de diffusion (défaut: 50)
---@return number[] image @Image générée RGB (H×W×3)
---
---**Exemple:**
---```lua
--- FluxModel.eval()  -- Mode inference
--- local image = FluxModel.generate("a serene lake at sunset", 50)
--- print("Image générée: " .. #image .. " pixels")
---```
function FluxModel.generate(prompt, num_steps) end

---Calculer la loss de diffusion pour l'entraînement.
---@param image number[] @Image RGB target
---@param tokens integer[] @Tokens du prompt
---@return number loss @Valeur de loss
---
---**Exemple:**
---```lua
--- FluxModel.train()  -- Mode training
--- local loss = FluxModel.computeDiffusionLoss(image, tokens)
--- print("Loss: " .. loss)
---```
function FluxModel.computeDiffusionLoss(image, tokens) end

---Définir le tokenizer pour les prompts.
---@param tokenizer_instance any @Instance du tokenizer
---
---**Exemple:**
---```lua
--- tokenizer.create(50000)
--- FluxModel.setPromptTokenizer(tokenizer)
---```
function FluxModel.setPromptTokenizer(tokenizer_instance) end

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

---@class DatasetItem
---@field text_file? string Chemin du fichier texte
---@field image_file? string Chemin du fichier image
---@field audio_file? string Chemin du fichier audio
---@field video_file? string Chemin du fichier vidéo
---@field text? string Contenu texte (si chargé)
---@field width? int Largeur de l'image
---@field height? int Hauteur de l'image

---@class MimirDatasetAPI
local dataset = {}

---Charger un dataset depuis un répertoire.
---Attendu: répertoire contenant des fichiers textes / structure interne.
---@param dir string
---@return boolean ok
---@return string? err
function dataset.load(dir) end

---Récupérer un item du dataset par son index (1-based).
---Retourne une table avec les chemins et métadonnées de l'item.
---@param index integer Index de l'item (commence à 1)
---@return DatasetItem|nil item Item du dataset
---@return string? err Message d'erreur si échec
function dataset.get(index) end

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

---Purger / clear caches mémoire.
---@return boolean ok
---@return string? err
function memory.clear() end

---Usage actuel en MB (alias utilitaire).
---@return number mb
function memory.get_usage() end

---Définir une limite RAM (en MB).
---@param limit_mb number
---@return boolean ok
---@return string? err
function memory.set_limit(limit_mb) end

--=============================================================================
-- Module: guard (MemoryGuard) - API Ancienne
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
-- Module: MemoryGuard (API Moderne - Recommandée)
--=============================================================================

---@class MemoryGuardStats
---@field current_mb float @Utilisation RAM courante en MB
---@field peak_mb float @Pic d'utilisation en MB
---@field limit_mb float @Limite configurée en MB
---@field usage_percent float @Pourcentage d'utilisation

---@class MimirMemoryGuardAPI
local MemoryGuard = {}

---Définir la limite de mémoire RAM stricte.
---Accepte des valeurs en bytes (grands nombres) ou en GB (si <= 1000).
---@param limit number @Limite en bytes ou en GB (si valeur <= 1000)
---@return boolean ok @true si succès
---
---**Exemples:**
---```lua
--- -- Définir limite à 10 Go
--- MemoryGuard.setLimit(10 * 1024 * 1024 * 1024)  -- en bytes
--- MemoryGuard.setLimit(10)  -- en GB (auto-détecté car < 1000)
---```
function MemoryGuard.setLimit(limit) end

---Obtenir la limite de mémoire configurée.
---@return integer bytes @Limite en bytes
---
---**Exemple:**
---```lua
--- local limit = MemoryGuard.getLimit()
--- print(string.format("Limite: %.2f GB", limit / 1024 / 1024 / 1024))
---```
function MemoryGuard.getLimit() end

---Obtenir l'utilisation RAM courante.
---@return integer bytes @RAM utilisée actuellement en bytes
---
---**Exemple:**
---```lua
--- local current = MemoryGuard.getCurrentUsage()
--- local limit = MemoryGuard.getLimit()
--- local percent = (current / limit) * 100
--- print(string.format("RAM: %.2f%%", percent))
---```
function MemoryGuard.getCurrentUsage() end

---Obtenir le pic d'utilisation RAM.
---@return integer bytes @Pic d'utilisation en bytes depuis le démarrage
---
---**Exemple:**
---```lua
--- local peak = MemoryGuard.getPeakUsage()
--- print(string.format("Pic RAM: %.2f GB", peak / 1024 / 1024 / 1024))
---```
function MemoryGuard.getPeakUsage() end

---Récupérer toutes les statistiques en une seule fois.
---@return MemoryGuardStats stats @Structure contenant toutes les stats
---
---**Exemple:**
---```lua
--- local stats = MemoryGuard.getStats()
--- print("RAM courante: " .. stats.current_mb .. " MB")
--- print("Pic: " .. stats.peak_mb .. " MB")
--- print("Limite: " .. stats.limit_mb .. " MB")
--- print("Utilisation: " .. stats.usage_percent .. "%")
---```
function MemoryGuard.getStats() end

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
--- MemoryGuard.setLimit(10 * 1024 * 1024 * 1024)
--- -- ... opérations ...
--- MemoryGuard.printStats()  -- Affiche le rapport complet
---```
function MemoryGuard.printStats() end

---Réinitialiser les compteurs de statistiques.
---Remet à zéro le pic et les compteurs d'allocations/libérations.
---La limite configurée est préservée.
---
---**Exemple:**
---```lua
--- MemoryGuard.reset()
--- print("✓ Statistiques réinitialisées")
---```
function MemoryGuard.reset() end

--=============================================================================
-- Module: allocator (DynamicTensorAllocator)
--=============================================================================

---@class AllocatorConfig
---@field max_tensors? integer @Nombre max de tenseurs en mémoire
---@field offload_threshold_mb? float @Seuil de RAM pour offload
---@field swap_strategy? string @Stratégie de swap (lru, fifo, etc.)
---@field max_ram_gb? number @Limite RAM globale
---@field enable_compression? boolean @Activer compression LZ4

---@class MimirAllocatorAPI
local allocator = {}

---Configurer l'allocator dynamique (tenseurs, offload, compression).
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
local htop = {} --avec configuration optionnelle.
---@param config? table @Configuration: {enable_viz: bool, refresh_rate: int, ...}
---@return boolean ok
---@return string? err
function htop.create(config) end

---Activer/désactiver l'affichage htop.
---@param enabled boolean
---@return boolean ok
---@return string? err
function htop.enable(enabled) end

---Mettre à jour les métriques affichées dans htop.
---Les paramètres peuvent être passés via une table HtopMetrics ou individuellement.
---@param metrics HtopMetrics|table @Structure de métriques ou paramètres individuels
---@return boolean ok
---@return string? err
function htop.update(metricsatches, loss, avg_loss, lr,
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

---Créer la fenêtre visualiseur SFML avec titre et dimensions optionnels.
---@param title? string @Titre de la fenêtre (défaut: "Mimir Visualizer")
---@param width? integer @Largeur en pixels (défaut: 1280)
---@param height? integer @Hauteur en pixels (défaut: 720)
---@return boolean ok
---@return string? err
function viz.create(title, width, height) end

---Initialiser le visualiseur (ouvre la fenêtre SFML).
---@return boolean ok
---@return string? err
function viz.initialize() end

---Vérifier si la fenêtre est ouverte.
---@return boolean open
function viz.is_open() end

---Traiter les événements fenêtre (fermeture, clavier, souris).
function viz.process_events() end

---Mettre à jour et afficher le rendu de la fenêtre.
function viz.update() end

---Ajouter/afficher une image dans le visualiseur.
---Les pixels sont fournis sous forme de tableau plat (row-major).
---@param pixels number[] @Tableau de valeurs pixel [0-255], RGBA ou RGB
---@param width integer @Largeur de l'image
---@param height integer @Hauteur de l'image
---@param channels integer @Nombre de canaux (3=RGB, 4=RGBA)
---@return boolean ok
---@return string? err
function viz.add_image(pixels, width, height, channels) end

---Mettre à jour les métriques d'entraînement affichées.
---@param metrics VizMetrics|table @Métriques: epoch, loss, lr, memory_mb, etc.
---@return boolean ok
---@return string? err
function viz.update_metrics(metrics) end

---Ajouter un point à l'historique de loss (pour graphe).
---@param loss number @Valeur de loss
function viz.add_loss_point(loss) end

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
---@type MimirArchitecturesAPI
architectures = architectures
---@type FluxAPI
flux = flux
---@type FluxModelAPI
FluxModel = FluxModel
---@type MimirLayersAPI
layers = layers
---@type MimirTokenizerAPI
tokenizer = tokenizer
---@type MimirDatasetAPI
dataset = dataset
---@type MimirMemoryAPI
memory = memory
---@type MimirGuardAPI
guard = guard
---@type MimirMemoryGuardAPI
MemoryGuard = MemoryGuard
---@type MimirAllocatorAPI
allocator = allocator
---@type MimirHtopAPI
htop = htop
---@type MimirVizAPI
viz = viz
