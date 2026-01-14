-- Entraînement PonyXL (prompt -> tokenizer -> encoder -> latent -> decodeur -> image RGB)
-- Usage:
--   ./bin/mimir --lua scripts/training/train_ponyxl_ddpm.lua

log("╔═══════════════════════════════════════════════════════════════╗")
log("║     Train PonyXL (prompt->image, autoencoder RGB)            ║")
log("╚═══════════════════════════════════════════════════════════════╝")

local function try_call(fn, ...)
  local ok, res = pcall(fn, ...)
  if ok then return true, res end
  return false, res
end

-- Mémoire:
-- Le modèle + l'allocator peuvent consommer presque toute la limite MemoryGuard.
-- Or stb_image (chargement PNG) fait des reallocs et a besoin d'un peu de marge.
-- => on laisse volontairement un "headroom" entre MemoryGuard et l'allocator.
local MEM_GB = tonumber(os.getenv("MIMIR_MEM_GB") or "")
if not MEM_GB or MEM_GB <= 0 then MEM_GB = 12 end
local ALLOC_GB = tonumber(os.getenv("MIMIR_ALLOC_GB") or "")
-- Par défaut on garde ~3GB de marge pour stb_image + overhead runtime (GPU/driver, caches, etc.).
if not ALLOC_GB or ALLOC_GB <= 0 then ALLOC_GB = math.max(1.0, MEM_GB - 3.0) end

-- IMPORTANT: Allocator.configure peut (selon build) ajuster la limite MemoryGuard.
-- Pour garantir un headroom, on configure l'allocator d'abord, puis on fixe MemoryGuard en dernier.
if Mimir and Mimir.Allocator and Mimir.Allocator.configure then
  Mimir.Allocator.configure({
    max_ram_gb = ALLOC_GB,
    enable_compression = true
  })
end

if Mimir and Mimir.MemoryGuard and Mimir.MemoryGuard.setLimit then
  try_call(Mimir.MemoryGuard.setLimit, MEM_GB)
end

-- Accélération hardware (API actuelle: booléen). Désactivation: MIMIR_HW=0|off|false
local HW = os.getenv("MIMIR_HW")
local ENABLE_HW = true
if HW and #HW > 0 then
  local v = string.lower(HW)
  ENABLE_HW = not (v == "0" or v == "false" or v == "off" or v == "no")
end
if Mimir and Mimir.Model and Mimir.Model.set_hardware then
  pcall(Mimir.Model.set_hardware, ENABLE_HW)
end

local QUICK = (os.getenv("MIMIR_QUICK") == "1")
-- HTOP/monitoring: activé par défaut. Override:
--   MIMIR_ENABLE_HTOP=0 -> disable
--   MIMIR_ENABLE_HTOP=1 -> enable
local ENABLE_HTOP = (os.getenv("MIMIR_ENABLE_HTOP") ~= "0") or true

-- IMPORTANT: la taille du vocab du Tokenizer doit matcher cfg.max_vocab.
-- Sinon l'Encoder ignore les ids >= max_vocab et le signal texte se perd.
local vocab_size = QUICK and 4096 or 8192

local cfg = {
  -- Texte
  d_model = QUICK and 128 or 256,
  -- NB: utilisé côté C++ pour créer l'Encoder si un tokenizer est présent
  embed_dim = QUICK and 128 or 256,
  seq_len = QUICK and 64 or 128,
  max_vocab = vocab_size,

  -- Bottleneck latent
  hidden_dim = QUICK and 1024 or 2048,
  latent_dim = QUICK and 256 or 512,

  -- Entraînement triple-fault: original + 1..blur_levels versions floutées
  blur_levels = 4,

  image_w = 64,
  image_h = 64,
  image_c = 3,

  cfg_dropout_prob = 0.01,

  -- Forcer le modèle à utiliser le texte: parfois on retire l'image conditionnelle (x_in).
  -- Sans ça, il peut apprendre à reconstruire uniquement depuis l'image floutée et ignorer le prompt.
  cond_image_dropout_prob = QUICK and 0.0 or 0.05,
  cond_image_dropout_lr_scale = 0.25,
  cond_image_dropout_noise_std = 0.0,

  dropout = 0.0,

  optimizer = "adamw",
  weight_decay = 0.001,

  -- Warmup LR (en nombre de steps d'optimiseur). 0 = désactivé.
  warmup_steps = QUICK and 10 or 600,

  max_items = QUICK and 200 or 0,
  max_text_chars = 4096,

  -- Pré-traitement prompts (FR): extraire des mots-clés (stopwords filtrés) et
  -- les mettre en tête du prompt pour renforcer le signal texte->image.
  -- Exemple interne: "motscles... | phrase complète ..."
  prompt_keywords_prefix = true,
  prompt_keywords_max = 16,
  -- Probabilité d'utiliser uniquement les mots-clés (sans la phrase). 0 = jamais.
  prompt_keywords_only_prob = 0.2,

  -- Debug: afficher quelques prompts parsés au début
  prompt_debug_dump = false,
  prompt_debug_first = 5,
  prompt_debug_every = 0,
  prompt_debug_max_chars = 280,

  -- Checkpoints automatiques côté C++ (dans Mimir.Model.train)
  checkpoint_every_epochs = 2,
  checkpoint_dir = "checkpoint/PonyXL_DDPM",
  checkpoint_include_checksums = true,
  checkpoint_include_gradients = true,
  checkpoint_include_weight_deltas = true,
  checkpoint_include_optimizer_state = true,
  checkpoint_include_activations = true
}

local epochs = 100
local lr = QUICK and 5e-4 or 1e-4

do
  local e = tonumber(os.getenv("MIMIR_EPOCHS") or "")
  if e and e > 0 then
    epochs = math.floor(e)
  end

  local mi = tonumber(os.getenv("MIMIR_MAX_ITEMS") or "")
  if mi and mi >= 0 then
    cfg.max_items = math.floor(mi)
  end

  local cke = tonumber(os.getenv("MIMIR_CKPT_EVERY_EPOCHS") or "")
  if cke and cke >= 0 then
    cfg.checkpoint_every_epochs = math.floor(cke)
  end

  local ckd = os.getenv("MIMIR_CKPT_DIR")
  if ckd and #ckd > 0 then
    cfg.checkpoint_dir = ckd
  end

  local lre = tonumber(os.getenv("MIMIR_LR") or "")
  if lre and lre > 0 then
    lr = lre
  end
end

log(string.format(
  "Config: d_model=%d seq_len=%d max_vocab=%d hidden_dim=%d latent_dim=%d blur_levels=%d img=%dx%dx%d",
  cfg.d_model, cfg.seq_len, cfg.max_vocab, cfg.hidden_dim, cfg.latent_dim, cfg.blur_levels, cfg.image_w, cfg.image_h, cfg.image_c
))
log(string.format("Train: epochs=%d lr=%.6g warmup_steps=%d", epochs, lr, cfg.warmup_steps or 0))

if ENABLE_HTOP and Mimir and Mimir.Htop and Mimir.Htop.create then
  pcall(Mimir.Htop.create, false)
end

-- NOTE: le dataset n'est pas versionné dans ce repo.
-- Définis `MIMIR_DATASET_ROOT` pour pointer vers ton dossier (ex: ../tensor/datasets.old).
local dataset_root = os.getenv("MIMIR_DATASET_ROOT") or "../dataset"
log("Dataset: " .. dataset_root)

local ok_ds, num_items_or_err = Mimir.Dataset.load(dataset_root)
if not ok_ds then
  error("Dataset.load failed: " .. tostring(num_items_or_err))
end

local num_items = tonumber(num_items_or_err) or 0
log("✓ Dataset chargé | items=" .. tostring(num_items))

-- Tokenizer: construit un vocab à partir d'un échantillon
local ok_tok, err_tok = Mimir.Tokenizer.create(vocab_size)
if ok_tok == false then
  error("Tokenizer.create failed: " .. tostring(err_tok))
end

local tok = Mimir.Tokenizer
local ensure = tok.tokenize_ensure or tok.tokenize
if not ensure then
  error("Tokenizer: tokenize/tokenize_ensure indisponible")
end

local vocab_build_items = math.min(num_items, QUICK and 50 or 500)
log("• Construction vocab (échantillon: " .. tostring(vocab_build_items) .. ")...")
for i = 1, vocab_build_items do
  local item = Mimir.Dataset.get(i)
  local text = item and item.text
  if (not text or #text == 0) and item and item.text_file then
    local f = io.open(item.text_file, "r")
    if f then
      text = f:read("*a")
      f:close()
    end
  end
  if text and #text > 0 then
    ensure(text)
  end
  if (i % 100 == 0) or (i == vocab_build_items) then
    log("  - vocab progress: " .. i .. "/" .. vocab_build_items)
  end
end

log("✓ Tokenizer prêt")

-- NOTE: la boucle d'entraînement PonyXL_DDPM (côté C++) entraîne directement sur le dataset
-- (texte + image). Il n'est pas nécessaire de préparer des séquences ici.

local ok_create, err_create = Mimir.Model.create("ponyxl_ddpm", cfg)
if not ok_create then
  error("Model.create('ponyxl_ddpm') failed: " .. tostring(err_create))
end

local ok_build, params_or_err = Mimir.Model.build()
if not ok_build then
  error("Model.build failed: " .. tostring(params_or_err))
end

local ok_alloc, err_alloc = Mimir.Model.allocate_params()
if not ok_alloc then
  error("Model.allocate_params failed: " .. tostring(err_alloc))
end

local ok_init, err_init = Mimir.Model.init_weights("xavier", 1337)
if not ok_init then
  error("Model.init_weights failed: " .. tostring(err_init))
end

log("✓ Model prêt | params=" .. tostring(Mimir.Model.total_params()))

local ok_train, err_train = Mimir.Model.train(epochs, lr)
if not ok_train then
  error("Model.train failed: " .. tostring(err_train))
end

log("✓ Terminé (checkpoints auto dans " .. tostring(cfg.checkpoint_dir) .. ")")
local ok_save, err_save = Mimir.Serialization.save(cfg.checkpoint_dir, "raw_folder", {
  save_encoder = true,
  save_optimizer = true,
  save_tokenizer = true,
  include_checksums = true,
  include_git_info = true,
  include_gradients = true,
  include_activations = true,
  include_optimizer_state = true,
  include_weight_deltas = true

})

if not ok_save then

  error("Mimir.Serialization.save() failed: " .. tostring(err_save))
end