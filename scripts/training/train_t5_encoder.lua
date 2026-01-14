#!/usr/bin/env mimir --lua

--[[
Train PonyXL (texte -> image) (dataset ../tensor/datasets.old)
=================================================

Le modèle `ponyxl` est un modèle texte -> image (grayscale) :
- input: prompt texte (dataset linkables: text -> image)
- target: image 125x125 (normalisée en [-1,1])

⚠️ Le runtime Mímir est float-only pour les Embedding tokens; on utilise donc
des embeddings déterministes (sinusoïdes) à partir des tokens.

Usage:
  bin/mimir --lua scripts/training/train_t5_encoder.lua

Note: l'option `--lua <script>` (voir `src/main.cpp`) ne propage pas les args au script.
Pour changer la taille du modèle / steps / lr, modifie les variables en haut de ce fichier.
]]

log("╔═══════════════════════════════════════════════════════════════╗")
log("║     Train PonyXL (texte->image) (../tensor/datasets.old)      ║")
log("╚═══════════════════════════════════════════════════════════════╝")

-- ---------------------------------------------------------------------------
-- 0) Runtime / mémoire
-- ---------------------------------------------------------------------------
local function try_call(fn, ...)
  local ok, res = pcall(fn, ...)
  if ok then return true, res end
  return false, res
end

if Mimir and Mimir.MemoryGuard and Mimir.MemoryGuard.setLimit then
  try_call(Mimir.MemoryGuard.setLimit, 10) -- GB
end

if Mimir and Mimir.Allocator and Mimir.Allocator.configure then
  local ok_alloc, err_alloc = Mimir.Allocator.configure({
    max_ram_gb = 10.0,
    enable_compression = true,
    swap_strategy = "lru"
  })
  if ok_alloc then
    log("✓ Allocateur configuré (10GB + compression)")
  else
    log("⚠️  Allocateur: configure() a échoué: " .. tostring(err_alloc))
  end
end

if Mimir and Mimir.Model and Mimir.Model.set_hardware then
  Mimir.Model.set_hardware("auto")
end

-- ---------------------------------------------------------------------------
-- Réglages (modifie ici)
-- ---------------------------------------------------------------------------
local QUICK = false
local ENABLE_HTOP = true

-- ---------------------------------------------------------------------------
-- 1) Config modèle
-- ---------------------------------------------------------------------------
local cfg = {
  d_model = QUICK and 64 or 128,
  seq_len = QUICK and 32 or 64,
  hidden_dim = QUICK and 512 or 1024,
  num_hidden_layers = QUICK and 2 or 3,
  image_w = 64,
  image_h = 64,
  dropout = 0.0,

  -- training hints (utilisés côté LuaScripting/trainModel)
  optimizer = "adamw",
  max_items = QUICK and 200 or 0,
  max_text_chars = 4096
}

local epochs = QUICK and 1 or 300
local lr = QUICK and 5e-4 or 2e-4

log(string.format("Config: d_model=%d seq_len=%d hidden_dim=%d layers=%d img=%dx%d", cfg.d_model, cfg.seq_len, cfg.hidden_dim, cfg.num_hidden_layers, cfg.image_w, cfg.image_h))
log(string.format("Train: epochs=%d lr=%.6g", epochs, lr))

-- ---------------------------------------------------------------------------
-- 1.25) Monitoring (Htop)
-- ---------------------------------------------------------------------------
if ENABLE_HTOP and Mimir and Mimir.Htop and Mimir.Htop.create then
  -- arg1 = enable_viz (optionnel)
  pcall(Mimir.Htop.create, false)
end

-- ---------------------------------------------------------------------------
-- 1.5) Dataset + Tokenizer (texte -> tokens)
-- ---------------------------------------------------------------------------
local dataset_root = "../tensor/datasets.old"
log("Dataset: " .. dataset_root)

-- Charger le dataset en mode linkables, et fixer la taille image cible (125x125)
-- min_modalities=2 => on veut au moins texte+image
local ok_ds, num_items_or_err = Mimir.Dataset.load(dataset_root, cfg.image_w, cfg.image_h, 2)
if not ok_ds then
  error("Dataset.load failed: " .. tostring(num_items_or_err))
end

local num_items = tonumber(num_items_or_err) or 0
log("✓ Dataset chargé | items=" .. tostring(num_items))

-- Tokenizer
local vocab_size = 50000
if Mimir and Mimir.Tokenizer and Mimir.Tokenizer.create then
  local ok_tok, err_tok = Mimir.Tokenizer.create(vocab_size)
  if ok_tok == false then
    error("Tokenizer.create failed: " .. tostring(err_tok))
  end
else
  error("Tokenizer API indisponible")
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

local pad_id = (tok.pad_id and tok.pad_id()) or 0
log("✓ Tokenizer prêt | pad_id=" .. tostring(pad_id))

-- ---------------------------------------------------------------------------
-- 2) Build modèle
-- ---------------------------------------------------------------------------
local ok_create, err_create = Mimir.Model.create("ponyxl", cfg)
if not ok_create then
  error("Model.create('ponyxl') failed: " .. tostring(err_create))
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
-- ---------------------------------------------------------------------------
-- 3) Entraînement (C++: PonyXLModel::trainOnDatasetItems via Mimir.Model.train)
-- ---------------------------------------------------------------------------
local ok_train, err_train = Mimir.Model.train(epochs, lr)
if not ok_train then
  error("Model.train failed: " .. tostring(err_train))
end

Mimir.Serialization.save("checkpoint/PonyXL", "raw_folder", {
  save_optimizer= true
})

log("✓ Terminé")
