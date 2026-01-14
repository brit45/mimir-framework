#!/usr/bin/env mimir --lua
-- ============================================================================
-- Mímir Framework - Simple REAL Training with VAE (API-compliant)
-- Dataset: ../tensor/datasets.old (text + image, handled by C++ loader)
--
-- Uses ONLY official API from mimir-api.lua:
--   dataset.load, dataset.prepare_sequences
--   model.create, architectures.vae, model.build, model.train
--   model.save, model.set_hardware (optional), Allocator.configure (optional)
-- ============================================================================

log("╔═══════════════════════════════════════════════════════════════╗")
log("║     Mímir - VAE Training Pipeline (Simple & Real)             ║")
log("╚═══════════════════════════════════════════════════════════════╝")

local function _mimir_add_module_path()
  local ok, info = pcall(debug.getinfo, 1, "S")
  if not ok or type(info) ~= "table" then return end
  local src = info.source
  if type(src) ~= "string" or src:sub(1, 1) ~= "@" then return end
  local dir = src:sub(2):match("(.*/)")
  if not dir then return end
  package.path = package.path .. ";" .. dir .. "../modules/?.lua;" .. dir .. "../modules/?/init.lua"
end

_mimir_add_module_path()
local Arch = require("arch")

local model = (type(_G.Mimir) == "table" and type(Mimir.Model) == "table") and Mimir.Model or _G.model
local dataset = (type(_G.Mimir) == "table" and type(Mimir.Dataset) == "table") and Mimir.Dataset or _G.dataset

-- ---------------------------------------------------------------------------
-- Runtime / memory (optional, but coherent with your framework)
-- ---------------------------------------------------------------------------
log("\n🔧 Runtime setup...")

-- Optional: choose backend (cpu/vulkan/auto)
-- model.set_hardware("auto")

-- Optional: allocator (if you want strict RAM cap)
-- Allocator.configure({ max_ram_gb = 8.0, enable_compression = true })

-- ---------------------------------------------------------------------------
-- Dataset
-- ---------------------------------------------------------------------------
local dataset_dir = "../tensor/datasets.old"
log("\n📦 Loading dataset: " .. dataset_dir)

local ok, err = dataset.load(dataset_dir)
if not ok then
  error("dataset.load failed: " .. tostring(err))
end
log("✓ Dataset loaded")

-- Prepare sequences for training (required by model.train)
-- Even if the dataset contains images, your internal loader may still prepare
-- text sequences or other training buffers here.
local max_seq_len = 128
log("🧩 Preparing sequences (max_len=" .. max_seq_len .. ") ...")

ok, err = dataset.prepare_sequences(max_seq_len)
if not ok then
  error("dataset.prepare_sequences failed: " .. tostring(err))
end
log("✓ Sequences prepared")

-- ---------------------------------------------------------------------------
-- Model: VAE
-- ---------------------------------------------------------------------------
log("\n🧠 Building VAE model...")

-- Minimal VAE config (fits your stub)
-- For image datasets, input_dim is typically: C*H*W
-- If your loader uses a fixed resolution internally, keep these consistent with it.
local C = 3
local H = 64
local W = 64
local input_dim = C * H * W

local vae_cfg = {
  input_dim = input_dim,
  latent_dim = 64,
  encoder_hidden = 512,
  decoder_hidden = 512
}

local cfg, warn = Arch.build_config("vae", vae_cfg)
if warn then
  log("⚠️  " .. tostring(warn))
end

ok, err = model.create("vae", cfg)
if not ok then
  error("model.create('vae') failed: " .. tostring(err))
end

log("✓ VAE built")
log("📊 Params: " .. tostring(model.total_params()))

ok, err = model.allocate_params()
if not ok then
  error("model.allocate_params failed: " .. tostring(err))
end

ok, err = model.init_weights("xavier", 1337)
if not ok then
  error("model.init_weights failed: " .. tostring(err))
end
log("✓ Weights initialized")

-- ---------------------------------------------------------------------------
-- Train
-- ---------------------------------------------------------------------------
local epochs = 10000
local lr = 3e-4

dataset.prepare_sequences(12)

log("\n🚀 Training...")
log("  epochs=" .. epochs .. ", lr=" .. lr)

model.train(epochs, lr)

log("✅ Training done")

-- ---------------------------------------------------------------------------
-- Save
-- ---------------------------------------------------------------------------
local out_dir = "checkpoints/vae_simple"
log("\n💾 Saving: " .. out_dir)

ok, err = Mimir.Serialization.save(out_dir .. ".safetensors", "safetensors")
if not ok then
  error("Mimir.Serialization.save failed: " .. tostring(err))
end
log("✓ Saved")

log("\n🏁 Finished.")
