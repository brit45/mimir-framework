#!/usr/bin/env lua
-- ══════════════════════════════════════════════════════════════
--  TEMPLATE SCRIPT - Pipeline Only (Mímir)
--
--  Objectif:
--    - Exemple minimal d'utilisation de `scripts/modules/pipeline_api.lua`
--    - Sans parseur d'arguments (pas de `args.lua`)
--
--  Usage:
--    ./bin/mimir --lua scripts/templates/template_pipeline_only.lua
--
--  (Optionnel via env)
--    MIMIR_DATASET=path/to/dataset.bin   -> lance un entraînement
--    MIMIR_EPOCHS=1                      -> epochs (défaut: 1)
--    MIMIR_LR=0.0003                     -> learning rate (défaut: 0.0003)
--    MIMIR_SAVE=checkpoints/run.safetensors -> sauvegarde
-- ═════════════════════════════════════════════════════════════=

local RUN_TAG = os.getenv("MIMIR_RUN") or "pipeline_only"

local function log(msg)
  if type(_G.log) == "function" then
    _G.log(msg)
  else
    print(msg)
  end
end

-- ---------------------------------------------------------------------------
-- Système: MemoryGuard + Allocator (recommandé avant build)
-- ---------------------------------------------------------------------------

if type(_G.Mimir) == "table" then
  if type(Mimir.MemoryGuard) == "table" and type(Mimir.MemoryGuard.setLimit) == "function" then
    pcall(Mimir.MemoryGuard.setLimit, 10) -- 10 GB
  end

  if type(Mimir.Allocator) == "table" and type(Mimir.Allocator.configure) == "function" then
    pcall(Mimir.Allocator.configure, {
      max_ram_gb = 10.0,
      enable_compression = true,
      swap_strategy = "lru",
    })
  end
end

-- ---------------------------------------------------------------------------
-- Pipeline
-- ---------------------------------------------------------------------------

local P = dofile("scripts/modules/pipeline_api.lua")

-- Config volontairement petite (rapide à builder)
local cfg = {
  vocab_size = 32000,
  seq_len = 64,
  d_model = 128,
  num_layers = 2,
  num_heads = 4,
  mlp_hidden = 256,
  dropout = 0.0,
  causal = true,
}

local pipe = P.Transformer(cfg)

-- Options "non filtrées" par Pipeline API: on peut les injecter après coup.
pipe.config.init = os.getenv("MIMIR_INIT") or "xavier"
pipe.config.seed = tonumber(os.getenv("MIMIR_SEED") or "") or 1337
pipe.config.dtype = os.getenv("MIMIR_DTYPE") -- si votre runtime supporte le dtype via pipeline

log("\n== Pipeline (only) ==")
log("run=" .. tostring(RUN_TAG))

local ok_build, params_or_err = pipe:build()
if ok_build == false then
  error("build() a échoué: " .. tostring(params_or_err))
end
log("✓ build() OK | params=" .. tostring(params_or_err))

-- Entraînement optionnel (si dataset fourni)
local dataset_path = os.getenv("MIMIR_DATASET")
if dataset_path and dataset_path ~= "" then
  local epochs = tonumber(os.getenv("MIMIR_EPOCHS") or "") or 1
  local lr = tonumber(os.getenv("MIMIR_LR") or "") or 0.0003
  log("\n▶ train(dataset=" .. tostring(dataset_path) .. ", epochs=" .. tostring(epochs) .. ", lr=" .. tostring(lr) .. ")")

  local ok_train, err_train = pipe:train(dataset_path, epochs, lr)
  if ok_train == false then
    error("train() a échoué: " .. tostring(err_train))
  end
end

-- Sauvegarde optionnelle
local save_path = os.getenv("MIMIR_SAVE")
if save_path and save_path ~= "" then
  log("\n💾 save(" .. tostring(save_path) .. ")")
  local ok_save, err_save = pipe:save(save_path)
  if ok_save == false then
    error("save() a échoué: " .. tostring(err_save))
  end
end

log("\n✓ Terminé")
