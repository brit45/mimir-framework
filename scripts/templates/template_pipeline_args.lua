#!/usr/bin/env lua
-- ══════════════════════════════════════════════════════════════
--  TEMPLATE SCRIPT - Args + Pipeline (Mímir)
--
--  Objectif:
--    - Utiliser `scripts/modules/args.lua` (flags + overrides)
--    - Piloter un pipeline via `scripts/modules/pipeline_api.lua`
--
--  Usage:
--    ./bin/mimir --lua scripts/templates/template_pipeline_args.lua -- \
--      --dataset dataset_2/tokenizer.bin \
--      --epochs 1 --lr 0.0003 \
--      --save checkpoints/my_run.safetensors \
--      --d-model 256 --layers 4 --heads 8 --seq-len 128 \
--      --override mlp_hidden=1024 \
--      --viz
--
--  Notes:
--    - `--override a.b.c=value` supporte bool/number/null/string.
--    - `args.lua` supporte `--no-flag` (ex: `--no-train`).
-- ═════════════════════════════════════════════════════════════=

local Args = dofile("scripts/modules/args.lua")
local P = dofile("scripts/modules/pipeline_api.lua")

local opts = Args.parse(arg) or {}

local function log(msg)
  if type(_G.log) == "function" then
    _G.log(msg)
  else
    print(msg)
  end
end

-- ---------------------------------------------------------------------------
-- Système: MemoryGuard + Allocator (optionnels)
-- ---------------------------------------------------------------------------

do
  local mem_gb = Args.get_num(opts, "mem-gb", 10)
  local alloc_gb = Args.get_num(opts, "alloc-gb", mem_gb)
  local compression = Args.get_bool(opts, "compression", true)

  if type(_G.Mimir) == "table" then
    if type(Mimir.MemoryGuard) == "table" and type(Mimir.MemoryGuard.setLimit) == "function" then
      pcall(Mimir.MemoryGuard.setLimit, mem_gb)
    end

    if type(Mimir.Allocator) == "table" and type(Mimir.Allocator.configure) == "function" then
      pcall(Mimir.Allocator.configure, {
        max_ram_gb = alloc_gb,
        enable_compression = compression,
        swap_strategy = "lru",
      })
    end
  end
end

-- ---------------------------------------------------------------------------
-- Config pipeline (Transformer) + overrides
-- ---------------------------------------------------------------------------

local cfg = {
  vocab_size = Args.get_int(opts, "vocab", 32000),
  seq_len = Args.get_int(opts, "seq-len", 64),
  d_model = Args.get_int(opts, "d-model", 128),
  num_layers = Args.get_int(opts, "layers", 2),
  num_heads = Args.get_int(opts, "heads", 4),
  mlp_hidden = Args.get_int(opts, "mlp", 256),
  dropout = Args.get_num(opts, "dropout", 0.0),
  causal = Args.get_bool(opts, "causal", true),
}

cfg = Args.apply_overrides(cfg, opts)

local pipe = P.Transformer(cfg)

-- Options extra (pas dans la whitelist `allowed` du pipeline API)
pipe.config.init = Args.get_str(opts, "init", "xavier")
pipe.config.seed = Args.get_int(opts, "seed", 1337)
pipe.config.dtype = Args.get_str(opts, "dtype", nil)

local run_tag = Args.get_str(opts, "run", "pipeline_args")

log("\n== Pipeline (args) ==")
log("run=" .. tostring(run_tag))

local ok_build, params_or_err = pipe:build()
if ok_build == false then
  error("build() a échoué: " .. tostring(params_or_err))
end
log("✓ build() OK | params=" .. tostring(params_or_err))

-- ---------------------------------------------------------------------------
-- Train (optionnel)
-- ---------------------------------------------------------------------------

local do_train = Args.get_bool(opts, "train", true)
if do_train then
  local dataset_path = Args.get_str(opts, "dataset", nil)
  if dataset_path == nil or dataset_path == "" then
    log("ℹ️  Pas de --dataset => skip train() (utilisez --no-train pour enlever ce message)")
  else
    local epochs = Args.get_int(opts, "epochs", 1)
    local lr = Args.get_num(opts, "lr", 0.0003)
    log("\n▶ train(dataset=" .. tostring(dataset_path) .. ", epochs=" .. tostring(epochs) .. ", lr=" .. tostring(lr) .. ")")

    local ok_train, err_train = pipe:train(dataset_path, epochs, lr)
    if ok_train == false then
      error("train() a échoué: " .. tostring(err_train))
    end
  end
end

-- ---------------------------------------------------------------------------
-- Save (optionnel)
-- ---------------------------------------------------------------------------

local save_path = Args.get_str(opts, "save", nil)
if save_path ~= nil and save_path ~= "" then
  log("\n💾 save(" .. tostring(save_path) .. ")")
  local ok_save, err_save = pipe:save(save_path)
  if ok_save == false then
    error("save() a échoué: " .. tostring(err_save))
  end
end

log("\n✓ Terminé")
