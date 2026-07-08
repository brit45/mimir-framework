#!/usr/bin/env mimir --lua
local Help = dofile("scripts/modules/help_cli.lua")
Help.auto_exit_help()

-- Smoke test: force disk spill into .mimir-spill, then verify cleanup-on-exit.
-- Goal: create spill files during execution (under memory pressure) then exit normally.

local function now_s() return os.clock() end
local function fmt_s(s) return string.format("%.3f s", s) end

local function count_spill_files()
  local p = io.popen("ls -1 .mimir-spill 2>/dev/null | wc -l")
  if not p then return -1 end
  local out = p:read("*a")
  p:close()
  return tonumber(out) or -1
end

local function apply_dtype(cfg)
  local dtype = (type(cfg) == "table" and cfg.dtype) or os.getenv("MIMIR_DTYPE")
  if dtype == nil then return true end
  if type(Mimir) ~= "table" or type(Mimir.model) ~= "table" or type(Mimir.model.dtype) ~= "function" then
    return true
  end
  local ok, dt_or_err = Mimir.model.dtype(dtype)
  if ok == false then
    log("❌ Mimir.model.dtype failed: " .. tostring(dt_or_err))
    return false
  end
  return true
end

log("\n=== spill_cleanup_smoke.lua ===")

-- Use a small limit to reliably trigger eviction.
local max_ram_gb = 0.5
local enable_compression = true
local vocab_size = 10000
local seq_len = 128

if Mimir.MemoryGuard and Mimir.MemoryGuard.setLimit then
  local ok, err = Mimir.MemoryGuard.setLimit(max_ram_gb)
  if ok == false then
    log("❌ Mimir.MemoryGuard.setLimit failed: " .. tostring(err))
    return
  end
  log(string.format("🛡️  MemoryGuard limit set to %.1f GB", max_ram_gb))
else
  log("⚠️ Mimir.MemoryGuard.setLimit not available")
end

if Mimir.Allocator and Mimir.Allocator.configure then
  local ok, err = Mimir.Allocator.configure({
    max_ram_gb = max_ram_gb,
    enable_compression = enable_compression,
    swap_strategy = "lru",
  })
  if ok == false then
    log("❌ Mimir.Allocator.configure failed: " .. tostring(err))
    return
  end
else
  log("⚠️ Mimir.Allocator.configure not available")
end

if Mimir.Tokenizer and Mimir.Tokenizer.create then
  Mimir.Tokenizer.create(vocab_size)
end
if Mimir.Tokenizer and Mimir.Tokenizer.set_max_length then
  Mimir.Tokenizer.set_max_length(seq_len)
end

local level = { name = "Small", layers = 4, dim = 256, heads = 8 }
local base_cfg = {}
if Mimir.Architectures and Mimir.Architectures.default_config then
  local b = Mimir.Architectures.default_config("transformer")
  if type(b) == "table" then base_cfg = b end
end

local cfg = base_cfg
cfg.vocab_size = vocab_size
cfg.d_model = level.dim
cfg.num_layers = level.layers
cfg.num_heads = level.heads
cfg.mlp_hidden = level.dim * 4
cfg.output_dim = level.dim
cfg.seq_len = seq_len
cfg.padding_idx = 0
cfg.causal = false

log(string.format("🧪 Building transformer %s | %dL × %dD × %dH", level.name, level.layers, level.dim, level.heads))

local t0 = now_s()
local ok, err = Mimir.Model.create("transformer", cfg)
if ok == false then
  log("❌ Mimir.Model.create failed: " .. tostring(err))
  return
end

if apply_dtype(cfg) == false then
  return
end

local oka, erra = Mimir.Model.allocate_params()
if oka == false then
  log("❌ Mimir.Model.allocate_params failed: " .. tostring(erra))
  return
end

local oki, erri = Mimir.Model.init_weights("xavier", 42)
if oki == false then
  log("❌ Mimir.Model.init_weights failed: " .. tostring(erri))
  return
end

log("⏱️ Completed in " .. fmt_s(now_s() - t0))

local n = count_spill_files()
if n >= 0 then
  log(string.format("📁 .mimir-spill files during run: %d", n))
else
  log("📁 .mimir-spill files during run: (count unavailable)")
end

log("✅ Smoke test done; exiting normally (atexit cleanup should empty .mimir-spill).")
