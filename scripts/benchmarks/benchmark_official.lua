#!/usr/bin/env mimir --lua
-- ============================================================================
--  Mímir Framework - OFFICIAL BENCHMARK (Init Speed + Memory Guard)
--  Objectif:
--    - Bench du boot système + construction des modèles
--    - Mesure séparation: build / allocate_params / init_weights
--    - Mesure mémoire: current + peak (MemoryGuard) + stats allocator/memory
--
-- Usage:
--   mimir --lua scripts/benchmarks/benchmark_official.lua
--   mimir --lua scripts/benchmarks/benchmark_official.lua --safe
--   mimir --lua scripts/benchmarks/benchmark_official.lua --full
--   mimir --lua scripts/benchmarks/benchmark_official.lua --extreme
--   mimir --lua scripts/benchmarks/benchmark_official.lua --iters 10
--   mimir --lua scripts/benchmarks/benchmark_official.lua --no-compress
-- ============================================================================

-- -----------------------------
-- Small helpers
-- -----------------------------
local function has(flag)
  for _, a in ipairs(arg or {}) do
    if a == flag then return true end
  end
  return false
end

local function get_arg_value(key, default)
  for i = 1, #(arg or {}) do
    if arg[i] == key and arg[i + 1] ~= nil then
      return arg[i + 1]
    end
  end
  return default
end

local function now_s()
  return os.clock()
end

local function fmt_ms(s)
  return string.format("%.3f ms", s * 1000.0)
end

local function fmt_s(s)
  return string.format("%.3f s", s)
end

local function safe_call(fn, ...)
  if fn == nil then return false, "missing fn" end
  return fn(...)
end

local function get_guard_stats()
  if not (Mimir and Mimir.MemoryGuard) then return nil end
  local fn = Mimir.MemoryGuard["getStats"] or Mimir.MemoryGuard["get_stats"]
  if type(fn) == "function" then return fn() end
  return nil
end

local function get_memory_stats()
  if not (Mimir and Mimir.Memory) then return nil end
  local fn = Mimir.Memory["getStats"] or Mimir.Memory["get_stats"]
  if type(fn) == "function" then return fn() end
  return nil
end

local function get_allocator_stats()
  if not (Mimir and Mimir.Allocator) then return nil end
  local fn = Mimir.Allocator["getStats"] or Mimir.Allocator["get_stats"]
  if type(fn) == "function" then return fn() end
  return nil
end

local function merge_into(dst, src)
  if type(dst) ~= "table" or type(src) ~= "table" then return dst end
  for k, v in pairs(src) do dst[k] = v end
  return dst
end

local function line()
  log(string.rep("─", 70))
end

local function section(title)
  log("\n" .. string.rep("=", 70))
  log("  " .. title)
  log(string.rep("=", 70))
end

local function mem_snapshot(tag)
  local gs = get_guard_stats()
  local ms = get_memory_stats()
  local as = get_allocator_stats()

  log("🧠 Memory Snapshot: " .. tag)

  if gs then
    log(string.format("  Guard: current=%.1f MB | peak=%.1f MB | limit=%.1f MB | usage=%.1f%%",
      gs["current_mb"] or gs["current"] or -1,
      gs["peak_mb"] or gs["peak"] or -1,
      gs["limit_mb"] or gs["limit"] or -1,
      gs["usage_percent"] or gs["usage"] or -1))
  else
    log("  Guard: (stats unavailable)")
  end

  if ms then
    log(string.format("  MemoryMgr: current=%.1f MB | peak=%.1f MB | usage=%.1f%%",
      ms.current_mb or -1, ms.peak_mb or -1, ms.usage_percent or -1))
  else
    -- fallback
    if Mimir.Memory and Mimir.Memory.get_usage then
      log(string.format("  MemoryMgr: usage=%.1f MB", Mimir.Memory.get_usage()))
    else
      log("  MemoryMgr: (stats unavailable)")
    end
  end

  if as then
    log(string.format("  Allocator: tensors=%d | loaded=%d",
      as["tensor_count"] or as["tensors"] or as["num_tensors"] or -1,
      as["loaded_count"] or as["loaded"] or as["loaded_tensors"] or -1))
  else
    log("  Allocator: (stats unavailable)")
  end
end

local function try_cleanup()
  -- Reset peaks if possible
  if Mimir.MemoryGuard and Mimir.MemoryGuard.reset then Mimir.MemoryGuard.reset() end

  -- Clear internal RAM caches if available
  if Mimir.Memory and Mimir.Memory.clear then
    local ok = Mimir.Memory.clear()
    if ok == false then
      -- ignore error string variants
    end
  end
end

-- -----------------------------
-- Config bench
-- -----------------------------
local function env_str(name, default)
  local v = os.getenv(name)
  if v == nil or v == "" then return default end
  return v
end

local function env_num(name, default)
  local v = os.getenv(name)
  if v == nil or v == "" then return default end
  local n = tonumber(v)
  if n == nil then return default end
  return n
end

-- Note: le binaire `mimir --lua ...` ne transmet pas toujours les args au script.
-- Donc on supporte aussi les variables d'environnement:
--   MIMIR_BENCH_MODE=safe|full|extreme
--   MIMIR_BENCH_ITERS=...
--   MIMIR_BENCH_RAM_GB=...
--   MIMIR_BENCH_SEQ=...
--   MIMIR_BENCH_VOCAB=...
--   MIMIR_BENCH_COMPRESS=0|1

local mode = env_str("MIMIR_BENCH_MODE", "safe")
if has("--safe") then mode = "safe" end
if has("--full") then mode = "full" end
if has("--extreme") then mode = "extreme" end

local iters = tonumber(get_arg_value("--iters", tostring(env_num("MIMIR_BENCH_ITERS", 3)))) or 3
if iters < 1 then iters = 1 end

local enable_compression = true
if has("--no-compress") then enable_compression = false end

local env_compress = os.getenv("MIMIR_BENCH_COMPRESS")
if env_compress == "0" or env_compress == "false" then enable_compression = false end
if env_compress == "1" or env_compress == "true" then enable_compression = true end

local max_ram_gb = tonumber(get_arg_value("--ram", tostring(env_num("MIMIR_BENCH_RAM_GB", 10)))) or 10.0
local seq_len = math.floor(tonumber(get_arg_value("--seq", tostring(env_num("MIMIR_BENCH_SEQ", 128)))) or 128)
local vocab_size = math.floor(tonumber(get_arg_value("--vocab", tostring(env_num("MIMIR_BENCH_VOCAB", 10000)))) or 10000)

-- Bench levels (Transformer only, car tu veux init/build speed + RAM)
local levels = {
  { name = "Warmup", layers = 2,  dim = 128,  heads = 4 },
  { name = "Small",  layers = 4,  dim = 256,  heads = 8 },
  { name = "Medium", layers = 6,  dim = 384,  heads = 8 },
  { name = "Large",  layers = 8,  dim = 512,  heads = 8 },
  { name = "XL",     layers = 12, dim = 768,  heads = 12 },
  { name = "XXL",    layers = 16, dim = 1024, heads = 16 },
}

if mode == "safe" then
  levels = {
    { name = "Warmup", layers = 2, dim = 128, heads = 4 },
    { name = "Small",  layers = 4, dim = 256, heads = 8 },
    { name = "Medium", layers = 6, dim = 384, heads = 8 },
    { name = "Large",  layers = 8, dim = 512, heads = 8 },
  }
elseif mode == "extreme" then
  levels = {
    { name = "Warmup", layers = 2,  dim = 128,  heads = 4 },
    { name = "Small",  layers = 4,  dim = 256,  heads = 8 },
    { name = "Medium", layers = 6,  dim = 384,  heads = 8 },
    { name = "Large",  layers = 8,  dim = 512,  heads = 8 },
    { name = "XL",     layers = 12, dim = 768,  heads = 12 },
    { name = "XXL",    layers = 16, dim = 1024, heads = 16 },
    { name = "ULTRA",  layers = 24, dim = 1536, heads = 24 },
  }
end

-- -----------------------------
-- Header
-- -----------------------------
log("╔═══════════════════════════════════════════════════════════════════╗")
log("║         Mímir OFFICIAL BENCHMARK - Init Speed & Memory           ║")
log("║     build / allocate_params / init_weights + peak MemoryGuard     ║")
log("╚═══════════════════════════════════════════════════════════════════╝")

log("")
log("📊 Configuration:")
log("  Mode:           " .. mode)
log("  Iterations:     " .. iters)
log("  Vocab size:     " .. vocab_size)
log("  Seq length:     " .. seq_len)
log(string.format("  RAM limit:      %.1f GB", max_ram_gb))
log("  Compression:    " .. (enable_compression and "ON (LZ4)" or "OFF"))

-- -----------------------------
-- Setup allocator / guard
-- -----------------------------
section("Phase 1: Runtime / Memory Setup")

if Mimir.MemoryGuard and Mimir.MemoryGuard.setLimit then
  local ok, err = Mimir.MemoryGuard.setLimit(max_ram_gb)
  if ok == false then
    log("❌ Mimir.MemoryGuard.setLimit failed: " .. tostring(err))
  else
    log(string.format("🛡️  MemoryGuard limit set to %.1f GB", max_ram_gb))
  end
else
  log("⚠️ Mimir.MemoryGuard.setLimit not available")
end

if Mimir.Allocator and Mimir.Allocator.configure then
  local ok, err = Mimir.Allocator.configure({
    max_ram_gb = max_ram_gb,
    enable_compression = enable_compression,
    -- valeurs safe par défaut
    swap_strategy = "lru",
  })
  if ok == false then
    log("❌ Mimir.Allocator.configure failed: " .. tostring(err))
    return
  end
  log("🚀 DynamicTensorAllocator configured")
else
  log("⚠️ Mimir.Allocator.configure not available (bench degraded)")
end

-- Optional hardware accel (si présent)
if Mimir.Model and Mimir.Model.hardware_caps and Mimir.Model.set_hardware then
  local hw = Mimir.Model.hardware_caps()
  log("\n🔧 Hardware Caps:")
  log(string.format("  • AVX2:  %s", hw.avx2 and "✓" or "✗"))
  log(string.format("  • FMA:   %s", hw.fma and "✓" or "✗"))
  log(string.format("  • F16C:  %s", hw.f16c and "✓" or "✗"))
  log(string.format("  • BMI2:  %s", hw.bmi2 and "✓" or "✗"))

  local ok = select(1, pcall(Mimir.Model.set_hardware, true))
  if not ok then pcall(Mimir.Model.set_hardware, "auto") end
  log("✓ Hardware acceleration requested")
end

mem_snapshot("after setup")

-- -----------------------------
-- Tokenizer ONCE (do not pollute model init)
-- -----------------------------
section("Phase 2: Tokenizer (One-time)")

local t0 = now_s()
local ok_tok = true
if Mimir.Tokenizer and Mimir.Tokenizer.create then
  ok_tok = Mimir.Tokenizer.create(vocab_size)
  log("Tokenizer created (vocab_max=" .. vocab_size .. ")")
else
  ok_tok = false
  log("⚠️ Mimir.Tokenizer.create not available")
end
local tok_time = now_s() - t0
log("⏱️ Tokenizer time: " .. fmt_s(tok_time))

if Mimir.Tokenizer and Mimir.Tokenizer.set_max_length then
  local ok, err = Mimir.Tokenizer.set_max_length(seq_len)
  if ok == false then
    log("⚠️ Mimir.Tokenizer.set_max_length failed: " .. tostring(err))
  end
end

mem_snapshot("after tokenizer")

-- -----------------------------
-- Bench loop
-- -----------------------------
section("Phase 3: Model Init Bench (Transformer)")

local results = {}

local function bench_one(level)
  line()
  log(string.format("🧪 Test: %s | %dL × %dD × %dH | iters=%d",
    level.name, level.layers, level.dim, level.heads, iters))

  local base_cfg = {}
  if Mimir.Architectures and Mimir.Architectures.default_config then
    local b = Mimir.Architectures.default_config("transformer")
    if type(b) == "table" then base_cfg = b end
  end

  local cfg = merge_into(base_cfg, {
    vocab_size = vocab_size,
    d_model = level.dim,
    num_layers = level.layers,
    num_heads = level.heads,
    mlp_hidden = level.dim * 4,
    output_dim = level.dim,
    seq_len = seq_len,
    padding_idx = 0,
    causal = false,
  })

  local times_build = 0.0
  local times_alloc = 0.0
  local times_init  = 0.0
  local params_last = 0

  -- reset peaks for this test (so peak reflects this test, not whole run)
  if Mimir.MemoryGuard and Mimir.MemoryGuard.reset then Mimir.MemoryGuard.reset() end

  for i = 1, iters do
    -- cleanup between iterations to reduce cache accumulation
    try_cleanup()

    -- ---- build (registry create) ----
    local tb = now_s()
    local ok, err = Mimir.Model.create("transformer", cfg)
    if ok == false then
      return false, "Mimir.Model.create failed: " .. tostring(err)
    end
    params_last = (Mimir.Model.total_params and Mimir.Model.total_params()) or params_last
    times_build = times_build + (now_s() - tb)

    -- ---- allocate ----
    local ta = now_s()
    local oka, erra = Mimir.Model.allocate_params()
    if oka == false then
      return false, "Mimir.Model.allocate_params failed: " .. tostring(erra)
    end
    times_alloc = times_alloc + (now_s() - ta)

    -- ---- init weights ----
    local ti = now_s()
    local oki, erri = Mimir.Model.init_weights("xavier", 42 + i)
    if oki == false then
      return false, "Mimir.Model.init_weights failed: " .. tostring(erri)
    end
    times_init = times_init + (now_s() - ti)
  end

  local gs = get_guard_stats()
  local peak_mb = gs and gs.peak_mb or -1
  local cur_mb  = gs and gs.current_mb or -1

  local avg_build = times_build / iters
  local avg_alloc = times_alloc / iters
  local avg_init  = times_init  / iters
  local avg_total = avg_build + avg_alloc + avg_init

  results[#results + 1] = {
    name = level.name,
    layers = level.layers,
    dim = level.dim,
    params = params_last,
    build_s = avg_build,
    alloc_s = avg_alloc,
    init_s = avg_init,
    total_s = avg_total,
    peak_mb = peak_mb,
    cur_mb = cur_mb
  }

  log("✅ OK")
  log("  Params:    " .. tostring(params_last))
  log("  build:     " .. fmt_ms(avg_build))
  log("  allocate:  " .. fmt_ms(avg_alloc))
  log("  init:      " .. fmt_ms(avg_init))
  log("  total:     " .. fmt_ms(avg_total))
  if peak_mb >= 0 then
    log(string.format("  Guard mem: current=%.1f MB | peak=%.1f MB", cur_mb, peak_mb))
  end

  return true, nil
end

local bench_start = now_s()
for _, lvl in ipairs(levels) do
  local ok, err = bench_one(lvl)
  if not ok then
    log("❌ Bench failed on " .. lvl.name .. ": " .. tostring(err))
    mem_snapshot("on failure")
    break
  end
end
local bench_total = now_s() - bench_start

-- -----------------------------
-- Summary
-- -----------------------------
section("SUMMARY")

log("Test              Layers  Dim      Params     build(ms)  alloc(ms)  init(ms)   total(ms)  peak(MB)")
log(string.rep("-", 94))

for _, r in ipairs(results) do
  log(string.format("%-16s  %5d  %4d  %10s  %8.1f  %8.1f  %8.1f  %9.1f  %7.1f",
    r.name,
    r.layers,
    r.dim,
    tostring(r.params),
    r.build_s * 1000.0,
    r.alloc_s * 1000.0,
    r.init_s * 1000.0,
    r.total_s * 1000.0,
    r.peak_mb
  ))
end

log(string.rep("-", 94))
log("Total bench time: " .. fmt_s(bench_total))

mem_snapshot("end")
log("\n✅ Official benchmark finished.")
