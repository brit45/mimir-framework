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

local function line()
    log(string.rep("─", 70))
end

local function section(title)
    log("\n" .. string.rep("=", 70))
    log("  " .. title)
    log(string.rep("=", 70))
end

local function mem_snapshot(tag)
    local gs = Mimir.MemoryGuard and Mimir.MemoryGuard.getStats and Mimir.MemoryGuard.getStats() or nil
    local ms = Mimir.Memory and Mimir.Memory.get_stats and Mimir.Memory.get_stats() or nil
    local as = Mimir.Allocator and Mimir.Allocator.get_stats and Mimir.Allocator.get_stats() or nil

    log("🧠 Memory Snapshot: " .. tag)

    if gs then
        log(string.format("  Guard: current=%.1f MB | peak=%.1f MB | limit=%.1f MB | usage=%.1f%%",
            gs.current_mb or -1, gs.peak_mb or -1, gs.limit_mb or -1, gs.usage_percent or -1))
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
            as.tensor_count or -1, as.loaded_count or -1))
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
local mode = "default"
if has("--safe") then mode = "safe" end
if has("--full") then mode = "full" end
if has("--extreme") then mode = "extreme" end

local iters = tonumber(get_arg_value("--iters", "5")) or 5
if iters < 1 then iters = 1 end

local enable_compression = true
if has("--no-compress") then enable_compression = false end

local max_ram_gb = tonumber(get_arg_value("--ram", "10")) or 10.0
local max_seq_len = tonumber(get_arg_value("--seq", "512")) or 512
local vocab_size = tonumber(get_arg_value("--vocab", "50000")) or 50000

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
log("  Max seq length: " .. max_seq_len)
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
        -- valeurs safe par défaut (tu peux ajuster si tu exposes plus)
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

    if hw.avx2 or hw.fma then
        Mimir.Model.set_hardware("auto")
        log("✓ Hardware acceleration enabled")
    else
        log("ℹ️  Hardware acceleration not enabled")
    end
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
    local ok, err = Mimir.Tokenizer.set_max_length(max_seq_len)
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

    local cfg = {
        vocab_size = vocab_size,
        embed_dim = level.dim,
        num_layers = level.layers,
        num_heads = level.heads,
        d_ff = level.dim * 4,
        max_seq_len = max_seq_len,
        dropout = 0.0
    }

    local times_build = 0.0
    local times_alloc = 0.0
    local times_init  = 0.0
    local params_last = 0

    -- reset peaks for this test (so peak reflects this test, not whole run)
    if Mimir.MemoryGuard and Mimir.MemoryGuard.reset then Mimir.MemoryGuard.reset() end

    for i = 1, iters do
        -- cleanup between iterations to reduce cache accumulation
        try_cleanup()

        -- unique-ish name to avoid confusion in logs; the engine may still reuse internals
        local model_name = "transformer"

        -- ---- build ----
        local tb = now_s()
        local ok, err = Mimir.Model.create(model_name, cfg)
        if ok == false then
            return false, "Mimir.Model.create failed: " .. tostring(err)
        end
        local okb, params, errb = Mimir.Model.build()
        if okb == false then
            return false, "Mimir.Model.build failed: " .. tostring(errb)
        end
        params_last = params or params_last
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

    local gs = Mimir.MemoryGuard and Mimir.MemoryGuard.getStats and Mimir.MemoryGuard.getStats() or nil
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
