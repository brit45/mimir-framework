#!/usr/bin/env mimir --lua
-- ================================================================
-- Mímir Benchmark Script - CPU Performance Tests
-- (Updated for registry-based architectures + int-token forward)
-- ================================================================

-- ================================================================
-- Runtime setup (template)
-- ================================================================
local function safe_call(fn, ...)
    if type(fn) ~= "function" then return false, "missing fn" end
    return fn(...)
end

local function configure_runtime(limit_gb)
    limit_gb = limit_gb or 10.0

    if Mimir and Mimir.MemoryGuard and Mimir.MemoryGuard.setLimit then
        local ok, err = safe_call(Mimir.MemoryGuard.setLimit, limit_gb)
        if ok == false then log("⚠️ MemoryGuard.setLimit failed: " .. tostring(err)) end
    end

    if Mimir and Mimir.Allocator and Mimir.Allocator.configure then
        local ok, err = safe_call(Mimir.Allocator.configure, {
            max_ram_gb = limit_gb,
            enable_compression = true,
            swap_strategy = "lru",
        })
        if ok == false then log("⚠️ Allocator.configure failed: " .. tostring(err)) end
    end

    if Mimir and Mimir.Model and Mimir.Model.set_hardware then
        -- Selon versions: bool (true/false) ou string ("auto"/"cpu").
        local ok = select(1, pcall(Mimir.Model.set_hardware, true))
        if not ok then pcall(Mimir.Model.set_hardware, "auto") end
    end
end

configure_runtime(10.0)

log("╔═══════════════════════════════════════════════════════════════════╗")
log("║           Mímir Framework - Suite de Benchmarks (v2.3+)          ║")
log("║                    CPU-Only Deep Learning                         ║")
log("╚═══════════════════════════════════════════════════════════════════╝")

-- Configuration
local config = {
    vocab_size = 10000,
    d_model_small = 128,
    d_model_medium = 256,
    d_model_large = 512,
    num_layers_small = 2,
    num_layers_medium = 4,
    num_layers_large = 8,
    seq_len = 128
}

-- Parse mode
-- Note: le binaire `mimir --lua ...` ne transmet pas toujours les args au script.
-- Donc on supporte aussi `MIMIR_BENCH_MODE=quick|full`.
local mode = os.getenv("MIMIR_BENCH_MODE") or "standard"

if arg and #arg > 0 then
    for i = 1, #arg do
        if arg[i] == "--quick" then
            mode = "quick"
        elseif arg[i] == "--full" then
            mode = "full"
        end
    end
end

if mode == "quick" then
    config.d_model_medium = 128
    config.num_layers_medium = 2
elseif mode == "full" then
    config.vocab_size = 20000
else
    mode = "standard"
end

log("\n📊 Configuration: " .. mode)
log("  Vocab size:     " .. config.vocab_size)
log("  Seq length:     " .. config.seq_len)

local function timer(name, func)
    log("\n⏱️  " .. name)
    local start = os.clock()
    func()
    local elapsed = os.clock() - start
    log("   ✓ Temps: " .. string.format("%.3f", elapsed) .. "s")
    return elapsed
end

local Tokenizer = (Mimir and Mimir.Tokenizer) or tokenizer
local function tokenizer_create(vocab)
    if Tokenizer and Tokenizer.create then
        local ok, err = safe_call(Tokenizer.create, vocab)
        if ok == false then error("Tokenizer.create failed: " .. tostring(err)) end
        return true
    end
    error("Tokenizer.create not available")
end

-- ================================================================
-- Benchmark 1: Tokenizer
-- ================================================================
log("\n" .. string.rep("=", 70))
log("  Benchmark 1: Tokenizer Performance")
log(string.rep("=", 70))

timer("Création tokenizer (" .. config.vocab_size .. " tokens)", function()
    tokenizer_create(config.vocab_size)
end)

-- Helper: Transformer config (registry)
local function make_transformer_cfg(d_model, num_layers, num_heads)
    return {
        vocab_size = config.vocab_size,
        d_model = d_model,
        num_layers = num_layers,
        num_heads = num_heads,
        mlp_hidden = d_model * 4,
        output_dim = d_model,
        seq_len = config.seq_len,
        padding_idx = 0,
        causal = false
    }
end

-- ================================================================
-- Benchmark 2: Model Creation - Small
-- ================================================================
log("\n" .. string.rep("=", 70))
log("  Benchmark 2: Création Modèle (Petit)")
log(string.rep("=", 70))

local small_config = make_transformer_cfg(config.d_model_small, config.num_layers_small, 4)

timer("Transformer " .. config.num_layers_small .. "L x " .. config.d_model_small .. "D", function()
    tokenizer_create(config.vocab_size)

    local cfg = small_config
    if Mimir and Mimir.Architectures and Mimir.Architectures.default_config then
        local base = Mimir.Architectures.default_config("transformer")
        if type(base) == "table" then
            for k, v in pairs(cfg) do base[k] = v end
            cfg = base
        end
    end

    Mimir.Model.create("transformer", cfg)
    Mimir.Model.allocate_params()
    Mimir.Model.init_weights("xavier", 42)
    log("   📊 Paramètres: " .. tostring(Mimir.Model.total_params()))
end)

-- ================================================================
-- Benchmark 3: Model Creation - Medium
-- ================================================================
log("\n" .. string.rep("=", 70))
log("  Benchmark 3: Création Modèle (Moyen)")
log(string.rep("=", 70))

local medium_config = make_transformer_cfg(config.d_model_medium, config.num_layers_medium, 8)

timer("Transformer " .. config.num_layers_medium .. "L x " .. config.d_model_medium .. "D", function()
    tokenizer_create(config.vocab_size)

    local cfg = medium_config
    if Mimir and Mimir.Architectures and Mimir.Architectures.default_config then
        local base = Mimir.Architectures.default_config("transformer")
        if type(base) == "table" then
            for k, v in pairs(cfg) do base[k] = v end
            cfg = base
        end
    end

    Mimir.Model.create("transformer", cfg)
    Mimir.Model.allocate_params()
    Mimir.Model.init_weights("xavier", 42)
    log("   📊 Paramètres: " .. tostring(Mimir.Model.total_params()))
end)

-- ================================================================
-- Benchmark 4: Model Creation - Large
-- ================================================================
if mode ~= "quick" then
    log("\n" .. string.rep("=", 70))
    log("  Benchmark 4: Création Modèle (Grand)")
    log(string.rep("=", 70))

    local large_config = make_transformer_cfg(config.d_model_large, config.num_layers_large, 16)

    timer("Transformer " .. config.num_layers_large .. "L x " .. config.d_model_large .. "D", function()
        tokenizer_create(config.vocab_size)

        local cfg = large_config
        if Mimir and Mimir.Architectures and Mimir.Architectures.default_config then
            local base = Mimir.Architectures.default_config("transformer")
            if type(base) == "table" then
                for k, v in pairs(cfg) do base[k] = v end
                cfg = base
            end
        end

        Mimir.Model.create("transformer", cfg)
        Mimir.Model.allocate_params()
        Mimir.Model.init_weights("xavier", 42)
        log("   📊 Paramètres: " .. tostring(Mimir.Model.total_params()))
    end)
end

-- ================================================================
-- Benchmark 5: Causal vs Non-causal
-- ================================================================
log("\n" .. string.rep("=", 70))
log("  Benchmark 5: Transformer causal vs non-causal")
log(string.rep("=", 70))

timer("Transformer non-causal", function()
    tokenizer_create(config.vocab_size)
    local cfg = make_transformer_cfg(config.d_model_medium, config.num_layers_medium, 8)
    cfg.causal = false
    Mimir.Model.create("transformer", cfg)
    Mimir.Model.allocate_params()
    Mimir.Model.init_weights("xavier", 42)
    log("   📊 Paramètres: " .. tostring(Mimir.Model.total_params()))
end)

timer("Transformer causal", function()
    tokenizer_create(config.vocab_size)
    local cfg = make_transformer_cfg(config.d_model_medium, config.num_layers_medium, 8)
    cfg.causal = true
    Mimir.Model.create("transformer", cfg)
    Mimir.Model.allocate_params()
    Mimir.Model.init_weights("xavier", 42)
    log("   📊 Paramètres: " .. tostring(Mimir.Model.total_params()))
end)

-- ================================================================
-- Benchmark 6: Sérialisation
-- ================================================================
log("\n" .. string.rep("=", 70))
log("  Benchmark 6: Sérialisation (Save/Load)")
log(string.rep("=", 70))

local checkpoint_path = "/tmp/mimir_benchmark_checkpoint"
local checkpoint_path_st = checkpoint_path .. ".safetensors"

tokenizer_create(config.vocab_size)
Mimir.Model.create("transformer", small_config)
Mimir.Model.allocate_params()
Mimir.Model.init_weights("xavier", 42)

timer("Save checkpoint", function()
    Mimir.Serialization.save(checkpoint_path_st, "safetensors")
end)

local size_cmd = "du -sh " .. checkpoint_path_st .. " 2>/dev/null | cut -f1"
local handle = io.popen(size_cmd)
if handle then
    local out = handle:read("*a") or ""
    handle:close()
    local size = out:gsub("%s+", "")
    if size ~= "" then log("   📦 Taille: " .. size) end
end

timer("Load checkpoint", function()
    tokenizer_create(config.vocab_size)
    Mimir.Model.create("transformer", small_config)
    Mimir.Model.allocate_params()
    Mimir.Serialization.load(checkpoint_path_st)
end)

pcall(os.remove, checkpoint_path_st)

-- ================================================================
-- Benchmark 7: Estimation Mémoire
-- ================================================================
log("\n" .. string.rep("=", 70))
log("  Benchmark 7: Estimation Mémoire par Configuration")
log(string.rep("=", 70))

local function estimate_memory(layers, dim, vocab)
    -- Estimation grossière: params * 4 bytes (float32)
    local params_per_layer = dim * dim * 4  -- attention (approx)
    params_per_layer = params_per_layer + (dim * dim * 8)  -- ffn (approx)
    local total_params = params_per_layer * layers
    total_params = total_params + (vocab * dim)  -- embedding
    local memory_mb = (total_params * 4) / (1024 * 1024)
    return memory_mb, total_params
end

log("\nConfiguration          Layers  Dim    Mémoire    Paramètres")
log(string.rep("-", 70))

local configs_to_test = {
    {"Tiny", 2, 64, config.vocab_size},
    {"Small", 2, 128, config.vocab_size},
    {"Medium", 4, 256, config.vocab_size},
    {"Large", 8, 512, config.vocab_size},
    {"XL", 12, 768, config.vocab_size}
}

for _, cfg in ipairs(configs_to_test) do
    local name, layers, dim, vocab = cfg[1], cfg[2], cfg[3], cfg[4]
    local mem_mb, params = estimate_memory(layers, dim, vocab)
    log(string.format("%-20s %4d    %4d   %7.1f MB  %8.1f M",
                     name, layers, dim, mem_mb, params / 1000000))
end

-- ================================================================
-- Résumé
-- ================================================================
log("\n" .. string.rep("=", 70))
log("  Résumé des Benchmarks")
log(string.rep("=", 70))

log("\n✅ Benchmarks terminés avec succès!")
log("🚀 Mode: " .. mode)
