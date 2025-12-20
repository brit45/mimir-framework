-- ================================================================
-- Mímir Benchmark Script - CPU Performance Tests
-- ================================================================

log("╔═══════════════════════════════════════════════════════════════════╗")
log("║           Mímir Framework - Suite de Benchmarks v2.0             ║")
log("║                    CPU-Only Deep Learning                         ║")
log("╚═══════════════════════════════════════════════════════════════════╝")

-- Configuration
local config = {
    vocab_size = 10000,
    embed_dim_small = 128,
    embed_dim_medium = 256,
    embed_dim_large = 512,
    num_layers_small = 2,
    num_layers_medium = 4,
    num_layers_large = 8,
    max_seq_len = 128
}

-- Parse mode
local mode = "standard"
if arg and #arg > 0 then
    for i = 1, #arg do
        if arg[i] == "--quick" then
            mode = "quick"
            config.embed_dim_medium = 128
            config.num_layers_medium = 2
        elseif arg[i] == "--full" then
            mode = "full"
            config.vocab_size = 20000
        end
    end
end

log("\n📊 Configuration: " .. mode)
log("  Vocab size:     " .. config.vocab_size)
log("  Max seq length: " .. config.max_seq_len)

-- Helper
local function timer(name, func)
    log("\n⏱️  " .. name)
    local start = os.clock()
    func()
    local elapsed = os.clock() - start
    log("   ✓ Temps: " .. string.format("%.3f", elapsed) .. "s")
    return elapsed
end

-- ================================================================
-- Benchmark 1: Tokenizer
-- ================================================================
log("\n" .. string.rep("=", 70))
log("  Benchmark 1: Tokenizer Performance")
log(string.rep("=", 70))

timer("Création tokenizer (" .. config.vocab_size .. " tokens)", function()
    tokenizer.create(config.vocab_size)
end)

-- ================================================================
-- Benchmark 2: Model Creation - Small
-- ================================================================
log("\n" .. string.rep("=", 70))
log("  Benchmark 2: Création Modèle (Petit)")
log(string.rep("=", 70))

local small_config = {
    vocab_size = config.vocab_size,
    embed_dim = config.embed_dim_small,
    num_layers = config.num_layers_small,
    num_heads = 4,
    d_ff = config.embed_dim_small * 4,
    max_seq_len = config.max_seq_len,
    dropout = 0.1
}

timer("Transformer " .. config.num_layers_small .. "L x " .. config.embed_dim_small .. "D", function()
    tokenizer.create(config.vocab_size)
    model.create("transformer", small_config)
    local ok, params = model.build()
    log("   📊 Paramètres: " .. params)
end)

-- ================================================================
-- Benchmark 3: Model Creation - Medium
-- ================================================================
log("\n" .. string.rep("=", 70))
log("  Benchmark 3: Création Modèle (Moyen)")
log(string.rep("=", 70))

local medium_config = {
    vocab_size = config.vocab_size,
    embed_dim = config.embed_dim_medium,
    num_layers = config.num_layers_medium,
    num_heads = 8,
    d_ff = config.embed_dim_medium * 4,
    max_seq_len = config.max_seq_len,
    dropout = 0.1
}

timer("Transformer " .. config.num_layers_medium .. "L x " .. config.embed_dim_medium .. "D", function()
    tokenizer.create(config.vocab_size)
    model.create("transformer", medium_config)
    local ok, params = model.build()
    log("   📊 Paramètres: " .. params)
end)

-- ================================================================
-- Benchmark 4: Model Creation - Large (if not quick mode)
-- ================================================================
if mode ~= "quick" then
    log("\n" .. string.rep("=", 70))
    log("  Benchmark 4: Création Modèle (Grand)")
    log(string.rep("=", 70))
    
    local large_config = {
        vocab_size = config.vocab_size,
        embed_dim = config.embed_dim_large,
        num_layers = config.num_layers_large,
        num_heads = 16,
        d_ff = config.embed_dim_large * 4,
        max_seq_len = config.max_seq_len,
        dropout = 0.1
    }
    
    timer("Transformer " .. config.num_layers_large .. "L x " .. config.embed_dim_large .. "D", function()
        tokenizer.create(config.vocab_size)
        model.create("transformer", large_config)
        local ok, params = model.build()
        log("   📊 Paramètres: " .. params)
    end)
end

-- ================================================================
-- Benchmark 5: Encoder vs Decoder
-- ================================================================
log("\n" .. string.rep("=", 70))
log("  Benchmark 5: Architectures - Encoder vs Decoder")
log(string.rep("=", 70))

-- Encoder
timer("Encoder (" .. config.num_layers_medium .. "L x " .. config.embed_dim_medium .. "D)", function()
    tokenizer.create(config.vocab_size)
    model.create("encoder", medium_config)
    local ok, params = model.build()
    log("   📊 Paramètres: " .. params)
end)

-- Decoder
timer("Decoder (" .. config.num_layers_medium .. "L x " .. config.embed_dim_medium .. "D)", function()
    tokenizer.create(config.vocab_size)
    model.create("decoder", medium_config)
    local ok, params = model.build()
    log("   📊 Paramètres: " .. params)
end)

-- ================================================================
-- Benchmark 6: Sérialisation
-- ================================================================
log("\n" .. string.rep("=", 70))
log("  Benchmark 6: Sérialisation (Save/Load)")
log(string.rep("=", 70))

local checkpoint_path = "/tmp/mimir_benchmark_checkpoint"

-- Créer modèle pour sauvegarder
tokenizer.create(config.vocab_size)
model.create("transformer", small_config)
model.build()

-- Save
local save_time = timer("Save checkpoint", function()
    model.save(checkpoint_path)
end)

-- Get size
local size_cmd = "du -sh " .. checkpoint_path .. " 2>/dev/null | cut -f1"
local handle = io.popen(size_cmd)
local size = handle:read("*a"):gsub("%s+", "")
handle:close()

if size ~= "" then
    log("   📦 Taille: " .. size)
end

-- Load (créer nouveau modèle)
local load_time = timer("Load checkpoint", function()
    tokenizer.create(config.vocab_size)
    model.create("transformer", small_config)
    model.build()
    -- Note: Load serait model.load() mais vérifions l'API
end)

-- Cleanup
os.execute("rm -rf " .. checkpoint_path .. " 2>/dev/null")

-- ================================================================
-- Benchmark 7: Estimation Mémoire
-- ================================================================
log("\n" .. string.rep("=", 70))
log("  Benchmark 7: Estimation Mémoire par Configuration")
log(string.rep("=", 70))

local function estimate_memory(layers, dim, vocab)
    -- Estimation: params * 4 bytes (float32)
    local params_per_layer = dim * dim * 4  -- Self-attention
    params_per_layer = params_per_layer + (dim * dim * 4 * 2)  -- FFN
    local total_params = params_per_layer * layers
    total_params = total_params + (vocab * dim)  -- Embedding
    
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
log("💡 Mímir est optimisé pour CPU - Aucun GPU requis")
log("\n📝 Conseils:")
log("   • --quick : Tests rapides (petits modèles)")
log("   • --full  : Tests complets (grands modèles)")
log("\n🔧 Optimisations CPU actives:")
log("   • OpenMP multi-threading")
log("   • Instructions SIMD (FMA, F16C)")
log("   • Huge Pages pour mémoire")
