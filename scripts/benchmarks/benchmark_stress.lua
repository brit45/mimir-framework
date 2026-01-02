-- ================================================================
-- Mímir STRESS BENCHMARK - Force la Machine au Maximum
-- Test intensif des capacités CPU et mémoire
-- ================================================================

log("╔═══════════════════════════════════════════════════════════════════╗")
log("║         Mímir STRESS BENCHMARK - CPU Performance Test            ║")
log("║                  ⚠️  TEST INTENSIF - FORCE LA MACHINE            ║")
log("╚═══════════════════════════════════════════════════════════════════╝")

-- Configuration INTENSE
local config = {
    -- Grands modèles
    vocab_size = 50000,
    
    -- Tests progressifs
    tests = {
        {name = "Warmup", layers = 2, dim = 128, heads = 4, iterations = 5},
        {name = "Small", layers = 4, dim = 256, heads = 8, iterations = 10},
        {name = "Medium", layers = 6, dim = 384, heads = 12, iterations = 15},
        {name = "Large", layers = 8, dim = 512, heads = 16, iterations = 20},
        {name = "XL", layers = 12, dim = 768, heads = 16, iterations = 25},
        {name = "XXL", layers = 16, dim = 1024, heads = 16, iterations = 30}
    },
    
    max_seq_len = 512,
    dropout = 0.1
}

-- Parse arguments
local skip_xxl = false
if arg and #arg > 0 then
    for i = 1, #arg do
        if arg[i] == "--safe" then
            log("\n🛡️  Mode SAFE activé - Limité à Large (8 layers)")
            config.tests = {
                config.tests[1],  -- Warmup
                config.tests[2],  -- Small
                config.tests[3],  -- Medium
                config.tests[4]   -- Large
            }
        elseif arg[i] == "--extreme" then
            log("\n🔥 Mode EXTREME activé - Ajout de modèles GIGANTESQUES")
            table.insert(config.tests, {name = "ULTRA", layers = 24, dim = 1536, heads = 24, iterations = 40})
        end
    end
end

log("\n📊 Configuration du stress test:")
log("  Vocab:          " .. config.vocab_size)
log("  Max seq length: " .. config.max_seq_len)
log("  Niveaux:        " .. #config.tests)

-- Helper pour mesurer temps
local function timer(name, func)
    log("\n⏱️  " .. name)
    local start = os.clock()
    local success, result = pcall(func)
    local elapsed = os.clock() - start
    
    if success then
        log("   ✓ Temps: " .. string.format("%.3f", elapsed) .. "s")
        return elapsed, result
    else
        log("   ❌ ERREUR: " .. tostring(result))
        return nil, nil
    end
end

-- Helper pour estimer mémoire
local function estimate_memory(layers, dim, vocab)
    local params_per_layer = dim * dim * 4 + (dim * dim * 4 * 2)
    local total_params = params_per_layer * layers + (vocab * dim)
    local memory_mb = (total_params * 4) / (1024 * 1024)
    return memory_mb, total_params
end

-- Helper pour monitorer système
local function get_system_load()
    local handle = io.popen("top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | cut -d'%' -f1")
    local cpu = handle:read("*a"):gsub("%s+", "")
    handle:close()
    
    handle = io.popen("free -m | awk 'NR==2{print $3}'")
    local mem_used = handle:read("*a"):gsub("%s+", "")
    handle:close()
    
    handle = io.popen("free -m | awk 'NR==2{print $2}'")
    local mem_total = handle:read("*a"):gsub("%s+", "")
    handle:close()
    
    return tonumber(cpu) or 0, tonumber(mem_used) or 0, tonumber(mem_total) or 0
end

-- ================================================================
-- Phase 0: Info Système
-- ================================================================
log("\n" .. string.rep("=", 70))
log("  Phase 0: Informations Système")
log(string.rep("=", 70))

local cpu_load, mem_used, mem_total = get_system_load()
log("\n💻 État initial du système:")
log("   CPU Load:    " .. string.format("%.1f", cpu_load) .. "%")
log("   RAM utilisée: " .. mem_used .. " MB / " .. mem_total .. " MB")
log("   RAM libre:    " .. (mem_total - mem_used) .. " MB")

-- ================================================================
-- STRESS TEST - Création de modèles progressivement plus gros
-- ================================================================

local results = {}
local total_start = os.clock()

for test_idx, test in ipairs(config.tests) do
    log("\n" .. string.rep("=", 70))
    log("  Test " .. test_idx .. "/" .. #config.tests .. ": " .. test.name)
    log("  Config: " .. test.layers .. "L × " .. test.dim .. "D × " .. test.heads .. "H")
    log(string.rep("=", 70))
    
    local mem_estimate, params_estimate = estimate_memory(test.layers, test.dim, config.vocab_size)
    log("\n📊 Estimation:")
    log("   Paramètres: " .. string.format("%.1f", params_estimate / 1000000) .. " M")
    log("   Mémoire:    " .. string.format("%.1f", mem_estimate) .. " MB")
    
    -- Vérifier si on a assez de RAM
    local cpu_before, mem_before, mem_total = get_system_load()
    local mem_available = mem_total - mem_before
    
    if mem_estimate > mem_available * 0.8 then
        log("\n⚠️  ATTENTION: Mémoire insuffisante!")
        log("   Requis:     " .. string.format("%.0f", mem_estimate) .. " MB")
        log("   Disponible: " .. mem_available .. " MB")
        log("   ⏭️  Test ignoré pour éviter swap/crash")
        
        results[test.name] = {
            status = "SKIPPED",
            reason = "Insufficient memory"
        }
        break
    end
    
    local test_config = {
        vocab_size = config.vocab_size,
        embed_dim = test.dim,
        num_layers = test.layers,
        num_heads = test.heads,
        d_ff = test.dim * 4,
        max_seq_len = config.max_seq_len,
        dropout = config.dropout
    }
    
    -- Test 1: Création Tokenizer
    local tok_time = timer("Tokenizer (" .. config.vocab_size .. " vocab)", function()
        tokenizer.create(config.vocab_size)
    end)
    
    -- Test 2: Création Modèle
    local create_time, params = timer("Création modèle Transformer", function()
        Mimir.Model.create("transformer", test_config)
        local ok, p = Mimir.Model.build()
        return p
    end)
    
    if not create_time then
        log("❌ Échec création modèle - STOP")
        results[test.name] = {status = "FAILED", reason = "Model creation failed"}
        break
    end
    
    log("   📊 Paramètres réels: " .. (params or "0"))
    
    -- Test 3: Créations multiples (stress intensif)
    log("\n🔥 STRESS TEST: " .. test.iterations .. " créations consécutives...")
    local stress_start = os.clock()
    local stress_success = 0
    
    for i = 1, test.iterations do
        if i % 5 == 0 then
            log("   Itération " .. i .. "/" .. test.iterations .. "...")
        end
        
        local success = pcall(function()
            tokenizer.create(config.vocab_size)
            Mimir.Model.create("transformer", test_config)
            Mimir.Model.build()
        end)
        
        if success then
            stress_success = stress_success + 1
        else
            log("   ⚠️  Échec itération " .. i)
        end
    end
    
    local stress_time = os.clock() - stress_start
    local avg_time = stress_time / test.iterations
    
    log("   ✓ Complétés: " .. stress_success .. "/" .. test.iterations)
    log("   ✓ Temps total: " .. string.format("%.3f", stress_time) .. "s")
    log("   ✓ Temps moyen: " .. string.format("%.3f", avg_time) .. "s")
    log("   ✓ Throughput: " .. string.format("%.2f", test.iterations / stress_time) .. " models/s")
    
    -- Test 4: Sérialisation sous charge
    local checkpoint_path = "/tmp/mimir_stress_" .. test.name .. ".safetensors"
    local save_time = timer("Sérialisation checkpoint", function()
        Mimir.Serialization.save(checkpoint_path, "safetensors")
    end)
    
    -- Get size
    local size_cmd = "du -sh " .. checkpoint_path .. " 2>/dev/null | cut -f1"
    local handle = io.popen(size_cmd)
    local size = handle:read("*a"):gsub("%s+", "")
    handle:close()
    
    if size ~= "" then
        log("   📦 Taille checkpoint: " .. size)
    end
    
    -- Cleanup
    os.execute("rm -rf " .. checkpoint_path .. " 2>/dev/null")
    
    -- État système après
    local cpu_after, mem_after, _ = get_system_load()
    log("\n💻 État système après:")
    log("   CPU Load:     " .. string.format("%.1f", cpu_after) .. "% (Δ " .. string.format("%+.1f", cpu_after - cpu_before) .. "%)")
    log("   RAM utilisée: " .. mem_after .. " MB (Δ " .. (mem_after - mem_before) .. " MB)")
    
    -- Sauvegarder résultats
    results[test.name] = {
        status = "SUCCESS",
        layers = test.layers,
        dim = test.dim,
        params = params or "0",
        create_time = create_time,
        stress_time = stress_time,
        avg_time = avg_time,
        iterations = test.iterations,
        throughput = test.iterations / stress_time,
        mem_used = mem_after - mem_before,
        checkpoint_size = size
    }
    
    -- Petite pause entre tests
    log("\n⏸️  Pause 2s avant prochain test...")
    os.execute("sleep 2")
    
end

local total_time = os.clock() - total_start

-- ================================================================
-- RÉSUMÉ FINAL
-- ================================================================
log("\n" .. string.rep("=", 70))
log("  📊 RÉSUMÉ DU STRESS TEST")
log(string.rep("=", 70))

log("\nTest              Status    Layers  Dim    Params   Temps Moy  Throughput")
log(string.rep("-", 70))

for _, test in ipairs(config.tests) do
    local r = results[test.name]
    if r then
        if r.status == "SUCCESS" then
            log(string.format("%-16s ✅ %-6s %4d   %5d   %6s   %7.3fs   %5.2f m/s",
                test.name,
                r.status,
                r.layers,
                r.dim,
                r.params,
                r.avg_time,
                r.throughput))
        elseif r.status == "SKIPPED" then
            log(string.format("%-16s ⏭️  %-6s %4d   %5d   ---     ---       ---",
                test.name,
                r.status,
                test.layers,
                test.dim))
        else
            log(string.format("%-16s ❌ %-6s %4d   %5d   ---     ---       ---",
                test.name,
                r.status,
                test.layers,
                test.dim))
        end
    end
end

log("\n" .. string.rep("-", 70))
log("Temps total du stress test: " .. string.format("%.2f", total_time) .. "s")

-- État final
local cpu_final, mem_final, mem_total = get_system_load()
log("\n💻 État final du système:")
log("   CPU Load:     " .. string.format("%.1f", cpu_final) .. "%")
log("   RAM utilisée: " .. mem_final .. " MB / " .. mem_total .. " MB")

-- Score de performance
log("\n🏆 SCORE DE PERFORMANCE:")
local successful_tests = 0
local total_models = 0
for _, r in pairs(results) do
    if r.status == "SUCCESS" then
        successful_tests = successful_tests + 1
        total_models = total_models + r.iterations
    end
end

log("   Tests réussis:    " .. successful_tests .. "/" .. #config.tests)
log("   Modèles créés:    " .. total_models)
log("   Temps moyen/test: " .. string.format("%.2f", total_time / #config.tests) .. "s")

if successful_tests == #config.tests then
    log("\n🎉 EXCELLENT! Tous les tests ont réussi!")
    log("💪 Votre machine peut gérer des gros modèles CPU-only!")
elseif successful_tests > #config.tests / 2 then
    log("\n👍 BON! La plupart des tests ont réussi")
    log("💡 Considérez ajouter plus de RAM pour les très gros modèles")
else
    log("\n⚠️  LIMITÉ: Seulement " .. successful_tests .. " tests réussis")
    log("💡 Votre machine est mieux adaptée aux petits/moyens modèles")
end

log("\n" .. string.rep("=", 70))
log("\n✅ Stress test terminé!")
log("\n📝 Options disponibles:")
log("   --safe    : Limite à 8 layers max (évite crash)")
log("   --extreme : Ajoute des modèles ULTRA (24 layers, 1536 dim)")
log("\n🚀 Mímir - CPU-Only Deep Learning")
log("💡 Aucun GPU requis, optimisé pour processeurs modernes")
