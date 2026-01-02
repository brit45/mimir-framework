-- ╔══════════════════════════════════════════════════════════════╗
-- ║  Test : API Unifiée avec Enhanced DebugJson v1.1.0          ║
-- ╚══════════════════════════════════════════════════════════════╝
--
-- Test de l'API unifiée Mimir.Serialization.save() avec les options
-- Enhanced Debug JSON v1.1.0 intégrées

print("╔════════════════════════════════════════════════════════════╗")
print("║     Test API Unifiée + Enhanced DebugJson v1.1.0        ║")
print("╚════════════════════════════════════════════════════════════╝\n")

-- Configuration mémoire
Allocator.configure({
    max_ram_gb = 2.0,
    enable_compression = false
})

print("✓ Allocateur configuré (2 GB)\n")

-- Créer un modèle Transformer simple
print("━━━ Création du modèle ━━━\n")

model.create("test_unified_api")

architectures.transformer({
    vocab_size = 500,
    d_model = 64,
    num_layers = 2,
    num_heads = 2,
    max_seq_len = 32
})

local ok_alloc, params = model.allocate_params()
if not ok_alloc then
    print("❌ Erreur allocation: " .. tostring(params))
    os.exit(1)
end

print(string.format("✓ Modèle créé: %d paramètres\n", params))

model.init_weights("xavier", 42)

-- Test 1: Sauvegarde DebugJson BASIQUE (legacy)
print("━━━ Test 1: DebugJson basique (legacy) ━━━\n")

local path1 = "/tmp/test_debug_basic.json"
local success1 = Mimir.Serialization.save(path1, "debug_json", {
    debug_max_values = 10
})

if not success1 then
    print("❌ Échec Test 1")
    os.exit(1)
end

print("✓ Test 1 OK: " .. path1)

local json1 = read_json(path1)
if not json1 then
    print("❌ JSON invalide")
    os.exit(1)
end

print(string.format("  format_version: %s", json1.format_version or "N/A"))
print(string.format("  features: %d\n", json1.features and #json1.features or 0))

-- Test 2: Sauvegarde DebugJson ENHANCED v1.1 avec TOUTES les options
print("━━━ Test 2: DebugJson Enhanced v1.1 (complet) ━━━\n")

local path2 = "/tmp/test_debug_enhanced.json"
local success2 = Mimir.Serialization.save(path2, "debug_json", {
    include_gradients = true,
    include_optimizer_state = true,
    max_values_per_tensor = 20,
    include_activations = false,
    include_checksums = true,
    include_weight_deltas = false,  -- Pas de snapshot précédent
    include_git_info = false,
    save_tokenizer = false,
    save_encoder = false
})

if not success2 then
    print("❌ Échec Test 2")
    os.exit(1)
end

print("✓ Test 2 OK: " .. path2)

local json2 = read_json(path2)
if not json2 then
    print("❌ JSON invalide")
    os.exit(1)
end

-- Vérifications
if json2.format_version ~= "1.1.0" then
    print(string.format("❌ Version incorrecte: %s", tostring(json2.format_version)))
    os.exit(1)
end
print(string.format("✓ format_version: %s", json2.format_version))

if not json2.features or #json2.features == 0 then
    print("❌ Pas de features")
    os.exit(1)
end

local expected_features = {
    "layer_config",
    "real_shapes",
    "optimizer_state",
    "checksums"
}

for _, feat in ipairs(expected_features) do
    local found = false
    for _, f in ipairs(json2.features) do
        if f == feat then
            found = true
            break
        end
    end
    if not found then
        print(string.format("❌ Feature manquante: %s", feat))
        os.exit(1)
    end
end

print(string.format("✓ features: %d (toutes présentes)", #json2.features))

if not json2.optimizer then
    print("❌ Pas de section optimizer")
    os.exit(1)
end
print(string.format("✓ optimizer: type=%s\n", json2.optimizer.type or "N/A"))

-- Test 3: Comparaison de taille
print("━━━ Test 3: Comparaison tailles ━━━\n")

local function get_file_size(path)
    local f = io.open(path, "r")
    if not f then return 0 end
    local size = f:seek("end")
    f:close()
    return size
end

local size1 = get_file_size(path1)
local size2 = get_file_size(path2)

print(string.format("  Basic:    %d bytes", size1))
print(string.format("  Enhanced: %d bytes", size2))
print(string.format("  Ratio:    %.2fx\n", size2 / math.max(1, size1)))

-- Test 4: Vérifier que SafeTensors fonctionne toujours
print("━━━ Test 4: SafeTensors (compatibilité) ━━━\n")

local path_st = "/tmp/test_unified.safetensors"
local success_st = Mimir.Serialization.save(path_st, "safetensors", {
    save_optimizer = true
})

if not success_st then
    print("❌ Échec SafeTensors")
    os.exit(1)
end

print("✓ SafeTensors OK: " .. path_st)

local size_st = get_file_size(path_st)
print(string.format("  Taille: %.2f MB\n", size_st / 1024 / 1024))

-- Résumé
print("╔════════════════════════════════════════════════════════════╗")
print("║                ✅ TOUS LES TESTS PASSÉS!                  ║")
print("╚════════════════════════════════════════════════════════════╝")

print("\n📋 Résumé API Unifiée:")
print("  • Format SafeTensors: ✅ Fonctionnel")
print("  • Format DebugJson legacy: ✅ Fonctionnel")
print("  • Format DebugJson v1.1.0: ✅ Fonctionnel")
print("  • Options Enhanced intégrées: ✅ OK")

print("\n✅ L'API unifiée Mimir.Serialization.save() supporte")
print("   toutes les options Enhanced Debug JSON v1.1.0!\n")
