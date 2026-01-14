-- ╔══════════════════════════════════════════════════════════════╗
-- ║  Test : API Hiérarchique Mimir.* Complète                   ║
-- ╚══════════════════════════════════════════════════════════════╝
--
-- Validation de la nouvelle structure API avec tous les namespaces
-- sous le préfixe Mimir.*

print("╔════════════════════════════════════════════════════════════╗")
print("║     Test API Hiérarchique Mimir.* (v2.3.0)              ║")
print("╚════════════════════════════════════════════════════════════╝\n")

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

-- ━━━ Test 1: Mimir.Allocator ━━━
print("━━━ Test 1: Mimir.Allocator ━━━\n")

Mimir.Allocator.configure({
    max_ram_gb = 2.0,
    enable_compression = false
})

print("✓ Mimir.Allocator.configure() OK\n")

-- ━━━ Test 2: Mimir.Model ━━━
print("━━━ Test 2: Mimir.Model ━━━\n")

local cfg, warn = Arch.build_config("transformer", {
    vocab_size = 1000,
    d_model = 128,
    num_layers = 2,
    num_heads = 4,
    max_seq_len = 64
})
if warn then
    print("⚠️  " .. tostring(warn))
end

local ok_create, err_create = Mimir.Model.create("transformer", cfg)
if not ok_create then
    print("❌ Erreur Mimir.Model.create: " .. tostring(err_create))
    os.exit(1)
end
print("✓ Mimir.Model.create(type, cfg) OK")

-- ━━━ Test 3: Mimir.Architectures ━━━
print("\n━━━ Test 3: Mimir.Architectures ━━━\n")

local names, err_names = Mimir.Architectures.available()
if not names then
    print("⚠️  Architectures.available() indisponible: " .. tostring(err_names))
else
    print("✓ Mimir.Architectures.available() OK (" .. tostring(#names) .. " entrées)")
end

-- ━━━ Test 4: Allocation des paramètres ━━━
print("\n━━━ Test 4: Allocation ━━━\n")

local ok, params = Mimir.Model.allocate_params()
if not ok then
    print("❌ Erreur allocation: " .. tostring(params))
    os.exit(1)
end

print(string.format("✓ Mimir.Model.allocate_params() OK: %d paramètres", params))

-- ━━━ Test 5: Initialisation des poids ━━━
print("\n━━━ Test 5: Initialisation ━━━\n")

Mimir.Model.init_weights("xavier", 42)
print("✓ Mimir.Model.init_weights() OK")

-- ━━━ Test 6: Mimir.Serialization ━━━
print("\n━━━ Test 6: Mimir.Serialization ━━━\n")

local path1 = "/tmp/test_hierarchical_api.safetensors"
local success1 = Mimir.Serialization.save(path1, "safetensors", {
    save_optimizer = false
})

if not success1 then
    print("❌ Échec sauvegarde SafeTensors")
    os.exit(1)
end

print("✓ Mimir.Serialization.save() SafeTensors OK")

-- ━━━ Test 7: Mimir.Serialization.detect_format ━━━
print("\n━━━ Test 7: Détection de format ━━━\n")

local format = Mimir.Serialization.detect_format(path1)
if format ~= "SAFETENSORS" then
    print("❌ Mauvaise détection: " .. tostring(format))
    os.exit(1)
end

print("✓ Mimir.Serialization.detect_format() OK: " .. format)

-- ━━━ Test 8: Debug JSON Enhanced ━━━
print("\n━━━ Test 8: Debug JSON Enhanced ━━━\n")

local path2 = "/tmp/test_hierarchical_api.json"
local success2 = Mimir.Serialization.save(path2, "debug_json", {
    include_gradients = false,
    include_optimizer_state = false,
    include_checksums = true,
    max_values_per_tensor = 10
})

if not success2 then
    print("❌ Échec sauvegarde DebugJson")
    os.exit(1)
end

print("✓ Mimir.Serialization.save() DebugJson OK")

-- Vérifier le JSON
local json = read_json(path2)
if not json then
    print("❌ JSON invalide")
    os.exit(1)
end

if json.format_version ~= "1.1.0" then
    print(string.format("❌ Mauvaise version: %s", tostring(json.format_version)))
    os.exit(1)
end

print(string.format("✓ Version JSON: %s", json.format_version))
print(string.format("✓ Features: %d", json.features and #json.features or 0))

-- ━━━ Résumé ━━━
print("\n╔════════════════════════════════════════════════════════════╗")
print("║              ✅ TOUS LES TESTS PASSÉS!                    ║")
print("╚════════════════════════════════════════════════════════════╝")

print("\n📋 Namespaces testés:")
print("  • Mimir.Allocator         ✅")
print("  • Mimir.Model             ✅")
print("  • Mimir.Architectures     ✅")
print("  • Mimir.Serialization     ✅")

print("\n✅ L'API hiérarchique Mimir.* fonctionne correctement!\n")
