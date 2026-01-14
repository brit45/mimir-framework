-- ╔══════════════════════════════════════════════════════════════╗
-- ║  Test : Formats de Sérialisation v2.3.0                    ║
-- ╚══════════════════════════════════════════════════════════════╝
--
-- Test les 3 formats de sérialisation:
--   1. SafeTensors (production)
--   2. RawFolder (debug)
--   3. DebugJson (inspection)

print("╔════════════════════════════════════════════════════════════╗")
print("║     Test Formats de Sérialisation v2.3.0                 ║")
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

local model = (type(_G.Mimir) == "table" and type(Mimir.Model) == "table") and Mimir.Model or _G.model

-- Configuration mémoire
Allocator.configure({
    max_ram_gb = 4.0,
    enable_compression = true
})

print("✓ Allocateur configuré (4 GB)\n")

-- Créer un petit modèle pour tester
print("━━━ Création du modèle de test ━━━\n")

local cfg, warn = Arch.build_config("transformer", {
    vocab_size = 500,
    d_model = 64,
    num_layers = 2,
    num_heads = 2,
    max_seq_len = 32
})
if warn then
    print("⚠️  " .. tostring(warn))
end

local ok_create, err_create = model.create("transformer", cfg)
if not ok_create then
    print("❌ Erreur création modèle: " .. tostring(err_create))
    os.exit(1)
end

local ok_alloc, params = model.allocate_params()
if not ok_alloc then
    print("❌ Erreur allocation: " .. tostring(params))
    os.exit(1)
end

print(string.format("✓ Modèle créé: %d paramètres (%.2f MB)\n", 
    params, params * 4 / 1024 / 1024))

model.init_weights("xavier", 42)

-- ══════════════════════════════════════════════════════════════
--  TEST 1: SafeTensors
-- ══════════════════════════════════════════════════════════════

print("━━━ TEST 1: SafeTensors ━━━\n")

local st_path = "/tmp/test_model.safetensors"

print("→ Sauvegarde SafeTensors...")
local ok_st_save, err_st_save = Mimir.Serialization.save(st_path, "safetensors")

if ok_st_save then
    print("  ✓ Sauvegarde réussie: " .. st_path)
else
    print("  ❌ ÉCHEC sauvegarde: " .. (err_st_save or "unknown"))
    os.exit(1)
end

print("→ Détection format...")
local format_st = Mimir.Serialization.detect_format(st_path)
if format_st == "SAFETENSORS" then
    print("  ✓ Format détecté: " .. format_st)
else
    print("  ❌ ÉCHEC détection: " .. tostring(format_st))
    os.exit(1)
end

print("→ Chargement SafeTensors...")
local ok_st_load, err_st_load = Mimir.Serialization.load(st_path)

if ok_st_load then
    print("  ✓ Chargement réussi")
else
    print("  ❌ ÉCHEC chargement: " .. (err_st_load or "unknown"))
    os.exit(1)
end

print("✅ TEST 1 PASSÉ\n")

-- ══════════════════════════════════════════════════════════════
--  TEST 2: RawFolder
-- ══════════════════════════════════════════════════════════════

print("━━━ TEST 2: RawFolder ━━━\n")

local rf_path = "/tmp/test_checkpoint/"

print("→ Sauvegarde RawFolder...")
local ok_rf_save, err_rf_save = Mimir.Serialization.save(rf_path, "raw_folder")

if ok_rf_save then
    print("  ✓ Sauvegarde réussie: " .. rf_path)
else
    print("  ❌ ÉCHEC sauvegarde: " .. (err_rf_save or "unknown"))
    os.exit(1)
end

print("→ Détection format...")
local format_rf = Mimir.Serialization.detect_format(rf_path)
if format_rf == "RAWFOLDER" then
    print("  ✓ Format détecté: " .. format_rf)
else
    print("  ❌ ÉCHEC détection: " .. tostring(format_rf))
    os.exit(1)
end

print("→ Chargement RawFolder (sans vérification checksums)...")
local ok_rf_load, err_rf_load = Mimir.Serialization.load(rf_path, "raw_folder")

if ok_rf_load then
    print("  ✓ Chargement réussi")
else
    print("  ❌ ÉCHEC chargement: " .. (err_rf_load or "unknown"))
    os.exit(1)
end

print("→ Chargement RawFolder (avec vérification checksums)...")
local ok_rf_check, err_rf_check = Mimir.Serialization.load(rf_path, "raw_folder", {
    verify_checksums = true
})

if ok_rf_check then
    print("  ✓ Chargement avec checksums réussi")
else
    print("  ❌ ÉCHEC vérification checksums: " .. (err_rf_check or "unknown"))
    os.exit(1)
end

print("✅ TEST 2 PASSÉ\n")

-- ══════════════════════════════════════════════════════════════
--  TEST 3: DebugJson
-- ══════════════════════════════════════════════════════════════

print("━━━ TEST 3: DebugJson ━━━\n")

local dj_path = "/tmp/test_debug.json"

print("→ Sauvegarde DebugJson...")
local ok_dj_save, err_dj_save = Mimir.Serialization.save(dj_path, "debug_json", {
    debug_max_values = 10
})

if ok_dj_save then
    print("  ✓ Sauvegarde réussie: " .. dj_path)
else
    print("  ❌ ÉCHEC sauvegarde: " .. (err_dj_save or "unknown"))
    os.exit(1)
end

print("→ Détection format...")
local format_dj = Mimir.Serialization.detect_format(dj_path)
if format_dj == "DEBUGJSON" then
    print("  ✓ Format détecté: " .. format_dj)
else
    print("  ❌ ÉCHEC détection: " .. tostring(format_dj))
    os.exit(1)
end

-- Note: DebugJson n'est pas chargeable (inspection seulement)
print("  ℹ️  DebugJson est un format d'inspection uniquement (pas de load)")

print("✅ TEST 3 PASSÉ\n")

-- ══════════════════════════════════════════════════════════════
--  TEST 4: Auto-détection
-- ══════════════════════════════════════════════════════════════

print("━━━ TEST 4: Auto-détection ━━━\n")

print("→ Chargement SafeTensors sans spécifier format...")
local ok_auto_st, err_auto_st = Mimir.Serialization.load(st_path)

if ok_auto_st then
    print("  ✓ Auto-détection SafeTensors OK")
else
    print("  ❌ ÉCHEC auto-détection: " .. (err_auto_st or "unknown"))
    os.exit(1)
end

print("→ Chargement RawFolder sans spécifier format...")
local ok_auto_rf, err_auto_rf = Mimir.Serialization.load(rf_path)

if ok_auto_rf then
    print("  ✓ Auto-détection RawFolder OK")
else
    print("  ❌ ÉCHEC auto-détection: " .. (err_auto_rf or "unknown"))
    os.exit(1)
end

print("✅ TEST 4 PASSÉ\n")

-- ══════════════════════════════════════════════════════════════
--  Vérification taille fichiers
-- ══════════════════════════════════════════════════════════════

print("━━━ Tailles des fichiers ━━━\n")

local function get_size(path)
    local cmd = "du -sh '" .. path .. "' 2>/dev/null | cut -f1"
    local handle = io.popen(cmd)
    local result = handle:read("*a"):gsub("%s+", "")
    handle:close()
    return result
end

print(string.format("  SafeTensors: %s (%s)", st_path, get_size(st_path)))
print(string.format("  RawFolder:   %s (%s)", rf_path, get_size(rf_path)))
print(string.format("  DebugJson:   %s (%s)", dj_path, get_size(dj_path)))

-- ══════════════════════════════════════════════════════════════
--  Nettoyage
-- ══════════════════════════════════════════════════════════════

print("\n━━━ Nettoyage ━━━\n")

os.remove(st_path)
os.execute("rm -rf " .. rf_path)
os.remove(dj_path)

print("✓ Fichiers temporaires supprimés")

-- ══════════════════════════════════════════════════════════════
--  Résumé
-- ══════════════════════════════════════════════════════════════

print("\n╔════════════════════════════════════════════════════════════╗")
print("║                    TOUS LES TESTS PASSÉS ✅               ║")
print("╚════════════════════════════════════════════════════════════╝")
print("")
print("Formats testés:")
print("  ✓ SafeTensors  - save/load/detect")
print("  ✓ RawFolder    - save/load/detect + checksums")
print("  ✓ DebugJson    - save/detect")
print("  ✓ Auto-detect  - SafeTensors & RawFolder")
print("")
print("🎉 API Serialization v2.3.0 fonctionnelle !")
