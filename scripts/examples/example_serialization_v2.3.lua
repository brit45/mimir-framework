-- ╔══════════════════════════════════════════════════════════════╗
-- ║  Exemple : Utilisation de la Serialization API v2.3.0       ║
-- ╚══════════════════════════════════════════════════════════════╝
--
-- Ce script démontre l'utilisation complète de la nouvelle API
-- de sérialisation avec les 3 formats disponibles.

print("╔════════════════════════════════════════════════════════════╗")
print("║     Serialization API v2.3.0 - Exemple Complet            ║")
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

-- ══════════════════════════════════════════════════════════════
--  Configuration Mémoire (OBLIGATOIRE)
-- ══════════════════════════════════════════════════════════════

Mimir.Allocator.configure({
    max_ram_gb = 10.0,
    enable_compression = true
})

print("✓ Allocateur configuré (limite: 10 GB)")

-- ══════════════════════════════════════════════════════════════
--  Création d'un Modèle Simple
-- ══════════════════════════════════════════════════════════════

print("\n━━━ Création du modèle ━━━\n")

-- Créer un petit Transformer pour la démo
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
    print("❌ Erreur création modèle: " .. tostring(err_create))
    os.exit(1)
end

local ok, params = Mimir.Model.allocate_params()
if not ok then
    print("❌ Erreur allocation: " .. tostring(params))
    os.exit(1)
end

print(string.format("✓ Modèle créé: %d paramètres (%.2f MB)", 
    params, params * 4 / 1024 / 1024))

Mimir.Model.init_weights("xavier", 42)
print("✓ Poids initialisés (xavier)")

-- ══════════════════════════════════════════════════════════════
--  FORMAT 1: SafeTensors (Production)
-- ══════════════════════════════════════════════════════════════

print("\n━━━ Format 1: SafeTensors (Production) ━━━\n")

local safetensors_path = "demo_model.safetensors"

print("Sauvegarde en SafeTensors...")
local ok_save, err_save = Mimir.Serialization.save(
    safetensors_path, 
    "safetensors",
    {
        save_tokenizer = true,
        save_encoder = true,
        save_optimizer = true,
        include_git_info = false
    }
)

if ok_save then
    print("✓ Sauvegarde réussie: " .. safetensors_path)
    print("  • Format: SafeTensors (compatible HuggingFace)")
    print("  • Interopérable avec PyTorch/TensorFlow")
    print("  • Header JSON + données binaires contiguës")
else
    print("❌ Erreur sauvegarde: " .. (err_save or "unknown"))
end

-- Détection automatique du format
print("\nDétection automatique du format...")
local detected_format, err_detect = Mimir.Serialization.detect_format(safetensors_path)
if detected_format then
    print("✓ Format détecté: " .. detected_format)
else
    print("❌ Erreur détection: " .. (err_detect or "unknown"))
end

-- Chargement avec détection automatique
print("\nChargement avec auto-détection...")
local ok_load, err_load = Mimir.Serialization.load(safetensors_path)

if ok_load then
    print("✓ Chargement réussi depuis SafeTensors")
else
    print("❌ Erreur chargement: " .. (err_load or "unknown"))
end

-- ══════════════════════════════════════════════════════════════
--  FORMAT 2: RawFolder (Debug avec Checksums)
-- ══════════════════════════════════════════════════════════════

print("\n━━━ Format 2: RawFolder (Debug) ━━━\n")

local rawfolder_path = "demo_checkpoint/"

print("Sauvegarde en RawFolder...")
local ok_raw, err_raw = Mimir.Serialization.save(
    rawfolder_path,
    "raw_folder",
    {
        save_tokenizer = true,
        save_encoder = true
    }
)

if ok_raw then
    print("✓ Sauvegarde réussie: " .. rawfolder_path)
    print("  • Format: RawFolder (structure lisible)")
    print("  • Checksums SHA256 pour validation")
    print("  • Idéal pour développement et debugging")
    print("  • Git-friendly (versioning)")
else
    print("❌ Erreur sauvegarde: " .. (err_raw or "unknown"))
end

-- Chargement avec vérification des checksums
print("\nChargement avec vérification checksums...")
local ok_load_raw, err_load_raw = Mimir.Serialization.load(
    rawfolder_path,
    "raw_folder",
    {
        verify_checksums = true
    }
)

if ok_load_raw then
    print("✓ Chargement réussi avec validation checksums")
else
    print("❌ Erreur chargement: " .. (err_load_raw or "unknown"))
end

-- ══════════════════════════════════════════════════════════════
--  FORMAT 3: DebugJson (Inspection)
-- ══════════════════════════════════════════════════════════════

print("\n━━━ Format 3: DebugJson (Inspection) ━━━\n")

local debugjson_path = "demo_debug.json"

print("Sauvegarde en DebugJson...")
local ok_debug, err_debug = Mimir.Serialization.save(
    debugjson_path,
    "debug_json",
    {
        debug_max_values = 20  -- Nombre max de valeurs à inclure
    }
)

if ok_debug then
    print("✓ Sauvegarde réussie: " .. debugjson_path)
    print("  • Format: DebugJson (JSON lisible)")
    print("  • Statistiques: min, max, mean, std")
    print("  • Échantillons de valeurs pour analyse")
    print("  • Idéal pour inspection et debugging")
else
    print("❌ Erreur sauvegarde: " .. (err_debug or "unknown"))
end

-- ══════════════════════════════════════════════════════════════
--  Comparaison des Formats
-- ══════════════════════════════════════════════════════════════

print("\n━━━ Comparaison des Formats ━━━\n")

print("┌─────────────┬──────────────┬────────────────┬─────────────────┐")
print("│ Format      │ Use Case     │ Avantages      │ Inconvénients   │")
print("├─────────────┼──────────────┼────────────────┼─────────────────┤")
print("│ SafeTensors │ Production   │ • Portable     │ • Binaire       │")
print("│             │ Partage      │ • Compatible   │ • Moins lisible │")
print("│             │              │ • Performant   │                 │")
print("├─────────────┼──────────────┼────────────────┼─────────────────┤")
print("│ RawFolder   │ Debug        │ • Lisible      │ • Plus lent     │")
print("│             │ Dev          │ • Checksums    │ • Plus gros     │")
print("│             │              │ • Git-friendly │                 │")
print("├─────────────┼──────────────┼────────────────┼─────────────────┤")
print("│ DebugJson   │ Inspection   │ • Très lisible │ • Très gros     │")
print("│             │ Analyse      │ • Stats inclus │ • Lent          │")
print("│             │              │ • Échantillons │ • Pas pour load │")
print("└─────────────┴──────────────┴────────────────┴─────────────────┘")

-- ══════════════════════════════════════════════════════════════
--  Recommandations
-- ══════════════════════════════════════════════════════════════

print("\n━━━ Recommandations ━━━\n")

print("✅ Production / Partage:")
print("   → Utilisez SafeTensors")
print("   → Compatible avec PyTorch/TensorFlow")
print("")

print("✅ Développement / Debug:")
print("   → Utilisez RawFolder")
print("   → Checksums pour validation")
print("")

print("✅ Inspection / Analyse:")
print("   → Utilisez DebugJson")
print("   → Statistiques et échantillons")
print("")

-- ══════════════════════════════════════════════════════════════
--  Nettoyage
-- ══════════════════════════════════════════════════════════════

print("\n━━━ Nettoyage ━━━\n")

-- Optionnel: supprimer les fichiers de démo
-- os.remove(safetensors_path)
-- os.execute("rm -rf " .. rawfolder_path)
-- os.remove(debugjson_path)

print("✓ Exemple terminé!")
print("")
print("📚 Documentation complète: docs/SAVE_LOAD.md")
print("📦 Code source: src/Serialization/")
