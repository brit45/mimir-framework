#!/usr/bin/env lua
-- ══════════════════════════════════════════════════════════════
--  Exemple Simple - Mímir Framework API Lua
-- ══════════════════════════════════════════════════════════════

print("\n╔════════════════════════════════════════════════════════╗")
print("║      Exemple Simple - API Lua Mímir                   ║")
print("╚════════════════════════════════════════════════════════╝\n")

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

print("🔧 Configuration Système:")
Mimir.Allocator.configure({
    max_ram_gb = 10.0,
    enable_compression = true
})
print("✓ Allocateur configuré (limite: 10 GB)\n")

-- 1. Vérifier les capacités hardware
print("🔧 Capacités Hardware:")
local hw = Mimir.Model.hardware_caps()
print(string.format("  • AVX2:  %s", hw.avx2 and "✓" or "✗"))
print(string.format("  • FMA:   %s", hw.fma and "✓" or "✗"))
print(string.format("  • F16C:  %s", hw.f16c and "✓" or "✗"))
print(string.format("  • BMI2:  %s", hw.bmi2 and "✓" or "✗"))

-- Activer l'accélération hardware si disponible
if hw.avx2 or hw.fma then
    Mimir.Model.set_hardware(true)
    print("\n✓ Accélération hardware activée\n")
end

-- 2. Créer un modèle simple (Transformer)
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Création d'un Transformer")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

-- Configuration simple
local config = {
    vocab_size = 5000,
    d_model = 256,
    num_layers = 4,
    num_heads = 4,
    max_seq_len = 512
}

local cfg, warn = Arch.build_config("transformer", config)
if warn then
    print("⚠️  " .. tostring(warn))
end

local ok_create, err_create = Mimir.Model.create("transformer", cfg)
if not ok_create then
    print("❌ ERREUR: Impossible de créer le modèle: " .. tostring(err_create))
    os.exit(1)
end

print("✓ Modèle Transformer créé via registre")
print(string.format("  • Vocabulaire: %d tokens", config.vocab_size))
print(string.format("  • Dimension: %d", config.d_model))
print(string.format("  • Couches: %d", config.num_layers))
print(string.format("  • Têtes d'attention: %d", config.num_heads))

-- 3. Allouer et initialiser
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Allocation des Paramètres")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

local success, param_count = Mimir.Model.allocate_params()

if not success then
    print("❌ ERREUR: Impossible d'allouer les paramètres!")
    print("⚠️  Limite de 10 GB atteinte ou modèle trop grand")
    print("💡 Solution: Réduire d_model, num_layers ou vocab_size")
    os.exit(1)
end

if success then
    print(string.format("\n✓ Paramètres alloués: %d", param_count))
    print(string.format("  • Taille (FP32): %.2f MB", param_count * 4 / 1024 / 1024))
    print(string.format("  • Taille (FP16): %.2f MB", param_count * 2 / 1024 / 1024))
else
    print("❌ Échec de l'allocation")
    return
end

Mimir.Model.init_weights("he", 42)
print("\n✓ Poids initialisés (méthode He, seed=42)")

-- 4. Tokenizer (optionnel)
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Tokenizer")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

Mimir.Tokenizer.create(config.vocab_size)
print("✓ Tokenizer créé")

local test_text = "Bonjour le monde"
local tokens = Mimir.Tokenizer.tokenize(test_text)
print(string.format("✓ Texte tokenizé: '%s'", test_text))
print("  Tokens: " .. table.concat(tokens, ", "))

-- 5. Sauvegarder
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Sauvegarde")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

-- Créer le répertoire checkpoints s'il n'existe pas
os.execute("mkdir -p checkpoints")

Mimir.Serialization.save("checkpoints/simple_transformer.safetensors", "safetensors")
print("✓ Modèle sauvegardé: checkpoints/simple_transformer.safetensors")

Mimir.Tokenizer.save("checkpoints/simple_tokenizer.json")
print("✓ Tokenizer sauvegardé: checkpoints/simple_tokenizer.json")

-- 6. Résumé
print("\n╔════════════════════════════════════════════════════════╗")
print("║                  Résumé                                ║")
print("╠════════════════════════════════════════════════════════╣")
print(string.format("║  Modèle:       Transformer (%d params)%s║", param_count, string.rep(" ", 17 - #tostring(param_count))))
print(string.format("║  Vocabulaire:  %d tokens%s║", config.vocab_size, string.rep(" ", 26 - #tostring(config.vocab_size))))
print(string.format("║  Contexte:     %d tokens%s║", config.max_seq_len, string.rep(" ", 26 - #tostring(config.max_seq_len))))
print("║  Hardware:     " .. (hw.avx2 and "AVX2" or (hw.fma and "FMA" or "CPU")) .. " optimized                            ║")
print("╚════════════════════════════════════════════════════════╝\n")

print("✅ Exemple terminé avec succès!\n")
