#!/usr/bin/env lua
-- ══════════════════════════════════════════════════════════════
--  Exemple Simple - Mímir Framework API Lua
-- ══════════════════════════════════════════════════════════════

print("\n╔════════════════════════════════════════════════════════╗")
print("║      Exemple Simple - API Lua Mímir                   ║")
print("╚════════════════════════════════════════════════════════╝\n")

-- 0. Configuration allocateur (OBLIGATOIRE!)
print("🔧 Configuration Système:")
allocator.configure({
    max_ram_gb = 10.0,
    enable_compression = true
})
print("✓ Allocateur configuré (limite: 10 GB)\n")

-- 1. Vérifier les capacités hardware
print("🔧 Capacités Hardware:")
local hw = model.hardware_caps()
print(string.format("  • AVX2:  %s", hw.avx2 and "✓" or "✗"))
print(string.format("  • FMA:   %s", hw.fma and "✓" or "✗"))
print(string.format("  • F16C:  %s", hw.f16c and "✓" or "✗"))
print(string.format("  • BMI2:  %s", hw.bmi2 and "✓" or "✗"))

-- Activer l'accélération hardware si disponible
if hw.avx2 or hw.fma then
    model.set_hardware(true)
    print("\n✓ Accélération hardware activée\n")
end

-- 2. Créer un modèle simple (Transformer)
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Création d'un Transformer")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

model.create("my_transformer")
print("✓ Modèle créé")

-- Configuration simple
local config = {
    vocab_size = 5000,
    d_model = 256,
    num_layers = 4,
    num_heads = 4,
    max_seq_len = 512
}

architectures.transformer(config)
print("✓ Architecture Transformer construite")
print(string.format("  • Vocabulaire: %d tokens", config.vocab_size))
print(string.format("  • Dimension: %d", config.d_model))
print(string.format("  • Couches: %d", config.num_layers))
print(string.format("  • Têtes d'attention: %d", config.num_heads))

-- 3. Allouer et initialiser
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Allocation des Paramètres")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

local success, param_count = model.allocate_params()

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

model.init_weights("he", 42)
print("\n✓ Poids initialisés (méthode He, seed=42)")

-- 4. Tokenizer (optionnel)
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Tokenizer")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

tokenizer.create(config.vocab_size)
print("✓ Tokenizer créé")

local test_text = "Bonjour le monde"
local tokens = tokenizer.tokenize(test_text)
print(string.format("✓ Texte tokenizé: '%s'", test_text))
print("  Tokens: " .. table.concat(tokens, ", "))

-- 5. Sauvegarder
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Sauvegarde")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

-- Créer le répertoire checkpoints s'il n'existe pas
os.execute("mkdir -p checkpoints")

model.save("checkpoints/simple_transformer.safetensors")
print("✓ Modèle sauvegardé: checkpoints/simple_transformer.safetensors")

tokenizer.save("checkpoints/simple_tokenizer.json")
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
