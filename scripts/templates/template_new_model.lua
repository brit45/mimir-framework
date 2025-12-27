#!/usr/bin/env lua
-- ══════════════════════════════════════════════════════════════
--  TEMPLATE SCRIPT - Mímir Framework v2.0
--  Utilisez ce template pour créer vos propres modèles
-- ══════════════════════════════════════════════════════════════

log("\n╔════════════════════════════════════════════════════════╗")
log("║      Template Script - Mímir Framework v2.0           ║")
log("╚════════════════════════════════════════════════════════╝\n")

-- ══════════════════════════════════════════════════════════════
--  ÉTAPE 1: CONFIGURATION SYSTÈME (OBLIGATOIRE!)
-- ══════════════════════════════════════════════════════════════

log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  1. Configuration Système")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

-- ⚠️ CRITIQUE: Toujours configurer l'allocateur en premier!
-- Cela active la limite stricte de 10 GB et la protection contre les crashs OOM
allocator.configure({
    max_ram_gb = 10.0,              -- Limite stricte (ne pas dépasser!)
    enable_compression = true       -- Compression LZ4 (~50% économie RAM)
})
log("✓ Allocateur configuré (limite: 10 GB, compression LZ4)")

-- Vérifier et activer l'accélération hardware
local hw = model.hardware_caps()
log("\n🔧 Capacités Hardware:")
log(string.format("  • AVX2:  %s", hw.avx2 and "✓" or "✗"))
log(string.format("  • FMA:   %s", hw.fma and "✓" or "✗"))
log(string.format("  • F16C:  %s", hw.f16c and "✓" or "✗"))
log(string.format("  • BMI2:  %s", hw.bmi2 and "✓" or "✗"))

if hw.avx2 or hw.fma then
    model.set_hardware(true)
    log("\n✓ Accélération hardware activée\n")
end

-- ══════════════════════════════════════════════════════════════
--  ÉTAPE 2: CONFIGURATION DU MODÈLE
-- ══════════════════════════════════════════════════════════════

log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  2. Configuration du Modèle")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

-- 💡 CONSEIL: Utilisez des valeurs raisonnables pour rester sous 10 GB
-- Exemples de configs qui rentrent dans 10 GB:
--   • Transformer: d_model=512, num_layers=6-8
--   • ResNet-50: ~25M params
--   • UNet: base_channels=64-128
--   • Diffusion: image_size=256

local config = {
    -- OPTION 1: Transformer/GPT
    vocab_size = 30000,        -- 30k tokens (raisonnable)
    d_model = 512,             -- Dimension modérée
    num_layers = 6,            -- 6 couches OK
    num_heads = 8,
    max_seq_len = 512,
    dropout = 0.1,
    
    -- OPTION 2: Vision (décommenter si besoin)
    -- image_size = 224,
    -- channels = 3,
    -- num_classes = 1000,
    
    -- Training
    batch_size = 32,
    learning_rate = 0.001,
    epochs = 10
}

log("Configuration choisie:")
log(string.format("  • vocab_size: %d", config.vocab_size))
log(string.format("  • d_model: %d", config.d_model))
log(string.format("  • num_layers: %d", config.num_layers))

-- ══════════════════════════════════════════════════════════════
--  ÉTAPE 3: CRÉATION DU MODÈLE
-- ══════════════════════════════════════════════════════════════

log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  3. Création du Modèle")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

local model_name = "my_model"
model.create(model_name, config)
log("✓ Modèle '" .. model_name .. "' créé")

-- Construire l'architecture (choisir une option)
-- OPTION 1: Transformer
architectures.transformer(config)
log("✓ Architecture Transformer construite")

-- OPTION 2: ResNet (décommenter si besoin)
-- architectures.resnet(config)
-- log("✓ Architecture ResNet construite")

-- OPTION 3: UNet (décommenter si besoin)
-- architectures.unet(config)
-- log("✓ Architecture UNet construite")

-- ══════════════════════════════════════════════════════════════
--  ÉTAPE 4: ALLOCATION DES PARAMÈTRES (VÉRIFIER LE SUCCÈS!)
-- ══════════════════════════════════════════════════════════════

log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  4. Allocation des Paramètres")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

local success, param_count = model.allocate_params()

-- ⚠️ TOUJOURS vérifier le succès!
if not success then
    log("❌ ERREUR: Impossible d'allouer les paramètres!")
    log("⚠️  La limite de 10 GB a été atteinte")
    log("\n💡 Solutions possibles:")
    log("  1. Réduire d_model (ex: 512 → 256)")
    log("  2. Réduire num_layers (ex: 6 → 4)")
    log("  3. Réduire vocab_size (ex: 30000 → 20000)")
    log("  4. Choisir une architecture plus petite")
    os.exit(1)
end

-- Calcul de la mémoire utilisée
local memory_mb = param_count * 4 / (1024 * 1024)
log(string.format("✓ Paramètres alloués: %d", param_count))
log(string.format("  Mémoire utilisée: %.2f MB", memory_mb))
log(string.format("  Compression LZ4: ~%.2f MB en RAM", memory_mb * 0.5))

-- ══════════════════════════════════════════════════════════════
--  ÉTAPE 5: INITIALISATION DES POIDS
-- ══════════════════════════════════════════════════════════════

log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  5. Initialisation des Poids")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

-- Méthodes disponibles: "xavier", "he", "uniform"
success = model.init_weights("xavier", 42)  -- seed = 42 pour reproductibilité

if not success then
    log("❌ Échec de l'initialisation des poids")
    os.exit(1)
end

log("✓ Poids initialisés (méthode: xavier, seed: 42)")

-- ══════════════════════════════════════════════════════════════
--  ÉTAPE 6: TOKENIZER (SI MODÈLE NLP)
-- ══════════════════════════════════════════════════════════════

log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  6. Tokenizer (optionnel)")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

tokenizer.create(config.vocab_size)
tokenizer.add_token("[PAD]")
tokenizer.add_token("[UNK]")
tokenizer.add_token("[CLS]")
tokenizer.add_token("[SEP]")
log("✓ Tokenizer créé (vocab_size: " .. config.vocab_size .. ")")

-- ══════════════════════════════════════════════════════════════
--  ÉTAPE 7: DATASET (SI TRAINING)
-- ══════════════════════════════════════════════════════════════

log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  7. Dataset (optionnel)")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

-- Charger un dataset (décommenter si besoin)
-- dataset.load("path/to/dataset")
-- dataset.prepare_sequences(config.max_seq_len)
-- log("✓ Dataset chargé et préparé")

log("⚠️  Dataset non configuré (à ajouter selon vos besoins)")

-- ══════════════════════════════════════════════════════════════
--  ÉTAPE 8: TRAINING / INFERENCE
-- ══════════════════════════════════════════════════════════════

log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  8. Training / Inference")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

log("💡 Votre code d'entraînement ou d'inférence ici...")
log("\nExemple de boucle de training:")
log("  for epoch = 1, config.epochs do")
log("    loss = model.train_step(batch)")
log("    log('Epoch ' .. epoch .. ', Loss: ' .. loss)")
log("  end")

-- ══════════════════════════════════════════════════════════════
--  ÉTAPE 9: SAUVEGARDE (OPTIONNEL)
-- ══════════════════════════════════════════════════════════════

log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  9. Sauvegarde")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

-- Sauvegarder le modèle (décommenter si besoin)
-- local checkpoint_path = "checkpoints/" .. model_name .. ".bin"
-- model.save(checkpoint_path)
-- log("✓ Modèle sauvegardé: " .. checkpoint_path)

log("⚠️  Sauvegarde non configurée (à ajouter si nécessaire)")

-- ══════════════════════════════════════════════════════════════
--  RÉSUMÉ FINAL
-- ══════════════════════════════════════════════════════════════

log("\n╔════════════════════════════════════════════════════════╗")
log("║                    RÉSUMÉ                              ║")
log("╚════════════════════════════════════════════════════════╝\n")

log("✅ Modèle créé avec succès!")
log(string.format("  • Nom: %s", model_name))
log(string.format("  • Paramètres: %d", param_count))
log(string.format("  • Mémoire: %.2f MB (%.2f MB avec compression)", memory_mb, memory_mb * 0.5))
log(string.format("  • Limite RAM: 10 GB"))
log(string.format("  • Utilisation: %.1f%%", (memory_mb / (10 * 1024)) * 100))

log("\n💡 Prochaines étapes:")
log("  1. Ajouter votre code de training/inference (étape 8)")
log("  2. Charger un dataset si nécessaire (étape 7)")
log("  3. Sauvegarder le modèle après training (étape 9)")

log("\n📚 Documentation:")
log("  • docs/MEMORY_BEST_PRACTICES.md - Bonnes pratiques mémoire")
log("  • MEMORY_SAFETY_FIXES.md - Détails des correctifs")
log("  • README.md - Documentation générale")

log("\n✨ Bon apprentissage avec Mímir Framework! ✨\n")
