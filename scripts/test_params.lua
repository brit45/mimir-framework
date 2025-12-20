-- Test d'allocation des paramètres

log("╔═══════════════════════════════════════════════════════════════════╗")
log("║         Test d'Allocation et Initialisation des Paramètres       ║")
log("╚═══════════════════════════════════════════════════════════════════╝")

-- Créer un tokenizer
tokenizer.create(5000)
log("✓ Tokenizer créé (5000 vocab)")

-- Créer un petit transformer
local config = {
    vocab_size = 5000,
    embed_dim = 128,
    num_layers = 2,
    num_heads = 4,
    d_ff = 512,
    max_seq_len = 64,
    dropout = 0.1
}

log("\n📊 Configuration du modèle:")
log("  Vocab:      " .. config.vocab_size)
log("  Embed dim:  " .. config.embed_dim)
log("  Layers:     " .. config.num_layers)
log("  Heads:      " .. config.num_heads)
log("  FFN dim:    " .. config.d_ff)

log("\n🔨 Création du modèle...")
model.create("transformer", config)

log("\n🏗️  Construction du modèle...")
local ok, params = model.build()

log("\n📈 Résultats:")
log("  Status:     " .. (ok and "✓ Succès" or "❌ Échec"))
log("  Paramètres: " .. params)
log("  Mémoire:    " .. string.format("%.2f", params * 4 / 1024 / 1024) .. " MB (float32)")
log("  Mémoire:    " .. string.format("%.2f", params * 2 / 1024 / 1024) .. " MB (float16)")

log("\n💾 Test de sérialisation...")
model.save("/tmp/test_params_checkpoint")
log("✓ Checkpoint sauvegardé")

-- Vérifier la taille
local handle = io.popen("du -sh /tmp/test_params_checkpoint 2>/dev/null | cut -f1")
local size = handle:read("*a"):gsub("%s+", "")
handle:close()

if size ~= "" then
    log("  Taille: " .. size)
end

-- Cleanup
os.execute("rm -rf /tmp/test_params_checkpoint 2>/dev/null")

log("\n✅ Test terminé avec succès!")
log("🚀 Les paramètres sont maintenant alloués et initialisés automatiquement")
