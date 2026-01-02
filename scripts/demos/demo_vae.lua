-- ============================================================================
-- Mímir Framework - Demo VAE
-- Variational Autoencoder pour génération
-- ============================================================================

log("=" .. string.rep("=", 78))
log("🎨 Demo VAE - Variational Autoencoder")
log("=" .. string.rep("=", 78))

-- Configuration système
log("\n🔧 Configuration...")
Allocator.configure({max_ram_gb = 10.0, enable_compression = true})
model.set_hardware(true)
log("✓ Système configuré")

-- Configuration VAE
local config = {
    input_dim = 784,
    latent_dim = 128,
    encoder_hidden = {512, 256},
    decoder_hidden = {256, 512},
    use_batchnorm = false
}

-- Créer le modèle VAE
log("\n🏗️  Création du modèle VAE...")
local success, err = model.create("vae_model")
if not success then
    log("❌ Erreur: " .. (err or "inconnue"))
    return
end

-- Construire avec architectures API
success, err = architectures.vae(config)
if not success then
    log("❌ Erreur architecture: " .. (err or "inconnue"))
    return
end

log("✓ Architecture VAE construite")

-- Allouer et initialiser
success, params = model.allocate_params()
if success then
    model.init_weights("xavier", 42)
    log("✓ Modèle construit: " .. params .. " paramètres")
    log("  Mémoire: " .. string.format("%.2f MB", params * 4 / 1024 / 1024))
else
    log("❌ Erreur allocation")
    return
end

-- Afficher config
log("\n📋 Configuration:")
log("  Input:  " .. config.input_dim .. " dimensions")
log("  Latent: " .. config.latent_dim .. " dimensions")
log("  Hidden: " .. table.concat(config.hidden_dims, " → "))
log("  Beta:   " .. config.beta .. " (KL weight)")

-- Estimation architecture
log("\n🏛️  Architecture:")
log("  Encoder: " .. config.input_dim .. " → " .. config.hidden_dims[1] .. " → " .. config.hidden_dims[2] .. " → " .. config.latent_dim)
log("  Decoder: " .. config.latent_dim .. " → " .. config.hidden_dims[2] .. " → " .. config.hidden_dims[1] .. " → " .. config.input_dim)

-- Usage pour génération
log("\n💡 Cas d'usage:")
log("  • Génération d'images (MNIST, CelebA)")
log("  • Compression latente")
log("  • Interpolation dans l'espace latent")
log("  • Détection d'anomalies")
log("  • Génération conditionnelle")

-- Modes d'utilisation
log("\n🎯 Modes d'utilisation:")
log("  1. Encode: image → latent vector")
log("  2. Decode: latent vector → image")
log("  3. Generate: random noise → image")
log("  4. Reconstruct: image → encode → decode → image")

-- Sauvegarder
log("\n💾 Sauvegarde du modèle...")
os.execute("mkdir -p checkpoints")
Mimir.Serialization.save("checkpoints/vae_demo.safetensors", "safetensors")
log("✓ Modèle sauvegardé")

log("\n✓ Demo VAE terminé!")
log("=" .. string.rep("=", 78))
