-- ============================================================================
-- Mímir Framework - Demo Diffusion
-- Denoising Diffusion Models (Stable Diffusion style)
-- ============================================================================

log("=" .. string.rep("=", 78))
log("🌊 Demo Diffusion - Denoising Diffusion Models")
log("=" .. string.rep("=", 78))

-- Configuration système (OBLIGATOIRE!)
log("\n🔧 Configuration...")
-- ⚠️ Toujours configurer l'allocateur au début pour activer la limite de 10 GB
Allocator.configure({max_ram_gb = 10.0, enable_compression = true})
model.set_hardware(true)
log("✓ Allocateur configuré (limite: 10 GB, compression LZ4)")
log("✓ Accélération hardware activée")

-- Configuration Diffusion
local config = {
    image_size = 256,
    image_channels = 3,
    base_channels = 128,
    num_res_blocks = 2,
    channel_multipliers = {1, 2, 2, 2},
    attention_levels = {1, 2, 3},
    time_embed_dim = 512
}

-- Créer le modèle Diffusion
log("\n🏗️  Création du modèle Diffusion...")
local success, err = model.create("diffusion_model")
if not success then
    log("❌ Erreur: " .. (err or "inconnue"))
    return
end

-- Construire avec architectures API
success, err = architectures.diffusion(config)
if not success then
    log("❌ Erreur architecture: " .. (err or "inconnue"))
    return
end

log("✓ Architecture Diffusion construite")

-- Allouer et initialiser
log("\n💾 Allocation des paramètres...")
success, params = model.allocate_params()

if not success then
    log("❌ ERREUR: Impossible d'allouer les paramètres!")
    log("⚠️  Modèle trop grand pour la limite de 10 GB")
    log("💡 Solution: Réduire base_channels, num_res_blocks ou image_size")
    return
end

if success then
    model.init_weights("he", 42)
    log("✓ Modèle construit: " .. params .. " paramètres")
    log("  Mémoire: " .. string.format("%.2f MB", params * 4 / 1024 / 1024))
else
    log("❌ Erreur allocation")
    return
end

-- Afficher config
log("\n📋 Configuration:")
log("  Image size: " .. config.image_size .. "x" .. config.image_size)
log("  Channels:   " .. config.image_channels .. " (RGB)")
log("  Timesteps:  " .. config.timesteps)
log("  Base channels: " .. config.model_channels)
log("  Res blocks: " .. config.num_res_blocks)

-- Process de diffusion
log("\n🌊 Process de Diffusion:")
log("  Forward (Training):")
log("    x₀ (image) → ... → x_T (noise)")
log("    Ajouter du bruit progressivement")
log("  Reverse (Generation):")
log("    x_T (noise) → ... → x₀ (image)")
log("    Débruiter progressivement")

-- Estimation génération
log("\n⏱️  Estimation génération:")
log("  Timesteps: " .. config.timesteps)
log("  Mode rapide: ~50 steps (DDIM)")
log("  Mode qualité: " .. config.timesteps .. " steps (DDPM)")

-- Usage pour génération
log("\n💡 Cas d'usage:")
log("  • Text-to-image (Stable Diffusion)")
log("  • Image editing / inpainting")
log("  • Super-résolution")
log("  • Image-to-image translation")
log("  • Génération conditionnelle")
log("  • Variations d'images")

-- Avantages Diffusion
log("\n✨ Avantages vs GAN:")
log("  ✓ Training plus stable")
log("  ✓ Meilleure qualité d'images")
log("  ✓ Mode collapse évité")
log("  ✓ Contrôle précis du processus")
log("  ⚠ Génération plus lente")
log("  ⚠ Plus coûteux en calcul")

-- Variantes
log("\n🔀 Variantes:")
log("  • DDPM: Denoising Diffusion Probabilistic Models")
log("  • DDIM: Denoising Diffusion Implicit Models (rapide)")
log("  • Stable Diffusion: Text-to-image avec CLIP")
log("  • Latent Diffusion: Travaille dans l'espace latent")

-- Sauvegarder
log("\n💾 Sauvegarde du modèle...")
os.execute("mkdir -p checkpoints")
Mimir.Serialization.save("checkpoints/diffusion_demo.safetensors", "safetensors")
log("✓ Modèle sauvegardé")

log("\n✓ Demo Diffusion terminé!")
log("=" .. string.rep("=", 78))
