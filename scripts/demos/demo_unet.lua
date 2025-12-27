-- ============================================================================
-- Mímir Framework - Demo UNet
-- Segmentation d'images avec U-Net
-- ============================================================================

log("=" .. string.rep("=", 78))
log("🎨 Demo UNet - Segmentation d'Images")
log("=" .. string.rep("=", 78))

-- Configuration système
log("\n🔧 Configuration...")
allocator.configure({max_ram_gb = 10.0, enable_compression = true})
local hw = model.hardware_caps()
model.set_hardware(true)
log("✓ Allocateur et hardware configurés")

-- Configuration UNet
local config = {
    input_channels = 3,
    output_channels = 1,
    base_channels = 64,
    num_levels = 4,
    blocks_per_level = 2,
    use_attention = true,
    use_residual = true,
    use_batchnorm = true,
    dropout = 0.1
}

-- Créer le modèle UNet
log("\n🏗️  Création du modèle UNet...")
local success, err = model.create("unet_segmentation")
if not success then
    log("❌ Erreur: " .. (err or "inconnue"))
    return
end

-- Construire avec architectures API
success, err = architectures.unet(config)
if not success then
    log("❌ Erreur architecture: " .. (err or "inconnue"))
    return
end

log("✓ Architecture UNet construite")

-- Allouer et initialiser
success, params = model.allocate_params()
if not success then
    log("❌ Erreur allocation")
    return
end

model.init_weights("he", 42)
log("✓ Modèle construit: " .. params .. " paramètres")
log("  Mémoire: " .. string.format("%.2f MB", params * 4 / 1024 / 1024))

-- Afficher config
log("\n📋 Configuration:")
log("  Input:  " .. config.input_channels .. " channels (RGB)")
log("  Output: " .. config.output_channels .. " channel (mask)")
log("  Depth:  " .. config.num_levels .. " levels")
log("  Blocks: " .. config.blocks_per_level .. " per level")
log("  Attention: " .. (config.use_attention and "Yes" or "No"))
log("  Residual:  " .. (config.use_residual and "Yes" or "No"))

-- Estimation architecture
log("\n🏛️  Architecture (estimation):")
local total_blocks = config.num_levels * 2 * config.blocks_per_level
log("  Encoder: " .. config.num_levels * config.blocks_per_level .. " blocks")
log("  Decoder: " .. config.num_levels * config.blocks_per_level .. " blocks")
log("  Total:   " .. total_blocks .. " conv blocks")

-- Usage pour segmentation
log("\n💡 Cas d'usage:")
log("  • Segmentation médicale (tumeurs, organes)")
log("  • Détection d'objets (masques d'instances)")
log("  • Segmentation sémantique")
log("  • Inpainting d'images")

-- Sauvegarder
log("\n💾 Sauvegarde du modèle...")
os.execute("mkdir -p checkpoints")
model.save("checkpoints/unet_demo")
log("✓ Modèle sauvegardé")

log("\n✓ Demo UNet terminé!")
log("=" .. string.rep("=", 78))
