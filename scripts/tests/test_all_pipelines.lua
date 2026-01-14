-- ============================================================================
-- Mímir Framework - Test Complet Pipeline API
-- Test de tous les modèles disponibles
-- ============================================================================

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
local Pipeline = require("pipeline_api")


log("=" .. string.rep("=", 78))
log("🧪 Test Complet Pipeline API - Tous les Modèles")
log("=" .. string.rep("=", 78))

-- ============================================================================
-- Manager de pipelines
-- ============================================================================

local manager = Pipeline.PipelineManager:new()

-- ============================================================================
-- Test 1: Transformer (NLP)
-- ============================================================================

log("\n" .. string.rep("-", 78))
log("1️⃣  TRANSFORMER - Natural Language Processing")
log(string.rep("-", 78))

local transformer = Pipeline.Transformer({
    vocab_size = 5000,
    embed_dim = 256,
    num_layers = 4,
    num_heads = 8,
    d_ff = 1024,
    max_seq_len = 128,
    dropout = 0.1,
    model_type = "transformer"
})

local ok, params = transformer:build()
if ok then
    log("✓ Transformer construit: " .. params .. " paramètres")
    manager:add("transformer", transformer)
else
    log("❌ Échec Transformer")
end

-- ============================================================================
-- Test 2: UNet (Segmentation)
-- ============================================================================

log("\n" .. string.rep("-", 78))
log("2️⃣  U-NET - Image Segmentation")
log(string.rep("-", 78))

local unet = Pipeline.UNet({
    input_channels = 3,
    output_channels = 1,
    base_channels = 64,
    num_levels = 4,
    blocks_per_level = 2,
    use_attention = true,
    use_residual = true,
    dropout = 0.1
})

ok, params = unet:build()
if ok then
    log("✓ UNet construit: " .. params .. " paramètres")
    manager:add("unet", unet)
else
    log("❌ Échec UNet")
end

-- ============================================================================
-- Test 3: VAE (Generation)
-- ============================================================================

log("\n" .. string.rep("-", 78))
log("3️⃣  VAE - Variational Autoencoder")
log(string.rep("-", 78))

local vae = Pipeline.VAE({
    input_dim = 784,
    latent_dim = 128,
    hidden_dims = {512, 256},
    beta = 1.0
})

ok, params = vae:build()
if ok then
    log("✓ VAE construit: " .. params .. " paramètres")
    manager:add("vae", vae)
else
    log("❌ Échec VAE")
end

-- ============================================================================
-- Test 4: ViT (Vision Transformer)
-- ============================================================================

log("\n" .. string.rep("-", 78))
log("4️⃣  ViT - Vision Transformer")
log(string.rep("-", 78))

local vit = Pipeline.ViT({
    image_size = 224,
    patch_size = 16,
    num_classes = 1000,
    embed_dim = 768,
    num_layers = 12,
    num_heads = 12,
    d_ff = 3072,
    dropout = 0.1
})

ok, params = vit:build()
if ok then
    log("✓ ViT construit: " .. params .. " paramètres")
    manager:add("vit", vit)
else
    log("❌ Échec ViT")
end

-- ============================================================================
-- Test 5: GAN (Adversarial)
-- ============================================================================

log("\n" .. string.rep("-", 78))
log("5️⃣  GAN - Generative Adversarial Network")
log(string.rep("-", 78))

local gan = Pipeline.GAN({
    latent_dim = 100,
    image_channels = 3,
    image_size = 64,
    gen_channels = 64,
    disc_channels = 64
})

ok = gan:build()
if ok then
    log("✓ GAN construit (Generator + Discriminator)")
    manager:add("gan", gan)
else
    log("❌ Échec GAN")
end

-- ============================================================================
-- Test 6: Diffusion (Denoising)
-- ============================================================================

log("\n" .. string.rep("-", 78))
log("6️⃣  DIFFUSION - Denoising Diffusion Models")
log(string.rep("-", 78))

local diffusion = Pipeline.Diffusion({
    image_channels = 3,
    image_size = 256,
    timesteps = 1000,
    model_channels = 128,
    num_res_blocks = 2
})

ok, params = diffusion:build()
if ok then
    log("✓ Diffusion construit: " .. params .. " paramètres")
    manager:add("diffusion", diffusion)
else
    log("❌ Échec Diffusion")
end

-- ============================================================================
-- Test 7: ResNet (Classification)
-- ============================================================================

log("\n" .. string.rep("-", 78))
log("7️⃣  RESNET - Residual Network")
log(string.rep("-", 78))

local resnet = Pipeline.ResNet({
    num_classes = 1000,
    layers = {3, 4, 6, 3},  -- ResNet-50
    base_channels = 64,
    use_bottleneck = true
})

ok, params = resnet:build()
if ok then
    log("✓ ResNet construit: " .. params .. " paramètres")
    manager:add("resnet", resnet)
else
    log("❌ Échec ResNet")
end

-- ============================================================================
-- Test 8: MobileNet (Mobile)
-- ============================================================================

log("\n" .. string.rep("-", 78))
log("8️⃣  MOBILENET - Efficient Mobile Architecture")
log(string.rep("-", 78))

local mobilenet = Pipeline.MobileNet({
    num_classes = 1000,
    width_mult = 1.0,
    input_size = 224
})

ok, params = mobilenet:build()
if ok then
    log("✓ MobileNet construit: " .. params .. " paramètres")
    manager:add("mobilenet", mobilenet)
else
    log("❌ Échec MobileNet")
end

-- ============================================================================
-- Résumé
-- ============================================================================

log("\n" .. string.rep("=", 78))
log("📊 RÉSUMÉ - Pipelines Construits")
log(string.rep("=", 78))

manager:list()

-- Statistiques par domaine
log("\n🎯 Par domaine:")
log("  NLP:        Transformer")
log("  Vision:     UNet, ViT, ResNet, MobileNet")
log("  Generation: VAE, GAN, Diffusion")

-- Recommandations CPU
log("\n🖥️  Recommandations CPU-only:")
log("  ✓ MobileNet: Meilleur rapport vitesse/précision")
log("  ✓ ResNet: Performance éprouvée")
log("  ✓ Transformer: Excellent pour NLP")
log("  ⚠ ViT: Nécessite plus de calcul")
log("  ⚠ Diffusion: Génération lente (1000 steps)")

-- Cas d'usage
log("\n💡 Cas d'usage recommandés:")
log("  • Classification images: MobileNet ou ResNet")
log("  • Segmentation: UNet")
log("  • Génération texte: Transformer")
log("  • Génération images: VAE ou Diffusion")
log("  • Détection: ResNet (backbone)")

-- Tailles modèles
log("\n💾 Tailles (estimation):")
log("  Légers:  MobileNet (<5M params)")
log("  Moyens:  Transformer, ResNet (5-25M params)")
log("  Lourds:  ViT, Diffusion (>25M params)")

log("\n✓ Test complet terminé!")
log("  8 architectures disponibles")
log("  Pipeline API unifié")
log("  Prêt pour entraînement CPU-only")
log("=" .. string.rep("=", 78))
