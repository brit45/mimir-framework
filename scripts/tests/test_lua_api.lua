#!/usr/bin/env lua5.3
-- Script de démonstration de l'API Lua complète
-- Mímir Framework - Model & ModelArchitectures API

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Mímir Framework - API Lua Complète")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

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
local tokenizer = (type(_G.Mimir) == "table" and type(Mimir.Tokenizer) == "table") and Mimir.Tokenizer or _G.tokenizer

-- ============================================================================
-- Test 1: Vérification des capacités hardware
-- ============================================================================

print("\n🔧 Capacités Hardware:")
local hw = model.hardware_caps()

print("  AVX2:  " .. (hw.avx2 and "✓" or "✗"))
print("  FMA:   " .. (hw.fma and "✓" or "✗"))
print("  F16C:  " .. (hw.f16c and "✓" or "✗"))
print("  BMI2:  " .. (hw.bmi2 and "✓" or "✗"))

-- Activer l'accélération hardware
model.set_hardware(true)
print("\n✓ Accélération hardware activée")

-- ============================================================================
-- Test 2: Création d'un modèle UNet
-- ============================================================================

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Test 1: Architecture UNet")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

-- Configuration UNet (legacy-friendly)
local unet_config = {
    input_channels = 3,
    output_channels = 1,
    base_channels = 64,
    num_levels = 4
}

local unet_cfg, unet_warn = Arch.build_config("unet", unet_config)
if unet_warn then
    print("⚠️  " .. tostring(unet_warn))
end

local success, err = model.create("unet", unet_cfg)
if not success then
    print("❌ Erreur création modèle UNet: " .. (err or "inconnue"))
    os.exit(1)
end
print("✓ UNet créé via registre")

-- Allouer les paramètres
success, count = model.allocate_params()
if success then
    print(string.format("✓ Paramètres alloués: %d (%.2f MB)", count, count * 2 / 1024 / 1024))
else
    print("❌ Erreur allocation: " .. (count or "inconnue"))
end

-- Initialiser les poids
success, err = model.init_weights("he", 42)
if success then
    print("✓ Poids initialisés (méthode He)")
else
    print("❌ Erreur initialisation: " .. (err or "inconnue"))
end

local total = model.total_params()
print(string.format("📊 Nombre total de paramètres: %d", total))

-- ============================================================================
-- Test 3: Architecture Transformer (GPT-style)
-- ============================================================================

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Test 2: Architecture Transformer")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

local transformer_config = {
    vocab_size = 10000,
    d_model = 512,
    num_layers = 6,
    num_heads = 8
}

local transformer_cfg, transformer_warn = Arch.build_config("transformer", transformer_config)
if transformer_warn then
    print("⚠️  " .. tostring(transformer_warn))
end

success, err = model.create("transformer", transformer_cfg)
if success then
    print("✓ Transformer créé via registre")
    print(string.format("  • Vocab size: %d", transformer_config.vocab_size))
    print(string.format("  • Model dim: %d", transformer_config.d_model))
    print(string.format("  • Layers: %d", transformer_config.num_layers))
    print(string.format("  • Heads: %d", transformer_config.num_heads))
else
    print("❌ Erreur: " .. (err or "inconnue"))
end

success, count = model.allocate_params()
success = model.init_weights("xavier", 123)

local total_transformer = model.total_params()
print(string.format("📊 Nombre total de paramètres: %d (%.2f MB)", 
      total_transformer, total_transformer * 2 / 1024 / 1024))

-- ============================================================================
-- Test 4: Vision Transformer (ViT)
-- ============================================================================

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Test 3: Vision Transformer (ViT)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

local vit_config = {
    image_size = 224,
    patch_size = 16,
    num_classes = 1000,
    d_model = 768,
    num_layers = 12
}

local vit_cfg, vit_warn = Arch.build_config("vit", vit_config)
if vit_warn then
    print("⚠️  " .. tostring(vit_warn))
end

success, err = model.create("vit", vit_cfg)
if success then
    print("✓ ViT créé via registre")
    print(string.format("  • Image: %dx%d", vit_config.image_size, vit_config.image_size))
    print(string.format("  • Patches: %dx%d", vit_config.patch_size, vit_config.patch_size))
    print(string.format("  • Classes: %d", vit_config.num_classes))
else
    print("❌ Erreur: " .. (err or "inconnue"))
end

model.allocate_params()
model.init_weights("xavier", 456)

local total_vit = model.total_params()
print(string.format("📊 Nombre total de paramètres: %d (%.2f MB)", 
      total_vit, total_vit * 2 / 1024 / 1024))

-- ============================================================================
-- Test 5: VAE (Variational Autoencoder)
-- ============================================================================

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Test 4: Variational Autoencoder (VAE)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

local vae_config = {
    input_dim = 784,    -- 28x28 pour MNIST
    latent_dim = 64
}

local vae_cfg, vae_warn = Arch.build_config("vae", vae_config)
if vae_warn then
    print("⚠️  " .. tostring(vae_warn))
end

success, err = model.create("vae", vae_cfg)
if success then
    print("✓ VAE créé via registre")
    print(string.format("  • Input: %d", vae_config.input_dim))
    print(string.format("  • Latent: %d", vae_config.latent_dim))
else
    print("❌ Erreur: " .. (err or "inconnue"))
end

model.allocate_params()
model.init_weights("xavier", 789)

local total_vae = model.total_params()
print(string.format("📊 Nombre total de paramètres: %d", total_vae))

-- ============================================================================
-- Test 6: GAN (Generative Adversarial Network)
-- ============================================================================

local total_gen = 0
local total_disc = 0

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Test 5: GAN")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("⚠️  GAN n'est pas disponible dans l'API v2.3 (skip)")

-- ============================================================================
-- Test 7: Diffusion Model
-- ============================================================================

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Test 6: Diffusion Model (DDPM)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

local diffusion_config = {
    image_size = 32,
    base_channels = 128
}

local diffusion_cfg, diffusion_warn = Arch.build_config("diffusion", diffusion_config)
if diffusion_warn then
    print("⚠️  " .. tostring(diffusion_warn))
end

success, err = model.create("diffusion", diffusion_cfg)
if success then
    print("✓ Diffusion créé via registre")
    print(string.format("  • Image: %dx%d", diffusion_config.image_size, diffusion_config.image_size))
    print(string.format("  • Base channels: %d", diffusion_config.base_channels))
else
    print("❌ Erreur: " .. (err or "inconnue"))
end

model.allocate_params()
model.init_weights("he", 333)

local total_diff = model.total_params()
print(string.format("📊 Nombre total de paramètres: %d (%.2f MB)", 
      total_diff, total_diff * 2 / 1024 / 1024))

-- ============================================================================
-- Test 8: ResNet
-- ============================================================================

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Test 7: ResNet-50")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

local resnet_config = {
    num_classes = 1000
}

local resnet_cfg, resnet_warn = Arch.build_config("resnet", resnet_config)
if resnet_warn then
    print("⚠️  " .. tostring(resnet_warn))
end

success, err = model.create("resnet", resnet_cfg)
if success then
    print("✓ ResNet créé via registre")
    print(string.format("  • Classes: %d", resnet_config.num_classes))
else
    print("❌ Erreur: " .. (err or "inconnue"))
end

model.allocate_params()
model.init_weights("he", 444)

local total_resnet = model.total_params()
print(string.format("📊 Nombre total de paramètres: %d (%.2f MB)", 
      total_resnet, total_resnet * 2 / 1024 / 1024))

-- ============================================================================
-- Test 9: MobileNetV2
-- ============================================================================

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Test 8: MobileNetV2")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

local mobilenet_config = {
    num_classes = 1000,
    width_multiplier = 1.0
}

local mobilenet_cfg, mobilenet_warn = Arch.build_config("mobilenet", mobilenet_config)
if mobilenet_warn then
    print("⚠️  " .. tostring(mobilenet_warn))
end

success, err = model.create("mobilenet", mobilenet_cfg)
if success then
    print("✓ MobileNet créé via registre")
    print(string.format("  • Classes: %d", mobilenet_config.num_classes))
    print(string.format("  • Width: %.1fx", mobilenet_config.width_multiplier))
else
    print("❌ Erreur: " .. (err or "inconnue"))
end

model.allocate_params()
model.init_weights("he", 555)

local total_mobile = model.total_params()
print(string.format("📊 Nombre total de paramètres: %d (%.2f MB)", 
      total_mobile, total_mobile * 2 / 1024 / 1024))

-- ============================================================================
-- Test 10: Tokenizer API
-- ============================================================================

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Test 9: Tokenizer API")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

success = tokenizer.create(10000)
if success then
    print("✓ Tokenizer créé (vocab_size=10000)")
else
    print("❌ Erreur création tokenizer")
end

local vocab_size = tokenizer.vocab_size()
print(string.format("  • Taille du vocabulaire: %d", vocab_size))

-- Test tokenization
local text = "Bonjour le monde ! Ceci est un test."
local tokens = tokenizer.tokenize(text)
if tokens then
    print(string.format("  • Tokens: %s", table.concat(tokens, ", ")))
    
    local reconstructed = tokenizer.detokenize(tokens)
    print(string.format("  • Reconstructed: %s", reconstructed))
else
    print("  ⚠️  Tokenization non disponible")
end

-- ============================================================================
-- Récapitulatif
-- ============================================================================

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Récapitulatif")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

print("\n📊 Taille des architectures:")
print(string.format("  • UNet (4 levels):          %8d params (%.1f MB)", total, total * 2 / 1024 / 1024))
print(string.format("  • Transformer (6 layers):   %8d params (%.1f MB)", total_transformer, total_transformer * 2 / 1024 / 1024))
print(string.format("  • ViT (12 layers):          %8d params (%.1f MB)", total_vit, total_vit * 2 / 1024 / 1024))
print(string.format("  • VAE (64 latent):          %8d params", total_vae))
print(string.format("  • GAN Generator:            %8d params", total_gen))
print(string.format("  • GAN Discriminator:        %8d params", total_disc))
print(string.format("  • Diffusion (DDPM):         %8d params (%.1f MB)", total_diff, total_diff * 2 / 1024 / 1024))
print(string.format("  • ResNet-50:                %8d params (%.1f MB)", total_resnet, total_resnet * 2 / 1024 / 1024))
print(string.format("  • MobileNetV2:              %8d params (%.1f MB)", total_mobile, total_mobile * 2 / 1024 / 1024))

print("\n✅ Tests d'API Lua terminés avec succès!")
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
