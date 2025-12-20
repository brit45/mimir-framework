#!/usr/bin/env lua5.3
-- Script de démonstration de l'API Lua complète
-- Mímir Framework - Model & ModelArchitectures API

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Mímir Framework - API Lua Complète")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

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

local success, err = model.create("unet")
if not success then
    print("❌ Erreur création modèle: " .. (err or "inconnue"))
    os.exit(1)
end
print("✓ Modèle créé")

-- Configuration UNet
local unet_config = {
    input_channels = 3,
    output_channels = 1,
    base_channels = 64,
    num_levels = 4
}

success, err = architectures.unet(unet_config)
if not success then
    print("❌ Erreur construction UNet: " .. (err or "inconnue"))
    os.exit(1)
end
print("✓ Architecture UNet construite")

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

-- Créer un nouveau modèle
model.create("transformer")
print("✓ Modèle créé")

local transformer_config = {
    vocab_size = 10000,
    d_model = 512,
    num_layers = 6,
    num_heads = 8
}

success, err = architectures.transformer(transformer_config)
if success then
    print("✓ Architecture Transformer construite")
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

model.create("vit")
print("✓ Modèle créé")

local vit_config = {
    image_size = 224,
    patch_size = 16,
    num_classes = 1000,
    d_model = 768,
    num_layers = 12
}

success, err = architectures.vit(vit_config)
if success then
    print("✓ Architecture ViT construite")
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

model.create("vae")
print("✓ Modèle créé")

local vae_config = {
    input_dim = 784,    -- 28x28 pour MNIST
    latent_dim = 64
}

success, err = architectures.vae(vae_config)
if success then
    print("✓ Architecture VAE construite")
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

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Test 5: GAN (Generator)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

model.create("gan_generator")
print("✓ Modèle créé")

local gan_config = {
    latent_dim = 100,
    image_size = 64
}

success, err = architectures.gan("generator", gan_config)
if success then
    print("✓ Architecture GAN Generator construite")
    print(string.format("  • Latent: %d", gan_config.latent_dim))
    print(string.format("  • Output: %dx%d", gan_config.image_size, gan_config.image_size))
else
    print("❌ Erreur: " .. (err or "inconnue"))
end

model.allocate_params()
model.init_weights("he", 111)

local total_gen = model.total_params()
print(string.format("📊 Paramètres Generator: %d", total_gen))

-- Discriminator
print("\n  Test 5b: GAN (Discriminator)")

model.create("gan_discriminator")
success = architectures.gan("discriminator", gan_config)
if success then
    print("✓ Architecture GAN Discriminator construite")
end

model.allocate_params()
model.init_weights("he", 222)

local total_disc = model.total_params()
print(string.format("📊 Paramètres Discriminator: %d", total_disc))

-- ============================================================================
-- Test 7: Diffusion Model
-- ============================================================================

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Test 6: Diffusion Model (DDPM)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

model.create("diffusion")
print("✓ Modèle créé")

local diffusion_config = {
    image_size = 32,
    base_channels = 128
}

success, err = architectures.diffusion(diffusion_config)
if success then
    print("✓ Architecture Diffusion construite")
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

model.create("resnet")
print("✓ Modèle créé")

local resnet_config = {
    num_classes = 1000
}

success, err = architectures.resnet(resnet_config)
if success then
    print("✓ Architecture ResNet-50 construite")
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

model.create("mobilenet")
print("✓ Modèle créé")

local mobilenet_config = {
    num_classes = 1000,
    width_multiplier = 1.0
}

success, err = architectures.mobilenet(mobilenet_config)
if success then
    print("✓ Architecture MobileNetV2 construite")
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
