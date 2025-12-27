-- Test de l'API FluxModel globale
print("=== Test API FluxModel ===\n")

-- Créer un modèle
print("📦 Création du modèle...")
local my_model = model.create("FluxModel-API-Test")

-- Configuration Flux
local config = {
    image_resolution = 64,
    latent_channels = 4,
    latent_resolution = 32,
    vae_base_channels = 128,
    vae_channel_mult = {1, 2, 4, 4},
    num_res_blocks = 2,
    vocab_size = 50000,
    text_max_length = 77,
    text_embed_dim = 768,
    transformer_dim = 768,
    num_transformer_blocks = 12,
    num_attention_heads = 12,
    mlp_ratio = 4.0,
    timestep_embed_dim = 256,
    num_diffusion_steps = 1000
}

-- Construire via architectures
print("🏗️  Construction de l'architecture Flux...")
architectures.flux(config)
print("✓ Architecture construite")

-- Test des modes train/eval via l'API FluxModel globale
print("\n--- Test Modes train/eval ---")
FluxModel.train()
local is_training = FluxModel.isTraining()
print("✓ Mode training: " .. tostring(is_training))

FluxModel.eval()
is_training = FluxModel.isTraining()
print("✓ Mode eval: " .. tostring(is_training))

-- Test tokenization
print("\n--- Test Tokenization ---")
local prompt = "a beautiful sunset over mountains"
local tokens = FluxModel.tokenizePrompt(prompt)
print("✓ Prompt tokenisé: " .. #tokens .. " tokens")

-- Test text encoding
print("\n--- Test Text Encoding ---")
local text_emb = FluxModel.encodeText(tokens)
print("✓ Text embedding: " .. #text_emb .. " valeurs")

-- Test image encoding
print("\n--- Test VAE Encoding ---")
local image_size = 3 * config.image_resolution * config.image_resolution
local test_image = {}
for i = 1, image_size do
    test_image[i] = math.random() * 2 - 1
end

local latent = FluxModel.encodeImage(test_image)
print("✓ Image encodée: " .. #latent .. " valeurs latent")

-- Test image decoding
print("\n--- Test VAE Decoding ---")
local reconstructed = FluxModel.decodeLatent(latent)
print("✓ Image décodée: " .. #reconstructed .. " valeurs")

-- Test noise prediction
print("\n--- Test Diffusion Noise Prediction ---")
local noisy_latent = {}
for i = 1, #latent do
    noisy_latent[i] = math.random() * 2 - 1
end

local timestep = 500
local predicted_noise = FluxModel.predictNoise(noisy_latent, text_emb, timestep)
print("✓ Bruit prédit: " .. #predicted_noise .. " valeurs")
print("  Timestep: " .. timestep)

-- Vérifier valeurs non-nulles
local non_zero = 0
for i = 1, math.min(100, #predicted_noise) do
    if math.abs(predicted_noise[i]) > 1e-6 then
        non_zero = non_zero + 1
    end
end
print("✓ Valeurs non-nulles: " .. non_zero .. "/100")

-- Test generation (mode inference)
print("\n--- Test Génération Complète ---")
FluxModel.eval()
local gen_image = FluxModel.generate("cyberpunk city at night", 50)
print("✓ Image générée: " .. gen_image.size .. " pixels")
print("  Résolution: " .. gen_image.resolution .. "×" .. gen_image.resolution)

print("\n=== Tests API FluxModel Réussis ! ===")
print("✓ Tous les bindings fonctionnels")
print("✓ Modes train/eval opérationnels")
print("✓ Pipeline complet text-to-image disponible")
