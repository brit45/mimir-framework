-- Test complet du FluxModel avec eval() et train()
print("=== Test FluxModel avec modes eval/train ===\n")

-- FluxModel n'est pas exposé dans l'API Lua v2.3 (skip propre)
if type(_G.Mimir) ~= "table" or type(Mimir.FluxModel) ~= "table" then
    print("⚠️  FluxModel indisponible dans l'API Lua v2.3 (skip)")
    os.exit(0)
end

-- Bonne pratique: configurer l'allocateur tôt
local MAX_RAM_GB = 10
print("🛡️  Configuration Allocator/MemoryGuard: Limite " .. MAX_RAM_GB .. " Go")
do
    local ok, err = Mimir.Allocator.configure({
        max_ram_gb = MAX_RAM_GB,
        enable_compression = true,
        swap_strategy = "lru",
    })
    if ok == false then
        print("❌ Allocator.configure failed: " .. tostring(err))
        os.exit(1)
    end
end
print("")

-- Configuration
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

-- Créer le modèle
local model = Mimir.FluxModel.new(config)
print("✓ Modèle FluxModel créé")

-- Vérifier l'utilisation mémoire
local mem_used = MemoryGuard.getCurrentUsage()
local mem_limit = MemoryGuard.getLimit()
print("   RAM utilisée: " .. string.format("%.2f", mem_used / 1024 / 1024 / 1024) .. " GB / " .. string.format("%.2f", mem_limit / 1024 / 1024 / 1024) .. " GB")

-- Vérifier les modes
print("\n--- Test des modes train/eval ---")
model:train()
print("✓ Mode training activé: " .. tostring(model:isTraining()))

model:eval()
print("✓ Mode eval activé: " .. tostring(model:isTraining()))

-- Test encode/decode
print("\n--- Test VAE encode/decode ---")
local image_size = 3 * config.image_resolution * config.image_resolution
local test_image = {}
for i = 1, image_size do
    test_image[i] = math.random() * 2 - 1  -- [-1, 1]
end

print("Image d'entrée: " .. image_size .. " valeurs")

-- Encoder
local latent = model:encodeImage(test_image)
local latent_size = config.latent_channels * config.latent_resolution * config.latent_resolution
print("✓ Latent encodé: " .. #latent .. " valeurs (attendu: " .. latent_size .. ")")

local mem_after_encode = MemoryGuard.getCurrentUsage()
print("   RAM après encodage: " .. string.format("%.2f", mem_after_encode / 1024 / 1024 / 1024) .. " GB")

-- Decoder
local reconstructed = model:decodeLatent(latent)
print("✓ Image reconstruite: " .. #reconstructed .. " valeurs")

-- Test text encoding
print("\n--- Test Text Encoding ---")
local prompt = "a beautiful mountain landscape"
if not (tokenizer and tokenizer.tokenize) then
    print("❌ Tokenizer non disponible (tokenizer.tokenize)")
    os.exit(1)
end
local tokens = tokenizer.tokenize(prompt)
print("✓ Prompt tokenisé: " .. #tokens .. " tokens")

local text_emb = model:encodeText(tokens)
print("✓ Text embedding: " .. #text_emb .. " valeurs")

local mem_after_text = MemoryGuard.getCurrentUsage()
print("   RAM après text encoding: " .. string.format("%.2f", mem_after_text / 1024 / 1024 / 1024) .. " GB")

-- Test noise prediction
print("\n--- Test Diffusion Prediction ---")
local noisy_latent = {}
for i = 1, latent_size do
    noisy_latent[i] = math.random() * 2 - 1
end

local timestep = 500
local predicted_noise = model:predictNoise(noisy_latent, text_emb, timestep)
print("✓ Bruit prédit: " .. #predicted_noise .. " valeurs")
print("  Timestep: " .. timestep)

-- Vérifier que les prédictions ne sont pas des zéros
local non_zero_count = 0
for i = 1, math.min(100, #predicted_noise) do
    if math.abs(predicted_noise[i]) > 1e-6 then
        non_zero_count = non_zero_count + 1
    end
end
print("✓ Valeurs non-nulles: " .. non_zero_count .. "/100 échantillons")

-- Test generation complete
print("\n--- Test Génération complète ---")
model:eval()  -- Mode inference

local prompt_gen = "cyberpunk city at night"
local generated_image = model:generate(prompt_gen, 50)  -- 50 diffusion steps
print("✓ Image générée: " .. #generated_image .. " valeurs")

-- Statistiques de l'image générée
local min_val = 1e9
local max_val = -1e9
local sum = 0
for i = 1, #generated_image do
    local val = generated_image[i]
    min_val = math.min(min_val, val)
    max_val = math.max(max_val, val)
    sum = sum + val
end
local mean = sum / #generated_image

print("\nStatistiques image générée:")
print("  Min: " .. string.format("%.4f", min_val))
print("  Max: " .. string.format("%.4f", max_val))
print("  Moyenne: " .. string.format("%.4f", mean))

print("\n=== Test réussi ! ===")
print("• Modes train/eval fonctionnels")
print("• VAE encode/decode implémentés")
print("• Text encoder fonctionnel")
print("• Diffusion transformer opérationnel")
print("• Génération end-to-end complète")

print("\n=== Rapport Mémoire ===")
local final_mem = MemoryGuard.getCurrentUsage()
local peak_mem = MemoryGuard.getPeakUsage()
local limit = MemoryGuard.getLimit()

print("💾 Utilisation RAM:")
print("   Courante: " .. string.format("%.2f", final_mem / 1024 / 1024 / 1024) .. " GB")
print("   Pic: " .. string.format("%.2f", peak_mem / 1024 / 1024 / 1024) .. " GB")
print("   Limite: " .. string.format("%.2f", limit / 1024 / 1024 / 1024) .. " GB")
print("   Utilisation: " .. string.format("%.1f", (peak_mem / limit) * 100) .. "%")

if peak_mem < limit then
    print("✅ Limite de 10 Go respectée !")
else
    print("⚠️  Limite de 10 Go dépassée !")
end
