-- Test de la limite mémoire à 10 Go avec FluxModel
print("=== Test Limite Mémoire 10 Go ===\n")

-- Bonne pratique: configurer l'allocateur tôt (configure aussi la limite du guard)
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

-- Vérifier la limite
local limit = MemoryGuard.getLimit()
print("✓ Limite configurée: " .. string.format("%.2f", limit / 1024 / 1024 / 1024) .. " GB")

-- Créer un modèle Flux via l'API model
do
    local ok, err = model.create("FluxModel-Test")
    if ok == false then
        print("❌ model.create failed: " .. tostring(err))
        os.exit(1)
    end
end

-- Configuration Flux
local config = {
    image_resolution = 64,  -- Réduit pour tester
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

-- Construire le modèle Flux
print("\n📦 Construction du modèle Flux...")
local ok_arch, err_arch = architectures.flux(config)
if ok_arch == false then
    print("❌ architectures.flux failed: " .. tostring(err_arch))
    os.exit(1)
end
print("✓ Modèle construit")

-- Vérifier l'utilisation mémoire après construction
local mem_after_build = MemoryGuard.getCurrentUsage()
print("   RAM après construction: " .. string.format("%.2f", mem_after_build / 1024 / 1024 / 1024) .. " GB")

-- Allouer les paramètres
print("\n⚙️  Allocation des paramètres...")
model.allocate_params()
print("✓ Paramètres alloués")

local mem_after_alloc = MemoryGuard.getCurrentUsage()
print("   RAM après allocation: " .. string.format("%.2f", mem_after_alloc / 1024 / 1024 / 1024) .. " GB")

-- Initialiser les poids
print("\n🎲 Initialisation des poids...")
model.init_weights("xavier")
print("✓ Poids initialisés")

local mem_after_init = MemoryGuard.getCurrentUsage()
print("   RAM après initialisation: " .. string.format("%.2f", mem_after_init / 1024 / 1024 / 1024) .. " GB")

-- Test d'un forward pass simple
print("\n🔄 Test forward pass...")
local input_size = 3 * config.image_resolution * config.image_resolution
local test_input = {}
for i = 1, input_size do
    test_input[i] = math.random() * 2 - 1
end

local output, forward_err = model.forward(test_input, false)
if output == nil then
    print("❌ model.forward failed: " .. tostring(forward_err))
    os.exit(1)
end
print("✓ Forward pass réussi")
print("   Output size: " .. #output)

local mem_after_forward = MemoryGuard.getCurrentUsage()
print("   RAM après forward: " .. string.format("%.2f", mem_after_forward / 1024 / 1024 / 1024) .. " GB")

-- Rapport final
print("\n=== Rapport Mémoire Final ===")
local final_mem = MemoryGuard.getCurrentUsage()
local peak_mem = MemoryGuard.getPeakUsage()
local limit_mem = MemoryGuard.getLimit()

print("💾 Utilisation RAM:")
print("   Courante: " .. string.format("%.2f", final_mem / 1024 / 1024 / 1024) .. " GB")
print("   Pic: " .. string.format("%.2f", peak_mem / 1024 / 1024 / 1024) .. " GB")
print("   Limite: " .. string.format("%.2f", limit_mem / 1024 / 1024 / 1024) .. " GB")
print("   Utilisation: " .. string.format("%.1f", (peak_mem / limit_mem) * 100) .. "%")

if peak_mem < limit_mem then
    print("\n✅ Limite de 10 Go respectée !")
    print("   Marge restante: " .. string.format("%.2f", (limit_mem - peak_mem) / 1024 / 1024 / 1024) .. " GB")
else
    print("\n⚠️  Limite de 10 Go dépassée !")
    print("   Dépassement: " .. string.format("%.2f", (peak_mem - limit_mem) / 1024 / 1024 / 1024) .. " GB")
end

-- Statistiques détaillées
print("\n📊 Statistiques MemoryGuard:")
MemoryGuard.printStats()

print("\n=== Test terminé ===")
