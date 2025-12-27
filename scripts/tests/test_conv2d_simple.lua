-- ╔═══════════════════════════════════════════════════════════════╗
-- ║  Test Conv2D Simplifié - Validation                          ║
-- ╚═══════════════════════════════════════════════════════════════╝

print("\n╔═══════════════════════════════════════════════════════════════╗")
print("║   Mímir - Test Conv2D Simplifié                            ║")
print("╚═══════════════════════════════════════════════════════════════╝\n")

-- Configuration
local cfg = {
    in_channels = 3,
    out_channels = 16,
    height = 8,
    width = 8,
    kernel = 3,
    seed = 42
}

math.randomseed(cfg.seed)

local vec_size = cfg.in_channels * cfg.height * cfg.width
print(string.format("📊 Configuration:"))
print(string.format("   • Input: %dx%dx%d", cfg.height, cfg.width, cfg.in_channels))
print(string.format("   • Output channels: %d", cfg.out_channels))
print(string.format("   • Kernel size: %dx%d", cfg.kernel, cfg.kernel))
print(string.format("   • Vector size: %d\n", vec_size))

-- Créer le modèle
model.create("test_conv2d", cfg)

-- Calculer params: weights + bias
local num_params = (cfg.kernel * cfg.kernel * cfg.in_channels * cfg.out_channels) + cfg.out_channels
print(string.format("📦 Paramètres: %d\n", num_params))

-- Ajouter la couche Conv2D
local ok = model.push_layer("conv1", "Conv2d", num_params)
if not ok then
    print("❌ Échec push_layer")
    return
end
print("✓ Couche Conv2D ajoutée")

-- Allouer et initialiser
ok = model.allocate_params()
if not ok then
    print("❌ Échec allocate_params")
    return
end
print("✓ Paramètres alloués")

ok = model.init_weights("xavier_uniform")
if not ok then
    print("❌ Échec init_weights")
    return
end
print("✓ Poids initialisés\n")

-- Créer des données de test
print("▶ Test Forward pass...")
local input = {}
for i = 1, vec_size do
    input[i] = (math.random() * 2.0 - 1.0)
end

-- Forward
local output = model.forward(input)
if not output then
    print("❌ Forward pass échoué")
    return
end

print(string.format("✓ Forward pass réussi!"))
print(string.format("   • Input size: %d", #input))
print(string.format("   • Output size: %d", #output))

-- Statistiques
local sum, min_val, max_val = 0, math.huge, -math.huge
for i = 1, #output do
    local v = output[i]
    sum = sum + v
    if v < min_val then min_val = v end
    if v > max_val then max_val = v end
end
local mean = sum / #output

print(string.format("\n📈 Statistiques:"))
print(string.format("   • Min: %.4f", min_val))
print(string.format("   • Max: %.4f", max_val))
print(string.format("   • Mean: %.4f", mean))

-- Test backward
print("\n▶ Test Backward pass...")
local grad = {}
for i = 1, #output do
    grad[i] = 1.0
end

ok = model.backward(grad)
if not ok then
    print("❌ Backward pass échoué")
    return
end
print("✓ Backward pass réussi!")

print("\n╔═══════════════════════════════════════════════════════════════╗")
print("║   ✅ Test Conv2D complété avec succès!                      ║")
print("╚═══════════════════════════════════════════════════════════════╝")
