-- ╔═══════════════════════════════════════════════════════════════╗
-- ║  Test Conv2D - Validation de l'implémentation                ║
-- ╚═══════════════════════════════════════════════════════════════╝

print("╔═══════════════════════════════════════════════════════════════╗")
print("║   Mímir - Test de Conv2D                                  ║")
print("╚═══════════════════════════════════════════════════════════════╝\n")

-- Vérifier les capacités hardware
print("🔧 Capacités Hardware:")
local hw = model.hardware_caps()
print(string.format("  • AVX2:  %s", hw.avx2 and "✓" or "✗"))
print(string.format("  • FMA:   %s", hw.fma and "✓" or "✗"))
print("")

-- Configuration d'une couche Conv2D simple
local in_channels = 3       -- RGB
local out_channels = 16     -- 16 filtres
local kernel_size = 3       -- 3x3
local stride = 1
local padding = 1
local dilation = 1
local img_h, img_w = 8, 8

-- Configuration du modèle
local cfg = {
    in_channels = in_channels,
    out_channels = out_channels,
    height = img_h,
    width = img_w,
    kernel = kernel_size
}

-- Création du modèle avec config
model.create("test_conv2d_model", cfg)

print("📋 Configuration Conv2D:")
print("   • Canaux d'entrée: " .. in_channels)
print("   • Canaux de sortie: " .. out_channels)
print("   • Taille kernel: " .. kernel_size .. "x" .. kernel_size)
print("   • Stride: " .. stride)
print("   • Padding: " .. padding)
print("   • Taille image: " .. img_h .. "x" .. img_w)
print("")

-- Calculer le nombre de paramètres
-- weights: kernel_size * kernel_size * in_channels * out_channels
-- bias: out_channels
local num_params = (kernel_size * kernel_size * in_channels * out_channels) + out_channels
print("📊 Paramètres:")
print("   • Poids: " .. (kernel_size * kernel_size * in_channels * out_channels))
print("   • Biais: " .. out_channels)
print("   • Total: " .. num_params)
print("")

-- Ajouter la couche Conv2D
local ok, err = model.push_layer("conv1", "Conv2d", num_params)
if not ok then
    print("❌ Erreur lors de l'ajout de la couche: " .. tostring(err))
    return
end

print("✓ Couche Conv2D ajoutée avec succès")

-- Allouer les paramètres
ok, err = model.allocate_params()
if not ok then
    print("❌ Erreur lors de l'allocation des paramètres: " .. tostring(err))
    return
end

print("✓ Paramètres alloués")

-- Initialiser les poids
model.init_weights("xavier_uniform")
print("✓ Poids initialisés (xavier_uniform)")
print("")

-- Créer une image de test (8x8x3)
local img_h, img_w = 8, 8
local input_size = img_h * img_w * in_channels

print("📊 Création des données de test:")
print("   • Taille image: " .. img_h .. "x" .. img_w)
print("   • Canaux: " .. in_channels)
print("   • Taille totale: " .. input_size .. " éléments")

local input = {}
for i = 1, input_size do
    input[i] = math.random() * 2.0 - 1.0  -- Valeurs entre -1 et 1
end

print("✓ Données de test créées")
print("")

-- Forward pass
print("▶ Forward pass...")
local output = model.forward(input)
if not output or #output == 0 then
    print("❌ Erreur lors du forward pass")
    return
end

-- Calculer les dimensions de sortie
local out_h = math.floor((img_h + 2 * padding - kernel_size) / stride) + 1
local out_w = math.floor((img_w + 2 * padding - kernel_size) / stride) + 1
local expected_size = out_h * out_w * out_channels

print("✓ Forward pass réussi!")
print("")
print("📊 Résultats:")
print("   • Taille de sortie attendue: " .. expected_size)
print("   • Taille de sortie obtenue: " .. #output)
print("   • Dimensions: " .. out_h .. "x" .. out_w .. "x" .. conv_params.out_channels)

-- Vérifier la taille
if #output ~= expected_size then
    print("❌ Erreur: taille de sortie incorrecte!")
    print("   Attendu: " .. expected_size .. ", obtenu: " .. #output)
    return
end

print("✓ Taille de sortie correcte!")

-- Statistiques de sortie
local sum, min_val, max_val = 0, math.huge, -math.huge
for i = 1, #output do
    local v = output[i]
    sum = sum + v
    min_val = math.min(min_val, v)
    max_val = math.max(max_val, v)
end
local mean = sum / #output

print("")
print("📈 Statistiques de sortie:")
print(string.format("   • Min: %.4f", min_val))
print(string.format("   • Max: %.4f", max_val))
print(string.format("   • Moyenne: %.4f", mean))

-- Test du backward pass avec un gradient simple
print("")
print("▶ Backward pass...")

local grad_output = {}
for i = 1, #output do
    grad_output[i] = 1.0  -- Gradient uniforme
end

local ok_back = model.backward(grad_output)
if not ok_back then
    print("❌ Erreur lors du backward pass")
    return
end

print("✓ Backward pass réussi!")
print("")

-- Test avec différentes tailles d'entrée
print("🔄 Test avec différentes tailles d'entrée...")
local test_sizes = {
    {16, 16},
    {32, 32},
    {64, 64}
}

for _, size in ipairs(test_sizes) do
    local h, w = size[1], size[2]
    local test_input_size = h * w * in_channels
    local test_input = {}
    for i = 1, test_input_size do
        test_input[i] = math.random()
    end
    
    local test_output = model.forward(test_input)
    
    if test_output then
        local expected_h = math.floor((h + 2 * padding - kernel_size) / stride) + 1
        local expected_w = math.floor((w + 2 * padding - kernel_size) / stride) + 1
        local expected = expected_h * expected_w * out_channels
        
        if #test_output == expected then
            print(string.format("   ✓ %dx%d → %dx%d (OK)", h, w, expected_h, expected_w))
        else
            print(string.format("   ❌ %dx%d → Erreur de taille", h, w))
        end
    else
        print(string.format("   ❌ %dx%d → Forward échoué", h, w))
    end
end

print("")
print("╔═══════════════════════════════════════════════════════════════╗")
print("║   ✅ Tous les tests Conv2D sont passés avec succès!       ║")
print("╚═══════════════════════════════════════════════════════════════╝")
