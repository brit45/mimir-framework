-- ╔═══════════════════════════════════════════════════════════════╗
-- ║  Test Dimensions Dynamiques Conv2D                           ║
-- ╚═══════════════════════════════════════════════════════════════╝

print("╔═══════════════════════════════════════════════════════════════╗")
print("║   Test Dimensions Dynamiques Conv2D                        ║")
print("╚═══════════════════════════════════════════════════════════════╝\n")

local function test_config(name, config)
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Test: " .. name)
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    -- Configuration
    print(string.format("   • Input: %dx%dx%d", config.height, config.width, config.in_channels))
    print(string.format("   • Output: %d channels", config.out_channels))
    print(string.format("   • Kernel: %dx%d", config.kernel, config.kernel))
    
    -- Créer le modèle
    model.create("test_" .. name, config)
    
    -- Calculer le nombre de paramètres
    local num_params = (config.kernel * config.kernel * config.in_channels * config.out_channels) 
                       + config.out_channels
    
    -- Ajouter la couche
    model.push_layer("conv1", "Conv2d", num_params)
    model.allocate_params()
    model.init_weights("xavier_uniform")
    
    -- Créer l'input
    local input_size = config.height * config.width * config.in_channels
    local input = {}
    for i = 1, input_size do
        input[i] = math.random() * 2.0 - 1.0
    end
    
    -- Forward pass
    local output = model.forward(input)
    
    if output and #output > 0 then
        print(string.format("   ✓ Forward pass OK - Output size: %d", #output))
        
        -- Calculer la taille attendue
        local expected_h = math.floor((config.height + 2 * (config.padding or 0) - config.kernel) / (config.stride or 1)) + 1
        local expected_w = math.floor((config.width + 2 * (config.padding or 0) - config.kernel) / (config.stride or 1)) + 1
        local expected_size = expected_h * expected_w * config.out_channels
        
        if #output == expected_size then
            print(string.format("   ✓ Dimensions correctes: %dx%dx%d", expected_h, expected_w, config.out_channels))
        else
            print(string.format("   ⚠ Dimensions incorrectes: attendu %d, obtenu %d", expected_size, #output))
        end
    else
        print("   ❌ Forward pass échoué")
    end
    
    print("")
end

-- Test 1: Petite image 8x8
test_config("small_8x8", {
    in_channels = 3,
    out_channels = 16,
    height = 8,
    width = 8,
    kernel = 3,
    stride = 1,
    padding = 1
})

-- Test 2: Image moyenne 32x32
test_config("medium_32x32", {
    in_channels = 3,
    out_channels = 32,
    height = 32,
    width = 32,
    kernel = 3,
    stride = 1,
    padding = 1
})

-- Test 3: Grande image 64x64 avec plus de canaux
test_config("large_64x64", {
    in_channels = 64,
    out_channels = 128,
    height = 64,
    width = 64,
    kernel = 3,
    stride = 1,
    padding = 1
})

-- Test 4: Stride 2 (downsampling)
test_config("stride2_32x32", {
    in_channels = 16,
    out_channels = 32,
    height = 32,
    width = 32,
    kernel = 3,
    stride = 2,
    padding = 1
})

-- Test 5: Kernel 5x5
test_config("kernel5_16x16", {
    in_channels = 8,
    out_channels = 16,
    height = 16,
    width = 16,
    kernel = 5,
    stride = 1,
    padding = 2
})

-- Test 6: Très petite image 4x4
test_config("tiny_4x4", {
    in_channels = 1,
    out_channels = 8,
    height = 4,
    width = 4,
    kernel = 3,
    stride = 1,
    padding = 1
})

print("╔═══════════════════════════════════════════════════════════════╗")
print("║   ✅ Tous les tests de dimensions dynamiques terminés!     ║")
print("╚═══════════════════════════════════════════════════════════════╝")
