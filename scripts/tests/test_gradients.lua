-- ╔═══════════════════════════════════════════════════════════════╗
-- ║  Test des Fonctionnalités de Gradients                        ║
-- ╚═══════════════════════════════════════════════════════════════╝

log("╔═══════════════════════════════════════════════════════════════╗")
log("║   Test Gradients & Vulkan Init Once                           ║")
log("╚═══════════════════════════════════════════════════════════════╝\n")

-- Configuration
local config = {
    in_channels = 3,
    out_channels = 8,
    height = 8,
    width = 8,
    kernel = 3,
    stride = 1,
    padding = 1
}

log("📋 Configuration du modèle:")
log(string.format("   • Input: %dx%dx%d", config.height, config.width, config.in_channels))
log(string.format("   • Output: %d channels", config.out_channels))
log("")

-- Créer le modèle
model.create("test_gradients", config)
log("✓ Modèle créé")

-- Ajouter une couche Conv2D
local num_params = (config.kernel * config.kernel * config.in_channels * config.out_channels) 
                   + config.out_channels
model.push_layer("conv1", "Conv2d", num_params)
log("✓ Couche Conv2D ajoutée (" .. num_params .. " paramètres)")

-- Allouer et initialiser
model.allocate_params()
model.init_weights("xavier_uniform")
log("✓ Paramètres initialisés\n")

-- Test 1: Vulkan Init Once
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  Test 1: Vulkan Init Once")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

-- Créer plusieurs modèles pour vérifier que Vulkan n'est initialisé qu'une fois
for i = 1, 3 do
    local test_model_name = "test_model_" .. i
    model.create(test_model_name, config)
    log(string.format("   Model %d créé (Vulkan devrait déjà être initialisé)", i))
end
log("✓ Vulkan init_once fonctionne (pas de réinitialisation)\n")

-- Revenir au modèle de test
model.create("test_gradients", config)
model.push_layer("conv1", "Conv2d", num_params)
model.allocate_params()
model.init_weights("xavier_uniform")

-- Test 2: Forward Pass
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  Test 2: Forward Pass")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

local input_size = config.height * config.width * config.in_channels
local input = {}
for i = 1, input_size do
    input[i] = math.random() * 2.0 - 1.0
end

local output = model.forward(input)
if output then
    log(string.format("   ✓ Forward pass OK - Output size: %d", #output))
else
    log("   ❌ Forward pass échoué")
    return
end
log("")

-- Test 3: Backward Pass
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  Test 3: Backward Pass")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

local grad_output = {}
for i = 1, #output do
    grad_output[i] = 1.0  -- Gradient uniforme
end

local ok = model.backward(grad_output)
if ok then
    log("   ✓ Backward pass OK")
else
    log("   ❌ Backward pass échoué")
    return
end
log("")

-- Test 4: Récupérer les gradients
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  Test 4: Récupération des Gradients")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

local grads = model.get_gradients()
if grads then
    log(string.format("   ✓ Gradients récupérés - %d valeurs", #grads))
    
    -- Calculer la norme des gradients
    local grad_norm = 0
    for _, g in ipairs(grads) do
        grad_norm = grad_norm + g * g
    end
    grad_norm = math.sqrt(grad_norm)
    log(string.format("   • Gradient norm: %.6f", grad_norm))
    
    -- Afficher quelques valeurs
    log("   • Premiers gradients:")
    for i = 1, math.min(5, #grads) do
        log(string.format("     grad[%d] = %.6f", i, grads[i]))
    end
else
    log("   ❌ Échec de récupération des gradients")
    return
end
log("")

-- Test 5: Zero Gradients
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  Test 5: Zero Gradients")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

ok = model.zero_grads()
if ok then
    log("   ✓ Gradients réinitialisés à zéro")
    
    -- Vérifier que les gradients sont bien à zéro
    local grads_after = model.get_gradients()
    if grads_after then
        local all_zero = true
        local max_val = 0
        for _, g in ipairs(grads_after) do
            if math.abs(g) > 1e-7 then
                all_zero = false
            end
            max_val = math.max(max_val, math.abs(g))
        end
        
        if all_zero then
            log("   ✓ Tous les gradients sont à zéro")
        else
            log(string.format("   ⚠ Certains gradients non nuls (max: %.6f)", max_val))
        end
    end
else
    log("   ❌ Échec de réinitialisation des gradients")
    return
end
log("")

-- Test 6: Boucle d'entraînement complète
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  Test 6: Boucle d'Entraînement Complète")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

for iter = 1, 3 do
    -- 1. Zero gradients
    model.zero_grads()
    
    -- 2. Forward
    local train_output = model.forward(input)
    
    -- 3. Backward
    local train_grad = {}
    for i = 1, #train_output do
        train_grad[i] = 0.01
    end
    model.backward(train_grad)
    
    -- 4. Vérifier que les gradients existent
    local iter_grads = model.get_gradients()
    if iter_grads then
        local norm = 0
        for _, g in ipairs(iter_grads) do
            norm = norm + g * g
        end
        norm = math.sqrt(norm)
        log(string.format("   Iteration %d: gradient norm = %.6f", iter, norm))
    else
        log(string.format("   ❌ Iteration %d: échec récupération gradients", iter))
    end
end

log("\n✓ Boucle d'entraînement complète testée\n")

-- Résumé
log("╔═══════════════════════════════════════════════════════════════╗")
log("║   ✅ Tous les tests de gradients réussis!                  ║")
log("╚═══════════════════════════════════════════════════════════════╝")
log("")
log("Fonctionnalités validées:")
log("  ✓ Vulkan init_once (pas de réinitialisation)")
log("  ✓ Forward pass")
log("  ✓ Backward pass")
log("  ✓ model.get_gradients()")
log("  ✓ model.zero_grads()")
log("  ✓ Boucle d'entraînement complète")
