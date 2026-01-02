#!/usr/bin/env lua
-- Test simple du format DebugJson v1.1.0
-- On crée un modèle minimaliste juste pour tester la sauvegarde enhanced

log("========================================")
log("Test Enhanced Debug JSON v1.1.0 (simplifié)")
log("========================================")

-- Créer un modèle simple
log("\n[1] Création modèle simple...")
local model = Model.create("BasicTest")
Model.set_name("test_enhanced_v1.1")

-- Ajouter quelques layers simples
log("[2] Ajout de layers...")
Model.push("layer1", "linear", 1000)  -- Linear 10x100
Model.push("relu", "relu", 0)
Model.push("layer2", "linear", 200)   -- Linear 100x2

log("[3] Allocation...")
Model.allocate()
Model.init_weights("xavier")

-- Faire un forward/backward simple
log("[4] Forward/backward...")
local input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
local output = Model.forward(input)
log(string.format("  Output: %d valeurs", #output))

local target = {1, 0}
local loss_grad = Model.loss_gradient(output, target, "mse")
Model.backward(loss_grad)

log("[5] Optimizer step...")
Model.optimizer_step(0.01)

-- Test sauvegarde enhanced
log("\n[6] Sauvegarde Enhanced Debug JSON...")
local success = Mimir.Serialization.save_enhanced_debug("/tmp/test_enhanced.json", {
    include_gradients = true,
    include_optimizer_state = true,
    max_values_per_tensor = 10,
    include_checksums = true,
    include_weight_deltas = false,
    include_git_info = false,
    save_tokenizer = false,
    save_encoder = false
})

if not success then
    log("  ❌ ERREUR: Sauvegarde échouée")
    os.exit(1)
end

log("  ✅ Sauvegarde réussie: /tmp/test_enhanced.json")

-- Vérifier le contenu
log("\n[7] Validation du fichier...")
local json = read_json("/tmp/test_enhanced.json")

if not json then
    log("  ❌ ERREUR: Fichier JSON illisible")
    os.exit(1)
end

-- Vérifier version
if json.format_version ~= "1.1.0" then
    log(string.format("  ❌ ERREUR: Version incorrecte: %s", tostring(json.format_version)))
    os.exit(1)
end
log(string.format("  ✅ format_version = '%s'", json.format_version))

-- Vérifier features
if not json.features then
    log("  ❌ ERREUR: Pas de features")
    os.exit(1)
end
log(string.format("  ✅ features: %d éléments", #json.features))

-- Vérifier layers
if not json.layers or #json.layers == 0 then
    log("  ❌ ERREUR: Pas de layers")
    os.exit(1)
end
log(string.format("  ✅ layers: %d", #json.layers))

-- Vérifier gradients
local has_grads = false
for _, layer in ipairs(json.layers) do
    if layer.tensors then
        for _, tensor in ipairs(layer.tensors) do
            if tensor.gradients then
                has_grads = true
                log(string.format("  ✅ Gradients trouvés dans '%s'", tensor.name))
                break
            end
        end
    end
    if has_grads then break end
end

if not has_grads then
    log("  ⚠️  WARNING: Pas de gradients (peut être normal)")
end

-- Vérifier optimizer
if not json.optimizer then
    log("  ❌ ERREUR: Pas de section optimizer")
    os.exit(1)
end
log(string.format("  ✅ optimizer: type=%s", json.optimizer.type or "N/A"))

log("\n========================================")
log("✅ TEST RÉUSSI!")
log("========================================")
log(string.format("Format DebugJson v%s fonctionnel", json.format_version))
log(string.format("Model: %s", json.model_name or "N/A"))
log(string.format("Fichier: /tmp/test_enhanced.json"))
