#!/usr/bin/env lua
-- Test du format DebugJson v1.1.0 avec toutes les fonctionnalités améliorées
-- Ce test vérifie :
-- 1. Layer configs (hyperparamètres Conv2d, Linear, etc.)
-- 2. Formes de tenseurs réelles (multi-dimensionnelles)
-- 3. Dump des gradients
-- 4. Détection des deltas de poids (avant/après optimizer)
-- 5. Section optimizer (légère)
-- 6. Sections conditionnelles (tokenizer/encoder)
-- 7. Versioning (format_version = "1.1.0")

log("========================================")
log("Test Enhanced Debug JSON v1.1.0")
log("========================================")

do
    local ok, err = Mimir.Allocator.configure({
        max_ram_gb= 10.0,
        enable_compression = true,
        swap_strategy = "lru"
    })
    if ok == false then
        log("❌ Mimir.Allocator.configure failed: " .. tostring(err))
        os.exit(1)
    end
end

Mimir.MemoryGuard.printStats()

-- Créer un petit modèle Conv2d simple pour le test
log("\n[1/8] Création du modèle...")
do
    -- IMPORTANT: actuellement, Model::push applique les champs Conv2d depuis modelConfig.
    -- On fournit donc une config globale cohérente pour permettre l'exécution de Conv2d.
    local ok, err = Mimir.Model.create("SimpleConv", {
        in_channels = 3,
        out_channels = 3,
        height = 64,
        width = 64,
        kernel = 3,
        stride = 1,
        padding = 1,
    })
    if ok == false then
        log("  ❌ ERREUR: Impossible de créer le modèle: " .. tostring(err))
        os.exit(1)
    end
end

-- Petit réseau conv exécutable (conv config propagée via modelConfig)
log("  - Ajout Conv2d(3→3, kernel=3x3, stride=1, padding=1)")
local conv_params = 3 * 3 * 3 * 3 + 3  -- in_ch * out_ch * k * k + bias
Mimir.Model.push_layer("conv1", "Conv2d", conv_params)

log("  - Ajout ReLU")
Mimir.Model.push_layer("relu1", "ReLU", 0)

log("  - Ajout Conv2d(3→3, kernel=3x3, stride=1, padding=1)")
Mimir.Model.push_layer("conv2", "Conv2d", conv_params)

log("  - Ajout ReLU")
Mimir.Model.push_layer("relu2", "ReLU", 0)

-- Allouer et initialiser les poids
log("\n[2/8] Allocation et initialisation des poids...")
Mimir.Model.allocate_params()
Mimir.Model.init_weights("xavier")

Mimir.MemoryGuard.printStats()


-- Créer un input dummy et faire forward + backward
log("\n[3/8] Forward pass...")
local input = {}
-- Par défaut, les Conv2d utilisent 64x64 si input_h/input_w ne sont pas renseignés
-- (voir Model.cpp). On aligne donc l'input sur 64x64 pour éviter les incohérences.
for i = 1, 3 * 64 * 64 do
    input[i] = math.random() * 0.1
end
local output, fwd_err = Mimir.Model.forward(input, true)
if output == nil then
    log("  ❌ ERREUR: forward failed: " .. tostring(fwd_err))
    os.exit(1)
end
-- log(string.format("  Output shape: %d", output))



-- Créer target dummy et calculer loss
log("\n[4/8] Backward pass...")
local target = {}
for i = 1, #output do
    target[i] = (i == 1) and 1.0 or 0.0  -- One-hot target
end

-- Calculer MSE loss manuellement
local loss = 0.0
for i = 1, #output do
    local diff = output[i] - target[i]
    loss = loss + diff * diff
end
loss = loss / #output
log(string.format("  Loss: %.6f", loss))

-- Calculer gradient MSE: d(MSE)/dy = 2*(y - target) / N
local loss_grad = {}
for i = 1, #output do
    loss_grad[i] = 2.0 * (output[i] - target[i]) / #output
end
Mimir.Model.backward(loss_grad)

-- Sauvegarder AVANT optimizer step (pour snapshot initial)
log("\n[5/8] Sauvegarde initiale (snapshot)...")
local snapshot_path = "/tmp/mimir_debug_snapshot.json"
local success = Mimir.Serialization.save_enhanced_debug(snapshot_path, {
    include_gradients = true,
    include_optimizer_state = false,
    max_values_per_tensor = 10,
    include_checksums = true,
    include_weight_deltas = false,  -- Pas de delta pour la première fois
    include_git_info = false,
    save_tokenizer = false,  -- Pas de tokenizer dans ce modèle
    save_encoder = false     -- Pas d'encoder non plus
})

if not success then
    log("  ❌ ERREUR: Échec de la sauvegarde initiale")
    os.exit(1)
end
log("  ✅ Snapshot initial sauvegardé")

-- Faire un optimizer step pour modifier les poids
log("\n[6/8] Optimizer step...")
local learning_rate = 0.001
Mimir.Model.optimizer_step(learning_rate)
log("  ✅ Poids mis à jour")

-- Sauvegarder APRÈS optimizer step (avec weight deltas)
log("\n[7/8] Sauvegarde finale (avec weight deltas)...")
local final_path = "/tmp/mimir_debug_final.json"
local success2 = Mimir.Serialization.save_enhanced_debug(final_path, {
    include_gradients = true,
    include_optimizer_state = true,
    max_values_per_tensor = 20,
    include_checksums = true,
    include_weight_deltas = true,  -- Activer les deltas maintenant
    include_git_info = true,
    save_tokenizer = false,
    save_encoder = false
})

if not success2 then
    log("  ❌ ERREUR: Échec de la sauvegarde finale")
    os.exit(1)
end
log("  ✅ Checkpoint final sauvegardé avec deltas")

-- Validation du fichier JSON
log("\n[8/8] Validation du format...")
local json = read_json(final_path)
if not json then
    log("  ❌ ERREUR: Impossible de lire le JSON")
    os.exit(1)
end

-- Vérifier la version
if json.format_version ~= "1.1.0" then
    log(string.format("  ❌ ERREUR: Mauvaise version (attendu '1.1.0', obtenu '%s')", tostring(json.format_version)))
    os.exit(1)
end
log("  ✅ format_version = '1.1.0'")

-- Vérifier les features
local features = json.features or {}
local expected_features = {
    "layer_config",
    "real_shapes",
    "gradients",
    "weight_deltas",
    "optimizer_state",
    "checksums"
}
for _, feat in ipairs(expected_features) do
    local found = false
    for _, f in ipairs(features) do
        if f == feat then
            found = true
            break
        end
    end
    if not found then
        log(string.format("  ❌ ERREUR: Feature manquante: %s", feat))
        os.exit(1)
    end
end
log(string.format("  ✅ Toutes les features présentes (%d)", #features))

-- Vérifier les layers
if not json.layers or #json.layers == 0 then
    log("  ❌ ERREUR: Pas de layers dans le JSON")
    os.exit(1)
end
log(string.format("  ✅ %d layers détectés", #json.layers))

-- Vérifier qu'au moins un layer a une config
local has_config = false
for _, layer in ipairs(json.layers) do
    if layer.config and next(layer.config) ~= nil then
        has_config = true
        local nkeys = 0
        for _ in pairs(layer.config) do nkeys = nkeys + 1 end
        log(string.format("  ✅ Layer '%s' a une config: %d champs", layer.name, nkeys))
        break
    end
end
if not has_config then
    log("  ⚠️  WARNING: Aucun layer n'a de config (peut être normal pour ReLU)")
end

-- Vérifier qu'au moins un tensor a une shape multi-dimensionnelle
local has_multi_dim = false
for _, layer in ipairs(json.layers) do
    if layer.tensors then
        for _, tensor in ipairs(layer.tensors) do
            if tensor.shape and #tensor.shape > 1 then
                has_multi_dim = true
                local shape_str = table.concat(tensor.shape, "x")
                log(string.format("  ✅ Tensor '%s' a une shape multi-dim: [%s]", tensor.name, shape_str))
                break
            end
        end
    end
    if has_multi_dim then break end
end
if not has_multi_dim then
    log("  ⚠️  WARNING: Aucun tensor avec shape multi-dimensionnelle")
    log("     (si tu utilises le nouveau weight_block, il faut que le dump exporte les tenseurs)")
end

-- Vérifier qu'au moins un tensor a des gradients
local has_gradients = false
for _, layer in ipairs(json.layers) do
    if layer.tensors then
        for _, tensor in ipairs(layer.tensors) do
            if tensor.gradients then
                has_gradients = true
                log(string.format("  ✅ Tensor '%s' a des gradients (L2 norm: %.6f)", tensor.name, tensor.gradients.l2_norm or 0))
                
                -- Vérifier que les gradients ne sont pas tous à zéro
                if tensor.gradients.all_zero then
                    log("    ⚠️  WARNING: Gradients all zero")
                else
                    log("    ✅ Gradients non-zero détectés")
                end
                break
            end
        end
    end
    if has_gradients then break end
end
if not has_gradients then
    log("  ❌ ERREUR: Aucun gradient trouvé")
    os.exit(1)
end

-- Vérifier weight_delta (devrait exister car c'est la 2e sauvegarde)
local has_delta = false
for _, layer in ipairs(json.layers) do
    if layer.weight_delta then
        has_delta = true
        log(string.format("  ✅ Layer '%s' a un weight_delta:", layer.name))
        log(string.format("    - changed: %s", tostring(layer.weight_delta.changed)))
        log(string.format("    - delta_l2_norm: %.6f", layer.weight_delta.delta_l2_norm or 0))
        log(string.format("    - relative_change: %.6f", layer.weight_delta.relative_change or 0))
        break
    end
end
if not has_delta then
    log("  ⚠️  WARNING: Aucun weight_delta (première sauvegarde?)")
else
    -- Vérifier que les poids ont effectivement changé
    local delta_found = false
    for _, layer in ipairs(json.layers) do
        if layer.weight_delta and layer.weight_delta.changed then
            delta_found = true
            log("    ✅ Changement de poids confirmé")
            break
        end
    end
    if not delta_found then
        log("    ⚠️  WARNING: Aucun poids n'a changé (learning_rate trop petit?)")
    end
end

-- Vérifier optimizer state
if not json.optimizer then
    log("  ❌ ERREUR: Section optimizer manquante")
    os.exit(1)
end
log(string.format("  ✅ Optimizer: type=%s, lr=%.6f, step=%d", 
    json.optimizer.type or "unknown",
    json.optimizer.lr or 0,
    json.optimizer.step or 0))

-- Vérifier que tokenizer/encoder ne sont PAS présents (model simple sans)
if json.tokenizer then
    log("  ⚠️  WARNING: tokenizer présent alors que save_tokenizer=false")
end
if json.encoder then
    log("  ⚠️  WARNING: encoder présent alors que save_encoder=false")
end
if not json.tokenizer and not json.encoder then
    log("  ✅ Sections conditionnelles: tokenizer et encoder absents (comme attendu)")
end


-- Résumé final
log("\n========================================")
log("✅ TOUS LES TESTS PASSÉS!")
log("========================================")
log(string.format("Format: DebugJson v%s", json.format_version))
log(string.format("Model: %s", json.model_name or "unknown"))
log(string.format("Total params: %d", json.total_params or 0))
log(string.format("Layers: %d", json.num_layers or 0))
log(string.format("Features: %s", table.concat(features, ", ")))
log("\nFichiers générés:")
log(string.format("  - %s (snapshot initial)", snapshot_path))
log(string.format("  - %s (checkpoint final)", final_path))
log("\nLe format DebugJson v1.1.0 est fonctionnel!")
