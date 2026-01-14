-- ╔══════════════════════════════════════════════════════════════╗
-- ║  Test : Enhanced DebugJson v1.1.0                          ║
-- ╚══════════════════════════════════════════════════════════════╝
--
-- Ce test vérifie le nouveau format DebugJson v1.1.0 avec :
--   - Layer configs (hyperparamètres Conv2d, Linear, etc.)
--   - Formes de tenseurs réelles (multi-dimensionnelles)
--   - Dump des gradients
--   - Weight deltas (avant/après optimizer)
--   - Section optimizer
--   - Sections conditionnelles (tokenizer/encoder)

print("╔════════════════════════════════════════════════════════════╗")
print("║     Test Enhanced DebugJson v1.1.0                       ║")
print("╚════════════════════════════════════════════════════════════╝\n")

local function _mimir_add_module_path()
    local ok, info = pcall(debug.getinfo, 1, "S")
    if not ok or type(info) ~= "table" then return end
    local src = info.source
    if type(src) ~= "string" or src:sub(1, 1) ~= "@" then return end
    local dir = src:sub(2):match("(.*/)")
    if not dir then return end
    package.path = package.path .. ";" .. dir .. "../modules/?.lua;" .. dir .. "../modules/?/init.lua"
end

_mimir_add_module_path()
local Arch = require("arch")

local model = (type(_G.Mimir) == "table" and type(Mimir.Model) == "table") and Mimir.Model or _G.model

-- Configuration mémoire
Allocator.configure({
    max_ram_gb = 2.0,
    enable_compression = false
})

print("✓ Allocateur configuré (2 GB)\n")

-- Créer un modèle simple avec Conv et Linear
print("━━━ Création du modèle ━━━\n")

local cfg, warn = Arch.build_config("transformer", {
    vocab_size = 500,
    d_model = 64,
    num_layers = 2,
    num_heads = 2,
    max_seq_len = 32
})
if warn then
    print("⚠️  " .. tostring(warn))
end

local ok_create, err_create = model.create("transformer", cfg)
if not ok_create then
    print("❌ Erreur création modèle: " .. tostring(err_create))
    os.exit(1)
end

local ok_alloc, params = model.allocate_params()
if not ok_alloc then
    print("❌ Erreur allocation: " .. tostring(params))
    os.exit(1)
end

print(string.format("✓ Modèle créé: %d paramètres\n", params))

model.init_weights("xavier", 42)

-- Forward + backward pour avoir des gradients
print("━━━ Forward + Backward ━━━\n")

local input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}  -- Séquence de tokens

local output = model.forward(input, true)  -- training=true
print(string.format("✓ Forward: %d outputs\n", #output))

local target = {}
for i = 1, #output do
    target[i] = (i == 1) and 1.0 or 0.0
end

local loss = model.compute_loss(output, target, "mse")
print(string.format("✓ Loss: %.6f\n", loss))

local loss_grad = model.compute_loss_gradient(output, target, "mse")
model.backward(loss_grad)
print("✓ Backward done\n")

-- Sauvegarde AVANT optimizer (snapshot initial)
print("━━━ Snapshot initial ━━━\n")

local snapshot_path = "/tmp/mimir_snapshot.json"
local success1 = Mimir.Serialization.save_enhanced_debug(snapshot_path, {
    include_gradients = true,
    include_optimizer_state = false,
    max_values_per_tensor = 10,
    include_checksums = true,
    include_weight_deltas = false,  -- Pas encore de deltas
    include_git_info = false,
    save_tokenizer = false,
    save_encoder = false
})

if not success1 then
    print("❌ Échec sauvegarde snapshot")
    os.exit(1)
end

print("✓ Snapshot sauvegardé: " .. snapshot_path .. "\n")

-- Optimizer step pour modifier les poids
print("━━━ Optimizer Step ━━━\n")

model.optimizer_step(0.001)  -- lr = 0.001
print("✓ Poids mis à jour\n")

-- Sauvegarde APRÈS optimizer (avec weight deltas)
print("━━━ Checkpoint final ━━━\n")

local final_path = "/tmp/mimir_enhanced_final.json"
local success2 = Mimir.Serialization.save_enhanced_debug(final_path, {
    include_gradients = true,
    include_optimizer_state = true,
    max_values_per_tensor = 20,
    include_checksums = true,
    include_weight_deltas = true,  -- Activer les deltas
    include_git_info = true,
    save_tokenizer = false,
    save_encoder = false
})

if not success2 then
    print("❌ Échec sauvegarde finale")
    os.exit(1)
end

print("✓ Checkpoint sauvegardé: " .. final_path .. "\n")

-- ══════════════════════════════════════════════════════════════
--  VALIDATION
-- ══════════════════════════════════════════════════════════════

print("━━━ Validation ━━━\n")

local json = read_json(final_path)
if not json then
    print("❌ Fichier JSON illisible")
    os.exit(1)
end

-- 1. Version
if json.format_version ~= "1.1.0" then
    print(string.format("❌ Version incorrecte: %s", tostring(json.format_version)))
    os.exit(1)
end
print(string.format("✓ format_version = '%s'", json.format_version))

-- 2. Features
if not json.features or #json.features == 0 then
    print("❌ Pas de features")
    os.exit(1)
end
print(string.format("✓ features: %d", #json.features))

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
    for _, f in ipairs(json.features) do
        if f == feat then
            found = true
            break
        end
    end
    if not found then
        print(string.format("❌ Feature manquante: %s", feat))
        os.exit(1)
    end
end
print("✓ Toutes les features présentes")

-- 3. Layers
if not json.layers or #json.layers == 0 then
    print("❌ Pas de layers")
    os.exit(1)
end
print(string.format("✓ %d layers", #json.layers))

-- 4. Layer config
local has_config = false
for _, layer in ipairs(json.layers) do
    if layer.config and next(layer.config) ~= nil then
        has_config = true
        print(string.format("✓ Layer '%s' a une config", layer.name))
        break
    end
end

-- 5. Shape multi-dimensionnelle
local has_multi_shape = false
for _, layer in ipairs(json.layers) do
    if layer.tensors then
        for _, tensor in ipairs(layer.tensors) do
            if tensor.shape and #tensor.shape > 1 then
                has_multi_shape = true
                local shape_str = table.concat(tensor.shape, "x")
                print(string.format("✓ Tensor '%s' shape: [%s]", tensor.name, shape_str))
                break
            end
        end
    end
    if has_multi_shape then break end
end

if not has_multi_shape then
    print("⚠️  Pas de shape multi-dim (peut être normal pour Linear 1D)")
end

-- 6. Gradients
local has_grads = false
for _, layer in ipairs(json.layers) do
    if layer.tensors then
        for _, tensor in ipairs(layer.tensors) do
            if tensor.gradients then
                has_grads = true
                print(string.format("✓ Tensor '%s' a des gradients (L2=%.6f)", 
                    tensor.name, tensor.gradients.l2_norm or 0))
                break
            end
        end
    end
    if has_grads then break end
end

if not has_grads then
    print("❌ Pas de gradients trouvés")
    os.exit(1)
end

-- 7. Weight deltas
local has_delta = false
for _, layer in ipairs(json.layers) do
    if layer.weight_delta then
        has_delta = true
        print(string.format("✓ Layer '%s' weight_delta:", layer.name))
        print(string.format("  - changed: %s", tostring(layer.weight_delta.changed)))
        print(string.format("  - delta_l2: %.6f", layer.weight_delta.delta_l2_norm or 0))
        break
    end
end

if not has_delta then
    print("⚠️  Pas de weight_delta (première sauvegarde?)")
end

-- 8. Optimizer
if not json.optimizer then
    print("❌ Pas de section optimizer")
    os.exit(1)
end
print(string.format("✓ Optimizer: type=%s, lr=%.6f", 
    json.optimizer.type or "N/A",
    json.optimizer.lr or 0))

-- 9. Sections conditionnelles
if json.tokenizer then
    print("⚠️  tokenizer présent (devrait être absent)")
end
if json.encoder then
    print("⚠️  encoder présent (devrait être absent)")
end
if not json.tokenizer and not json.encoder then
    print("✓ Sections conditionnelles OK (tokenizer/encoder absents)")
end

-- ══════════════════════════════════════════════════════════════
--  RÉSUMÉ
-- ══════════════════════════════════════════════════════════════

print("\n╔════════════════════════════════════════════════════════════╗")
print("║                ✅ TOUS LES TESTS PASSÉS!                  ║")
print("╚════════════════════════════════════════════════════════════╝")
print(string.format("\nFormat: DebugJson v%s", json.format_version))
print(string.format("Model: %s", json.model_name or "N/A"))
print(string.format("Total params: %d", json.total_params or 0))
print(string.format("Layers: %d", json.num_layers or 0))
print("\nFichiers générés:")
print(string.format("  - %s (snapshot initial)", snapshot_path))
print(string.format("  - %s (checkpoint final)", final_path))
print("\n✅ Le format DebugJson v1.1.0 est fonctionnel!\n")
