-- ╔══════════════════════════════════════════════════════════════╗
-- ║  Test Minimal : Enhanced DebugJson v1.1.0                  ║
-- ╚══════════════════════════════════════════════════════════════╝
--
-- Test minimal pour vérifier que la nouvelle API save_enhanced_debug
-- fonctionne et produit un fichier JSON v1.1.0 valide

print("╔════════════════════════════════════════════════════════════╗")
print("║     Test Minimal Enhanced DebugJson v1.1.0               ║")
print("╚════════════════════════════════════════════════════════════╝\n")

-- Configuration mémoire
Allocator.configure({
    max_ram_gb = 4.0,
    enable_compression = true
})

print("✓ Allocateur configuré (4 GB)\n")

-- Créer un modèle Transformer (comme dans test_serialization_formats.lua)
print("━━━ Création du modèle ━━━\n")

model.create("test_enhanced")

architectures.transformer({
    vocab_size = 500,
    d_model = 64,
    num_layers = 2,
    num_heads = 2,
    max_seq_len = 32
})

local ok_alloc, params = model.allocate_params()
if not ok_alloc then
    print("❌ Erreur allocation: " .. tostring(params))
    os.exit(1)
end

print(string.format("✓ Modèle créé: %d paramètres (%.2f MB)\n", 
    params, params * 4 / 1024 / 1024))

model.init_weights("xavier", 42)

-- Test de la nouvelle fonction save_enhanced_debug
print("━━━ Test save_enhanced_debug ━━━\n")

local test_path = "/tmp/test_enhanced_v1.1.json"

-- Vérifier que la fonction existe
if not Mimir.Serialization.save_enhanced_debug then
    print("❌ Fonction save_enhanced_debug non trouvée dans Mimir.Serialization")
    os.exit(1)
end

print("✓ Fonction save_enhanced_debug trouvée")

-- Appeler la fonction avec des options
local success = Mimir.Serialization.save_enhanced_debug(test_path, {
    include_gradients = false,  -- Pas de gradients pour test minimal
    include_optimizer_state = true,
    max_values_per_tensor = 10,
    include_checksums = true,
    include_weight_deltas = false,
    include_git_info = false,
    save_tokenizer = false,
    save_encoder = false
})

if not success then
    print("❌ Échec sauvegarde enhanced debug JSON")
    os.exit(1)
end

print("✓ Sauvegarde réussie: " .. test_path .. "\n")

-- Vérifier que le fichier existe
local file = io.open(test_path, "r")
if not file then
    print("❌ Fichier non créé")
    os.exit(1)
end

local content = file:read("*all")
file:close()

print(string.format("✓ Fichier créé (%d bytes)\n", #content))

-- Lire et valider le JSON
print("━━━ Validation JSON ━━━\n")

local json = read_json(test_path)
if not json then
    print("❌ JSON invalide")
    os.exit(1)
end

print("✓ JSON valide")

-- Vérifier format_version
if json.format_version ~= "1.1.0" then
    print(string.format("❌ Mauvaise version: '%s' (attendu '1.1.0')", 
        tostring(json.format_version)))
    os.exit(1)
end

print(string.format("✓ format_version = '%s'", json.format_version))

-- Vérifier features
if not json.features or #json.features == 0 then
    print("❌ Pas de features")
    os.exit(1)
end

print(string.format("✓ features présentes: %d", #json.features))

-- Vérifier que certaines features attendues sont là
local expected = {"layer_config", "real_shapes", "optimizer_state", "checksums"}
for _, feat in ipairs(expected) do
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

print("✓ Toutes les features attendues présentes")

-- Vérifier layers
if not json.layers or #json.layers == 0 then
    print("❌ Pas de layers")
    os.exit(1)
end

print(string.format("✓ %d layers détectés", #json.layers))

-- Vérifier qu'au moins un layer a une config
local has_config = false
for _, layer in ipairs(json.layers) do
    if layer.config and next(layer.config) ~= nil then
        has_config = true
        print(string.format("✓ Layer '%s' a une config", layer.name))
        break
    end
end

-- Vérifier optimizer section
if not json.optimizer then
    print("❌ Pas de section optimizer")
    os.exit(1)
end

print(string.format("✓ Optimizer section présente (type: %s)", 
    json.optimizer.type or "N/A"))

-- Vérifier model_name
if not json.model_name then
    print("❌ Pas de model_name")
    os.exit(1)
end

print(string.format("✓ model_name = '%s'", json.model_name))

-- Vérifier total_params
if not json.total_params then
    print("❌ Pas de total_params")
    os.exit(1)
end

print(string.format("✓ total_params = %d", json.total_params))

-- Vérifier num_layers
if not json.num_layers then
    print("❌ Pas de num_layers")
    os.exit(1)
end

print(string.format("✓ num_layers = %d", json.num_layers))

-- Résumé
print("\n╔════════════════════════════════════════════════════════════╗")
print("║                ✅ TOUS LES TESTS PASSÉS!                  ║")
print("╚════════════════════════════════════════════════════════════╝")

print("\n📋 Résumé:")
print(string.format("  • Format: DebugJson v%s", json.format_version))
print(string.format("  • Model: %s", json.model_name))
print(string.format("  • Parameters: %d", json.total_params))
print(string.format("  • Layers: %d", json.num_layers))
print(string.format("  • Features: %d", #json.features))
print(string.format("  • Fichier: %s", test_path))

print("\n✅ Le format Enhanced DebugJson v1.1.0 est fonctionnel!\n")
