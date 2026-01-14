-- ============================================================================
-- Mímir Framework - Demo ResNet
-- Residual Network pour classification
-- ============================================================================

log("=" .. string.rep("=", 78))
log("🏗️  Demo ResNet - Residual Network")
log("=" .. string.rep("=", 78))

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

local Allocator = (type(_G.Mimir) == "table" and type(Mimir.Allocator) == "table") and Mimir.Allocator or _G.Allocator
local model = (type(_G.Mimir) == "table" and type(Mimir.Model) == "table") and Mimir.Model or _G.model

-- Configuration système
log("\n🔧 Configuration...")
Allocator.configure({max_ram_gb = 10.0, enable_compression = true})
model.set_hardware(true)
log("✓ Système configuré")

-- Configuration ResNet-50
local config = {
    num_classes = 1000,
    layers = {3, 4, 6, 3},
    base_channels = 64,
    use_bottleneck = true
}

-- Créer le modèle ResNet
log("\n🏗️  Création du modèle ResNet...")
local resnet_input = {
    num_classes = config.num_classes,
    base_channels = config.base_channels,
}
if type(config.layers) == "table" and #config.layers >= 4 then
    resnet_input.blocks1 = config.layers[1]
    resnet_input.blocks2 = config.layers[2]
    resnet_input.blocks3 = config.layers[3]
    resnet_input.blocks4 = config.layers[4]
end

local cfg, warn = Arch.build_config("resnet", resnet_input)
if warn then
    log("⚠️  " .. tostring(warn))
end

local success, err = model.create("resnet", cfg)
if not success then
    log("❌ Erreur création modèle: " .. (err or "inconnue"))
    return
end

log("✓ ResNet créé via registre")

-- Allouer et initialiser
success, params = model.allocate_params()
if success then
    model.init_weights("he", 42)
    log("✓ Modèle construit: " .. params .. " paramètres")
    log("  Mémoire: " .. string.format("%.2f MB", params * 4 / 1024 / 1024))
else
    log("❌ Erreur allocation")
    return
end

-- Calcul total blocks
local total_blocks = 0
for _, l in ipairs(config.layers) do
    total_blocks = total_blocks + l
end

-- Afficher config
log("\n📋 Configuration:")
log("  Classes: " .. config.num_classes)
log("  Layout:  " .. table.concat(config.layers, ", ") .. " (ResNet-50)")
log("  Total blocks: " .. total_blocks)
log("  Base channels: " .. config.base_channels)
log("  Bottleneck: " .. (config.use_bottleneck and "Yes" or "No"))

-- Estimation architecture
log("\n🏛️  Architecture:")
log("  Conv1: 7x7, " .. config.base_channels)
log("  Layer1: " .. config.layers[1] .. " blocks, " .. config.base_channels)
log("  Layer2: " .. config.layers[2] .. " blocks, " .. config.base_channels * 2)
log("  Layer3: " .. config.layers[3] .. " blocks, " .. config.base_channels * 4)
log("  Layer4: " .. config.layers[4] .. " blocks, " .. config.base_channels * 8)
log("  FC: " .. config.num_classes .. " classes")

-- Concept clé: Residual connections
log("\n🔑 Concept clé - Residual Connections:")
log("  y = F(x) + x")
log("  • Permet d'entraîner des réseaux très profonds")
log("  • Évite le vanishing gradient")
log("  • Facilite l'apprentissage de fonctions identité")

-- Variantes ResNet
log("\n🔀 Variantes ResNet:")
log("  • ResNet-18:  [2,2,2,2] → ~11M params")
log("  • ResNet-34:  [3,4,6,3] → ~21M params")
log("  • ResNet-50:  [3,4,6,3] + bottleneck → ~25M params")
log("  • ResNet-101: [3,4,23,3] → ~44M params")
log("  • ResNet-152: [3,8,36,3] → ~60M params")

-- Usage pour classification
log("\n💡 Cas d'usage:")
log("  • Classification d'images (ImageNet)")
log("  • Transfer learning (backbone)")
log("  • Object detection (Faster R-CNN)")
log("  • Segmentation (DeepLab)")
log("  • Feature extraction")

-- Avantages
log("\n✨ Avantages:")
log("  ✓ Architecture éprouvée (2015)")
log("  ✓ Très bonne performance")
log("  ✓ Entraînement stable")
log("  ✓ Transfer learning efficace")
log("  ✓ Nombreux modèles pré-entraînés")

-- Sauvegarder
log("\n💾 Sauvegarde du modèle...")
os.execute("mkdir -p checkpoints")
Mimir.Serialization.save("checkpoints/resnet_demo.safetensors", "safetensors")
log("✓ Modèle sauvegardé")

log("\n✓ Demo ResNet terminé!")
log("=" .. string.rep("=", 78))
