-- ============================================================================
-- Mímir Framework - Demo MobileNet
-- Efficient CNN pour mobile/edge devices
-- ============================================================================

log("=" .. string.rep("=", 78))
log("📱 Demo MobileNet - Efficient Mobile Architecture")
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

-- Configuration MobileNetV2
local config = {
    num_classes = 1000,
    width_multiplier = 1.0,
    resolution = 224
}

-- Créer le modèle MobileNet
log("\n🏗️  Création du modèle MobileNet...")
local mobilenet_input = {
    num_classes = config.num_classes,
    image_size = config.resolution,
}

local cfg, warn = Arch.build_config("mobilenet", mobilenet_input)
if warn then
    log("⚠️  " .. tostring(warn))
end

local success, err = model.create("mobilenet", cfg)
if not success then
    log("❌ Erreur création modèle: " .. (err or "inconnue"))
    return
end

log("✓ MobileNet créé via registre")

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

-- Estimation params selon width_mult
local width_mult = config.width_multiplier or 1.0
local input_size = config.resolution or 224
local estimated_params = 3.4 * (width_mult ^ 2)  -- ~3.4M pour width_mult=1.0

-- Afficher config
log("\n📋 Configuration:")
log("  Classes: " .. config.num_classes)
log("  Width multiplier: " .. width_mult)
log("  Input size: " .. input_size .. "x" .. input_size)
log("  Params estimés: ~" .. string.format("%.1fM", estimated_params))

-- Concept clé: Depthwise Separable Convolutions
log("\n🔑 Concept clé - Depthwise Separable:")
log("  Standard Conv: C_in × C_out × K × K")
log("  Depthwise: C_in × K × K  (par channel)")
log("  Pointwise: C_in × C_out × 1 × 1")
log("  Réduction: ~8-9× moins de calculs")

-- Architecture MobileNetV2
log("\n🏛️  Architecture:")
log("  • Conv 3x3 standard")
log("  • 17 × Inverted Residual blocks")
log("  • Conv 1x1")
log("  • Global Average Pooling")
log("  • FC " .. config.num_classes)

-- Variantes width multiplier
log("\n⚖️  Variantes (width multiplier):")
log("  • 0.5:  ~1M params, latency ~40ms")
log("  • 0.75: ~2M params, latency ~60ms")
log("  • 1.0:  ~3.4M params, latency ~80ms")
log("  • 1.4:  ~6M params, latency ~120ms")

-- Usage pour mobile/edge
log("\n💡 Cas d'usage:")
log("  • Classification mobile (smartphones)")
log("  • Edge devices (Raspberry Pi, etc.)")
log("  • Real-time inference")
log("  • Embedded vision systems")
log("  • IoT applications")

-- Avantages
log("\n✨ Avantages:")
log("  ✓ Très léger (CPU-friendly)")
log("  ✓ Latence faible")
log("  ✓ Bonne accuracy/efficiency tradeoff")
log("  ✓ Adapté pour Mímir (CPU-only)")
log("  ✓ Scalable (width_mult)")

-- Comparaison vs ResNet
log("\n⚖️  MobileNetV2 vs ResNet-50:")
log("  Params: 3.4M vs 25M  (~7× plus léger)")
log("  FLOPs:  300M vs 4B   (~13× plus rapide)")
log("  Accuracy: ~72% vs ~76% (ImageNet Top-1)")

-- Recommandations CPU
log("\n🖥️  Recommandations Mímir CPU:")
log("  • width_mult=1.0 pour accuracy")
log("  • width_mult=0.75 pour vitesse")
log("  • width_mult=0.5 pour edge extrême")
log("  • Batch size=1 pour inférence temps réel")

-- Sauvegarder
log("\n💾 Sauvegarde du modèle...")
os.execute("mkdir -p checkpoints")
Mimir.Serialization.save("checkpoints/mobilenet_demo.safetensors", "safetensors")
log("✓ Modèle sauvegardé")

log("\n✓ Demo MobileNet terminé!")
log("=" .. string.rep("=", 78))
