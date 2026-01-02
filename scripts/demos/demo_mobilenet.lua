-- ============================================================================
-- Mímir Framework - Demo MobileNet
-- Efficient CNN pour mobile/edge devices
-- ============================================================================

log("=" .. string.rep("=", 78))
log("📱 Demo MobileNet - Efficient Mobile Architecture")
log("=" .. string.rep("=", 78))

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
local success, err = model.create("mobilenet_model")
if not success then
    log("❌ Erreur: " .. (err or "inconnue"))
    return
end

-- Construire avec architectures API
success, err = architectures.mobilenet(config)
if not success then
    log("❌ Erreur architecture: " .. (err or "inconnue"))
    return
end

log("✓ Architecture MobileNetV2 construite")

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
local estimated_params = 3.4 * (config.width_mult ^ 2)  -- ~3.4M pour width_mult=1.0

-- Afficher config
log("\n📋 Configuration:")
log("  Classes: " .. config.num_classes)
log("  Width multiplier: " .. config.width_mult)
log("  Input size: " .. config.input_size .. "x" .. config.input_size)
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
