#!/usr/bin/env lua5.3
-- ============================================================================
-- TEST SIMPLE: Multi-Input API Smoke Test
-- ============================================================================
-- Vérifie que l'API model.set_layer_io() fonctionne sans erreur
-- Version: 2.3.0 - Multi-Input Support

log("🧪 Test Simple: Multi-Input API")
log("================================\n")

-- Test 1: Créer un modèle simple
log("1️⃣  Création du modèle...")
local ok = model.create("test_api")
if ok then
    log("   ✅ Model créé\n")
else
    log("   ❌ Échec création\n")
    os.exit(1)
end

-- Test 2: Ajouter des layers
log("2️⃣  Ajout de layers...")
model.push_layer("layer1", "Linear", 100)
model.push_layer("layer2", "ReLU", 0)
model.push_layer("layer3", "Linear", 50)
model.push_layer("layer4", "Add", 0)
log("   ✅ 4 layers ajoutés\n")

-- Test 3: Configurer les I/O
log("3️⃣  Configuration des I/O avec set_layer_io()...")
local result1 = model.set_layer_io("layer1", {"x"}, "hidden")
local result2 = model.set_layer_io("layer3", {"hidden"}, "out")
local result3 = model.set_layer_io("layer4", {"x", "out"}, "x")

if result1 and result2 and result3 then
    log("   ✅ set_layer_io() fonctionne")
    log("      • layer1: x → hidden")
    log("      • layer3: hidden → out")
    log("      • layer4: [x, out] → x")
else
    log("   ❌ set_layer_io() a échoué")
    if not result1 then log("      × layer1 failed") end
    if not result2 then log("      × layer3 failed") end
    if not result3 then log("      × layer4 failed") end
end

log("\n================================")
log("✅ API Test Complete")
log("================================")
