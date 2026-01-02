#!/usr/bin/env lua
-- ══════════════════════════════════════════════════════════════
--  Test Complet des Nouveaux Layers - Mímir Framework v2.1
--  Architecture Unifée: LayerTypes + LayerOps
-- ══════════════════════════════════════════════════════════════

log("\n╔════════════════════════════════════════════════════════╗")
log("║   Test Complet Système de Layers Unifié              ║")
log("╚════════════════════════════════════════════════════════╝\n")

Allocator.configure({
    max_ram_gb = 10.0,
    enable_compression = true
})

-- ══════════════════════════════════════════════════════════════
--  TEST 1: Modèle NLP (Embedding + LayerNorm + Linear + Softmax)
-- ══════════════════════════════════════════════════════════════

log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  TEST 1: NLP Pipeline")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

local success, err = model.create("nlp_model", {})
if not success then
    log("❌ Échec création modèle NLP: " .. (err or ""))
    os.exit(1)
end

-- Petite architecture NLP
-- Input: sequence de tokens -> Embedding -> LayerNorm -> Linear -> GELU -> Linear -> Softmax
local vocab_size = 1000
local embed_dim = 128
local hidden_dim = 256
local num_classes = 10

-- Embedding
model.push_layer("embedding", "Embedding", vocab_size * embed_dim)

-- Layer Norm
model.push_layer("ln1", "LayerNorm", embed_dim * 2)  -- gamma + beta

-- Linear 1 (Projection)
model.push_layer("linear1", "Linear", embed_dim * hidden_dim + hidden_dim)

-- GELU Activation
model.push_layer("gelu1", "GELU", 0)

-- Linear 2 (Classification)
model.push_layer("linear2", "Linear", hidden_dim * num_classes + num_classes)

-- Softmax
model.push_layer("softmax", "Softmax", 0)

log("✓ Architecture NLP construite (6 layers)")

-- Allocation et init
success, num_params = model.allocate_params()
if not success then
    log("❌ Échec allocation")
    os.exit(1)
end
log(string.format("✓ Paramètres alloués: %d", num_params))

success = model.init_weights("xavier", 42)
if not success then
    log("❌ Échec init poids")
    os.exit(1)
end
log("✓ Poids initialisés\n")

-- Forward pass avec input factice (embedding accepte des floats pour ce test)
local seq_length = 10
local input_nlp = {}
for i = 1, embed_dim * seq_length do
    input_nlp[i] = math.random() * 2 - 1
end

log(string.format("Forward pass NLP: %d valeurs", #input_nlp))
local output_nlp = model.forward(input_nlp)

if output_nlp and #output_nlp > 0 then
    log(string.format("✓ Output NLP: %d valeurs", #output_nlp))
else
    log("❌ Forward pass NLP échoué")
    os.exit(1)
end

log("")

-- ══════════════════════════════════════════════════════════════
--  TEST 2: Modèle CV (Conv + BN + MaxPool + AvgPool + Linear)
-- ══════════════════════════════════════════════════════════════

log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  TEST 2: CV Pipeline avec Pooling")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

success, err = model.create("cv_model", {
    in_channels = 3,
    base_channels = 16,
    height = 32,
    width = 32
})
if not success then
    log("❌ Échec création modèle CV: " .. (err or ""))
    os.exit(1)
end

-- Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d -> 
-- Conv2d -> BatchNorm2d -> ReLU -> GlobalAvgPool2d -> Linear

-- Conv1
model.push_layer("conv1", "Conv2d", 3 * 16 * 3 * 3)

-- BN1
model.push_layer("bn1", "BatchNorm2d", 16 * 2)

-- ReLU1
model.push_layer("relu1", "ReLU", 0)

-- MaxPool
model.push_layer("maxpool1", "MaxPool2d", 0)

-- Conv2
model.push_layer("conv2", "Conv2d", 16 * 32 * 3 * 3)

-- BN2
model.push_layer("bn2", "BatchNorm2d", 32 * 2)

-- ReLU2
model.push_layer("relu2", "ReLU", 0)

-- GlobalAvgPool
model.push_layer("gap", "GlobalAvgPool2d", 0)

-- Linear (classification)
model.push_layer("fc", "Linear", 32 * 10 + 10)

log("✓ Architecture CV construite (9 layers)")

success, num_params = model.allocate_params()
if not success then
    log("❌ Échec allocation CV")
    os.exit(1)
end
log(string.format("✓ Paramètres alloués: %d", num_params))

success = model.init_weights("he", 123)
if not success then
    log("❌ Échec init poids CV")
    os.exit(1)
end
log("✓ Poids initialisés\n")

-- Forward pass
local input_cv_size = 3 * 32 * 32
local input_cv = {}
for i = 1, input_cv_size do
    input_cv[i] = math.random()
end

log(string.format("Forward pass CV: %d valeurs (3×32×32)", input_cv_size))
local output_cv = model.forward(input_cv)

if output_cv and #output_cv > 0 then
    log(string.format("✓ Output CV: %d valeurs", #output_cv))
else
    log("❌ Forward pass CV échoué")
    os.exit(1)
end

log("")

-- ══════════════════════════════════════════════════════════════
--  TEST 3: Modèle avec GroupNorm et GELU/SiLU
-- ══════════════════════════════════════════════════════════════

log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  TEST 3: Normalization Avancée + Activations")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

success, err = model.create("norm_model", {
    in_channels = 32,
    height = 16,
    width = 16
})
if not success then
    log("❌ Échec création modèle norm: " .. (err or ""))
    os.exit(1)
end

-- Conv -> GroupNorm -> SiLU -> Conv -> GELU -> AvgPool

model.push_layer("conv_a", "Conv2d", 32 * 32 * 3 * 3)
model.push_layer("gn1", "GroupNorm", 32 * 2)
model.push_layer("silu", "SiLU", 0)
model.push_layer("conv_b", "Conv2d", 32 * 32 * 3 * 3)
model.push_layer("gelu", "GELU", 0)
model.push_layer("avgpool", "AvgPool2d", 0)

log("✓ Architecture Norm construite (6 layers)")

success, num_params = model.allocate_params()
if not success then
    log("❌ Échec allocation norm")
    os.exit(1)
end
log(string.format("✓ Paramètres alloués: %d", num_params))

success = model.init_weights("xavier", 456)
if not success then
    log("❌ Échec init poids norm")
    os.exit(1)
end
log("✓ Poids initialisés\n")

local input_norm_size = 32 * 16 * 16
local input_norm = {}
for i = 1, input_norm_size do
    input_norm[i] = math.random() * 2 - 1
end

log(string.format("Forward pass Norm: %d valeurs (32×16×16)", input_norm_size))
local output_norm = model.forward(input_norm)

if output_norm and #output_norm > 0 then
    log(string.format("✓ Output Norm: %d valeurs", #output_norm))
else
    log("❌ Forward pass Norm échoué")
    os.exit(1)
end

log("")

-- ══════════════════════════════════════════════════════════════
--  TEST 4: Flatten et Reshape
-- ══════════════════════════════════════════════════════════════

log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  TEST 4: Shape Operations (Flatten/Reshape)")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

success, err = model.create("shape_model", {})
if not success then
    log("❌ Échec création modèle shape: " .. (err or ""))
    os.exit(1)
end

-- Conv -> Flatten -> Linear -> Reshape

model.push_layer("conv_s", "Conv2d", 3 * 16 * 3 * 3)
model.push_layer("flatten", "Flatten", 0)
model.push_layer("linear_s", "Linear", 16 * 8 * 8 * 128 + 128)
model.push_layer("reshape", "Reshape", 0)

log("✓ Architecture Shape construite (4 layers)")

success, num_params = model.allocate_params()
if not success then
    log("❌ Échec allocation shape")
    os.exit(1)
end
log(string.format("✓ Paramètres alloués: %d", num_params))

success = model.init_weights("xavier", 789)
if not success then
    log("❌ Échec init poids shape")
    os.exit(1)
end
log("✓ Poids initialisés\n")

local input_shape = {}
for i = 1, 3 * 10 * 10 do
    input_shape[i] = math.random()
end

log(string.format("Forward pass Shape: %d valeurs", #input_shape))
local output_shape = model.forward(input_shape)

if output_shape and #output_shape > 0 then
    log(string.format("✓ Output Shape: %d valeurs", #output_shape))
else
    log("❌ Forward pass Shape échoué")
    os.exit(1)
end

log("")

-- ══════════════════════════════════════════════════════════════
--  RÉSUMÉ FINAL
-- ══════════════════════════════════════════════════════════════

log("╔════════════════════════════════════════════════════════╗")
log("║              RÉSUMÉ DES TESTS                          ║")
log("╚════════════════════════════════════════════════════════╝\n")

log("✅ Tous les tests réussis!")
log("")
log("📊 Layers testés:")
log("  ✓ Embedding (NLP)")
log("  ✓ LayerNorm")
log("  ✓ Linear (Dense)")
log("  ✓ GELU, SiLU (Activations avancées)")
log("  ✓ Softmax")
log("  ✓ Conv2d (2D Convolution)")
log("  ✓ BatchNorm2d")
log("  ✓ ReLU")
log("  ✓ MaxPool2d (2D)")
log("  ✓ AvgPool2d")
log("  ✓ GlobalAvgPool2d (Adaptive)")
log("  ✓ GroupNorm")
log("  ✓ Flatten")
log("  ✓ Reshape")
log("")

local mem_stats = MemoryGuard.getStats()
log("💾 Mémoire:")
log(string.format("  • Utilisée: %.2f MB", mem_stats.current_mb))
log(string.format("  • Pic: %.2f MB", mem_stats.peak_mb))
log("")

log("✨ Système de Layers Unifié: ✅ Validé! ✨\n")
