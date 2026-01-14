-- ================================================================
-- Example Training Script for Mímir Framework
-- Demonstrates model creation, training, and saving
-- ================================================================

log("════════════════════════════════════════════════")
log("  Mímir Framework - Example Training Script")
log("════════════════════════════════════════════════")

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

-- ================================================================
-- 0. Configuration allocateur et hardware (OBLIGATOIRE!)
-- ================================================================
log("\n[0/6] Configuration système...")

-- ⚠️ IMPORTANT: Toujours configurer l'allocateur en premier!
-- Cela active la limite de 10 GB et la compression LZ4
Mimir.Allocator.configure({
    max_ram_gb = 10.0,              -- Limite stricte
    enable_compression = true       -- Compression LZ4 (~50% économie)
})
log("✓ Allocateur configuré (limite: 10 GB, compression activée)")

local hw = Mimir.Model.hardware_caps()
log("✓ Hardware: AVX2=" .. (hw.avx2 and "oui" or "non") .. ", FMA=" .. (hw.fma and "oui" or "non"))
Mimir.Model.set_hardware(true)

-- ================================================================
-- 1. Configuration
-- ================================================================
log("\n[1/6] Configuration...")

local config = {
    num_layers = 6,
    d_model = 512,
    num_heads = 8,
    vocab_size = 10000,
    max_seq_len = 256,
    dropout = 0.1,
    use_prenorm = true
}

log("✓ Model config: " .. config.num_layers .. " layers, " .. config.d_model .. " dims")

-- ================================================================
-- 2. Create Tokenizer
-- ================================================================
log("\n[2/6] Creating tokenizer...")

Mimir.Tokenizer.create(config.vocab_size)
log("✓ Tokenizer created with vocab size: " .. config.vocab_size)

-- ================================================================
-- 3. Create Model
-- ================================================================
log("\n[3/6] Creating model...")

-- Build config (encoder-style: causal=false)
local model_config = {
    vocab_size = config.vocab_size,
    max_seq_len = config.max_seq_len,
    d_model = config.d_model,
    num_heads = config.num_heads,
    num_layers = config.num_layers,
    d_ff = config.d_model * 4,
    dropout = config.dropout,
    causal = false
}

local cfg, warn = Arch.build_config("transformer", model_config)
if warn then
    log("⚠️  " .. tostring(warn))
end

local success, error_msg = Mimir.Model.create("transformer", cfg)
if not success then
    log("❌ Error creating model: " .. (error_msg or "unknown"))
    return
end

log("✓ Model created via registre")

-- Allocate and initialize
success, num_params = Mimir.Model.allocate_params()
if not success then
    log("❌ Error allocating parameters")
    return
end

success = Mimir.Model.init_weights("xavier", 42)
if success then
    log("✓ Model built with " .. num_params .. " parameters")
    log("  Mémoire: " .. string.format("%.2f MB", num_params * 4 / 1024 / 1024))
else
    log("❌ Error initializing weights")
    return
end

-- ================================================================
-- 4. Load Dataset
-- ================================================================
log("\n[4/6] Loading dataset...")

local dataset_ok, num_items = Mimir.Dataset.load("datasets.old/text")

if dataset_ok then
    log("✓ Dataset loaded: " .. num_items .. " items")
    
    -- Prepare sequences
    local seq_ok, num_sequences = Mimir.Dataset.prepare_sequences(256)
    
    if seq_ok then
        log("✓ Prepared " .. num_sequences .. " sequences of length 256")
    else
        log("⚠️  Warning: Could not prepare sequences")
    end
else
    log("⚠️  Warning: Dataset not found, continuing anyway...")
end

-- ================================================================
-- 5. Training
-- ================================================================
log("\n[5/6] Training...")

local num_epochs = 5
local learning_rate = 0.0001

log("Starting training: " .. num_epochs .. " epochs, LR=" .. learning_rate)

-- Training loop avec learning rate schedule
for epoch = 1, num_epochs do
    local lr = learning_rate * math.exp(-0.1 * (epoch - 1))
    log("Epoch " .. epoch .. "/" .. num_epochs .. " (LR: " .. string.format("%.6f", lr) .. ")")
    Mimir.Model.train(1, lr)
end

log("✓ Training completed successfully")

-- ================================================================
-- 6. Save Model
-- ================================================================
log("\n[6/6] Saving model...")

os.execute("mkdir -p checkpoints")
local save_ok = Mimir.Serialization.save("checkpoints/example_encoder.safetensors", "safetensors")

if save_ok then
    log("✓ Model saved to: checkpoints/example_encoder")
    Mimir.Tokenizer.save("checkpoints/example_encoder/tokenizer.json")
    log("✓ Tokenizer saved")
else
    log("❌ Failed to save model")
end

-- ================================================================
-- 7. Test Inference
-- ================================================================
log("\n[7/7] Testing inference...")

local result = Mimir.Model.infer("This is a test input for the encoder model")
if result and result ~= "" then
    log("✓ Inference result: " .. result)
else
    log("⚠️  Inference non disponible")
end

-- ================================================================
-- Done
-- ================================================================
log("\n════════════════════════════════════════════════")
log("✅ Script completed successfully!")
log("════════════════════════════════════════════════")
