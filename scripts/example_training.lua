-- ================================================================
-- Example Training Script for Mímir Framework
-- Demonstrates model creation, training, and saving
-- ================================================================

log("════════════════════════════════════════════════")
log("  Mímir Framework - Example Training Script")
log("════════════════════════════════════════════════")

-- ================================================================
-- 1. Configuration
-- ================================================================
log("\n[1/5] Configuration...")

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
log("\n[2/5] Creating tokenizer...")

tokenizer.create(config.vocab_size)
log("✓ Tokenizer created with vocab size: " .. config.vocab_size)

-- ================================================================
-- 3. Create Model
-- ================================================================
log("\n[3/5] Creating model...")

local success, error_msg = model.create("encoder", config)

if not success then
    log("❌ Error creating model: " .. (error_msg or "unknown"))
    return
end

log("✓ EncoderModel created")

-- Build architecture
local ok, num_params = model.build()

if ok then
    log("✓ Model built with " .. num_params .. " parameters")
else
    log("❌ Error building model")
    return
end

-- ================================================================
-- 4. Load Dataset
-- ================================================================
log("\n[4/5] Loading dataset...")

local dataset_ok, num_items = dataset.load("datasets.old/text")

if dataset_ok then
    log("✓ Dataset loaded: " .. num_items .. " items")
    
    -- Prepare sequences
    local seq_ok, num_sequences = dataset.prepare_sequences(256)
    
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
log("\n[5/5] Training...")

local num_epochs = 5
local learning_rate = 0.0001

log("Starting training: " .. num_epochs .. " epochs, LR=" .. learning_rate)

local train_ok = model.train(num_epochs, learning_rate)

if train_ok then
    log("✓ Training completed successfully")
else
    log("❌ Training failed")
    return
end

-- ================================================================
-- 6. Save Model
-- ================================================================
log("\nSaving model...")

local save_ok = model.save("checkpoints/example_encoder")

if save_ok then
    log("✓ Model saved to: checkpoints/example_encoder")
else
    log("❌ Failed to save model")
end

-- ================================================================
-- 7. Test Inference
-- ================================================================
log("\nTesting inference...")

local result = model.infer("This is a test input for the encoder model")
log("Inference result: " .. result)

-- ================================================================
-- Done
-- ================================================================
log("\n════════════════════════════════════════════════")
log("✓ Script completed successfully!")
log("════════════════════════════════════════════════")
