-- ================================================================
-- GPT-style Decoder Model Example
-- Text generation with autoregressive decoder
-- ================================================================

log("════════════════════════════════════════════════")
log("  GPT-style Decoder Model Training")
log("════════════════════════════════════════════════")

-- Configuration allocateur
log("\n🔧 Configuration allocateur...")
allocator.configure({
    max_ram_gb = 10.0,
    enable_compression = true
})
log("✓ Allocateur configuré (10 GB, compression LZ4)")

-- Hardware capabilities
log("\n🔧 Capacités Hardware:")
local hw = model.hardware_caps()
log("  AVX2: " .. (hw.avx2 and "✓" or "✗"))
log("  FMA:  " .. (hw.fma and "✓" or "✗"))
model.set_hardware(true)
log("✓ Accélération hardware activée")

-- Create tokenizer
log("\n📚 Tokenizer...")
tokenizer.create(50000)
log("✓ Tokenizer created (50k vocab)")

-- GPT-2 style configuration
log("\n🏗️  Configuration GPT-2...")
local gpt_config = {
    vocab_size = 50000,
    max_seq_len = 1024,
    d_model = 768,
    num_layers = 12,
    num_heads = 12,
    d_ff = 3072,
    dropout = 0.1,
    causal = true
}

-- Create decoder model
local success, err = model.create("gpt_decoder")
if not success then
    log("❌ Erreur: " .. (err or "inconnue"))
    return
end

-- Build with architectures API
success, err = architectures.transformer(gpt_config)
if not success then
    log("❌ Erreur architecture: " .. (err or "inconnue"))
    return
end
log("✓ Architecture Transformer construite")

-- Allocate and initialize
success, params = model.allocate_params()
if not success then
    log("❌ Erreur allocation")
    return
end

success = model.init_weights("he", 42)
log("✓ Model built: " .. params .. " parameters")
log("  Mémoire: " .. string.format("%.2f MB", params * 4 / 1024 / 1024))

-- Load text dataset
dataset.load("datasets.old/text")
dataset.prepare_sequences(512)
log("✓ Dataset prepared")

-- Training avec learning rate schedule
log("\n🔥 Training for text generation...")
local epochs = 20
local base_lr = 0.0003

for epoch = 1, epochs do
    -- Cosine decay
    local lr = base_lr * 0.5 * (1 + math.cos(math.pi * epoch / epochs))
    log("Epoch " .. epoch .. "/" .. epochs .. " (LR: " .. string.format("%.6f", lr) .. ")")
    model.train(1, lr)
end
log("✓ Training terminé")

-- Save
log("\n💾 Sauvegarde...")
os.execute("mkdir -p checkpoints")
model.save("checkpoints/gpt_decoder")
tokenizer.save("checkpoints/gpt_decoder/tokenizer.json")
log("✓ Model et tokenizer sauvegardés")

-- Generate text
log("\nGenerating text...")
local prompt = "Once upon a time"
local generated = model.infer(prompt)
log("Prompt: " .. prompt)
log("Generated: " .. generated)
