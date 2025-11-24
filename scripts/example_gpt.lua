-- ================================================================
-- GPT-style Decoder Model Example
-- Text generation with autoregressive decoder
-- ================================================================

log("════════════════════════════════════════════════")
log("  GPT-style Decoder Model Training")
log("════════════════════════════════════════════════")

-- Create tokenizer
tokenizer.create(50000)
log("✓ Tokenizer created (50k vocab)")

-- GPT-2 style configuration
local gpt_config = {
    vocab_size = 50000,
    embed_dim = 768,
    num_layers = 12,
    num_heads = 12,
    d_ff = 3072,
    max_seq_len = 1024,
    dropout = 0.1,
    use_causal_mask = true,
    temperature = 0.8,
    top_k = 50,
    top_p = 0.9
}

-- Create decoder model
model.create("decoder", gpt_config)
log("✓ DecoderModel created (GPT-style)")

-- Build architecture
local ok, params = model.build()
log("✓ Model built: " .. params .. " parameters")

-- Load text dataset
dataset.load("datasets.old/text")
dataset.prepare_sequences(512)
log("✓ Dataset prepared")

-- Training
log("\nTraining for text generation...")
model.train(20, 0.0003)

-- Save
model.save("checkpoints/gpt_decoder")
log("✓ Model saved")

-- Generate text
log("\nGenerating text...")
local prompt = "Once upon a time"
local generated = model.infer(prompt)
log("Prompt: " .. prompt)
log("Generated: " .. generated)
