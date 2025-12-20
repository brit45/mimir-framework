-- Test verbose d'allocation

log("Test d'allocation des paramètres avec verbosité")

tokenizer.create(1000)
log("✓ Tokenizer")

local config = {
    vocab_size = 1000,
    embed_dim = 64,
    num_layers = 1,
    num_heads = 2,
    d_ff = 256,
    max_seq_len = 32,
    dropout = 0.1
}

log("\nCréation...")
model.create("transformer", config)

log("\nBuild...")
local ok, params = model.build()

log("\n✓ Paramètres: " .. params)

-- Tester l'accès explicite à allocate_params
log("\nTest allocate_params explicite...")
local ok2, params2 = model.allocate_params()
log("  Result: " .. tostring(ok2))
log("  Params: " .. tostring(params2))

-- Tester init_weights
log("\nTest init_weights...")
local ok3 = model.init_weights("he", 42)
log("  Result: " .. tostring(ok3))

log("\n✅ Done")
