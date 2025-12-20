#!/usr/bin/env mimir --lua

log("=== Test du Tokenizer ===\n")

-- 1. Créer tokenizer
tokenizer.create(100000)
log("✓ Tokenizer créé\n")

-- 2. Ajouter quelques mots
local test_words = {"hello", "world", "machine", "learning", "neural", "network"}

for _, word in ipairs(test_words) do
    local id = tokenizer.add_token(word)
    log(string.format("  Ajouté '%s' → ID %d", word, id))
end

log(string.format("\n✓ Vocab size: %d\n", tokenizer.vocab_size()))

-- 3. Test tokenization
local test_texts = {
    "hello world",
    "machine learning",
    "neural network training"
}

log("=== Test de tokenization ===\n")

for _, text in ipairs(test_texts) do
    log("Text: \"" .. text .. "\"")
    
    local tokens = tokenizer.tokenize(text)
    
    log("  Tokens: [")
    for i, tok_id in ipairs(tokens) do
        local token_str = tokenizer.get_token_by_id(tok_id)
        log(string.format("    %d: %s", tok_id, token_str))
    end
    log("  ]")
    
    local decoded = tokenizer.detokenize(tokens)
    log("  Decoded: \"" .. decoded .. "\"\n")
end

-- 4. Test tokens spéciaux
log("=== Tokens spéciaux ===")
log(string.format("  PAD: %d", tokenizer.pad_id()))
log(string.format("  UNK: %d", tokenizer.unk_id()))
log(string.format("  SEQ: %d", tokenizer.seq_id()))
log(string.format("  MOD: %d", tokenizer.mod_id()))
log(string.format("  MAG: %d", tokenizer.mag_id()))

log("\n✓ Test terminé")
