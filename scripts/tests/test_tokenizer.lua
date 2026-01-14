#!/usr/bin/env mimir --lua

log("=== Test du Tokenizer ===\n")

local function script_dir()
    local src = debug.getinfo(1, "S").source
    if type(src) ~= "string" then return "." end
    if src:sub(1, 1) == "@" then src = src:sub(2) end
    return (src:match("^(.*)/[^/]*$") or ".")
end

local dataset_path = (arg and arg[1]) or (script_dir() .. "/fixtures/tokenizer_dataset.txt")

-- 1. Créer tokenizer
tokenizer.create(50000)
log("✓ Tokenizer créé\n")

-- 2. Entraîner le vocabulaire à partir d'un dataset texte
log("=== Entraînement du tokenizer sur dataset ===")
log("Dataset: " .. dataset_path)

local f = io.open(dataset_path, "r")
if not f then
    error("Impossible d'ouvrir le dataset: " .. dataset_path)
end

local line_count = 0
for line in f:lines() do
    if line and #line > 0 then
        tokenizer.ensure_vocab_from_text(line)
        line_count = line_count + 1
        if (line_count % 2000) == 0 then
            log(string.format("  ... %d lignes (vocab=%d)", line_count, tokenizer.vocab_size()))
        end
    end
end
f:close()

log(string.format("✓ Dataset chargé: %d lignes", line_count))
log(string.format("✓ Vocab size après entraînement: %d\n", tokenizer.vocab_size()))

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
