#!/usr/bin/env mimir --lua

-- Entraînement simple du tokenizer sur un dataset texte.
-- Le dataset peut être :
--  - un fichier (1 ligne = 1 exemple)
--  - un dossier (scan récursif des fichiers .txt/.md/.jsonl)
--
-- Usage:
--   ./scripts/training/train_tokenizer.lua <dataset_path> [max_vocab] [out.json]

local function script_dir()
    local src = debug.getinfo(1, "S").source
    if type(src) ~= "string" then return "." end
    if src:sub(1, 1) == "@" then src = src:sub(2) end
    return (src:match("^(.*)/[^/]*$") or ".")
end

local function to_number(s, default)
    local n = tonumber(s)
    if n == nil then return default end
    return n
end

local function shell_quote(s)
    if s == nil then return "''" end
    s = tostring(s)
    return "'" .. s:gsub("'", "'\\''") .. "'"
end

local function is_dir(path)
    local cmd = "test -d " .. shell_quote(path) .. " >/dev/null 2>&1"
    local ok, why, code = os.execute(cmd)
    if type(ok) == "number" then
        return ok == 0
    end
    if type(ok) == "boolean" then
        return ok
    end
    if why == "exit" and code == 0 then
        return true
    end
    return false
end

local function ensure_parent_dir(filepath)
    local dir = tostring(filepath):match("^(.*)/[^/]*$")
    if dir and #dir > 0 then
        os.execute("mkdir -p " .. shell_quote(dir))
    end
end

local function iter_text_files(root)
    -- .txt (principal), .md (docs/corpus), .jsonl (lignes JSON) : on tokenise la ligne brute.
    local find_cmd = table.concat({
        "find ", shell_quote(root),
        " -type f ",
        "( -name '*.txt' -o -name '*.md' -o -name '*.jsonl' )",
        " -print"
    })
    local p = io.popen(find_cmd)
    if not p then return function() return nil end end
    return function()
        local line = p:read("*l")
        if line == nil then p:close() end
        return line
    end
end

-- NOTE: selon la façon dont Mímir lance Lua, la table globale `arg` peut être absente.
-- On supporte donc aussi les variables d'environnement :
--   TOKENIZER_DATASET, TOKENIZER_MAX_VOCAB, TOKENIZER_OUT
local dataset_path = (arg and arg[1]) or os.getenv("TOKENIZER_DATASET") or (script_dir() .. "/../tests/fixtures/tokenizer_dataset.txt")
local max_vocab = to_number((arg and arg[2]) or os.getenv("TOKENIZER_MAX_VOCAB"), 50000)
local out_path = (arg and arg[3]) or os.getenv("TOKENIZER_OUT") or "build/tokenizer_dataset.json"

log("=== Entraînement Tokenizer (dataset) ===")
log("Dataset: " .. dataset_path)
log("max_vocab: " .. tostring(max_vocab))
log("out: " .. out_path .. "\n")

tokenizer.create(max_vocab)

local line_count = 0
local file_count = 0

local function train_on_file(path)
    local f = io.open(path, "r")
    if not f then
        log("[WARN] Impossible d'ouvrir: " .. tostring(path))
        return
    end
    file_count = file_count + 1
    for line in f:lines() do
        if line and #line > 0 then
            tokenizer.ensure_vocab_from_text(line)
            line_count = line_count + 1
            if (line_count % 20000) == 0 then
                log(string.format("  ... %d lignes (fichiers=%d, vocab=%d)", line_count, file_count, tokenizer.vocab_size()))
            end
        end
    end
    f:close()
end

local function train_with_datasetloader(root_dir)
    log("Mode: datasetloader (dataset.load)")
    local ok, n_or_err = Mimir.Dataset.load(root_dir)
    if not ok then
        log("[WARN] dataset.load a échoué: " .. tostring(n_or_err))
        return false
    end

    local n_items = tonumber(n_or_err) or 0
    if n_items <= 0 then
        log("[WARN] Dataset vide")
        return true
    end

    for i = 1, n_items do
        local item = dataset.get(i)
        if item then
            if item.text and #tostring(item.text) > 0 then
                tokenizer.ensure_vocab_from_text(tostring(item.text))
                line_count = line_count + 1
            elseif item.text_file and #tostring(item.text_file) > 0 then
                train_on_file(tostring(item.text_file))
            end

            if (i % 2000) == 0 then
                log(string.format("  ... %d items (lignes=%d, fichiers=%d, vocab=%d)", i, line_count, file_count, tokenizer.vocab_size()))
            end
        end
    end

    return true
end

if is_dir(dataset_path) then
    -- Préférer le datasetloader (standard Mímir) pour gérer datasets multimodaux.
    -- Fallback: scan récursif si datasetloader échoue.
    local ok = train_with_datasetloader(dataset_path)
    if not ok then
        log("Mode: dossier (fallback scan récursif)")
        for file in iter_text_files(dataset_path) do
            if file == nil then break end
            train_on_file(file)
        end
    end
else
    log("Mode: fichier")
    train_on_file(dataset_path)
end

log(string.format("\n✓ Terminé: %d lignes (%d fichiers)", line_count, file_count))
log(string.format("✓ Vocab size: %d", tokenizer.vocab_size()))

-- Optionnel: stats
-- tokenizer.print_stats()

ensure_parent_dir(out_path)
tokenizer.save(out_path)
log("✓ Tokenizer sauvegardé")
