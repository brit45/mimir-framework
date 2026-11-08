local Bootstrap = dofile(ROOTWORK.."/scripts/modules/causal_lm_tokenizer.lua")
local path = "/tmp/mimir_causal_lm_tokenizer_bootstrap.json"
os.remove(path)

local corpus = table.concat({
  "alpha beta gamma delta epsilon",
  "the quick brown fox jumps over the lazy dog",
  "un petit corpus causal reproductible",
}, "\n")

-- Missing tokenizer: generate, validate and persist one.
local tokens, status = Bootstrap.ensure({
  corpus = corpus,
  path = path,
  vocab_size = 128,
  padding_idx = 0,
  seq_len = 8,
})
assert(type(tokens) == "table" and #tokens > 8)
assert(status.generated == true)
assert(status.replaced_loaded_tokenizer == false)
local saved = io.open(path, "rb")
assert(saved, "generated tokenizer was not persisted")
saved:close()

-- A vocabulary with too many assigned IDs must be replaced before building the model.
assert(Mimir.Tokenizer.create(256))
local oversized_words = {}
for index = 1, 120 do oversized_words[index] = "oversizedtoken" .. index end
assert(Mimir.Tokenizer.ensure_vocab_from_text(table.concat(oversized_words, " ")))
assert(tonumber(Mimir.Tokenizer.vocab_size()) > 128)
assert(Mimir.Tokenizer.save(path))

local replaced_tokens, replaced = Bootstrap.ensure({
  corpus = corpus,
  path = path,
  vocab_size = 128,
  padding_idx = 0,
  seq_len = 8,
})
assert(type(replaced_tokens) == "table" and #replaced_tokens > 8)
assert(replaced.generated == true)
assert(replaced.replaced_loaded_tokenizer == true)
assert(tonumber(Mimir.Tokenizer.get_max_vocab()) == 128)
assert(tonumber(Mimir.Tokenizer.vocab_size()) <= 128)

-- A compatible tokenizer must cover new words with BPE and persist its IDs.
assert(Mimir.Tokenizer.create(128))
assert(Mimir.Tokenizer.ensure_vocab_from_text("alpha beta"))
assert(Mimir.Tokenizer.save(path))
local old_vocab_size = tonumber(Mimir.Tokenizer.vocab_size())
local composed_tokens, composed = Bootstrap.ensure({
  corpus = "alpha beta zéphyrly novamatière",
  path = path,
  vocab_size = 128,
  padding_idx = 0,
  seq_len = 8,
})
assert(type(composed_tokens) == "table")
assert(composed.generated == false and composed.composed == true)
assert(composed.added_tokens >= 2)
assert(tonumber(Mimir.Tokenizer.vocab_size()) > old_vocab_size)
local unk_id = tonumber(Mimir.Tokenizer.unk_id())
local new_word_ids = Mimir.Tokenizer.tokenize("zéphyrly")
assert(new_word_ids[2] ~= unk_id, "composed word still maps to UNK")

assert(Mimir.Tokenizer.create(128))
assert(Mimir.Tokenizer.load(path))
local reloaded_ids = Mimir.Tokenizer.tokenize("zéphyrly")
assert(reloaded_ids[2] ~= tonumber(Mimir.Tokenizer.unk_id()),
       "composed word was not persisted")

-- Identical corpus: no second BPE study and no tokenizer file rewrite.
local function contents()
  local file = assert(io.open(path, "rb"))
  local text = file:read("*a")
  file:close()
  return text
end
local serialized = contents()
local saved_merges = tonumber(Mimir.Tokenizer.bpe_merge_count())
local reused_tokens, reused = Bootstrap.ensure({
  corpus = "alpha beta zéphyrly novamatière", path = path,
  vocab_size = 128, padding_idx = 0, seq_len = 8,
})
assert(type(reused_tokens) == "table")
assert(reused.generated == false and reused.tokenizer_reused == true)
assert(reused.tokenizer_bpe_updated == false and reused.added_tokens == 0)
assert(tonumber(Mimir.Tokenizer.bpe_merge_count()) == saved_merges)
assert(contents() == serialized, "covered corpus rewrote the tokenizer")

-- The reserved tokens plus the corpus alphabet cannot fit into 64 slots.
-- Insufficient capacity must fail, rather than silently train on UNK.
os.remove(path)
local ok, err = pcall(Bootstrap.ensure, {
  corpus = corpus, path = path, vocab_size = 64, padding_idx = 0, seq_len = 8,
})
assert(not ok and tostring(err):find("increase model.vocab_size", 1, true))
os.remove(path)
print("PASS causal LM tokenizer bootstrap")
