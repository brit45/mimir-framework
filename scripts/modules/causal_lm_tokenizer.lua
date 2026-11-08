---@class MimirCausalTokenizerModule
local M = {}

local function readable(path)
  if type(path) ~= "string" or path == "" then return false end
  local file = io.open(path, "rb")
  if not file then return false end
  file:close()
  return true
end

local function tokenizer_compatibility(model_vocab, padding_idx, probe_text)
  local actual = tonumber(Mimir.Tokenizer.vocab_size())
  local maximum = tonumber(Mimir.Tokenizer.get_max_vocab())
  if not actual or actual < 7 then
    return false, "vocabulary is missing required special tokens"
  end
  if actual > model_vocab then
    return false, string.format("vocabulary has %d entries, model supports %d", actual, model_vocab)
  end
  if maximum and maximum > model_vocab then
    return false, string.format("tokenizer capacity is %d, model supports %d", maximum, model_vocab)
  end

  local specials = {
    pad = tonumber(Mimir.Tokenizer.pad_id()),
    unk = tonumber(Mimir.Tokenizer.unk_id()),
    seq = tonumber(Mimir.Tokenizer.seq_id()),
    mod = tonumber(Mimir.Tokenizer.mod_id()),
    mag = tonumber(Mimir.Tokenizer.mag_id()),
  }
  for name, id in pairs(specials) do
    if not id or id < 0 or id >= model_vocab then
      return false, string.format("invalid %s token id: %s", name, tostring(id))
    end
  end
  if specials.pad ~= padding_idx then
    return false, string.format(
      "PAD id %d differs from model padding_idx %d", specials.pad, padding_idx)
  end

  local tokenize_probe = type(Mimir.Tokenizer.tokenize_bpe) == "function"
      and Mimir.Tokenizer.tokenize_bpe or Mimir.Tokenizer.tokenize
  local tokens, token_err = tokenize_probe(probe_text)
  if type(tokens) ~= "table" then
    return false, "probe tokenization failed: " .. tostring(token_err)
  end
  for i, id in ipairs(tokens) do
    if not id or id < 0 or id >= model_vocab or id % 1 ~= 0 then
      return false, string.format("probe token #%d is outside model vocabulary: %s", i, tostring(id))
    end
  end
  return true, nil, tokens
end

local function count_unknown(tokens)
  local unknown_id = tonumber(Mimir.Tokenizer.unk_id())
  local count = 0
  if unknown_id then
    for _, token_id in ipairs(tokens) do
      if token_id == unknown_id then count = count + 1 end
    end
  end
  return count
end

---@param options MimirCausalTokenizerOptions
---@return TokenIds tokens
---@return MimirCausalTokenizerInfo info
function M.ensure(options)
  assert(type(options) == "table", "causal tokenizer options must be a table")
  local corpus = assert(options.corpus, "causal tokenizer corpus is required")
  local model_vocab = math.floor(assert(tonumber(options.vocab_size), "causal tokenizer vocab_size is required"))
  local padding_idx = math.floor(tonumber(options.padding_idx) or 0)
  local seq_len = math.floor(tonumber(options.seq_len) or 512)
  local path = options.path or ""
  local bpe_merges = math.floor(tonumber(options.bpe_merges) or 1000)
  assert(bpe_merges > 0, "causal LM requires a positive tokenizer.bpe_merges value")
  local loaded = false
  local reason

  local function compose_current_tokenizer()
    local old_vocab_size = tonumber(Mimir.Tokenizer.vocab_size()) or 0
    local old_merges = tonumber(Mimir.Tokenizer.bpe_merge_count()) or 0
    local before_ok, _, before_tokens = tokenizer_compatibility(model_vocab, padding_idx, corpus)
    -- BPE base symbols must get space before whole-word additions or merges.
    -- A loaded BPE that already covers the corpus needs no second study.
    local needs_study = old_merges == 0 or not before_ok or count_unknown(before_tokens or {}) > 0
    if needs_study then
      local learned, learn_err = Mimir.Tokenizer.learn_bpe({corpus}, bpe_merges)
      assert(learned ~= false, "tokenizer BPE learning failed: " .. tostring(learn_err))
    end

    local compatible, why, tokens = tokenizer_compatibility(model_vocab, padding_idx, corpus)
    local unknown_tokens = type(tokens) == "table" and count_unknown(tokens) or nil
    local vocab_size = tonumber(Mimir.Tokenizer.vocab_size()) or 0
    return tokens, {
      compatible = compatible,
      reason = why,
      unknown_tokens = unknown_tokens,
      vocab_size = vocab_size,
      added_tokens = math.max(0, vocab_size - old_vocab_size),
      composed_tokens = math.max(0, vocab_size - old_vocab_size),
      bpe_added_tokens = math.max(0, vocab_size - old_vocab_size),
      bpe_merges = tonumber(Mimir.Tokenizer.bpe_merge_count()) or 0,
      bpe_updated = needs_study,
    }
  end

  if readable(path) then
    local ok, err = Mimir.Tokenizer.load(path)
    if ok ~= false then
      loaded = true
      local old_vocab_size = tonumber(Mimir.Tokenizer.vocab_size()) or 0
      local old_max_vocab = tonumber(Mimir.Tokenizer.get_max_vocab()) or 0
      if old_vocab_size > model_vocab then
        reason = string.format("loaded vocabulary has %d entries, model supports %d",
                               old_vocab_size, model_vocab)
      else
        local capacity_ok, capacity_err = Mimir.Tokenizer.set_max_vocab(model_vocab)
        assert(capacity_ok ~= false, "tokenizer capacity update failed: " .. tostring(capacity_err))
        Mimir.Tokenizer.set_max_length(seq_len)

        local tokens, composed = compose_current_tokenizer()
        if composed.compatible and composed.unknown_tokens == 0 then
          local capacity_changed = old_max_vocab ~= model_vocab
          local changed = composed.added_tokens > 0 or capacity_changed or composed.bpe_updated
          if changed then
            local saved, save_err = Mimir.Tokenizer.save(path)
            assert(saved ~= false, "composed tokenizer save failed: " .. tostring(save_err))
          end
          return tokens, {
            generated = false,
            composed = composed.composed_tokens > 0 or capacity_changed,
            tokenizer_reused = not changed,
            tokenizer_bpe_updated = composed.bpe_updated,
            added_tokens = composed.added_tokens,
            composed_tokens = composed.composed_tokens,
            bpe_added_tokens = composed.bpe_added_tokens,
            unknown_tokens = composed.unknown_tokens,
            bpe_merges = composed.bpe_merges,
            bpe_merges_requested = bpe_merges,
            path = path,
            vocab_size = composed.vocab_size,
            max_vocab = tonumber(Mimir.Tokenizer.get_max_vocab()),
          }
        end
        reason = composed.reason or string.format(
          "%d corpus tokens still map to <UNK> after BPE", composed.unknown_tokens or 0)
      end
    else
      reason = "load failed: " .. tostring(err)
    end
  else
    reason = "file is absent or unreadable"
  end

  local created, create_err = Mimir.Tokenizer.create(model_vocab)
  assert(created ~= false, "Tokenizer.create failed: " .. tostring(create_err))
  Mimir.Tokenizer.set_max_length(seq_len)
  local tokens, composed = compose_current_tokenizer()
  assert(composed.compatible,
         "generated tokenizer is incompatible with model: " .. tostring(composed.reason))
  assert(composed.unknown_tokens == 0,
         string.format("BPE cannot represent %d corpus tokens; increase model.vocab_size",
                       composed.unknown_tokens))

  if path ~= "" then
    local saved, save_err = Mimir.Tokenizer.save(path)
    assert(saved ~= false, "generated tokenizer save failed: " .. tostring(save_err))
  end

  return tokens, {
    generated = true,
    tokenizer_reused = false,
    tokenizer_bpe_updated = composed.bpe_updated,
    replaced_loaded_tokenizer = loaded,
    composed = composed.composed_tokens > 0,
    added_tokens = composed.added_tokens,
    composed_tokens = composed.composed_tokens,
    bpe_added_tokens = composed.bpe_added_tokens,
    unknown_tokens = composed.unknown_tokens,
    bpe_merges = composed.bpe_merges,
    bpe_merges_requested = bpe_merges,
    reason = reason,
    path = path,
    vocab_size = composed.vocab_size,
    max_vocab = tonumber(Mimir.Tokenizer.get_max_vocab()),
  }
end

---@param options MimirCausalTokenizerOptions
---@return TokenIds tokens
---@return MimirCausalTokenizerInfo info
function M.load_prepared(options)
  assert(type(options) == "table", "causal tokenizer options must be a table")
  local path = assert(options.path, "prepared causal tokenizer path is required")
  local corpus = assert(options.corpus, "causal tokenizer corpus is required")
  local model_vocab = math.floor(assert(tonumber(options.vocab_size), "causal tokenizer vocab_size is required"))
  local padding_idx = math.floor(tonumber(options.padding_idx) or 0)
  local seq_len = math.floor(tonumber(options.seq_len) or 512)
  assert(readable(path), "prepared causal tokenizer is missing: " .. tostring(path))
  local loaded, load_err = Mimir.Tokenizer.load(path)
  assert(loaded ~= false, "prepared causal tokenizer load failed: " .. tostring(load_err))
  Mimir.Tokenizer.set_max_length(seq_len)

  local compatible, why, tokens = tokenizer_compatibility(model_vocab, padding_idx, corpus)
  assert(compatible, "prepared tokenizer is incompatible: " .. tostring(why))
  local unknown_tokens = count_unknown(tokens)
  assert(unknown_tokens == 0,
         string.format("prepared tokenizer leaves %d corpus tokens as <UNK>", unknown_tokens))
  return tokens, {
    generated = false,
    composed = false,
    added_tokens = 0,
    unknown_tokens = unknown_tokens,
    bpe_merges = tonumber(Mimir.Tokenizer.bpe_merge_count()) or 0,
    path = path,
    vocab_size = tonumber(Mimir.Tokenizer.vocab_size()) or 0,
    max_vocab = tonumber(Mimir.Tokenizer.get_max_vocab()) or 0,
  }
end

return M
