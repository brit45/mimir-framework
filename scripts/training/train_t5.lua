#!/usr/bin/env lua
-- Native T5 encoder pretraining with a tied masked-span denoising head.
-- ./run_mimir.sh --conf configs/t5.json
-- This is not the encoder-decoder T5 span-infilling objective.
local DEFAULT_CONFIG_PATH = "configs/t5.json"
local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local opts = Args.parse(arg) or {}
local cli_config_path =
    Args.get_str(opts, "conf", Args.get_str(opts, "config", DEFAULT_CONFIG_PATH))

if type(CONF) ~= "table"
    or type(CONF.model) ~= "table"
    or type(CONF.training) ~= "table"
    or type(CONF.dataset) ~= "table" then
  local loaded, load_err = read_json(cli_config_path)
  assert(type(loaded) == "table",
         "cannot load T5 config '" .. cli_config_path
         .. "': " .. tostring(load_err))
  CONF = loaded
  print("[t5] configuration: " .. cli_config_path)
end

assert(type(CONF.model) == "table"
       and type(CONF.training) == "table"
       and type(CONF.dataset) == "table",
       "T5 config requires model, training and dataset sections")

CONF.tokenizer = type(CONF.tokenizer) == "table" and CONF.tokenizer or {}

local function apply_cli(tbl, field, arg_name, getter)
  if opts[arg_name] ~= nil then
    tbl[field] = getter(opts, arg_name, tbl[field])
  end
end

apply_cli(CONF.dataset, "text_path", "corpus", Args.get_str)
apply_cli(CONF.dataset, "text_path", "text-path", Args.get_str)
apply_cli(CONF.tokenizer, "path", "tokenizer", Args.get_str)
apply_cli(CONF.tokenizer, "bpe_merges", "bpe-merges", Args.get_int)

apply_cli(CONF.model, "vocab_size", "vocab-size", Args.get_int)
apply_cli(CONF.model, "seq_len", "seq-len", Args.get_int)
apply_cli(CONF.model, "d_model", "d-model", Args.get_int)
apply_cli(CONF.model, "num_layers", "layers", Args.get_int)
apply_cli(CONF.model, "num_heads", "heads", Args.get_int)
apply_cli(CONF.model, "mlp_hidden", "mlp-hidden", Args.get_int)
apply_cli(CONF.model, "dtype", "dtype", Args.get_str)

apply_cli(CONF.training, "steps", "steps", Args.get_int)
apply_cli(CONF.training, "learning_rate", "learning-rate", Args.get_num)
apply_cli(CONF.training, "learning_rate", "lr", Args.get_num)
apply_cli(CONF.training, "optimizer", "optimizer", Args.get_str)
apply_cli(CONF.training, "seed", "seed", Args.get_int)
apply_cli(CONF.training, "weight_decay", "weight-decay", Args.get_num)
apply_cli(CONF.training, "grad_clip_norm", "grad-clip-norm", Args.get_num)
apply_cli(CONF.training, "log_every", "log-every", Args.get_int)
apply_cli(CONF.training, "save_every", "save-every", Args.get_int)
apply_cli(CONF.training, "checkpoint_dir", "checkpoint-dir", Args.get_str)
apply_cli(CONF.training, "amp", "amp", Args.get_bool)
apply_cli(CONF.training, "amp_dtype", "amp-dtype", Args.get_str)
apply_cli(CONF.training, "ddp", "ddp", Args.get_bool)

-- Les overrides structurés ont la priorité sur les raccourcis CLI.
CONF = Args.apply_overrides(CONF, opts)

local function require_readable_file(path, label)
  assert(type(path) == "string" and path ~= "",
         label .. " path is missing from the T5 configuration")
  local file, open_err = io.open(path, "rb")
  assert(file,
         string.format("%s not found or unreadable: %s (%s). "
                       .. "Update '%s' or pass another --conf file.",
                       label, path, tostring(open_err), cli_config_path))
  return file
end

local model_cfg = CONF.model
local train_cfg = CONF.training
local tokenizer_cfg = CONF.tokenizer or {}
local distributed_enabled = train_cfg.ddp == true
local rank, world_size = 0, 1
if distributed_enabled then
  local dist_ok, dist_err = Mimir.Distributed.initialize()
  assert(dist_ok, "DDP initialization failed: " .. tostring(dist_err))
  rank = Mimir.Distributed.rank()
  world_size = Mimir.Distributed.world_size()
  assert(world_size > 1, "DDP needs at least two MPI processes (launch with mpirun)")
end
local is_main_process = not distributed_enabled or Mimir.Distributed.is_main_process()
local seq_len = assert(tonumber(model_cfg.seq_len))
local vocab_size = assert(tonumber(model_cfg.vocab_size))
local architecture = "t5"
model_cfg.encoder_only = false

math.randomseed(tonumber(train_cfg.seed) or 1337)

local corpus_file = require_readable_file(CONF.dataset.text_path, "training corpus")
local corpus = corpus_file:read("*a")
corpus_file:close()

local tokenizer_path = tokenizer_cfg.path
if opts["checkpoint-dir"] and not opts.tokenizer then
  tokenizer_path = train_cfg.checkpoint_dir .. "/tokenizer.json"
end
if type(tokenizer_path) ~= "string" or tokenizer_path == "" then
  tokenizer_path = (train_cfg.checkpoint_dir or "checkpoint/t5") .. "/tokenizer.json"
  tokenizer_cfg.path = tokenizer_path
  CONF.tokenizer = tokenizer_cfg
end

local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")
local tokenizer_parent = tokenizer_path:match("^(.*)/[^/]+$")
if tokenizer_parent and tokenizer_parent ~= "" then FS.mkdir_p(tokenizer_parent) end
local TextTokenizer = dofile(ROOTWORK.."/scripts/modules/causal_lm_tokenizer.lua")
local function ensure_tokenizer()
  return TextTokenizer.ensure({
    corpus = corpus,
    path = tokenizer_path,
    vocab_size = vocab_size,
    padding_idx = tonumber(model_cfg.padding_idx) or 0,
    seq_len = seq_len,
    bpe_merges = tokenizer_cfg.bpe_merges,
  })
end
local tokens, tokenizer_status
if distributed_enabled then
  if is_main_process then
    tokens, tokenizer_status = ensure_tokenizer()
    local barrier_ok, barrier_err = Mimir.Distributed.barrier()
    assert(barrier_ok, "DDP tokenizer barrier failed: " .. tostring(barrier_err))
  else
    local barrier_ok, barrier_err = Mimir.Distributed.barrier()
    assert(barrier_ok, "DDP tokenizer barrier failed: " .. tostring(barrier_err))
    tokens, tokenizer_status = ensure_tokenizer()
  end
else
  tokens, tokenizer_status = ensure_tokenizer()
end
if is_main_process and tokenizer_status.generated then
  print(string.format(
    "[t5] tokenizer généré: path=%s vocab=%d/%d reason=%s",
    tostring(tokenizer_status.path),
    tonumber(tokenizer_status.vocab_size) or 0,
    vocab_size,
    tostring(tokenizer_status.reason)))
elseif is_main_process then
  print(string.format(
    "[t5] tokenizer chargé: path=%s vocab=%d/%d",
    tostring(tokenizer_status.path),
    tonumber(tokenizer_status.vocab_size) or 0,
    vocab_size))
end
assert(#tokens >= seq_len, "corpus must contain at least seq_len tokens")

local cfg, cfg_err = Mimir.Architectures.default_config(architecture)
assert(cfg, tostring(cfg_err))
for key, value in pairs(model_cfg) do
  if key ~= "architecture" then cfg[key] = value end
end
for _, key in ipairs({"beta1", "beta2", "epsilon", "weight_decay", "grad_clip_norm"}) do
  if train_cfg[key] ~= nil then cfg[key] = train_cfg[key] end
end
assert(Mimir.Model.create(architecture, cfg))
if train_cfg.amp == true then
  local amp_dtype = train_cfg.amp_dtype or "bfloat16"
  local amp_ok, amp_err = Mimir.Model.set_autocast(amp_dtype)
  assert(amp_ok, "AMP initialization failed: " .. tostring(amp_err))
  if is_main_process then
    print(string.format("[t5] AMP=%s (Linear CPU autocast, accumulations float32)", amp_dtype))
  end
end
if distributed_enabled and is_main_process then
  print(string.format("[t5] DDP=%d ranks (MPI gradient average)", world_size))
end
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier", tonumber(train_cfg.seed) or 1337))
math.randomseed(math.floor(tonumber(train_cfg.seed) or 1337) + rank * 104729)

local mask_ratio = tonumber(train_cfg.mask_ratio) or 0.15
local mask_id = tonumber(train_cfg.mask_id) or (vocab_size - 1)
assert(mask_ratio > 0 and mask_ratio <= 1, "mask_ratio must be in (0,1]")
assert(mask_id >= 0 and mask_id < vocab_size and mask_id ~= (model_cfg.padding_idx or 0),
       "mask_id must be a non-padding vocabulary id")
local function sample()
  local first = math.random(1, #tokens - seq_len + 1)
  local input, target, selected = {}, {}, {}
  for i = 1, seq_len do
    input[i] = tokens[first + i - 1]
    target[i] = input[i]
  end
  -- Mask contiguous spans; only corrupted positions contribute to the loss.
  local count = math.max(1, math.floor(seq_len * mask_ratio))
  local start = math.random(1, seq_len - count + 1)
  for i = start, start + count - 1 do input[i] = mask_id; selected[i] = true end
  return input, target, selected, count
end

-- Mean cross-entropy and its exact gradient with respect to flattened logits.
local function cross_entropy(logits, target, selected, count)
  assert(#logits == seq_len * vocab_size,
         string.format("expected %d logits, got %d", seq_len * vocab_size, #logits))
  local grad, loss = {}, 0.0
  for position = 1, seq_len do
    local offset = (position - 1) * vocab_size
    local maximum = -math.huge
    for token = 1, vocab_size do
      maximum = math.max(maximum, logits[offset + token])
    end
    local denominator = 0.0
    for token = 1, vocab_size do
      denominator = denominator + math.exp(logits[offset + token] - maximum)
    end
    local gold = target[position] + 1 -- Mimir token ids are zero-based.
    assert(gold >= 1 and gold <= vocab_size, "target token outside vocabulary")
    if selected[position] then loss = loss + (math.log(denominator) + maximum - logits[offset + gold]) end
    for token = 1, vocab_size do
      local probability = math.exp(logits[offset + token] - maximum) / denominator
      grad[offset + token] = selected[position] and (probability - (token == gold and 1.0 or 0.0)) / count or 0
    end
  end
  return loss / count, grad
end

local steps = tonumber(train_cfg.steps) or 1000
assert(steps >= 1 and steps % 1 == 0, "steps must be a positive integer")
local learning_rate = tonumber(train_cfg.learning_rate) or 3e-4
local optimizer = train_cfg.optimizer or "adamw"
local log_every = tonumber(train_cfg.log_every) or 10
local save_every = tonumber(train_cfg.save_every) or 500
local checkpoint_dir = train_cfg.checkpoint_dir or "checkpoint/t5"
assert(log_every >= 1 and save_every >= 1, "log_every and save_every must be positive")

for step = 1, steps do
  local input, target, selected, count = sample()
  assert(Mimir.Model.zero_grads())
  local logits, forward_err =
      Mimir.Model.forward({text_ids = input}, true)
  assert(logits, "forward failed: " .. tostring(forward_err))
  local loss, gradient = cross_entropy(logits, target, selected, count)
  local backward_ok, backward_err = Mimir.Model.backward(gradient)
  assert(backward_ok, "backward failed: " .. tostring(backward_err))
  local step_ok, step_err = Mimir.Model.optimizer_step(learning_rate, optimizer)
  assert(step_ok, "optimizer step failed: " .. tostring(step_err))

  if is_main_process and (step == 1 or step % log_every == 0) then
    print(string.format("step=%d loss=%.6f perplexity=%.3f",
                        step, loss, math.exp(math.min(loss, 20))))
  end
  if is_main_process and step % save_every == 0 then
    local path = string.format("%s/step-%07d", checkpoint_dir, step)
    local save_ok, save_err = Mimir.Serialization.save(path, "raw_folder", {
      save_optimizer = true,
      save_tokenizer = true
    })
    assert(save_ok, "checkpoint save failed: " .. tostring(save_err))
  end
end

if is_main_process then
  assert(Mimir.Serialization.save(checkpoint_dir .. "/final", "raw_folder", {
    save_optimizer = true, save_tokenizer = true, save_encoder = false
  }))
  print("T5_TRAIN_OK " .. checkpoint_dir .. "/final")
end
if distributed_enabled then
  local barrier_ok, barrier_err = Mimir.Distributed.barrier()
  assert(barrier_ok, "DDP shutdown barrier failed: " .. tostring(barrier_err))
  assert(Mimir.Distributed.shutdown())
end
