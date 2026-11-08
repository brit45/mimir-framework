#!/usr/bin/env lua
-- Thin CLI for the native CausalLMModel generation and conversation-memory API.
-- Prompts are passed in full; inference sizes the native graph at runtime.

local Args = dofile(ROOTWORK .. "/scripts/modules/args.lua")
local FS = dofile(ROOTWORK .. "/scripts/modules/fs.lua")
local CheckpointResume = dofile(ROOTWORK .. "/scripts/modules/checkpoint_resume.lua")
local opts = Args.parse(arg) or {}
local config_path = Args.get_str(opts, "conf",
    Args.get_str(opts, "config", "configs/causal_lm.json"))
local config, config_err = read_json(config_path)
assert(type(config) == "table" and type(config.model) == "table",
       "invalid causal LM config '" .. config_path .. "': " .. tostring(config_err))
config = Args.apply_overrides(config, opts)

local checkpoint = Args.get_str(opts, "checkpoint", "")
local tokenizer_path = Args.get_str(opts, "tokenizer", "")
local prompt = Args.get_str(opts, "prompt", "")
local prompt_file = Args.get_str(opts, "prompt-file", "")
local interactive = Args.get_bool(opts, "interactive", false)
local memory_disabled = Args.get_bool(opts, "no-memory", false)
local checkpoint_format = Args.get_str(opts, "format", "raw_folder")
local memory_path = Args.get_str(opts, "memory-file", "")

local function fail(message)
  error("[causal_lm_generate] " .. tostring(message), 0)
end

if checkpoint == "" then fail("pass --checkpoint <trained-checkpoint>") end
local saved_model_config
if checkpoint_format == "raw_folder" then
  checkpoint = CheckpointResume.resolve_dir(checkpoint)
  if not checkpoint then fail("no loadable raw_folder checkpoint") end
  local architecture_json, architecture_err = read_json(FS.join(checkpoint, "model", "architecture.json"))
  if type(architecture_json) ~= "table" then fail("checkpoint architecture: " .. tostring(architecture_err)) end
  saved_model_config = architecture_json.model_config
elseif checkpoint_format == "safetensors" then
  local Tools = dofile(ROOTWORK .. "/scripts/modules/mpk_tools.lua")
  local info, info_err = Tools.checkpoint(checkpoint, checkpoint_format)
  if type(info) ~= "table" then fail("checkpoint architecture: " .. tostring(info_err)) end
  saved_model_config = info.config
else
  fail("--format must be raw_folder or safetensors")
end
if type(saved_model_config) ~= "table" or saved_model_config.type ~= "causal_lm" then
  fail("checkpoint model_config must describe causal_lm")
end
if tokenizer_path == "" then
  local saved_tokenizer = FS.join(checkpoint, "tokenizer", "tokenizer.json")
  tokenizer_path = checkpoint_format == "raw_folder" and FS.file_exists(saved_tokenizer) and saved_tokenizer or
    (type(config.tokenizer) == "table" and config.tokenizer.path or "")
end
if tokenizer_path == "" then fail("pass --tokenizer <tokenizer.json>") end
if prompt ~= "" and prompt_file ~= "" then
  fail("use either --prompt or --prompt-file, not both")
end
if interactive and prompt_file ~= "" then
  fail("--prompt-file is for one-shot generation; omit --interactive")
end
if prompt_file ~= "" then
  local file, open_err = io.open(prompt_file, "rb")
  if not file then fail("cannot read prompt file '" .. prompt_file .. "': " .. tostring(open_err)) end
  local contents, read_err = file:read("*a")
  file:close()
  if contents == nil then
    fail("cannot read prompt file '" .. prompt_file .. "': " .. tostring(read_err))
  end
  prompt = contents
end
if prompt == "" and not interactive then fail("pass --prompt <text> or use --interactive") end

local loaded_tokenizer, tokenizer_err = Mimir.Tokenizer.load(tokenizer_path)
if not loaded_tokenizer then fail("Tokenizer.load: " .. tostring(tokenizer_err)) end

-- Serialized topology is authoritative: launch config may describe another run.
local model_cfg = saved_model_config
local architecture = "causal_lm"
local tokenizer_vocab = tonumber(Mimir.Tokenizer.vocab_size())
local capacity = tonumber(model_cfg.vocab_size)
if not tokenizer_vocab or not capacity or tokenizer_vocab ~= capacity then
  fail(string.format("tokenizer vocab (%s) differs from checkpoint vocabulary (%s)",
      tostring(tokenizer_vocab), tostring(capacity)))
end
model_cfg.vocab_size = tokenizer_vocab
-- CausalLMModel sizes its inference graph from the complete prompt at runtime.

local model_config = {}
for key, value in pairs(model_cfg) do
  if key ~= "architecture" then model_config[key] = value end
end
model_config.causal_arch_version = model_cfg.causal_arch_version or 1
model_config.local_conv_kernel = model_cfg.local_conv_kernel or 0
model_config.local_conv_dilation = model_cfg.local_conv_dilation or 1
model_config.fused_swiglu = model_cfg.fused_swiglu == true
if type(model_config.dtype) == "string" then
  local aliases = { F32 = "float32", FP32 = "float32", F16 = "float16", FP16 = "float16",
    BF16 = "bfloat16", F64 = "float64", FP64 = "float64" }
  model_config.dtype = aliases[model_config.dtype:upper()] or model_config.dtype
end
local created, create_err = Mimir.Model.create(architecture, model_config)
if not created then fail("Model.create: " .. tostring(create_err)) end
local allocated, allocate_err = Mimir.Model.allocate_params()
if not allocated then fail("Model.allocate_params: " .. tostring(allocate_err)) end
local loaded, load_err = Mimir.Serialization.load(checkpoint, checkpoint_format, {
  load_optimizer = false,
  load_tokenizer = false,
  load_encoder = false,
  strict_mode = true,
  apply_model_config = false,
  load_layer_metadata = false,
})
if not loaded then fail("Serialization.load: " .. tostring(load_err)) end

local system_prompt = Args.get_str(opts, "system-prompt", "")
if system_prompt == "" and type(config.chat) == "table" then
  system_prompt = type(config.chat.system_prompt) == "string"
      and config.chat.system_prompt or ""
end

local generation = {
  max_new_tokens = Args.get_int(opts, "max-new-tokens", 64),
  temperature = Args.get_num(opts, "temperature", 0.8),
  top_k = Args.get_int(opts, "top-k", 40),
  top_p = Args.get_num(opts, "top-p", 0.95),
  seed = Args.get_int(opts, "seed", 1337),
  stop_id = Args.get_int(opts, "stop-id", -1),
  system_prompt = system_prompt,
  use_memory = not memory_disabled,
}

if interactive then
  if memory_path == "" and not memory_disabled then
    local training = type(config.training) == "table" and config.training or {}
    memory_path = (training.checkpoint_dir or "checkpoints/lina_chat") .. "/conversation_memory.json"
  end
  local memory_count = 0
  if not memory_disabled then
    local count, count_err = Mimir.Model.causal_lm_memory_size(memory_path)
    if count == nil then fail("load conversation memory: " .. tostring(count_err)) end
    memory_count = count
  end
  print("Chat CausalLM. /quit termine, /reset efface la mémoire.")
  if memory_disabled then
    print("Mémoire désactivée.")
  else
    print(string.format("Mémoire chargée : %d tour(s) (%s)", memory_count, memory_path))
  end
  while true do
    io.write("Vous> ")
    io.flush()
    local message = io.read("*l")
    if not message or message == "/quit" or message == "/exit" then break end
    if message == "/reset" or message == "/forget" then
      if memory_disabled then print("Mémoire désactivée.")
      else
        local ok, err = Mimir.Model.causal_lm_reset_memory(memory_path)
        if not ok then print("Mémoire non effacée : " .. tostring(err))
        else print("Mémoire de conversation effacée.") end
      end
    elseif message == "/memory" then
      if memory_disabled then print("Mémoire désactivée.")
      else
        local count, err = Mimir.Model.causal_lm_memory_size(memory_path)
        if count == nil then print("Mémoire illisible : " .. tostring(err))
        else print(string.format("%d tour(s) mémorisé(s).", count)) end
      end
    elseif message ~= "" then
      local response, err = Mimir.Model.causal_lm_chat(message, memory_path, generation)
      if response == nil then fail("native chat: " .. tostring(err)) end
      print("Assistant> " .. response)
    end
  end
else
  local response, err = Mimir.Model.causal_lm_generate(prompt, generation)
  if response == nil then fail("native generation: " .. tostring(err)) end
  print(response)
end
