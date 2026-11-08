-- Thin configuration entry point. Dataset preparation, BPE, BOS/EOS framing,
-- control tags, sampling, loss, optimizer steps and metrics belong to the
-- native CausalLMModel. seq_len is the training window; inference grows to the
-- complete prompt length at runtime.

local DEFAULT_CONFIG_PATH = "configs/causal_lm.json"
local Args = dofile(ROOTWORK .. "/scripts/modules/args.lua")
local FS = dofile(ROOTWORK .. "/scripts/modules/fs.lua")
local CheckpointResume = dofile(ROOTWORK .. "/scripts/modules/checkpoint_resume.lua")
-- DDP must initialize before monitor threads start; rank 0 opens the UI later.
local opts = Args.parse(arg, {defer_monitor = true}) or {}
local config_path = Args.get_str(opts, "conf", Args.get_str(opts, "config", DEFAULT_CONFIG_PATH))

if
  type(CONF) ~= "table" or type(CONF.model) ~= "table" or type(CONF.training) ~= "table" or
    type(CONF.dataset) ~= "table"
 then
  local loaded, load_err = read_json(config_path)
  assert(type(loaded) == "table", "cannot load causal LM config '" .. config_path .. "': " .. tostring(load_err))
  CONF = loaded
end
assert(
  type(CONF.model) == "table" and type(CONF.training) == "table" and type(CONF.dataset) == "table",
  "causal LM config requires model, training and dataset sections"
)
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
apply_cli(CONF.model, "seq_len", "train-window", Args.get_int)
apply_cli(CONF.model, "d_model", "d-model", Args.get_int)
apply_cli(CONF.model, "num_layers", "layers", Args.get_int)
apply_cli(CONF.model, "num_heads", "heads", Args.get_int)
apply_cli(CONF.model, "num_kv_heads", "kv-heads", Args.get_int)
apply_cli(CONF.model, "mlp_hidden", "mlp-hidden", Args.get_int)
apply_cli(CONF.model, "latent_dim", "latent-dim", Args.get_int)
apply_cli(CONF.model, "local_conv_kernel", "local-conv-kernel", Args.get_int)
apply_cli(CONF.model, "local_conv_dilation", "local-conv-dilation", Args.get_int)
apply_cli(CONF.model, "fused_swiglu", "fused-swiglu", Args.get_bool)
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
apply_cli(CONF.training, "resume", "resume", Args.get_bool)
apply_cli(CONF.training, "amp", "amp", Args.get_bool)
apply_cli(CONF.training, "amp_dtype", "amp-dtype", Args.get_str)
apply_cli(CONF.training, "ddp", "ddp", Args.get_bool)
CONF = Args.apply_overrides(CONF, opts)

local model_cfg = CONF.model
local train_cfg = CONF.training
local tokenizer_cfg = CONF.tokenizer

local checkpoint_dir = train_cfg.checkpoint_dir or "checkpoints/causal_lm"
local resume_from = Args.get_str(opts, "resume-from", Args.get_str(opts, "resume-dir", nil))
local resume_requested = resume_from ~= nil or
  Args.get_bool(opts, "resume", train_cfg.resume == true) == true
local resume_dir = nil
local resume_architecture = nil
local resume_model_config = nil
if resume_requested then
  resume_dir = resume_from or CheckpointResume.resolve_dir(checkpoint_dir)
  if resume_dir then
    local architecture_path = FS.join(resume_dir, "model", "architecture.json")
    local tokenizer_checkpoint_path = FS.join(resume_dir, "tokenizer", "tokenizer.json")
    local ok_architecture, loaded_architecture = pcall(read_json, architecture_path)
    assert(ok_architecture and type(loaded_architecture) == "table",
      "checkpoint de reprise illisible: " .. architecture_path)
    assert(loaded_architecture.model_name == "causal_lm" or
      loaded_architecture.model_type == "causal_lm" or
      (type(loaded_architecture.model_config) == "table" and
        (loaded_architecture.model_config.architecture == "causal_lm" or
         loaded_architecture.model_config.type == "causal_lm")),
      "le checkpoint sélectionné n'est pas un modèle causal_lm: " .. tostring(resume_dir))
    assert(CheckpointResume.file_exists(tokenizer_checkpoint_path),
      "tokenizer absent du checkpoint: " .. tokenizer_checkpoint_path)
    assert(type(loaded_architecture.model_config) == "table",
      "model_config absent du checkpoint: " .. tostring(resume_dir))
    local training_path = FS.join(resume_dir, "model", "training.json")
    local ok_training, loaded_training = pcall(read_json, training_path)
    assert(ok_training and type(loaded_training) == "table" and loaded_training.has_optimizer == true,
      "état d'optimiseur absent ou illisible dans le checkpoint: " .. training_path)
    resume_architecture = loaded_architecture
    resume_model_config = loaded_architecture.model_config
  elseif resume_from then
    error("checkpoint demandé introuvable ou invalide: " .. tostring(resume_from))
  else
    print("[causal_lm] aucun checkpoint trouvé; démarrage depuis une initialisation neuve")
    resume_requested = false
    train_cfg.resume = false
  end
end

local function check_cli_topology(option, saved_key, label, fallback_key, legacy_default)
  local requested = opts[option]
  if requested == nil or requested == true or not resume_model_config then return end
  local saved = resume_model_config[saved_key]
  if saved == nil and fallback_key then saved = resume_model_config[fallback_key] end
  if saved == nil then saved = legacy_default end
  local requested_number = tonumber(requested)
  local saved_number = tonumber(saved)
  if requested_number ~= nil and saved_number ~= nil and requested_number ~= saved_number then
    error(string.format(
      "--%s=%s est incompatible avec la topologie du checkpoint (%s=%s); retire cette option pour reprendre avec la topologie enregistrée",
      option, tostring(requested), label, tostring(saved)))
  end
end

if resume_model_config then
  check_cli_topology("seq-len", "seq_len", "seq_len")
  check_cli_topology("train-window", "seq_len", "seq_len")
  check_cli_topology("d-model", "d_model", "d_model")
  check_cli_topology("layers", "num_layers", "num_layers")
  check_cli_topology("heads", "num_heads", "num_heads")
  check_cli_topology("kv-heads", "num_kv_heads", "num_kv_heads")
  check_cli_topology("mlp-hidden", "mlp_hidden", "mlp_hidden")
  check_cli_topology("latent-dim", "latent_dim", "latent_dim")
  check_cli_topology("local-conv-kernel", "local_conv_kernel", "local_conv_kernel", nil, 0)
  check_cli_topology("local-conv-dilation", "local_conv_dilation", "local_conv_dilation", nil, 1)
  if opts["fused-swiglu"] ~= nil then
    local saved = resume_model_config.fused_swiglu == true
    assert(Args.get_bool(opts, "fused-swiglu", saved) == saved,
      "--fused-swiglu est incompatible avec la topologie du checkpoint")
  end
  check_cli_topology("vocab-size", "tokenizer_capacity", "tokenizer_capacity", "vocab_size")

  local saved_seed = tonumber(resume_model_config.seed)
  if saved_seed == nil and type(resume_model_config.training_state) == "table" then
    saved_seed = tonumber(resume_model_config.training_state.initial_seed)
  end
  if saved_seed ~= nil then
    if opts.seed ~= nil and tonumber(opts.seed) ~= saved_seed then
      error(string.format("--seed=%s est incompatible avec la graine de reprise (%d)",
        tostring(opts.seed), saved_seed))
    end
    train_cfg.seed = saved_seed
  end

  local function has_config_override(path)
    local overrides = opts.override
    if overrides == nil then return false end
    if type(overrides) ~= "table" then overrides = {overrides} end
    for _, expression in ipairs(overrides) do
      if tostring(expression):match("^" .. path:gsub("%.", "%%.") .. "=") then
        return true
      end
    end
    return false
  end
  local saved_learning_rate = tonumber(resume_model_config.learning_rate)
  for _, field in ipairs({
    {"local_conv_kernel", 0}, {"local_conv_dilation", 1}, {"fused_swiglu", false},
  }) do
    if has_config_override("model." .. field[1]) then
      local saved = resume_model_config[field[1]]
      if saved == nil then saved = field[2] end
      assert(model_cfg[field[1]] == saved,
        "model." .. field[1] .. " est incompatible avec la topologie du checkpoint")
    end
  end
  if saved_learning_rate and opts["learning-rate"] == nil and opts.lr == nil and
      not has_config_override("training.learning_rate") then
    train_cfg.learning_rate = saved_learning_rate
  end
  for _, key in ipairs({"amp", "amp_dtype", "grad_clip_norm"}) do
    local cli_key = key:gsub("_", "-")
    local saved = resume_model_config[key]
    if saved ~= nil and opts[cli_key] == nil and
        not has_config_override("training." .. key) then
      train_cfg[key] = saved
    end
  end
  if resume_model_config.ddp ~= nil then
    local saved_ddp = resume_model_config.ddp == true
    if opts.ddp ~= nil and Args.get_bool(opts, "ddp", saved_ddp) ~= saved_ddp then
      error("le mode DDP du checkpoint ne peut pas être changé pendant la reprise")
    end
    if not has_config_override("training.ddp") then
      train_cfg.ddp = saved_ddp
    end
  end
  local saved_optimizer_fields = {
    {"optimizer", "optimizer"}, {"beta1", "beta1"}, {"beta2", "beta2"},
    {"epsilon", "epsilon"}, {"eps", "eps"}, {"weight_decay", "weight-decay"},
    {"rmsprop_alpha", "rmsprop-alpha"}, {"min_lr", "min-lr"},
    {"decay_rate", "decay-rate"}, {"decay_steps", "decay-steps"},
    {"decay_strategy", "decay-strategy"}, {"warmup_steps", "warmup-steps"},
    {"adafactor_clip_threshold", "adafactor-clip-threshold"},
    {"adafactor_decay_rate", "adafactor-decay-rate"},
    {"adafactor_eps2", "adafactor-eps2"}, {"adafactor_beta1", "adafactor-beta1"},
    {"adafactor_scale_parameter", "adafactor-scale-parameter"},
    {"adafactor_relative_step", "adafactor-relative-step"},
  }
  for _, field in ipairs(saved_optimizer_fields) do
    local config_key, cli_key = field[1], field[2]
    local saved = resume_model_config[config_key]
    if saved ~= nil then
      local requested = opts[cli_key]
      local overridden = has_config_override("training." .. config_key)
      if overridden then requested = train_cfg[config_key] end
      if requested ~= nil and requested ~= true then
        local requested_number, saved_number = tonumber(requested), tonumber(saved)
        local differs
        if requested_number ~= nil and saved_number ~= nil then
          differs = requested_number ~= saved_number
        else
          differs = tostring(requested) ~= tostring(saved)
        end
        if differs then
          error(string.format("--%s ne peut pas modifier l'état optimiseur sauvegardé (%s=%s)",
            cli_key, config_key, tostring(saved)))
        end
      end
      train_cfg[config_key] = saved
    end
  end

  -- La partition validation et la tokenizer font partie de l'état du run.
  -- Les options CLI explicites restent vérifiables, mais les valeurs de la
  -- configuration courante ne doivent pas redécouper les exemples au hasard.
  local validation_keys = {
    {"validate_every_steps", "validate-every-steps"},
    {"validate_items", "validate-items"},
    {"validate_holdout_frac", "validate-holdout-frac"},
    {"validate_holdout_items", "validate-holdout-items"},
    {"validate_holdout", "validate-holdout"},
  }
  for _, pair in ipairs(validation_keys) do
    local saved_value = resume_model_config[pair[1]]
    if saved_value ~= nil and opts[pair[2]] == nil then
      train_cfg[pair[1]] = saved_value
    end
  end
  train_cfg.resume = true
end

Args.apply_validation_config(train_cfg, opts, {
  validate_every_steps = tonumber(train_cfg.validate_every_steps) or 100,
  validate_items = tonumber(train_cfg.validate_items) or 8,
  validate_holdout_frac = tonumber(train_cfg.validate_holdout_frac) or 0.1,
  validate_holdout_items = tonumber(train_cfg.validate_holdout_items) or 0,
  validate_holdout = train_cfg.validate_holdout ~= false,
  validate_save_debug = train_cfg.validate_save_debug == true,
})
local architecture = model_cfg.architecture or "causal_lm"
if resume_model_config then
  -- Reprise stricte: le checkpoint reste la source de vérité pour la topo.
  for key, value in pairs(resume_model_config) do
    model_cfg[key] = value
  end
  model_cfg.architecture = architecture
  model_cfg.vocab_size = tonumber(resume_model_config.tokenizer_capacity) or
    tonumber(resume_model_config.vocab_size) or model_cfg.vocab_size
  model_cfg.causal_arch_version = resume_model_config.causal_arch_version or 1
  model_cfg.local_conv_kernel = resume_model_config.local_conv_kernel or 0
  model_cfg.local_conv_dilation = resume_model_config.local_conv_dilation or 1
  model_cfg.fused_swiglu = resume_model_config.fused_swiglu == true
end
local training_window = tonumber(model_cfg.seq_len)
assert(
  training_window and training_window >= 2 and training_window == math.floor(training_window),
  "model.seq_len/--train-window must be an integer of at least 2"
)
model_cfg.seq_len = training_window
local seed = math.floor(tonumber(train_cfg.seed) or 1337)
local distributed_enabled = train_cfg.ddp == true
local rank, world_size = 0, 1
if distributed_enabled then
  local ok, err = Mimir.Distributed.initialize()
  assert(ok, "DDP initialization failed: " .. tostring(err))
  rank = Mimir.Distributed.rank()
  world_size = Mimir.Distributed.world_size()
  assert(world_size > 1, "DDP needs at least two MPI processes (launch with mpirun)")
end
local is_main = not distributed_enabled or Mimir.Distributed.is_main_process()
if resume_model_config and type(resume_model_config.training_state) == "table" then
  local saved_world_size = tonumber(resume_model_config.training_state.world_size)
  if saved_world_size and saved_world_size ~= world_size then
    error(string.format("la reprise DDP exige le même nombre de processus (checkpoint=%d, actuel=%d)",
      saved_world_size, world_size))
  end
end

local tokenizer_path = tokenizer_cfg.path
if resume_dir then
  tokenizer_path = FS.join(resume_dir, "tokenizer", "tokenizer.json")
end
if type(tokenizer_path) ~= "string" or tokenizer_path == "" then
  tokenizer_path = checkpoint_dir .. "/tokenizer.json"
end
local corpus_path = assert(CONF.dataset.text_path, "dataset.text_path is required")
local bpe_merges = math.max(1, math.floor(tonumber(tokenizer_cfg.bpe_merges) or 1000))
local steps = math.max(1, math.floor(tonumber(train_cfg.steps) or 1000))
local learning_rate = tonumber(train_cfg.learning_rate) or 3e-4
assert(learning_rate > 0, "training.learning_rate must be positive")

local config, config_err = Mimir.Architectures.default_config(architecture)
assert(type(config) == "table", "default_config failed: " .. tostring(config_err))
for key, value in pairs(model_cfg) do
  if key ~= "architecture" then
    config[key] = value
  end
end
for key, value in pairs(train_cfg) do
  config[key] = value
end
if resume_model_config then
  for key, value in pairs(resume_model_config) do
    config[key] = value
  end
  for key, value in pairs(train_cfg) do
    config[key] = value
  end
  -- Runtime choices for this invocation are allowed, while topology, tokenizer
  -- metadata and validation split stay tied to the checkpoint.
  config.vocab_size = tonumber(resume_model_config.tokenizer_capacity) or
    tonumber(resume_model_config.vocab_size) or config.vocab_size
  config.seq_len = tonumber(resume_model_config.seq_len) or config.seq_len
  config.seed = seed
  config.resume = true
end
config.checkpoint_dir = checkpoint_dir
config.log_every = math.max(1, math.floor(tonumber(train_cfg.log_every) or 10))
config.save_every = math.max(0, math.floor(tonumber(train_cfg.save_every) or 500))
config.seed = seed
local latent_dim = tonumber(config.latent_dim)
if latent_dim == nil or latent_dim < 0 then
  local d_model = math.max(2, math.floor(tonumber(config.d_model) or 256))
  latent_dim = math.max(1, math.min(128, math.floor(d_model / 2)))
end
config.latent_dim = math.floor(latent_dim)

-- Checkpoints use SafeTensors dtype names; Model.create expects runtime names.
if type(config.dtype) == "string" then
  local aliases = { F32 = "float32", FP32 = "float32", F16 = "float16", FP16 = "float16",
    BF16 = "bfloat16", F64 = "float64", FP64 = "float64" }
  config.dtype = aliases[config.dtype:upper()] or config.dtype
end

local created, create_err = Mimir.Model.create(architecture, config)
assert(created, "Model.create failed: " .. tostring(create_err))
if is_main then
  print(
    string.format(
      "[causal_lm] fenêtre d'entraînement=%d tokens; la longueur du prompt d'inférence est dynamique",
      training_window
    )
  )
  print(string.format("[causal_lm] bottleneck latent contextuel: %d dimensions par position", config.latent_dim))
  print(string.format("[causal_lm] convolution locale: noyau=%d dilatation=%d; SwiGLU natif=%s",
    tonumber(config.local_conv_kernel) or 3, tonumber(config.local_conv_dilation) or 1,
    tostring(config.fused_swiglu ~= false)))
end

if train_cfg.amp == true then
  local dtype = train_cfg.amp_dtype or "bfloat16"
  local ok, err = Mimir.Model.set_autocast(dtype)
  assert(ok, "AMP initialization failed: " .. tostring(err))
  if is_main then
    print("[causal_lm] AMP=" .. tostring(dtype))
  end
end

local prepared, prepare_result = Mimir.Model.causal_lm_prepare_corpus(corpus_path, tokenizer_path, bpe_merges, seed)
assert(prepared, "native corpus/BPE preparation failed: " .. tostring(prepare_result))
if resume_model_config then
  assert(prepare_result.tokenizer_reused == true,
    "la reprise a modifié ou remplacé le tokenizer du checkpoint; vérifie le corpus et sa couverture BPE")
  assert(prepare_result.tokenizer_bpe_updated ~= true,
    "la reprise ne peut pas réapprendre le BPE; utilise le corpus/tokenizer d'origine")
  local prepared_metrics = {
    training_samples = prepare_result.samples,
    training_tokens = prepare_result.tokens,
    dataset_samples = prepare_result.dataset_samples,
    validation_samples = prepare_result.validation_samples,
    validation_tokens = prepare_result.validation_tokens,
  }
  for _, key in ipairs({"training_samples", "training_tokens", "dataset_samples", "validation_samples", "validation_tokens"}) do
    local saved = tonumber(resume_model_config[key])
    local actual = tonumber(prepared_metrics[key])
    if saved ~= nil and actual ~= nil then
      assert(saved == actual, string.format(
        "le corpus préparé diffère du checkpoint (%s: sauvegardé=%d, actuel=%d)", key, saved, actual))
    end
  end
end
if is_main then
  print("[causal_lm] format: <BOS> ... <EOS>; balises <SYSTEM>/<USER>/<CONTEXT>/<REFLECT>/<ASSISTANT>/<FINAL>")
  if prepare_result.tokenizer_reused then
    print(
      string.format(
        "[causal_lm] tokenizer réutilisé: vocab=%d/%d merges=%d",
        tonumber(prepare_result.vocab_size) or 0,
        tonumber(prepare_result.vocab_capacity) or 0,
        tonumber(prepare_result.bpe_merges) or 0
      )
    )
  end
  print(
    string.format(
      "[causal_lm] dataset natif: train_samples=%d validation_items=%d validation_tokens=%d holdout=%s tokens=%d vocab=%d/%d merges=%d",
      tonumber(prepare_result.samples) or 0,
      tonumber(prepare_result.validation_samples) or 0,
      tonumber(prepare_result.validation_tokens) or 0,
      tostring(prepare_result.validation_holdout == true),
      tonumber(prepare_result.tokens) or 0,
      tonumber(prepare_result.vocab_size) or 0,
      tonumber(prepare_result.vocab_capacity) or 0,
      tonumber(prepare_result.bpe_merges) or 0
    )
  )
  local validation_active = (tonumber(train_cfg.validate_every_steps) or 0) > 0 and
    (tonumber(train_cfg.validate_items) or 0) > 0
  print(string.format(
    "[causal_lm] validation %s: toutes les %d étapes, %d items maximum, source=%s",
    validation_active and "active" or "désactivée",
    tonumber(train_cfg.validate_every_steps) or 0,
    tonumber(train_cfg.validate_items) or 0,
    tostring(prepare_result.validation_source or "training")
  ))
end

local allocated, allocate_err = Mimir.Model.allocate_params()
assert(allocated, "allocate_params failed: " .. tostring(allocate_err))
if resume_dir then
  local loaded, load_err = Mimir.Serialization.load(resume_dir, "raw_folder", {
    load_encoder = false,
    load_tokenizer = true,
    load_optimizer = true,
    strict_mode = true,
    validate_checksums = true,
    apply_model_config = false,
  })
  assert(loaded, "échec du chargement du checkpoint de reprise: " .. tostring(load_err))
  local state = type(resume_model_config.training_state) == "table" and
    resume_model_config.training_state or {}
  local training_path = FS.join(resume_dir, "model", "training.json")
  local training_state = read_json(training_path)
  local resumed_step = tonumber(training_state.step) or
    tonumber(type(training_state.loop) == "table" and training_state.loop.global_step) or
    tonumber(state.global_step) or 0
  if tonumber(state.global_step) and tonumber(state.global_step) ~= resumed_step then
    error(string.format("état incohérent: optimizer.step=%d mais model_config.global_step=%d",
      resumed_step, tonumber(state.global_step)))
  end
  if is_main then
    print(string.format("[causal_lm] reprise depuis %s (step=%d)", tostring(resume_dir), resumed_step))
  end
else
  local initialized, init_err = Mimir.Model.init_weights("xavier", seed)
  assert(initialized, "init_weights failed: " .. tostring(init_err))
end
if is_main then
  Args.start_monitor(opts)
end
if distributed_enabled and is_main then
  print(string.format("[causal_lm] DDP=%d ranks (MPI gradient average)", world_size))
end

local completed, train_result = Mimir.Model.causal_lm_train(steps, learning_rate, seed + rank * 104729)
assert(completed, "native CausalLMModel training failed: " .. tostring(train_result))
if is_main then
  local initial_step = 0
  if resume_model_config and type(resume_model_config.training_state) == "table" then
    initial_step = tonumber(resume_model_config.training_state.global_step) or 0
  end
  print(string.format("[causal_lm] entraînement terminé: %d pas supplémentaires, step global=%d/%d",
    train_result, initial_step + train_result, steps))
end

if distributed_enabled then
  local ok, err = Mimir.Distributed.barrier()
  assert(ok, "DDP shutdown barrier failed: " .. tostring(err))
  assert(Mimir.Distributed.shutdown())
end
