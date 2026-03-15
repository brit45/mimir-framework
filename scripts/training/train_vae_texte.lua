-- Train VAEText (vae_text) sur un dataset de textes (text_file).
--
-- Usage:
--   ./bin/mimir --lua scripts/training/train_vae_texte.lua -- \
--     --dataset-root ../dataset_2 \
--     --out-dir checkpoint/vae_text_trained \
--     --epochs 5 --lr 1e-4 \
--     --seq-len 256 --d-model 256 --latent-tokens 32 \
--     --kl-beta 0.01 --logvar-clip-min -6 --logvar-clip-max 2 \
--     --mem-gb 15 --alloc-gb 15 --max-items 0
--
-- Viz (optionnel):
--   --viz --viz-taps-max-frames 12 --viz-taps-max-side 64
--   --viz-title "VAEText" --viz-width 1600 --viz-height 900 --viz-fps 60
--   --viz-hide-activation-blocks false
--
-- TUI (optionnel):
--   --htop (ou --tux)

local Args = dofile("scripts/modules/args.lua")
local opts = Args.parse(arg) or {}

local Ckpt = dofile("scripts/modules/checkpoint_resume.lua")

local function opt_num(k, d)
  local v = opts[k]
  if v == nil then return d end
  local n = tonumber(v)
  if n == nil then return d end
  return n
end

local function opt_int(k, d)
  return math.floor(opt_num(k, d))
end

local function opt_str(k, d)
  local v = opts[k]
  if v == nil or v == true then return d end
  return tostring(v)
end

local function opt_bool(k, d)
  local v = opts[k]
  if v == nil then return d end
  if v == true or v == false then return v end
  v = tostring(v):lower()
  if v == "1" or v == "true" or v == "yes" or v == "on" then return true end
  if v == "0" or v == "false" or v == "no" or v == "off" then return false end
  return d
end

local BaseTok = dofile("scripts/modules/base_tokenizer.lua")

local function assert_ok(ok, err, msg)
  if ok == false then
    error((msg or "Operation failed") .. ": " .. tostring(err))
  end
end

local MEM_GB = opt_num("mem-gb", 15)
local ALLOC_GB = opt_num("alloc-gb", MEM_GB)
local ENABLE_COMPRESSION = opt_bool("compression", opt_bool("compress", true))

if Mimir and Mimir.Allocator and Mimir.Allocator.configure then
  Mimir.Allocator.configure({max_ram_gb = ALLOC_GB, enable_compression = ENABLE_COMPRESSION})
end
if Mimir and Mimir.MemoryGuard and Mimir.MemoryGuard.setLimit then
  pcall(Mimir.MemoryGuard.setLimit, MEM_GB)
end
if Mimir and Mimir.Model and Mimir.Model.set_hardware then
  pcall(Mimir.Model.set_hardware, opt_bool("hw", true))
end

local dataset_root = opt_str("dataset-root", "../dataset_2")
local out_dir = opt_str("out-dir", "checkpoint/vae_text_trained")
local RESUME = opt_bool("resume", true)
local epochs = opt_int("epochs", 5)
local lr = opt_num("lr", 1e-4)
local seed = opt_int("seed", opt_int("init-seed", 1337))

local cfg, err = Mimir.Architectures.default_config("vae_text")
assert(type(cfg) == "table", "default_config(vae_text) failed: " .. tostring(err))

cfg.seq_len = opt_int("seq-len", cfg.seq_len or 256)
cfg.d_model = opt_int("d-model", cfg.d_model or 256)
cfg.num_layers = opt_int("layers", cfg.num_layers or 4)
cfg.num_heads = opt_int("heads", cfg.num_heads or 8)
cfg.mlp_hidden = opt_int("mlp-hidden", cfg.mlp_hidden or 1024)
cfg.latent_tokens = opt_int("latent-tokens", cfg.latent_tokens or 32)
cfg.proj_dim = opt_int("proj-dim", cfg.proj_dim or 256)
cfg.stochastic_latent = opt_bool("stochastic-latent", cfg.stochastic_latent ~= false)

-- Training knobs consumed by Model::trainStepVAEText
cfg.align_weight = opt_num("align-weight", cfg.align_weight or 0.0)
cfg.kl_beta = opt_num("kl-beta", cfg.kl_beta or 0.01)
cfg.kl_warmup_steps = opt_int("kl-warmup-steps", cfg.kl_warmup_steps or 0)
cfg.recon_loss = opt_str("recon-loss", cfg.recon_loss or "mse")
cfg.logvar_clip_min = opt_num("logvar-clip-min", cfg.logvar_clip_min or -6.0)
cfg.logvar_clip_max = opt_num("logvar-clip-max", cfg.logvar_clip_max or 2.0)
cfg.grad_accum_steps = opt_int("grad-accum-steps", cfg.grad_accum_steps or 1)
cfg.grad_clip_norm = opt_num("grad-clip-norm", cfg.grad_clip_norm or 1.0)
cfg.max_items = opt_int("max-items", cfg.max_items or 0)
cfg.log_every = opt_int("log-every", cfg.log_every or 10)

-- IMPORTANT: utilisé côté C++ pour le shuffle/ordre dataset.
cfg.seed = seed

-- Viz taps (consommés côté C++ si viz active)
cfg.viz_taps_max_frames = opt_int("viz-taps-max-frames", cfg.viz_taps_max_frames or 12)
cfg.viz_taps_max_side = opt_int("viz-taps-max-side", cfg.viz_taps_max_side or 64)

cfg.optimizer = opt_str("optimizer", cfg.optimizer or "adamw")
cfg.beta1 = opt_num("beta1", cfg.beta1 or 0.9)
cfg.beta2 = opt_num("beta2", cfg.beta2 or 0.999)
cfg.epsilon = opt_num("epsilon", cfg.epsilon or 1e-8)
cfg.weight_decay = opt_num("weight-decay", cfg.weight_decay or 1e-6)

-- IMPORTANT: base tokenizer commun
local base_tok_path = opt_str("base-tokenizer", BaseTok.default_path())
do
  local ok_bt, err_bt = BaseTok.load_base({
    path = base_tok_path,
    max_vocab = opt_int("max-vocab", cfg.vocab_size or 32000),
    require = true,
  })
  assert(ok_bt == true, "Base tokenizer: " .. tostring(err_bt))
end
cfg.vocab_size = BaseTok.vocab_size()
cfg.tokenizer_frozen = true

-- Checkpoint dir (utilisé pour interrupt checkpoints côté C++)
cfg.checkpoint_dir = out_dir

log("VAEText train")
log(string.format("  dataset_root=%s", dataset_root))
log(string.format("  out_dir=%s", out_dir))
log(string.format("  seq_len=%d d_model=%d latent_tokens=%d proj_dim=%d", cfg.seq_len, cfg.d_model, cfg.latent_tokens, cfg.proj_dim))
log(string.format("  layers=%d heads=%d mlp_hidden=%d", cfg.num_layers, cfg.num_heads, cfg.mlp_hidden))
log(string.format("  epochs=%d lr=%g kl_beta=%g", epochs, lr, cfg.kl_beta))
log(string.format("  logvar_clip=[%g,%g] grad_accum_steps=%d", cfg.logvar_clip_min, cfg.logvar_clip_max, cfg.grad_accum_steps))
log(string.format("  base_tokenizer=%s vocab_size=%d", base_tok_path, cfg.vocab_size))
log(string.format("  seed=%d", seed))

-- Dataset
local ok_ds, n_or_err = Mimir.Dataset.load(dataset_root)
assert_ok(ok_ds, n_or_err, "Dataset.load failed")
log("✓ Dataset chargé: " .. tostring(n_or_err))

-- Modèle
assert_ok(Mimir.Model.create("vae_text", cfg), nil, "Model.create(vae_text) failed")
local params = Mimir.Model.total_params()
log("✓ Model créé (registry): params=" .. tostring(params))

local ok_alloc, err_alloc = Mimir.Model.allocate_params()
assert_ok(ok_alloc, err_alloc, "Model.allocate_params failed")

local resumed_from = nil
if RESUME and Ckpt and Ckpt.resolve_dir then
  local resume_dir = Ckpt.resolve_dir(out_dir)
  if resume_dir then
    log("↩︎ Resume: chargement checkpoint: " .. tostring(resume_dir))
    local load_opts = {
      load_encoder = true,
      load_tokenizer = true,
      load_optimizer = true,
      strict_mode = false,
      validate_checksums = true
    }
    local ok_load, err_load = Mimir.Serialization.load(resume_dir, "raw_folder", load_opts)
    assert_ok(ok_load, err_load, "Serialization.load(resume) failed")
    resumed_from = resume_dir
  end
end

if not resumed_from then
  local init_method = opt_str("init", "xavier")
  local ok_init, err_init = Mimir.Model.init_weights(init_method, seed)
  assert_ok(ok_init, err_init, "Model.init_weights failed")
end

-- Entraînement
local ok_train, err_train = Mimir.Model.train(epochs, lr)
assert_ok(ok_train, err_train, "Model.train failed")

-- Sauvegarde
os.execute("mkdir -p '" .. out_dir:gsub("'", "'\\''") .. "' 2>/dev/null")
local ok_save, err_save = Mimir.Serialization.save(out_dir, "raw_folder", {
  save_optimizer = true,
  save_tokenizer = true,
  save_encoder = true,
  include_checksums = true,
  include_git_info = true,
  include_gradients = true,
  include_activations = true,
  include_optimizer_state = true,
  include_weight_deltas = true,
})
assert_ok(ok_save, err_save, "Serialization.save failed")

log("✓ Checkpoint VAEText écrit: " .. out_dir)
