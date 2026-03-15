-- Train VAEConv (vae_conv) sur un dataset d'images (image_file).
--
-- Usage:
--   ./bin/mimir --lua scripts/training/train_vae_conv.lua -- \
--     --dataset-root ../dataset \
--     --out-dir checkpoint/vae_conv_trained \
--     --epochs 5 --lr 1e-4 \
--     --warmup-steps 200 --kl-warmup-steps 200 \
--     --autosave-every-epochs 1 \
--     --image-w 64 --image-h 64 --latent-h 8 --latent-w 8 --latent-c 4 \
--     --base-channels 32 --kl-beta 1.0 \
--     --mem-gb 15 --alloc-gb 15 --max-items 0
--
-- Viz (optionnel):
--   --viz --viz-taps-max-frames 12 --viz-taps-max-side 64
--   --viz-title "VAEConv" --viz-width 1600 --viz-height 900 --viz-fps 60
--   --viz-hide-activation-blocks false
--
-- TUI (optionnel):
--   --htop (ou --tux)
--
-- Robustesse:
--   --validate-every 200 --validate-items 8 --validate-holdout-frac 0.01
--   --triple-fault true --fault-every 500
--
-- Perf:
--   La compression mémoire de l'allocateur peut coûter cher CPU sur des gros modèles.
--   Si vous avez assez de RAM, essayez: --compression false

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

-- Par défaut on conserve le comportement existant (compression activée).
-- Désactivez pour accélérer si la RAM le permet.
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

local dataset_root = opt_str("dataset-root", "../dataset")
local out_dir = opt_str("out-dir", "checkpoint/vae_conv_trained")
local RESUME = opt_bool("resume", true)
local epochs = opt_int("epochs", 5)
local lr = opt_num("lr", 1e-4)
local seed = opt_int("seed", opt_int("init-seed", 4242))

local cfg, err = Mimir.Architectures.default_config("vae_conv")
assert(type(cfg) == "table", "default_config(vae_conv) failed: " .. tostring(err))

cfg.image_w = opt_int("image-w", cfg.image_w or 64)
cfg.image_h = opt_int("image-h", cfg.image_h or 64)
cfg.image_c = opt_int("image-c", cfg.image_c or 3)

cfg.latent_h = opt_int("latent-h", cfg.latent_h or 0)
cfg.latent_w = opt_int("latent-w", cfg.latent_w or 0)
cfg.latent_c = opt_int("latent-c", cfg.latent_c or 256)

cfg.base_channels = opt_int("base-channels", cfg.base_channels or 32)

-- Latent stochastique (réparameterisation). Important pour éviter un "AE déterministe" pénalisé au KL.
-- NOTE: côté C++, Reparameterize utilise mu directement si training=false OU stochastic_latent=false.
cfg.stochastic_latent = opt_bool("stochastic-latent", cfg.stochastic_latent or false)

-- Texte (optionnel)
cfg.text_cond = opt_bool("text-cond", cfg.text_cond or false)
cfg.seq_len = opt_int("seq-len", cfg.seq_len or 64)
cfg.text_d_model = opt_int("text-d-model", cfg.text_d_model or 64)
cfg.proj_dim = opt_int("proj-dim", cfg.proj_dim or (cfg.text_cond and 64 or 0))
cfg.align_weight = opt_num("align-weight", cfg.align_weight or 0.1)

-- IMPORTANT: base tokenizer commun (si texte activé)
local base_tok_path = opt_str("base-tokenizer", BaseTok.default_path())
do
  local ok_bt, err_bt = BaseTok.load_base({
    path = base_tok_path,
    max_vocab = opt_int("max-vocab", cfg.vocab_size or 50000),
    require = cfg.text_cond == true,
  })
  assert(ok_bt == true, "Base tokenizer: " .. tostring(err_bt))
end

if cfg.text_cond then
  cfg.vocab_size = BaseTok.vocab_size()
  cfg.tokenizer_frozen = true

  -- IMPORTANT: `Model::trainStepVAEText` exige les projections (img_proj + text_proj)
  -- donc proj_dim doit être > 0.
  if (cfg.proj_dim or 0) <= 0 then
    cfg.proj_dim = 64
  end
end

-- Si latent_h/w non fournis, on dérive (downsample x8 par défaut)
if (cfg.latent_h or 0) <= 0 then cfg.latent_h = math.max(1, math.floor(cfg.image_h / 8)) end
if (cfg.latent_w or 0) <= 0 then cfg.latent_w = math.max(1, math.floor(cfg.image_w / 8)) end

-- Renseigne `latent_dim` pour éviter l'inférence ambiguë (et garder la compat training/inférence).
cfg.latent_dim = math.max(1, (cfg.latent_h or 0) * (cfg.latent_w or 0) * (cfg.latent_c or 0))

-- Options d'entraînement consommées côté C++ (LuaScripting.cpp)
cfg.optimizer = opt_str("optimizer", cfg.optimizer or "adamw")
cfg.beta1 = opt_num("beta1", cfg.beta1 or 0.9)
cfg.beta2 = opt_num("beta2", cfg.beta2 or 0.999)
cfg.epsilon = opt_num("epsilon", cfg.epsilon or 1e-8)
-- VAEConv: le weight decay trop fort dégrade souvent la reconstruction.
cfg.weight_decay = opt_num("weight-decay", cfg.weight_decay or 1e-8)

cfg.decay_strategy = opt_str("decay-strategy", cfg.decay_strategy or "cosine")

cfg.kl_beta = opt_num("kl-beta", cfg.kl_beta or 1.0)
-- Stabilisation VAE (consommée côté C++ par Model::trainStepVAE)
-- Par défaut: ramp-up du KL sur ~1/2 époque (dataset ~1967 linkables)
cfg.kl_warmup_steps = opt_int("kl-warmup-steps", cfg.kl_warmup_steps or 100)

-- Recon loss (consommé côté C++ par Model::trainStepVAE)
cfg.recon_loss = opt_str("recon-loss", cfg.recon_loss or "l1")

-- Losses additionnelles (optionnelles)
cfg.ssim_weight = opt_num("ssim-weight", cfg.ssim_weight or 0.0)
cfg.ssim_mode = opt_str("ssim-mode", cfg.ssim_mode or "ssim") -- "ssim" ou "ms_ssim"
cfg.ssim_k1 = opt_num("ssim-k1", cfg.ssim_k1 or 0.01)
cfg.ssim_k2 = opt_num("ssim-k2", cfg.ssim_k2 or 0.03)
cfg.ssim_L = opt_num("ssim-L", cfg.ssim_L or 2.0)

cfg.spectral_weight = opt_num("spectral-weight", cfg.spectral_weight or 0.0)
cfg.spectral_scales = opt_int("spectral-scales", cfg.spectral_scales or 1)

cfg.perceptual_weight = opt_num("perceptual-weight", cfg.perceptual_weight or 0.0)
cfg.perceptual_arch = opt_str("perceptual-arch", cfg.perceptual_arch or "vgg16_feat")
cfg.perceptual_checkpoint = opt_str("perceptual-ckpt", cfg.perceptual_checkpoint or "")
cfg.perceptual_base_channels = opt_int("perceptual-base-channels", cfg.perceptual_base_channels or 8)

cfg.adv_weight = opt_num("adv-weight", cfg.adv_weight or 0.0)
cfg.adv_disc_arch = opt_str("adv-disc-arch", cfg.adv_disc_arch or "patch_discriminator")
cfg.adv_disc_lr = opt_num("adv-disc-lr", cfg.adv_disc_lr or lr)
cfg.adv_disc_base_channels = opt_int("adv-disc-base-channels", cfg.adv_disc_base_channels or 32)

-- Paramètres recon loss
cfg.huber_delta = opt_num("huber-delta", cfg.huber_delta or 1.0)
cfg.charbonnier_eps = opt_num("charbonnier-eps", cfg.charbonnier_eps or 1e-3)
cfg.nll_sigma = opt_num("nll-sigma", cfg.nll_sigma or 1.0)

-- Warmup LR (consommé côté C++ via Optimizer.warmup_steps)
-- Important: le scheduler applique le warmup sur `opt.initial_lr`.
cfg.warmup_steps = opt_int("lr-warmup-steps", opt_int("warmup-steps", cfg.warmup_steps or 200))

-- Autosave (consommé côté C++ dans LuaScripting::lua_trainModel)
-- 0 = désactiver. 1 = sauvegarde à chaque epoch.
cfg.autosave_every_epochs = opt_int("autosave-every-epochs", opt_int("autosave_every_epochs", cfg.autosave_every_epochs or 1))

-- Marqueurs (Wasserstein/Temporal) qui modulent la loss de reconstruction côté C++.
-- Par défaut: désactivé (0.0) pour conserver un training identique.
cfg.marker_wass_scale = opt_num("marker-wass-scale", cfg.marker_wass_scale or 0.0)
cfg.marker_temp_scale = opt_num("marker-temp-scale", cfg.marker_temp_scale or 0.0)
cfg.marker_warmup_steps = opt_int("marker-warmup-steps", cfg.marker_warmup_steps or 0)
cfg.marker_scale_max = opt_num("marker-scale-max", cfg.marker_scale_max or 10.0)
-- Clamp logvar plus serré => std dans ~[exp(-3), exp(1)] = [0.05, 2.7]
cfg.logvar_clip_min = opt_num("logvar-clip-min", cfg.logvar_clip_min or -6.0)
cfg.logvar_clip_max = opt_num("logvar-clip-max", cfg.logvar_clip_max or 2.0)
-- Clip grad pour éviter un emballement en milieu d'époque
cfg.grad_clip_norm = opt_num("grad-clip-norm", cfg.grad_clip_norm or 1.0)
cfg.grad_accum_steps = opt_int("grad-accum-steps", cfg.grad_accum_steps or 1)

cfg.max_items = opt_int("max-items", cfg.max_items or 0)
cfg.log_every = opt_int("log-every", cfg.log_every or 10)

-- IMPORTANT: utilisé côté C++ pour le shuffle/ordre dataset.
cfg.seed = seed

-- Viz taps (consommés côté C++ si viz active)
cfg.viz_taps_max_frames = opt_int("viz-taps-max-frames", cfg.viz_taps_max_frames or 24)
cfg.viz_taps_max_side = opt_int("viz-taps-max-side", cfg.viz_taps_max_side or 256)

-- Checkpoints/validation (consommés côté C++ dans Mimir.Model.train)
cfg.checkpoint_dir = out_dir

Args.apply_validation_config(cfg, opts)

cfg.triple_fault = opt_bool("triple-fault", false)
cfg.triple_fault_every_steps = opt_int("fault-every", opt_int("triple-fault-every", 5))

log("VAEConv train")
log(string.format("  dataset_root=%s", dataset_root))
log(string.format("  out_dir=%s", out_dir))
log(string.format("  image=%dx%dx%d", cfg.image_w, cfg.image_h, cfg.image_c))
log(string.format("  latent=%dx%dx%d", cfg.latent_h, cfg.latent_w, cfg.latent_c))
log(string.format("  base_channels=%d", cfg.base_channels))
log(string.format("  stochastic_latent=%s", tostring(cfg.stochastic_latent)))
if cfg.text_cond then
  log(string.format("  text_cond=true seq_len=%d vocab_size=%d text_d_model=%d proj_dim=%d align_weight=%g",
    cfg.seq_len or 0, cfg.vocab_size or 0, cfg.text_d_model or 0, cfg.proj_dim or 0, cfg.align_weight or 0.0))
  log(string.format("  base_tokenizer=%s", base_tok_path))
end
log(string.format("  epochs=%d lr=%g kl_beta=%g", epochs, lr, cfg.kl_beta))
log(string.format("  seed=%d", seed))
log(string.format("  warmup: lr_warmup_steps=%d kl_warmup_steps=%d", cfg.warmup_steps or 0, cfg.kl_warmup_steps or 0))
log(string.format("  logvar_clip=[%g,%g] grad_clip_norm=%g",
  cfg.logvar_clip_min, cfg.logvar_clip_max, cfg.grad_clip_norm))
log(string.format("  autosave_every_epochs=%d", cfg.autosave_every_epochs or 0))
log(string.format("  markers: wass_scale=%g temp_scale=%g warmup_steps=%d scale_max=%g",
  cfg.marker_wass_scale or 0.0, cfg.marker_temp_scale or 0.0, cfg.marker_warmup_steps or 0, cfg.marker_scale_max or 0.0))
log(string.format("  grad_accum_steps=%d", cfg.grad_accum_steps))
if (cfg.validate_every_steps or 0) > 0 then
  log(string.format("  validate_every_steps=%d validate_items=%d holdout_frac=%.4g",
    cfg.validate_every_steps, cfg.validate_items, cfg.validate_holdout_frac))
end
if cfg.triple_fault then
  log(string.format("  triple_fault=true fault_every=%d", cfg.triple_fault_every_steps))
end

-- Heuristiques de qualité: si le latent est trop petit ou si KL+latent déterministe => détails lissés.
do
  local image_dim = (cfg.image_w or 0) * (cfg.image_h or 0) * (cfg.image_c or 0)
  local latent_dim = (cfg.latent_h or 0) * (cfg.latent_w or 0) * (cfg.latent_c or 0)
  if image_dim > 0 and latent_dim > 0 then
    local ratio = latent_dim / image_dim
    if ratio < 0.05 then
      log(string.format("⚠️  Latent très compressé: latent_dim=%d (%.3g x image_dim=%d). Risque de perte de détails.", latent_dim, ratio, image_dim))
    end
  end
  if (cfg.stochastic_latent == false) and ((cfg.kl_beta or 0) > 0) then
    log("⚠️  stochastic_latent=false avec KL>0: le KL pousse mu→0 et lisse les détails. Essaye --stochastic-latent true, ou baisse --kl-beta (ex: 0.01) si tu veux garder un encodeur quasi-déterministe.")
  end
end

-- Dataset
local ok_ds, n_or_err = Mimir.Dataset.load(dataset_root, cfg.image_w, cfg.image_h, cfg.text_cond and 2 or 1, true, 'dataset_cache.json', 10240, true)
assert_ok(ok_ds, n_or_err, "Dataset.load failed")
log("✓ Dataset chargé: " .. tostring(n_or_err))

-- Modèle
assert_ok(Mimir.Model.create("vae_conv", cfg), nil, "Model.create(vae_conv) failed")
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
if ok_train == false and tostring(err_train) == "STOP_REQUESTED" then
  log("⛔ Stop demandé via Viz: autosave effectué. Sauvegarde finale (complète) puis fin du programme.")

  -- On ré-écrit dans le dernier dossier epoch_* (incluant *_stop si présent)
  local last_dir = nil
  if Ckpt and Ckpt.find_latest_epoch_dir then
    last_dir = Ckpt.find_latest_epoch_dir(out_dir)
  end
  if not last_dir and Ckpt and Ckpt.resolve_dir then
    last_dir = Ckpt.resolve_dir(out_dir)
  end
  if not last_dir then
    last_dir = out_dir
  end

  os.execute("mkdir -p '" .. tostring(last_dir):gsub("'", "'\\''") .. "' 2>/dev/null")
  local ok_save_stop, err_save_stop = Mimir.Serialization.save(last_dir, "raw_folder", {
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
  assert_ok(ok_save_stop, err_save_stop, "Serialization.save(stop) failed")
  log("✓ Checkpoint STOP écrit: " .. tostring(last_dir))
  return
end
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

log("✓ Checkpoint VAEConv écrit: " .. out_dir)
