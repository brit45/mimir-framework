#!/usr/bin/env lua
---@diagnostic disable: undefined-field, need-check-nil
-- Entraînement DDPM latent – générique (ponyxl_dppm / ponyxl_ddpm / …)
-- Le dataset est enregistré auprès de Mimir mais jamais itéré depuis Lua ;
-- Mimir.Model.train() gère entièrement données, validation, autosave et steps.
--
-- Exemple :
--   OMP_NUM_THREADS=10 ./run_mimir.sh --lua scripts/training/ponyxl_ddpm_train.lua --no-htop \
--     --dataset "dataset_2" \
--     --out-dir checkpoint/ponyxl_dppm-512_latent-64-64-32_base-8_float16 \
--     --arch ldm_unet \
--     --image-w 512 --image-h 512 --image-c 3 \
--     --vae-checkpoint checkpoint/vae_conv/epoch_0002_stop \
--     --kl-beta 1.0 --kl-warmup 600 --seed 4242 \
--     --epochs 5 --lr 1e-5 --lr-warmup-steps 400 --optimizer adamw \
--     --log-every 1 --validate-every-steps 64 --validate-items 4 --validate-holdout-frac 0.1 \
--     --viz --resume 0 --viz-hide-activation-blocks=true --viz-hide-normalisation-blocks=true \
--     --dtype "fp16"

-- ── Logging ───────────────────────────────────────────────────────────────────

local _raw_log = _G.log or function(m) io.stdout:write(tostring(m or ""), "\n") end

local function log(msg)
  local s = tostring(msg or "")
  s = s:gsub("^%s+", "")
  _raw_log(s)
end

local function die(msg)
  log("ERROR: " .. tostring(msg))
  os.exit(1)
end

-- ── dtype ─────────────────────────────────────────────────────────────────────
-- Prend la chaîne dtype directement (ex: "fp16", "fp32").

local function apply_dtype(dtype_str)
  local dtype = dtype_str or os.getenv("MIMIR_DTYPE")
  if not dtype or dtype == "" then return true end
  if type(Mimir) ~= "table" then return true end
  local fn = (type(Mimir.model) == "table" and type(Mimir.model.dtype) == "function" and Mimir.model.dtype)
          or (type(Mimir.Model) == "table" and type(Mimir.Model.dtype) == "function" and Mimir.Model.dtype)
  if not fn then return true end
  local ok, err = fn(dtype)
  if not ok then die("dtype invalide: " .. tostring(err)) end
  return true
end

-- ── Args ──────────────────────────────────────────────────────────────────────

local Args = dofile("scripts/modules/args.lua")
local opts = Args.parse(arg) or {}

local function opt_bool(k, d)
  local v = opts[k]
  if v == nil then return d end
  if v == true or v == false then return v end
  v = tostring(v):lower()
  if v == "1" or v == "true" or v == "yes" or v == "on" then return true end
  if v == "0" or v == "false" or v == "no" or v == "off" then return false end
  return d
end

local function opt_num(k, d)
  local v = opts[k]
  if v == nil then return d end
  local n = tonumber(v)
  if n == nil then return d end
  return n
end

-- ── Helpers ────────────────────────────────────────────────────────────────────

local function shell_quote(s)
  s = tostring(s or "")
  return "'" .. s:gsub("'", "'\"'\"'") .. "'"
end

local function file_exists(path)
  local f = io.open(path, "rb")
  if f then f:close(); return true end
  return false
end

local function list_dir_names(path)
  local out = {}
  local p = io.popen("ls -1 " .. shell_quote(path) .. " 2>/dev/null", "r")
  if not p then return out end
  for line in p:lines() do
    if line and #line > 0 then out[#out + 1] = line end
  end
  p:close()
  return out
end

local function resolve_checkpoint_dir(path_in)
  local p = tostring(path_in or "")
  if p == "" then return p end

  if file_exists(p .. "/model/architecture.json") or file_exists(p .. "/architecture.json") then
    return p
  end

  if p:match("/model/?$") then
    p = p:gsub("/model/?$", "")
    if file_exists(p .. "/model/architecture.json") then return p end
  end

  local best_name, best_epoch, best_stop = nil, -1, false
  for _, name in ipairs(list_dir_names(p)) do
    local digits, suffix = name:match("^epoch_(%d+)(.*)$")
    if digits then
      local e = tonumber(digits) or -1
      local sfx = tostring(suffix or "")
      local is_stop = (sfx:lower():find("stop", 1, true) ~= nil) or (sfx:lower():find("final", 1, true) ~= nil)
      if e > best_epoch then
        best_epoch, best_name, best_stop = e, name, is_stop
      elseif e == best_epoch and best_name ~= nil then
        if (not best_stop and is_stop) or (best_stop == is_stop and name > best_name) then
          best_name, best_stop = name, is_stop
        end
      end
    end
  end

  if best_name then
    return p .. "/" .. best_name
  end
  return p
end

-- ── Diffusion schedule ─────────────────────────────────────────────────────────────────

local function clamp(x, a, b)
  if x < a then return a end
  if x > b then return b end
  return x
end

local function ddpm_alpha_bar_end(T, beta0, beta1)
  T = math.max(2, math.floor(tonumber(T) or 2))
  beta0 = clamp(tonumber(beta0) or 0, 0.0, 0.999)
  beta1 = clamp(tonumber(beta1) or 0, 0.0, 0.999)
  local log_ab = 0.0
  for i = 0, T - 1 do
    local frac = (T > 1) and (i / (T - 1)) or 0.0
    local beta = clamp(beta0 + (beta1 - beta0) * frac, 0.0, 0.999)
    log_ab = log_ab + math.log(math.max(1e-12, 1.0 - beta))
  end
  return math.exp(log_ab)
end

local function solve_beta_end(T, beta_start, target_ab_end)
  T = math.max(2, math.floor(tonumber(T) or 2))
  beta_start = clamp(tonumber(beta_start) or 1e-4, 0.0, 0.999)
  target_ab_end = clamp(tonumber(target_ab_end) or 4e-5, 1e-8, 0.999)

  local lo = beta_start
  local hi = 0.999
  for _ = 1, 40 do
    local mid = 0.5 * (lo + hi)
    local ab = ddpm_alpha_bar_end(T, beta_start, mid)
    if ab > target_ab_end then
      lo = mid
    else
      hi = mid
    end
  end
  return clamp(0.5 * (lo + hi), beta_start, 0.999)
end

-- ── Hardware / allocator ─────────────────────────────────────────────────────────────

local MEM_GB   = opt_num("mem-gb",   15)
local ALLOC_GB = opt_num("alloc-gb", MEM_GB)

if Mimir and Mimir.Allocator and Mimir.Allocator.configure then
  Mimir.Allocator.configure({ max_ram_gb = ALLOC_GB, enable_compression = opt_bool("compress", false) })
end
if Mimir and Mimir.MemoryGuard and Mimir.MemoryGuard.setLimit then
  pcall(Mimir.MemoryGuard.setLimit, MEM_GB)
end
if Mimir and Mimir.Model and Mimir.Model.set_hardware then
  pcall(Mimir.Model.set_hardware, opt_bool("hw", true))
end

-- ── Options ───────────────────────────────────────────────────────────────────

local ARCH         = Args.get_str(opts, "arch",     "ldm_unet")

local DATASET_DIR  = Args.get_str(opts, "dataset",  "dataset_2")
local DATASET_W    = Args.get_int(opts, "image-w",  Args.get_int(opts, "dataset-w", 512))
local DATASET_H    = Args.get_int(opts, "image-h",  Args.get_int(opts, "dataset-h", 512))
local DATASET_C    = Args.get_int(opts, "image-c",  3)
local DATASET_MIN_MOD = Args.get_int(opts, "dataset-min-modalities", 2)

local EPOCHS       = Args.get_int(opts, "epochs", 5)
local LR           = Args.get_num(opts, "lr",     1e-5)
local TRAIN_SEED   = Args.get_int(opts, "seed",   4242)
local INIT_SEED    = Args.get_int(opts, "init-seed", TRAIN_SEED)
local MAX_ITEMS    = Args.get_int(opts, "max-items", 0)

local TOKENIZER_PATH    = Args.get_str(opts, "tokenizer",  "checkpoint/base_tokenizer/tokenizer.json")
local DESIRED_MAX_VOCAB = Args.get_int(opts, "max-vocab",  32000)

local VAE_CKPT = resolve_checkpoint_dir(
  Args.get_str(opts, "vae-checkpoint", "checkpoint/vae_conv/epoch_0002_stop")
)

local OUT_DIR = Args.get_str(opts, "out-dir", "checkpoint/ldm_unet")

-- --resume 0 → false,  --resume / --resume 1 → true
local RESUME  = opt_bool("resume", false)
local DTYPE   = Args.get_str(opts, "dtype", nil)

local SAVE_PRETRAIN_DEBUG = opt_bool("save-pretrain-jsondebug", true)
local PRETRAIN_DEBUG_PATH = Args.get_str(opts, "pretrain-jsondebug", OUT_DIR .. "/pretrain_debug.json")

-- DDPM schedule
local TRAIN_TIMESTEPS     = Args.get_int(opts, "train-timesteps", Args.get_int(opts, "ddpm-steps", 1000))
local BETA_START          = opt_num("beta-start",         1e-4)
local BETA_END            = opt_num("beta-end",           0.02)
local TARGET_AB_END       = opt_num("ddpm-alpha-bar-end", opt_num("alpha-bar-end", 4e-6))
local AUTO_BETA_END       = opt_bool("auto-beta-end", opts["beta-end"] == nil)
if AUTO_BETA_END then
  BETA_END = solve_beta_end(TRAIN_TIMESTEPS, BETA_START, TARGET_AB_END)
end
local DDPM_STEPS_PER_IMAGE = Args.get_int(opts, "ddpm-steps-per-image", 1)

-- Loss / conditioning
local RECON_LOSS        = Args.get_str(opts, "recon-loss",        "mse")
local TIMESTEP_COND     = Args.get_str(opts, "timestep-cond",     "log_snr")
local LOSS_WEIGHTING    = Args.get_str(opts, "loss-weighting",    "none")
local MIN_SNR_GAMMA     = opt_num("min-snr-gamma",    5.0)
local OUTPUT_ACTIVATION = Args.get_str(opts, "output-activation", "linear")
local CFG_DROPOUT       = opt_num("cfg-dropout",      0.10)

-- KL  –  --kl-warmup est l’alias court de --kl-warmup-steps
local KL_BETA           = opt_num("kl-beta",           0.0)
local KL_WARMUP_STEPS   = Args.get_int(opts, "kl-warmup-steps",
                             Args.get_int(opts, "kl-warmup", 0))
local LOGVAR_CLIP_MIN   = opt_num("logvar-clip-min",  -10.0)
local LOGVAR_CLIP_MAX   = opt_num("logvar-clip-max",   10.0)

-- Auxiliary image-space loss
local IMG_LOSS_WEIGHT   = opt_num("img-loss-weight", 0.0)
local IMG_LOSS_EVERY    = Args.get_int(opts, "img-loss-every",
                             Args.get_int(opts, "img-loss-every-steps", 0))

-- Optimizer
local OPTIMIZER         = Args.get_str(opts, "optimizer",       "adamw")
local WARMUP_STEPS      = Args.get_int(opts, "lr-warmup-steps",
                             Args.get_int(opts, "warmup-steps", 400))
local WEIGHT_DECAY      = opt_num("weight-decay",    0.0)
local DECAY_STRATEGY    = Args.get_str(opts, "decay-strategy",  "linear")

-- Text encoder
local TEXT_CTX_LEN      = Args.get_int(opts, "text-ctx-len",  1300)
local TEXT_MEANPOOL     = opt_bool("text-meanpool", false)

-- U-Net
local UNET_DEPTH        = Args.get_int(opts, "unet-depth",             3)
local UNET_BLOCKS       = Args.get_int(opts, "unet-blocks-per-level",  1)
local UNET_BOTTLENECK   = Args.get_int(opts, "unet-bottleneck-blocks", 1)

-- Validation
local VALIDATE_EVERY         = Args.get_int(opts, "validate-every-steps",
                                  Args.get_int(opts, "validate-every", 100))
local VALIDATE_ITEMS         = math.max(1, Args.get_int(opts, "validate-items", 4))
local VALIDATE_HOLDOUT       = opt_bool("validate-holdout", true)
local VALIDATE_HOLDOUT_FRAC  = opt_num("validate-holdout-frac",  0.01)
local VALIDATE_HOLDOUT_ITEMS = Args.get_int(opts, "validate-holdout-items", 6)
local VALIDATE_SEED          = Args.get_int(opts, "validate-seed", 4242)
local VALIDATE_T             = Args.get_int(opts, "validate-t",   1000)

-- Misc
local LOG_EVERY             = Args.get_int(opts, "log-every", 1)
local AUTOSAVE_EVERY        = Args.get_int(opts, "autosave-every", 800)
local AUTOSAVE_EVERY_EPOCHS = Args.get_int(opts, "autosave-every-epochs",
                                 (AUTOSAVE_EVERY > 0) and 1 or 0)
local INIT_WEIGHTS          = opt_bool("init-weights", true)

-- Visualizer
local VIZ_DDPM            = opt_bool("viz-ddpm", true)
local VIZ_DDPM_EVERY      = Args.get_int(opts, "viz-ddpm-every", 200)
local VIZ_DDPM_STEPS      = Args.get_int(opts, "viz-ddpm-steps", 1)
local VIZ_TAPS_MAX_FRAMES = Args.get_int(opts, "viz-taps",
                               Args.get_int(opts, "viz-taps-max-frames", 64))
local VIZ_TAPS_MAX_SIDE   = Args.get_int(opts, "viz-taps-size",
                               Args.get_int(opts, "viz-taps-max-side", 720))
local VIZ_HIDE_ACT        = opt_bool("viz-hide-activation-blocks",   false)
local VIZ_HIDE_NORM       = opt_bool("viz-hide-normalisation-blocks", false)

-- ── Tokenizer ─────────────────────────────────────────────────────────────────────

log("init tokenizer")
Mimir.Tokenizer.load(TOKENIZER_PATH)
if Mimir.Tokenizer.set_max_vocab then
  pcall(Mimir.Tokenizer.set_max_vocab, DESIRED_MAX_VOCAB)
end
Mimir.Tokenizer.ensure_vocab_from_text("pony horse snow forest portrait")

local TOKENIZER_VOCAB = (Mimir.Tokenizer.get_max_vocab and Mimir.Tokenizer.get_max_vocab())
                     or Mimir.Tokenizer.vocab_size()
log(string.format("tokenizer: path=%s vocab_size=%d max_vocab=%d",
  TOKENIZER_PATH,
  tonumber(Mimir.Tokenizer.vocab_size() or 0) or 0,
  tonumber(TOKENIZER_VOCAB or 0) or 0))

-- ── Dataset (enregistrement uniquement – jamais itéré depuis Lua) ─────────────

log("load dataset")
local ok_ds, n_or_err = Mimir.Dataset.load(DATASET_DIR, DATASET_W, DATASET_H, DATASET_MIN_MOD)
if not ok_ds then die(n_or_err or "Dataset.load échoué") end
local DATASET_TOTAL = math.floor(tonumber(n_or_err) or 0)
if DATASET_TOTAL <= 0 then die("dataset vide") end
log(string.format("dataset: total=%d dir=%s w=%d h=%d c=%d",
  DATASET_TOTAL, DATASET_DIR, DATASET_W, DATASET_H, DATASET_C))

-- ── Modèle ────────────────────────────────────────────────────────────────────

log("create model: arch=" .. ARCH)
---@type any
local cfg = (Mimir.Architectures.default_config and Mimir.Architectures.default_config(ARCH)) or {}
if type(cfg) ~= "table" then cfg = {} end

-- Général
cfg.max_vocab              = TOKENIZER_VOCAB
cfg.seed                   = TRAIN_SEED

-- Images
cfg.image_w                = DATASET_W
cfg.image_h                = DATASET_H
cfg.image_c                = DATASET_C

-- VAE / checkpoint
cfg.vae_checkpoint         = VAE_CKPT
cfg.base_tokenizer_path    = TOKENIZER_PATH
cfg.checkpoint_dir         = OUT_DIR

-- DDPM
cfg.ddpm_steps             = TRAIN_TIMESTEPS
cfg.ddpm_beta_start        = BETA_START
cfg.ddpm_beta_end          = BETA_END
cfg.ddpm_steps_per_image   = DDPM_STEPS_PER_IMAGE

-- Loss / conditioning
cfg.recon_loss             = RECON_LOSS
cfg.timestep_cond          = TIMESTEP_COND
cfg.loss_weighting         = LOSS_WEIGHTING
cfg.min_snr_gamma          = MIN_SNR_GAMMA
cfg.output_activation      = OUTPUT_ACTIVATION
cfg.cfg_dropout_prob       = CFG_DROPOUT

-- KL
cfg.kl_beta                = KL_BETA
cfg.kl_warmup_steps        = KL_WARMUP_STEPS
cfg.logvar_clip_min        = LOGVAR_CLIP_MIN
cfg.logvar_clip_max        = LOGVAR_CLIP_MAX

-- Auxiliary
cfg.img_loss_weight        = IMG_LOSS_WEIGHT
cfg.img_loss_every_steps   = IMG_LOSS_EVERY

-- Text
cfg.text_ctx_len           = TEXT_CTX_LEN
cfg.text_bottleneck_meanpool = TEXT_MEANPOOL
cfg.text_clip_like         = opt_bool("text-clip-like", cfg.text_clip_like ~= false)
cfg.global_ctx_tokens      = Args.get_int(opts, "global-ctx-tokens",
                               tonumber(cfg.global_ctx_tokens or 0) or 0)
cfg.sdxl_time_cond         = opt_bool("sdxl-time-cond", cfg.sdxl_time_cond ~= false)

-- U-Net
cfg.unet_depth             = UNET_DEPTH
cfg.unet_blocks_per_level  = UNET_BLOCKS
cfg.unet_bottleneck_blocks = UNET_BOTTLENECK

-- Optimizer
cfg.optimizer              = OPTIMIZER
cfg.beta1                  = 0.9
cfg.beta2                  = 0.999
cfg.weight_decay           = WEIGHT_DECAY
cfg.decay_strategy         = DECAY_STRATEGY
cfg.warmup_steps           = WARMUP_STEPS

-- Logging / autosave
cfg.log_every              = LOG_EVERY
cfg.max_items              = MAX_ITEMS
cfg.autosave_every_epochs  = AUTOSAVE_EVERY_EPOCHS

-- Validation
cfg.validate_every_steps   = VALIDATE_EVERY
cfg.validate_items         = VALIDATE_ITEMS
cfg.validate_holdout       = VALIDATE_HOLDOUT
cfg.validate_holdout_frac  = VALIDATE_HOLDOUT_FRAC
cfg.validate_holdout_items = VALIDATE_HOLDOUT_ITEMS
cfg.validate_seed          = VALIDATE_SEED
cfg.validate_t             = VALIDATE_T

-- Visualizer
cfg.viz_taps_max_frames              = VIZ_TAPS_MAX_FRAMES
cfg.viz_taps_max_side                = VIZ_TAPS_MAX_SIDE
cfg.viz_hide_activation_blocks       = VIZ_HIDE_ACT
cfg.viz_hide_normalisation_blocks    = VIZ_HIDE_NORM
if VIZ_DDPM then
  cfg.viz_ddpm_every_steps = VIZ_DDPM_EVERY
  cfg.viz_ddpm_num_steps   = VIZ_DDPM_STEPS
end

-- Structured captions
cfg.caption_structured_enable       = opt_bool("caption-structured-enable",
                                        cfg.caption_structured_enable ~= false)
cfg.caption_structured_canonicalize = opt_bool("caption-structured-canonicalize",
                                        cfg.caption_structured_canonicalize ~= false)
cfg.caption_tags_dropout_prob       = opt_num("caption-tags-dropout",
                                        tonumber(cfg.caption_tags_dropout_prob or 0.0) or 0.0)
cfg.caption_contexte_dropout_prob   = opt_num("caption-contexte-dropout",
                                        tonumber(cfg.caption_contexte_dropout_prob or 0.0) or 0.0)
cfg.caption_mentalite_dropout_prob  = opt_num("caption-mentalite-dropout",
                                        tonumber(cfg.caption_mentalite_dropout_prob or 0.0) or 0.0)
cfg.caption_texte_dropout_prob      = opt_num("caption-texte-dropout",
                                        tonumber(cfg.caption_texte_dropout_prob or 0.0) or 0.0)
-- term-frequency boost (optionnel)
if cfg.caption_kv_enable ~= nil then
  cfg.caption_kv_enable = opt_bool("caption-kv-enable", cfg.caption_kv_enable ~= false)
end
if cfg.term_freq_boost_enable ~= nil then
  cfg.term_freq_boost_enable             = opt_bool("term-freq-boost-enable",
                                             cfg.term_freq_boost_enable ~= false)
  cfg.term_freq_boost_use_tokens         = opt_bool("term-freq-boost-use-tokens",
                                             cfg.term_freq_boost_use_tokens ~= false)
  cfg.term_freq_boost_use_keywords       = opt_bool("term-freq-boost-use-keywords",
                                             cfg.term_freq_boost_use_keywords ~= false)
  cfg.term_freq_boost_top_k              = Args.get_int(opts, "term-freq-boost-top-k",
                                             tonumber(cfg.term_freq_boost_top_k or 0) or 0)
  cfg.term_freq_boost_repeat             = Args.get_int(opts, "term-freq-boost-repeat",
                                             tonumber(cfg.term_freq_boost_repeat or 0) or 0)
  cfg.term_freq_boost_start_step         = Args.get_int(opts, "term-freq-boost-start-step",
                                             tonumber(cfg.term_freq_boost_start_step or 0) or 0)
  cfg.term_freq_boost_update_every_steps = Args.get_int(opts, "term-freq-boost-update-every-steps",
                                             tonumber(cfg.term_freq_boost_update_every_steps or 0) or 0)
end

-- Overrides CLI  (--override key=value …)
if Args.apply_overrides and opts and opts.override ~= nil then
  local ok_ov, err_ov = pcall(Args.apply_overrides, cfg, opts)
  if not ok_ov then die(err_ov) end
  log("overrides applied")
end

-- ── Log config ────────────────────────────────────────────────────────────────

log(string.format("arch=%s  vae=%s  out=%s", ARCH, VAE_CKPT, OUT_DIR))
log(string.format("ddpm: T=%d beta_start=%.4g beta_end=%.4g alpha_bar_end=%.4g steps/img=%d",
  cfg.ddpm_steps, cfg.ddpm_beta_start, cfg.ddpm_beta_end,
  ddpm_alpha_bar_end(cfg.ddpm_steps, cfg.ddpm_beta_start, cfg.ddpm_beta_end),
  cfg.ddpm_steps_per_image))
log(string.format("loss: recon=%s timestep_cond=%s weighting=%s min_snr=%.4g out=%s cfg_drop=%.2f",
  cfg.recon_loss, cfg.timestep_cond, cfg.loss_weighting,
  cfg.min_snr_gamma, cfg.output_activation, cfg.cfg_dropout_prob))
log(string.format("kl: beta=%.4g warmup=%d logvar=[%.4g,%.4g]",
  cfg.kl_beta, cfg.kl_warmup_steps, cfg.logvar_clip_min, cfg.logvar_clip_max))
log(string.format("opt: %s lr=%.4g warmup=%d decay=%s wd=%.4g",
  cfg.optimizer, LR, cfg.warmup_steps, cfg.decay_strategy, cfg.weight_decay))
log(string.format("unet: depth=%d blocks/level=%d bottleneck=%d",
  cfg.unet_depth, cfg.unet_blocks_per_level, cfg.unet_bottleneck_blocks))
log(string.format("text: ctx_len=%d meanpool=%s clip_like=%s",
  cfg.text_ctx_len, tostring(cfg.text_bottleneck_meanpool), tostring(cfg.text_clip_like)))
log(string.format("viz: taps(frames=%d,side=%d) ddpm(every=%d,steps=%d) hide(act=%s,norm=%s)",
  cfg.viz_taps_max_frames, cfg.viz_taps_max_side,
  VIZ_DDPM_EVERY, VIZ_DDPM_STEPS,
  tostring(VIZ_HIDE_ACT), tostring(VIZ_HIDE_NORM)))
if DTYPE and DTYPE ~= "" then log("dtype=" .. DTYPE) end
if RESUME then log("resume=true (cherche checkpoint dans " .. OUT_DIR .. ")") end

-- ── Create + allocate + init ──────────────────────────────────────────────────

local ok_create, err_create = Mimir.Model.create(ARCH, cfg)
if not ok_create then die(err_create or "Model.create échoué") end

apply_dtype(DTYPE)

log("allocate params")
Mimir.Model.allocate_params()

if RESUME and (file_exists(OUT_DIR .. "/model/architecture.json")
            or file_exists(OUT_DIR .. "/architecture.json")) then
  log("loading checkpoint: " .. OUT_DIR)
  local ok_load, err_load = Mimir.Serialization.load(OUT_DIR, "raw_folder")
  if not ok_load then
    log("WARN: resume échoué (" .. tostring(err_load) .. ") – init from scratch")
    if INIT_WEIGHTS then Mimir.Model.init_weights("xavier", INIT_SEED) end
  else
    log("checkpoint chargé")
  end
else
  if INIT_WEIGHTS then
    Mimir.Model.init_weights("xavier", INIT_SEED)
  end
end

-- ── Debug JSON (pre-train) ────────────────────────────────────────────────────

if SAVE_PRETRAIN_DEBUG then
  log("save pretrain debug_json: " .. PRETRAIN_DEBUG_PATH)
  local ok_dbg, err_dbg = Mimir.Serialization.save(PRETRAIN_DEBUG_PATH, "debug_json", {
    include_checksums       = true,
    include_git_info        = true,
    include_optimizer_state = true,
    max_values_per_tensor   = 16,
    save_tokenizer          = true,
    save_encoder            = true,
  })
  if not ok_dbg then
    log("WARN: pretrain debug_json save failed: " .. tostring(err_dbg))
  end
end

-- ── Entraînement ─────────────────────────────────────────────────────────────

log(string.format("start training: epochs=%d lr=%.4g max_items=%d", EPOCHS, LR, MAX_ITEMS))

local ok_train, steps_or_err = Mimir.Model.train(EPOCHS, LR)
if not ok_train then
  if steps_or_err == "STOP_REQUESTED" then
    log("stop demandé via Viz")
  else
    die(steps_or_err or "Model.train échoué")
  end
else
  log("training complete (steps=" .. tostring(steps_or_err) .. ")")
end

-- ── Sauvegarde finale ─────────────────────────────────────────────────────────

os.execute("mkdir -p " .. shell_quote(OUT_DIR))
local ok_save, err_save = Mimir.Serialization.save(OUT_DIR, "raw_folder", {
  save_optimizer          = true,
  save_tokenizer          = true,
  save_encoder            = true,
  include_checksums       = true,
  include_git_info        = true,
  include_gradients       = true,
  include_activations     = true,
  include_optimizer_state = true,
  include_weight_deltas   = true,
})
if not ok_save then die("save échoué: " .. tostring(err_save)) end
log("saved to " .. OUT_DIR)
