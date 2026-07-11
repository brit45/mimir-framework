#!/usr/bin/env lua
---@diagnostic disable: undefined-field, need-check-nil, inject-field

local Args = dofile("scripts/modules/args.lua")
local FS = dofile("scripts/modules/fs.lua")
local opts = Args.parse(arg) or {}

local function log(msg)
  local f = _G.log or function(m) io.stdout:write(tostring(m or ""), "\n") end
  f((tostring(msg or ""):gsub("^%s+", "")))
end

local function die(msg)
  log("ERROR: " .. tostring(msg))
  os.exit(1)
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

local function apply_dtype(cfg)
  local dtype = (type(cfg) == "table" and cfg.dtype) or os.getenv("MIMIR_DTYPE")
  if dtype == nil then return true end
  if type(Mimir) ~= "table" then return true end

  local dtype_fn = nil
  if type(Mimir.model) == "table" and type(Mimir.model.dtype) == "function" then
    dtype_fn = Mimir.model.dtype
  elseif type(Mimir.Model) == "table" and type(Mimir.Model.dtype) == "function" then
    dtype_fn = Mimir.Model.dtype
  end
  if dtype_fn == nil then return true end

  local ok, dt_or_err = dtype_fn(dtype)
  if not ok then die("dtype invalide: " .. tostring(dt_or_err)) end
  return true
end

local function mkdir_p(path)
  FS.mkdir_p(path)
end

local function dirname(path)
  return FS.dirname(path)
end

local function write_ppm_rgb_u8(path, pixels, w, h)
  if type(pixels) ~= "table" then die("pixels invalide") end
  w = math.floor(tonumber(w) or 0)
  h = math.floor(tonumber(h) or 0)
  if w <= 0 or h <= 0 then die("w/h invalides") end
  local expected = w * h * 3
  if #pixels ~= expected then
    die("taille pixels invalide: got=" .. tostring(#pixels) .. " expected=" .. tostring(expected))
  end

  local parent = dirname(path)
  if parent and parent ~= "" then mkdir_p(parent) end

  local f, err = io.open(path, "wb")
  if not f then die("open(" .. tostring(path) .. ") a échoué: " .. tostring(err)) end
  ---@cast f file*
  f:write(string.format("P6\n%d %d\n255\n", w, h))

  local chunk = {}
  local n = 0
  local CHUNK = 8192
  for i = 1, expected do
    n = n + 1
    chunk[n] = string.char(math.max(0, math.min(255, math.floor(tonumber(pixels[i]) or 0))))
    if n >= CHUNK then
      f:write(table.concat(chunk, "", 1, n))
      n = 0
    end
  end
  if n > 0 then f:write(table.concat(chunk, "", 1, n)) end
  f:close()
end

local function usage()
  log("Usage:")
  log("  ./bin/mimir --lua scripts/training/ponyxl_ddpm_direct_train.lua -- \\")
  log("    --image path/to/image.png \\")
  log("    --prompt \"ton prompt\" \\")
  log("    --vae-checkpoint checkpoint/vae_conv_.../epoch_XXXX \\")
  log("    [--steps 200] [--lr 1e-5] [--dataset-w 512] [--dataset-h 512] \\")
  log("    [--tokenizer checkpoint/base_tokenizer/tokenizer.json] \\")
  log("    [--checkpoint checkpoint/ponyxl_ddpm_prev] [--out checkpoints/ponyxl_ddpm_direct] \\")
  log("    [--sample-out out.ppm] [--sample-steps 30]")
end

if opt_bool("help", false) or opt_bool("h", false) then
  usage()
  return
end

local IMAGE_PATH = Args.get_str(opts, "image", "")
local PROMPT = Args.get_str(opts, "prompt", "")
local VAE_CKPT = Args.get_str(opts, "vae-checkpoint", "")
local TOKENIZER_PATH = Args.get_str(opts, "tokenizer", "checkpoint/base_tokenizer/tokenizer.json")
local CKPT = Args.get_str(opts, "checkpoint", "")
local OUT_DIR = Args.get_str(opts, "out", "checkpoints/ponyxl_ddpm_direct")
local SAMPLE_OUT = Args.get_str(opts, "sample-out", "")

local STEPS = Args.get_int(opts, "steps", 200)
local LR = Args.get_num(opts, "lr", 1e-5)
local TRAIN_SEED = Args.get_int(opts, "seed", 4242)
local INIT_SEED = Args.get_int(opts, "init-seed", TRAIN_SEED)
local IMAGE_W = Args.get_int(opts, "dataset-w", 512)
local IMAGE_H = Args.get_int(opts, "dataset-h", 512)
local TRAIN_TIMESTEPS = Args.get_int(opts, "train-timesteps", 1000)
local DDPM_STEPS_PER_IMAGE = Args.get_int(opts, "ddpm-steps-per-image", 1)
local TEXT_CTX_LEN = Args.get_int(opts, "text-ctx-len", 1300)
local TOKENIZER_MAX_VOCAB = Args.get_int(opts, "tokenizer-max-vocab", 32000)
local OPTIMIZER = Args.get_str(opts, "optimizer", "adamw")
local LOG_EVERY = Args.get_int(opts, "log-every", 10)
local SAVE_EVERY = Args.get_int(opts, "save-every", 0)
local SAMPLE_STEPS = Args.get_int(opts, "sample-steps", 30)
local GUIDANCE = Args.get_num(opts, "guidance", 1.0)
local INIT_WEIGHTS = opt_bool("init-weights", true)
local CALIBRATE_VAE = opt_bool("calibrate-vae", true)

if IMAGE_PATH == "" then die("--image requis") end
if PROMPT == "" then die("--prompt requis") end
if VAE_CKPT == "" then die("--vae-checkpoint requis") end
if STEPS <= 0 then die("--steps doit être > 0") end
if IMAGE_W <= 0 or IMAGE_H <= 0 then die("--dataset-w/--dataset-h doivent être > 0") end

if TOKENIZER_PATH ~= "" then
  log("[ponyxl_ddpm_direct_train] init tokenizer: " .. tostring(TOKENIZER_PATH))
  Mimir.Tokenizer.load(TOKENIZER_PATH)
  if Mimir.Tokenizer.set_max_vocab then
    pcall(Mimir.Tokenizer.set_max_vocab, TOKENIZER_MAX_VOCAB)
  end
end

local img, err_img = Mimir.IO.read_image_rgb_u8(IMAGE_PATH, IMAGE_W, IMAGE_H)
if img == nil then die(err_img or ("Impossible de charger l'image: " .. tostring(IMAGE_PATH))) end
if type(img) ~= "table" or type(img.image) ~= "table" then die("lecture image invalide") end

local cfg = Mimir.Architectures.default_config("ponyxl_ddpm")
cfg.seed = TRAIN_SEED
cfg.vae_checkpoint = VAE_CKPT
cfg.image_w = IMAGE_W
cfg.image_h = IMAGE_H
cfg.image_c = 3
cfg.text_ctx_len = TEXT_CTX_LEN
cfg.max_vocab = TOKENIZER_MAX_VOCAB
cfg.ddpm_steps = TRAIN_TIMESTEPS
cfg.ddpm_steps_per_image = DDPM_STEPS_PER_IMAGE
cfg.recon_loss = Args.get_str(opts, "recon-loss", cfg.recon_loss or "mse")
cfg.optimizer = OPTIMIZER
cfg.beta1 = Args.get_num(opts, "beta1", 0.9)
cfg.beta2 = Args.get_num(opts, "beta2", 0.999)
cfg.weight_decay = Args.get_num(opts, "weight-decay", 0.0)
cfg.warmup_steps = Args.get_int(opts, "warmup-steps", 0)
cfg.cfg_dropout_prob = Args.get_num(opts, "cfg-dropout", tonumber(cfg.cfg_dropout_prob or 0.1) or 0.1)
cfg.timestep_cond = Args.get_str(opts, "timestep-cond", cfg.timestep_cond or "log_snr")
cfg.loss_weighting = Args.get_str(opts, "loss-weighting", cfg.loss_weighting or "none")
cfg.min_snr_gamma = Args.get_num(opts, "min-snr-gamma", tonumber(cfg.min_snr_gamma or 5.0) or 5.0)
cfg.output_activation = Args.get_str(opts, "output-activation", cfg.output_activation or "linear")
cfg.global_ctx_tokens = Args.get_int(opts, "global-ctx-tokens", tonumber(cfg.global_ctx_tokens or 0) or 0)
cfg.text_clip_like = opt_bool("text-clip-like", cfg.text_clip_like ~= false)
cfg.text_bottleneck_meanpool = opt_bool("text-meanpool", cfg.text_bottleneck_meanpool == true)
cfg.sdxl_time_cond = opt_bool("sdxl-time-cond", cfg.sdxl_time_cond ~= false)
cfg.unet_depth = Args.get_int(opts, "unet-depth", tonumber(cfg.unet_depth or 3) or 3)
cfg.unet_blocks_per_level = Args.get_int(opts, "unet-blocks-per-level", tonumber(cfg.unet_blocks_per_level or 1) or 1)
cfg.unet_bottleneck_blocks = Args.get_int(opts, "unet-bottleneck-blocks", tonumber(cfg.unet_bottleneck_blocks or 1) or 1)
cfg.img_loss_weight = Args.get_num(opts, "img-loss-weight", tonumber(cfg.img_loss_weight or 0.0) or 0.0)
cfg.img_loss_every_steps = Args.get_int(opts, "img-loss-every-steps", tonumber(cfg.img_loss_every_steps or 0) or 0)
cfg.kl_beta = Args.get_num(opts, "kl-beta", tonumber(cfg.kl_beta or 0.0) or 0.0)
cfg.kl_warmup_steps = Args.get_int(opts, "kl-warmup-steps", tonumber(cfg.kl_warmup_steps or 0) or 0)
cfg.logvar_clip_min = Args.get_num(opts, "logvar-clip-min", tonumber(cfg.logvar_clip_min or -10.0) or -10.0)
cfg.logvar_clip_max = Args.get_num(opts, "logvar-clip-max", tonumber(cfg.logvar_clip_max or 10.0) or 10.0)

if Args.apply_overrides and opts and opts.override ~= nil then
  local ok_ov, err_ov = pcall(Args.apply_overrides, cfg, opts)
  if not ok_ov then die(err_ov) end
end

log("[ponyxl_ddpm_direct_train] create model: ponyxl_ddpm")
local ok_create, err_create = Mimir.Model.create("ponyxl_ddpm", cfg)
if not ok_create then die(err_create or "Model.create a échoué") end

apply_dtype(cfg)

local ok_alloc, err_alloc = Mimir.Model.allocate_params()
if not ok_alloc then die(err_alloc or "allocate_params a échoué") end
if INIT_WEIGHTS and Mimir.Model.init_weights then
  local ok_init, err_init = Mimir.Model.init_weights("xavier", INIT_SEED)
  if ok_init == false then die(err_init or "init_weights a échoué") end
end

if CKPT ~= "" then
  log("[ponyxl_ddpm_direct_train] load checkpoint: " .. tostring(CKPT))
  local ok_load, err_load = Mimir.Serialization.load(CKPT)
  if not ok_load then die(err_load or "Serialization.load a échoué") end
end

if CALIBRATE_VAE then
  local m, err_m = Mimir.Model.ponyxl_ddpm_vae_mu_moments(img.image, img.width, img.height)
  if not m then
    die(err_m or "calibrage vae_scale a échoué")
  end
  local n = tonumber(m.n) or 0
  if n > 0 then
    local sum = tonumber(m.sum) or 0.0
    local sumsq = tonumber(m.sumsq) or 0.0
    local mean = sum / n
    local var = (sumsq / n) - (mean * mean)
    if var < 1e-12 then var = 1e-12 end
    local std = math.sqrt(var)
    local scale = 1.0 / std
    local ok_set, err_set = Mimir.Model.ponyxl_ddpm_set_vae_scale(scale)
    if not ok_set then
      die(err_set or "ponyxl_ddpm_set_vae_scale a échoué")
    end
    local cur = Mimir.Model.ponyxl_ddpm_get_vae_scale()
    log(string.format(
      "[ponyxl_ddpm_direct_train] calibrate vae_scale: n=%d mean=%.6g std=%.6g -> scale=%.6g (cur=%.6g)",
      n,
      mean,
      std,
      scale,
      tonumber(cur) or scale
    ))
  end
end

mkdir_p(OUT_DIR)

log(string.format("[ponyxl_ddpm_direct_train] image=%s resized=%dx%d steps=%d lr=%.6g", tostring(IMAGE_PATH), IMAGE_W, IMAGE_H, STEPS, LR))
log(string.format("[ponyxl_ddpm_direct_train] prompt=%s", tostring(PROMPT)))

local avg_loss = 0.0
---@type fun(...: any): any, any
local pony_train_step = Mimir.Model.ponyxl_ddpm_train_step
for step = 1, STEPS do
  local meta = {
    epoch = 1,
    total_epochs = 1,
    batch = step,
    total_batches = STEPS,
    avg_loss = avg_loss,
    dataset_i = step,
  }

  local st, err_step = pony_train_step(
    PROMPT,
    img.image,
    img.width,
    img.height,
    LR,
    OPTIMIZER,
    meta
  )
  if not st then die(err_step or ("ponyxl_ddpm_train_step a échoué à l'étape " .. tostring(step))) end

  local cur_loss = tonumber(st.loss) or 0.0
  if step == 1 then avg_loss = cur_loss else avg_loss = (avg_loss * (step - 1) + cur_loss) / step end

  if step == 1 or step == STEPS or (LOG_EVERY > 0 and (step % LOG_EVERY) == 0) then
    log(string.format(
      "[ponyxl_ddpm_direct_train] step=%d/%d loss=%.6f t=%.4f grad_norm=%.6f grad_max=%.6f kl=%.6f",
      step,
      STEPS,
      cur_loss,
      tonumber(st.timestep) or 0.0,
      tonumber(st.grad_norm) or 0.0,
      tonumber(st.grad_max_abs) or 0.0,
      tonumber(st.kl_divergence) or 0.0
    ))
  end

  if SAVE_EVERY > 0 and (step % SAVE_EVERY) == 0 then
    local ckpt_dir = string.format("%s/step_%06d", OUT_DIR, step)
    mkdir_p(ckpt_dir)
    local ok_save0, err_save0 = Mimir.Serialization.save(ckpt_dir, "raw_folder", {
      include_git_info = true,
      include_checksums = false,
    })
    if not ok_save0 then die(err_save0 or ("save step " .. tostring(step) .. " a échoué")) end
  end
end

local ok_save, err_save = Mimir.Serialization.save(OUT_DIR, "raw_folder", {
  include_git_info = true,
  include_checksums = true,
})
if not ok_save then die(err_save or "save final a échoué") end
log("[ponyxl_ddpm_direct_train] checkpoint écrit: " .. tostring(OUT_DIR))

if SAMPLE_OUT ~= "" then
  local pixels, w, h, c_or_err = Mimir.Model.ponyxl_ddpm_text2img(PROMPT, TRAIN_SEED, SAMPLE_STEPS, GUIDANCE, 0)
  if not pixels then die(c_or_err or "ponyxl_ddpm_text2img a échoué après entraînement") end
  local c = tonumber(c_or_err) or 3
  if c ~= 3 then die("channels inattendu: " .. tostring(c)) end
  write_ppm_rgb_u8(SAMPLE_OUT, pixels, w, h)
  log("[ponyxl_ddpm_direct_train] sample écrit: " .. tostring(SAMPLE_OUT))
end

log("[ponyxl_ddpm_direct_train] terminé")