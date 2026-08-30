#!/usr/bin/env lua
---@diagnostic disable: undefined-field, need-check-nil

local Args = dofile("scripts/modules/args.lua")
local Checkpoint = dofile("scripts/modules/checkpoint_resume.lua")

local function die(message)
  error("[lumen_text2img] " .. tostring(message), 0)
end

local function check(ok, err, operation)
  if not ok then die(operation .. ": " .. tostring(err or "échec")) end
end

local function write_ppm(path, pixels, width, height, channels)
  local file, err = io.open(path, "wb")
  if not file then return false, err end
  file:write(string.format("P6\n%d %d\n255\n", width, height))
  local chunk = {}
  for pixel = 0, width * height - 1 do
    local offset = pixel * channels
    local red = pixels[offset + 1] or 0
    local green = channels == 1 and red or (pixels[offset + 2] or 0)
    local blue = channels == 1 and red or (pixels[offset + 3] or 0)
    chunk[#chunk + 1] = string.char(math.max(0, math.min(255, red)))
    chunk[#chunk + 1] = string.char(math.max(0, math.min(255, green)))
    chunk[#chunk + 1] = string.char(math.max(0, math.min(255, blue)))
    if #chunk >= 8192 then
      file:write(table.concat(chunk))
      chunk = {}
    end
  end
  if #chunk > 0 then file:write(table.concat(chunk)) end
  file:close()
  return true
end

local opts = Args.parse(arg)
local prompt = Args.get_str(opts, "prompt", "a cinematic portrait, detailed lighting")
local seed = Args.get_int(opts, "seed", 1337)
local steps = Args.get_int(opts, "steps", 30)
local guidance = Args.get_num(opts, "guidance", 5.0)
local checkpoint = Args.get_str(opts, "checkpoint", "")
local tokenizer = Args.get_str(opts, "tokenizer", "checkpoint/base_tokenizer/tokenizer.json")
local output = Args.get_str(opts, "out", "lumen-output.ppm")

local cfg, cfg_err = Mimir.Architectures.default_config("lumen_diffusion")
if type(cfg) ~= "table" then die(cfg_err or "architecture lumen_diffusion indisponible") end

cfg.image_w = Args.get_int(opts, "image-w", cfg.image_w)
cfg.image_h = Args.get_int(opts, "image-h", cfg.image_h)
cfg.image_c = Args.get_int(opts, "image-c", cfg.image_c)
cfg.latent_w = Args.get_int(opts, "latent-w", cfg.latent_w)
cfg.latent_h = Args.get_int(opts, "latent-h", cfg.latent_h)
cfg.latent_c = Args.get_int(opts, "latent-c", cfg.latent_c)
cfg.vae_checkpoint = Args.get_str(opts, "vae-checkpoint", cfg.vae_checkpoint or "")
cfg.vae_base_channels = Args.get_int(opts, "vae-base-channels", cfg.vae_base_channels or 16)
cfg.vae_stochastic_latent = Args.get_bool(opts, "vae-stochastic-latent",
  cfg.vae_stochastic_latent ~= false)
cfg.vae_use_resnet = Args.get_bool(opts, "vae-resnet", cfg.vae_use_resnet ~= false)
cfg.vae_use_attn = Args.get_bool(opts, "vae-attn", cfg.vae_use_attn ~= false)
cfg.vae_use_skip_connections = Args.get_bool(opts, "vae-use-skip-connections",
  cfg.vae_use_skip_connections ~= false)
cfg.vae_use_encoder_prior = Args.get_bool(opts, "vae-encoder-prior",
  cfg.vae_use_encoder_prior ~= false)
cfg.vae_enc_norm = Args.get_str(opts, "vae-enc-norm", cfg.vae_enc_norm or "none")
cfg.vae_dec_norm = Args.get_str(opts, "vae-dec-norm", cfg.vae_dec_norm or "groupnorm")
cfg.vae_decoder_upsample = Args.get_str(opts, "vae-decoder-upsample",
  cfg.vae_decoder_upsample or "nearest_conv")
cfg.vae_enc_gn_groups = Args.get_int(opts, "vae-enc-gn-groups", cfg.vae_enc_gn_groups or 16)
cfg.vae_dec_gn_groups = Args.get_int(opts, "vae-dec-gn-groups", cfg.vae_dec_gn_groups or 32)
cfg.vae_attn_heads = Args.get_int(opts, "vae-attn-heads", cfg.vae_attn_heads or 4)
cfg.vae_attn_max_tokens = Args.get_int(opts, "vae-attn-max-tokens",
  cfg.vae_attn_max_tokens or 4096)
cfg.vae_resnet_max_tokens = Args.get_int(opts, "vae-resnet-max-tokens",
  cfg.vae_resnet_max_tokens or 4096)
cfg.vae_scale = Args.get_num(opts, "vae-scale", cfg.vae_scale or 0.0)
cfg.vae_shift = Args.get_num(opts, "vae-shift", cfg.vae_shift or 0.0)
cfg.patch_size = Args.get_int(opts, "patch-size", cfg.patch_size)
cfg.hidden_size = Args.get_int(opts, "hidden-size", cfg.hidden_size)
cfg.depth = Args.get_int(opts, "depth", cfg.depth)
cfg.mlp_ratio = Args.get_num(opts, "mlp-ratio", cfg.mlp_ratio)
cfg.seed = seed

if cfg.image_w <= 0 or cfg.image_h <= 0 then die("--image-w et --image-h doivent être positifs") end
if cfg.latent_w <= 0 or cfg.latent_h <= 0 or cfg.latent_c <= 0 then
  die("--latent-w, --latent-h et --latent-c doivent être positifs")
end
if cfg.vae_checkpoint == "" then
  die("--vae-checkpoint est requis (dossier epoch_* du VAEConv entraîné)")
end
local resolved_vae_checkpoint = Checkpoint.resolve_dir(cfg.vae_checkpoint)
if not resolved_vae_checkpoint then
  die("--vae-checkpoint ne contient aucun checkpoint VAEConv exploitable: " ..
    cfg.vae_checkpoint)
end
if resolved_vae_checkpoint ~= cfg.vae_checkpoint then
  print("[lumen_text2img] checkpoint VAE résolu: " .. resolved_vae_checkpoint)
  cfg.vae_checkpoint = resolved_vae_checkpoint
end
if cfg.patch_size < 1 or cfg.latent_w % cfg.patch_size ~= 0 or
    cfg.latent_h % cfg.patch_size ~= 0 then
  die("--patch-size doit diviser exactement --latent-w et --latent-h")
end
if cfg.image_c ~= 1 and cfg.image_c ~= 3 and cfg.image_c ~= 4 then
  die("--image-c doit valoir 1, 3 ou 4")
end

if tokenizer ~= "" then
  local ok_tokenizer, tokenizer_err = Mimir.Tokenizer.load(tokenizer)
  check(ok_tokenizer, tokenizer_err, "Tokenizer.load")
end
local ok_create, create_err = Mimir.Model.create("lumen_diffusion", cfg)
check(ok_create, create_err, "Model.create")
local ok_allocate, allocate_err = Mimir.Model.allocate_params()
check(ok_allocate, allocate_err, "Model.allocate_params")

if checkpoint ~= "" then
  local ok_load, load_err = Mimir.Serialization.load(checkpoint, "raw_folder", {
    load_tokenizer = true,
    load_encoder = false,
    load_optimizer = false,
    strict_mode = true,
  })
  check(ok_load, load_err, "Serialization.load")
else
  local ok_init, init_err = Mimir.Model.init_weights("xavier", seed)
  check(ok_init, init_err, "Model.init_weights")
  io.stderr:write("[lumen_text2img] avertissement: débruiteur non entraîné (aucun --checkpoint); " ..
    "utiliser OMP_NUM_THREADS=1 pour un smoke test bit-reproductible\n")
end

local pixels, width_or_error, height, channels =
  Mimir.Model.lumen_text2img(prompt, seed, steps, guidance)
if type(pixels) ~= "table" then die(width_or_error or "génération impossible") end

local ok_write, write_err = write_ppm(output, pixels, width_or_error, height, channels)
if not ok_write then die("écriture PPM: " .. tostring(write_err)) end
print(string.format("[lumen_text2img] image écrite: %s (%dx%dx%d, seed=%d)",
  output, width_or_error, height, channels, seed))