---@diagnostic disable: undefined-global, undefined-field, need-check-nil, param-type-mismatch, assign-type-mismatch

-- PonyXL-DDPM text2img with real HuggingFace VAE decoder.
-- Usage:
--   ./bin/mimir --lua scripts/inferences/ponyxl_ddpm_text2img_hf_vae.lua -- \
--     --diffusion-checkpoint checkpoint/PonyXL_SDXL/epoch_xxxx_or_model.safetensors \
--     --vae-checkpoint ../ponyxl.safetensors \
--     --prompt "a pony in snowy forest" \
--     --tokenizer checkpoint/base_tokenizer/tokenizer.json \
--     --vae-mapping-json tools/ponyxl_vae_decoder_mapping.json \
--     --out out_hf_vae.ppm \
--     --seed 12345 --steps 20 --guidance 5.0
--
-- Smoke test sans checkpoint diffusion compatible disponible:
--   ./bin/mimir --lua scripts/inferences/ponyxl_ddpm_text2img_hf_vae.lua -- \
--     --skip-diffusion-load \
--     --override latent_in_dim=4 \
--     --override latent_seq_len=4096 \
--     --override latent_h=64 \
--     --override latent_w=64 \
--     --vae-checkpoint ../ponyxl.safetensors \
--     --prompt "a pony in snowy forest" \
--     --out out_hf_vae.ppm

local Args = dofile("scripts/modules/args.lua")

local function logf(fmt, ...)
  local msg = string.format(fmt, ...)
  if type(log) == "function" then log(msg) else print(msg) end
end

local function die(msg)
  local text = "[ponyxl_ddpm_text2img_hf_vae] ❌ " .. tostring(msg)
  if type(log) == "function" then log(text) else print(text) end
  error(tostring(msg))
end

local function apply_dtype(cfg)
  local dtype = (type(cfg) == "table" and cfg.dtype) or os.getenv("MIMIR_DTYPE")
  if dtype == nil then return true end
  if type(Mimir) ~= "table" or type(Mimir.model) ~= "table" or type(Mimir.model.dtype) ~= "function" then
    return true
  end
  local ok, dt_or_err = Mimir.model.dtype(dtype)
  if not ok then die("dtype invalide: " .. tostring(dt_or_err)) end
  return true
end

local function shell_quote(s)
  s = tostring(s or "")
  return "'" .. s:gsub("'", "'\\''") .. "'"
end

local function mkdir_p(path)
  local dir = tostring(path or "")
  if dir == "" then return end
  os.execute("mkdir -p " .. shell_quote(dir) .. " >/dev/null 2>&1")
end

local function dirname(path)
  path = tostring(path or "")
  return path:match("^(.*)/[^/]*$")
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

  local f, err = io.open(path, "wb")
  if not f then die("open(" .. tostring(path) .. ") a échoué: " .. tostring(err)) end
  ---@cast f file*
  f:write(string.format("P6\n%d %d\n255\n", w, h))

  local chunk = {}
  local n = 0
  local CHUNK = 8192
  for i = 1, #pixels do
    local v = math.floor(tonumber(pixels[i]) or 0)
    if v < 0 then v = 0 end
    if v > 255 then v = 255 end
    n = n + 1
    chunk[n] = string.char(v)
    if n >= CHUNK then
      f:write(table.concat(chunk))
      n = 0
    end
  end
  if n > 0 then
    f:write(table.concat(chunk, "", 1, n))
  end

  f:close()
end

local function float_image_to_u8(image)
  if type(image) ~= "table" then die("image float invalide") end
  local out = {}
  for i = 1, #image do
    local x = tonumber(image[i]) or 0.0
    local v = math.floor(((x + 1.0) * 127.5) + 0.5)
    if v < 0 then v = 0 end
    if v > 255 then v = 255 end
    out[i] = v
  end
  return out
end

local opts = Args.parse(arg)

local PROMPT = Args.get_str(opts, "prompt", "")
if PROMPT == "" then die("--prompt requis") end

local DIFFUSION_CKPT = Args.get_str(opts, "diffusion-checkpoint", Args.get_str(opts, "checkpoint", Args.get_str(opts, "ckpt", "")))
local VAE_CKPT = Args.get_str(opts, "vae-checkpoint", Args.get_str(opts, "checkpoint", Args.get_str(opts, "ckpt", "../ponyxl.safetensors")))
local SKIP_DIFFUSION_LOAD = Args.get_bool(opts, "skip-diffusion-load", false)

if not SKIP_DIFFUSION_LOAD and DIFFUSION_CKPT == "" then die("--diffusion-checkpoint requis (ou --skip-diffusion-load)") end
if VAE_CKPT == "" then die("--vae-checkpoint requis") end

local TOKENIZER_PATH = Args.get_str(opts, "tokenizer", "")
local VAE_MAPPING_JSON = Args.get_str(opts, "vae-mapping-json", "tools/ponyxl_vae_decoder_mapping.json")
local OUT = Args.get_str(opts, "out", "out_hf_vae.ppm")
local OUT_DIR = dirname(OUT)
if OUT_DIR and OUT_DIR ~= "" then mkdir_p(OUT_DIR) end

local SEED = Args.get_int(opts, "seed", 12345)
local STEPS = Args.get_int(opts, "steps", 20)
local GUIDANCE = Args.get_num(opts, "guidance", Args.get_num(opts, "cfg", 5.0))
local DESIRED_MAX_VOCAB = Args.get_int(opts, "max-vocab", 0)

if TOKENIZER_PATH ~= "" then
  logf("[ponyxl_ddpm_text2img_hf_vae] init tokenizer: %s", tostring(TOKENIZER_PATH))
  Mimir.Tokenizer.load(TOKENIZER_PATH)
  if DESIRED_MAX_VOCAB > 0 and Mimir.Tokenizer.set_max_vocab then
    pcall(Mimir.Tokenizer.set_max_vocab, DESIRED_MAX_VOCAB)
  end
end

local pony_cfg, pony_cfg_err = Mimir.Architectures.default_config("ponyxl_ddpm")
if type(pony_cfg) ~= "table" then die("default_config(ponyxl_ddpm) a échoué: " .. tostring(pony_cfg_err)) end
pony_cfg.seed = SEED

if Args.apply_overrides and opts and opts.override ~= nil then
  local ok_ov, err_ov = pcall(Args.apply_overrides, pony_cfg, opts)
  if not ok_ov then die(err_ov) end
end

logf("[ponyxl_ddpm_text2img_hf_vae] create model: ponyxl_ddpm")
local ok_create, err_create = Mimir.Model.create("ponyxl_ddpm", pony_cfg)
if not ok_create then die(err_create or "Model.create(ponyxl_ddpm) a échoué") end
apply_dtype(pony_cfg)

local ok_alloc, alloc_err = Mimir.Model.allocate_params()
if not ok_alloc then die(alloc_err or "allocate_params ponyxl_ddpm a échoué") end

if SKIP_DIFFUSION_LOAD then
  logf("[ponyxl_ddpm_text2img_hf_vae] init random diffusion weights (skip load)")
  local ok_init, err_init = Mimir.Model.init_weights("xavier", SEED)
  if not ok_init then die(err_init or "init_weights ponyxl_ddpm a échoué") end
else
  logf("[ponyxl_ddpm_text2img_hf_vae] load diffusion checkpoint: %s", tostring(DIFFUSION_CKPT))
  local ok_load, err_load = Mimir.Serialization.load(DIFFUSION_CKPT)
  if not ok_load then die(err_load or "Serialization.load diffusion a échoué") end
end

logf("[ponyxl_ddpm_text2img_hf_vae] sample latent: steps=%d seed=%d guidance=%.4g", STEPS, SEED, GUIDANCE)
local latent, latent_w, latent_h, latent_c_or_err = Mimir.Model.ponyxl_ddpm_text2img_latent(PROMPT, SEED, STEPS, GUIDANCE)
if not latent then die(latent_c_or_err or "ponyxl_ddpm_text2img_latent a échoué") end

local latent_c = tonumber(latent_c_or_err) or 0
if latent_c <= 0 then die("latent_c invalide: " .. tostring(latent_c_or_err)) end
logf("[ponyxl_ddpm_text2img_hf_vae] latent=%d (%dx%dx%d)", #latent, tonumber(latent_w) or 0, tonumber(latent_h) or 0, latent_c)

local pony_image_w = math.floor(tonumber(pony_cfg.image_w) or 512)
local pony_image_h = math.floor(tonumber(pony_cfg.image_h) or 512)
local lw_num = math.floor(tonumber(latent_w) or 0)
local lh_num = math.floor(tonumber(latent_h) or 0)
if lw_num <= 0 or lh_num <= 0 then die("dimensions latentes invalides") end
if pony_image_w % lw_num ~= 0 or pony_image_h % lh_num ~= 0 or (pony_image_w / lw_num) ~= 8 or (pony_image_h / lh_num) ~= 8 then
  die(string.format("latent spatial scale incompatible with hf_vae_decoder: image=%dx%d latent=%dx%d (x8 requis)", pony_image_w, pony_image_h, lw_num, lh_num))
end
if latent_c ~= 4 then
  die(string.format("latent_c=%d incompatible with hf_vae_decoder; utilisez un modèle diffusion HF-latent ou des overrides latent_in_dim=4 latent_h=64 latent_w=64 latent_seq_len=4096", latent_c))
end

local vae_cfg, vae_cfg_err = Mimir.Architectures.default_config("hf_vae_decoder")
if type(vae_cfg) ~= "table" then die("default_config(hf_vae_decoder) a échoué: " .. tostring(vae_cfg_err)) end
vae_cfg.image_w = pony_image_w
vae_cfg.image_h = pony_image_h
vae_cfg.latent_w = lw_num
vae_cfg.latent_h = lh_num
vae_cfg.latent_c = latent_c
vae_cfg.dtype = os.getenv("MIMIR_DTYPE") or vae_cfg.dtype

logf("[ponyxl_ddpm_text2img_hf_vae] create model: hf_vae_decoder")
local ok_vae_create, err_vae_create = Mimir.Model.create("hf_vae_decoder", vae_cfg)
if not ok_vae_create then die(err_vae_create or "Model.create(hf_vae_decoder) a échoué") end
apply_dtype(vae_cfg)

local ok_vae_alloc, err_vae_alloc = Mimir.Model.allocate_params()
if not ok_vae_alloc then die(err_vae_alloc or "allocate_params hf_vae_decoder a échoué") end

logf("[ponyxl_ddpm_text2img_hf_vae] load VAE checkpoint: %s", tostring(VAE_CKPT))
logf("[ponyxl_ddpm_text2img_hf_vae] load VAE mapping: %s", tostring(VAE_MAPPING_JSON))
local ok_vae_load, err_vae_load = Mimir.Serialization.load(VAE_CKPT, "safetensors", {
  load_tokenizer = false,
  load_encoder = false,
  load_optimizer = false,
  strict_mode = true,
  mapping_json = VAE_MAPPING_JSON,
})
if not ok_vae_load then die(err_vae_load or "Serialization.load hf_vae_decoder a échoué") end

local image, fwd_err = Mimir.Model.forward(latent, false)
if type(image) ~= "table" then die(fwd_err or "forward hf_vae_decoder a échoué") end

local pixels = float_image_to_u8(image)
write_ppm_rgb_u8(OUT, pixels, tonumber(vae_cfg.image_w) or 0, tonumber(vae_cfg.image_h) or 0)
logf("[ponyxl_ddpm_text2img_hf_vae] ✅ écrit: %s (%dx%d)", tostring(OUT), tonumber(vae_cfg.image_w) or 0, tonumber(vae_cfg.image_h) or 0)