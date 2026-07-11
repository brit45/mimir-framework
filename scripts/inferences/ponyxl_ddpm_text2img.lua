-- PonyXL-DDPM text2img (Lua)
-- Usage:
--   ./bin/mimir --lua scripts/inferences/ponyxl_ddpm_text2img.lua -- \
--     --checkpoint checkpoint/PonyXL_SDXL/epoch_xxxx_or_model.safetensors \
--     --prompt "a pony in snowy forest" \
--     --out out.ppm \
--     --seed 12345 --steps 50 --guidance 5.0 \
--     --vae-checkpoint checkpoint/vae_conv_cpu_512_latent-64-64-32_base-64-2/epoch_0003_stop \
--     --tokenizer checkpoint/base_tokenizer/tokenizer.json

local Args = dofile("scripts/modules/args.lua")
local FS = dofile("scripts/modules/fs.lua")

local function logf(fmt, ...)
  if type(log) == "function" then
    log(string.format(fmt, ...))
  else
    print(string.format(fmt, ...))
  end
end

local function die(msg)
  if type(log) == "function" then log("[ponyxl_ddpm_text2img] ❌ " .. tostring(msg)) end
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

  local f, err = io.open(path, "wb")
  if not f then die("open(" .. tostring(path) .. ") a échoué: " .. tostring(err)) end
  ---@cast f file*
  f:write(string.format("P6\n%d %d\n255\n", w, h))

  local chunk = {}
  local n = 0
  local CHUNK = 8192
  for i = 1, #pixels do
    local v = pixels[i]
    v = math.floor(tonumber(v) or 0)
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

-- ---------------------------------------------------------------------------
-- Args
-- ---------------------------------------------------------------------------

local opts = Args.parse(arg)

local PROMPT = Args.get_str(opts, "prompt", "")
if PROMPT == "" then die("--prompt requis") end

local CKPT = Args.get_str(opts, "checkpoint", Args.get_str(opts, "ckpt", ""))
if CKPT == "" then die("--checkpoint requis") end

local OUT = Args.get_str(opts, "out", "out.ppm")
local OUT_DIR = dirname(OUT)
if OUT_DIR and OUT_DIR ~= "" then mkdir_p(OUT_DIR) end

local SEED = Args.get_int(opts, "seed", 12345)
local STEPS = Args.get_int(opts, "steps", 50)
local GUIDANCE = Args.get_num(opts, "guidance", Args.get_num(opts, "cfg", 1.0))
local MAX_SIDE = Args.get_int(opts, "max-side", 0)

local TOKENIZER_PATH = Args.get_str(opts, "tokenizer", "")
local DESIRED_MAX_VOCAB = Args.get_int(opts, "max-vocab", 0)

local DEFAULT_VAE = "checkpoint/vae_conv_cpu_512_latent-64-64-32_base-64-2/epoch_0003_stop"
local VAE_CKPT = Args.get_str(opts, "vae-checkpoint", DEFAULT_VAE)

-- ---------------------------------------------------------------------------
-- Model init
-- ---------------------------------------------------------------------------

if TOKENIZER_PATH ~= "" then
  logf("[ponyxl_ddpm_text2img] init tokenizer: %s", tostring(TOKENIZER_PATH))
  Mimir.Tokenizer.load(TOKENIZER_PATH)
  if DESIRED_MAX_VOCAB > 0 and Mimir.Tokenizer.set_max_vocab then
    pcall(Mimir.Tokenizer.set_max_vocab, DESIRED_MAX_VOCAB)
  end
end

logf("[ponyxl_ddpm_text2img] create model: ponyxl_ddpm")
local cfg = Mimir.Architectures.default_config("ponyxl_ddpm")

cfg.seed = SEED
cfg.vae_checkpoint = VAE_CKPT

if Args.apply_overrides and opts and opts.override ~= nil then
  local ok_ov, err_ov = pcall(Args.apply_overrides, cfg, opts)
  if not ok_ov then die(err_ov) end
end

local ok_create, err_create = Mimir.Model.create("ponyxl_ddpm", cfg)
if not ok_create then die(err_create or "Model.create a échoué") end

apply_dtype(cfg)

local ok_alloc, params_or_err = Mimir.Model.allocate_params()
if not ok_alloc then die(params_or_err or "allocate_params a échoué") end

logf("[ponyxl_ddpm_text2img] load checkpoint: %s", tostring(CKPT))
local ok_load, err_load = Mimir.Serialization.load(CKPT)
if not ok_load then die(err_load or "Serialization.load a échoué") end

-- ---------------------------------------------------------------------------
-- Generate
-- ---------------------------------------------------------------------------

logf("[ponyxl_ddpm_text2img] generate: steps=%d seed=%d guidance=%.4g", STEPS, SEED, GUIDANCE)
local pixels, w, h, c_or_err = Mimir.Model.ponyxl_ddpm_text2img(PROMPT, SEED, STEPS, GUIDANCE, MAX_SIDE)
if not pixels then die(c_or_err or "ponyxl_ddpm_text2img a échoué") end

local c = tonumber(c_or_err) or 3
if c ~= 3 then
  die("channels inattendu: " .. tostring(c) .. " (attendu 3)")
end

write_ppm_rgb_u8(OUT, pixels, w, h)
logf("[ponyxl_ddpm_text2img] ✅ écrit: %s (%dx%d)", tostring(OUT), tonumber(w) or 0, tonumber(h) or 0)
