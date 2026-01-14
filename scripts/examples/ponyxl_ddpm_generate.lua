-- Génération PonyXL (prompt->image) pour le nouveau modèle "ponyxl_ddpm"
-- (NOTE: malgré le nom, ce n'est plus un DDPM; c'est un autoencoder conditionné par le prompt)
--
-- Prérequis:
--   - entraîner via scripts/training/train_ponyxl_ddpm.lua
--   - disposer d'un checkpoint raw_folder (ex: checkpoint/PonyXL_DDPM/epoch_0002)
--
-- Usage:
--   ./bin/mimir --lua scripts/examples/ponyxl_ddpm_generate.lua

local PROMPT = os.getenv("MIMIR_PROMPT") or "dog"
local CHECKPOINT_BASE = os.getenv("MIMIR_PONYXL_CKPT") or "checkpoint/PonyXL_DDPM"
local INIT_MODE = os.getenv("MIMIR_INIT") or "noise" -- "zeros" | "noise"
local INIT_IMAGE = os.getenv("MIMIR_INIT_IMAGE") -- chemin vers une image d'init (PPM P6 recommandé)
local INIT_IMAGE_ALPHA = tonumber(os.getenv("MIMIR_INIT_IMAGE_ALPHA") or "1.0") or 1.0
local INIT_IMAGE_RESIZE = os.getenv("MIMIR_INIT_IMAGE_RESIZE") or "bilinear" -- "bilinear" | "nearest"
local SEED = tonumber(os.getenv("MIMIR_SEED") or "1331") or 1331
local STEPS = tonumber(os.getenv("MIMIR_STEPS") or "10") or 45
local ALPHA = tonumber(os.getenv("MIMIR_ALPHA") or "0.050") or 1.0  -- 1.0 = remplace entièrement par la prédiction
local NOISE_STD = tonumber(os.getenv("MIMIR_NOISE_STD") or "0.1") or 0.0
local NOISE_DECAY = tonumber(os.getenv("MIMIR_NOISE_DECAY") or "0.5") or 0.5
local NOISE_KIND = os.getenv("MIMIR_NOISE_KIND") or "peltier" -- "gaussian" | "peltier" | "fbm"
local PELTIER_LAMBDA = tonumber(os.getenv("MIMIR_PELTIER_LAMBDA") or "4.0") or 2.0
local NOISE_OCTAVES = tonumber(os.getenv("MIMIR_NOISE_OCTAVES") or "4") or 4
local NOISE_LACUNARITY = tonumber(os.getenv("MIMIR_NOISE_LACUNARITY") or "2.0") or 2.0
local NOISE_GAIN = tonumber(os.getenv("MIMIR_NOISE_GAIN") or "0.5") or 0.5

-- Pré-traitement avant réinjection (améliore souvent la netteté):
-- - lissage léger (anti-bruit)
-- - unsharp mask (accentuation des détails)
-- - contraste (expansion douce)
-- Désactivation: MIMIR_PREPROCESS=0|off|false
local PREPROCESS = os.getenv("MIMIR_PREPROCESS") or "0"
local PRE_SMOOTH = tonumber(os.getenv("MIMIR_PRE_SMOOTH") or "0.08") or 0.08
local PRE_SHARPEN = tonumber(os.getenv("MIMIR_PRE_SHARPEN") or "0.60") or 0.60
local PRE_CONTRAST = tonumber(os.getenv("MIMIR_PRE_CONTRAST") or "1.05") or 1.05
local PRE_CLAMP = os.getenv("MIMIR_PRE_CLAMP") or "1"

-- Mode "prompt only": ne réinjecte pas l'image prédite dans l'entrée du modèle.
-- Le modèle reçoit toujours [text_vec, image_base] où image_base est fixe (zeros/noise/init_image).
-- Désactivation: MIMIR_PROMPT_ONLY=0|off|false
local PROMPT_ONLY = os.getenv("MIMIR_PROMPT_ONLY") or "1"
local OUT_W = tonumber(os.getenv("MIMIR_OUT_W") or "125") or 64
local OUT_H = tonumber(os.getenv("MIMIR_OUT_H") or "125") or 64
local UPSCALE = os.getenv("MIMIR_UPSCALE") or "bilinear" -- "bilinear" | "nearest" | "none"

-- Pyramide multi-scale (64→128→256→…): "auto" ou liste "64,128,256,512,1024".
-- Attention: ça améliore souvent la netteté via réinjection, mais ne remplace pas un vrai modèle de super-resolution.
local PYRAMID = os.getenv("MIMIR_PYRAMID") or "off" -- "off" | "auto" | liste
local PYRAMID_STEPS = tonumber(os.getenv("MIMIR_PYRAMID_STEPS") or "2") or 2
local PYRAMID_ALPHA = tonumber(os.getenv("MIMIR_PYRAMID_ALPHA") or "1.0") or 1.0

log("╔═══════════════════════════════════════════════════════════════╗")
log("║   PonyXL Generate (prompt->image, autoencoder RGB)            ║")
log("╚═══════════════════════════════════════════════════════════════╝")

if Mimir and Mimir.MemoryGuard and Mimir.MemoryGuard.setLimit then
  pcall(Mimir.MemoryGuard.setLimit, 10)
end

if Mimir and Mimir.Allocator and Mimir.Allocator.configure then
  pcall(Mimir.Allocator.configure, {
    max_ram_gb = 10.0,
    enable_compression = true,
    swap_strategy = "lru"
  })
end

if Mimir and Mimir.Model and Mimir.Model.set_hardware then
  Mimir.Model.set_hardware("opencl")
end

local function file_exists(path)
  local f = io.open(path, "rb")
  if f then f:close(); return true end
  return false
end

local function find_latest_epoch_dir(base)
  local p = io.popen("ls -1d " .. base .. "/epoch_* 2>/dev/null | sort | tail -n 1")
  if not p then return nil end
  local line = p:read("*l")
  p:close()
  if line and #line > 0 then return line end
  return nil
end

local ckpt_dir = find_latest_epoch_dir(CHECKPOINT_BASE) or CHECKPOINT_BASE
local arch_path = ckpt_dir .. "/model/architecture.json"
if not file_exists(arch_path) then
  error("Checkpoint introuvable: " .. tostring(arch_path) .. " (entraîne d'abord ou corrige CHECKPOINT_BASE)")
end

local arch = read_json(arch_path)
if not arch or type(arch) ~= "table" then
  error("architecture.json invalide: " .. tostring(arch_path))
end

local cfg = arch.model_config or {}
cfg.seed = cfg.seed or SEED
cfg.image_w = cfg.image_w or cfg.image_width or 64
cfg.image_h = cfg.image_h or cfg.image_height or 64
cfg.image_c = cfg.image_c or 3
cfg.d_model = cfg.d_model or 256
cfg.seq_len = cfg.seq_len or 128
cfg.max_vocab = cfg.max_vocab or 8192
cfg.hidden_dim = cfg.hidden_dim or 2048
cfg.latent_dim = cfg.latent_dim or 512

log("📦 Checkpoint: " .. ckpt_dir)
log(string.format("Config: d_model=%d seq_len=%d max_vocab=%d hidden_dim=%d latent_dim=%d img=%dx%dx%d",
  cfg.d_model, cfg.seq_len, cfg.max_vocab, cfg.hidden_dim, cfg.latent_dim, cfg.image_w, cfg.image_h, cfg.image_c
))
log(string.format("Génération: steps=%d alpha=%.3f noise_std=%.4f noise_decay=%.3f init=%s seed=%d",
  STEPS, ALPHA, NOISE_STD, NOISE_DECAY, tostring(INIT_MODE), cfg.seed
))
log(string.format("Noise: kind=%s peltier_lambda=%.3f octaves=%d", tostring(NOISE_KIND), PELTIER_LAMBDA, math.floor(NOISE_OCTAVES)))
log(string.format("Preprocess: enabled=%s smooth=%.3f sharpen=%.3f contrast=%.3f",
  tostring(PREPROCESS), PRE_SMOOTH, PRE_SHARPEN, PRE_CONTRAST
))
log(string.format("Input: prompt_only=%s", tostring(PROMPT_ONLY)))
log(string.format("Sortie: %dx%d (upscale=%s)", OUT_W, OUT_H, tostring(UPSCALE)))

local ok_create, err_create = Mimir.Model.create("ponyxl_ddpm", cfg)
if not ok_create then error("Model.create failed: " .. tostring(err_create)) end

local ok_build, err_build = Mimir.Model.build()
if not ok_build then error("Model.build failed: " .. tostring(err_build)) end

local ok_alloc, err_alloc = Mimir.Model.allocate_params()
if not ok_alloc then error("allocate_params failed: " .. tostring(err_alloc)) end

local ok_load, err_load = Mimir.Serialization.load(ckpt_dir, "raw_folder")
if not ok_load then error("Serialization.load failed: " .. tostring(err_load)) end

if not (Mimir and Mimir.Model and Mimir.Model.encode_prompt) then
  error("Mimir.Model.encode_prompt indisponible (recompile bin/mimir)")
end

-- IMPORTANT:
-- `encode_prompt` (côté C++) utilise `tokenize()` (non-mutant). Si le tokenizer du checkpoint
-- a un vocab trop petit, beaucoup de prompts deviennent identiques (OOV -> <UNK>) => le prompt
-- ne change rien.
-- Pour que le prompt influence réellement l'image, on "enseigne" au tokenizer les tokens du prompt
-- AVANT d'appeler encode_prompt. Ça ne touche pas à la seed/steps/init_image.
-- if PROMPT and #PROMPT > 0 and Mimir and Mimir.Tokenizer and Mimir.Tokenizer.tokenize_ensure then
--   pcall(Mimir.Tokenizer.tokenize_ensure, PROMPT)
-- end

local text_vec, err_text = Mimir.Model.encode_prompt(PROMPT)
if not text_vec then error("encode_prompt failed: " .. tostring(err_text)) end
if #text_vec ~= cfg.d_model then
  error("encode_prompt: taille inattendue: got " .. tostring(#text_vec) .. " expected " .. tostring(cfg.d_model))
end

math.randomseed(math.floor(SEED))
local function randn()
  local u1 = math.max(1e-12, math.random())
  local u2 = math.random()
  return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
end

-- Bruit multi-échelle spatialement corrélé (fBm) pour init et injection.
-- Objectif: éviter un bruit blanc trop "pixel" qui donne souvent des sorties bouillies
-- dans un autoencodeur conditionné par texte + réinjection.
local function compute_mean_var(t)
  local n = #t
  if n <= 0 then return 0.0, 0.0 end
  local sum = 0.0
  for i = 1, n do sum = sum + (tonumber(t[i]) or 0.0) end
  local mean = sum / n
  local vsum = 0.0
  for i = 1, n do
    local d = (tonumber(t[i]) or 0.0) - mean
    vsum = vsum + d * d
  end
  local var = vsum / n
  return mean, var
end

local function clamp_int(v, lo, hi)
  if v < lo then return lo end
  if v > hi then return hi end
  return v
end

local function noise_get_src(src, w, h, c, x, y, ch)
  x = clamp_int(x, 0, w - 1)
  y = clamp_int(y, 0, h - 1)
  ch = clamp_int(ch, 0, c - 1)
  local idx = (y * w + x) * c + ch + 1
  return tonumber(src[idx] or 0.0) or 0.0
end

local function noise_sample_bilinear(src, sw, sh, c, xf, yf, ch)
  local x0 = math.floor(xf)
  local y0 = math.floor(yf)
  local x1 = x0 + 1
  local y1 = y0 + 1
  local tx = xf - x0
  local ty = yf - y0

  local v00 = noise_get_src(src, sw, sh, c, x0, y0, ch)
  local v10 = noise_get_src(src, sw, sh, c, x1, y0, ch)
  local v01 = noise_get_src(src, sw, sh, c, x0, y1, ch)
  local v11 = noise_get_src(src, sw, sh, c, x1, y1, ch)

  local a = v00 + (v10 - v00) * tx
  local b = v01 + (v11 - v01) * tx
  return a + (b - a) * ty
end

local function upsample_noise(src, sw, sh, c, dw, dh)
  local dst = {}
  local sx = (sw - 1) / math.max(1, dw - 1)
  local sy = (sh - 1) / math.max(1, dh - 1)
  local di = 1
  for oy = 0, dh - 1 do
    local yf = oy * sy
    for ox = 0, dw - 1 do
      local xf = ox * sx
      for ch = 0, c - 1 do
        dst[di] = noise_sample_bilinear(src, sw, sh, c, xf, yf, ch)
        di = di + 1
      end
    end
  end
  return dst
end

local function make_fbm_noise(w, h, c, octaves, lacunarity, gain)
  local out = {}
  local n = w * h * c
  for i = 1, n do out[i] = 0.0 end

  local amp = 1.0
  local freq = 1.0

  local oct = math.max(1, math.floor(octaves or 4))
  local lac = (lacunarity and lacunarity > 1.0) and lacunarity or 2.0
  local g = (gain and gain > 0.0 and gain < 1.0) and gain or 0.5

  for o = 1, oct do
    -- Taille de la grille: plus petite => plus basse fréquence
    local gw = math.max(2, math.floor(w / freq))
    local gh = math.max(2, math.floor(h / freq))

    -- Génère une grille gaussienne (par canal)
    local grid = {}
    local gi = 1
    for y = 0, gh - 1 do
      for x = 0, gw - 1 do
        for ch = 0, c - 1 do
          grid[gi] = randn()
          gi = gi + 1
        end
      end
    end

    local up = upsample_noise(grid, gw, gh, c, w, h)
    for i = 1, n do
      out[i] = out[i] + up[i] * amp
    end
    amp = amp * g
    freq = freq * lac
  end

  -- Normalise ~variance 1
  local mean, var = compute_mean_var(out)
  local inv_std = (var > 1e-12) and (1.0 / math.sqrt(var)) or 1.0
  for i = 1, n do
    out[i] = (out[i] - mean) * inv_std
  end
  return out
end

-- Bruit "peltier": approximation de shot-noise (Skellam = Poisson(λ) - Poisson(λ)), normalisée.
-- Note: ce n'est pas un modèle physique complet; c'est une distribution discrète à queues plus lourdes.
local function poisson_knuth(lambda)
  if lambda <= 0 then return 0 end
  local L = math.exp(-lambda)
  local k = 0
  local p = 1.0
  repeat
    k = k + 1
    p = p * math.random()
  until p <= L
  return k - 1
end

local function peltier_noise_unit(lambda)
  -- Skellam centré: var = 2λ, donc /sqrt(2λ) -> variance ~1.
  local lam = math.max(1e-6, lambda)
  local a = poisson_knuth(lam)
  local b = poisson_knuth(lam)
  return (a - b) / math.sqrt(2.0 * lam)
end

local function sample_noise(noise_std)
  if not noise_std or noise_std <= 0.0 then return 0.0 end
  if NOISE_KIND == "peltier" then
    return peltier_noise_unit(PELTIER_LAMBDA) * noise_std
  end
  return randn() * noise_std
end

local img_dim = cfg.image_w * cfg.image_h * cfg.image_c
local x_img = {}
if INIT_MODE == "noise" then
  if NOISE_KIND == "fbm" then
    local n0 = make_fbm_noise(cfg.image_w, cfg.image_h, cfg.image_c, NOISE_OCTAVES, NOISE_LACUNARITY, NOISE_GAIN)
    for i = 1, img_dim do x_img[i] = n0[i] end
  else
    for i = 1, img_dim do x_img[i] = randn() end
  end
else
  for i = 1, img_dim do x_img[i] = 0.0 end
end

-- Image de base fixe (utilisée en mode prompt-only)
local x_base = {}
for i = 1, img_dim do x_base[i] = x_img[i] end

local function lerp(a, b, t)
  return a + (b - a) * t
end

local function clamp_f32(v, lo, hi)
  if v < lo then return lo end
  if v > hi then return hi end
  return v
end

local function is_truthy(v)
  if v == nil then return false end
  local s = tostring(v):lower()
  return not (s == "0" or s == "false" or s == "off" or s == "no")
end

local function preprocess_enabled()
  return is_truthy(PREPROCESS)
end

local function preprocess_clamp_enabled()
  return is_truthy(PRE_CLAMP)
end

-- Buffers réutilisés pour éviter du GC à chaque step.
local _pre_blur = {}
local _pre_out = {}

local function box_blur_3x3_into(dst, src, w, h, c)
  local di = 1
  for yy = 0, h - 1 do
    local y0 = yy - 1
    local y1 = yy
    local y2 = yy + 1
    for xx = 0, w - 1 do
      local x0 = xx - 1
      local x1 = xx
      local x2 = xx + 1
      for ch = 0, c - 1 do
        local s = 0.0
        s = s + noise_get_src(src, w, h, c, x0, y0, ch)
        s = s + noise_get_src(src, w, h, c, x1, y0, ch)
        s = s + noise_get_src(src, w, h, c, x2, y0, ch)
        s = s + noise_get_src(src, w, h, c, x0, y1, ch)
        s = s + noise_get_src(src, w, h, c, x1, y1, ch)
        s = s + noise_get_src(src, w, h, c, x2, y1, ch)
        s = s + noise_get_src(src, w, h, c, x0, y2, ch)
        s = s + noise_get_src(src, w, h, c, x1, y2, ch)
        s = s + noise_get_src(src, w, h, c, x2, y2, ch)
        dst[di] = s / 9.0
        di = di + 1
      end
    end
  end
end

-- Applique lissage+accentuation+contraste sur une image normalisée [-1,1] avant réinjection.
-- Retourne une table réutilisée (ne pas la modifier ailleurs).
local function preprocess_for_reinject(src)
  if not preprocess_enabled() then
    return src
  end

  local w, h, c = cfg.image_w, cfg.image_h, cfg.image_c
  box_blur_3x3_into(_pre_blur, src, w, h, c)

  local smooth = clamp_f32(PRE_SMOOTH or 0.0, 0.0, 1.0)
  local sharpen = PRE_SHARPEN or 0.0
  local contrast = PRE_CONTRAST or 1.0
  local do_clamp = preprocess_clamp_enabled()

  for i = 1, img_dim do
    local v = tonumber(src[i] or 0.0) or 0.0
    local b = tonumber(_pre_blur[i] or 0.0) or 0.0

    -- 1) lissage léger
    local sm = v + (b - v) * smooth

    -- 2) unsharp mask (accentuation): sm + k*(sm - blur)
    local sh = sm + sharpen * (sm - b)

    -- 3) contraste
    local out = sh * contrast
    if do_clamp then
      out = clamp_f32(out, -1.0, 1.0)
    end
    _pre_out[i] = out
  end

  return _pre_out
end

-- Resize utilitaires dédiés à l'image d'init (évite de dépendre des helpers définis plus bas).
local function init_get_src(src, w, h, c, x, y, ch)
  x = clamp_int(x, 0, w - 1)
  y = clamp_int(y, 0, h - 1)
  ch = clamp_int(ch, 0, c - 1)
  local idx = (y * w + x) * c + ch + 1
  return tonumber(src[idx] or 0.0) or 0.0
end

local function init_sample_nearest(src, sw, sh, c, xf, yf, ch)
  local x = math.floor(xf + 0.5)
  local y = math.floor(yf + 0.5)
  return init_get_src(src, sw, sh, c, x, y, ch)
end

local function init_sample_bilinear(src, sw, sh, c, xf, yf, ch)
  local x0 = math.floor(xf)
  local y0 = math.floor(yf)
  local x1 = x0 + 1
  local y1 = y0 + 1
  local tx = xf - x0
  local ty = yf - y0

  local v00 = init_get_src(src, sw, sh, c, x0, y0, ch)
  local v10 = init_get_src(src, sw, sh, c, x1, y0, ch)
  local v01 = init_get_src(src, sw, sh, c, x0, y1, ch)
  local v11 = init_get_src(src, sw, sh, c, x1, y1, ch)

  local a = v00 + (v10 - v00) * tx
  local b = v01 + (v11 - v01) * tx
  return a + (b - a) * ty
end

local function init_resize_image(src, sw, sh, c, dw, dh, mode)
  local dst = {}
  local sx = (sw - 1) / math.max(1, dw - 1)
  local sy = (sh - 1) / math.max(1, dh - 1)
  local sampler = (mode == "nearest") and init_sample_nearest or init_sample_bilinear
  local di = 1
  for oy = 0, dh - 1 do
    local yf = oy * sy
    for ox = 0, dw - 1 do
      local xf = ox * sx
      for ch = 0, c - 1 do
        dst[di] = sampler(src, sw, sh, c, xf, yf, ch)
        di = di + 1
      end
    end
  end
  return dst
end

local function ppm_read_token(f)
  while true do
    local ch = f:read(1)
    if not ch then return nil end
    if ch == "#" then
      f:read("*l")
    elseif ch:match("%s") then
      -- skip
    else
      local tok = { ch }
      while true do
        local c2 = f:read(1)
        if not c2 or c2:match("%s") then break end
        if c2 == "#" then
          f:read("*l")
          break
        end
        tok[#tok + 1] = c2
      end
      return table.concat(tok)
    end
  end
end

local function load_ppm_p6_float(path)
  local f = io.open(path, "rb")
  if not f then return nil, "Impossible d'ouvrir: " .. tostring(path) end

  local magic = ppm_read_token(f)
  if magic ~= "P6" then
    f:close()
    return nil, "Format non supporté (attendu PPM P6). Convertis via: `convert input.png -colorspace RGB output.ppm`"
  end

  local w = tonumber(ppm_read_token(f) or "")
  local h = tonumber(ppm_read_token(f) or "")
  local maxv = tonumber(ppm_read_token(f) or "")
  if not w or not h or not maxv or w <= 0 or h <= 0 then
    f:close()
    return nil, "Header PPM invalide"
  end
  if maxv <= 0 or maxv > 65535 then
    f:close()
    return nil, "PPM maxval invalide: " .. tostring(maxv)
  end

  -- Après maxv, il y a au moins un whitespace avant les bytes.
  local sep = f:read(1)
  if not sep then
    f:close()
    return nil, "PPM tronqué"
  end
  if not sep:match("%s") then
    -- Rare, mais on remet le byte dans le flux via un buffer minimal
    f:seek("cur", -1)
  end

  local bytes_per_sample = (maxv <= 255) and 1 or 2
  local expected = math.floor(w * h * 3 * bytes_per_sample)
  local raw = f:read(expected)
  f:close()
  if not raw or #raw ~= expected then
    return nil, "PPM tronqué: attendu " .. tostring(expected) .. " bytes, got " .. tostring(raw and #raw or 0)
  end

  local scale = 2.0 / maxv
  local out = {}
  local oi = 1
  if bytes_per_sample == 1 then
    for i = 1, #raw do
      local b = string.byte(raw, i)
      out[oi] = b * scale - 1.0
      oi = oi + 1
    end
  else
    local ri = 1
    while ri <= #raw do
      local hi = string.byte(raw, ri)
      local lo = string.byte(raw, ri + 1)
      local v = hi * 256 + lo
      out[oi] = v * scale - 1.0
      oi = oi + 1
      ri = ri + 2
    end
  end
  return { pixels = out, w = w, h = h, c = 3 }, nil
end

-- Si une image est fournie, elle initialise (ou mélange) l'entrée image du modèle.
if INIT_IMAGE and #INIT_IMAGE > 0 then
  local alpha_img = math.max(0.0, math.min(1.0, INIT_IMAGE_ALPHA))
  local img, err_img = load_ppm_p6_float(INIT_IMAGE)
  if not img then
    error("INIT_IMAGE: " .. tostring(err_img))
  end
  if img.c ~= cfg.image_c then
    error("INIT_IMAGE: channels inattendus: got " .. tostring(img.c) .. " expected " .. tostring(cfg.image_c))
  end

  local resized = img.pixels
  if img.w ~= cfg.image_w or img.h ~= cfg.image_h then
    resized = init_resize_image(img.pixels, img.w, img.h, img.c, cfg.image_w, cfg.image_h, INIT_IMAGE_RESIZE)
  end
  if #resized ~= img_dim then
    error("INIT_IMAGE: taille inattendue après resize: got " .. tostring(#resized) .. " expected " .. tostring(img_dim))
  end

  for i = 1, img_dim do
    x_img[i] = lerp(x_img[i], resized[i], alpha_img)
  end

  -- Met à jour aussi la base fixe
  for i = 1, img_dim do
    x_base[i] = x_img[i]
  end
  log(string.format("INIT_IMAGE: %s (%dx%d) -> (%dx%d) alpha=%.3f resize=%s",
    tostring(INIT_IMAGE), img.w, img.h, cfg.image_w, cfg.image_h, alpha_img, tostring(INIT_IMAGE_RESIZE)
  ))
end

local function parse_pyramid(str)
  if not str or str == "" or str == "off" or str == "0" then return nil end
  if str == "auto" then return "auto" end
  local sizes = {}
  for token in string.gmatch(str, "[^,; ]+") do
    local n = tonumber(token)
    if n and n > 0 then sizes[#sizes + 1] = math.floor(n) end
  end
  if #sizes < 2 then return nil end
  table.sort(sizes)
  return sizes
end

local function compute_minmax(t)
  local mn = math.huge
  local mx = -math.huge
  for i = 1, #t do
    local v = tonumber(t[i] or 0.0) or 0.0
    if v < mn then mn = v end
    if v > mx then mx = v end
  end
  if mn == math.huge then mn = 0.0 end
  if mx == -math.huge then mx = 0.0 end
  return mn, mx
end

local input = {}
local y

local function run_step(step_i, noise_std, img_in)
  local expected_len = #text_vec + img_dim
  local idx = 1
  for i = 1, #text_vec do
    input[idx] = text_vec[i]
    idx = idx + 1
  end
  local img = img_in or x_img
  if noise_std and noise_std > 0.0 then
    if NOISE_KIND == "fbm" then
      local nf = make_fbm_noise(cfg.image_w, cfg.image_h, cfg.image_c, NOISE_OCTAVES, NOISE_LACUNARITY, NOISE_GAIN)
      for i = 1, img_dim do
        input[idx] = img[i] + nf[i] * noise_std
        idx = idx + 1
      end
    else
      for i = 1, img_dim do
        input[idx] = img[i] + sample_noise(noise_std)
        idx = idx + 1
      end
    end
  else
    for i = 1, img_dim do
      input[idx] = img[i]
      idx = idx + 1
    end
  end

  -- Vérifie que le prompt (text_vec) est bien en premier: [text_vec..., image...]
  if (idx - 1) ~= expected_len then
    error("Input packing invalide (step " .. tostring(step_i) .. "): got " .. tostring(idx - 1) .. " expected " .. tostring(expected_len))
  end
  -- Nettoie une éventuelle queue si l'input a déjà servi avec une longueur différente.
  input[expected_len + 1] = nil

  local y_step, err_fwd = Mimir.Model.forward(input, false)
  if not y_step then error("Model.forward failed (step " .. tostring(step_i) .. "): " .. tostring(err_fwd)) end
  if #y_step ~= img_dim then
    error("forward: taille output inattendue (step " .. tostring(step_i) .. "): got " .. tostring(#y_step) .. " expected " .. tostring(img_dim))
  end
  return y_step
end

local steps = math.max(1, math.floor(STEPS))
local alpha = math.max(0.0, math.min(1.0, ALPHA))
local noise = math.max(0.0, NOISE_STD)

local function prompt_only_enabled()
  local s = tostring(PROMPT_ONLY):lower()
  return not (s == "0" or s == "false" or s == "off" or s == "no")
end

local prompt_only = prompt_only_enabled()
if prompt_only then
  log("ℹ️  PROMPT_ONLY actif: réinjection + pyramide désactivées")
end

for s = 1, steps do
  y = run_step(s, noise, prompt_only and x_base or x_img)
  if not prompt_only then
    y = preprocess_for_reinject(y)
  end

  -- Ré-injection: on affine en repartant de la prédiction précédente.
  if not prompt_only then
    if alpha >= 1.0 then
      x_img = y
    elseif alpha <= 0.0 then
      -- garde x_img tel quel
    else
      for i = 1, img_dim do
        x_img[i] = lerp(x_img[i], y[i], alpha)
      end
    end
  end

  local mn, mx = compute_minmax(y)
  log(string.format("step %d/%d: out[min=%.4f max=%.4f] noise_std=%.4f",
    s, steps, mn, mx, noise
  ))

  noise = noise * NOISE_DECAY
end

-- Sortie finale potentielle (si pyramide activée)
local y_work = nil
local work_w, work_h = cfg.image_w, cfg.image_h

local function clamp_u8(v)
  if v < 0 then return 0 end
  if v > 255 then return 255 end
  return v
end

local function get_src(y_src, w, h, c, x, y, ch)
  x = clamp_int(x, 0, w - 1)
  y = clamp_int(y, 0, h - 1)
  ch = clamp_int(ch, 0, c - 1)
  local idx = (y * w + x) * c + ch + 1
  return tonumber(y_src[idx] or 0.0) or 0.0
end

local function sample_nearest(y_src, sw, sh, c, xf, yf, ch)
  local x = math.floor(xf + 0.5)
  local y = math.floor(yf + 0.5)
  return get_src(y_src, sw, sh, c, x, y, ch)
end

local function sample_bilinear(y_src, sw, sh, c, xf, yf, ch)
  local x0 = math.floor(xf)
  local y0 = math.floor(yf)
  local x1 = x0 + 1
  local y1 = y0 + 1
  local tx = xf - x0
  local ty = yf - y0

  local v00 = get_src(y_src, sw, sh, c, x0, y0, ch)
  local v10 = get_src(y_src, sw, sh, c, x1, y0, ch)
  local v01 = get_src(y_src, sw, sh, c, x0, y1, ch)
  local v11 = get_src(y_src, sw, sh, c, x1, y1, ch)

  local a = v00 + (v10 - v00) * tx
  local b = v01 + (v11 - v01) * tx
  return a + (b - a) * ty
end

local function resize_image(src, sw, sh, c, dw, dh, mode)
  local dst = {}
  local sx = (sw - 1) / math.max(1, dw - 1)
  local sy = (sh - 1) / math.max(1, dh - 1)
  local sampler = (mode == "nearest") and sample_nearest or sample_bilinear
  local di = 1
  for oy = 0, dh - 1 do
    local yf = oy * sy
    for ox = 0, dw - 1 do
      local xf = ox * sx
      for ch = 0, c - 1 do
        dst[di] = sampler(src, sw, sh, c, xf, yf, ch)
        di = di + 1
      end
    end
  end
  return dst
end

local function resize_into(dst, src, sw, sh, c, dw, dh, mode)
  local sx = (sw - 1) / math.max(1, dw - 1)
  local sy = (sh - 1) / math.max(1, dh - 1)
  local sampler = (mode == "nearest") and sample_nearest or sample_bilinear
  local di = 1
  for oy = 0, dh - 1 do
    local yf = oy * sy
    for ox = 0, dw - 1 do
      local xf = ox * sx
      for ch = 0, c - 1 do
        dst[di] = sampler(src, sw, sh, c, xf, yf, ch)
        di = di + 1
      end
    end
  end
end

-- Pyramide multi-scale (optionnelle)
do
  if prompt_only then
    -- En mode prompt-only on évite toute boucle qui réinjecte l'image dans le modèle.
    -- La sortie finale reste y (avec upscale simple si demandé).
    goto pyramid_done
  end

  local pyramid = parse_pyramid(PYRAMID)
  local pyr_steps = math.max(0, math.floor(PYRAMID_STEPS))
  local pyr_alpha = math.max(0.0, math.min(1.0, PYRAMID_ALPHA))
  local out_w = math.max(1, math.floor(OUT_W))
  local out_h = math.max(1, math.floor(OUT_H))

  if pyramid and pyramid == "auto" then
    local target = math.max(out_w, out_h)
    local size = math.max(cfg.image_w, cfg.image_h)
    local sizes = { size }
    while size < target do
      size = size * 2
      sizes[#sizes + 1] = size
    end
    pyramid = sizes
  end

  if type(pyramid) == "table" and #pyramid >= 2 and pyr_steps > 0 then
    local base = math.max(cfg.image_w, cfg.image_h)
    if pyramid[1] ~= base then
      pyramid[#pyramid + 1] = base
      table.sort(pyramid)
    end

    y_work = y
    work_w, work_h = cfg.image_w, cfg.image_h

    local x_small = {}
    for i = 1, img_dim do x_small[i] = 0.0 end

    for stage = 2, #pyramid do
      local target = pyramid[stage]
      local tw, th = target, target
      log(string.format("pyramid stage %d/%d: %dx%d -> %dx%d (steps=%d alpha=%.3f)",
        stage - 1, #pyramid - 1, work_w, work_h, tw, th, pyr_steps, pyr_alpha
      ))

      y_work = resize_image(y_work, work_w, work_h, cfg.image_c, tw, th, UPSCALE)
      work_w, work_h = tw, th

      local stage_noise = noise
      for ps = 1, pyr_steps do
        resize_into(x_small, y_work, work_w, work_h, cfg.image_c, cfg.image_w, cfg.image_h, "bilinear")
        local y_small = run_step(string.format("pyr%d.%d", stage, ps), stage_noise, x_small)
        local y_up = resize_image(y_small, cfg.image_w, cfg.image_h, cfg.image_c, work_w, work_h, UPSCALE)

        if pyr_alpha >= 1.0 then
          y_work = y_up
        elseif pyr_alpha <= 0.0 then
          -- ne change rien
        else
          for i = 1, work_w * work_h * cfg.image_c do
            y_work[i] = lerp(y_work[i], y_up[i], pyr_alpha)
          end
        end

        stage_noise = stage_noise * NOISE_DECAY
      end
    end
  end

  ::pyramid_done::
end

local out_w = math.max(1, math.floor(OUT_W))
local out_h = math.max(1, math.floor(OUT_H))
local y_src = y_work or y
local src_w = y_work and work_w or cfg.image_w
local src_h = y_work and work_h or cfg.image_h
local do_upscale = (UPSCALE ~= "none") and (out_w ~= src_w or out_h ~= src_h)

local out_path = "generated.ppm"

local f = io.open(out_path, "wb")
if not f then error("Impossible d'ouvrir: " .. tostring(out_path)) end
f:write(string.format("P6\n%d %d\n255\n", out_w, out_h))

if not y_src then error("Aucun output modèle (y=nil)") end

if not do_upscale then
  local bytes = {}
  local bi = 1
  for i = 1, src_w * src_h * cfg.image_c do
    local v = tonumber(y_src[i] or 0.0) or 0.0
    local u = clamp_u8(math.floor((v * 0.5 + 0.5) * 255.0 + 0.5))
    bytes[bi] = string.char(u)
    bi = bi + 1
  end
  f:write(table.concat(bytes))
else
  local sw, sh, c = src_w, src_h, cfg.image_c
  local sx = (sw - 1) / math.max(1, out_w - 1)
  local sy = (sh - 1) / math.max(1, out_h - 1)
  local sampler
  if UPSCALE == "nearest" then
    sampler = sample_nearest
  else
    sampler = sample_bilinear
  end

  -- Écriture stream par ligne pour éviter un gros buffer Lua.
  for oy = 0, out_h - 1 do
    local row = {}
    local ri = 1
    local yf = oy * sy
    for ox = 0, out_w - 1 do
      local xf = ox * sx
      for ch = 0, c - 1 do
        local v = sampler(y_src, sw, sh, c, xf, yf, ch)
        local u = clamp_u8(math.floor((v * 0.5 + 0.5) * 255.0 + 0.5))
        row[ri] = string.char(u)
        ri = ri + 1
      end
    end
    f:write(table.concat(row))
  end
end
f:close()

log("✓ Image générée: " .. out_path)
log("(Astuce: pour PNG: `convert generated.ppm generated.png`)")
