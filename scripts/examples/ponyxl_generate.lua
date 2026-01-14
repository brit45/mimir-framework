 
-- Génération PonyXL (texte -> image) - Mímir
-- Usage:
--   bin/mimir --lua scripts/examples/ponyxl_generate.lua
-- Puis édite PROMPT ci-dessous.

local PROMPT = "dragonoïd"
-- PonyXL ici est un modèle "one-shot" (texte -> image) : il ne conditionne pas sur une image courante.
-- Donc une boucle de steps ne peut pas "dénoyer" comme une diffusion; au mieux c'est un lissage.
-- Par défaut on sort directement l'image; tu peux activer un auto-contraste pour éviter une image grise/bruitée.
local STEPS = 25
local GUIDANCE_SCALE = 4.0
local STEP_SIZE = 1.0
local AUTO_CONTRAST = true
local OUT_PATH = "checkpoint/PonyXL/out.pgm"
local TOKENIZER_PATH = "checkpoint/PonyXL/tokenizer/tokenizer.json"

log("╔═══════════════════════════════════════════════════════════════╗")
log("║            PonyXL Generate (prompt -> image)                  ║")
log("╚═══════════════════════════════════════════════════════════════╝")

-- Runtime/mémoire
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
  Mimir.Model.set_hardware("auto")
end

-- Config modèle (doit matcher l'entraînement / le checkpoint)
local cfg = {
  d_model = 128,
  seq_len = 64,
  hidden_dim = 1024,
  num_hidden_layers = 3,
  image_w = 64,
  image_h = 64,
  dropout = 0.0,
}

-- Créer/build le modèle avant load()
local ok_create, err_create = Mimir.Model.create("ponyxl", cfg)
if not ok_create then
  error("Model.create('ponyxl') failed: " .. tostring(err_create))
end

local ok_build, params_or_err = Mimir.Model.build()
if not ok_build then
  error("Model.build failed: " .. tostring(params_or_err))
end

local ok_alloc, err_alloc = Mimir.Model.allocate_params()
if not ok_alloc then
  error("Model.allocate_params failed: " .. tostring(err_alloc))
end

-- Charger checkpoint PonyXL (remplit les poids)
local ok_load, err_load = Mimir.Serialization.load("checkpoint/PonyXL", "raw_folder")
if not ok_load then
  error("Serialization.load('checkpoint/PonyXL') failed: " .. tostring(err_load))
end

-- Tokenizer: IMPORTANT -> charger celui du checkpoint pour matcher l'entraînement.
-- Un tokenizer "vierge" donne souvent une sortie qui ressemble à du bruit.
if not (Mimir and Mimir.Tokenizer and Mimir.Tokenizer.create) then
  error("Tokenizer API indisponible")
end

local vocab_size = 50000
local ok_tok, err_tok = Mimir.Tokenizer.create(vocab_size)
if ok_tok == false then
  error("Tokenizer.create failed: " .. tostring(err_tok))
end

local tok = Mimir.Tokenizer
if tok.load then
  local ok_load_tok, err_load_tok = tok.load(TOKENIZER_PATH)
  if ok_load_tok then
    log("✓ Tokenizer chargé: " .. TOKENIZER_PATH)
  else
    log("⚠️  Tokenizer.load a échoué, fallback tokenizer vierge: " .. tostring(err_load_tok))
  end
else
  log("⚠️  Tokenizer.load indisponible, fallback tokenizer vierge")
end

local ensure = tok.tokenize_ensure or tok.tokenize
if not ensure then
  error("Tokenizer: tokenize/tokenize_ensure indisponible")
end

-- Embedding déterministe (doit matcher PonyXLModel::embedTextDeterministic)
local function embed_tokens(tokens, seq_len, d_model, pad_id)
  local x = {}
  local idx = 1
  for pos = 1, seq_len do
    local tid = tonumber(tokens[pos] or pad_id) or 0
    for d = 1, d_model do
      x[idx] = math.sin(0.013 * tid + 0.17 * pos + 0.007 * d)
      idx = idx + 1
    end
  end
  return x
end

local function pad_or_truncate(tokens, seq_len, pad_id)
  local out = {}
  for i = 1, seq_len do
    out[i] = tokens[i] or pad_id
  end
  return out
end

local seq_len = cfg.seq_len
local d_model = cfg.d_model
local image_w = cfg.image_w
local image_h = cfg.image_h
local pad_id = (tok.pad_id and tok.pad_id()) or 0

-- Tokenize prompts (cond / uncond) pour guidance
local tokens_cond = ensure(PROMPT)
if type(tokens_cond) ~= "table" or #tokens_cond == 0 then
  error("Tokenization (cond) a échoué")
end

-- "Uncond": PonyXL n'est pas entraîné en classifier-free guidance.
-- On utilise donc un prompt neutre (padding) stable plutôt que tokenize("").
local tokens_uncond = {}

tokens_cond = pad_or_truncate(tokens_cond, seq_len, pad_id)
tokens_uncond = pad_or_truncate(tokens_uncond, seq_len, pad_id)

local x_cond = embed_tokens(tokens_cond, seq_len, d_model, pad_id)
local x_uncond = embed_tokens(tokens_uncond, seq_len, d_model, pad_id)

local N = image_w * image_h

-- Les entrées étant constantes, on calcule les sorties 1 seule fois.
local y_uncond, err_u = Mimir.Model.forward(x_uncond, false)
if not y_uncond then
  error("Model.forward(uncond) failed: " .. tostring(err_u))
end

local y_cond, err_c = Mimir.Model.forward(x_cond, false)
if not y_cond then
  error("Model.forward(cond) failed: " .. tostring(err_c))
end

-- Stats simples (diagnostic)
do
  local sum_c, sum_u, sum2_c, sum2_u = 0.0, 0.0, 0.0, 0.0
  for i = 1, N do
    local vc = tonumber(y_cond[i] or 0) or 0
    local vu = tonumber(y_uncond[i] or 0) or 0
    sum_c = sum_c + vc
    sum_u = sum_u + vu
    sum2_c = sum2_c + vc * vc
    sum2_u = sum2_u + vu * vu
  end
  local mean_c = sum_c / N
  local mean_u = sum_u / N
  local var_c = math.max(0.0, (sum2_c / N) - mean_c * mean_c)
  local var_u = math.max(0.0, (sum2_u / N) - mean_u * mean_u)
  log(string.format("Pred stats | cond: mean=%.4f std=%.4f | uncond: mean=%.4f std=%.4f", mean_c, math.sqrt(var_c), mean_u, math.sqrt(var_u)))
end

-- Boucle multi-steps + guidance
local img = {}
for i = 1, N do
  local u = tonumber(y_uncond[i] or 0) or 0
  local c = tonumber(y_cond[i] or 0) or 0
  -- Guidance optionnelle: si GUIDANCE_SCALE=0 => sortie = uncond.
  -- Si tu veux juste l'image conditionnée, mets GUIDANCE_SCALE=1 et (uncond) sera un prompt neutre.
  local guided = u + GUIDANCE_SCALE * (c - u)
  img[i] = guided
end

-- Optionnel: lissage "steps" (uniquement esthétique, n'ajoute pas d'information)
if STEPS and STEPS > 1 then
  local acc = {}
  for i = 1, N do acc[i] = 0.0 end
  for step = 1, STEPS do
    for i = 1, N do
      acc[i] = acc[i] + STEP_SIZE * (img[i] - acc[i])
    end
    if step == 1 or step == STEPS or (step % 5 == 0) then
      log(string.format("Step %d/%d (guidance=%.2f)", step, STEPS, GUIDANCE_SCALE))
    end
  end
  img = acc
else
  log(string.format("Step 1/1 (guidance=%.2f)", GUIDANCE_SCALE))
end

-- Écrire une image PGM (grayscale)
local function clamp01(v)
  if v < 0 then return 0 end
  if v > 1 then return 1 end
  return v
end

local function clamp(v, lo, hi)
  if v < lo then return lo end
  if v > hi then return hi end
  return v
end

-- Mapping vers [0,255]
local minv, maxv = 1e30, -1e30
if AUTO_CONTRAST then
  for i = 1, N do
    local v = tonumber(img[i] or 0) or 0
    if v < minv then minv = v end
    if v > maxv then maxv = v end
  end
  if not (maxv > minv + 1e-8) then
    AUTO_CONTRAST = false
  else
    log(string.format("Auto-contrast: min=%.4f max=%.4f", minv, maxv))
  end
end

local f = assert(io.open(OUT_PATH, "wb"))
f:write("P5\n")
f:write(string.format("%d %d\n", image_w, image_h))
f:write("255\n")

for i = 1, N do
  local v = tonumber(img[i] or 0) or 0
  local u
  if AUTO_CONTRAST then
    u = (v - minv) / (maxv - minv)
    u = clamp01(u)
  else
    -- Mapping "raw" attendu par l'entraînement ([-1,1] -> [0,1])
    u = (v + 1.0) * 0.5
    u = clamp01(u)
  end
  local b = math.floor(u * 255 + 0.5)
  f:write(string.char(b))
end
f:close()

log("✓ Image écrite: " .. OUT_PATH)
