#!/usr/bin/env lua
---@diagnostic disable: undefined-field, need-check-nil

-- PonyXL / SDXL monolithic SafeTensors loader via external_safetensors_base.
--
-- Ce script charge le checkpoint externe dans le modèle base de compatibilité.
-- Il sert de point d'entrée "inférence"/préchargement pour travailler avec le
-- safetensors monolithique tel quel, sans conversion préalable.
--
-- Limite actuelle: external_safetensors_base est un conteneur de compatibilité
-- non-exécutable. Le script valide et prépare le chargement, mais ne fait pas
-- encore de text2img complet.
--
-- Exemples:
--   1) Charger tout le checkpoint:
--      ./bin/mimir --lua scripts/inferences/ponyxl_external_safetensors_base.lua -- \
--        --mode full --checkpoint ../ponyxl.safetensors --memory-gb 32
--
--   2) Charger seulement text_encoder_2:
--      ./bin/mimir --lua scripts/inferences/ponyxl_external_safetensors_base.lua -- \
--        --mode subset --component text_encoder_2 --checkpoint ../ponyxl.safetensors
--
--   3) Charger un préfixe arbitraire:
--      ./bin/mimir --lua scripts/inferences/ponyxl_external_safetensors_base.lua -- \
--        --mode subset --include model.diffusion_model. --max-tensors 128

local Args = dofile("scripts/modules/args.lua")

local COMPONENT_PREFIXES = {
  unet = "model.diffusion_model.",
  vae = "first_stage_model.",
  text_encoder = "conditioner.embedders.0.",
  text_encoder_2 = "conditioner.embedders.1.",
}

local function logf(fmt, ...)
  local msg = string.format(fmt, ...)
  if type(log) == "function" then
    log(msg)
  else
    print(msg)
  end
end

local function die(msg)
  local text = "[ponyxl_external_safetensors_base] ❌ " .. tostring(msg)
  if type(log) == "function" then
    log(text)
  else
    print(text)
  end
  os.exit(1)
end

local function ok_or_die(ok, err, ctx)
  if ok then return end
  die((ctx or "operation") .. ": " .. tostring(err or "unknown"))
end

local function apply_dtype(cfg)
  local dtype = (type(cfg) == "table" and cfg.dtype) or os.getenv("MIMIR_DTYPE")
  if dtype == nil then return end
  if type(Mimir) ~= "table" or type(Mimir.model) ~= "table" or type(Mimir.model.dtype) ~= "function" then
    return
  end
  local ok, dt_or_err = Mimir.model.dtype(dtype)
  ok_or_die(ok, dt_or_err, "Mimir.model.dtype(" .. tostring(dtype) .. ")")
end

local function trim(s)
  s = tostring(s or "")
  s = s:gsub("^%s+", "")
  s = s:gsub("%s+$", "")
  return s
end

local function split_csv(raw)
  if raw == nil or raw == true then return {} end
  local out = {}
  for part in tostring(raw):gmatch("[^,]+") do
    part = trim(part)
    if part ~= "" then
      out[#out + 1] = part
    end
  end
  return out
end

local function merge_prefixes(dst, src)
  local seen = {}
  for i = 1, #dst do
    seen[dst[i]] = true
  end
  for i = 1, #src do
    local value = src[i]
    if value ~= "" and not seen[value] then
      dst[#dst + 1] = value
      seen[value] = true
    end
  end
  return dst
end

local function collect_component_prefixes(raw)
  local out = {}
  for _, name in ipairs(split_csv(raw)) do
    local prefix = COMPONENT_PREFIXES[name]
    if not prefix then
      die("component inconnu: " .. tostring(name) .. " (attendus: unet, vae, text_encoder, text_encoder_2)")
    end
    out[#out + 1] = prefix
  end
  return out
end

local function table_len(t)
  if type(t) ~= "table" then return 0 end
  return #t
end

local opts = Args.parse(arg)

local mode = tostring(Args.get_str(opts, "mode", "full")):lower()
if mode ~= "full" and mode ~= "subset" then
  die("--mode doit être 'full' ou 'subset'")
end

local checkpoint_path = Args.get_str(opts, "checkpoint", Args.get_str(opts, "ckpt", "../ponyxl.safetensors"))
if checkpoint_path == "" then
  die("--checkpoint requis")
end

local memory_gb = Args.get_num(opts, "memory-gb", 32.0)
local dtype = Args.get_str(opts, "dtype", os.getenv("MIMIR_DTYPE") or "float16")
local requested_include = split_csv(opts.include)
local requested_exclude = split_csv(opts.exclude)
local component_prefixes = collect_component_prefixes(opts.component)

local include_prefixes = {}
if mode == "subset" then
  merge_prefixes(include_prefixes, component_prefixes)
  merge_prefixes(include_prefixes, requested_include)
  if #include_prefixes == 0 then
    include_prefixes = { COMPONENT_PREFIXES.text_encoder_2 }
  end
elseif #requested_include > 0 or #component_prefixes > 0 then
  merge_prefixes(include_prefixes, component_prefixes)
  merge_prefixes(include_prefixes, requested_include)
end

local default_max_tensors = (mode == "full") and 0 or 32
local max_tensors = Args.get_int(opts, "max-tensors", default_max_tensors)

local cfg, err_cfg = Mimir.Architectures.default_config("external_safetensors_base")
if type(cfg) ~= "table" then
  die("default_config(external_safetensors_base) a échoué: " .. tostring(err_cfg))
end

cfg.source_safetensors = checkpoint_path
cfg.include_prefixes = include_prefixes
cfg.exclude_prefixes = requested_exclude
cfg.max_tensors = max_tensors
cfg.dtype = dtype

if Args.apply_overrides and opts and opts.override ~= nil then
  local ok_ov, err_ov = pcall(Args.apply_overrides, cfg, opts)
  if not ok_ov then die(err_ov) end
end

logf("[ponyxl_external_safetensors_base] mode=%s checkpoint=%s", mode, tostring(checkpoint_path))
logf("[ponyxl_external_safetensors_base] dtype=%s memory_gb=%.2f", tostring(cfg.dtype), tonumber(memory_gb) or 0.0)
logf("[ponyxl_external_safetensors_base] include_prefixes=%d exclude_prefixes=%d max_tensors=%d",
  table_len(cfg.include_prefixes), table_len(cfg.exclude_prefixes), tonumber(cfg.max_tensors) or 0)
if table_len(cfg.include_prefixes) > 0 then
  for i = 1, #cfg.include_prefixes do
    logf("  include[%d] = %s", i, tostring(cfg.include_prefixes[i]))
  end
end
if table_len(cfg.exclude_prefixes) > 0 then
  for i = 1, #cfg.exclude_prefixes do
    logf("  exclude[%d] = %s", i, tostring(cfg.exclude_prefixes[i]))
  end
end

ok_or_die(Mimir.MemoryGuard.setLimit(memory_gb), nil, "MemoryGuard.setLimit")
ok_or_die(Mimir.Allocator.configure({
  max_ram_gb = memory_gb,
  enable_compression = true,
  swap_strategy = "lru",
}), nil, "Allocator.configure")

local ok_create, err_create = Mimir.Model.create("external_safetensors_base", cfg)
ok_or_die(ok_create, err_create, "Model.create(external_safetensors_base)")
apply_dtype(cfg)

local ok_alloc, err_alloc = Mimir.Model.allocate_params()
ok_or_die(ok_alloc, err_alloc, "Model.allocate_params")

local ok_load, err_load = Mimir.Serialization.load(checkpoint_path, "safetensors", {
  load_tokenizer = false,
  load_encoder = false,
  load_optimizer = false,
})
ok_or_die(ok_load, err_load, "Serialization.load")

logf("[ponyxl_external_safetensors_base] ✅ checkpoint chargé")
logf("[ponyxl_external_safetensors_base] ℹ️ modèle base prêt pour compatibilité/inspection, pas encore text2img exécutable")