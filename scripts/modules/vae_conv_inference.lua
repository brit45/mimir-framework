-- Checkpoint compatibility for native VAEConv inference. This module only
-- resolves configuration and weights; graph construction stays in C++.
local M = {}
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")
local Ckpt = dofile(ROOTWORK.."/scripts/modules/checkpoint_resume.lua")
local JSON = dofile(ROOTWORK.."/scripts/modules/mpk.lua")
local function fail(message) error("[vae_conv_inference] "..tostring(message), 2) end
local function copy(values)
  local result = {}; for k, v in pairs(values or {}) do result[k] = v end; return result
end
local function str(value, fallback)
  if value == nil or value == true then return fallback end
  return tostring(value)
end
local function json_file(path)
  local ok, value = pcall(read_json, path)
  if not ok or type(value) ~= "table" then fail("JSON absent ou invalide: "..path) end
  return value
end
local function size(path)
  local f = io.open(path, "rb"); if not f then return nil end
  local n = f:seek("end"); f:close(); return n
end
local function read_bytes(path, offset, count)
  local f, err = io.open(path, "rb"); if not f then fail(err) end
  local position = f:seek("set", offset)
  local bytes = position and f:read(count); f:close()
  if not bytes or #bytes ~= count then fail("données tronquées: "..path) end
  return bytes
end
local function integer(n) return type(n) == "number" and n >= 0 and n % 1 == 0 end
local function product(shape)
  if type(shape) ~= "table" then fail("shape de poids absente") end
  local n = 1
  for _, d in ipairs(shape) do
    if not integer(d) or d < 1 then fail("shape de poids invalide") end
    n = n * d
  end
  return n
end
local function conv_mode(value)
  local mode = tostring(value or "2d"):lower():match("^%s*(.-)%s*$")
  mode = ({["2"]="2d", ["3"]="3d", ["4"]="4d", conv2d="2d", conv3d="3d", conv4d="4d"})[mode] or mode
  if mode ~= "2d" and mode ~= "3d" and mode ~= "4d" then fail("--conv-mode invalide: "..mode) end
  return mode
end
local function read_safe(path)
  local file_size = size(path)
  if not file_size or file_size < 10 then fail("SafeTensors absent ou tronqué: "..path) end
  local length = string.unpack("<I8", read_bytes(path, 0, 8))
  if length < 2 or length > 64*1024*1024 or length > file_size-8 then fail("en-tête SafeTensors invalide") end
  local header, err = JSON.decode_json(read_bytes(path, 8, length))
  if type(header) ~= "table" then fail(err or "en-tête SafeTensors invalide") end
  return header, 8+length, file_size
end
local function safe_entry(ctx, entry)
  local offsets = type(entry) == "table" and entry.data_offsets
  if type(offsets) ~= "table" or not integer(offsets[1]) or not integer(offsets[2])
      or offsets[2] < offsets[1] or ctx.data_offset+offsets[2] > ctx.file_size then
    fail("offsets SafeTensors invalides")
  end
  return ctx.data_offset+offsets[1], offsets[2]-offsets[1]
end

-- Restore options from the saved native graph, including older model_config
-- objects that predate these options. No layers are created here.
local function saved_options(cfg, layers)
  if type(layers) ~= "table" or #layers == 0 then return end
  local rank, context, detail, resnet, attention = "2d", false, false, false, false
  local norms, groups = {enc="none", dec="none"}, {enc=0, dec=0}
  local res_tokens, attn_tokens, heads = 0, 0, 0
  cfg.use_encoder_prior, cfg.use_skip_connections, cfg.text_cond = false, false, false
  for _, layer in ipairs(layers) do
    local name, kind = tostring(layer.name), tostring(layer.type):lower()
    if kind == "conv4d" then rank = "4d" elseif kind == "conv3d" and rank ~= "4d" then rank = "3d" end
    if name:find("/latent_context/", 1, true) then context = true end
    if name:find("/detail/", 1, true) then detail = true end
    if name:match("/latent_context/project$") or name == "vae_conv/enc/detail/project" then
      cfg.latent_context_channels = tonumber(layer.out_channels) or cfg.latent_context_channels
    end
    if name == "vae_conv/enc/conv_in" then
      cfg.image_w, cfg.image_h = layer.input_width or cfg.image_w, layer.input_height or cfg.image_h
      cfg.image_c, cfg.base_channels = layer.in_channels or cfg.image_c, layer.out_channels or cfg.base_channels
    elseif name == "vae_conv/enc/mu" then
      cfg.latent_w, cfg.latent_h = layer.input_width or cfg.latent_w, layer.input_height or cfg.latent_h
      cfg.latent_c = layer.out_channels or cfg.latent_c
    elseif name == "vae_conv/enc/logvar" and layer.use_bias ~= nil then
      cfg.logvar_head_bias = layer.use_bias == true
    end
    if name == "vae_conv/z_prior_bias" then cfg.use_encoder_prior = true end
    if name:match("^vae_conv/dec/up%d+/skip_proj$") then cfg.use_skip_connections = true end
    if name == "vae_conv/txt/tok_emb" then
      cfg.text_cond = true
      cfg.vocab_size, cfg.text_d_model, cfg.seq_len = layer.vocab_size or cfg.vocab_size,
        layer.embed_dim or cfg.text_d_model, layer.seq_len or cfg.seq_len
    elseif name == "vae_conv/txt_proj" then cfg.proj_dim = layer.out_features or cfg.proj_dim end
    local side = name:match("^vae_conv/(enc)/") or name:match("^vae_conv/(dec)/")
    if side then
      if kind == "groupnorm" or kind == "layernorm" then
        if norms[side] ~= "none" and norms[side] ~= kind then fail("normalisations mixtes non prises en charge: "..side) end
        norms[side] = kind
        groups[side] = math.max(groups[side], tonumber(layer.num_groups) or 0)
      end
      local tokens = (tonumber(layer.input_height) or 0)*(tonumber(layer.input_width) or 0)
      if name:match("/res0/conv1$") or name:match("/res/conv1$") or name:match("/bot_res/conv1$") then
        resnet, res_tokens = true, math.max(res_tokens, tokens)
      end
      if kind == "selfattention" then
        attention, attn_tokens = true, math.max(attn_tokens, tonumber(layer.seq_len) or tokens)
        heads = math.max(heads, tonumber(layer.num_heads) or 0)
      end
    end
    if name:match("^vae_conv/dec/up%d+/up$") then
      cfg.decoder_upsample = ({convtranspose2d="conv_transpose", upsamplenearest="nearest_conv",
        upsamplebilinear="bilinear_conv", pixelshuffle="pixel_shuffle"})[kind] or cfg.decoder_upsample
    end
  end
  cfg.conv_mode, cfg.latent_spatial_context, cfg.latent_detail_path = rank, context, detail
  cfg.resnet, cfg.attention = resnet, attention
  cfg.use_resnet, cfg.use_attention, cfg.use_attn = nil, nil, nil
  cfg.enc_norm, cfg.dec_norm = norms.enc, norms.dec
  if groups.enc > 0 then cfg.enc_gn_groups = groups.enc end
  if groups.dec > 0 then cfg.dec_gn_groups = groups.dec end
  if heads > 0 then cfg.attn_heads = heads end
  if res_tokens > 0 then cfg.resnet_max_tokens = res_tokens end
  if attn_tokens > 0 then cfg.attn_max_tokens = attn_tokens end
end

function M.resolve(opts)
  opts = opts or {}
  local conf = type(CONF) == "table" and CONF or {}
  local external
  if opts.config ~= nil then
    if type(opts.config) ~= "string" then fail("--config exige un chemin JSON") end
    external = json_file(opts.config)
    conf = external
  end
  local model = type(conf.model) == "table" and conf.model or
    (external and (external.model_config or external.modelConfig or external)) or {}
  local training = type(conf.training) == "table" and conf.training or {}
  local inference = type(conf.inference) == "table" and conf.inference or {}
  local ctx = {checkpoint=str(opts.checkpoint, str(opts.ckpt,
    inference.checkpoint or training.out_dir or "checkpoint/vae_conv-generique.v8"))}
  ctx.format = str(opts.format, "auto"):lower()
  ctx.format = ({rawfolder="raw_folder", safetensor="safetensors"})[ctx.format] or ctx.format
  if ctx.format == "auto" or ctx.format == "" then
    ctx.format = (ctx.checkpoint:lower():match("%.safetensors$") or ctx.checkpoint:lower():match("%.st$"))
      and "safetensors" or "raw_folder"
  end
  local arch
  if ctx.format == "raw_folder" then
    ctx.checkpoint = Ckpt.resolve_dir(ctx.checkpoint)
    if not ctx.checkpoint then fail("aucun checkpoint raw_folder chargeable") end
    arch = json_file(FS.join(ctx.checkpoint, "model", "architecture.json"))
    local manifest = json_file(FS.join(ctx.checkpoint, "manifest.json"))
    ctx.tensors = {}
    for _, entry in ipairs(manifest.tensor_index or {}) do ctx.tensors[entry.name] = entry end
  elseif ctx.format == "safetensors" then
    ctx.tensors, ctx.data_offset, ctx.file_size = read_safe(ctx.checkpoint)
    local entry = ctx.tensors["model/architecture_json"]
    if entry then
      local offset, count = safe_entry(ctx, entry)
      if entry.dtype ~= "U8" or count > 64*1024*1024 then fail("architecture SafeTensors invalide") end
      local err; arch, err = JSON.decode_json(read_bytes(ctx.checkpoint, offset, count))
      if type(arch) ~= "table" then fail(err or "architecture SafeTensors illisible") end
    end
  else fail("format non supporté: "..ctx.format) end
  local saved = arch and (arch.model_config or arch.modelConfig)
  if type(saved) ~= "table" then
    if ctx.format ~= "safetensors" or next(model) == nil then
      fail("model_config absent; fournir la configuration correspondante via --config pour un ancien export SafeTensors")
    end
    saved = model
  end
  if saved.type and saved.type ~= "vae_conv" then fail("checkpoint attendu: vae_conv") end
  local cfg, err = Mimir.Architectures.default_config("vae_conv")
  if type(cfg) ~= "table" then fail(err) end
  -- Never inherit topology from the launch configuration when it is serialized.
  for k, v in pairs(saved) do cfg[k] = v end
  cfg.latent_spatial_context = saved.latent_spatial_context == true
  cfg.latent_detail_path = saved.latent_detail_path == true
  cfg.logvar_head_bias = saved.logvar_head_bias == true
  cfg.conv_mode = conv_mode(saved.conv_mode)
  saved_options(cfg, arch and arch.layers)
  for _, source in ipairs({{"--conv-mode", opts["conv-mode"]}, {"--conv_mode", opts.conv_mode},
      {"CONF.model.conv_mode", model.conv_mode}, {"CONF.training.conv_mode", training.conv_mode}}) do
    if source[2] ~= nil and conv_mode(source[2]) ~= cfg.conv_mode then
      fail("configuration incompatible avec le checkpoint: "..source[1].." (checkpoint="..cfg.conv_mode..")")
    end
  end
  for _, key in ipairs({"image_w","image_h","image_c","latent_w","latent_h","latent_c","base_channels"}) do
    local cli = key:gsub("_", "-")
    local value = tonumber(cfg[key])
    if not value or value <= 0 or value % 1 ~= 0 then fail("dimension invalide: "..key) end
    cfg[key] = value
    if opts[cli] ~= nil and tonumber(opts[cli]) ~= value then
      fail("configuration incompatible avec le checkpoint: --"..cli)
    end
  end
  if cfg.image_c ~= 3 then fail("les scripts d'image exigent image_c=3") end
  cfg.latent_dim = cfg.latent_w*cfg.latent_h*cfg.latent_c
  cfg.stochastic_latent = false
  local dtype = str(opts.dtype, cfg.dtype or os.getenv("MIMIR_DTYPE"))
  local aliases = {F32="float32", FP32="float32", F16="float16", FP16="float16", BF16="bfloat16"}
  cfg.dtype = dtype and (aliases[dtype:upper()] or dtype:lower()) or nil
  ctx.cfg, ctx.dtype, ctx.layers = cfg, cfg.dtype, {}
  for _, layer in ipairs(arch and arch.layers or {}) do ctx.layers[layer.name] = layer end
  return ctx
end

local float_sizes = {F32=4, FLOAT32=4, F16=2, FLOAT16=2, BF16=2, BFLOAT16=2, F64=8, FLOAT64=8}
local function weights(ctx, name, expected)
  local entry = ctx.tensors[name..(ctx.format == "raw_folder" and "_weights" or "/weights")]
  if ctx.format == "safetensors" then entry = entry or ctx.tensors[name] end
  if not entry then fail("Poids manquants: "..name) end
  local metadata, path, offset, count
  if ctx.format == "raw_folder" then
    if type(entry.json_file) ~= "string" then fail("métadonnées de poids absentes: "..name) end
    metadata = json_file(FS.join(ctx.checkpoint, entry.json_file))
    if type(metadata.data_file) ~= "string" then fail("fichier de poids absent: "..name) end
    path, offset = FS.join(ctx.checkpoint, "tensors", metadata.data_file), 0
    count = size(path)
  else
    metadata, path = entry, ctx.checkpoint
    offset, count = safe_entry(ctx, entry)
  end
  local n, dtype = product(metadata.shape), tostring(metadata.dtype):upper()
  if n ~= expected then fail("shape de poids incompatible: "..name.." (checkpoint="..n..", modèle="..expected..")") end
  if not float_sizes[dtype] then fail("dtype de poids non pris en charge: "..name.." ("..dtype..")") end
  if count ~= n*float_sizes[dtype] then fail("taille de poids invalide ou fichier absent: "..name) end
  return path, offset, count, dtype
end

function M.create_and_load(ctx, architecture, output)
  local cfg = copy(ctx.cfg); cfg.inference_output_tensor = output
  local ok, err = Mimir.Model.create(architecture, cfg)
  if ok == false then fail("Model.create: "..tostring(err)) end
  if ctx.dtype then
    ok, err = Mimir.Model.dtype(ctx.dtype); if ok == false then fail(err) end
  end
  local layers = Mimir.Model.get_layers()
  local producers, needed = {}, {}
  for _, layer in ipairs(layers) do producers[layer.output] = layer end
  local function visit(tensor)
    local layer = producers[tensor]; if not layer or needed[layer.name] then return end
    needed[layer.name] = true
    for _, input in ipairs(layer.inputs or {}) do visit(input) end
  end
  visit(output)
  if not producers[output] then fail("sortie native inconnue: "..tostring(output)) end
  for _, layer in ipairs(layers) do
    if needed[layer.name] and (tonumber(layer.param_count) or 0) > 0 and not layer.name:match("/zero_enc_skip$") then
      weights(ctx, layer.name, layer.param_count)
      local saved = ctx.layers[layer.name]
      if saved and tostring(saved.type):lower() ~= tostring(layer.type):lower() then fail("type de couche incompatible: "..layer.name) end
      for _, key in ipairs({"in_channels","out_channels","input_height","input_width","kernel_size",
          "kernel_h","kernel_w","stride","stride_h","stride_w","padding","pad_h","pad_w",
          "dilation","dilation_h","dilation_w","groups","num_groups","eps","num_heads","use_bias"}) do
        if saved and saved[key] ~= nil and layer[key] ~= nil and saved[key] ~= layer[key] then
          fail("configuration de couche incompatible: "..layer.name.."."..key)
        end
      end
      for _, key in ipairs({"input_spatial_shape","kernel_shape","stride_shape","padding_shape","dilation_shape"}) do
        if saved and type(saved[key]) == "table" and type(layer[key]) == "table" then
          if #saved[key] ~= #layer[key] then fail("géométrie incompatible: "..layer.name.."."..key) end
          for i, value in ipairs(saved[key]) do
            if value ~= layer[key][i] then fail("géométrie incompatible: "..layer.name.."."..key) end
          end
        end
      end
    end
  end
  ok, err = Mimir.Model.allocate_params(); if ok == false then fail(err) end
  ok, err = Mimir.Serialization.load(ctx.checkpoint, ctx.format, {
    load_encoder=false, load_tokenizer=false, load_optimizer=false,
    strict_mode=false, validate_checksums=true, apply_model_config=false,
    load_layer_metadata=false,
  })
  if ok == false then fail("Serialization.load: "..tostring(err)) end
end

-- RAW stores mu, while the native decoder consumes the operational z. Keep
-- the historical RAW/edition contract, applying the learned prior exactly once.
function M.decoder_input(ctx, latent)
  if not ctx.cfg.use_encoder_prior then return latent end
  if not ctx.prior then
    local path, offset, count, dtype = weights(ctx, "vae_conv/z_prior_bias", ctx.cfg.latent_dim)
    local bytes, prior = read_bytes(path, offset, count), {}
    for i=1,ctx.cfg.latent_dim do
      local value
      if dtype == "F32" or dtype == "FLOAT32" then value = string.unpack("<f", bytes, (i-1)*4+1)
      elseif dtype == "F64" or dtype == "FLOAT64" then
        -- The native checkpoint reader converts F64 weights to float32.
        value = string.unpack("<d", bytes, (i-1)*8+1)
        value = string.unpack("<f", string.pack("<f", value))
      else
        local bits = string.unpack("<I2", bytes, (i-1)*2+1)
        if dtype == "BF16" or dtype == "BFLOAT16" then
          value = string.unpack("<f", string.pack("<I4", bits << 16))
        else
          local sign, exponent, fraction = bits >> 15, (bits >> 10) & 31, bits & 1023
          if exponent == 31 then value = fraction == 0 and math.huge or (0/0)
          elseif exponent == 0 then value = fraction*2^-24
          else value = (1+fraction/1024)*2^(exponent-15) end
          if sign ~= 0 then value = -value end
        end
      end
      if value ~= value or math.abs(value) == math.huge then fail("prior non fini") end
      prior[i] = value
    end
    ctx.prior = prior
  end
  local result = {}; for i=1,#latent do result[i] = latent[i]+ctx.prior[i] end
  return result
end

return M
