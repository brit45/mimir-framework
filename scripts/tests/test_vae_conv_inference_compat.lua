-- Exercise the public inference/editor CLIs against synthetic native graphs.
-- No training checkpoint or user dataset is read or modified.
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")
local PPM = dofile(ROOTWORK.."/scripts/modules/image_ppm.lua")
local JSON = dofile(ROOTWORK.."/scripts/modules/mpk.lua")
local directory = os.getenv("MIMIR_VAE_INFERENCE_TEST_DIR")
if not directory or directory == "" then
  directory = os.tmpname()
  os.remove(directory)
end
assert(FS.mkdir_p(directory), "Cannot create temporary test directory")
assert(Mimir.Allocator.configure({max_ram_gb=1, enable_compression=false}))

local function read_bytes(path)
  local file = assert(io.open(path, "rb"))
  local bytes = assert(file:read("*a"))
  assert(file:close())
  return bytes
end

local function write_bytes(path, bytes)
  local file = assert(io.open(path, "wb"))
  assert(file:write(bytes))
  assert(file:close())
end

local subprocesses = 0
local function run(script, name, arguments, expect_failure)
  subprocesses = subprocesses + 1
  local executable = os.getenv("MIMIR_VAE_INFERENCE_TEST_BIN")
    or FS.join(ROOTWORK, FS.is_windows() and "bin/mimir.exe" or "run_mimir.sh")
  local command = {FS.quote(executable), "--lua", FS.quote(FS.join(ROOTWORK, script)), "--"}
  for _, value in ipairs(arguments) do command[#command+1] = FS.quote(value) end
  local log_path = FS.join(directory, name..".log")
  command[#command+1] = ">"..FS.quote(log_path).." 2>&1"
  local ok, why, code = os.execute(table.concat(command, " "))
  local success = ok == true or ok == 0
  if expect_failure then
    assert(not success, name.." unexpectedly succeeded; log: "..log_path)
  else
    assert(success, name.." failed ("..tostring(why).." "..tostring(code).."):\n"..read_bytes(log_path))
  end
  return read_bytes(log_path)
end

local function same_image(actual_path, expected_path, label)
  local actual, expected = assert(PPM.read(actual_path)), assert(PPM.read(expected_path))
  assert(actual.width == expected.width and actual.height == expected.height, label..": dimensions differ")
  assert(#actual.pixels == #expected.pixels, label..": pixel count differs")
  for index=1,#expected.pixels do
    assert(actual.pixels[index] == expected.pixels[index], label..": pixel "..index.." differs")
  end
end

local width, height, latent_dim = 8, 8, 8
local pixels = {}
for index=1,width*height*3 do pixels[index] = ((index*37+13)%256)/127.5-1 end
-- A P6 payload can begin with whitespace or '#'; these are pixels, not header.
pixels[1], pixels[2], pixels[3] = 10/127.5-1, 32/127.5-1, 35/127.5-1
local input = FS.join(directory, "input.ppm")
assert(PPM.write(input, pixels, width, height))
local source = assert(PPM.read(input))
local normalized = {}
for index,value in ipairs(source.pixels) do normalized[index] = value/127.5-1 end

local infer_script = "scripts/inferences/infer_vae_conv.lua"
local edit_script = "scripts/inferences/edit_vae_conv.lua"
local function roundtrip(checkpoint, name, expected, config)
  local raw = FS.join(directory, name..".raw")
  local decoded = FS.join(directory, name.."_decoded.ppm")
  local edited = FS.join(directory, name.."_edited.ppm")
  local checkpoint_option, input_option, output_option = "--checkpoint", "--input", "--output"
  if name == "2d" then checkpoint_option, input_option, output_option = "--ckpt", "--in", "--out" end
  local encode_args = {
    "--encode", checkpoint_option, checkpoint, input_option, input, output_option, raw, "--mem-gb", "1",
  }
  if config then encode_args[#encode_args+1], encode_args[#encode_args+2] = "--config", config end
  run(infer_script, name.."_encode", encode_args)
  local bytes = read_bytes(raw)
  assert(#bytes == latent_dim*4, name..": RAW must contain exactly CHW float32 values")
  for index=1,latent_dim do
    local value = string.unpack("<f", bytes, (index-1)*4+1)
    assert(value == value and math.abs(value) < math.huge, name..": latent is not finite")
  end
  local decode_args = {
    "--decode", checkpoint_option, checkpoint, input_option, raw, output_option, decoded, "--mem-gb", "1",
  }
  if config then decode_args[#decode_args+1], decode_args[#decode_args+2] = "--config", config end
  run(infer_script, name.."_decode", decode_args)
  local edit_args = {
    "--checkpoint", checkpoint, "--input", input, "--output", edited,
    "--headless", "--strength", "0", "--mix", "1", "--mem-gb", "1",
  }
  if config then edit_args[#edit_args+1], edit_args[#edit_args+2] = "--config", config end
  run(edit_script, name.."_edit", edit_args)
  same_image(decoded, expected, name..": decode versus original native graph")
  same_image(edited, decoded, name..": editor strength=0 versus encode/decode")
  return raw, decoded
end

local checkpoints, references = {}, {}
local legacy_safe, legacy_safe_config
for _,case in ipairs({
  {name="2d", mode="2d"},
  {name="3d", mode="3d"},
  {name="4d", mode="4d", prior=true},
  {name="2d_bilinear", mode="2d", upsample="bilinear_conv", norm="layernorm"},
  {name="2d_pixel_shuffle", mode="2d", upsample="pixel_shuffle", norm="layernorm"},
}) do
  local name, mode = case.name, case.mode
  local cfg = assert(Mimir.Architectures.default_config("vae_conv"))
  cfg.image_w, cfg.image_h, cfg.image_c = width, height, 3
  cfg.latent_w, cfg.latent_h, cfg.latent_c, cfg.latent_dim = 2, 2, 2, latent_dim
  cfg.base_channels, cfg.conv_mode, cfg.dtype = 4, mode, "float32"
  cfg.logvar_head_bias = mode ~= "2d"
  cfg.resnet, cfg.attention = mode == "3d", mode == "3d"
  cfg.attn_heads, cfg.attn_max_tokens = 2, 16
  cfg.use_skip_connections, cfg.use_encoder_prior, cfg.stochastic_latent = false, case.prior == true, false
  cfg.latent_spatial_context, cfg.latent_detail_path = mode ~= "2d", mode ~= "2d"
  cfg.latent_context_channels = 4
  cfg.enc_norm, cfg.dec_norm = mode == "2d" and "none" or "groupnorm", mode == "2d" and "none" or "groupnorm"
  if case.norm then cfg.enc_norm, cfg.dec_norm = case.norm, case.norm end
  if case.upsample then cfg.decoder_upsample = case.upsample end
  cfg.enc_gn_groups, cfg.dec_gn_groups = 2, 2
  cfg.text_cond = mode == "4d"
  cfg.vocab_size, cfg.seq_len, cfg.text_d_model, cfg.proj_dim = 32, 4, 4, 4
  assert(Mimir.Model.create("vae_conv", cfg))
  assert(Mimir.Model.allocate_params())
  assert(Mimir.Model.init_weights("he", 123))
  local checkpoint = FS.join(directory, "checkpoint_"..name)
  assert(Mimir.Serialization.save(checkpoint, "raw_folder", {
    save_optimizer=false, save_encoder=false, save_tokenizer=false,
  }))
  if case.prior then
    local manifest = assert(read_json(FS.join(checkpoint, "manifest.json")))
    local prior
    for _,entry in ipairs(manifest.tensor_index) do
      if entry.name == "vae_conv/z_prior_bias_weights" then prior = read_bytes(FS.join(checkpoint, entry.bin_file)) end
    end
    assert(prior and #prior == latent_dim*4, "The fixture must serialize a spatial CHW prior")
    local nonzero = false
    for index=1,latent_dim do
      local value = string.unpack("<f", prior, (index-1)*4+1)
      assert(value == value and math.abs(value) < math.huge, "Prior must be finite")
      if math.abs(value) > 1e-5 then nonzero = true end
    end
    assert(nonzero, "A zero prior would not test its application at decode time")
  end
  local reference = FS.join(directory, "reference_"..name..".ppm")
  local reconstructed = assert(Mimir.Model.forward(normalized, false))
  assert(#reconstructed == width*height*3, mode..": native inference output must be reconstruction only")
  for _,value in ipairs(reconstructed) do
    assert(value == value and math.abs(value) < math.huge, mode..": native reconstruction is not finite")
  end
  assert(PPM.write(reference, reconstructed, width, height))
  checkpoints[name], references[name] = checkpoint, reference
  roundtrip(checkpoint, name, reference)

  if mode == "4d" then
    local safe = FS.join(directory, "checkpoint_4d.safetensors")
    assert(Mimir.Serialization.save(safe, "safetensors", {
      save_optimizer=false, save_encoder=false, save_tokenizer=false,
    }))
    roundtrip(safe, "4d_safetensors", reference)
  end
  if name == "2d" then
    local safe = FS.join(directory, "checkpoint_2d.safetensors")
    assert(Mimir.Serialization.save(safe, "safetensors", {
      save_optimizer=false, save_encoder=false, save_tokenizer=false,
    }))
    local bytes = read_bytes(safe)
    local length = string.unpack("<I8", bytes)
    local header = assert(JSON.decode_json(bytes:sub(9, 8+length)))
    local architecture_entry = assert(header["model/architecture_json"], "The native SafeTensors fixture must contain its graph")
    local begin_offset, end_offset = architecture_entry.data_offsets[1], architecture_entry.data_offsets[2]
    header["model/architecture_json"] = nil
    -- SafeTensors requires contiguous payload ranges. Remove the metadata
    -- bytes as well, and retain each remaining tensor's exact binary values.
    for key,entry in pairs(header) do
      if key ~= "__metadata__" and entry.data_offsets[1] >= end_offset then
        entry.data_offsets[1] = entry.data_offsets[1]-(end_offset-begin_offset)
        entry.data_offsets[2] = entry.data_offsets[2]-(end_offset-begin_offset)
      end
    end
    local payload = bytes:sub(9+length)
    payload = payload:sub(1, begin_offset)..payload:sub(end_offset+1)
    local encoded = assert(JSON.encode_json(header))
    encoded = encoded..string.rep(" ", (8-#encoded%8)%8)
    legacy_safe = FS.join(directory, "legacy_without_architecture.safetensors")
    write_bytes(legacy_safe, string.pack("<I8", #encoded)..encoded..payload)
    legacy_safe_config = FS.join(directory, "legacy_safe_config.json")
    assert(write_json(legacy_safe_config, {model=cfg}))
  end
end

local missing_config_output = FS.join(directory, "missing_config.raw")
run(infer_script, "safe_missing_config", {
  "--encode", "--checkpoint", legacy_safe, "--input", input, "--output", missing_config_output,
  "--mem-gb", "1",
}, true)
assert(not FS.file_exists(missing_config_output), "A SafeTensors export without its graph requires --config")
roundtrip(legacy_safe, "2d_safe_external_config", references["2d"], legacy_safe_config)

-- The native serialized layers remain authoritative when older metadata omits
-- options added later, or carries a stale decoder normalization field.
for _,mode in ipairs({"2d", "4d"}) do
  local checkpoint = checkpoints[mode]
  local path = FS.join(checkpoint, "model", "architecture.json")
  local architecture = assert(read_json(path))
  local original = read_bytes(path)
  local manifest_path = FS.join(checkpoint, "manifest.json")
  local manifest_original = read_bytes(manifest_path)
  for _,key in ipairs({"model_config", "modelConfig"}) do
    local cfg = architecture[key]
    if type(cfg) == "table" then
      cfg.conv_mode, cfg.decoder_upsample = nil, nil
      cfg.logvar_head_bias = nil
      cfg.latent_spatial_context, cfg.latent_detail_path, cfg.latent_context_channels = nil, nil, nil
      cfg.dec_norm, cfg.dtype = "none", "F32"
    end
  end
  if mode == "4d" then
    -- Historical variance branches did not contain these feature/norm layers.
    -- Deterministic encoding needs mu, not newly initialized variance weights.
    for index=#architecture.layers,1,-1 do
      if tostring(architecture.layers[index].name):match("^vae_conv/enc/logvar_pre_") then
        table.remove(architecture.layers, index)
      end
    end
    local manifest = assert(read_json(manifest_path))
    for index=#manifest.tensor_index,1,-1 do
      if tostring(manifest.tensor_index[index].name):match("^vae_conv/enc/logvar_pre_") then
        table.remove(manifest.tensor_index, index)
      end
    end
    assert(write_json(manifest_path, manifest))
  end
  assert(write_json(path, architecture))
  roundtrip(checkpoint, mode.."_legacy_metadata", references[mode])
  write_bytes(path, original)
  write_bytes(manifest_path, manifest_original)
end

local source_output = FS.join(directory, "source_mix0.ppm")
run(edit_script, "source_mix0", {
  "--checkpoint", checkpoints["3d"], "--input", input, "--output", source_output,
  "--headless", "--strength", "1", "--mix", "0", "--mem-gb", "1",
})
same_image(source_output, input, "mix=0 must preserve RGB source pixels")

for _,script in ipairs({infer_script, edit_script}) do
  local name = script == infer_script and "infer_conv_mismatch" or "edit_conv_mismatch"
  local output = FS.join(directory, name..(script == infer_script and ".raw" or ".ppm"))
  local arguments = {"--checkpoint", checkpoints["3d"], "--input", input, "--output", output,
    "--conv-mode", "2d", "--mem-gb", "1"}
  arguments[#arguments+1] = script == infer_script and "--encode" or "--headless"
  run(script, name, arguments, true)
  assert(not FS.file_exists(output), name..": incompatible graph must not produce an output")
end

-- Partial native loads must reject missing trained tensors, including weights
-- used by the additional latent branches, before emitting a plausible image.
local manifest_path = FS.join(checkpoints["4d"], "manifest.json")
local manifest_original = read_bytes(manifest_path)
local manifest = assert(read_json(manifest_path))
local removed = false
for index,entry in ipairs(assert(manifest.tensor_index)) do
  if entry.name == "vae_conv/dec/detail/project_weights" then
    table.remove(manifest.tensor_index, index)
    removed = true
    break
  end
end
assert(removed, "The 4D fixture must contain detail reconstruction weights")
assert(write_json(manifest_path, manifest))
local missing_output = FS.join(directory, "missing_weights.ppm")
run(edit_script, "missing_weights", {
  "--checkpoint", checkpoints["4d"], "--input", input, "--output", missing_output,
  "--headless", "--mem-gb", "1",
}, true)
assert(not FS.file_exists(missing_output), "Missing weights must not produce an output")
write_bytes(manifest_path, manifest_original)

print(string.format("VAE_CONV_INFERENCE_COMPAT_OK: %d native CLI processes (%s)", subprocesses, directory))
