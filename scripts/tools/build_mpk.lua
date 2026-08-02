---@diagnostic disable: undefined-global

-- Build an MPK file (Mimir Package Template).
--
-- Usage:
--   ./bin/mimir --lua scripts/tools/build_mpk.lua -- \
--     --name my_vae_pack \
--     --type vae_conv \
--     --author bri45 \
--     --description "VAEConv baseline package" \
--     --out exports/my_vae_pack.mpk

local Args = dofile("scripts/modules/args.lua")
local FS = dofile("scripts/modules/fs.lua")
local MPK = dofile("scripts/modules/mpk.lua")
local MPKLayers = dofile("scripts/modules/mpk_layers.lua")

local function log(...)
  local out = {}
  for i = 1, select("#", ...) do
    out[#out + 1] = tostring(select(i, ...))
  end
  io.stdout:write(table.concat(out, " ") .. "\n")
end

local function die(msg)
  io.stderr:write("[build_mpk] " .. tostring(msg) .. "\n")
  os.exit(1)
end

local function print_usage()
  log("Usage:")
  log("  mimir --lua scripts/tools/build_mpk.lua -- --name <name> --type <arch> --out <file.mpk> [options]")
  log("")
  log("Options:")
  log("  --name <string>               Package name (required)")
  log("  --type <architecture>         Model architecture type for registry load (required)")
  log("  --author <string>             Header author (default: unknown)")
  log("  --created-at <iso8601>        Header created_at (default: current UTC)")
  log("  --modifiable / --no-modifiable Header modifiable flag (default: true)")
  log("  --viz / --no-viz              Header viz_specified flag (default: false)")
  log("  --config-json <path>          Base config JSON source")
  log("  --from-registry               Use Mimir.Architectures.default_config(type)")
  log("  --structure-json <path>       Model structure JSON source")
  log("  --template <name>             model_structure template: vae_conv|unet|ponyxl_ddpm|auto")
  log("  --description <text>          Human description text")
  log("  --description-file <path>     Description text file")
  log("  (output)                      Always writes modern Visu-like MPK pseudocode")
  log("  --out <path.mpk>              Output MPK path (required, .mpk)")
  log("  --compile [path.mpk.bin]      Also emit opaque optimized binary-v4")
  log("  --help                        Show this help")
end

local function read_json_or_die(path, label)
  local obj, err = MPK.read_json_file(path)
  if type(obj) ~= "table" then
    die((label or "json") .. " read failed: " .. tostring(err or path))
  end
  return obj
end

local function read_text_or_die(path, label)
  local txt, err = MPK.read_text_file(path)
  if txt == nil then
    die((label or "text") .. " read failed: " .. tostring(err or path))
  end
  return txt
end

local opts = Args.parse(arg) or {}

if Args.has(opts, "help") then
  print_usage()
  return
end
if Args.has(opts, "j" .. "son") or Args.has(opts, "bin" .. "ary") then
  die("--json/--binary removed: write pseudocode, then use compile_mpk.lua for binary output")
end

local name = Args.get_str(opts, "name", "")
local model_type = Args.get_str(opts, "type", "")
local out_path = Args.get_str(opts, "out", "")

if name == "" then die("missing --name") end
if model_type == "" then die("missing --type") end
if out_path == "" then die("missing --out") end
if not out_path:lower():match("%.mpk$") then die("--out must end with .mpk") end

local author = Args.get_str(opts, "author", "unknown")
local created_at = Args.get_str(opts, "created-at", nil)
local modifiable = Args.get_bool(opts, "modifiable", true)
local viz_specified = Args.get_bool(opts, "viz", false)
local description = Args.get_str(opts, "description", "")
local description_file = Args.get_str(opts, "description-file", "")
if description_file ~= "" then
  description = read_text_or_die(description_file, "description-file")
end

local base_config = {}
local cfg_json = Args.get_str(opts, "config-json", "")
local from_registry = Args.get_bool(opts, "from-registry", false)

if cfg_json ~= "" then
  base_config = read_json_or_die(cfg_json, "config-json")
elseif from_registry then
  if type(_G.Mimir) ~= "table" or type(Mimir.Architectures) ~= "table" then
    die("--from-registry requires running through ./bin/mimir")
  end
  local cfg, err = Mimir.Architectures.default_config(model_type)
  if type(cfg) ~= "table" then
    die("default_config(" .. tostring(model_type) .. ") failed: " .. tostring(err))
  end
  local cfg_tbl = {}
  local cfg_any = cfg or {}
  for k, v in pairs(cfg_any) do
    cfg_tbl[k] = v
  end
  base_config = cfg_tbl
end

local model_structure = {
  architecture = model_type,
  generated_by = "scripts/tools/build_mpk.lua",
}

local structure_json = Args.get_str(opts, "structure-json", "")
local template_name = Args.get_str(opts, "template", "auto")
if structure_json ~= "" then
  model_structure = read_json_or_die(structure_json, "structure-json")
elseif template_name ~= "" then
  if template_name == "auto" then
    model_structure = MPK.model_structure_template(model_type)
  else
    model_structure = MPK.model_structure_template(template_name)
  end
  model_structure.architecture = model_type
  model_structure.generated_by = "scripts/tools/build_mpk.lua"
elseif type(_G.Mimir) == "table" and type(Mimir.Architectures) == "table" then
  local entry = Mimir.Architectures.info(model_type)
  if type(entry) == "table" then
    model_structure = {
      architecture = model_type,
      registry_name = entry.name,
      registry_description = entry.description,
      default_config = entry.config,
      generated_by = "scripts/tools/build_mpk.lua",
    }
  end
end

local ok_graph, err_graph = MPKLayers.normalize_graph_in_place(model_structure)
if not ok_graph then
  die("graph validation failed: " .. tostring(err_graph))
end

local pkg, err_build = MPK.build({
  name = name,
  type = model_type,
  author = author,
  created_at = created_at,
  modifiable = modifiable,
  viz_specified = viz_specified,
  base_config = base_config,
  model_structure = model_structure,
  description = description,
  container = "pseudocode",
})
if not pkg then
  die("build failed: " .. tostring(err_build))
end

local parent = FS.dirname(out_path)
if parent and parent ~= "" then
  FS.mkdir_p(parent)
end

local ok_write, err_write = MPK.write(out_path, pkg)
if not ok_write then
  die("write failed: " .. tostring(err_write))
end

local compiled_path = nil
if opts.compile ~= nil and opts.compile ~= false then
  compiled_path = type(opts.compile) == "string" and opts.compile or
    (out_path .. ".bin")
  if not compiled_path:lower():match("%.mpk%.bin$") then
    die("--compile output must end with .mpk.bin")
  end
  local compiled_parent = FS.dirname(compiled_path)
  if compiled_parent and compiled_parent ~= "" then FS.mkdir_p(compiled_parent) end
  local ok_compile, err_compile = MPK.compile(out_path, compiled_path)
  if not ok_compile then die("compile failed: " .. tostring(err_compile)) end
end

log("[build_mpk] OK")
log("  out:        " .. out_path)
log("  name:       " .. tostring(pkg.header.name))
log("  type:       " .. tostring(pkg.header.type))
log("  author:     " .. tostring(pkg.header.author))
log("  created_at: " .. tostring(pkg.header.created_at))
log("  modifiable: " .. tostring(pkg.header.modifiable))
log("  viz:        " .. tostring(pkg.header.viz_specified))
log("  container:  " .. tostring(pkg.container))
if type(pkg.header.checksum) == "table" then
  log("  checksum:   " .. tostring(pkg.header.checksum.algorithm) .. ":" .. tostring(pkg.header.checksum.value))
end
log("  size(bytes):" .. tostring(pkg.header.size))
if compiled_path then log("  binary-v4:  " .. compiled_path) end
log("")
log("Registry load example:")
log("  ./bin/mimir --lua scripts/tools/load_mpk.lua -- --in " .. out_path .. " --create")
