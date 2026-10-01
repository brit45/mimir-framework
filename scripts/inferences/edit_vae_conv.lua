-- Éditeur latent VAEConv natif Mímir, avec UI commune Qt / GTK / SFML / Web.
local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
Help.auto_exit_help({
  script_path = "scripts/inferences/edit_vae_conv.lua",
  description = "Édition d'image par modification du latent VAEConv. Export PPM RGB. Backend Viz : "..Mimir.Viz.backend(),
  options = {
    "--checkpoint PATH : checkpoint raw_folder VAEConv entraîné (requis)",
    "--input PATH : image PNG/JPEG/BMP ou PPM P6/P3 (requise)",
    "--output PATH : image PPM à enregistrer (défaut outputs/edited.ppm)",
    "--strength N : décalage initial du canal latent (défaut 0)",
    "--channel N : canal latent, de 1 à latent_c (défaut 1)",
    "--mix N : mélange source/résultat VAE, de 0 à 1 (défaut 1)",
    "--headless : calculer et enregistrer une fois, sans interface",
    "--mem-gb N : budget mémoire (défaut 8 Go)",
    "Boutons : force ±, canal suivant, mélange ±, réinitialiser, enregistrer.",
    "H : aide et backend actif. Le backend est sélectionné à la compilation.",
    "Les canaux latents ne sont pas des commandes sémantiques (couleur, objet...).",
  },
  examples = {"./bin/mimir --lua scripts/inferences/edit_vae_conv.lua -- --checkpoint checkpoint/vae/epoch_0010 --input photo.ppm --output outputs/edition.ppm"},
})
local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")
local Ckpt = dofile(ROOTWORK.."/scripts/modules/checkpoint_resume.lua")
local PPM = dofile(ROOTWORK.."/scripts/modules/image_ppm.lua")
local opts = Args.parse(arg) or {}
local allowed = {checkpoint=true,input=true,output=true,strength=true,channel=true,mix=true,headless=true,["mem-gb"]=true}
for key in pairs(opts) do assert(allowed[key], "Option inconnue : "..tostring(key)) end
local function number(key, default)
  local value = opts[key] == nil and default or tonumber(opts[key])
  assert(value and value == value and math.abs(value) < math.huge, "Nombre invalide : "..key)
  return value
end
assert(type(opts.checkpoint) == "string", "--checkpoint requis")
assert(type(opts.input) == "string", "--input requis")
local output = opts.output or "outputs/edited.ppm"
assert(type(output) == "string" and output:lower():match("%.ppm$"), "--output doit être un fichier .ppm")
assert(output ~= opts.input, "Choisir une sortie distincte de l'image source")
local checkpoint = assert(Ckpt.resolve_dir(opts.checkpoint), "Checkpoint raw_folder introuvable")
local architecture = read_json(FS.join(checkpoint, "model", "architecture.json"))
local cfg = assert(architecture.model_config or architecture.modelConfig, "model_config absent")
assert(cfg.type == "vae_conv", "Checkpoint vae_conv attendu")
for _, key in ipairs({"image_w","image_h","image_c","latent_w","latent_h","latent_c"}) do
  assert(type(cfg[key]) == "number" and cfg[key] > 0 and cfg[key] == math.floor(cfg[key]), "Dimension invalide : "..key)
end
assert(cfg.image_c == 3, "PPM exige image_c=3")
cfg.latent_dim = cfg.latent_w * cfg.latent_h * cfg.latent_c
cfg.stochastic_latent = false
cfg.dtype = ({F32="float32",F16="float16",BF16="bfloat16"})[tostring(cfg.dtype):upper()] or cfg.dtype
-- La topologie sérialisée prévaut sur les anciens champs de configuration.
cfg.dec_norm = "none"
for _, layer in ipairs(architecture.layers or {}) do
  if tostring(layer.name):match("^vae_conv/dec/") then
    local kind = tostring(layer.type):lower()
    if kind == "groupnorm" then cfg.dec_norm = "groupnorm"; break end
    if kind == "layernorm" then cfg.dec_norm = "layernorm" end
  end
end
local strength, channel, mix = number("strength", 0), number("channel", 1), number("mix", 1)
assert(math.abs(strength) <= 10, "--strength doit être entre -10 et 10")
assert(channel >= 1 and channel <= cfg.latent_c and channel == math.floor(channel), "--channel hors limites")
assert(mix >= 0 and mix <= 1, "--mix doit être entre 0 et 1")
local headless = Args.get_bool(opts, "headless", false)
assert(headless or Mimir.Viz.backend() ~= "NONE",
  "Viz désactivée : compiler avec -DMIMIR_VIZ_BACKEND=WEB (ou QT/GTK/SFML), ou utiliser --headless")
local mem = number("mem-gb", 8)
assert(mem > 0, "--mem-gb doit être positif")
assert(Mimir.Allocator.configure({max_ram_gb=mem, enable_compression=true}))
local function load_model(name)
  assert(Mimir.Model.create(name, cfg))
  assert(Mimir.Model.allocate_params())
  -- Le décodeur charge un sous-ensemble du checkpoint complet. Vérifier
  -- explicitement chaque poids attendu avant le chargement partiel.
  local manifest = read_json(FS.join(checkpoint, "manifest.json"))
  local tensors = {}
  for _, entry in ipairs(assert(manifest.tensor_index, "tensor_index absent")) do tensors[entry.name] = entry end
  for _, layer in ipairs(Mimir.Model.get_layers()) do
    if layer.param_count > 0 and not layer.name:match("/zero_enc_skip$") then
      local entry = assert(tensors[layer.name.."_weights"], "Poids manquants : "..layer.name)
      local metadata = read_json(FS.join(checkpoint, entry.json_file))
      local count = 1
      for _, size in ipairs(assert(metadata.shape, "shape absent")) do count = count * size end
      assert(count == layer.param_count, "Topologie incompatible : "..layer.name)
      assert(({F32=true,F16=true,BF16=true,F64=true,float32=true,float16=true,bfloat16=true,float64=true})[metadata.dtype], "dtype non supporté")
      assert(type(metadata.data_file)=="string", "data_file absent")
    end
  end
  assert(Mimir.Serialization.load(checkpoint, "raw_folder", {
    load_encoder=false, load_tokenizer=false, load_optimizer=false,
    strict_mode=name=="vae_conv", validate_checksums=true,
  }))
end
local image
if opts.input:lower():match("%.ppm$") then
  image = assert(PPM.read(opts.input))
else
  local loaded = assert(Mimir.IO.read_image_rgb_u8(opts.input,cfg.image_w,cfg.image_h))
  image = {pixels=loaded.image,width=loaded.width,height=loaded.height}
end
local source = image.pixels
if image.width ~= cfg.image_w or image.height ~= cfg.image_h then
  source = PPM.resize(source, image.width, image.height, cfg.image_w, cfg.image_h)
end
local input = {}
for i, value in ipairs(source) do input[i] = value / 127.5 - 1 end
load_model("vae_conv")
local packed = assert(Mimir.Model.forward(input, false))
local image_dim, latent_dim = #input, cfg.latent_dim
assert(#packed == image_dim + 2 * latent_dim, "Sortie VAEConv inattendue")
local original = {}
for i = 1, latent_dim do original[i] = packed[image_dim+i] end
packed = nil
-- Seul le décodeur reste chargé durant les interactions.
load_model("vae_conv_decode")
local result, preview
local function compute()
  local latent, plane = {}, cfg.latent_w * cfg.latent_h
  for i = 1, latent_dim do
    latent[i] = original[i] + (math.floor((i-1)/plane)+1 == channel and strength or 0)
  end
  local decoded = assert(Mimir.Model.forward(latent, false))
  assert(#decoded == image_dim, "Sortie décodeur inattendue")
  result, preview = {}, {}
  for i = 1, image_dim do
    local value = input[i] * (1-mix) + decoded[i] * mix
    assert(value == value and math.abs(value) < math.huge, "Sortie non finie")
    result[i] = math.max(-1, math.min(1, value))
    preview[i] = math.floor((result[i]+1)*127.5+0.5)
  end
end
local function save()
  local directory = FS.dirname(output)
  if directory and directory ~= "" then FS.mkdir_p(directory) end
  assert(PPM.write(output, result, cfg.image_w, cfg.image_h))
  print("Image enregistrée : "..output)
end
compute()
if headless then save(); return end
local UI = dofile(ROOTWORK.."/scripts/modules/viz_ui.lua")
assert(Mimir.Viz.create({visualization={enabled=true,window_width=1200,window_height=800,
  window_title="Mímir • Éditeur VAEConv",fps_limit=30,image_grid_cols=2,image_grid_rows=1}}))
local view_w, view_h = 1200, 800
local function scene(initial)
  local cols = math.max(1, math.floor((view_w-24)/167))
  local top = 32 + math.ceil(7/cols)*50
  local panels = {}
  for id = 0, 6 do panels[#panels+1] = {id=id,visible=id==2} end
  panels[3] = {id=2,visible=true,x=12,y=top,w=math.max(220,view_w-24),h=math.max(140,view_h-top-20),
    title=string.format("Source / éditions • Canal %d/%d • Force %.2f • Mélange %.0f%%",channel,cfg.latent_c,strength,mix*100)}
  local controls = {}
  for i, entry in ipairs({{"less","Force −"},{"more","Force +"},{"channel","Canal suivant"},
      {"mix_less","Mélange −"},{"mix_more","Mélange +"},{"reset","Réinitialiser"},{"save","Enregistrer"}}) do
    controls[#controls+1] = {id=entry[1],label=entry[2],x=12+((i-1)%cols)*167,y=20+math.floor((i-1)/cols)*50,w=155,h=38}
  end
  if not initial then panels = {{id=2,title=panels[3].title}} end
  return {panels=panels,controls=controls,image_size=math.max(64,math.min(512,math.floor((view_w-80)/2))),help="Force : ±0.1 sur un canal latent. Mélange : source ↔ VAE. Enregistrer : sortie PPM."}
end
local ui = UI.new(scene(true))
local function publish()
  assert(Mimir.Viz.add_image(source,cfg.image_w,cfg.image_h,3,"Source"))
  assert(Mimir.Viz.add_image(preview,cfg.image_w,cfg.image_h,3,
    string.format("Canal %d • Force %.2f • Mélange %.0f%%",channel,strength,mix*100)))
  ui:configure(scene())
end
local function edit(fn)
  return function() fn(); compute(); publish() end
end
ui:on("less",edit(function() strength=math.max(-10,strength-0.1) end))
ui:on("more",edit(function() strength=math.min(10,strength+0.1) end))
ui:on("channel",edit(function() channel=channel%cfg.latent_c+1 end))
ui:on("mix_less",edit(function() mix=math.max(0,mix-0.1) end))
ui:on("mix_more",edit(function() mix=math.min(1,mix+0.1) end))
ui:on("reset",edit(function() strength=0; channel=1; mix=1 end))
ui:on("save",function()
  save()
  ui:configure({panels={{id=2,title="Enregistré : "..output}}})
end)
local function adapt_window(event)
  if event.width and (view_w ~= event.width or view_h ~= event.height) then
    view_w, view_h = event.width, event.height
    ui:configure(scene(true))
  end
end
ui:on("resize", adapt_window)
ui:on("configured", adapt_window)
publish()
print("Viz : "..Mimir.Viz.backend().." • H : aide • Enregistrer pour écrire la sortie")
local ok, err = xpcall(function()
  while Mimir.Viz.is_open() do ui:dispatch(30) end
end, debug.traceback)
Mimir.Viz.set_enabled(false)
if not ok then error(err) end
