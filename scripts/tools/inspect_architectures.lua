---@diagnostic disable: undefined-global, undefined-field

-- Outil CLI d'inspection des architectures du framework.
--
-- Usage:
--   ./bin/mimir --lua scripts/tools/inspect_architectures.lua -- [options]
--
-- Options:
--   -a, --show-archs        Liste les architectures disponibles (+ dtypes)
--   -l, --list <arch>       Sélectionne une architecture par son nom
--   -p, --params            Affiche les paramètres de l'archi sélectionnée
--   --layers                Affiche les layers de l'archi sélectionnée
--   --stats                 Affiche les statistiques théoriques (params/flops)
--   -d, --dtypes            Liste les dtypes pris en charge par le framework
--   --json                  Export JSON complet du registre (toutes archs + params)
--   -h, --help              Affiche cette aide

-- log() peut être absent hors du binaire mimir : on fournit un fallback.
local log = rawget(_G, "log")
if type(log) ~= "function" then
  log = function(...)
    print(...)
  end
end

local Args = dofile("scripts/modules/args.lua")

-- ---------------------------------------------------------------------------
-- Couleurs ANSI (désactivées si NO_COLOR ou sortie non-TTY)
-- ---------------------------------------------------------------------------
local COLOR_ENABLED = true
do
  if os.getenv("NO_COLOR") ~= nil then
    COLOR_ENABLED = false
  end
end

local C = {
  reset = "\27[0m",
  bold = "\27[1m",
  dim = "\27[2m",
  red = "\27[31m",
  green = "\27[32m",
  yellow = "\27[33m",
  blue = "\27[34m",
  magenta = "\27[35m",
  cyan = "\27[36m",
  gray = "\27[90m",
}

local function colorize(s, ...)
  s = tostring(s)
  if not COLOR_ENABLED then return s end
  local codes = { ... }
  if #codes == 0 then return s end
  return table.concat(codes) .. s .. C.reset
end

-- ---------------------------------------------------------------------------
-- Helpers d'alignement (UTF-8) + tableau ASCII (style analyze_model.lua)
-- ---------------------------------------------------------------------------
local function utf8_len_safe(s)
  s = tostring(s or "")
  if type(_G.utf8) == "table" and type(_G.utf8.len) == "function" then
    local ok, n = pcall(_G.utf8.len, s)
    if ok and type(n) == "number" then return n end
  end
  local i, n, bytes = 1, 0, #s
  while i <= bytes do
    local c = s:byte(i)
    if not c then break end
    if c < 0x80 then i = i + 1
    elseif c < 0xE0 then i = i + 2
    elseif c < 0xF0 then i = i + 3
    else i = i + 4 end
    n = n + 1
  end
  return n
end

local function utf8_sub_chars(s, n_chars)
  s = tostring(s or "")
  n_chars = math.floor(tonumber(n_chars) or 0)
  if n_chars <= 0 then return "" end
  if type(_G.utf8) == "table" and type(_G.utf8.offset) == "function" then
    local ok, off = pcall(_G.utf8.offset, s, n_chars + 1)
    if ok and type(off) == "number" then return s:sub(1, off - 1) end
  end
  local i, bytes, count = 1, #s, 0
  while i <= bytes and count < n_chars do
    local c = s:byte(i)
    if not c then break end
    if c < 0x80 then i = i + 1
    elseif c < 0xE0 then i = i + 2
    elseif c < 0xF0 then i = i + 3
    else i = i + 4 end
    count = count + 1
  end
  return s:sub(1, i - 1)
end

local function trunc(s, max_len)
  s = tostring(s or "")
  max_len = math.floor(tonumber(max_len) or 80)
  if utf8_len_safe(s) <= max_len then return s end
  if max_len < 6 then return utf8_sub_chars(s, max_len) end
  return utf8_sub_chars(s, max_len - 3) .. "..."
end

local function pad_right(s, w)
  s = tostring(s or "")
  w = math.floor(tonumber(w) or utf8_len_safe(s))
  local len = utf8_len_safe(s)
  if len >= w then return s end
  return s .. string.rep(" ", w - len)
end

-- Construit un tableau ASCII coloré.
--   columns: { {key=, title=, align="left|right", max=, color=<ansi>}, ... }
--   rows:    { { [key]=value, ... }, ... }
local function make_table(columns, rows)
  local widths = {}
  for ci = 1, #columns do
    local col = columns[ci]
    widths[ci] = utf8_len_safe(col.title or col.key or ("col" .. ci))
  end
  for ri = 1, #rows do
    for ci = 1, #columns do
      local col = columns[ci]
      local s = trunc(rows[ri][col.key], col.max or 120)
      local w = utf8_len_safe(s)
      if w > widths[ci] then widths[ci] = w end
    end
  end

  local function sep(ch)
    local parts = { "+" }
    for ci = 1, #columns do
      parts[#parts + 1] = string.rep(ch, widths[ci] + 2)
      parts[#parts + 1] = "+"
    end
    return colorize(table.concat(parts), C.gray)
  end

  local bar = colorize("|", C.gray)
  local out = {}
  out[#out + 1] = sep("-")

  do
    local parts = { bar }
    for ci = 1, #columns do
      local col = columns[ci]
      local title = pad_right(col.title or col.key or "", widths[ci])
      parts[#parts + 1] = " " .. colorize(title, C.bold, C.cyan) .. " "
      parts[#parts + 1] = bar
    end
    out[#out + 1] = table.concat(parts)
  end

  out[#out + 1] = sep("=")

  for ri = 1, #rows do
    local row = rows[ri]
    local parts = { bar }
    for ci = 1, #columns do
      local col = columns[ci]
      local s = trunc(row[col.key], col.max or 120)
      local cell
      if (col.align or "left") == "right" then
        cell = string.rep(" ", math.max(0, widths[ci] - utf8_len_safe(s))) .. s
      else
        cell = pad_right(s, widths[ci])
      end
      if col.color then cell = colorize(cell, col.color) end
      parts[#parts + 1] = " " .. cell .. " "
      parts[#parts + 1] = bar
    end
    out[#out + 1] = table.concat(parts)
  end

  out[#out + 1] = sep("-")
  return table.concat(out, "\n")
end

-- Le module args.lua gère les flags longs (`--show-archs`, `--list`, `--params`).
-- Les flags courts (`-a`, `-l`, `-p`, `-h`) ne commencent pas par `--`, ils sont
-- donc collectés dans la liste des positionnels : on les analyse ici.
local function parse_flags(argv)
  local opts_long, pos = Args.parse(argv or {})

  local opts = {
    show_archs = Args.has(opts_long, "show-archs"),
    params     = Args.has(opts_long, "params"),
    dtypes     = Args.has(opts_long, "dtypes"),
    help       = Args.has(opts_long, "help"),
    layers     = Args.has(opts_long, "layers"),
    stats      = Args.has(opts_long, "stats"),
    json_out   = Args.has(opts_long, "json"),
    arch       = Args.get_str(opts_long, "list", nil),
  }

  -- Scan des formes courtes depuis les positionnels.
  for i = 1, #pos do
    local v = pos[i]
    if v == "-a" then
      opts.show_archs = true
    elseif v == "-p" then
      opts.params = true
    elseif v == "-d" then
      opts.dtypes = true
    elseif v == "-h" then
      opts.help = true
    elseif v == "-l" then
      opts.arch = pos[i + 1]
    end
  end

  return opts
end

local function print_usage()
  log(colorize("Usage: ", C.bold) .. "mimir --lua scripts/tools/inspect_architectures.lua -- [options]")
  log("")
  log(colorize("Options:", C.bold, C.cyan))
  log("  " .. colorize("-a, --show-archs", C.green) .. "        Liste les architectures disponibles (+ dtypes)")
  log("  " .. colorize("-l, --list <arch>", C.green) .. "       Sélectionne une architecture par son nom")
  log("  " .. colorize("-p, --params", C.green) .. "            Affiche les paramètres de l'archi sélectionnée")
  log("  " .. colorize("    --layers", C.green) .. "            Affiche les layers de l'archi sélectionnée")
  log("  " .. colorize("    --stats", C.green) .. "             Affiche les statistiques théoriques (params par layer)")
  log("  " .. colorize("-d, --dtypes", C.green) .. "            Liste les dtypes pris en charge par le framework")
  log("  " .. colorize("    --json", C.green) .. "              Export JSON complet du registre")
  log("  " .. colorize("-h, --help", C.green) .. "              Affiche cette aide")
end

-- Aplatit récursivement une config en lignes { key=<chemin>, value=<scalaire> }.
local function flatten_config(cfg, prefix, rows)
  rows = rows or {}
  prefix = prefix or ""
  local keys = {}
  for k in pairs(cfg) do
    keys[#keys + 1] = k
  end
  table.sort(keys, function(x, y) return tostring(x) < tostring(y) end)
  for _, k in ipairs(keys) do
    local v = cfg[k]
    local path = (prefix == "") and tostring(k) or (prefix .. "." .. tostring(k))
    if type(v) == "table" then
      flatten_config(v, path, rows)
    else
      local val
      if type(v) == "string" then
        val = '"' .. v .. '"'
      else
        val = tostring(v)
      end
      rows[#rows + 1] = { key = path, value = val, vtype = type(v) }
    end
  end
  return rows
end

-- Affiche une config sous forme de tableau coloré (clé / valeur / type).
local function print_config_table(cfg)
  local rows = flatten_config(cfg, "", {})
  if #rows == 0 then
    log(colorize("\t(aucun paramètre)", C.dim))
    return
  end
  local columns = {
    { key = "key", title = "Paramètre", align = "left", color = C.yellow, max = 40 },
    { key = "value", title = "Valeur", align = "left", color = C.green, max = 60 },
    { key = "vtype", title = "Type", align = "left", color = C.magenta, max = 12 },
  }
  log(make_table(columns, rows))
end

-- dtype effectif d'une architecture: champ `config.dtype` s'il est présent,
-- sinon le défaut global du framework ("float32").
local function arch_default_dtype(entry)
  if type(entry) == "table" and type(entry.config) == "table" then
    local dt = entry.config.dtype
    if type(dt) == "string" and dt ~= "" then
      return dt
    end
  end
  return "float32"
end

local function list_archs()
  -- Lecture complète du registry : nom + description + config par défaut.
  local entries, err = Mimir.Architectures.info()
  if type(entries) ~= "table" then
    log(colorize("[ERROR] ", C.bold, C.red) .. tostring(err))
    return nil
  end
  log("\n" .. colorize("* Architectures supportées par le Framework :", C.bold, C.blue))
  local rows = {}
  for _, entry in ipairs(entries) do
    rows[#rows + 1] = {
      name = tostring(entry.name),
      dtype = arch_default_dtype(entry),
      description = (type(entry.description) == "string") and entry.description or "",
    }
  end
  local columns = {
    { key = "name", title = "Architecture", align = "left", color = C.cyan, max = 32 },
    { key = "dtype", title = "dtype défaut", align = "left", color = C.green, max = 12 },
    { key = "description", title = "Description", align = "left", color = nil, max = 72 },
  }
  log(make_table(columns, rows))
  return entries
end

-- Affiche le tableau des dtypes disponibles dans le framework.
local function list_dtypes()
  local dtypes, err = Mimir.Architectures.dtypes()
  if type(dtypes) ~= "table" then
    log(colorize("[ERROR] ", C.bold, C.red) .. tostring(err))
    return nil
  end
  log("\n" .. colorize("* dtypes pris en charge par le Framework :", C.bold, C.blue))
  local rows = {}
  for _, dt in ipairs(dtypes) do
    rows[#rows + 1] = {
      name = tostring(dt.name),
      kind = tostring(dt.kind),
      bytes = tostring(dt.bytes),
      aliases = tostring(dt.aliases),
    }
  end
  local columns = {
    { key = "name", title = "dtype", align = "left", color = C.cyan, max = 12 },
    { key = "kind", title = "Famille", align = "left", color = C.magenta, max = 8 },
    { key = "bytes", title = "Octets", align = "right", color = C.green, max = 6 },
    { key = "aliases", title = "Alias acceptés", align = "left", color = C.yellow, max = 40 },
  }
  log(make_table(columns, rows))
  return dtypes
end

local function show_params(arch)
  -- Lecture complète de l'entrée du registry pour cette architecture.
  local entry, err = Mimir.Architectures.info(arch)
  if type(entry) ~= "table" then
    log(colorize("[ERROR] ", C.bold, C.red) .. "info(" .. tostring(arch) .. ") a échoué: " .. tostring(err))
    return false
  end
  log("\n" .. colorize("* Architecture ", C.bold, C.blue) .. colorize("'" .. tostring(entry.name) .. "'", C.bold, C.cyan) .. colorize(" :", C.bold, C.blue))
  if type(entry.description) == "string" and entry.description ~= "" then
    log("  " .. colorize("description: ", C.bold) .. entry.description)
  end
  log("  " .. colorize("dtype par défaut: ", C.bold) .. colorize(arch_default_dtype(entry), C.green))
  log("  " .. colorize("paramètres (config par défaut) :", C.bold))
  if type(entry.config) == "table" then
    print_config_table(entry.config)
  else
    log(colorize("\t(aucune config)", C.dim))
  end
  return true
end

-- Instancie l'architecture dans le contexte courant et retourne true/false.
local function instantiate_arch(arch)
  local ok, err = Mimir.Model.create(arch)
  if not ok then
    log(colorize("[ERROR] ", C.bold, C.red) .. "Impossible d'instancier '" .. arch .. "': " .. tostring(err))
    return false
  end
  Mimir.Model.allocate_params()
  return true
end

-- --layers: affiche le tableau des layers du modèle courant.
local function show_layers(arch)
  log("\n" .. colorize("* Layers – ", C.bold, C.blue) .. colorize(arch, C.bold, C.cyan))
  if not instantiate_arch(arch) then return false end
  local layers = Mimir.Model.get_layers()
  if type(layers) ~= "table" or #layers == 0 then
    log(colorize("\t(aucun layer)", C.dim))
    return true
  end

  -- Construire la colonne "shape" à partir des champs disponibles.
  local function layer_shape(la)
    local t = la.type or ""
    if t == "Linear" then
      if la.seq_len > 0 then
        return string.format("[%d, %d→%d]", la.seq_len, la.in_features, la.out_features)
      end
      return string.format("[%d→%d]", la.in_features, la.out_features)
    elseif t == "Embedding" then
      return string.format("[%d×%d]", la.vocab_size, la.in_features > 0 and la.in_features or la.embed_dim)
    elseif t == "Conv2d" or t == "ConvTranspose2d" then
      return string.format("[%d→%d, k%d s%d p%d]", la.in_channels, la.out_channels, la.kernel_size, la.stride, la.padding)
    elseif t == "GroupNorm" or t == "LayerNorm" then
      return string.format("[c%d]", la.in_channels > 0 and la.in_channels or la.in_features)
    elseif t == "MultiHeadAttention" or t == "SelfAttention" or t == "CrossAttention" then
      return string.format("[seq%d, d%d, h%d]", la.seq_len, la.embed_dim, la.num_heads)
    elseif t == "Reshape" then
      return "→reshape"
    elseif t == "Permute" then
      return "→permute"
    elseif t == "Concat" then
      return "→concat"
    end
    return ""
  end

  local rows = {}
  for _, la in ipairs(layers) do
    local inputs_str = table.concat(la.inputs or {}, ", ")
    rows[#rows + 1] = {
      idx    = tostring(la.index),
      name   = tostring(la.name),
      ltype  = tostring(la.type),
      shape  = layer_shape(la),
      params = la.param_count > 0 and tostring(la.param_count) or "-",
      output = tostring(la.output),
      inputs = inputs_str,
    }
  end
  local columns = {
    { key = "idx",    title = "#",       align = "right",  color = C.gray,    max = 5  },
    { key = "name",   title = "Nom",     align = "left",   color = C.yellow,  max = 52 },
    { key = "ltype",  title = "Type",    align = "left",   color = C.cyan,    max = 22 },
    { key = "shape",  title = "Shape",   align = "left",   color = C.magenta, max = 28 },
    { key = "params", title = "Params",  align = "right",  color = C.green,   max = 12 },
    { key = "output", title = "Output",  align = "left",   color = C.dim,     max = 36 },
  }
  log(make_table(columns, rows))
  return true
end

-- --stats: statistiques théoriques (params par layer + totaux par type).
local function show_stats(arch)
  log("\n" .. colorize("* Stats – ", C.bold, C.blue) .. colorize(arch, C.bold, C.cyan))
  if not instantiate_arch(arch) then return false end

  local layers = Mimir.Model.get_layers()
  local total_params = Mimir.Model.total_params()
  if type(layers) ~= "table" then layers = {} end

  -- Agrégation par type.
  local by_type = {}       -- type → {count, params}
  local top_layers = {}    -- top-20 par params

  for _, la in ipairs(layers) do
    local t = la.type or "?"
    if not by_type[t] then by_type[t] = { count = 0, params = 0 } end
    by_type[t].count  = by_type[t].count  + 1
    by_type[t].params = by_type[t].params + la.param_count
    if la.param_count > 0 then
      top_layers[#top_layers + 1] = la
    end
  end

  -- Top-20 layers par params desc.
  table.sort(top_layers, function(a, b) return a.param_count > b.param_count end)
  local top_n = math.min(20, #top_layers)

  log(colorize("  Total paramètres: ", C.bold) .. colorize(string.format("%d", total_params), C.green)
    .. colorize(string.format("  (%.2f M)", total_params / 1e6), C.gray))
  log(colorize("  Total layers: ", C.bold) .. colorize(tostring(#layers), C.yellow))
  log("")

  -- Tableau par type.
  local type_rows = {}
  for t, v in pairs(by_type) do
    type_rows[#type_rows + 1] = {
      ltype  = t,
      count  = tostring(v.count),
      params = string.format("%d", v.params),
      pct    = total_params > 0
               and string.format("%.1f%%", 100.0 * v.params / total_params)
               or  "0%",
    }
  end
  table.sort(type_rows, function(a, b)
    return tonumber(a.params) > tonumber(b.params)
  end)
  log(colorize("  Répartition par type :", C.bold))
  log(make_table(
    {
      { key = "ltype",  title = "Type",      align = "left",  color = C.cyan,    max = 24 },
      { key = "count",  title = "Nb",        align = "right", color = C.yellow,  max = 6  },
      { key = "params", title = "Paramètres",align = "right", color = C.green,   max = 14 },
      { key = "pct",    title = "%",         align = "right", color = C.magenta, max = 7  },
    },
    type_rows
  ))

  -- Top N layers.
  if top_n > 0 then
    log(colorize(string.format("\n  Top-%d layers (params desc) :", top_n), C.bold))
    local top_rows = {}
    for i = 1, top_n do
      local la = top_layers[i]
      top_rows[#top_rows + 1] = {
        rank   = tostring(i),
        name   = tostring(la.name),
        ltype  = tostring(la.type),
        params = string.format("%d", la.param_count),
        pct    = total_params > 0
                 and string.format("%.1f%%", 100.0 * la.param_count / total_params)
                 or  "0%",
      }
    end
    log(make_table(
      {
        { key = "rank",   title = "#",          align = "right", color = C.gray,    max = 4  },
        { key = "name",   title = "Layer",       align = "left",  color = C.yellow,  max = 52 },
        { key = "ltype",  title = "Type",        align = "left",  color = C.cyan,    max = 22 },
        { key = "params", title = "Paramètres",  align = "right", color = C.green,   max = 14 },
        { key = "pct",    title = "%",           align = "right", color = C.magenta, max = 7  },
      },
      top_rows
    ))
  end
  return true
end

-- --json: exporte tout le registre en JSON (stdout).
local function export_json()
  local entries, err = Mimir.Architectures.info()
  if type(entries) ~= "table" then
    log(colorize("[ERROR] ", C.bold, C.red) .. tostring(err))
    return false
  end

  -- Sérialisation JSON minimale (pas de dépendance externe).
  local function to_json(v, indent)
    indent = indent or 0
    local sp = string.rep("  ", indent)
    local sp2 = string.rep("  ", indent + 1)
    local t = type(v)
    if t == "nil" then
      return "null"
    elseif t == "boolean" then
      return tostring(v)
    elseif t == "number" then
      if v == math.floor(v) and math.abs(v) < 2^53 then
        return string.format("%d", v)
      end
      return string.format("%.10g", v)
    elseif t == "string" then
      -- Échapper les caractères JSON.
      return '"' .. v:gsub('\\', '\\\\'):gsub('"', '\\"')
                     :gsub('\n', '\\n'):gsub('\r', '\\r')
                     :gsub('\t', '\\t') .. '"'
    elseif t == "table" then
      -- Détecter si c'est un tableau.
      local is_array = (#v > 0)
      if is_array then
        local parts = {}
        for i = 1, #v do
          parts[#parts + 1] = sp2 .. to_json(v[i], indent + 1)
        end
        return "[\n" .. table.concat(parts, ",\n") .. "\n" .. sp .. "]"
      else
        local keys = {}
        for k in pairs(v) do keys[#keys + 1] = k end
        table.sort(keys, function(a, b) return tostring(a) < tostring(b) end)
        local parts = {}
        for _, k in ipairs(keys) do
          parts[#parts + 1] = sp2 .. '"' .. tostring(k) .. '": ' .. to_json(v[k], indent + 1)
        end
        return "{\n" .. table.concat(parts, ",\n") .. "\n" .. sp .. "}"
      end
    end
    return '"[unsupported:' .. t .. ']"'
  end

  -- Construire la structure complète.
  local registry = {}
  for _, entry in ipairs(entries) do
    -- Optionnel: instancier pour obtenir le total_params réel.
    local total = 0
    local layer_list = {}
    local ok_inst = pcall(function()
      Mimir.Model.create(entry.name)
      Mimir.Model.allocate_params()
      total = Mimir.Model.total_params()
      local layers = Mimir.Model.get_layers()
      if type(layers) == "table" then
        for _, la in ipairs(layers) do
          layer_list[#layer_list + 1] = {
            index       = la.index,
            name        = la.name,
            type        = la.type,
            param_count = la.param_count,
            output      = la.output,
            inputs      = la.inputs,
          }
        end
      end
    end)

    registry[#registry + 1] = {
      name         = entry.name,
      description  = entry.description,
      dtype        = arch_default_dtype(entry),
      config       = entry.config,
      total_params = ok_inst and total or nil,
      layers       = ok_inst and layer_list or nil,
    }
  end

  io.write(to_json({ registry = registry }, 0))
  io.write("\n")
  io.flush()
  return true
end

local opts = parse_flags(arg)

if opts.help then
  print_usage()
  return
end

-- --json : export complet, indépendant des autres flags.
if opts.json_out then
  export_json()
  return
end

-- Aucun flag utile : on affiche l'aide.
if not opts.show_archs and not opts.arch and not opts.dtypes then
  print_usage()
  return
end

if opts.show_archs then
  list_archs()
  list_dtypes()
elseif opts.dtypes then
  list_dtypes()
end

if opts.arch then
  if not opts.params and not opts.layers and not opts.stats then
    -- -l sans aucune sous-option : confirmer l'existence + hint.
    local entry = Mimir.Architectures.info(opts.arch)
    if type(entry) == "table" then
      log("\n" .. colorize("Architecture '" .. opts.arch .. "' disponible.", C.green)
        .. " Options: " .. colorize("-p", C.bold) .. " params · "
        .. colorize("--layers", C.bold) .. " layers · "
        .. colorize("--stats", C.bold) .. " stats")
    else
      log("\n" .. colorize("[ERROR] ", C.bold, C.red) .. "Architecture inconnue: " .. tostring(opts.arch))
    end
    return
  end

  if opts.params  then show_params(opts.arch)  end
  if opts.layers  then show_layers(opts.arch)  end
  if opts.stats   then show_stats(opts.arch)   end
end