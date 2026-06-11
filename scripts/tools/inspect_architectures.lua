---@diagnostic disable: undefined-global, undefined-field

-- Outil CLI d'inspection des architectures du framework.
--
-- Usage:
--   ./bin/mimir --lua scripts/tools/inspect_architectures.lua -- [options]
--
-- Options:
--   -a, --show-archs        Liste les architectures disponibles dans le framework.
--   -l, --list <arch>       Sélectionne une architecture par son nom.
--   -p, --params            Affiche les paramètres (config par défaut) de l'archi
--                           sélectionnée via -l/--list.
--   -h, --help              Affiche cette aide.
--
-- Exemples:
--   ... -- -a
--   ... -- --show-archs
--   ... -- -l vae_conv -p
--   ... -- --list vae_conv --params

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
    params = Args.has(opts_long, "params"),
    dtypes = Args.has(opts_long, "dtypes"),
    help = Args.has(opts_long, "help"),
    arch = Args.get_str(opts_long, "list", nil),
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
  log("  " .. colorize("-d, --dtypes", C.green) .. "            Liste les dtypes pris en charge par le framework")
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

local opts = parse_flags(arg)

if opts.help then
  print_usage()
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
  -- `-d` seul (sans `-a`) : afficher uniquement les dtypes.
  list_dtypes()
end

if opts.arch then
  if opts.params then
    show_params(opts.arch)
  else
    -- -l sans -p : on confirme juste que l'architecture existe.
    local entry = Mimir.Architectures.info(opts.arch)
    if type(entry) == "table" then
      log("\n" .. colorize("Architecture '" .. opts.arch .. "' disponible.", C.green)
        .. " Ajoutez " .. colorize("-p/--params", C.bold) .. " pour voir ses paramètres.")
    else
      log("\n" .. colorize("[ERROR] ", C.bold, C.red) .. "Architecture inconnue: " .. tostring(opts.arch))
    end
  end
end