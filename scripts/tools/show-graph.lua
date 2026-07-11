---@diagnostic disable: undefined-global, undefined-field

--[[
  show-graph.lua — Visualisation métriques d'entraînement
  Équivalent Lua de tools/show-graph.py
  Génère un rapport HTML interactif (Chart.js).

  Usage (via Mimir):
    ./bin/mimir --lua scripts/tools/show-graph.lua -- [CSV] [OPTIONS]
  Usage (Lua ≥ 5.3 standalone):
    lua scripts/tools/show-graph.lua [CSV] [OPTIONS]

  OPTIONS:
    [CSV]                    CSV à analyser (défaut: checkpoints/loss_history.csv)
    --csv PATH               Chemin CSV unique ou pattern glob (*part0.csv, ...)
    --csv-dir DIR            Dossier contenant des CSV *part[0-9].csv à fusionner
    --model NAME             Nom du modèle (auto-détection si absent)
    --algo NAME              Algo/loss (mse, l1, charbonnier, huber, ...)
    --checkpoint-dir DIR     Dossier run checkpoint (auto-détection si absent)
    -n, --no-interactive     Pas de prompts stdin
    --out PATH               HTML de sortie (défaut: ./graph_report.html)
    --watch                  Mode surveillance : régénère le rapport dès que le CSV change
                             et ouvre le navigateur automatiquement.
    --watch-interval N       Intervalle de polling en secondes (défaut: 2)
    -h, --help               Aide
]]

-- ══════════════════════════════════════════════════════════════
-- HELPERS FICHIERS / SYSTÈME
-- ══════════════════════════════════════════════════════════════

local FS = dofile("scripts/modules/fs.lua")

local function file_exists(p)
  return FS.file_exists(p)
end

-- ══════════════════════════════════════════════════════════════
-- GLOB / DÉTECTION *part[0-9].csv
-- ══════════════════════════════════════════════════════════════

-- Retourne la liste triée des fichiers *part[0-9]+.csv dans un dossier,
-- ou nil si aucun trouvé.
local function find_part_csvs(dir)
  local names = FS.list_dir(dir)
  local found = {}
  for _, name in ipairs(names) do
    if name:match("part%d+%.csv$") then
      found[#found+1] = FS.join(dir, name)
    end
  end
  if #found == 0 then return nil end
  table.sort(found)   -- tri lexicographique = tri numérique sur part0..part9
  return found
end

-- Idem depuis un chemin CSV : si le chemin contient "part[0-9]" on cherche
-- les autres parties dans le même dossier.
local function expand_part_csvs(csv_path)
  if not csv_path:match("part%d+%.csv$") then return { csv_path } end
  local dir = csv_path:match("^(.*[/\\])") or "./"
  -- Retire le slash final pour ls
  dir = dir:gsub("[/\\]$", "")
  if dir == "" then dir = "." end
  local parts = find_part_csvs(dir)
  return parts or { csv_path }
end

-- Fusionne plusieurs tables CSV (même structure d'en-têtes, concatène les lignes).
-- Réassigne `step` de façon continue si la colonne existe.
local function merge_csvs(parts_data)
  if #parts_data == 1 then return parts_data[1] end
  -- Vérifier que tous ont les mêmes colonnes (on prend les headers du 1er)
  local merged = {
    headers = parts_data[1].headers,
    df      = {},
    n       = 0,
    parts   = #parts_data,
  }
  for _, h in ipairs(merged.headers) do merged.df[h] = {} end

  local step_offset = 0
  for pi, part in ipairs(parts_data) do
    local step_max_here = 0
    for i = 1, part.n do
      for _, h in ipairs(merged.headers) do
        local v = (part.df[h] or {})[i]
        if h == "step" and type(v) == "number" then
          v = v + step_offset
          if v - step_offset > step_max_here then step_max_here = v - step_offset end
        end
        if v ~= nil then
          local col = merged.df[h]
          col[#col+1] = v
        end
      end
      merged.n = merged.n + 1
    end
    -- Offset pour la prochaine partie : max step de cette partie + 1
    if part.df.step and #part.df.step > 0 then
      local last = part.df.step[#part.df.step]
      if type(last) == "number" then
        step_offset = step_offset + last + 1
      end
    end
  end
  return merged
end

local function read_file(p)
  local f = io.open(p, "r"); if not f then return nil end
  local s = f:read("*a"); f:close(); return s
end

local function write_file(p, s)
  local f = io.open(p, "w"); if not f then return false end
  f:write(s); f:close(); return true
end

local function is_dir(p)
  return FS.is_dir(p)
end

local function ls(dir)
  return FS.list_dir(dir)
end

local function file_mtime(p)
  local f = io.open(p, "rb")
  if not f then return 0 end
  local r = f:seek("end")
  f:close()
  return tonumber(r) or 0
end

-- mtime combinée d'une liste de fichiers (max)
local function mtimes_combined(paths)
  local mx = 0
  for _, p in ipairs(paths) do
    local m = file_mtime(p); if m > mx then mx = m end
  end
  return mx
end

-- Définie après repo_root() (forward-declaration Lua)
local open_browser

-- sleep portable (Lua n'a pas sleep natif sans LuaSocket)
local function sleep(sec)
  os.execute("sleep " .. tostring(sec))
end

local function file_size(p)
  local f = io.open(p, "rb"); if not f then return 0 end
  local s = f:seek("end"); f:close(); return s or 0
end

local function human_bytes(n)
  if     n < 1024       then return n .. " B"
  elseif n < 1048576    then return string.format("%.1f KiB", n / 1024)
  elseif n < 1073741824 then return string.format("%.1f MiB", n / 1048576)
  else                       return string.format("%.2f GiB", n / 1073741824)
  end
end

local function script_dir()
  local ok, info = pcall(debug.getinfo, 1, "S")
  if ok and info and info.source and info.source:sub(1,1) == "@" then
    return info.source:sub(2):match("^(.*[/\\])") or "./"
  end
  return "./"
end

local function repo_root()
  local sd = script_dir()
  return sd:match("^(.*[/\\])scripts[/\\]tools[/\\]?") or "./"
end

-- ══════════════════════════════════════════════════════════════
-- PARAMÈTRES UI (viz_ui_settings.json)
-- ══════════════════════════════════════════════════════════════

local function ui_settings_path()
  return repo_root() .. "viz_ui_settings.json"
end

-- Lit viz_ui_settings.json.
-- Extrait uniquement les clés connues de show-graph :
--   t.browser    (string top-level)
--   t.showgraph  (sous-objet string pairs)
-- Le reste du fichier n'est jamais désérialisé ni réécrit en entier.
local function load_ui_settings()
  local p = ui_settings_path()
  local s = read_file(p)
  if not s then return {} end
  local t = {}
  local br = s:match('"browser"%s*:%s*"([^"]*)"')
  if br then t.browser = br end
  -- Extraction inline du sous-objet "showgraph" (json_obj défini plus loin)
  local sg_start = s:find('"showgraph"%s*:%s*{')
  if sg_start then
    local depth, i = 0, sg_start
    while i <= #s do
      local c = s:sub(i, i)
      if c == '{' then depth = depth + 1
      elseif c == '}' then
        depth = depth - 1
        if depth == 0 then
          local sg_raw = s:sub(sg_start, i)
          local sg = {}
          for k, v in sg_raw:gmatch('"([^"]+)"%s*:%s*"([^"]*)"') do sg[k] = v end
          t.showgraph = sg
          break
        end
      end
      i = i + 1
    end
  end
  return t
end

-- Applique des patches dans viz_ui_settings.json SANS toucher aux autres clés.
--   patches.key = "string"  → remplace/insère la paire top-level "key": "value"
--   patches.key = { ... }   → remplace/insère le sous-objet top-level "key": { ... }
local function patch_ui_settings(patches)
  local p = ui_settings_path()
  local raw = read_file(p)
  if not raw or raw:match("^%s*$") then raw = "{}\n" end

  -- Position du dernier '}' dans la chaîne
  local function last_rbrace(s)
    for i = #s, 1, -1 do if s:sub(i,i) == '}' then return i end end
  end

  for k, v in pairs(patches) do
    if type(v) == "string" then
      local ek  = k:gsub('([^%w_])', '%%%1')
      local new = string.format('"%s": "%s"', k, v:gsub('"', '\\"'))
      local n
      raw, n = raw:gsub('"' .. ek .. '"%s*:%s*"[^"]*"', new, 1)
      if n == 0 then
        local pos = last_rbrace(raw)
        if pos then
          local before = raw:sub(1, pos - 1)
          local sep = before:match('[^%s,{]%s*$') and ',\n  ' or '  '
          raw = before .. sep .. new .. '\n' .. raw:sub(pos)
        end
      end
    elseif type(v) == "table" then
      local inner = {}
      for sk, sv in pairs(v) do
        inner[#inner+1] = string.format('    "%s": "%s"', sk, tostring(sv):gsub('"', '\\"'))
      end
      table.sort(inner)
      local block = '  "' .. k .. '": {\n'
                    .. (#inner > 0 and table.concat(inner, ',\n') .. '\n' or '')
                    .. '  }'

      local ek    = k:gsub('([^%w_])', '%%%1')
      local s_pos = raw:find('"' .. ek .. '"%s*:%s*{')
      if s_pos then
        -- Remplacer le bloc existant (comptage d'accolades)
        local depth, i = 0, s_pos
        while i <= #raw do
          local c = raw:sub(i, i)
          if c == '{' then depth = depth + 1
          elseif c == '}' then
            depth = depth - 1
            if depth == 0 then
              -- Début de ligne (indentation incluse)
              local ls = s_pos
              while ls > 1 and raw:sub(ls-1, ls-1) ~= '\n' do ls = ls - 1 end
              raw = raw:sub(1, ls - 1) .. block .. raw:sub(i + 1)
              break
            end
          end
          i = i + 1
        end
      else
        -- Insérer avant le dernier '}'
        local pos = last_rbrace(raw)
        if pos then
          local before = raw:sub(1, pos - 1)
          local sep = before:match('[^%s,{]%s*$') and ',\n' or '\n'
          raw = before .. sep .. block .. '\n' .. raw:sub(pos)
        end
      end
    end
  end

  local tmp = p .. '.tmp'
  if write_file(tmp, raw) then os.rename(tmp, p) end
end

-- Alias utilisé par choose_and_save_browser (compat descendante).
local function save_ui_settings(t)
  patch_ui_settings(t)
end

-- Retourne la liste des navigateurs installés sur le système.
local function detect_browsers()
  local candidates = {
    { cmd = os.getenv("BROWSER"),   label = "$BROWSER" },
    { cmd = "sensible-browser",     label = "sensible-browser" },
    { cmd = "x-www-browser",        label = "x-www-browser" },
    { cmd = "firefox",              label = "Firefox" },
    { cmd = "firefox-esr",          label = "Firefox ESR" },
    { cmd = "chromium-browser",     label = "Chromium" },
    { cmd = "chromium",             label = "Chromium" },
    { cmd = "google-chrome",        label = "Google Chrome" },
    { cmd = "google-chrome-stable", label = "Google Chrome (stable)" },
    { cmd = "brave-browser",        label = "Brave" },
  }
  local found, seen = {}, {}
  for _, c in ipairs(candidates) do
    if c.cmd and c.cmd ~= "" and not seen[c.cmd] then
      local ck = io.popen('command -v "' .. c.cmd .. '" 2>/dev/null')
      local bin = ck and ck:read("*l"); if ck then ck:close() end
      if bin and bin ~= "" then
        seen[c.cmd] = true
        found[#found+1] = { cmd = c.cmd, label = c.label, bin = bin }
      end
    end
  end
  return found
end

-- Propose interactivement la liste des navigateurs installés,
-- sauvegarde le choix dans viz_ui_settings.json et retourne la commande.
local function choose_and_save_browser()
  local browsers = detect_browsers()
  io.stderr:write("\n🌐 Premier lancement — quel navigateur souhaitez-vous utiliser ?\n")
  for i, b in ipairs(browsers) do
    io.stderr:write(string.format("  [%d] %-30s  %s\n", i, b.label, b.bin))
  end
  local xdg_idx = #browsers + 1
  io.stderr:write(string.format("  [%d] xdg-open (système par défaut)\n", xdg_idx))
  io.stderr:write(string.format("Votre choix [1-%d] (défaut: 1) : ", xdg_idx))
  io.stderr:flush()
  local line = io.read("*l")
  local choice = math.max(1, math.min(xdg_idx, tonumber(line) or 1))
  local cmd, label
  if choice <= #browsers then
    cmd   = browsers[choice].cmd
    label = browsers[choice].label
  else
    cmd   = "xdg-open"
    label = "xdg-open"
  end
  io.stderr:write("✓ Navigateur choisi : " .. label .. "\n")
  patch_ui_settings({ browser = cmd })
  io.stderr:write("💾 Préférence sauvegardée dans viz_ui_settings.json\n\n")
  return cmd
end

-- Ouvre le rapport HTML dans le navigateur préféré.
-- Au premier appel (aucune préférence enregistrée) demande à l'utilisateur.
open_browser = function(path)
  -- Si le chemin est déjà une URL http(s), l'utiliser telle quelle.
  local url
  if path:match("^https?://") then
    url = path
  else
    local abs = path:sub(1,1) == "/" and path
                or ((os.getenv("PWD") or ".") .. "/" .. path)
    url = "file://" .. abs
  end
  local settings = load_ui_settings()
  local browser  = settings.browser
  if not browser or browser == "" then
    browser = choose_and_save_browser()
  end
  os.execute('"' .. browser .. '" "' .. url .. '" >/dev/null 2>&1 &')
end

-- ══════════════════════════════════════════════════════════════
-- JSON MINIMAL (extraction de champs scalaires depuis architecture.json)
-- ══════════════════════════════════════════════════════════════

local function json_str(text, key)
  return text:match('"' .. key .. '"%s*:%s*"([^"]*)"')
end

local function json_num(text, key)
  return tonumber(text:match('"' .. key .. '"%s*:%s*(%-?%d+%.?%d*[eE]?[+-]?%d*)'))
end

-- Extraire un sous-objet JSON { ... } à la clé donnée
local function json_obj(text, key)
  local s = text:find('"' .. key .. '"%s*:%s*{')
  if not s then return nil end
  local d, i = 0, s
  while i <= #text do
    local c = text:sub(i, i)
    if c == '{' then d = d + 1
    elseif c == '}' then
      d = d - 1; if d == 0 then return text:sub(s, i) end
    end
    i = i + 1
  end
  return nil
end

-- ══════════════════════════════════════════════════════════════
-- CSV PARSER
-- ══════════════════════════════════════════════════════════════

local function parse_csv(path)
  local f = io.open(path, "r")
  if not f then return nil, "Fichier introuvable: " .. path end
  local first = f:read("*l")
  if not first then f:close(); return nil, "CSV vide" end

  local headers = {}
  for col in (first .. ","):gmatch("([^,]*),") do
    headers[#headers+1] = col:match("^%s*(.-)%s*$")
  end

  local df = {}
  for _, h in ipairs(headers) do df[h] = {} end
  local n = 0

  for line in f:lines() do
    if line:match("[^%s,]") then
      n = n + 1
      local ci = 0
      for val in (line .. ","):gmatch("([^,]*),") do
        ci = ci + 1
        if headers[ci] then
          local v = val:match("^%s*(.-)%s*$")
          df[headers[ci]][n] = tonumber(v) or v
        end
      end
    end
  end
  f:close()
  return { headers = headers, df = df, n = n }
end

-- ══════════════════════════════════════════════════════════════
-- STATISTIQUES
-- ══════════════════════════════════════════════════════════════

local function col_stats(t)
  if not t or #t == 0 then return { min=0, max=0, mean=0, std=0, n=0 } end
  local n, s, mn, mx = #t, 0, t[1], t[1]
  for i = 1, n do
    local v = t[i]
    if type(v) == "number" then
      s = s + v
      if v < mn then mn = v end
      if v > mx then mx = v end
    end
  end
  local mu = s / n
  local var = 0
  for i = 1, n do
    local v = t[i]; if type(v) == "number" then var = var + (v - mu)^2 end
  end
  return { min=mn, max=mx, mean=mu, std=math.sqrt(var / n), n=n }
end

local function histogram(t, bins)
  bins = bins or 40
  if not t or #t == 0 then return {}, {} end
  local s = col_stats(t)
  local range = s.max - s.min
  if range == 0 then return { tostring(s.min) }, { #t } end
  local bw = range / bins
  local counts, labels = {}, {}
  for i = 1, bins do
    counts[i] = 0
    labels[i] = string.format("%.4g", s.min + (i - 0.5) * bw)
  end
  for _, v in ipairs(t) do
    if type(v) == "number" then
      local b = math.max(1, math.min(bins, math.floor((v - s.min) / bw) + 1))
      counts[b] = counts[b] + 1
    end
  end
  return labels, counts
end

local function pearson(a, b)
  local n = math.min(#a, #b); if n < 2 then return 0 end
  local sa, sb = 0, 0
  for i = 1, n do
    if type(a[i]) == "number" and type(b[i]) == "number" then sa = sa + a[i]; sb = sb + b[i] end
  end
  local ma, mb = sa / n, sb / n
  local num, da2, db2 = 0, 0, 0
  for i = 1, n do
    if type(a[i]) == "number" and type(b[i]) == "number" then
      local da, db = a[i] - ma, b[i] - mb
      num = num + da * db; da2 = da2 + da * da; db2 = db2 + db * db
    end
  end
  local d = math.sqrt(da2 * db2); return d > 0 and num / d or 0
end

-- Extrait les points de validation (val_loss numérique) avec le train_loss associé.
-- Retourne un tableau trié par step de { step, val_loss, val_mse, train_loss, gap }.
local function extract_val_points(df, n)
  local pts = {}
  if not df or not df.val_loss then return pts end

  -- Tableau trié (step, loss) pour recherche du train_loss le plus proche (bin search).
  local ts = {}
  local step_arr = df.step or {}
  local loss_arr = df.loss or {}
  for i = 1, n do
    local s, l = step_arr[i], loss_arr[i]
    if type(s) == "number" and type(l) == "number" then
      ts[#ts+1] = { s = s, l = l }
    end
  end
  table.sort(ts, function(a, b) return a.s < b.s end)

  local function nearest_loss(target)
    if #ts == 0 or type(target) ~= "number" then return nil end
    local lo, hi = 1, #ts
    while lo < hi do
      local mid = math.floor((lo + hi) / 2)
      if ts[mid].s < target then lo = mid + 1 else hi = mid end
    end
    local best = ts[lo]
    if lo > 1 and math.abs(ts[lo-1].s - target) < math.abs(best.s - target) then
      best = ts[lo-1]
    end
    return best.l
  end

  local vst_arr = df.val_step or {}
  local vmse_arr = df.val_mse  or {}

  for i = 1, n do
    local vl = df.val_loss[i]
    if type(vl) == "number" then
      local cs  = step_arr[i]
      local vs  = vst_arr[i]
      local vm  = vmse_arr[i]
      local actual = (type(vs) == "number" and vs >= 0) and vs
                     or (type(cs) == "number" and cs or i)
      local tl = nearest_loss(actual)
      pts[#pts+1] = {
        step       = actual,
        val_loss   = vl,
        val_mse    = type(vm) == "number" and vm or nil,
        train_loss = tl,
        gap        = tl and (vl - tl) or nil,
      }
    end
  end
  table.sort(pts, function(a, b) return a.step < b.step end)
  return pts
end

-- Retire toutes les lignes appartenant à un cycle de validation :
-- pendant la validation, l'optimiseur ne fait pas de mise à jour, donc opt_step reste
-- constant. Toutes les lignes qui partagent le même opt_step qu'une ligne de résultat
-- val_loss (y compris les items intermédiaires qui n'ont pas encore val_loss rempli)
-- sont exclues des métriques d'entraînement, sauf la PREMIÈRE occurrence (la vraie
-- étape d'entraînement qui a déclenché la validation).
-- Retourne (train_df, train_n, n_val_excluded).
local function filter_train_rows(df, headers, n)
  if not df.val_loss then return df, n, 0 end

  -- Clé pivot : opt_step est le compteur d'optimizer steps (ne bouge pas pendant val).
  -- Fallback sur step si opt_step absent du CSV.
  local step_key = (df.opt_step and #df.opt_step > 0) and "opt_step" or "step"

  -- Collecter les valeurs pivot présentes dans les lignes de résultat val_loss
  local val_pivot = {}
  for i = 1, n do
    if type(df.val_loss[i]) == "number" then
      local sv = df[step_key] and df[step_key][i]
      if type(sv) == "number" then val_pivot[sv] = true end
    end
  end
  if not next(val_pivot) then return df, n, 0 end

  -- Conserver seulement la première ligne pour chaque pivot (= étape d'entraînement)
  local first_seen = {}
  local new_df = {}
  for _, h in ipairs(headers) do new_df[h] = {} end
  local new_n = 0

  for i = 1, n do
    local sv = df[step_key] and df[step_key][i]
    if type(sv) == "number" and val_pivot[sv] then
      if not first_seen[sv] then
        first_seen[sv] = true
        new_n = new_n + 1
        for _, h in ipairs(headers) do new_df[h][new_n] = df[h][i] end
      end
      -- lignes suivantes avec le même pivot = items + résumé de validation → ignorées
    else
      new_n = new_n + 1
      for _, h in ipairs(headers) do new_df[h][new_n] = df[h][i] end
    end
  end
  return new_df, new_n, n - new_n
end

-- Auto-détection des paramètres de calibration depuis le CSV :
--   validate_every_steps : fréquence de validation (en opt_steps)
--   validate_items       : nombre d'images évaluées par validation
--   n_dataset            : nombre d'items dans le dataset (par epoch)
local function detect_validation_params(df, n)
  local p = { validate_every_steps = nil, validate_items = nil, n_dataset = nil }

  -- n_dataset : colonne total_batches (constante dans le CSV)
  if df.total_batches and #df.total_batches > 0 then
    p.n_dataset = df.total_batches[1]
  end

  if not df.val_loss then return p end

  local step_key = (df.opt_step and #df.opt_step > 0) and "opt_step" or "step"

  -- Collecter les opt_steps de validation dans l'ordre
  local seen, sorted = {}, {}
  for i = 1, n do
    if type(df.val_loss[i]) == "number" then
      local sv = df[step_key] and df[step_key][i]
      if type(sv) == "number" and not seen[sv] then
        seen[sv] = true; sorted[#sorted+1] = sv
      end
    end
  end
  table.sort(sorted)

  -- validate_every_steps : mode des espacements (valeur la plus fréquente)
  if #sorted >= 2 then
    local freq = {}
    for k = 2, math.min(#sorted, 12) do
      local d = sorted[k] - sorted[k-1]
      freq[d] = (freq[d] or 0) + 1
    end
    local best_d, best_f = nil, 0
    for d, f in pairs(freq) do
      if f > best_f then best_d = d; best_f = f end
    end
    p.validate_every_steps = best_d
  end

  -- validate_items : lignes avec le même pivot que le 1er résultat val, moins 1 (training row)
  if #sorted > 0 then
    local sample = sorted[1]
    local cnt = 0
    for i = 1, n do
      local sv = df[step_key] and df[step_key][i]
      if sv == sample then cnt = cnt + 1 end
    end
    -- cnt = 1 training + N items intermédiaires + 1 résumé val_loss
    -- items évalués = cnt - 1 (on exclut la ligne training)
    p.validate_items = math.max(0, cnt - 1)
  end

  return p
end

-- ══════════════════════════════════════════════════════════════
-- SÉRIALISATION JS
-- ══════════════════════════════════════════════════════════════

-- Tableau JS à partir d'une table Lua (nombres ou chaînes)
local function js_arr(t, stride)
  stride = stride or 1
  local p = {}
  for i = 1, #t, stride do
    local v = t[i]
    if type(v) == "number" then
      p[#p+1] = (v ~= v) and "null" or string.format("%.7g", v)
    elseif type(v) == "string" then
      p[#p+1] = '"' .. v:gsub('"', '\\"') .. '"'
    else
      p[#p+1] = "null"
    end
  end
  return "[" .. table.concat(p, ",") .. "]"
end

-- Tableau de points {x,y} pour Chart.js scatter/line
local function js_xy(xs, ys, stride)
  stride = stride or 1
  local p = {}
  local n = math.min(#xs, #ys)
  for i = 1, n, stride do
    if type(xs[i]) == "number" and type(ys[i]) == "number" then
      p[#p+1] = string.format("{x:%.7g,y:%.7g}", xs[i], ys[i])
    end
  end
  return "[" .. table.concat(p, ",") .. "]"
end

-- Chaîne JS (avec guillemets, échappée)
local function js_s(s)
  if s == nil then return "null" end
  return '"' .. tostring(s):gsub('"', '\\"'):gsub('\n', '\\n') .. '"'
end

-- Couleur CSS pour la matrice de corrélation : -1=rouge, 0=blanc, +1=bleu
local function corr_bg(r)
  r = math.max(-1, math.min(1, r or 0))
  if r >= 0 then
    local g = math.floor(255 * (1 - r)); return string.format("rgb(%d,%d,255)", g, g)
  else
    local gb = math.floor(255 * (1 + r)); return string.format("rgb(255,%d,%d)", gb, gb)
  end
end

-- Sérialise les colonnes CSV (strided) en JSON pour les mises à jour DOM partielles.
-- Appelée depuis generate() après chaque régénération du HTML.
local function gen_data_json(df, n, val_pts)
  local stride = math.max(1, math.ceil(n / 2000))
  local buf = {}
  local function e(s) buf[#buf+1] = s end
  e(string.format('{"stride":%d,"n":%d', stride, n))
  for _, c in ipairs({
    "step","opt_step","loss","avg_loss","learning_rate","mse","kl_divergence",
    "wasserstein","entropy_diff","moment_mismatch","spatial_coherence",
    "temporal_consistency","timestep","opt_eps","epoch",
  }) do
    if df[c] and #df[c] > 0 then
      e(',"' .. c .. '":' .. js_arr(df[c], stride))
    end
  end
  if val_pts and #val_pts > 0 then
    local vp = {}
    for _, p in ipairs(val_pts) do
      vp[#vp+1] = string.format(
        '{"step":%.7g,"val_loss":%.7g,"val_mse":%s,"train_loss":%s,"gap":%s}',
        p.step, p.val_loss,
        p.val_mse    and string.format("%.7g", p.val_mse)    or "null",
        p.train_loss and string.format("%.7g", p.train_loss) or "null",
        p.gap        and string.format("%.7g", p.gap)        or "null"
      )
    end
    e(',"val_pts":[' .. table.concat(vp, ",") .. "]")
  else
    e(',"val_pts":[]')
  end
  e("}")
  return table.concat(buf)
end

-- ══════════════════════════════════════════════════════════════
-- AUTO-DÉTECTION CHECKPOINT
-- ══════════════════════════════════════════════════════════════

local function find_latest_run()
  local ckpt = repo_root() .. "checkpoint"
  if not is_dir(ckpt) then return nil end
  local best, bmt = nil, 0
  for _, name in ipairs(ls(ckpt)) do
    if name ~= "base_tokenizer" and not name:match("^%.") then
      local p = ckpt .. "/" .. name
      if is_dir(p) then
        local mt = file_mtime(p); if mt > bmt then bmt = mt; best = p end
      end
    end
  end
  return best
end

local function find_latest_epoch(run_dir)
  local best, bmt = nil, 0
  for _, name in ipairs(ls(run_dir)) do
    if name:match("^epoch_") then
      local p = run_dir .. "/" .. name
      if is_dir(p) and file_exists(p .. "/model/architecture.json") then
        local mt = file_mtime(p); if mt > bmt then bmt = mt; best = p end
      end
    end
  end
  return best
end

local function load_meta(run_dir)
  local m = { source = run_dir }
  local ep = find_latest_epoch(run_dir)
  if not ep then return m end
  m.source = (run_dir:match("[^/]+$") or run_dir) .. "/" .. (ep:match("[^/]+$") or ep)
  local text = read_file(ep .. "/model/architecture.json")
  if not text then return m end
  local mc = json_obj(text, "model_config")
  if mc then
    m.model = json_str(mc, "type")
    m.algo  = json_str(mc, "recon_loss")
    if m.algo == "charbonnier" then
      local e = json_num(mc, "charbonnier_eps")
      if e then m.algo_details = string.format("eps=%g", e) end
    elseif m.algo == "huber" then
      local d = json_num(mc, "huber_delta")
      if d then m.algo_details = string.format("delta=%g", d) end
    elseif m.algo and m.algo:match("nll") then
      local sg = json_num(mc, "nll_sigma")
      if sg then m.algo_details = string.format("sigma=%g", sg) end
    end
    m.checkpoint_dir = json_str(mc, "checkpoint_dir") or run_dir
  else
    m.model = json_str(text, "model_name")
  end
  return m
end

-- ══════════════════════════════════════════════════════════════
-- GÉNÉRATION HTML
-- ══════════════════════════════════════════════════════════════

local PALETTE = {
  loss     = "#2E86AB",
  recon    = "#A23B72",
  lr       = "#F18F01",
  kl       = "#E63946",
  wass     = "#F4A261",
  entropy  = "#2A9D8F",
  moment   = "#E76F51",
  spatial  = "#52B788",
  temporal = "#8338EC",
  timestep = "#457B9D",
  opt_eps  = "#6A4C93",
  val_loss = "#FF9F1C",
  val_recon= "#E040FB",
}

local EPOCH_PAL = {
  "#2E86AB","#A23B72","#F18F01","#E63946","#2A9D8F",
  "#8338EC","#F4A261","#52B788","#E76F51","#457B9D",
  "#E9C46A","#06D6A0","#EF476F","#118AB2","#FFD166",
}

local function gen_html(ctx)
  local df      = ctx.df
  local n       = ctx.n
  local rc      = ctx.recon_col
  local rl      = ctx.recon_label
  local epochs  = ctx.epochs

  local stride = math.max(1, math.ceil(n / 2000))

  -- Helpers locaux
  local function jd(c)
    return js_arr(df[c] or {}, stride)
  end
  local function jdxy(xc, yc)
    return js_xy(df[xc] or {}, df[yc] or {}, stride)
  end
  local function has(c)
    return df[c] ~= nil and #df[c] > 0
  end
  local steps = jd("step")

  -- Histogrammes
  local hlbl_loss, hcnt_loss
  if has("loss") then hlbl_loss, hcnt_loss = histogram(df.loss, 40) end
  local hlbl_rc, hcnt_rc
  if rc and has(rc) then hlbl_rc, hcnt_rc = histogram(df[rc], 40) end

  -- Colonnes pour matrice de corrélation
  local corr_cols = {}
  for _, c in ipairs({"loss", rc, "learning_rate", "kl_divergence",
                      "wasserstein", "timestep", "opt_eps", "epoch"}) do
    if c and has(c) then corr_cols[#corr_cols+1] = c end
  end

  -- Datasets par epoch (scatter+showLine)
  local ep_ds = {}
  if has("epoch") and has("loss") and has("step") then
    local ep_map = {}
    for i = 1, n do
      local e = df.epoch[i]
      if type(e) == "number" then
        if not ep_map[e] then ep_map[e] = { sx = {}, sy = {} } end
        local d = ep_map[e]
        d.sx[#d.sx+1] = df.step[i]
        d.sy[#d.sy+1] = df.loss[i]
      end
    end
    for idx, e in ipairs(epochs) do
      local ed = ep_map[e]
      if ed then
        local est = math.max(1, math.ceil(#ed.sx / 500))
        local col = EPOCH_PAL[((idx - 1) % #EPOCH_PAL) + 1]
        ep_ds[#ep_ds+1] = string.format(
          "{label:'Epoch %d',data:%s,borderColor:'%s',backgroundColor:'transparent',"
          .. "borderWidth:1.5,showLine:true,pointRadius:0,tension:0.2}",
          e, js_xy(ed.sx, ed.sy, est), col)
      end
    end
  end

  -- Tableau HTML de corrélation
  local corr_tbl = ""
  if #corr_cols > 0 then
    corr_tbl = corr_tbl .. "<tr><th></th>"
    for _, c in ipairs(corr_cols) do
      corr_tbl = corr_tbl .. "<th>" .. (c == rc and rl or c) .. "</th>"
    end
    corr_tbl = corr_tbl .. "</tr>\n"
    for _, ca in ipairs(corr_cols) do
      corr_tbl = corr_tbl .. "<tr><th>" .. (ca == rc and rl or ca) .. "</th>"
      for _, cb in ipairs(corr_cols) do
        local r = pearson(df[ca], df[cb])
        corr_tbl = corr_tbl .. string.format(
          '<td style="background:%s;color:%s">%.2f</td>',
          corr_bg(r), math.abs(r) > 0.5 and "#fff" or "#222", r)
      end
      corr_tbl = corr_tbl .. "</tr>\n"
    end
  end

  -- Registre des instances Chart.js pour les mises à jour DOM partielles.
  -- Chaque builder y stocke {type, ...} ; sérialisé dans window._sg_chart_meta.
  local chart_meta = {}

  -- ── Builders de cartes Chart.js ────────────────────────────

  -- Clé x commune à tous les graphes : opt_step si disponible (pas de trous), sinon step.
  local x_key = has("opt_step") and "opt_step" or "step"

  -- Sérialise les frontières d'epoch en annotations Chart.js (valeurs réelles opt_step).
  local function epoch_annotations_js()
    local ep_bnd = ctx.epoch_boundaries or {}
    if #ep_bnd == 0 then return "{}" end
    local parts = {}
    for _, b in ipairs(ep_bnd) do
      parts[#parts+1] = string.format(
        'ep%d:{type:\'line\',scaleID:\'x\',value:%g,'
        .. 'borderColor:\'rgba(255,255,255,0.18)\',borderWidth:1,borderDash:[4,4],'
        .. 'label:{display:true,content:\'E%d\',position:\'start\','
        .. 'color:\'#8b949e\',font:{size:8},padding:{y:2}}}',
        b.epoch, b.step, b.epoch)
    end
    return '{' .. table.concat(parts, ',') .. '}'
  end

  -- Graphe en ligne : scatter+showLine avec x_key comme abscisse réelle (axe linéaire).
  -- Tous les graphes partagent le même espace x → annotations d'epoch alignées.
  local function card_line(id, title, col_name, color, log_y)
    if not has(col_name) then return "" end
    local yscale = log_y and "logarithmic" or "linear"
    local lbl    = (col_name == rc) and rl or col_name
    chart_meta[id] = string.format('{type:"line",xcol:"%s",ycol:"%s"}', x_key, col_name)
    return string.format(
      '<div class="card"><h3>%s</h3><canvas id="%s"></canvas></div>\n'
      .. "<script>window._sg_charts[%s]=new Chart(document.getElementById(%s),{type:'scatter',"
      .. "data:{datasets:[{label:%s,data:%s,"
      .. "borderColor:'%s',backgroundColor:'%s22',fill:true,"
      .. "borderWidth:1.5,tension:0.2,pointRadius:0,showLine:true}]},"
      .. "options:{responsive:true,maintainAspectRatio:false,animation:false,"
      .. "plugins:{legend:{display:false},annotation:{annotations:%s}},"
      .. "scales:{x:{type:'linear',ticks:{maxTicksLimit:8}},y:{type:'%s'}}}});</script>\n",
      title, id, js_s(id), js_s(id), js_s(lbl), jdxy(x_key, col_name),
      color, color, epoch_annotations_js(), yscale)
  end

  -- Histogramme (barres)
  local function card_bar(id, title, labels, counts, color)
    if not labels or #labels == 0 then return "" end
    return string.format(
      '<div class="card"><h3>%s</h3><canvas id="%s"></canvas></div>\n'
      .. "<script>new Chart(document.getElementById(%s),{type:'bar',"
      .. "data:{labels:%s,datasets:[{label:'',data:%s,"
      .. "backgroundColor:'%s88',borderColor:'%s',borderWidth:1}]},"
      .. "options:{responsive:true,maintainAspectRatio:false,animation:false,"
      .. "plugins:{legend:{display:false}},"
      .. "scales:{x:{ticks:{maxTicksLimit:10}},y:{ticks:{maxTicksLimit:6}}}}});</script>\n",
      title, id, js_s(id), js_arr(labels), js_arr(counts), color, color)
  end

  -- Nuage de points
  local function card_scatter(id, title, xcol, ycol, color, xl, yl)
    if not has(xcol) or not has(ycol) then return "" end
    chart_meta[id] = string.format('{type:"scatter",xcol:"%s",ycol:"%s"}', xcol, ycol)
    return string.format(
      '<div class="card"><h3>%s</h3><canvas id="%s"></canvas></div>\n'
      .. "<script>window._sg_charts[%s]=new Chart(document.getElementById(%s),{type:'scatter',"
      .. "data:{datasets:[{label:'',data:%s,"
      .. "backgroundColor:'%s55',borderColor:'transparent',"
      .. "pointRadius:2,pointHoverRadius:3}]},"
      .. "options:{responsive:true,maintainAspectRatio:false,animation:false,"
      .. "plugins:{legend:{display:false}},"
      .. "scales:{x:{title:{display:true,text:%s}},"
      .. "y:{title:{display:true,text:%s}}}}});</script>\n",
      title, id, js_s(id), js_s(id), jdxy(xcol, ycol), color, js_s(xl), js_s(yl))
  end

  -- ── Assemblage HTML ────────────────────────────────────────

  local out = {}
  local function emit(s) out[#out+1] = s end

  -- DOCTYPE + head
  emit("<!DOCTYPE html>\n<html lang='fr'>\n<head>\n")
  emit("<meta charset='utf-8'>\n")
  emit("<meta name='viewport' content='width=device-width,initial-scale=1'>\n")
  emit("<title>Training Metrics — " .. (ctx.model or "Mimir") .. "</title>\n")
  emit("<script src='https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js'></script>\n")
  emit("<script src='https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js'></script>\n")
  emit("<style>\n")
  emit(":root{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#c9d1d9;--muted:#8b949e;--accent:#58a6ff}\n")
  emit("*{box-sizing:border-box;margin:0;padding:0}\n")
  -- Fondu entrant à chaque chargement de page
  emit("@keyframes _sg_in{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}\n")
  emit("body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;")
  emit("background:var(--bg);color:var(--text);padding:24px;")
  emit("animation:_sg_in 0.35s ease both}\n")
  emit("header{background:var(--card);border:1px solid var(--border);border-radius:10px;")
  emit("padding:20px 24px;margin-bottom:24px}\n")
  emit("header h1{font-size:1.4rem;color:var(--accent);margin-bottom:12px}\n")
  emit(".meta{display:flex;flex-wrap:wrap;gap:10px;font-size:.82rem;color:var(--muted)}\n")
  emit(".meta span{background:#21262d;padding:4px 10px;border-radius:6px;white-space:nowrap}\n")
  emit(".meta b{color:var(--text)}\n")
  emit("section{margin-bottom:28px}\n")
  emit("section>h2{font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);")
  emit("margin-bottom:12px;border-bottom:1px solid var(--border);padding-bottom:6px}\n")
  emit(".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px}\n")
  emit(".card{background:var(--card);border:1px solid var(--border);border-radius:8px;")
  emit("padding:14px;min-height:270px}\n")
  emit(".card h3{font-size:.74rem;font-weight:600;color:var(--muted);margin-bottom:10px;")
  emit("text-transform:uppercase;letter-spacing:.05em}\n")
  emit(".card canvas{height:210px!important;width:100%!important}\n")
  emit(".card.full{grid-column:1/-1;min-height:auto}\n")
  emit(".card.full canvas{height:280px!important}\n")
  emit(".scroll{overflow-x:auto;margin-top:8px}\n")
  emit("table.corr{border-collapse:collapse;font-size:.76rem}\n")
  emit("table.corr th,table.corr td{padding:5px 9px;border:1px solid var(--border);")
  emit("text-align:center;min-width:54px}\n")
  emit("table.corr th{background:#21262d;color:var(--muted);font-weight:600}\n")
  emit("table.stats{width:100%;border-collapse:collapse;font-size:.81rem}\n")
  emit("table.stats th{background:#21262d;color:var(--muted);padding:6px 10px;")
  emit("text-align:left;font-weight:600;border-bottom:1px solid var(--border)}\n")
  emit("table.stats td{padding:5px 10px;border-bottom:1px solid #21262d;font-family:monospace}\n")
  emit("table.stats tr:hover td{background:#21262d44}\n")
  emit("footer{text-align:center;font-size:.72rem;color:var(--muted);margin-top:24px;")
  emit("padding-top:14px;border-top:1px solid var(--border)}\n")
  -- Bouton de sauvegarde (visible uniquement en mode watch)
  emit(".sg-save-btn{position:fixed;top:18px;right:20px;z-index:999;")
  emit("background:#238636;color:#fff;border:1px solid #2ea043;border-radius:6px;")
  emit("padding:7px 14px;font-size:.8rem;font-weight:600;cursor:pointer;")
  emit("transition:background 0.2s,transform 0.1s}\n")
  emit(".sg-save-btn:hover{background:#2ea043}\n")
  emit(".sg-save-btn:active{transform:scale(0.96)}\n")
  emit(".sg-toast{position:fixed;bottom:24px;right:20px;z-index:999;")
  emit("background:#161b22;border:1px solid #30363d;border-radius:8px;")
  emit("padding:10px 16px;font-size:.8rem;color:#c9d1d9;")
  emit("opacity:0;transform:translateY(8px);")
  emit("transition:opacity 0.25s,transform 0.25s;pointer-events:none}\n")
  emit(".sg-toast.sg-show{opacity:1;transform:none}\n")
  emit("</style>\n")
  -- Script d'initialisation : registre des charts + fonctions de mise à jour DOM.
  -- _sg_apply(d)       : met à jour les datasets Chart.js sans rechargement de page.
  -- _sg_fetch_update() : récupère /graph_data.json puis appelle _sg_apply.
  emit("<script>\n")
  emit("window._sg_charts={};\n")
  emit("function _sg_apply(d){\n")
  emit("  var M=window._sg_chart_meta||{},C=window._sg_charts||{};\n")
  emit("  function xy(xc,yc){\n")
  emit("    var xs=d[xc]||[],ys=d[yc]||[],p=[];\n")
  emit("    for(var i=0;i<Math.min(xs.length,ys.length);i++)\n")
  emit("      if(typeof xs[i]==='number'&&typeof ys[i]==='number')\n")
  emit("        p.push({x:xs[i],y:ys[i]});\n")
  emit("    return p;\n")
  emit("  }\n")
  emit("  for(var id in M){\n")
  emit("    var m=M[id],ch=C[id];if(!ch)continue;\n")
  emit("    if(m.type==='line'){\n")
  emit("      ch.data.datasets[0].data=xy(m.xcol,m.ycol);\n")
  emit("      ch.update('none');\n")
  emit("    }else if(m.type==='scatter'){\n")
  emit("      ch.data.datasets[0].data=xy(m.xcol,m.ycol);\n")
  emit("      ch.update('none');\n")
  emit("    }else if(m.type==='val_overlay'){\n")
  emit("      ch.data.datasets[0].data=xy(d.opt_step?'opt_step':'step','loss');\n")
  emit("      ch.data.datasets[1].data=(d.val_pts||[]).map(function(p){return{x:p.step,y:p.val_loss};});\n")
  emit("      ch.update('none');\n")
  emit("    }else if(m.type==='val_gap'){\n")
  emit("      var vp=(d.val_pts||[]).filter(function(p){return p.gap!=null;});\n")
  emit("      ch.data.labels=vp.map(function(p){return String(Math.floor(p.step));});\n")
  emit("      var gv=vp.map(function(p){return p.gap;});\n")
  emit("      ch.data.datasets[0].data=gv;\n")
  emit("      ch.data.datasets[0].backgroundColor=gv.map(function(v){return v>0?'#E6394666':'#2A9D8F66';});\n")
  emit("      ch.data.datasets[0].borderColor=gv.map(function(v){return v>0?'#E63946':'#2A9D8F';});\n")
  emit("      ch.update('none');\n")
  emit("    }\n")
  emit("  }\n")
  emit("  var b=document.body;\n")
  emit("  b.style.transition='opacity 0.12s ease';\n")
  emit("  b.style.opacity='0.65';\n")
  emit("  setTimeout(function(){b.style.opacity='1';},160);\n")
  emit("}\n")
  emit("function _sg_fetch_update(){\n")
  emit("  fetch('/graph_data.json')\n")
  emit("    .then(function(r){return r.json();})\n")
  emit("    .then(_sg_apply)\n")
  emit("    .catch(function(){});\n")
  emit("}\n")
  emit("</script>\n")
  emit("</head>\n<body>\n")

  -- Header
  emit("<header>\n<h1>📈 Training Metrics Dashboard</h1>\n<div class='meta'>\n")
  if ctx.model then
    emit("  <span>🧠 <b>Modèle :</b> " .. ctx.model .. "</span>\n")
  end
  if ctx.algo then
    local a = ctx.algo .. (ctx.algo_details and (" [" .. ctx.algo_details .. "]") or "")
    emit("  <span>📐 <b>Algo :</b> " .. a .. "</span>\n")
  end
  if ctx.meta and ctx.meta.source then
    emit("  <span>📦 <b>Source :</b> " .. ctx.meta.source .. "</span>\n")
  end
  emit(string.format("  <span>📊 <b>Steps :</b> %d &nbsp;|&nbsp; <b>Epochs :</b> %d</span>\n", n, #epochs))
  if ctx.n_dataset and ctx.n_dataset > 0 then
    emit(string.format("  <span>🗂️ <b>Dataset :</b> %d items/epoch</span>\n", ctx.n_dataset))
  end
  if ctx.validate_every_steps and ctx.validate_every_steps > 0 then
    local frac = (ctx.n_dataset and ctx.n_dataset > 0)
      and string.format(" = 1/%.4g epoch", ctx.n_dataset / ctx.validate_every_steps) or ""
    emit(string.format("  <span>✅ <b>Val :</b> tous les %d opt_steps%s</span>\n",
      ctx.validate_every_steps, frac))
  end
  if ctx.validate_items and ctx.validate_items > 0 then
    emit(string.format("  <span>🔬 <b>Items/val :</b> %d</span>\n", ctx.validate_items))
  end
  emit("  <span>📄 <b>CSV :</b> " .. ctx.csv_path .. " (" .. human_bytes(ctx.csv_size) .. ")</span>\n")
  emit("</div>\n</header>\n")
  -- Bouton de sauvegarde (mode --watch uniquement)
  if ctx.sse_port and ctx.sse_port > 0 then
    emit(string.format(
      '<button class="sg-save-btn" onclick="_sg_save()">💾 Sauvegarder</button>\n'
      .. '<div class="sg-toast" id="sg_toast"></div>\n'
      .. '<script>\n'
      .. 'function _sg_save(){\n'
      .. '  fetch("http://127.0.0.1:%d/snapshot",{method:"POST"})\n'
      .. '    .then(function(r){return r.json();})\n'
      .. '    .then(function(j){\n'
      .. '      var name=j.saved||(j.file||"snapshot sauvegardé");\n'
      .. '      _sg_toast("✓ " + name);\n'
      .. '    })\n'
      .. '    .catch(function(e){_sg_toast("Erreur : "+e,true);});\n'
      .. '}\n'
      .. 'function _sg_toast(msg,err){\n'
      .. '  var t=document.getElementById("sg_toast");\n'
      .. '  t.textContent=msg;\n'
      .. '  t.style.borderColor=err?"#f85149":"#2ea043";\n'
      .. '  t.classList.add("sg-show");\n'
      .. '  setTimeout(function(){t.classList.remove("sg-show");},3500);\n'
      .. '}\n'
      .. '</script>\n', ctx.sse_port))
  end

  -- Section: métriques globales
  emit("\n<section><h2>Métriques globales</h2><div class='grid'>\n")
  emit(card_line("c_loss",  "Loss",               "loss",              PALETTE.loss,     false))
  if rc then
    emit(card_line("c_rc",  rl,                   rc,                  PALETTE.recon,    false))
  end
  emit(card_line("c_lr",    "Learning Rate",       "learning_rate",    PALETTE.lr,       true))
  emit(card_line("c_kl",    "KL Divergence",       "kl_divergence",    PALETTE.kl,       false))
  emit(card_line("c_wass",  "Wasserstein",         "wasserstein",      PALETTE.wass,     false))
  emit(card_line("c_ent",   "Entropy Δ",           "entropy_diff",     PALETTE.entropy,  false))
  emit(card_line("c_mom",   "Moment Mismatch",     "moment_mismatch",  PALETTE.moment,   false))
  emit(card_line("c_spat",  "Spatial Coherence",   "spatial_coherence",PALETTE.spatial,  false))
  emit(card_line("c_temp",  "Temporal Consistency","temporal_consistency",PALETTE.temporal,false))
  emit(card_line("c_ts",    "Timestep",            "timestep",         PALETTE.timestep, false))
  emit(card_line("c_eps",   "opt_eps",             "opt_eps",          PALETTE.opt_eps,  false))
  emit("</div></section>\n")

  -- Section: validation + écart objectif/erreurs
  local val_pts  = ctx.val_pts or {}
  local val_cols = {}
  for _, c in ipairs(ctx.headers or {}) do
    if c:match("^val_") and has(c) then val_cols[#val_cols+1] = c end
  end

  if #val_pts > 0 or #val_cols > 0 then
    emit("\n<section><h2>Validation &amp; Écart objectif</h2><div class='grid'>\n")

    if #val_pts > 0 then
      -- Sérialisation des points de validation
      local vsc, gsv, gss = {}, {}, {}
      for _, p in ipairs(val_pts) do
        vsc[#vsc+1] = string.format("{x:%.7g,y:%.7g}", p.step, p.val_loss)
        if p.gap then
          gsv[#gsv+1] = string.format("%.7g", p.gap)
          gss[#gss+1] = string.format("%d", math.floor(p.step))
        end
      end
      local val_js = "[" .. table.concat(vsc, ",") .. "]"
      local gap_v  = "[" .. table.concat(gsv, ",") .. "]"
      local gap_s  = '["' .. table.concat(gss, '","') .. '"]'

      -- Overlay : courbe train (ligne) + points val (scatter alignés sur step)
      emit('<div class="card full">\n')
      emit('<h3>Train vs Validation Loss')
      emit(' <small style="font-weight:normal;color:var(--muted)">')
      emit('&mdash; points = étapes de validation</small></h3>\n')
      emit('<canvas id="c_val_ov"></canvas></div>\n')
      chart_meta["c_val_ov"] = '{type:"val_overlay"}'
      emit('<script>\n')
      emit('window._sg_charts["c_val_ov"]=new Chart(document.getElementById("c_val_ov"),{\n')
      emit('  type:"scatter",\n')
      emit('  data:{datasets:[\n')
      emit('    {label:"Train loss",type:"line",\n')
      emit('     data:' .. jdxy(x_key, "loss") .. ',\n')
      emit('     borderColor:"' .. PALETTE.loss .. '",\n')
      emit('     backgroundColor:"' .. PALETTE.loss .. '22",\n')
      emit('     borderWidth:1.5,pointRadius:0,fill:true,showLine:true,tension:0.2,order:2},\n')
      emit('    {label:"Val loss",\n')
      emit('     data:' .. val_js .. ',\n')
      emit('     borderColor:"' .. PALETTE.val_loss .. '",\n')
      emit('     backgroundColor:"' .. PALETTE.val_loss .. 'CC",\n')
      emit('     pointRadius:6,pointHoverRadius:9,showLine:false,order:1}\n')
      emit('  ]},\n')
      emit('  options:{responsive:true,maintainAspectRatio:false,animation:false,\n')
      emit('    plugins:{legend:{display:true,labels:{boxWidth:10,font:{size:10}}},\n')
      emit('             annotation:{annotations:' .. epoch_annotations_js() .. '}},\n')
      emit('    scales:{x:{title:{display:true,text:"Step"},ticks:{maxTicksLimit:10}},\n')
      emit('            y:{title:{display:true,text:"Loss"}}}}\n')
      emit('});\n</script>\n')

      -- Histogramme des écarts val - train (rouge=surapprentissage, vert=sous)
      if #gsv > 0 then
        emit('<div class="card full">\n')
        emit('<h3>Écart Val &minus; Train')
        emit(' <small style="font-weight:normal;color:var(--muted)">')
        emit('&nbsp;<span style="color:#E63946">&#9632;</span> sur-apprentissage (val&gt;train)')
        emit('&nbsp;&nbsp;<span style="color:#2A9D8F">&#9632;</span> sous-apprentissage')
        emit('</small></h3>\n')
        emit('<canvas id="c_gap"></canvas></div>\n')
        chart_meta["c_gap"] = '{type:"val_gap"}'
        emit('<script>(function(){\n')
        emit('  var gv=' .. gap_v .. ', gs=' .. gap_s .. ';\n')
        emit('  var bg=gv.map(function(v){return v>0?"#E6394666":"#2A9D8F66";});\n')
        emit('  var brd=gv.map(function(v){return v>0?"#E63946":"#2A9D8F";});\n')
        emit('  window._sg_charts["c_gap"]=new Chart(document.getElementById("c_gap"),{\n')
        emit('    type:"bar",\n')
        emit('    data:{labels:gs,datasets:[{\n')
        emit('      label:"val-train",data:gv,backgroundColor:bg,borderColor:brd,borderWidth:1\n')
        emit('    }]},\n')
        emit('    options:{responsive:true,maintainAspectRatio:false,animation:false,\n')
        emit('      plugins:{legend:{display:false},\n')
        emit('               tooltip:{callbacks:{label:function(c){\n')
        emit('                 return(c.raw>0?"Surapprentissage":"Sous-appr.")+": "+c.raw.toFixed(5);\n')
        emit('               }}}},\n')
        emit('      scales:{\n')
        emit('        x:{title:{display:true,text:"Step (validation)"},ticks:{maxTicksLimit:15}},\n')
        emit('        y:{title:{display:true,text:"Écart (val-train)"},\n')
        emit('           grid:{color:function(c){\n')
        emit('             return c.tick&&c.tick.value===0?"#8b949e":"#30363d";\n')
        emit('           }}}\n')
        emit('      }}\n')
        emit('  });\n')
        emit('})();</script>\n')
      end
    end

    -- Val MSE (métrique secondaire) : scatter depuis val_pts si disponible.
    -- val_mse = eps-space MSE pour DDPM, KL pour VAE.
    -- val_step n'est pas une métrique à tracer (index de step uniquement).
    do
      local has_vm = false
      for _, p in ipairs(val_pts) do
        if p.val_mse then has_vm = true; break end
      end
      if has_vm then
        local vmsc = {}
        for _, p in ipairs(val_pts) do
          if p.val_mse then
            vmsc[#vmsc+1] = string.format("{x:%.7g,y:%.7g}", p.step, p.val_mse)
          end
        end
        local vm_js = "[" .. table.concat(vmsc, ",") .. "]"
        chart_meta["c_val_mse"] = '{type:"val_mse_scatter"}'
        emit('<div class="card full">\n')
        emit('<h3>Val MSE')
        emit(' <small style="font-weight:normal;color:var(--muted)">')
        emit('métrique secondaire &mdash; eps-MSE (DDPM) / KL (VAE)')
        emit('</small></h3>\n')
        emit('<canvas id="c_val_mse"></canvas></div>\n')
        emit('<script>\n')
        emit('window._sg_charts["c_val_mse"]=new Chart(document.getElementById("c_val_mse"),{\n')
        emit('  type:"scatter",\n')
        emit('  data:{datasets:[{label:"Val MSE",data:' .. vm_js .. ',\n')
        emit('    borderColor:"' .. PALETTE.val_recon .. '",\n')
        emit('    backgroundColor:"' .. PALETTE.val_recon .. 'CC",\n')
        emit('    pointRadius:6,pointHoverRadius:9,showLine:false}]},\n')
        emit('  options:{responsive:true,maintainAspectRatio:false,animation:false,\n')
        emit('    plugins:{legend:{display:false}},\n')
        emit('    scales:{x:{title:{display:true,text:"Step"},ticks:{maxTicksLimit:10}},\n')
        emit('            y:{title:{display:true,text:"Val MSE"}}}}\n')
        emit('});\n</script>\n')
      end
    end

    -- Autres colonnes val_* : exclure val_loss (overlay), val_mse (scatter ci-dessus),
    -- val_step (index de step, pas une métrique significative).
    for _, vc in ipairs(val_cols) do
      if vc ~= "val_loss" and vc ~= "val_mse" and vc ~= "val_step" then
        emit(card_line("c_" .. vc:gsub("[^%w]","_"),
          vc:gsub("^val_", "Val "), vc, PALETTE.val_loss, false))
      end
    end

    emit("</div></section>\n")
  end

  -- Section: loss par epoch
  if #ep_ds > 0 then
    emit("\n<section><h2>Loss par epoch</h2><div class='grid'>\n")
    emit("<div class='card full'><h3>Loss par epoch</h3>")
    emit("<canvas id='c_ep'></canvas></div>\n")
    emit(string.format(
      "<script>new Chart(document.getElementById('c_ep'),{type:'scatter',"
      .. "data:{datasets:[%s]},"
      .. "options:{responsive:true,maintainAspectRatio:false,animation:false,"
      .. "plugins:{legend:{display:%s,labels:{boxWidth:10,font:{size:8}}}},"
      .. "scales:{x:{title:{display:true,text:'Step'},ticks:{maxTicksLimit:10}},"
      .. "y:{title:{display:true,text:'Loss'}}},"
      .. "elements:{point:{radius:0}}}});</script>\n",
      table.concat(ep_ds, ","),
      (#epochs <= 20) and "true" or "false"))
    emit("</div></section>\n")
  end

  -- Section: distributions
  local has_dist = (hlbl_loss and #hlbl_loss > 0) or (hlbl_rc and #hlbl_rc > 0)
  if has_dist then
    emit("\n<section><h2>Distributions</h2><div class='grid'>\n")
    emit(card_bar("c_hl",  "Distribution Loss",       hlbl_loss, hcnt_loss, PALETTE.loss))
    if rc then
      emit(card_bar("c_hrc", "Distribution " .. rl,   hlbl_rc,  hcnt_rc,   PALETTE.recon))
    end
    emit("</div></section>\n")
  end

  -- Section: scatter / relations
  local has_scat = has("timestep") or (rc and has(rc)) or has("learning_rate")
  if has_scat then
    emit("\n<section><h2>Relations</h2><div class='grid'>\n")
    emit(card_scatter("c_sc1", "Loss vs Timestep",
      "timestep", "loss", PALETTE.timestep, "Timestep", "Loss"))
    if rc then
      emit(card_scatter("c_sc2", "Loss vs " .. rl,
        rc, "loss", PALETTE.recon, rl, "Loss"))
    end
    emit(card_scatter("c_sc3", "Loss vs LR",
      "learning_rate", "loss", PALETTE.lr, "Learning Rate", "Loss"))
    if rc and has("timestep") then
      emit(card_scatter("c_sc4", rl .. " vs Timestep",
        "timestep", rc, PALETTE.temporal, "Timestep", rl))
    end
    if has("kl_divergence") and has("timestep") then
      emit(card_scatter("c_sc5", "KL vs Timestep",
        "timestep", "kl_divergence", PALETTE.kl, "Timestep", "KL"))
    end
    emit("</div></section>\n")
  end

  -- Section: corrélation
  if corr_tbl ~= "" then
    emit("\n<section><h2>Corrélation (Pearson)</h2><div class='grid'>\n")
    emit("<div class='card full'><div class='scroll'><table class='corr'>\n")
    emit(corr_tbl)
    emit("</table></div></div>\n</div></section>\n")
  end

  -- Section: statistiques tabulaires
  emit("\n<section><h2>Statistiques</h2><div class='grid'>\n")
  emit("<div class='card full'><table class='stats'>\n")
  emit("<tr><th>Métrique</th><th>Min</th><th>Max</th><th>μ (mean)</th>")
  emit("<th>σ (std)</th><th>N</th></tr>\n")
  local function stat_row(lbl, cname)
    if not has(cname) then return end
    local s = col_stats(df[cname])
    emit(string.format(
      "<tr><td>%s</td><td>%.5g</td><td>%.5g</td><td>%.5g</td><td>%.5g</td><td>%d</td></tr>\n",
      lbl, s.min, s.max, s.mean, s.std, s.n))
  end
  stat_row("Loss",                "loss")
  if rc then stat_row(rl,         rc)               end
  stat_row("Learning Rate",       "learning_rate")
  stat_row("KL Divergence",       "kl_divergence")
  stat_row("Wasserstein",         "wasserstein")
  stat_row("Entropy Δ",           "entropy_diff")
  stat_row("Moment Mismatch",     "moment_mismatch")
  stat_row("Spatial Coherence",   "spatial_coherence")
  stat_row("Temporal Consistency","temporal_consistency")
  stat_row("Timestep",            "timestep")
  stat_row("opt_eps",             "opt_eps")
  emit("</table></div>\n</div></section>\n")

  emit("\n<footer>Généré par <b>show-graph.lua</b> — Mímir Framework</footer>\n")

  -- Émettre le registre de méta-données de charts pour _sg_apply() (côté JS).
  do
    local parts = {}
    for id, ms in pairs(chart_meta) do
      parts[#parts+1] = js_s(id) .. ":" .. ms
    end
    emit("<script>window._sg_chart_meta={" .. table.concat(parts, ",") .. "};</script>\n")
  end

  -- Listener SSE : mise à jour DOM partielle (pas de location.reload()).
  -- Fallback timer : reload complet si pas de serveur SSE.
  if ctx.sse_port and ctx.sse_port > 0 then
    -- _sg_fetch_update() définie dans le <script> d'init du <head>.
    emit('<script>\n')
    emit(string.format(
      'var _es=new EventSource("http://127.0.0.1:%d/events");\n', ctx.sse_port))
    emit('_es.onmessage=function(e){if(e.data==="update") _sg_fetch_update();};\n')
    emit('</script>\n')
  elseif ctx.watch_interval and ctx.watch_interval > 0 then
    -- Fallback timer : reload complet (sans serveur SSE)
    emit('<script>(function(){\n')
    emit('function _sg_reload(){\n')
    emit('  document.body.style.transition="opacity 0.3s ease,transform 0.3s ease";\n')
    emit('  document.body.style.opacity="0";\n')
    emit('  document.body.style.transform="translateY(-4px)";\n')
    emit('  setTimeout(function(){location.reload();},600);\n')
    emit('}\n')
    emit(string.format('setTimeout(_sg_reload,%d*1000);\n', ctx.watch_interval))
    emit('})();</script>\n')
  end

  emit("</body>\n</html>\n")

  return table.concat(out)
end

-- ══════════════════════════════════════════════════════════════
-- ARGUMENT PARSING
-- ══════════════════════════════════════════════════════════════

local function parse_args()
  local raw = rawget(_G, "arg") or {}
  local opts = {
    csv                   = nil,
    csv_dir               = nil,
    model                 = nil,
    algo                  = nil,
    checkpoint_dir        = nil,
    no_interactive        = false,
    out                   = "./graph_report.html",
    watch                 = false,
    watch_interval        = 2,
    validate_every_steps  = nil,
    validate_items        = nil,
    n_dataset             = nil,
  }
  local pos = {}
  local i = 1
  while i <= #raw do
    local a = raw[i]
    if a == "-h" or a == "--help" then
      io.write([[
show-graph.lua — Visualisation métriques d'entraînement
Génère un rapport HTML interactif (Chart.js).

Usage:
  ./bin/mimir --lua scripts/tools/show-graph.lua -- [CSV] [OPTIONS]
  lua scripts/tools/show-graph.lua [CSV] [OPTIONS]

OPTIONS:
  [CSV]                    CSV (défaut: checkpoints/loss_history.csv)
                           Accepte aussi les patterns *part[0-9].csv (fusion auto)
  --csv PATH               Alias (idem que positional)
  --csv-dir DIR            Dossier contenant des CSV *part[0-9].csv à fusionner
  --model NAME             Nom du modèle
  --algo NAME              Algo/loss (mse, l1, charbonnier, huber, ...)
  --checkpoint-dir DIR     Dossier run checkpoint (auto-détection si absent)
  -n, --no-interactive     Pas de prompts stdin
  --out PATH               HTML de sortie (défaut: ./graph_report.html)
  --watch                  Surveille le CSV et régénère le rapport à chaque changement
                           Ouvre le navigateur automatiquement au démarrage
  --watch-interval N       Intervalle de polling en secondes (défaut: 2)
  --validate-every-steps N Fréquence de validation (en opt_steps) — auto-détecté si absent
  --validate-items N       Nombre d'images évaluées par validation — auto-détecté si absent
  --n-dataset N            Taille du dataset (items par epoch) — auto-détecté si absent
  -h, --help               Cette aide
]])
      os.exit(0)
    elseif a == "--csv"            then opts.csv            = raw[i+1]; i = i+1
    elseif a:match("^--csv=")     then opts.csv            = a:sub(7)
    elseif a == "--csv-dir"        then opts.csv_dir        = raw[i+1]; i = i+1
    elseif a:match("^--csv%-dir=") then opts.csv_dir        = a:match("^--csv%-dir=(.+)")
    elseif a == "--model"          then opts.model          = raw[i+1]; i = i+1
    elseif a:match("^--model=")   then opts.model          = a:sub(9)
    elseif a == "--algo"           then opts.algo           = raw[i+1]; i = i+1
    elseif a:match("^--algo=")    then opts.algo           = a:sub(8)
    elseif a == "--checkpoint-dir" then opts.checkpoint_dir = raw[i+1]; i = i+1
    elseif a:match("^--checkpoint%-dir=") then
      opts.checkpoint_dir = a:match("^--checkpoint%-dir=(.+)")
    elseif a == "--out"            then opts.out            = raw[i+1]; i = i+1
    elseif a:match("^--out=")     then opts.out            = a:sub(7)
    elseif a == "--watch"          then opts.watch          = true
    elseif a == "--watch-interval" then
      opts.watch_interval = tonumber(raw[i+1]) or 2; i = i+1
    elseif a:match("^--watch%-interval=") then
      opts.watch_interval = tonumber(a:match("^--watch%-interval=(.+)")) or 2
    elseif a == "-n" or a == "--no-interactive" then
      opts.no_interactive = true
    elseif a == "--validate-every-steps" then
      opts.validate_every_steps = tonumber(raw[i+1]); i = i+1
    elseif a:match("^--validate%-every%-steps=") then
      opts.validate_every_steps = tonumber(a:match("^--validate%-every%-steps=(.+)"))
    elseif a == "--validate-items" then
      opts.validate_items = tonumber(raw[i+1]); i = i+1
    elseif a:match("^--validate%-items=") then
      opts.validate_items = tonumber(a:match("^--validate%-items=(.+)"))
    elseif a == "--n-dataset" then
      opts.n_dataset = tonumber(raw[i+1]); i = i+1
    elseif a:match("^--n%-dataset=") then
      opts.n_dataset = tonumber(a:match("^--n%-dataset=(.+)"))
    elseif not a:match("^%-")     then pos[#pos+1] = a
    end
    i = i+1
  end
  if not opts.csv and pos[1] then opts.csv = pos[1] end
  return opts
end

-- ══════════════════════════════════════════════════════════════
-- SERVEUR SSE LOCAL (mode --watch)
-- ══════════════════════════════════════════════════════════════

-- Trouve un port TCP libre en lisant /proc/net/tcp (Linux).
local function find_free_port(start)
  start = start or 7700
  for p = start, start + 200 do
    local hex = string.format("%04X", p):upper()
    local free = true
    local f = io.open("/proc/net/tcp", "r")
    if f then
      for line in f:lines() do
        if line:find(hex, 1, true) then free = false; break end
      end
      f:close()
    end
    if free then return p end
  end
  return start
end

-- Lance un serveur HTTP+SSE Python en arrière-plan.
-- Sert `serve_dir` statiquement et expose GET /events (SSE)
-- qui envoie "reload" quand le fichier signal change.
-- Retourne le chemin du fichier signal.
local function start_sse_server(serve_dir, port)
  local sig_file = string.format("/tmp/sg_sig_%d", port)
  write_file(sig_file, "init")

  local py_src = string.format(
[[import http.server, time, socketserver

PORT = %d
DIRECTORY = %q
SIGNAL = %q

class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=DIRECTORY, **k)
    def do_GET(self):
        if self.path == '/events':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()
            last = None
            while True:
                try:
                    with open(SIGNAL) as f:
                        v = f.read().strip()
                    if v != last:
                        last = v
                        if last != 'init':
                            self.wfile.write(b'data: update\n\n')
                            self.wfile.flush()
                    time.sleep(0.25)
                except Exception:
                    break
            return
        elif self.path == '/graph_data.json':
            import os as _os
            fp = _os.path.join(DIRECTORY, 'graph_data.json')
            try:
                with open(fp, 'rb') as f:
                    body = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception:
                self.send_response(404)
                self.end_headers()
            return
        return super().do_GET()
    def do_POST(self):
        if self.path == '/snapshot':
            import os as _os, shutil, datetime, json as _json
            ts = datetime.datetime.now().strftime('%%Y%%m%%d_%%H%%M%%S')
            src = _os.path.join(DIRECTORY, 'graph_report.html')
            dst = _os.path.join(DIRECTORY, 'graph_report_' + ts + '.html')
            try:
                shutil.copy2(src, dst)
                msg = _json.dumps({'saved': dst}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Length', str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
            except Exception as ex:
                err = _json.dumps({'error': str(ex)}).encode()
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Length', str(len(err)))
                self.end_headers()
                self.wfile.write(err)
            return
        self.send_response(405)
        self.end_headers()
    def log_message(self, *a): pass

class TS(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

TS(('127.0.0.1', PORT), H).serve_forever()
]], port, serve_dir, sig_file)

  local py_path = string.format("/tmp/sg_sse_%d.py", port)
  write_file(py_path, py_src)
  os.execute(string.format("python3 %q >/dev/null 2>&1 &", py_path))
  sleep(1)  -- laisser le serveur démarrer
  return sig_file
end

-- ══════════════════════════════════════════════════════════════
-- MAIN
-- ══════════════════════════════════════════════════════════════

local function main()
  local opts = parse_args()

  -- ── Résolution des chemins CSV ──────────────────────────────────────────
  -- Priorité : --csv-dir > --csv > auto-détection
  local csv_paths = nil   -- liste de chemins à charger (et fusionner si > 1)

  if opts.csv_dir then
    -- Mode dossier : cherche tous les *part[0-9].csv dedans
    local parts = find_part_csvs(opts.csv_dir)
    if parts and #parts > 0 then
      csv_paths = parts
      io.stderr:write(string.format("📂 CSV dir : %s (%d part(s) détectées)\n",
        opts.csv_dir, #parts))
    else
      -- Fallback : tous les .csv du dossier
      csv_paths = {}
      for _, name in ipairs(FS.list_dir(opts.csv_dir)) do
        if name:match("%.csv$") then
          csv_paths[#csv_paths + 1] = FS.join(opts.csv_dir, name)
        end
      end
      if #csv_paths == 0 then
        io.stderr:write("❌ Aucun CSV trouvé dans : " .. opts.csv_dir .. "\n")
        os.exit(1)
      end
      table.sort(csv_paths)
    end
  elseif opts.csv then
    -- Un chemin donné : peut être une *part.csv → expansion auto
    csv_paths = expand_part_csvs(opts.csv)
    if #csv_paths > 1 then
      io.stderr:write(string.format("🗂️  Pattern part CSV : %d fichier(s) détectés\n", #csv_paths))
    end
  else
    -- Auto-détection
    for _, c in ipairs({
      repo_root() .. "checkpoints/loss_history.csv",
      repo_root() .. "checkpoint/loss_history.csv",
      "./loss_history.csv",
    }) do
      if file_exists(c) then csv_paths = { c }; break end
    end
    if not csv_paths then
      csv_paths = { repo_root() .. "checkpoints/loss_history.csv" }
    end
  end

  -- ── Auto-détection checkpoint ──────────────────────────────────────────
  local meta = {}
  local ckpt_dir = opts.checkpoint_dir
  if not ckpt_dir then
    ckpt_dir = csv_paths[1]:match("^(.*checkpoint/[^/]+)") or find_latest_run()
  end
  if ckpt_dir then meta = load_meta(ckpt_dir) end

  local model        = opts.model or meta.model
  local algo         = opts.algo  or meta.algo
  local algo_details = (not opts.algo) and meta.algo_details or nil

  -- ── Auto-détection des paramètres de calibration (pré-parse du 1er CSV) ────────
  local cal = {
    validate_every_steps = opts.validate_every_steps,
    validate_items       = opts.validate_items,
    n_dataset            = opts.n_dataset,
  }
  do
    local settings = load_ui_settings()
    -- Charger depuis viz_ui_settings.json (sous-objet "showgraph") si non fourni en CLI
    local sg = type(settings.showgraph) == "table" and settings.showgraph or {}
    if not cal.validate_every_steps then
      cal.validate_every_steps = tonumber(sg.validate_every_steps)
    end
    if not cal.validate_items then
      cal.validate_items = tonumber(sg.validate_items)
    end
    if not cal.n_dataset then
      cal.n_dataset = tonumber(sg.n_dataset)
    end

    -- Tenter l'auto-détection depuis le 1er CSV disponible
    local first_csv = csv_paths[1]
    if file_exists(first_csv) then
      local d_pre = parse_csv(first_csv)
      if d_pre then
        local det = detect_validation_params(d_pre.df, d_pre.n)
        if not cal.validate_every_steps and det.validate_every_steps then
          cal.validate_every_steps = det.validate_every_steps
          io.stderr:write(string.format("🔍 Auto-détecté validate_every_steps=%d\n", cal.validate_every_steps))
        end
        if not cal.validate_items and det.validate_items then
          cal.validate_items = det.validate_items
          io.stderr:write(string.format("🔍 Auto-détecté validate_items=%d\n", cal.validate_items))
        end
        if not cal.n_dataset and det.n_dataset then
          cal.n_dataset = det.n_dataset
          io.stderr:write(string.format("🔍 Auto-détecté n_dataset=%d\n", cal.n_dataset))
        end
      end
    end
  end

  -- ── Prompts interactifs ────────────────────────────────────────────────
  if not opts.no_interactive and not opts.watch then
    if not model then
      io.stderr:write("Modèle entraîné (laisser vide si inconnu) : ")
      io.stderr:flush()
      local v = io.read("*l"); if v and v ~= "" then model = v end
    end
    if not algo then
      io.stderr:write("Algo/loss (mse, l1, charbonnier, ... ; vide = inconnu) : ")
      io.stderr:flush()
      local v = io.read("*l"); if v and v ~= "" then algo = v end
    end
    -- Calibration : demander si non trouvé automatiquement
    if not cal.validate_every_steps then
      io.stderr:write("Fréquence de validation (en opt_steps, ex: 50 ; vide = inconnu) : ")
      io.stderr:flush()
      local v = io.read("*l"); cal.validate_every_steps = tonumber(v) or nil
    end
    if not cal.validate_items then
      io.stderr:write("Nombre d'items par validation (ex: 6 ; vide = inconnu) : ")
      io.stderr:flush()
      local v = io.read("*l"); cal.validate_items = tonumber(v) or nil
    end
    if not cal.n_dataset then
      io.stderr:write("Taille dataset / epoch (items, ex: 4634 ; vide = inconnu) : ")
      io.stderr:flush()
      local v = io.read("*l"); cal.n_dataset = tonumber(v) or nil
    end
    -- Persister dans viz_ui_settings.json (sous-objet "showgraph", autres clés préservées)
    do
      local settings = load_ui_settings()
      local sg = type(settings.showgraph) == "table" and settings.showgraph or {}
      local changed = false
      if cal.validate_every_steps and tostring(sg.validate_every_steps) ~= tostring(cal.validate_every_steps) then
        sg.validate_every_steps = tostring(cal.validate_every_steps); changed = true
      end
      if cal.validate_items and tostring(sg.validate_items) ~= tostring(cal.validate_items) then
        sg.validate_items = tostring(cal.validate_items); changed = true
      end
      if cal.n_dataset and tostring(sg.n_dataset) ~= tostring(cal.n_dataset) then
        sg.n_dataset = tostring(cal.n_dataset); changed = true
      end
      if changed then
        patch_ui_settings({ showgraph = sg })
        io.stderr:write("💾 Paramètres de calibration sauvegardés dans viz_ui_settings.json\n")
      end
    end
  end

  -- ── Vérification existence CSV ─────────────────────────────────────────
  for _, p in ipairs(csv_paths) do
    if not file_exists(p) then
      io.stderr:write("❌ Fichier introuvable: " .. p .. "\n")
      io.stderr:write("   Vérifiez que le modèle a été entraîné.\n")
      os.exit(1)
    end
  end

  -- Upvalues partagées entre generate() et le bloc watch
  local sse_port   = nil   -- port du serveur SSE (nil = pas de SSE)
  local sig_file   = nil   -- fichier signal → déclenche reload côté browser
  local http_url   = nil   -- URL HTTP à ouvrir (remplace file://)
  local html_written = false  -- true après le premier rendu HTML complet

  -- Démarrage du serveur SSE AVANT le premier rendu :
  -- le HTML généré contiendra directement le bon port SSE.
  if opts.watch then
    local html_dir = opts.out:match("^(.*[/\\])") or "."
    html_dir = html_dir:gsub("[/\\]+$", "")
    if html_dir == "" then html_dir = "." end
    local port = find_free_port(7700)
    sig_file = start_sse_server(html_dir, port)
    sse_port = port
    local html_name = opts.out:match("[^/\\]+$") or "graph_report.html"
    http_url = string.format("http://127.0.0.1:%d/%s", port, html_name)
    io.stderr:write(string.format("🔌 Serveur SSE : http://127.0.0.1:%d\n", port))
  end

  -- ── Fonction de (re)génération du rapport ─────────────────────────────
  local function generate()
    -- Charger + fusionner
    local parts_data = {}
    for _, p in ipairs(csv_paths) do
      local d, err = parse_csv(p)
      if not d then
        io.stderr:write("❌ Erreur CSV (" .. p .. "): " .. (err or "?") .. "\n")
        return false
      end
      parts_data[#parts_data+1] = d
    end
    local data = merge_csvs(parts_data)
    local df = data.df

    -- Points de validation (val_loss sparse) pour le graphe d'écart
    -- (calculés sur le df complet avant filtrage, afin d'avoir le train_loss le plus proche)
    local val_pts = extract_val_points(df, data.n)
    -- Retirer les étapes de validation des métriques d'entraînement
    local train_df, train_n, _val_excl = filter_train_rows(df, data.headers, data.n)

    if data.parts and data.parts > 1 then
      io.stderr:write(string.format("🗂️  Fusion de %d parties → %d steps total\n",
        data.parts, data.n))
    end
    local val_skipped = data.n - train_n
    io.stderr:write(string.format("📊 %d steps d'entraînement%s | colonnes: %s\n", train_n,
      val_skipped > 0 and string.format(" (%d lignes val exclues)", val_skipped) or "",
      table.concat(data.headers, ", ")))

    -- Frontières d'epoch dans les données d'entraînement (opt_step comme abscisse réelle)
    local epoch_boundaries = {}  -- { { step=N, epoch=E }, ... }
    local ep_x_col = (train_df.opt_step and #train_df.opt_step > 0) and "opt_step" or "step"
    if train_df.epoch and train_df[ep_x_col] then
      local prev_e = nil
      for i = 1, train_n do
        local e = train_df.epoch[i]
        local s = train_df[ep_x_col][i]
        if type(e) == "number" and e ~= prev_e then
          if prev_e ~= nil then
            epoch_boundaries[#epoch_boundaries+1] = { step = s, epoch = e }
          end
          prev_e = e
        end
      end
    end

    -- Colonne reconstruction
    local recon_col = train_df.mse and #train_df.mse > 0 and "mse" or nil
    local algo_norm = (algo or ""):lower()
    local recon_label
    if recon_col then
      recon_label = (algo_norm ~= "" and algo_norm ~= "mse")
        and ("Reconstruction (" .. algo_norm .. ")") or "MSE"
    else
      recon_label = "Reconstruction"
    end

    -- Liste des epochs
    local ep_set, ep_list = {}, {}
    if train_df.epoch then
      for _, e in ipairs(train_df.epoch) do
        if type(e) == "number" and not ep_set[e] then
          ep_set[e] = true; ep_list[#ep_list+1] = e
        end
      end
      table.sort(ep_list)
    end

    local step_min = (train_df.step and #train_df.step > 0) and train_df.step[1]        or 0
    local step_max = (train_df.step and #train_df.step > 0) and train_df.step[#train_df.step] or 0
    io.stderr:write(string.format("   Epochs: %d | Steps: %d → %d\n",
      #ep_list, step_min, step_max))

    -- Répertoire de sortie (partagé par HTML et graph_data.json)
    local json_dir = opts.out:match("^(.*[/\\])") or ""
    if json_dir == "" then json_dir = "./" end
    local json_path = json_dir .. "graph_data.json"

    -- ── Écriture graph_data.json (mise à jour DOM partielle) ──────────────
    local function flush_json()
      local jdata = gen_data_json(train_df, train_n, val_pts)
      local jtmp = json_path .. ".tmp"
      if write_file(jtmp, jdata) then os.rename(jtmp, json_path) end
    end

    -- ── En mode watch après le premier rendu : JSON seul ──────────────────
    if html_written then
      flush_json()
      -- Notifier le navigateur → _sg_fetch_update() met à jour les charts
      if sig_file then write_file(sig_file, tostring(os.time())) end
      io.stderr:write(string.format("🔄 %s mis à jour\n", json_path))
      return true
    end

    -- ── Premier rendu (ou mode one-shot) : HTML complet ───────────────────
    io.stderr:write("🖌️  Génération du rapport HTML...\n")

    local total_bytes = 0
    for _, p in ipairs(csv_paths) do total_bytes = total_bytes + file_size(p) end

    local html = gen_html({
      csv_path       = #csv_paths == 1 and csv_paths[1]
                       or (csv_paths[1]:match("^(.*[/\\])") or "./") .. "["
                       .. #csv_paths .. " parts]",
      csv_size       = total_bytes,
      model          = model,
      algo           = algo,
      algo_details   = algo_details,
      meta           = meta,
      df                   = train_df,
      headers              = data.headers,
      n                    = train_n,
      epoch_boundaries     = epoch_boundaries,
      validate_every_steps = cal.validate_every_steps,
      validate_items       = cal.validate_items,
      n_dataset            = cal.n_dataset,
      epochs         = ep_list,
      recon_col      = recon_col,
      recon_label    = recon_label,
      val_pts        = val_pts,
      sse_port       = sse_port,
      watch_interval = (not sse_port and opts.watch) and opts.watch_interval or nil,
    })

    local tmp = opts.out .. ".tmp"
    if write_file(tmp, html) then
      os.rename(tmp, opts.out)
      html_written = true
      flush_json()
      io.stderr:write(string.format("💾 %s (%s)\n", opts.out, human_bytes(#html)))
      return true
    else
      io.stderr:write("❌ Impossible d'écrire : " .. opts.out .. "\n")
      return false
    end
  end

  -- ── Banner ─────────────────────────────────────────────────────────────
  local SEP = string.rep("═", 59)
  io.stderr:write(SEP .. "\n")
  io.stderr:write("        SHOW GRAPH — Mímir training metrics\n")
  io.stderr:write(SEP .. "\n")
  for _, p in ipairs(csv_paths) do
    io.stderr:write(string.format("📄 CSV     : %s (%s)\n", p, human_bytes(file_size(p))))
  end
  io.stderr:write(string.format("🧠 Modèle  : %s\n", model or "(inconnu)"))
  if algo then
    local a = algo .. (algo_details and (" [" .. algo_details .. "]") or "")
    io.stderr:write("📐 Algo    : " .. a .. "\n")
  end
  if meta.source then
    io.stderr:write("📦 Source  : " .. meta.source .. "\n")
  end
  if opts.watch then
    io.stderr:write(string.format("👁️  Watch   : intervalle %ds\n", opts.watch_interval))
  end
  io.stderr:write(SEP .. "\n\n")

  -- ── Premier rendu ──────────────────────────────────────────────────────
  io.stderr:write("⏳ Chargement du CSV...\n")
  local ok = generate()
  if not ok then os.exit(1) end

  if opts.watch then
    -- Ouvre le navigateur (http:// via serveur SSE, ou file:// en fallback)
    open_browser(http_url or opts.out)
    io.stderr:write("👁️  Surveillance SSE active — Ctrl+C pour arrêter\n")
    if http_url then
      io.stderr:write(string.format("   URL : %s\n", http_url))
    end

    local last_mtime = mtimes_combined(csv_paths)
    while true do
      sleep(opts.watch_interval)
      local cur = mtimes_combined(csv_paths)
      if cur ~= last_mtime then
        last_mtime = cur
        io.stderr:write("🔄 Changement détecté, régénération...\n")
        generate()  -- met à jour HTML + sig_file → SSE → browser
      end
    end
  else
    io.stderr:write(string.format("🌐 Ouvrir : xdg-open %s\n", opts.out))
    io.stderr:write("\n✓ Terminé.\n")
  end
end

main()
