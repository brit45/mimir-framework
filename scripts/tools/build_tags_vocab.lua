-- Build a tags vocabulary file from a dataset of (image + text) pairs.
--
-- It supports two input families:
--   1) text sidecars (`.txt`) under `--dataset-root`
--   2) COCO captions JSON (`captions_train2017.json` / `captions_val2017.json`)
--
-- It normalizes, counts frequencies, and writes a vocab file (one tag/token per line).
--
-- Usage:
--   ./bin/mimir --lua scripts/tools/build_tags_vocab.lua -- \
--     --dataset-root dataset_2 \
--     --out checkpoint/tags_vocab.txt \
--     --dataset-format auto \
--     --split-mode auto \
--     --lowercase true \
--     --min-freq 2 \
--     --top-k 5000
--
-- Notes:
-- - This tool only reads text files. It does not open images.
-- - Output is sorted by (freq desc, tag asc) for determinism.

local Args = dofile("scripts/modules/args.lua")
local opts = Args.parse(arg) or {}
local FS = dofile("scripts/modules/fs.lua")

local function opt_num(k, d)
  local v = opts[k]
  if v == nil then return d end
  local n = tonumber(v)
  if n == nil then return d end
  return n
end

local function opt_int(k, d)
  return math.floor(opt_num(k, d))
end

local function opt_str(k, d)
  local v = opts[k]
  if v == nil or v == true then return d end
  return tostring(v)
end

local function opt_bool(k, d)
  local v = opts[k]
  if v == nil then return d end
  if v == true or v == false then return v end
  v = tostring(v):lower()
  if v == "1" or v == "true" or v == "yes" or v == "on" then return true end
  if v == "0" or v == "false" or v == "no" or v == "off" then return false end
  return d
end

local function ensure_parent_dir(filepath)
  local dir = FS.dirname(filepath)
  if dir and #dir > 0 then
    FS.mkdir_p(dir)
  end
end

local function strip_extension(path)
  local s = tostring(path or "")
  local base = s:match("^(.*)%.([^.]+)$")
  return base or s
end

local function default_composition_out(vocab_path)
  return strip_extension(vocab_path) .. ".composition.json"
end

local function json_escape(s)
  s = tostring(s or "")
  s = s:gsub("\\", "\\\\")
  s = s:gsub('"', '\\"')
  s = s:gsub("\n", "\\n")
  s = s:gsub("\r", "\\r")
  s = s:gsub("\t", "\\t")
  return s
end

local function collect_txt_files(root, out)
  local entries = FS.list_dir(root)
  table.sort(entries)
  for _, name in ipairs(entries) do
    local full = FS.join(root, name)
    if FS.is_dir(full) then
      collect_txt_files(full, out)
    elseif name:match("%.txt$") then
      out[#out + 1] = full
    end
  end
end

local function trim(s)
  s = tostring(s or "")
  s = s:gsub("^%s+", "")
  s = s:gsub("%s+$", "")
  return s
end

local function normalize_spaces(s)
  -- Collapse whitespace to single spaces.
  s = s:gsub("[%s\t\r\n]+", " ")
  return trim(s)
end

local function decode_json_table(raw)
  local json_mod = rawget(_G, "json")
  if type(json_mod) == "table" and type(json_mod.decode) == "function" then
    local ok, v = pcall(json_mod.decode, raw)
    if ok and type(v) == "table" then return v end
  end

  local cjson_mod = rawget(_G, "cjson")
  if type(cjson_mod) == "table" and type(cjson_mod.decode) == "function" then
    local ok, v = pcall(cjson_mod.decode, raw)
    if ok and type(v) == "table" then return v end
  end

  return nil
end

local function read_all(path)
  local f = io.open(path, "r")
  if not f then return nil, "cannot open: " .. tostring(path) end
  local s = f:read("*a") or ""
  f:close()
  return s, nil
end

local function find_existing(paths)
  for _, p in ipairs(paths) do
    if FS.file_exists(p) then return p end
  end
  return nil
end

local function parent_dir(path)
  return FS.dirname(path)
end

local function basename(path)
  local p = tostring(path or "")
  local i = p:match("^.*()/")
  if not i then return p end
  return p:sub(i + 1)
end

local function guess_coco_annotations(dataset_root, explicit_path)
  if explicit_path and explicit_path ~= "" then
    return explicit_path
  end

  local root = tostring(dataset_root or "")
  local leaf = basename(root)
  local parent = parent_dir(root) or "."

  local candidates = {}
  if leaf == "train2017" then
    candidates[#candidates + 1] = FS.join(parent, "annotations/captions_train2017.json")
    candidates[#candidates + 1] = FS.join(root, "captions_train2017.json")
  elseif leaf == "val2017" then
    candidates[#candidates + 1] = FS.join(parent, "annotations/captions_val2017.json")
    candidates[#candidates + 1] = FS.join(root, "captions_val2017.json")
  else
    candidates[#candidates + 1] = FS.join(root, "annotations/captions_train2017.json")
    candidates[#candidates + 1] = FS.join(root, "captions_train2017.json")
    candidates[#candidates + 1] = FS.join(root, "annotations/captions_val2017.json")
    candidates[#candidates + 1] = FS.join(root, "captions_val2017.json")
  end

  return find_existing(candidates)
end

local function split_words(s)
  local out = {}
  local cur = {}
  local function flush()
    if #cur == 0 then return end
    out[#out + 1] = table.concat(cur)
    cur = {}
  end

  for i = 1, #s do
    local ch = s:sub(i, i)
    if ch:match("[%w_%-']") then
      cur[#cur + 1] = ch
    else
      flush()
    end
  end
  flush()
  return out
end

local dataset_root = opt_str("dataset-root", "dataset_2")
local out_path = opt_str("out", "checkpoint/tags_vocab.txt")
local lowercase = opt_bool("lowercase", true)
local min_freq = opt_int("min-freq", 1)
local top_k = opt_int("top-k", 0)
local max_files = opt_int("max-files", 0)
local dataset_format = opt_str("dataset-format", "auto") -- auto|txt|coco
local split_mode = opt_str("split-mode", "auto") -- auto|phrases|tokens|both
local coco_annotations = opt_str("coco-annotations", "")
local composition_out = opt_str("composition-out", default_composition_out(out_path))

dataset_format = tostring(dataset_format):lower()
split_mode = tostring(split_mode):lower()

if dataset_format ~= "auto" and dataset_format ~= "txt" and dataset_format ~= "coco" then
  error("dataset-format invalide (auto|txt|coco): " .. tostring(dataset_format))
end
if split_mode ~= "auto" and split_mode ~= "phrases" and split_mode ~= "tokens" and split_mode ~= "both" then
  error("split-mode invalide (auto|phrases|tokens|both): " .. tostring(split_mode))
end

if min_freq < 1 then min_freq = 1 end
if top_k < 0 then top_k = 0 end
if max_files < 0 then max_files = 0 end

log("=== build_tags_vocab ===")
log("- dataset_root=" .. tostring(dataset_root))
log("- out=" .. tostring(out_path))
log("- dataset_format=" .. tostring(dataset_format) .. " split_mode=" .. tostring(split_mode))
log("- lowercase=" .. tostring(lowercase))
log("- min_freq=" .. tostring(min_freq) .. " top_k=" .. tostring(top_k) .. " max_files=" .. tostring(max_files))
log("- composition_out=" .. tostring(composition_out))

local files = {}
collect_txt_files(dataset_root, files)

local detected_format = dataset_format
if detected_format == "auto" then
  if #files > 0 then
    detected_format = "txt"
  else
    local coco_json = guess_coco_annotations(dataset_root, coco_annotations)
    if coco_json then
      detected_format = "coco"
      coco_annotations = coco_json
    else
      error("Impossible de détecter le format dataset (ni .txt, ni captions COCO JSON)")
    end
  end
end

if detected_format == "txt" then
  if max_files > 0 and #files > max_files then
    local sliced = {}
    for i = 1, max_files do sliced[i] = files[i] end
    files = sliced
  end
  if #files == 0 then
    error("Aucun .txt trouvé sous dataset-root=" .. tostring(dataset_root))
  end
  log("- txt_files=" .. tostring(#files))
else
  if coco_annotations == "" then
    local guessed = guess_coco_annotations(dataset_root, nil)
    if guessed then coco_annotations = guessed end
  end
  if coco_annotations == "" or not FS.file_exists(coco_annotations) then
    error("Fichier COCO annotations introuvable. Utilise --coco-annotations <captions_*.json>")
  end
  log("- coco_annotations=" .. tostring(coco_annotations))
end

if split_mode == "auto" then
  if detected_format == "coco" then
    split_mode = "tokens"
  else
    split_mode = "phrases"
  end
end
log("- split_mode_effective=" .. tostring(split_mode))

local freq = {}
local item_freq = {}
local total_tags = 0
local total_samples = 0

local function add_tag(tag, seen)
  tag = normalize_spaces(tag)
  if tag == "" then return end
  if lowercase then tag = tag:lower() end
  freq[tag] = (freq[tag] or 0) + 1
  if seen and not seen[tag] then
    seen[tag] = true
    item_freq[tag] = (item_freq[tag] or 0) + 1
  end
  total_tags = total_tags + 1
end

local function process_text(txt)
  -- split on '.', puis optionnellement ajoute des tokens mot-à-mot.
  total_samples = total_samples + 1
  local seen = {}
  local cur = ""
  local function emit_piece(s)
    s = normalize_spaces(s)
    if s == "" then return end

    if split_mode == "phrases" or split_mode == "both" then
      add_tag(s, seen)
    end
    if split_mode == "tokens" or split_mode == "both" then
      local words = split_words(s)
      for _, w in ipairs(words) do
        add_tag(w, seen)
      end
    end
  end

  for i = 1, #txt do
    local ch = txt:sub(i, i)
    if ch == "." then
      emit_piece(cur)
      cur = ""
    else
      cur = cur .. ch
    end
  end
  emit_piece(cur)
end

local function unescape_json_string(s)
  s = s:gsub('\\"', '"')
  s = s:gsub("\\n", "\n")
  s = s:gsub("\\r", "\r")
  s = s:gsub("\\t", "\t")
  s = s:gsub("\\/", "/")
  s = s:gsub("\\\\", "\\")
  return s
end

local function process_coco_annotations(path)
  local raw, err = read_all(path)
  if not raw then error("lecture coco annotations échouée: " .. tostring(err)) end

  local decoded = decode_json_table(raw)
  local count = 0
  if type(decoded) == "table" and type(decoded.annotations) == "table" then
    for _, ann in ipairs(decoded.annotations) do
      if type(ann) == "table" and type(ann.caption) == "string" then
        process_text(ann.caption)
        count = count + 1
      end
    end
    return count
  end

  -- Fallback robuste si module JSON indisponible: extraction directe des champs "caption".
  for cap in raw:gmatch('"caption"%s*:%s*"(.-)"') do
    process_text(unescape_json_string(cap))
    count = count + 1
  end
  return count
end

local read_ok = 0
local read_fail = 0

if detected_format == "txt" then
  for _, path in ipairs(files) do
    local f = io.open(path, "r")
    if f then
      local content = f:read("*a") or ""
      f:close()
      process_text(content)
      read_ok = read_ok + 1
    else
      read_fail = read_fail + 1
    end
  end
else
  local n = process_coco_annotations(coco_annotations)
  read_ok = n
  read_fail = 0
end

log("- read_ok=" .. tostring(read_ok) .. " read_fail=" .. tostring(read_fail))
log("- total_tags_seen=" .. tostring(total_tags))

local items = {}
for tag, n in pairs(freq) do
  if n >= min_freq then
    table.insert(items, {
      tag = tag,
      n = n,
      item_n = item_freq[tag] or 0,
    })
  end
end

table.sort(items, function(a, b)
  if a.n ~= b.n then return a.n > b.n end
  return a.tag < b.tag
end)

if top_k > 0 and #items > top_k then
  while #items > top_k do table.remove(items) end
end

ensure_parent_dir(out_path)
local out = io.open(out_path, "w")
if not out then
  error("Impossible d'écrire: " .. tostring(out_path))
end

for _, it in ipairs(items) do
  out:write(it.tag)
  out:write("\n")
end
out:close()

if composition_out ~= "" and composition_out ~= "false" and composition_out ~= "none" then
  ensure_parent_dir(composition_out)
  local comp = io.open(composition_out, "w")
  if not comp then
    error("Impossible d'écrire la composition de classes: " .. tostring(composition_out))
  end

  local kept_total_tags = 0
  for _, it in ipairs(items) do
    kept_total_tags = kept_total_tags + (it.n or 0)
  end

  comp:write("{\n")
  comp:write("  \"dataset_root\": \"" .. json_escape(dataset_root) .. "\",\n")
  comp:write("  \"vocab_path\": \"" .. json_escape(out_path) .. "\",\n")
  comp:write("  \"dataset_format\": \"" .. json_escape(detected_format) .. "\",\n")
  comp:write("  \"split_mode\": \"" .. json_escape(split_mode) .. "\",\n")
  comp:write("  \"lowercase\": " .. tostring(lowercase) .. ",\n")
  comp:write("  \"total_samples\": " .. tostring(total_samples) .. ",\n")
  comp:write("  \"total_tags_seen\": " .. tostring(total_tags) .. ",\n")
  comp:write("  \"total_tags_kept\": " .. tostring(kept_total_tags) .. ",\n")
  comp:write("  \"num_classes\": " .. tostring(#items) .. ",\n")
  comp:write("  \"classes\": [\n")
  for i, it in ipairs(items) do
    local pos_items = math.max(0, tonumber(it.item_n) or 0)
    local neg_items = math.max(0, total_samples - pos_items)
    local raw_pos_weight = 1.0
    if pos_items > 0 then
      raw_pos_weight = neg_items / pos_items
    end
    if raw_pos_weight < 1e-6 then raw_pos_weight = 1e-6 end
    local tag_freq = 0.0
    if kept_total_tags > 0 then
      tag_freq = (it.n or 0) / kept_total_tags
    end
    local item_ratio = 0.0
    if total_samples > 0 then
      item_ratio = pos_items / total_samples
    end

    comp:write("    {\n")
    comp:write("      \"tag\": \"" .. json_escape(it.tag) .. "\",\n")
    comp:write("      \"count\": " .. tostring(it.n or 0) .. ",\n")
    comp:write("      \"sample_count\": " .. tostring(pos_items) .. ",\n")
    comp:write("      \"tag_frequency\": " .. string.format("%.12g", tag_freq) .. ",\n")
    comp:write("      \"sample_frequency\": " .. string.format("%.12g", item_ratio) .. ",\n")
    comp:write("      \"recommended_pos_weight\": " .. string.format("%.12g", raw_pos_weight) .. "\n")
    if i < #items then
      comp:write("    },\n")
    else
      comp:write("    }\n")
    end
  end
  comp:write("  ]\n")
  comp:write("}\n")
  comp:close()
  log("✓ composition de classes écrite: " .. tostring(composition_out))
end

log("✓ tags_vocab écrit: " .. tostring(out_path) .. " (classes=" .. tostring(#items) .. ")")
