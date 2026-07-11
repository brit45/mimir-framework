-- Build a tags vocabulary file from a dataset of (image + text) pairs.
--
-- It scans all `.txt` files under `--dataset-root`, splits on '.' (dot-separated tags/short phrases),
-- normalizes, counts frequencies, and writes a vocab file (one tag per line).
--
-- Usage:
--   ./bin/mimir --lua scripts/tools/build_tags_vocab.lua -- \
--     --dataset-root dataset_2 \
--     --out checkpoint/tags_vocab.txt \
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

local dataset_root = opt_str("dataset-root", "dataset_2")
local out_path = opt_str("out", "checkpoint/tags_vocab.txt")
local lowercase = opt_bool("lowercase", true)
local min_freq = opt_int("min-freq", 1)
local top_k = opt_int("top-k", 0)
local max_files = opt_int("max-files", 0)

if min_freq < 1 then min_freq = 1 end
if top_k < 0 then top_k = 0 end
if max_files < 0 then max_files = 0 end

log("=== build_tags_vocab ===")
log("- dataset_root=" .. tostring(dataset_root))
log("- out=" .. tostring(out_path))
log("- lowercase=" .. tostring(lowercase))
log("- min_freq=" .. tostring(min_freq) .. " top_k=" .. tostring(top_k) .. " max_files=" .. tostring(max_files))

local files = {}
collect_txt_files(dataset_root, files)
if max_files > 0 and #files > max_files then
  local sliced = {}
  for i = 1, max_files do sliced[i] = files[i] end
  files = sliced
end

if #files == 0 then
  error("Aucun .txt trouvé sous dataset-root=" .. tostring(dataset_root))
end

log("- txt_files=" .. tostring(#files))

local freq = {}
local total_tags = 0

local function add_tag(tag)
  tag = normalize_spaces(tag)
  if tag == "" then return end
  if lowercase then tag = tag:lower() end
  freq[tag] = (freq[tag] or 0) + 1
  total_tags = total_tags + 1
end

local function process_text(txt)
  -- split on '.'
  local cur = ""
  for i = 1, #txt do
    local ch = txt:sub(i, i)
    if ch == "." then
      add_tag(cur)
      cur = ""
    else
      cur = cur .. ch
    end
  end
  add_tag(cur)
end

local read_ok = 0
local read_fail = 0

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

log("- read_ok=" .. tostring(read_ok) .. " read_fail=" .. tostring(read_fail))
log("- total_tags_seen=" .. tostring(total_tags))

local items = {}
for tag, n in pairs(freq) do
  if n >= min_freq then
    table.insert(items, {tag = tag, n = n})
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

log("✓ tags_vocab écrit: " .. tostring(out_path) .. " (classes=" .. tostring(#items) .. ")")
