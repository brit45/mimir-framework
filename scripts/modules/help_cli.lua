local M = {}

local function read_file(path)
  if type(path) ~= "string" or path == "" then return nil end
  local f = io.open(path, "r")
  if not f then return nil end
  local ok, data = pcall(function()
    return f:read("*a")
  end)
  f:close()
  if not ok then return nil end
  return data
end

local function trim(s)
  return (tostring(s or ""):gsub("^%s+", ""):gsub("%s+$", ""))
end

local function split_lines(s)
  local out = {}
  for line in tostring(s or ""):gmatch("([^\n]*)\n?") do
    if line == "" and #out > 0 and out[#out] == "" then
      break
    end
    out[#out + 1] = line
  end
  return out
end

function M.should_show_help(argv)
  local a = argv or _G.arg or {}
  for i = 1, #a do
    local v = a[i]
    if v == "--help" or v == "-h" or v == "--h" then
      return true
    end
  end
  return false
end

function M.find_script_from_stack(start_level, max_level)
  local first = start_level or 2
  local last = max_level or 12
  local this_src = debug.getinfo(1, "S").source
  for lvl = first, last do
    local info = debug.getinfo(lvl, "S")
    if info and type(info.source) == "string" then
      local src = info.source
      if src:sub(1, 1) == "@" then
        local path = src:sub(2)
        if path:sub(-4) == ".lua" and src ~= this_src and not path:match("help_cli%.lua$") then
          return path
        end
      end
    end
  end
  return nil
end

local function infer_description(path)
  local data = read_file(path)
  if not data then return nil end
  local lines = split_lines(data)
  local comments = {}
  for i = 1, math.min(#lines, 80) do
    local line = lines[i]
    if line:match("^%s*%-%-") then
      local txt = trim((line:gsub("^%s*%-%-%s?", "")))
      local is_usage = txt:match("^[Uu]sage") or txt:match("^[Oo]ptions")
      local is_directive = txt:match("^@") or txt:match("^%-@")
      local is_separator = txt:match("^[=%-%*_#%.%s]+$") ~= nil
      if txt ~= "" and not is_usage and not is_directive and not is_separator then
        comments[#comments + 1] = txt
      end
    elseif line:match("^%s*$") then
      -- keep scanning through initial blank lines
    else
      break
    end
  end
  return comments[1]
end

local function add_option(set, key)
  if not key then return end
  local k = trim(key)
  if k == "" then return end
  k = k:gsub("_", "-")
  set[k] = true
end

local function infer_options(path)
  local data = read_file(path)
  if not data then return {} end
  local set = {}

  for key in data:gmatch('opt_%w+%s*%(%s*"([%w%-%_]+)"') do
    add_option(set, key)
  end
  for key in data:gmatch('Args%.get_%w+%s*%([^\n]-"([%w%-%_]+)"') do
    add_option(set, key)
  end
  -- Convention des scripts qui centralisent les getters dans un helper :
  -- apply_cli(section, "field", "cli-name", Args.get_int)
  for key in data:gmatch('apply_cli%s*%([^\n]-"[%w%-%_]+"%s*,%s*"([%w%-%_]+)"') do
    add_option(set, key)
  end
  for key in data:gmatch('opts%s*%[%s*"([%w%-%_]+)"%s*%]') do
    add_option(set, key)
  end

  local out = {}
  for k, _ in pairs(set) do
    out[#out + 1] = "--" .. k
  end
  table.sort(out)
  return out
end

function M.print_help(params)
  local p = params or {}
  local script_path = p.script_path or M.find_script_from_stack(3, 18) or "<script.lua>"
  local description = p.description or infer_description(script_path) or "Script Lua du projet Mimir."
  local options = p.options
  if type(options) ~= "table" then
    options = infer_options(script_path)
  end

  local common = p.common_flags
  if type(common) ~= "table" then
    common = {
      "--help, -h : affiche cette aide",
    }
  end

  print("Usage:")
  print("  ./bin/mimir --lua " .. script_path .. " -- [options]")
  print("")
  print("Description:")
  print("  " .. description)
  print("")
  print("Options detectees:")
  if #options == 0 then
    print("  (aucune option detectee automatiquement dans le script)")
  else
    for i = 1, #options do
      print("  " .. options[i])
    end
  end
  print("")
  print("Flags communs:")
  for i = 1, #common do
    print("  " .. tostring(common[i]))
  end
end

function M.auto_exit_help(params)
  if M.should_show_help(_G.arg) then
    M.print_help(params)
    os.exit(0)
  end
end

return M
