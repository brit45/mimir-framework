-- Checkpoint resume helper for raw_folder epoch_*, step_*, step-*, final and root layouts.
-- Usage:
--   local Ckpt = dofile(ROOTWORK.."/scripts/modules/checkpoint_resume.lua")
--   local dir = Ckpt.resolve_dir("checkpoint/MyModel")

---@class MimirCheckpointResumeModule
local M = {}
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")

local function file_exists(path)
  return FS.file_exists(path)
end

local function looks_like_raw_folder(dir)
  return file_exists(FS.join(dir, "model", "architecture.json"))
      or file_exists(FS.join(dir, "model", "model.safetensors"))
      or file_exists(FS.join(dir, "model.safetensors"))
end

local function find_latest_epoch_dir(base)
  base = tostring(base or "")
  if #base == 0 then return nil end

  local entries = FS.list_dir(base)
  local epochs = {}
  for _, name in ipairs(entries) do
    if tostring(name):match("^epoch_%d+") then
      local full = FS.join(base, name)
      if FS.is_dir(full) and looks_like_raw_folder(full) then
        epochs[#epochs + 1] = full
      end
    end
  end

  table.sort(epochs, function(a, b)
    local ea = tonumber(a:match("epoch_(%d+)[^/\\]*$")) or -1
    local eb = tonumber(b:match("epoch_(%d+)[^/\\]*$")) or -1
    if ea ~= eb then return ea < eb end
    return a < b
  end)
  if #epochs > 0 then
    return epochs[#epochs]
  end
  return nil
end

local function step_number(path_or_name)
  local name = tostring(path_or_name or "")
  return tonumber(name:match("step[_%-](%d+)[^/\\]*$")) or
         tonumber(name:match("^step[_%-](%d+)$")) or -1
end

function M.file_exists(path)
  return file_exists(path)
end

function M.find_latest_epoch_dir(base)
  return find_latest_epoch_dir(base)
end

-- Compare actual progress when metadata is available. A final/root checkpoint
-- may be newer than the last periodic epoch or step checkpoint.
local function checkpoint_step(path)
  if type(read_json) ~= "function" then return -1 end
  local training_path = FS.join(path, "model", "training.json")
  if not file_exists(training_path) then return -1 end
  local ok, state = pcall(read_json, training_path)
  if not ok or type(state) ~= "table" then return -1 end
  return tonumber(state.step) or
    (type(state.loop) == "table" and tonumber(state.loop.global_step)) or -1
end

function M.resolve_dir(base)
  base = tostring(base or "")
  if #base == 0 then return nil end
  local best = find_latest_epoch_dir(base)
  local best_step = best and checkpoint_step(best) or -1
  local candidates = {}
  for _, name in ipairs(FS.list_dir(base)) do
    if tostring(name):match("^step[_%-]%d+") then
      candidates[#candidates + 1] = FS.join(base, name)
    end
  end
  table.sort(candidates, function(a, b)
    return step_number(a) < step_number(b)
  end)
  candidates[#candidates + 1] = FS.join(base, "final")
  candidates[#candidates + 1] = base
  for _, path in ipairs(candidates) do
    if looks_like_raw_folder(path) then
      local step = checkpoint_step(path)
      if not best or step >= best_step then best, best_step = path, step end
    end
  end
  return best
end

return M
