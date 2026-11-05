-- Pure resolver regression: numeric epochs and partially written checkpoints.
local original_dofile = dofile
local candidates = {"epoch_9", "epoch_10", "epoch_11", "epoch_999_notes"}
local valid = { ["runs/epoch_9/model/architecture.json"] = true,
                ["runs/epoch_10/model/architecture.json"] = true }
local fs = {
  join = function(...) return table.concat({...}, "/") end,
  list_dir = function() return candidates end,
  is_dir = function() return true end,
  file_exists = function(path)
    if path:match("/training.json$") then return valid[path:gsub("training.json$", "architecture.json")] == true end
    return valid[path] == true
  end,
}
ROOTWORK = ROOTWORK or "."
dofile = function(path)
  if path:match("/scripts/modules/fs.lua$") then return fs end
  return original_dofile(path)
end
local ckpt = original_dofile(ROOTWORK .. "/scripts/modules/checkpoint_resume.lua")
dofile = original_dofile
assert(ckpt.resolve_dir("runs") == "runs/epoch_10")
assert(ckpt.resolve_dir("runs/epoch_10") == "runs/epoch_10")
assert(ckpt.resolve_dir("missing") == nil)
valid["runs/final/model/architecture.json"] = true
local original_read_json = read_json
read_json = function(path)
  if path == "runs/final/model/training.json" then return {step=15} end
  return {step=10}
end
assert(ckpt.resolve_dir("runs") == "runs/final")
valid["runs/model/architecture.json"] = true
read_json = function(path)
  if path == "runs/model/training.json" then return {step=20} end
  return {step=10}
end
assert(ckpt.resolve_dir("runs") == "runs")
valid["runs/model/architecture.json"] = nil
valid["runs/final/model/architecture.json"] = nil
candidates[#candidates+1] = "step_20"
valid["runs/step_20/model/architecture.json"] = true
read_json = function(path)
  return {step=path:match("step_20") and 20 or 10}
end
assert(ckpt.resolve_dir("runs") == "runs/step_20")
read_json = original_read_json
print("CHECKPOINT_RESUME_RESOLVER_OK")
