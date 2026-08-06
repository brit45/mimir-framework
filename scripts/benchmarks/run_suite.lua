local Help = dofile("scripts/modules/help_cli.lua")
Help.auto_exit_help()

local function has(flag)
  for _, value in ipairs(arg or {}) do
    if value == flag then return true end
  end
  return false
end

local function shell_quote(value)
  return "'" .. tostring(value):gsub("'", "'\\''") .. "'"
end

local function project_root()
  local conf_dir = type(CONF_DIR) == "string" and CONF_DIR or ""
  if conf_dir ~= "" then return conf_dir end
  return "."
end

local suite = "all"
for i = 1, #(arg or {}) do
  if arg[i] == "--suite" and arg[i + 1] then suite = arg[i + 1] end
end

local groups = {
  smoke = {
    {"dtype API", "scripts/benchmarks/dtype_api_smoke.lua"},
    {"spill cleanup", "scripts/benchmarks/spill_cleanup_smoke.lua"},
    {"NMS", "scripts/benchmarks/benchmark_nms.lua", "--quick"},
  },
  core = {
    {"NMS", "scripts/benchmarks/benchmark_nms.lua", "--quick"},
    {"attention", "scripts/benchmarks/benchmark_attention.lua"},
    {"convolution training", "scripts/benchmarks/benchmark_conv_train.lua"},
    {"général", "scripts/benchmarks/benchmark.lua", "--quick"},
    {"complet", "scripts/benchmarks/benchmark_complet.lua"},
    {"officiel", "scripts/benchmarks/benchmark_official.lua", "--safe", "--iters", "1"},
  },
  stress = {
    {"stress", "scripts/benchmarks/benchmark_stress.lua"},
  },
}

groups.all = {
  groups.smoke[1],
  groups.smoke[2],
  groups.smoke[3],
  groups.core[2],
  groups.core[3],
  groups.core[4],
  groups.core[5],
  groups.core[6],
  groups.stress[1],
}

local selected = groups[suite]
assert(selected, "suite inconnue: " .. tostring(suite))

local root = project_root()
local mimir_bin = os.getenv("MIMIR_BIN") or (root .. "/bin/mimir")
local failures = {}

log(string.format("Suite de benchmarks '%s': %d étape(s)", suite, #selected))
for index, entry in ipairs(selected) do
  local command = {
    "cd", shell_quote(root), "&&", shell_quote(mimir_bin),
    "--lua", shell_quote(entry[2]),
  }
  if #entry > 2 then
    command[#command + 1] = "--"
    for i = 3, #entry do command[#command + 1] = shell_quote(entry[i]) end
  end

  log(string.format("\n[%d/%d] %s", index, #selected, entry[1]))
  local ok, reason, code = os.execute(table.concat(command, " "))
  local passed = ok == true or ok == 0
  if not passed then
    failures[#failures + 1] = string.format(
      "%s (%s %s)", entry[1], tostring(reason), tostring(code))
    if not has("--keep-going") then break end
  end
end

if #failures > 0 then
  error("benchmarks en échec: " .. table.concat(failures, ", "))
end

log(string.format("\nSuite '%s' terminée avec succès.", suite))
