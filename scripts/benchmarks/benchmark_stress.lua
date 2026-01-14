-- scripts/benchmarks/benchmark_stress.lua
-- Benchmark "loss from output-gradient" (no targets needed)
-- Loss: L = 0.5 * sum(y^2)  => dL/dy = y
-- So we can compute loss from loss_grad only (loss_grad == y)

math.randomseed(1337)

local model = Mimir.Model

-- -----------------------------
-- Helpers
-- -----------------------------
local function must(ok, err, ctx)
  if ok == false or ok == nil then
    error("❌ FAIL: " .. tostring(ctx) .. " | " .. tostring(err))
  end
end

local function is_finite(x)
  return x == x and x ~= math.huge and x ~= -math.huge
end

local function vec_l2_and_maxabs(v)
  local s = 0.0
  local m = 0.0
  for i = 1, #v do
    local a = v[i]
    local aa = (a >= 0) and a or -a
    s = s + a * a
    if aa > m then m = aa end
  end
  return math.sqrt(s), m
end

local function vec_any_nan_inf(v)
  for i = 1, #v do
    if not is_finite(v[i]) then return true end
  end
  return false
end

local function make_input(cfg)
  local n = cfg.height * cfg.width * cfg.in_channels
  local x = {}
  for i = 1, n do
    x[i] = (math.random() * 2.0 - 1.0)
  end
  return x
end

-- Loss from gradient (since grad == output):
-- L = 0.5 * sum(grad^2)
local function loss_from_grad(loss_grad)
  local s = 0.0
  for i = 1, #loss_grad do
    local g = loss_grad[i]
    s = s + g * g
  end
  return 0.5 * s
end

local function hw_stats()
  if model.hardware_caps then
    local caps = model.hardware_caps()
    if caps then
      return string.format("AVX2=%s FMA=%s F16C=%s BMI2=%s",
        tostring(caps.avx2), tostring(caps.fma), tostring(caps.f16c), tostring(caps.bmi2))
    end
  end
  return "caps=?"
end

local function guard_stats()
  if Mimir and Mimir.MemoryGuard and Mimir.MemoryGuard.getStats then
    local g = Mimir.MemoryGuard.getStats()
    if g then
      return string.format("Guard current=%.1fMB peak=%.1fMB limit=%.1fMB usage=%.1f%%",
        g.current or 0, g.peak or 0, g.limit or 0, g.usage or 0)
    end
  end
  return "Guard stats unavailable"
end

local function alloc_stats()
  if Mimir and Mimir.Allocator and Mimir.Allocator.getStats then
    local a = Mimir.Allocator.getStats()
    if a then
      return string.format("Allocator tensors=%s loaded=%s",
        tostring(a.tensors), tostring(a.loaded))
    end
  end
  return "Allocator stats unavailable"
end

local function sep()
  log("────────────────────────────────────────────────────────────────────────")
end

-- -----------------------------
-- Runtime setup
-- -----------------------------
log("╔════════════════════════════════════════════════════════════════════╗")
log("║        Mímir BENCH — Loss from Output Gradient (CPU torture)       ║")
log("║   L = 0.5 * Σ(y²)  =>  dL/dy = y  (loss computed from gradients)   ║")
log("╚════════════════════════════════════════════════════════════════════╝")

-- Allocator / Guard
if Mimir and Mimir.Allocator and Mimir.Allocator.configure then
  local ok, err = Mimir.Allocator.configure({ max_ram_gb = 10.0, enable_compression = true })
  if ok == false then log("⚠️ Allocator.configure failed: " .. tostring(err)) end
end
if Mimir and Mimir.MemoryGuard and Mimir.MemoryGuard.setLimit then
  local ok, err = Mimir.MemoryGuard.setLimit(10.0)
  if ok == false then log("⚠️ MemoryGuard.set_limit_gb failed: " .. tostring(err)) end
end
if model.set_hardware then
  local ok, err = model.set_hardware("cpu")
  if ok == false then log("⚠️ set_hardware(cpu) failed: " .. tostring(err)) end
end

log("Hardware: " .. hw_stats())
log(guard_stats())
log(alloc_stats())
sep()

-- -----------------------------
-- Test definition (BasicMLP-only, robust)
-- -----------------------------
local LEVELS = {
  { name="Warmup", cfg={ input_dim=256,  hidden_dim=256,  output_dim=256,  hidden_layers=1, dropout=0.0 }, steps=200, lr=0.01 },
  { name="Small",  cfg={ input_dim=512,  hidden_dim=512,  output_dim=512,  hidden_layers=2, dropout=0.0 }, steps=250, lr=0.008 },
  { name="Medium", cfg={ input_dim=1024, hidden_dim=1024, output_dim=1024, hidden_layers=2, dropout=0.0 }, steps=200, lr=0.006 },
  { name="Large",  cfg={ input_dim=2048, hidden_dim=2048, output_dim=2048, hidden_layers=3, dropout=0.0 }, steps=150, lr=0.004 },
}

local function run_level(L)
  log(("TEST: %s | MLP %d -> %d (hidden=%d x %d)"):format(
    L.name,
    L.cfg.input_dim or 0,
    L.cfg.output_dim or 0,
    L.cfg.hidden_dim or 0,
    L.cfg.hidden_layers or 0))

  -- Registry-based creation
  must(model.create("basic_mlp", L.cfg), nil, "model.create(basic_mlp)")

  local t_alloc0 = os.clock()
  must(model.allocate_params(), nil, "allocate_params")
  local t_alloc = os.clock() - t_alloc0

  local t_init0 = os.clock()
  must(model.init_weights("xavier_uniform", 1337), nil, "init_weights")
  local t_init = os.clock() - t_init0

  local x = make_input({ height = 1, width = (L.cfg.input_dim or 0), in_channels = 1 })

  local loss0, loss_last, loss_best = nil, nil, 1e30
  local fwd_ms, bwd_ms, step_ms = 0.0, 0.0, 0.0

  for s = 1, L.steps do
    must(model.zero_grads(), nil, "zero_grads")

    -- forward (training=true)
    local t0 = os.clock()
    local y, err = model.forward(x, true)
    local t1 = os.clock()
    if not y then error("forward failed: " .. tostring(err)) end

    -- loss_grad = y  (because d/dy 0.5*sum(y^2) = y)
    -- compute loss from gradients only:
    local loss_grad = y
    local loss = loss_from_grad(loss_grad)

    if not loss0 then loss0 = loss end
    loss_last = loss
    if loss < loss_best then loss_best = loss end

    -- backward
    local t2 = os.clock()
    local okb, errb = model.backward(loss_grad)
    local t3 = os.clock()
    if okb == false then error("backward failed: " .. tostring(errb)) end

    -- gradients stats
    local grads, gerr = model.get_gradients()
    if not grads then error("get_gradients failed: " .. tostring(gerr)) end
    if vec_any_nan_inf(grads) then error("NaN/Inf detected in parameter gradients") end
    local g_l2, g_max = vec_l2_and_maxabs(grads)

    -- optimizer
    local t4 = os.clock()
    local oks, errs = model.optimizer_step(L.lr)
    local t5 = os.clock()
    if oks == false then error("optimizer_step failed: " .. tostring(errs)) end

    fwd_ms  = fwd_ms  + (t1 - t0) * 1000.0
    bwd_ms  = bwd_ms  + (t3 - t2) * 1000.0
    step_ms = step_ms + (t5 - t4) * 1000.0

    if s == 1 or s % math.floor(L.steps / 5) == 0 then
      log(string.format(
        "  step=%4d/%d | loss=%.6f (best=%.6f) | gradL2=%.6f gradMax=%.6f",
        s, L.steps, loss, loss_best, g_l2, g_max
      ))
    end
  end

  -- A simple pass condition: loss must go down a bit (not necessarily huge)
  local ok_drop = (loss_last < loss0 * 0.99)  -- -1%
  log(string.format("Result: loss %.6f -> %.6f (best %.6f) | drop_ok=%s",
    loss0 or 0, loss_last or 0, loss_best or 0, tostring(ok_drop)))

  log(string.format("Timing(avg): fwd=%.3fms | bwd=%.3fms | opt=%.3fms | steps=%d",
    fwd_ms / L.steps, bwd_ms / L.steps, step_ms / L.steps, L.steps
  ))
  log(guard_stats())
  log(alloc_stats())
  sep()

  return {
    name=L.name,
    params=(model.total_params and model.total_params() or 0),
    loss0=loss0, lossl=loss_last,
    fwd=fwd_ms / L.steps,
    bwd=bwd_ms / L.steps,
    opt=step_ms / L.steps,
    ok=ok_drop
  }
end

-- -----------------------------
-- Main loop
-- -----------------------------
local results = {}
for _, L in ipairs(LEVELS) do
  local ok, ret = pcall(run_level, L)
  if ok then
    results[#results+1] = ret
  else
    log("❌ TEST FAILED: " .. L.name .. " | " .. tostring(ret))
    sep()
  end
end

-- Stress: create many tiny models to test allocator churn
log("STRESS: create/destroy many tiny MLP models")
local stress_n = 120
local stress_cfg = { input_dim=256, hidden_dim=256, output_dim=256, hidden_layers=1, dropout=0.0 }
local tS0 = os.clock()
for i=1,stress_n do
  model.create("basic_mlp", stress_cfg)
  model.allocate_params()
  model.init_weights("xavier_uniform", i)
  local x = make_input({ height = 1, width = (stress_cfg.input_dim or 0), in_channels = 1 })
  local y = model.forward(x, false)
  if not y then error("stress forward failed at i="..i) end
end
local tS1 = os.clock()
log(string.format("Stress done: %d models | total=%0.3fs | avg=%0.2fms/model",
  stress_n, (tS1 - tS0), ((tS1 - tS0) * 1000.0) / stress_n
))
log(guard_stats())
log(alloc_stats())
sep()

-- Summary
log("SUMMARY")
for _, r in ipairs(results) do
  log(string.format("%-7s | params=%s | loss %.4f -> %.4f | fwd=%.3fms bwd=%.3fms opt=%.3fms | ok=%s",
    r.name, tostring(r.params), r.loss0 or 0, r.lossl or 0, r.fwd, r.bwd, r.opt, tostring(r.ok)
  ))
end
log("✅ benchmark_gradloss finished.")

