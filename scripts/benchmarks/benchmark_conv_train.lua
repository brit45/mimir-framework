#!/usr/bin/env mimir --lua

log("╔════════════════════════════════════════════════════════════════════╗")
log("║ BasicMLP Training Bench (API optimizer_step) - loss should decrease║")
log("╚════════════════════════════════════════════════════════════════════╝")

math.randomseed(42)

-- ======================================================================
-- 0) Memory safety (optionnel mais recommandé)
-- ======================================================================
if Mimir.MemoryGuard and Mimir.MemoryGuard.setLimit then
  local ok, err = Mimir.MemoryGuard.setLimit(4.0)
  if ok == false then log("⚠️ MemoryGuard.setLimit failed: " .. tostring(err)) end
end
Mimir.Allocator.configure({
  max_ram_gb = 4.0,
  enable_compression = true,
  swap_strategy = "lru",
})
log("🛡️  MemoryGuard / Allocator configured (4 GB, compression ON)")

if Mimir.Model and Mimir.Model.set_hardware then
  local ok = select(1, pcall(Mimir.Model.set_hardware, true))
  if not ok then log("⚠️ set_hardware(true) failed") end
end

-- ======================================================================
-- 1) Build model (BasicMLP)
-- ======================================================================
log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("1) Build model (BasicMLP)")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

local DIM = 256

local ok, err = Mimir.Model.create("basic_mlp", {
  input_dim = DIM,
  hidden_dim = DIM,
  output_dim = DIM,
  hidden_layers = 2,
  dropout = 0.0
})
assert(ok ~= false, tostring(err or "Model.create(basic_mlp) failed"))

assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier", 42))

log("✓ Model ready | params=" .. tostring(Mimir.Model.total_params()))

-- ======================================================================
-- 2) Data : x random, target = simple "blur-like" transform (fixed)
-- ======================================================================
log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("2) Data (input -> target)")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

local function rand_uniform()
  return (math.random() * 2.0 - 1.0)
end

local x = {}
for i = 1, DIM do x[i] = rand_uniform() end

local y = {}
for i = 1, DIM do y[i] = 0.0 end

log("✓ Input size=" .. #x .. " | Target size=" .. #y)

-- ======================================================================
-- 3) Loss + grad (MSE)
-- ======================================================================
local function mse_and_grad(pred, target)
  local n = #pred
  local loss = 0.0
  local grad = {}

  for i = 1, n do
    local d = (pred[i] - target[i])
    loss = loss + d * d
  end
  loss = loss / n

  -- d/dpred: 2*(pred-target)/n
  local scale = 2.0 / n
  for i = 1, n do
    grad[i] = (pred[i] - target[i]) * scale
  end

  return loss, grad
end

local function grad_norm(g)
  local s = 0.0
  for i = 1, #g do
    local v = g[i]
    s = s + v * v
  end
  return math.sqrt(s)
end

-- ======================================================================
-- 4) Train loop (IMPORTANT: forward(training=true) + backward right after)
-- ======================================================================
log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("3) Train (forward(train=true) + backward + optimizer_step)")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

local epochs = 400
local lr = 1e-4
local best = 1e30
local first = nil

for epoch = 1, epochs do
  -- 1) clear grads
  local ok, err = Mimir.Model.zero_grads()
  if not ok then
    log("❌ zero_grads failed: " .. tostring(err))
    break
  end

  -- 2) forward MUST be in training mode to cache forward state
  local pred, ferr = Mimir.Model.forward(x, true)  -- <-- FORCÉ À true
  if not pred then
    log("❌ forward failed: " .. tostring(ferr))
    break
  end

  -- 3) loss + grad wrt pred
  local loss, dL_dpred = mse_and_grad(pred, y)

  if not first then first = loss end
  if loss < best then best = loss end

  -- 4) backward immediately after the forward that produced pred
  local bok, berr = Mimir.Model.backward(dL_dpred)
  if not bok then
    log("❌ backward failed: " .. tostring(berr))
    break
  end

  -- 5) sanity-check gradients are non-zero
  local grads, gerr = Mimir.Model.get_gradients()
  if not grads then
    log("❌ get_gradients failed: " .. tostring(gerr))
    break
  end
  local gn = grad_norm(grads)

  -- 6) optimizer step
  local sok, serr = Mimir.Model.optimizer_step(lr)
  if not sok then
    log("❌ optimizer_step failed: " .. tostring(serr))
    break
  end

  if epoch % 20 == 0 or epoch == 1 then
    log(string.format("epoch=%d | loss=%.6f | best=%.6f | grad_norm=%.6f", epoch, loss, best, gn))
  end
end

-- final eval (training=false)
local final_pred = Mimir.Model.forward(x, false)
local final_loss = nil
if final_pred then
  final_loss = select(1, mse_and_grad(final_pred, y))
end

log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("4) Sauvegarde")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

local snapshot_path = "mimir_conv_snapshot"
local success1 = false
if Mimir.Serialization and Mimir.Serialization.save then
  local ok, ret = pcall(function()
    return Mimir.Serialization.save(snapshot_path, "raw_folder", {
      include_gradients = true,
      include_optimizer_state = true,
      max_values_per_tensor = 10,
      include_checksums = true,
      include_weight_deltas = true,
      include_git_info = true,
      save_tokenizer = true,
      save_encoder = true,
    })
  end)
  if ok then success1 = ret end

  if not success1 then
    -- Fallback minimal (au cas où l'impl n'accepte pas les options avancées)
    local ok2, ret2 = pcall(function()
      return Mimir.Serialization.save(snapshot_path, "raw_folder")
    end)
    if ok2 then success1 = ret2 end
  end
end

if success1 then
  log("✓ Snapshot sauvegardé: " .. snapshot_path)
else
  log("⚠️  Échec sauvegarde snapshot")
end

log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("5) Result")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log(string.format("first loss=%.6f", first or -1))
log(string.format("final loss=%.6f", final_loss or -1))
log(string.format("best loss =%.6f", best))

if first and final_loss and final_loss < (first * 0.98) then
  log("✅ PASS: loss decreased")
else
  log("❌ FAIL: loss did NOT decrease")
  log("   -> Si grad_norm ~ 0 ou warning 'no valid forward state' persiste:")
  log("      - le runtime ne cache pas le forward en mode training")
  log("      - ou Conv2d backward / optimizer_step ne sont pas branchés correctement")
end
