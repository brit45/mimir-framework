#!/usr/bin/env mimir --lua

log("╔════════════════════════════════════════════════════════════════════╗")
log("║ Conv Training Bench (API optimizer_step) - loss should decrease    ║")
log("╚════════════════════════════════════════════════════════════════════╝")

-- ======================================================================
-- 0) Memory safety (optionnel mais recommandé)
-- ======================================================================
Mimir.Allocator.configure({
  max_ram_gb = 4.0,
  enable_compression = true
})
log("🛡️  MemoryGuard / Allocator configured (4 GB, compression ON)")

-- ======================================================================
-- 1) Build model : Single Conv2d (3x3, same padding)
-- ======================================================================
log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("1) Build model (Conv2d only)")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

local H, W, C = 32, 32, 3
local OUT = 3
local K = 3

Mimir.Model.create("conv_train_api", {
  in_channels = C,
  out_channels = OUT,
  height = H,
  width = W,
  kernel = K,
  stride = 1,
  padding = 1
})

-- params conv: K*K*C*OUT + OUT (bias)
local params = (K * K * C * OUT) + OUT
assert(Mimir.Model.push_layer("conv", "Conv2d", params))

assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier_uniform", 42))

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

local N = H * W * C
local x = {}
for i = 1, N do x[i] = rand_uniform() end

-- target: une version “adoucie” (pas un vrai blur 2D ici, mais stable et déterministe)
-- (le but = vérifier que l’optimisation bouge bien les poids)
local y = {}
for i = 1, N do
  local a = x[i]
  local b = x[math.max(1, i - 1)]
  local c_ = x[math.min(N, i + 1)]
  y[i] = 0.8 * a + 0.1 * b + 0.1 * c_
end

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

local epochs = 20000
local lr = 6e-5  -- volontairement assez grand pour voir bouger vite (tu peux baisser ensuite)
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

  if epoch % 10 == 0 or epoch == 1 then
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
local success1 = Mimir.Serialization.save(snapshot_path, "raw_folder", {
    include_gradients = true,
    include_optimizer_state = true,
    max_values_per_tensor = 10,
    include_checksums = true,
    include_weight_deltas = true,  -- Pas encore de deltas
    include_git_info = true,
    save_tokenizer = true,
    save_encoder = true
})

if success1 then
    log("✓ Snapshot sauvegardé (Enhanced Debug JSON v1.1.0): " .. snapshot_path)
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
