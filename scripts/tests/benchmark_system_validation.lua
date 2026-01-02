#!/usr/bin/env mimir --lua
-- ============================================================================
-- Mímir - System Validation Benchmark (v2.3.x)
-- CPU-first, runtime+allocator+forward+train-probe+serialization
--
-- Contraintes:
--  - pas de print() -> log() only
--  - pas de global hr() -> défini ici
--  - Embedding: forward() doit recevoir TokenIds (int), pas float[]
--  - Linear config: on évite les graphes "Linear" manuels et on valide
--    multi-input via builders (UNet/ResNet/Diffusion) qui câblent tout.
-- ============================================================================

math.randomseed(os.time())

-- ----------------------------------------------------------------------------
-- Helpers
-- ----------------------------------------------------------------------------
local function hr(title)
  log("\n" .. string.rep("=", 70))
  log(title)
  log(string.rep("=", 70))
end

local function ok_or_die(ok, err, context)
  if not ok then
    error("❌ FAIL: " .. (context or "operation") .. (err and (": " .. tostring(err)) or ""))
  end
end

local function assert_non_nil(v, err, context)
  if v == nil then
    error("❌ FAIL: " .. (context or "value is nil") .. (err and (": " .. tostring(err)) or ""))
  end
  return v
end

local function safe_call(fn, ...)
  local ok, a, b, c = pcall(fn, ...)
  if not ok then return false, tostring(a) end
  return true, a, b, c
end

local function rand_uniform()
  return (math.random() * 2.0 - 1.0)
end

local function rand_vec(n)
  local t = {}
  for i = 1, n do t[i] = rand_uniform() end
  return t
end

local function mse_and_grad(pred, target)
  local n = #pred
  local loss = 0.0
  local grad = {}
  for i = 1, n do
    local d = pred[i] - target[i]
    loss = loss + d * d
    grad[i] = (2.0 * d) / n
  end
  return loss / n, grad
end

local function summarize_memory()
  -- MemoryGuard stats (si dispo)
  if Mimir and Mimir.MemoryGuard and (Mimir.MemoryGuard.getStats or Mimir.MemoryGuard.get_stats) then
    local fn = Mimir.MemoryGuard.getStats or Mimir.MemoryGuard.get_stats
    local ok, st = safe_call(fn)
    if ok and st then
      log(string.format("MemoryGuard: current=%.1fMB peak=%.1fMB limit=%.1fMB usage=%.1f%%",
        st.current_mb or 0, st.peak_mb or 0, st.limit_mb or 0, st.usage_percent or 0))
    end
  elseif Mimir and Mimir.Guard and (Mimir.Guard.getStats or Mimir.Guard.get_stats) then
    local fn = Mimir.Guard.getStats or Mimir.Guard.get_stats
    local ok, st = safe_call(fn)
    if ok and st then
      log(string.format("Guard: current=%.1fMB peak=%.1fMB limit=%.1fMB usage=%.1f%%",
        st.current_mb or 0, st.peak_mb or 0, st.limit_mb or 0, st.usage_percent or 0))
    end
  end

  -- Allocator stats (si dispo)
  if Mimir and Mimir.Allocator and (Mimir.Allocator.getStats or Mimir.Allocator.get_stats) then
    local fn = Mimir.Allocator.getStats or Mimir.Allocator.get_stats
    local ok, st = safe_call(fn)
    if ok and st then
      log(string.format("Allocator: tensors=%s loaded=%s",
        tostring(st.tensor_count or 0), tostring(st.loaded_count or 0)))
    end
  end
end

local function hardware_setup()
  local caps = Mimir.Model.hardware_caps()
  log("Hardware: AVX2=" .. tostring(caps.avx2) ..
      " FMA=" .. tostring(caps.fma) ..
      " F16C=" .. tostring(caps.f16c) ..
      " BMI2=" .. tostring(caps.bmi2))

  local ok, err = Mimir.Model.set_hardware("cpu")
  ok_or_die(ok, err, "Model.set_hardware(cpu)")
  log("✅ PASS: Model.set_hardware(cpu)")
end

-- ----------------------------------------------------------------------------
-- Test Runner
-- ----------------------------------------------------------------------------
local RESULTS = { total = 0, passed = 0, failed = 0, failures = {} }

local function run_test(name, fn)
  RESULTS.total = RESULTS.total + 1
  log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
  log("TEST: " .. name)
  log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

  local ok, err = pcall(fn)
  if ok then
    RESULTS.passed = RESULTS.passed + 1
    log("✅ PASS: " .. name)
  else
    RESULTS.failed = RESULTS.failed + 1
    local msg = tostring(err)
    table.insert(RESULTS.failures, { name = name, err = msg })
    log("❌ FAIL: " .. name)
    log("   " .. msg)
  end
end

-- ----------------------------------------------------------------------------
-- 0) Runtime / Allocator Setup
-- ----------------------------------------------------------------------------
hr("0) Runtime / Memory / Allocator Setup")

do
  local ok, err = Mimir.Allocator.configure({
    max_ram_gb = 10.0,
    enable_compression = true
  })
  ok_or_die(ok, err, "Allocator.configure(10GB, compression=ON)")
  log("✅ PASS: Allocator.configure | 10GB compression=ON")
  hardware_setup()
  summarize_memory()
end

-- ----------------------------------------------------------------------------
-- 1) Tiny Transformer (int tokens) - forward smoke test
-- ----------------------------------------------------------------------------
run_test("Tiny Transformer forward (TokenIds int input)", function()
  -- Petit modèle pour valider le chemin tokens->embedding->attention->head
  local cfg = {
    vocab_size = 256,
    embed_dim = 64,
    num_layers = 2,
    num_heads = 4,
    d_ff = 128,
    max_seq_len = 32,
    dropout = 0.0,
    causal = true
  }

  local ok, err = Mimir.Model.create("transformer", cfg)
  ok_or_die(ok, err, "Model.create(transformer)")

  ok, err = Mimir.Architectures.transformer(cfg)
  ok_or_die(ok, err, "Architectures.transformer")

  ok, err = Mimir.Model.allocate_params()
  ok_or_die(ok, err, "allocate_params")

  ok, err = Mimir.Model.init_weights("he", 42)
  ok_or_die(ok, err, "init_weights(he)")

  local params = Mimir.Model.total_params()
  log("Model ready | params=" .. tostring(params))

  -- IMPORTANT: token ids entiers (Lua 1..vocab_size, selon ton runtime)
  local tokens = {}
  for i = 1, 16 do
    tokens[i] = (i % 200) + 1
  end

  local out, ferr = Mimir.Model.forward(tokens, false)
  assert_non_nil(out, ferr, "Transformer.forward(TokenIds)")
  log("Forward OK | out_len=" .. tostring(#out))
end)

-- ----------------------------------------------------------------------------
-- 2) Conv2d Train-Probe (loss doit baisser)
-- ----------------------------------------------------------------------------
run_test("Conv2d Train-Probe (loss decreases)", function()
  -- Basé sur ton bench officiel conv train, mais en mode "validation"
  local H, W, C = 32, 32, 3
  local OUT, K = 3, 3

  local ok, err = Mimir.Allocator.configure({ max_ram_gb = 4.0, enable_compression = true })
  ok_or_die(ok, err, "Allocator.configure(4GB)")
  log("Allocator set to 4GB for conv probe")

  ok, err = Mimir.Model.create("conv_train_api", {
    in_channels = C,
    out_channels = OUT,
    height = H,
    width = W,
    kernel = K,
    stride = 1,
    padding = 1
  })
  ok_or_die(ok, err, "Model.create(conv_train_api)")

  local params_count = (K * K * C * OUT) + OUT
  ok, err = Mimir.Model.push_layer("conv", "Conv2d", params_count)
  ok_or_die(ok, err, "push_layer(Conv2d)")

  ok, err = Mimir.Model.allocate_params()
  ok_or_die(ok, err, "allocate_params")

  ok, err = Mimir.Model.init_weights("he", 123)
  ok_or_die(ok, err, "init_weights(he)")

  log("Model ready | params=" .. tostring(Mimir.Model.total_params()))

  local N = H * W * C
  local x = rand_vec(N)

  -- target stable/deterministic
  local y = {}
  for i = 1, N do
    local a = x[i]
    local b = x[math.max(1, i - 1)]
    local c_ = x[math.min(N, i + 1)]
    y[i] = 0.8 * a + 0.1 * b + 0.1 * c_
  end

  local steps = 200
  local lr = 0.01

  local best = 1e30
  local first_loss = nil
  local last_loss = nil

  for step = 1, steps do
    ok, err = Mimir.Model.zero_grads()
    ok_or_die(ok, err, "zero_grads")

    local pred, ferr = Mimir.Model.forward(x, true)
    assert_non_nil(pred, ferr, "forward(conv)")

    local loss, grad = mse_and_grad(pred, y)
    if not first_loss then first_loss = loss end
    last_loss = loss
    if loss < best then best = loss end

    ok, err = Mimir.Model.backward(grad)
    ok_or_die(ok, err, "backward")

    ok, err = Mimir.Model.optimizer_step(lr)
    ok_or_die(ok, err, "optimizer_step")

    if step == 1 or step % 20 == 0 then
      log(string.format("step=%d | loss=%.6f | best=%.6f", step, loss, best))
    end
  end

  -- critère simple : la loss doit baisser un minimum (pas besoin d’être énorme)
  log(string.format("loss: %.6f -> %.6f", first_loss, last_loss))
  if not (last_loss < first_loss) then
    error("loss did not decrease")
  end

  summarize_memory()
end)

-- ----------------------------------------------------------------------------
-- 3) UNet Forward (multi-input/skip connections inside)
-- ----------------------------------------------------------------------------
run_test("UNet forward smoke test (multi-input graph)", function()
  local cfg = {
    input_channels = 3,
    output_channels = 3,
    base_channels = 16,
    num_levels = 3,
    blocks_per_level = 1,
    use_attention = false,
    use_residual = true,
    dropout = 0.0,
    image_size = 32
  }

  local ok, err = Mimir.Model.create("unet", cfg)
  ok_or_die(ok, err, "Model.create(unet)")

  ok, err = Mimir.Architectures.unet(cfg)
  ok_or_die(ok, err, "Architectures.unet")

  ok, err = Mimir.Model.allocate_params()
  ok_or_die(ok, err, "allocate_params")

  ok, err = Mimir.Model.init_weights("he", 7)
  ok_or_die(ok, err, "init_weights(he)")

  local N = (cfg.input_channels or 3) * (cfg.image_size or 32) * (cfg.image_size or 32)
  local x = rand_vec(N)

  local out, ferr = Mimir.Model.forward(x, false)
  assert_non_nil(out, ferr, "UNet.forward")
  log("Forward OK | out_len=" .. tostring(#out) .. " | params=" .. tostring(Mimir.Model.total_params()))
end)

-- ----------------------------------------------------------------------------
-- 4) ResNet Forward (residual connections)
-- ----------------------------------------------------------------------------
run_test("ResNet forward smoke test (residual connections)", function()
  local cfg = {
    image_channels = 3,
    image_size = 32,
    num_classes = 10,
    blocks = 2,
    base_channels = 16
  }

  local ok, err = Mimir.Model.create("resnet", cfg)
  ok_or_die(ok, err, "Model.create(resnet)")

  ok, err = Mimir.Architectures.resnet(cfg)
  ok_or_die(ok, err, "Architectures.resnet")

  ok, err = Mimir.Model.allocate_params()
  ok_or_die(ok, err, "allocate_params")

  ok, err = Mimir.Model.init_weights("he", 99)
  ok_or_die(ok, err, "init_weights(he)")

  local N = (cfg.image_channels or 3) * (cfg.image_size or 32) * (cfg.image_size or 32)
  local x = rand_vec(N)

  local out, ferr = Mimir.Model.forward(x, false)
  assert_non_nil(out, ferr, "ResNet.forward")
  log("Forward OK | out_len=" .. tostring(#out) .. " | params=" .. tostring(Mimir.Model.total_params()))
end)

-- ----------------------------------------------------------------------------
-- 5) Serialization (Enhanced debug JSON + detect_format)
-- ----------------------------------------------------------------------------
run_test("Serialization smoke test (debug json + detect_format)", function()
  local path = "checkpoints/_bench_validation_debug.json"
  local ok, err = Mimir.Serialization.save_enhanced_debug(path, {
    include_weights_samples = true,
    sample_size = 128,
    include_optimizer = true,
    include_runtime = true
  })
  ok_or_die(ok, err, "Serialization.save_enhanced_debug")
  log("[DebugJsonDump] saved: " .. path)

  local fmt, ferr = Mimir.Serialization.detect_format(path)
  if fmt then
    log("detect_format: " .. tostring(fmt))
  else
    -- Sur un JSON debug, c’est possible que detect_format retourne nil (selon ton impl)
    log("detect_format: nil (" .. tostring(ferr or "no err") .. ")")
  end
end)

-- ----------------------------------------------------------------------------
-- Summary
-- ----------------------------------------------------------------------------
hr("SUMMARY")

log("Total tests:  " .. tostring(RESULTS.total))
log("✅ Passed:     " .. tostring(RESULTS.passed))
log("❌ Failed:     " .. tostring(RESULTS.failed))
local pass_rate = 0.0
if RESULTS.total > 0 then pass_rate = (RESULTS.passed / RESULTS.total) * 100.0 end
log(string.format("📈 Pass rate:  %.1f%%", pass_rate))

if RESULTS.failed > 0 then
  log("\nFailures:")
  for _, f in ipairs(RESULTS.failures) do
    log(" - " .. f.name .. " => " .. f.err)
  end
  log("\n⚠️  Some tests failed - voir erreurs ci-dessus")
else
  log("\n🎉 ALL TESTS PASSED - SYSTEM VALIDATED!")
end
