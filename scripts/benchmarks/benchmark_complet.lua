-- scripts/benchmark_full.lua
-- ╔════════════════════════════════════════════════════════════════════╗
-- ║     Mímir Framework - Full Benchmark Suite (CPU-FIRST, v2.3+)     ║
-- ║  Stress: build/alloc/init + forward compute + multi-input + I/O   ║
-- ╚════════════════════════════════════════════════════════════════════╝
--
-- Objectifs:
--  1) Pousser la construction/initialisation des modèles (scalabilité)
--  2) Mesurer le coût d'allocation + init weights (stress RAM/allocator)
--  3) Mesurer le forward compute (conv/linear/matmul) + multi-input routing
--  4) Détecter fuites mémoire (guard current/peak stable)
--  5) Tester save/load (si supporté)

-- NOTE:
--  - Utilise log() (API framework) plutôt que printf.
--  - Le benchmark reste "safe": tailles importantes mais ajustables en haut du script.

--==========================================================
-- Helpers
--==========================================================
local function now_ms()
  return os.clock() * 1000.0
end

local function fmt_ms(x) return string.format("%.3f ms", x) end
local function fmt_mb(x) return string.format("%.1f MB", x) end

local function hr(title)
  log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
  log(title)
  log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
end

local function guard_snapshot(tag)
  if Mimir.MemoryGuard and Mimir.MemoryGuard.getStats then
    local s = Mimir.MemoryGuard.getStats()
    if s then
      log(string.format("🧠 Guard snapshot: %s | current=%s | peak=%s | limit=%s | usage=%.1f%%",
        tag,
        fmt_mb(s.current_mb or s.current or 0),
        fmt_mb(s.peak_mb or s.peak or 0),
        fmt_mb(s.limit_mb or s.limit or 0),
        (s.usage_percent or 0)
      ))
    end
  end
  if Mimir.Memory and Mimir.Memory.get_stats then
    local m = Mimir.Memory.get_stats()
    if m then
      log(string.format("🧠 MemoryMgr snapshot: %s | current=%s | peak=%s | usage=%.1f%%",
        tag,
        fmt_mb(m.current_mb or m.current or 0),
        fmt_mb(m.peak_mb or m.peak or 0),
        (m.usage_percent or 0)
      ))
    end
  end
  if Mimir.Allocator and Mimir.Allocator.get_stats then
    local a = Mimir.Allocator.get_stats()
    if a then
      log(string.format("🧠 Allocator snapshot: %s | tensors=%s | loaded=%s",
        tag,
        tostring(a.tensor_count or a.tensors or a.num_tensors or "?"),
        tostring(a.loaded_count or a.loaded or a.loaded_tensors or "?")
      ))
    end
  end
end

local function assert_ok(ok, err, context)
  if not ok then
    error((context or "Operation failed") .. (err and (": " .. err) or ""), 2)
  end
end

local function gen_floats(n, scale)
  scale = scale or 1.0
  local t = {}
  for i = 1, n do
    -- déterministe-ish (pas besoin de math.randomseed)
    t[i] = ((i % 97) * 0.001) * scale
  end
  return t
end

local function gen_int_tokens(n, vocab)
  local t = {}
  vocab = vocab or 32000
  for i = 1, n do
    t[i] = (i * 1315423911) % vocab
  end
  return t
end

local function checksum(arr)
  local s = 0.0
  local mx = -1e30
  local mn = 1e30
  for i = 1, #arr do
    local v = arr[i]
    if v ~= v then return {nan=true} end -- NaN
    s = s + v
    if v > mx then mx = v end
    if v < mn then mn = v end
  end
  return {sum=s, mean=s / math.max(1, #arr), min=mn, max=mx}
end

local function log_stats(prefix, arr)
  local c = checksum(arr)
  if c.nan then
    log(prefix .. " stats: NaN detected ❌")
  else
    log(string.format("%s stats: min=%.6f | max=%.6f | mean=%.6f | sum=%.6f",
      prefix, c.min, c.max, c.mean, c.sum))
  end
end

--==========================================================
-- Configuration (ajuste ici)
--==========================================================
local CFG = {
  -- RAM budget
  limit_gb = 10,
  -- Allocator
  allocator = {
    max_ram_gb = 10,
    enable_compression = true,
    -- optionnels selon ton impl:
    max_tensors = 2000,
    offload_threshold_mb = 9000,
    swap_strategy = "lru",
  },

  -- Benchmark model sizes
  transformer = {
    {name="TINY",  cfg={vocab_size=12000, embed_dim=128, num_layers=2, num_heads=4, d_ff=512,  max_seq_len=128}},
    {name="SMALL", cfg={vocab_size=30000, embed_dim=256, num_layers=4, num_heads=8, d_ff=1024, max_seq_len=256}},
  },

  vit = {
    {name="ViT-S", cfg={image_size=128, patch_size=16, embed_dim=256, num_layers=4, num_heads=8, mlp_ratio=4.0, num_classes=1000}},
  },

  unet = {
    {name="UNet-S", cfg={input_channels=3, output_channels=3, base_channels=32, num_levels=3, blocks_per_level=2, use_attention=false, use_residual=true, dropout=0.0}},
  },

  vae = {
    {name="VAE-S", cfg={input_dim=64*64*3, latent_dim=128, encoder_hidden=512, decoder_hidden=512}},
  },

  resnet = {
    {name="ResNet", cfg={input_channels=3, num_classes=1000}},
  },

  diffusion = {
    {name="DDPM-S", cfg={image_channels=3, resolution=64, model_channels=64, num_res_blocks=2, use_attention=false, dropout=0.0, use_bottleneck=true}},
  },

  -- forward compute iterations
  forward_warmup = 3,
  forward_iters  = 10,

  -- input sizes (compute stress)
  image = {c=3, h=128, w=128},   -- UNet/ViT/ResNet/Diffusion image-like
  seq_len = 256,                -- Transformer token input
  seed = 42,
}

--==========================================================
-- Suite start
--==========================================================
log("╔════════════════════════════════════════════════════════════════════╗")
log("║   Mímir - Full Benchmark Suite (build/alloc/init/forward/multi)    ║")
log("╚════════════════════════════════════════════════════════════════════╝")

-- hardware caps
if Mimir.Model and Mimir.Model.hardware_caps then
  local caps = Mimir.Model.hardware_caps()
  if caps then
    log(string.format("🔧 Hardware caps: AVX2=%s FMA=%s F16C=%s BMI2=%s",
      tostring(caps.avx2), tostring(caps.fma), tostring(caps.f16c), tostring(caps.bmi2)))
  end
end

-- memory limits
hr("0) Memory setup")
if Mimir.MemoryGuard and Mimir.MemoryGuard.setLimit then
  assert_ok(Mimir.MemoryGuard.setLimit(CFG.limit_gb), nil, "Mimir.MemoryGuard.setLimit")
  log(string.format("🛡️  MemoryGuard limit set to %d GB", CFG.limit_gb))
end
if Mimir.Allocator and Mimir.Allocator.configure then
  local ok, err = Mimir.Allocator.configure(CFG.allocator)
  assert_ok(ok, err, "Mimir.Allocator.configure")
  log("🚀 DynamicTensorAllocator configured (compression=" .. tostring(CFG.allocator.enable_compression) .. ")")
end
if Mimir.Memory and Mimir.Memory.set_limit then
  local ok, err = Mimir.Memory.set_limit(CFG.limit_gb)
  assert_ok(ok, err, "Mimir.Memory.set_limit")
end
guard_snapshot("start")

--==========================================================
-- Core benchmark functions
--==========================================================
local function bench_model_build_alloc_init(model_type, cfg, init_method, seed)
  init_method = init_method or "xavier"
  seed = seed or CFG.seed

  local t0 = now_ms()
  local ok, err = Mimir.Model.create(model_type, cfg)
  assert_ok(ok, err, "Mimir.Model.create(" .. tostring(model_type) .. ")")

  local params = nil
  if Mimir.Model.build then
    local b0 = now_ms()
    local okb, p_or_err = Mimir.Model.build()
    local b1 = now_ms()
    if okb then
      params = p_or_err
    end
    -- build() peut être optionnel selon ton style de modèle; si okb=false, on continue
    log(string.format("   build: %s", fmt_ms(b1 - b0)))
  end

  local a0 = now_ms()
  local oka, aerr = Mimir.Model.allocate_params()
  assert_ok(oka, aerr, "Mimir.Model.allocate_params")
  local a1 = now_ms()

  local i0 = now_ms()
  local oki, ierr = Mimir.Model.init_weights(init_method, seed)
  assert_ok(oki, ierr, "Mimir.Model.init_weights")
  local i1 = now_ms()

  local total = now_ms() - t0
  local total_params = (Mimir.Model.total_params and Mimir.Model.total_params()) or params or -1

  return {
    params = total_params,
    build_ms = (Mimir.Model.build and (nil)) or 0,
    alloc_ms = (a1 - a0),
    init_ms  = (i1 - i0),
    total_ms = total
  }
end

local function bench_forward(model_label, input, training)
  training = training or false
  -- warmup
  for _ = 1, CFG.forward_warmup do
    local out = Mimir.Model.forward(input, training)
    if not out then error(model_label .. ": forward returned nil during warmup") end
  end

  local t0 = now_ms()
  local out = nil
  for _ = 1, CFG.forward_iters do
    out = Mimir.Model.forward(input, training)
    if not out then error(model_label .. ": forward returned nil") end
  end
  local t1 = now_ms()
  local avg = (t1 - t0) / CFG.forward_iters

  log(string.format("   forward: avg=%s over %d iters | out_size=%d",
    fmt_ms(avg), CFG.forward_iters, #out))
  log_stats("   output", out)
  return avg, out
end

local function section_builds(title, cases)
  hr(title)
  for _, c in ipairs(cases) do
    log(string.format("▶ %s (%s)", c.name, title))
    local r = bench_model_build_alloc_init(c.model_type, c.cfg, c.init or "xavier", CFG.seed)
    guard_snapshot("after " .. c.name)
    log(string.format("   params=%s | alloc=%s | init=%s | total=%s",
      tostring(r.params), fmt_ms(r.alloc_ms), fmt_ms(r.init_ms), fmt_ms(r.total_ms)))
  end
end

--==========================================================
-- 1) Build/Alloc/Init stress (multi-architectures)
--==========================================================
local build_cases = {}

for _, c in ipairs(CFG.transformer) do table.insert(build_cases, {name="Transformer-"..c.name, model_type="transformer", cfg=c.cfg}) end
for _, c in ipairs(CFG.vit)        do table.insert(build_cases, {name="ViT-"..c.name,         model_type="vit",        cfg=c.cfg}) end
for _, c in ipairs(CFG.unet)       do table.insert(build_cases, {name="UNet-"..c.name,        model_type="unet",       cfg=c.cfg}) end
for _, c in ipairs(CFG.vae)        do table.insert(build_cases, {name="VAE-"..c.name,         model_type="vae",        cfg=c.cfg}) end
for _, c in ipairs(CFG.resnet)     do table.insert(build_cases, {name="ResNet-"..c.name,      model_type="resnet",     cfg=c.cfg}) end
for _, c in ipairs(CFG.diffusion)  do table.insert(build_cases, {name="Diffusion-"..c.name,   model_type="diffusion",  cfg=c.cfg}) end

hr("1) Build / Allocate / Init (Stress)")
for _, c in ipairs(build_cases) do
  log(string.format("✅ Case: %s", c.name))
  local t0 = now_ms()
  local ok, err = Mimir.Model.create(c.model_type, c.cfg)
  assert_ok(ok, err, "create " .. c.name)

  local build_ms = 0
  if Mimir.Model.build then
    local b0 = now_ms()
    local okb, p = Mimir.Model.build()
    local b1 = now_ms()
    build_ms = b1 - b0
    if okb then
      log("   build ok | params=" .. tostring(p))
    else
      log("   build skipped/failed (ok=false) - continuing (manual layers?)")
    end
  end

  local a0 = now_ms()
  local oka, aerr = Mimir.Model.allocate_params()
  assert_ok(oka, aerr, "allocate " .. c.name)
  local a1 = now_ms()

  local i0 = now_ms()
  local oki, ierr = Mimir.Model.init_weights("xavier", CFG.seed)
  assert_ok(oki, ierr, "init " .. c.name)
  local i1 = now_ms()

  local t1 = now_ms()
  guard_snapshot("after " .. c.name)

  log(string.format("   build=%s | alloc=%s | init=%s | total=%s",
    fmt_ms(build_ms), fmt_ms(a1-a0), fmt_ms(i1-i0), fmt_ms(t1-t0)))
end

--==========================================================
-- 2) Forward compute benchmarks (representative)
--==========================================================
hr("2) Forward Compute Benchmarks (Real runtime)")
-- Inputs
local img_n = CFG.image.c * CFG.image.h * CFG.image.w
local img = gen_floats(img_n, 1.0)
local tokens = gen_int_tokens(CFG.seq_len, 30000)

-- UNet forward
do
  local cfg = CFG.unet[1].cfg
  log("▶ UNet forward ("..CFG.image.h.."x"..CFG.image.w.."x"..CFG.image.c..")")
  assert_ok(Mimir.Model.create("unet", cfg))
  if Mimir.Model.build then Mimir.Model.build() end
  Mimir.Model.allocate_params()
  Mimir.Model.init_weights("he", CFG.seed)
  bench_forward("unet", img, false)
  guard_snapshot("after UNet forward")
end

-- ResNet forward
do
  local cfg = CFG.resnet[1].cfg
  log("▶ ResNet forward ("..CFG.image.h.."x"..CFG.image.w.."x"..CFG.image.c..")")
  assert_ok(Mimir.Model.create("resnet", cfg))
  if Mimir.Model.build then Mimir.Model.build() end
  Mimir.Model.allocate_params()
  Mimir.Model.init_weights("he", CFG.seed)
  bench_forward("resnet", img, false)
  guard_snapshot("after ResNet forward")
end

-- ViT forward
do
  local cfg = CFG.vit[1].cfg
  log("▶ ViT forward (image_size="..tostring(cfg.image_size)..")")
  assert_ok(Mimir.Model.create("vit", cfg))
  if Mimir.Model.build then Mimir.Model.build() end
  Mimir.Model.allocate_params()
  Mimir.Model.init_weights("xavier", CFG.seed)
  local n = cfg.image_size * cfg.image_size * 3
  local vit_img = gen_floats(n, 1.0)
  bench_forward("vit", vit_img, false)
  guard_snapshot("after ViT forward")
end

-- Transformer forward (tokens)
do
  local cfg = CFG.transformer[2].cfg
  log("▶ Transformer forward (seq_len="..CFG.seq_len..")")
  assert_ok(Mimir.Model.create("transformer", cfg))
  if Mimir.Model.build then Mimir.Model.build() end
  Mimir.Model.allocate_params()
  Mimir.Model.init_weights("xavier", CFG.seed)
  bench_forward("transformer", tokens, false)
  guard_snapshot("after Transformer forward")
end

--==========================================================
-- 3) Multi-input routing stress (Residual + Concat + Split)
--==========================================================
hr("3) Multi-Input Stress (TensorStore routing)")

local has_set_io = (Mimir.Model.set_layer_io ~= nil)

local function require_set_io()
  if not has_set_io then
    error("Mimir.Model.set_layer_io() not available in API - cannot run multi-input stress")
  end
end

-- Residual block demo (Conv -> BN -> Add(x, main) -> ReLU)
do
  require_set_io()
  log("▶ Residual block (Conv/BN/Add)")
  local cfg = {in_channels=CFG.image.c, out_channels=64, height=CFG.image.h, width=CFG.image.w, kernel=3, stride=1, padding=1}

  assert_ok(Mimir.Model.create("residual_stress", cfg))

  -- conv1 (writes main)
  local conv_params = (3*3*64*64) + 64
  Mimir.Model.push_layer("conv1", "Conv2d", conv_params)
  Mimir.Model.set_layer_io("conv1", {"x"}, "main")

  -- bn1 (main -> main)
  Mimir.Model.push_layer("bn1", "BatchNorm2d", 64*2)
  Mimir.Model.set_layer_io("bn1", {"main"}, "main")

  -- add (x + main -> combined)
  Mimir.Model.push_layer("add", "Add", 0)
  Mimir.Model.set_layer_io("add", {"x","main"}, "combined")

  -- relu (combined -> x)
  Mimir.Model.push_layer("relu", "ReLU", 0)
  Mimir.Model.set_layer_io("relu", {"combined"}, "x")

  Mimir.Model.allocate_params()
  Mimir.Model.init_weights("he", CFG.seed)

  local in64 = gen_floats(1 * 64 * CFG.image.h * CFG.image.w, 1.0)
  bench_forward("residual_stress", in64, false)
  guard_snapshot("after Residual stress")
end

-- Split + 2 branches + Concat
do
  require_set_io()
  log("▶ Split + parallel conv + concat")
  local cfg = {in_channels=64, out_channels=64, height=CFG.image.h, width=CFG.image.w, kernel=3, stride=1, padding=1}
  assert_ok(Mimir.Model.create("split_parallel", cfg))

  Mimir.Model.push_layer("split", "Split", 0)
  Mimir.Model.set_layer_io("split", {"x"}, "parts") -- convention: writes parts_0 / parts_1

  -- branch 0
  Mimir.Model.push_layer("conv0", "Conv2d", (3*3*64*64)+64)
  Mimir.Model.set_layer_io("conv0", {"parts_0"}, "b0")

  -- branch 1 (5x5)
  local k = 5
  Mimir.Model.push_layer("conv1", "Conv2d", (k*k*64*64)+64)
  -- IMPORTANT: ton runtime doit lire kernel/stride/padding depuis Layer/config; sinon ce test devient "3x3"
  Mimir.Model.set_layer_io("conv1", {"parts_1"}, "b1")

  Mimir.Model.push_layer("cat", "Concat", 0)
  Mimir.Model.set_layer_io("cat", {"b0","b1"}, "x")

  Mimir.Model.allocate_params()
  Mimir.Model.init_weights("he", CFG.seed)

  local in64 = gen_floats(1 * 64 * CFG.image.h * CFG.image.w, 1.0)
  bench_forward("split_parallel", in64, false)
  guard_snapshot("after Split/Concat stress")
end

--==========================================================
-- 4) Save/Load benchmark (optional)
--==========================================================
hr("4) Serialization (Save/Load)")
do
  local cfg = CFG.transformer[1].cfg
  assert_ok(Mimir.Model.create("transformer", cfg))
  if Mimir.Model.build then Mimir.Model.build() end
  Mimir.Model.allocate_params()
  Mimir.Model.init_weights("xavier", CFG.seed)

  local save_path = "checkpoints/bench_tmp_model.safetensors"
  local t0 = now_ms()
  local oks, serr = Mimir.Serialization.save(save_path, "safetensors")
  assert_ok(oks, serr, "Mimir.Serialization.save")
  local t1 = now_ms()
  log("   save: " .. fmt_ms(t1 - t0))

  local t2 = now_ms()
  local okl, lerr = Mimir.Serialization.load(save_path)
  assert_ok(okl, lerr, "Mimir.Serialization.load")
  local t3 = now_ms()
  log("   load: " .. fmt_ms(t3 - t2))
  guard_snapshot("after save/load")
end

--==========================================================
-- 5) Leak check: repeat forwards and ensure guard current stable
--==========================================================
hr("5) Leak Check (repeat forward)")
do
  local cfg = CFG.unet[1].cfg
  assert_ok(Mimir.Model.create("unet", cfg))
  if Mimir.Model.build then Mimir.Model.build() end
  Mimir.Model.allocate_params()
  Mimir.Model.init_weights("he", CFG.seed)

  guard_snapshot("before leak loop")
  local loops = 30
  local t0 = now_ms()
  for i=1, loops do
    local out = Mimir.Model.forward(img, false)
    if not out then error("Leak check forward failed at iter " .. i) end
  end
  local t1 = now_ms()
  log(string.format("   %d forwards: total=%s | avg=%s",
    loops, fmt_ms(t1 - t0), fmt_ms((t1 - t0) / loops)))
  guard_snapshot("after leak loop")
end

--==========================================================
-- Done
--==========================================================
hr("✅ Benchmark finished")
guard_snapshot("end")
log("✅ Official full benchmark completed.")
