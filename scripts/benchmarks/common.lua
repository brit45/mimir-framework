-- Helpers partagés des benchmarks natifs Mímir.
local B = {}
function B.has(flag)
  for _, v in ipairs(arg or {}) do if v == flag then return true end end
  return false
end
function B.value(flag, env, default)
  for i, v in ipairs(arg or {}) do
    if v == flag then
      assert(arg[i + 1] and not arg[i + 1]:match('^%-%-'), 'valeur manquante: ' .. flag)
      return arg[i + 1]
    end
  end
  return (env and os.getenv(env)) or default
end
function B.number(flag, env, default, minimum, integer)
  local n = tonumber(B.value(flag, env, default))
  assert(n and n == n and n < math.huge and n >= (minimum or 0)
    and (not integer or n % 1 == 0), 'valeur invalide: ' .. flag)
  return n
end
function B.finite(v)
  return type(v) == 'number' and v == v and math.abs(v) < math.huge
end
function B.vector(v, expected)
  assert(type(v) == 'table' and #v > 0, 'sortie vide ou invalide')
  assert(not expected or #v == expected, 'taille de sortie inattendue: ' .. #v)
  for i, x in ipairs(v) do assert(B.finite(x), 'NaN/Inf à l’indice ' .. i) end
  return v
end
function B.floats(n)
  local x = {}
  for i = 1, n do x[i] = math.sin(i * 0.73) * 0.25 end
  return x
end
function B.tokens(n, vocab)
  local x = {}
  for i = 1, n do x[i] = 1 + (i % (vocab - 1)) end
  return x
end
-- os.clock mesure le CPU cumulé du processus, pas la latence murale.
-- Le binding Lua n'expose pas d'horloge monotone haute résolution.
B.now = os.clock
function B.measure(label, fn)
  local start = B.now()
  local result = fn()
  local elapsed = B.now() - start
  log(string.format('%s: cpu_ms=%.3f', label, elapsed * 1000))
  return result, elapsed
end
function B.setup(default_ram)
  local ram = B.number('--ram', 'MIMIR_BENCH_RAM_GB', default_ram or 2, 0.001)
  assert(ram <= 1000, '--ram doit être exprimé en GiB (<=1000)')
  assert(Mimir.MemoryGuard.setLimit(ram))
  assert(Mimir.Allocator.configure({max_ram_gb=ram,
    enable_compression=not B.has('--no-compress') and os.getenv('MIMIR_BENCH_COMPRESS') ~= '0'
      and os.getenv('MIMIR_BENCH_COMPRESS') ~= 'false'}))
  assert(Mimir.Model.set_hardware(not B.has('--no-accel')))
  local caps = Mimir.Model.hardware_caps()
  log(string.format('Chronométrage: CPU processus (os.clock), OMP_NUM_THREADS=%s, AVX2=%s, FMA=%s',
    os.getenv('OMP_NUM_THREADS') or 'auto', tostring(caps.avx2), tostring(caps.fma)))
  log('set_hardware contrôle les kernels accélérés; ce booléen ne sélectionne pas un backend GPU.')
end
function B.dtype()
  assert(Mimir.Model.dtype(B.value('--dtype', 'MIMIR_DTYPE', 'float32')))
end
function B.init()
  B.dtype()
  B.measure('allocate_params', function() assert(Mimir.Model.allocate_params()) end)
  B.measure('init_weights', function() assert(Mimir.Model.init_weights('xavier', 42)) end)
  log('params=' .. Mimir.Model.total_params() .. ' dtype=' .. Mimir.Model.dtype())
end
function B.create(name, cfg)
  B.measure('create ' .. name .. ' (graphe inclus)', function()
    assert(Mimir.Model.create(name, cfg))
  end)
  B.init()
end
function B.forward(input, training, expected)
  return B.vector(assert(Mimir.Model.forward(input, training or false)), expected)
end
function B.bench_forward(input, expected)
  local warmup = B.number('--warmup', nil, 1, 0, true)
  local iters = B.number('--iters', 'MIMIR_BENCH_ITERS', B.has('--quick') and 2 or 10, 1, true)
  for _ = 1, warmup do B.forward(input, false, expected) end
  local output, elapsed = B.measure('forward + transfert Lua + validation', function()
    local out
    for _ = 1, iters do out = B.forward(input, false, expected) end
    return out
  end)
  log(string.format('warmup=%d iters=%d cpu_ms/iter=%.3f output=%d', warmup, iters, elapsed*1000/iters, #output))
  return output
end
function B.snapshot(tag)
  local g, a = Mimir.MemoryGuard.getStats(), Mimir.Allocator.getStats()
  log(string.format('%s: guard_current_MiB=%.2f guard_peak_MiB=%.2f tensors=%d loaded=%d',
    tag, g.current_mb, g.peak_mb, a.tensor_count, a.loaded_count))
  -- Compteurs internes, pas le RSS; pic cumulatif, aucun reset avec modèle vivant.
end
function B.transformer(dim, layers, seq, vocab, causal)
  return {d_model=dim, num_layers=layers, num_heads=4, mlp_hidden=dim*4,
    output_dim=dim, seq_len=seq, vocab_size=vocab, padding_idx=0, causal=causal or false}
end
function B.roundtrip(cfg, input)
  B.create('basic_mlp', cfg)
  local before = B.forward(input)
  local path = os.tmpname()
  local ok, err = pcall(function()
    B.measure('save safetensors', function()
      assert(Mimir.Serialization.save(path, 'safetensors', {save_tokenizer=false, save_encoder=false}))
    end)
    B.create('basic_mlp', cfg)
    assert(Mimir.Model.init_weights('zeros', 0))
    local reset = B.forward(input, false, #before)
    local changed = false
    for i, v in ipairs(before) do if math.abs(v-reset[i]) > 1e-5 then changed=true end end
    assert(changed, 'roundtrip non discriminant: poids inchangés')
    B.measure('load safetensors', function()
      assert(Mimir.Serialization.load(path, 'safetensors', {load_tokenizer=false, load_encoder=false, strict_mode=true}))
    end)
    local after = B.forward(input, false, #before)
    for i, v in ipairs(before) do assert(math.abs(v-after[i]) < 1e-5, 'roundtrip différent') end
  end)
  os.remove(path)
  assert(ok, err)
end
return B
