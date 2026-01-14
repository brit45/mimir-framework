-- Entraînement minimal d'un modèle "basic_mlp" (régression) via API Lua.
-- Usage: ./bin/mimir --lua scripts/examples/train_basic_mlp.lua

math.randomseed(123)

local function mse_and_grad(pred, target)
  local n = #pred
  local grad = {}
  local mse = 0.0
  for i = 1, n do
    local d = pred[i] - target[i]
    mse = mse + d * d
  end
  mse = mse / n
  for i = 1, n do
    grad[i] = (2.0 / n) * (pred[i] - target[i])
  end
  return mse, grad
end

local function rand_vec(n)
  local t = {}
  for i = 1, n do
    t[i] = (math.random() * 2.0 - 1.0)
  end
  return t
end

-- 1) Vérifier le registry + récupérer une config par défaut
local names = Mimir.Architectures.available()
print("Architectures disponibles:", table.concat(names, ", "))

local cfg = assert(Mimir.Architectures.default_config("basic_mlp"), "default_config('basic_mlp') a retourné nil")

-- 2) Paramétrer une petite tâche: apprendre l'identité y = x
cfg.input_dim = 8
cfg.hidden_dim = 32
cfg.output_dim = 8
cfg.hidden_layers = 2
cfg.dropout = 0.0

assert(Mimir.Model.create("basic_mlp", cfg))
assert(Mimir.Model.build())
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier", 123))

local lr = 1e-2
local steps = 200

for step = 1, steps do
  local x = rand_vec(cfg.input_dim)
  local y = x -- identité

  assert(Mimir.Model.zero_grads())
  local pred = assert(Mimir.Model.forward(x, true))
  local loss, grad = mse_and_grad(pred, y)

  assert(Mimir.Model.backward(grad))
  assert(Mimir.Model.optimizer_step(lr, "adamw"))

  if step % 20 == 0 then
    print(string.format("step=%d/%d  loss=%.6f", step, steps, loss))
  end
end

print("OK: entraînement basic_mlp terminé")
