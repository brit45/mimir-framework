-- Public Lua graph construction and native runtime execution for Primer math.
local input = {-2.0, -0.5, 0.5, 2.0}
local unary = {
  {"SquaredReLU", function(x) return math.max(0, x)^2 end},
  {"Square", function(x) return x*x end},
  {"SqrtAbs", function(x) return math.sqrt(math.abs(x)) end},
  {"Exp", math.exp},
  {"LogAbs", function(x) return math.log(math.abs(x)) end},
  {"Abs", math.abs},
  {"ReciprocalNoNan", function(x) return x == 0 and 0 or 1/x end},
  {"Sign", function(x) return x > 0 and 1 or x < 0 and -1 or 0 end},
  {"Sin", math.sin}, {"Cos", math.cos},
}
local function near(actual, expected, label)
  assert(actual == actual and math.abs(actual - expected) <= 1e-5 * math.max(1, math.abs(expected)), label)
end
local function create(kind, options, canonical)
  assert(Mimir.Model.create_empty("primer_lua", {dtype = "float32"}))
  assert(Mimir.Model.set_hardware(false))
  options = options or {}
  options.inputs = options.inputs or {"__input__"}
  options.output = "x"
  local ok, err = Mimir.Model.push_layer("operation", kind, 0, options)
  assert(ok, tostring(err))
  assert(Mimir.Model.set_layer_io("operation", options.inputs, options.output))
  local allocated, count = Mimir.Model.allocate_params()
  assert(allocated, tostring(count))
  assert(count == 0, kind .. " must be stateless")
  local layers = assert(Mimir.Model.get_layers())
  assert(#layers == 1 and layers[1].type == (canonical or kind), "canonical layer type " .. kind)
end
for _, case in ipairs(unary) do
  create(case[1])
  assert(Mimir.Model.zero_grads())
  local output, err = Mimir.Model.forward(input, true)
  assert(output, tostring(err))
  assert(#output == #input)
  for i, value in ipairs(input) do near(output[i], case[2](value), case[1] .. " output") end
  assert(Mimir.Model.backward({0.5, -0.25, 0.75, 1.0}))
end
for _, kind in ipairs({"Maximum", "Minimum"}) do
  create(kind, {inputs = {"__input__", "bound"}})
  assert(Mimir.Model.zero_grads())
  local output = assert(Mimir.Model.forward({__input__ = input, bound = {0.5}}, true))
  for i, value in ipairs(input) do
    near(output[i], kind == "Maximum" and math.max(value, 0.5) or math.min(value, 0.5), kind .. " broadcast")
  end
  assert(Mimir.Model.backward({0.5, -0.25, 0.75, 1.0}))
end
local reduction = {ReduceSum = {-2.5, 2.5}, ReduceMean = {-1.25, 1.25},
                   ReduceMin = {-2, 0.5}, ReduceMax = {-0.5, 2}, ReduceProd = {1, 1}}
for _, kind in ipairs({"ReduceSum", "ReduceMean", "ReduceMin", "ReduceMax", "ReduceProd"}) do
  create(kind, {shape = {2, 2}, axis = -1})
  assert(Mimir.Model.zero_grads())
  local output = assert(Mimir.Model.forward(input, true))
  assert(#output == 2)
  for i, expected in ipairs(reduction[kind]) do near(output[i], expected, kind .. " axis") end
  assert(Mimir.Model.backward({0.5, -0.25}))
end
create("LowerTriangular", {shape = {2, 3}})
assert(Mimir.Model.zero_grads())
local triangular = assert(Mimir.Model.forward({1, 2, 3, 4, 5, 6}, true))
local expected = {1, 0, 0, 4, 5, 0}
for i, value in ipairs(expected) do near(triangular[i], value, "rectangular causal mask") end
assert(Mimir.Model.backward({1, 1, 1, 1, 1, 1}))

for _, alias in ipairs({{"ReLU2", "SquaredReLU"}, {"AbsRoot", "SqrtAbs"}, {"reduce_sum", "ReduceSum"}}) do
  create(alias[1], alias[2] == "ReduceSum" and {shape = {2, 2}, axis = -1} or {}, alias[2])
  local values = assert(Mimir.Model.forward(input, false))
  assert(#values == (alias[2] == "ReduceSum" and 2 or 4), "alias graph shape " .. alias[1])
end

-- A learned projection ensures the public backward/optimizer and SafeTensors
-- paths are exercised with actual parameter storage around the new operators.
local function learned_graph()
  assert(Mimir.Model.create_empty("primer_lua_checkpoint", {dtype = "float32"}))
  assert(Mimir.Model.set_hardware(false))
  assert(Mimir.Model.push_layer("projection", "Linear", 1,
      {inputs = {"__input__"}, output = "projected", in_features = 1, out_features = 1,
       seq_len = 4, use_bias = false}))
  assert(Mimir.Model.set_layer_io("projection", {"__input__"}, "projected"))
  assert(Mimir.Model.push_layer("squared", "SquaredReLU", 0, {inputs = {"projected"}, output = "squared"}))
  assert(Mimir.Model.set_layer_io("squared", {"projected"}, "squared"))
  assert(Mimir.Model.push_layer("reduce", "ReduceSum", 0, {inputs = {"squared"}, output = "x", axis = 0}))
  assert(Mimir.Model.set_layer_io("reduce", {"squared"}, "x"))
  assert(Mimir.Model.allocate_params())
end
learned_graph()
assert(Mimir.Model.init_weights("xavier", 343))
assert(Mimir.Model.zero_grads())
local initial = assert(Mimir.Model.forward(input, true))
assert(Mimir.Model.backward({1}))
local gradients = assert(Mimir.Model.get_gradients())
assert(#gradients == 1 and math.abs(gradients[1]) > 0, "learned projection must receive a gradient")
assert(Mimir.Model.optimizer_step(0.001, "sgd"))
local trained = assert(Mimir.Model.forward(input, false))
assert(trained[1] ~= initial[1], "optimizer did not update the learned projection")
local base = os.tmpname()
os.remove(base)
local checkpoint = base .. ".safetensors"
assert(Mimir.Serialization.save(checkpoint, "safetensors", {
  save_tokenizer = false, save_encoder = false, save_optimizer = false, include_git_info = false,
}))
learned_graph()
assert(Mimir.Serialization.load(checkpoint, "safetensors", {
  load_tokenizer = false, load_encoder = false, load_optimizer = false, strict_mode = true,
}))
local resumed = assert(Mimir.Model.forward(input, false))
near(resumed[1], trained[1], "checkpoint numerical parity")
os.remove(checkpoint)
print("PRIMER_LUA_NATIVE_FORWARD_BACKWARD_SERIALIZATION_OK")
