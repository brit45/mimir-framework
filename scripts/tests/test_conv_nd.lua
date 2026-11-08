-- Exercise the public model path, not a standalone convolution helper.
for rank = 3, 4 do
  local shape, spatial = {2, 2}, {}
  for axis = 1, rank do
    spatial[axis] = 2
    shape[#shape + 1] = 2
  end
  assert(Mimir.Model.create_empty("conv_nd_test", {}))
  assert(Mimir.Model.set_hardware(false))
  local ok, err = Mimir.Model.push_layer("conv", "Conv" .. rank .. "D", 0, {
    shape = shape, input_spatial_shape = spatial, kernel_shape = {1},
    in_channels = 2, out_channels = 2, groups = 2, use_bias = true,
  })
  assert(ok, tostring(err))
  local allocated, count = Mimir.Model.allocate_params()
  assert(allocated, tostring(count))
  assert(count == 4, "grouped 1x1 convolution requires two weights and two biases")
  assert(Mimir.Model.init_weights("xavier", 37))
  local layers = Mimir.Model.get_layers()
  assert(#layers == 1 and layers[1].type == "Conv" .. rank .. "d")
  assert(#layers[1].input_spatial_shape == rank and layers[1].kernel_shape[1] == 1)
  local input, gradient = {}, {}
  for i = 1, 4 * 2 ^ rank do
    input[i] = (i % 7 - 3) / 7
    gradient[i] = 0.01
  end
  assert(Mimir.Model.zero_grads())
  local output, forward_err = Mimir.Model.forward(input, true)
  assert(output, tostring(forward_err))
  assert(#output == #input)
  local backward, backward_err = Mimir.Model.backward(gradient)
  assert(backward, tostring(backward_err))
  local step, step_err = Mimir.Model.optimizer_step(0.01, "sgd")
  assert(step, tostring(step_err))
  local updated = assert(Mimir.Model.forward(input, false))
  local changed = false
  for i = 1, #output do
    assert(updated[i] == updated[i] and math.abs(updated[i]) < math.huge)
    changed = changed or output[i] ~= updated[i]
  end
  assert(changed, "optimizer must update the convolution")
end
local caps = Mimir.Model.hardware_caps()
assert(type(caps.simd) == "table" and type(caps.simd.selected) == "string")
print("CONV_ND_LUA_FORWARD_BACKWARD_OK")
