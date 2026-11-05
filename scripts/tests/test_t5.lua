-- Native CPU smoke: registry, denoising backward, optimizer and encoder export contract.
local cfg = assert(Mimir.Architectures.default_config("t5"))
for k,v in pairs({vocab_size=16,seq_len=4,d_model=8,num_layers=1,num_heads=2,mlp_hidden=16}) do cfg[k]=v end
assert(Mimir.Model.create("t5",cfg))
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier",42))
local input={2,15,4,5}
local function forward(training)
  local out,err=Mimir.Model.forward({text_ids=input},training)
  assert(out, tostring(err)); assert(#out==64)
  for _,v in ipairs(out) do assert(v==v and math.abs(v)<1e10) end
  return out
end
local before=forward(false)
assert(Mimir.Model.zero_grads())
local logits=forward(true)
local gradient={}
for i=1,#logits do gradient[i]=(i==19) and -1 or 0 end
assert(Mimir.Model.backward(gradient))
assert(Mimir.Model.optimizer_step(0.001,"adamw"))
local after=forward(false)
local difference=0
for i=1,#before do difference=difference+math.abs(before[i]-after[i]) end
assert(difference>1e-7, "optimizer did not change logits")
cfg.encoder_only=true
assert(Mimir.Model.create("t5",cfg))
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier",42))
local encoded=assert(Mimir.Model.forward({text_ids=input},false))
assert(#encoded==32)
print("T5_SMOKE_OK")
