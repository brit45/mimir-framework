-- Integration regression for the real Lua dataset training / checkpoint path.
local FS = dofile(ROOTWORK .. "/scripts/modules/fs.lua")
local root = os.tmpname()
os.remove(root)
assert(FS.mkdir_p(FS.join(root, "dataset")))
local source = assert(io.open(ROOTWORK .. "/logo.png", "rb"))
local data = source:read("*a"); source:close()
local image = assert(io.open(FS.join(root, "dataset", "sample.png"), "wb"))
image:write(data); image:close()
local cfg = assert(Mimir.Architectures.default_config("vae_conv"))
cfg.image_w=16; cfg.image_h=16; cfg.image_c=3
cfg.base_channels=4; cfg.latent_c=2; cfg.latent_h=4; cfg.latent_w=4
cfg.latent_dim=32; cfg.resnet=false; cfg.attention=false
cfg.stochastic_latent=false; cfg.text_cond=false
cfg.max_items=1; cfg.validate_every_steps=0; cfg.validate_items=0
cfg.autosave_every_epochs=1; cfg.checkpoint_dir=FS.join(root,"checkpoints")
cfg.optimizer="adam"; cfg.decay_strategy="cosine"; cfg.kl_beta=0
assert(Mimir.Dataset.load(FS.join(root,"dataset"),16,16,1,true,FS.join(root,"cache.json"),128,true))
local function create()
  assert(Mimir.Model.create("vae_conv",cfg))
  assert(Mimir.Model.allocate_params())
end
local function metadata(path)
  local value,err=read_json(FS.join(path,"model","training.json"))
  assert(type(value)=="table",tostring(err))
  return value
end
create()
assert(Mimir.Model.init_weights("he",123))
local ok,first_steps=Mimir.Model.train(1,0.001)
assert(ok,tostring(first_steps))
local first=FS.join(cfg.checkpoint_dir,"epoch_0001")
local before=metadata(first)
assert(before.loop.completed_epochs==1)
assert(before.step==first_steps)
create()
assert(Mimir.Serialization.load(first,"raw_folder",{load_optimizer=true,load_encoder=false,load_tokenizer=false,strict_mode=true}))
local resumed,steps=Mimir.Model.train(2,0.001)
assert(resumed,tostring(steps))
assert(steps==first_steps+2,"global training step restarted")
local after=metadata(FS.join(cfg.checkpoint_dir,"epoch_0003"))
assert(after.step==steps)
assert(after.total_steps==before.total_steps,"scheduler duration changed on resume")
assert(after.loop.completed_epochs==3)
assert(before.step==metadata(first).step,"resume overwrote the previous checkpoint")
assert(os.execute((FS.is_windows() and "rmdir /s /q " or "rm -rf ") .. FS.quote(root)))
print("TRAINING_RESUME_OK")
