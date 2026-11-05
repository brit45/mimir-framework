-- Exercise the real inference script with bounded model/Viz/file doubles.
ROOTWORK=ROOTWORK or "."
local original_dofile,original_open=dofile,io.open
local original_mimir,original_arg=Mimir,arg
local defaults={type="lumen_diffusion",image_w=8,image_h=8,image_c=3,latent_w=4,latent_h=4,latent_c=4,
  patch_size=2,hidden_size=8,depth=1,mlp_ratio=2,vae_base_channels=8,vae_scale=1,vae_checkpoint="fake-vae"}
local function run(mode)
  local calls,files,created,frame={}, {},0,0
  arg={"--allow-untrained","--vae-checkpoint","fake-vae","--vae-scale","1",
    "--tokenizer","fake-tokenizer","--prompt","CLI prompt","--seed","7","--steps","3",
    "--guidance","2.5","--out","cli.ppm"}
  if mode=="headless" then arg[#arg+1]="--headless" end
  if mode=="no-viz" then arg[#arg+1]="--no-viz" end
  local open=false
  Mimir={Architectures={default_config=function()
    local cfg={};for k,v in pairs(defaults) do cfg[k]=v end;return cfg
  end},Tokenizer={load=function() return true end},Model={
    create=function() return true end,allocate_params=function() return true end,init_weights=function() return true end,
    lumen_text2img=function(prompt,seed,steps,guidance)
      calls[#calls+1]={prompt=prompt,seed=seed,steps=steps,guidance=guidance}
      return {255,0,0,0,255,0},2,1,3
    end,
  },Viz={}}
  local viz=Mimir.Viz
  function viz.backend() return mode=="unavailable" and "NONE" or "WEB" end
  function viz.create() created=created+1;if mode=="create-failed" then return false,"offline" end;open=true;return true end
  function viz.is_open() return open end
  function viz.configure() return true end
  function viz.clear() return true end
  function viz.add_image() return true end
  function viz.set_enabled() open=false;return true end
  function viz.poll_events(timeout)
    if timeout==0 then return {} end
    frame=frame+1
    if frame>1 or mode=="cancel" then return {{type="click",id="cancel"}} end
    local result={}
    local function click(id) result[#result+1]={type="click",id=id} end
    local function text(s) for _,c in utf8.codes(s) do result[#result+1]={type="text",unicode=c} end end
    local function enter() result[#result+1]={type="key",code=58} end
    click("edit_prompt");text("UI prompt");enter()
    click("edit_seed");text("42");enter()
    click("edit_steps");text("9");enter()
    click("edit_guidance");text("1.25");enter()
    click("edit_output");text("gui.ppm");enter()
    click("generate")
    return result
  end
  dofile=function(path)
    if path:match("/scripts/modules/fs.lua$") then
      return {dirname=function() return "." end,mkdir_p=function() return true end}
    elseif path:match("/scripts/modules/checkpoint_resume.lua$") then
      return {resolve_dir=function(path_value) return path_value end}
    end
    return original_dofile(path)
  end
  io.open=function(path,mode_value)
    assert(mode_value=="wb","unexpected file access")
    local data=""
    return {write=function(_,value) data=data..value end,close=function() files[path]=data end}
  end
  local ok,err=xpcall(function() original_dofile(ROOTWORK.."/scripts/inferences/lumen_text2img.lua") end,debug.traceback)
  dofile=original_dofile;io.open=original_open
  assert(ok,err)
  if mode=="cancel" then
    assert(#calls==0 and next(files)==nil,"closing the form launched inference")
  elseif mode=="gui" then
    assert(#calls==1 and calls[1].prompt=="UI prompt" and calls[1].seed==42 and calls[1].steps==9 and calls[1].guidance==1.25)
    assert(files["gui.ppm"]=="P6\n2 1\n255\n"..string.char(255,0,0,0,255,0) and not files["cli.ppm"])
  else
    assert(#calls==1 and calls[1].prompt=="CLI prompt" and calls[1].seed==7 and calls[1].steps==3 and calls[1].guidance==2.5)
    assert(files["cli.ppm"] and not files["gui.ppm"])
    if mode=="headless" or mode=="no-viz" or mode=="unavailable" then assert(created==0) end
  end
end
for _,mode in ipairs({"gui","cancel","headless","no-viz","unavailable","create-failed"}) do run(mode) end
Mimir=original_mimir;arg=original_arg
print("LUMEN_INFERENCE_SCRIPT_OK actual script, edited arguments/output, cancel, CLI and unavailable Viz")
