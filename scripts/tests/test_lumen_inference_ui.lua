-- No model/checkpoint required: drive the actual Lua controller through Viz events.
ROOTWORK = ROOTWORK or "."
local UI=dofile(ROOTWORK.."/scripts/modules/lumen_inference_ui.lua")
local initial={prompt="un paysage",seed=1337,steps=30,guidance=5,output="image.ppm"}
local function click(id) return {type="click",id=id} end
local function key(code,control) return {type="key",code=code,control=control} end
local function typing(text)
  local result={}
  for _,code in utf8.codes(text) do result[#result+1]={type="text",unicode=code} end
  return result
end
local function events(...)
  local result={}
  for _,item in ipairs({...}) do
    if item.type then result[#result+1]=item else for _,event in ipairs(item) do result[#result+1]=event end end
  end
  return result
end
local function fake(frames,options)
  options=options or {}
  local viz={scenes={},images={},created=0,stopped=0,open=false,index=0}
  function viz.backend() return options.backend or "WEB" end
  function viz.is_open() return viz.open end
  function viz.create()
    viz.created=viz.created+1
    if options.fail then return false,"cannot open display" end
    viz.open=not options.closed;return true
  end
  function viz.configure(scene)
    assert(scene.events and type(scene.capture_keys)=="boolean")
    local ids={}
    for _,control in ipairs(scene.controls) do
      assert(not ids[control.id]);ids[control.id]=true
      assert(control.w>0 and control.h>0)
    end
    viz.scenes[#viz.scenes+1]=scene;return true
  end
  function viz.poll_events(timeout)
    if timeout==0 then return {} end
    viz.index=viz.index+1
    return frames[viz.index] or {click("cancel")}
  end
  function viz.clear() viz.images={};return true end
  function viz.add_image(pixels,w,h,c,caption)
    viz.images[#viz.images+1]={pixels=pixels,w=w,h=h,c=c,caption=caption};return true
  end
  function viz.set_enabled(enabled) assert(not enabled);viz.open=false;viz.stopped=viz.stopped+1;return true end
  Mimir={Viz=viz}
  return viz
end
local calls={}
local function generate(settings)
  assert(Mimir.Viz.scenes[1].controls[2].id=="edit_prompt")
  calls[#calls+1]=settings
  return {12,34,56},1,1,3
end
local viz=fake({{click("cancel")}})
assert(UI.run(initial,generate) and #calls==0 and viz.stopped==1,"cancellation inferred anyway")
viz=fake({},{closed=true})
assert(UI.run(initial,generate) and #calls==0,"immediate window close launched CLI inference")
viz=fake({{key(36)}})
assert(UI.run(initial,generate) and #calls==0,"Escape inferred instead of cancelling")
for _,options in ipairs({{backend="NONE"},{fail=true}}) do
  viz=fake({},options)
  assert(not UI.run(initial,generate) and #calls==0,"unavailable Viz launched inference")
end
viz=fake({})
assert(not UI.run(initial,generate,{disabled=true}) and viz.created==0,"headless opened Viz")

viz=fake({
  events(click("edit_prompt"),typing("été"),key(59),typing("é"),key(58),
    click("edit_seed"),typing("42"),key(60),typing("12"),key(58),
    click("edit_guidance"),typing("2.5"),key(58),
    click("edit_output"),typing("résultat.ppm"),key(58),click("generate"),click("generate")),
  events(click("edit_seed"),typing("9"),key(36),click("generate")),
  events(click("reset"),click("generate")),
  {click("cancel")},
})
assert(UI.run(initial,generate) and #calls==3,"queued duplicate generation was accepted")
assert(calls[1].prompt=="été" and calls[1].seed==42 and calls[1].steps==12 and calls[1].guidance==2.5)
assert(calls[1].output=="résultat.ppm" and calls[2].seed==42,"edited seed cancellation changed settings")
assert(calls[3].prompt==initial.prompt and calls[3].seed==1337 and initial.output=="image.ppm")
assert(viz.images[1].w==1 and viz.images[1].c==3 and viz.stopped==1)

for _,case in ipairs({{"steps","0"},{"steps","1.5"},{"seed","2147483648"},
    {"guidance","-1"},{"guidance","1e100"},{"output"," "}}) do
  local before=#calls
  viz=fake({events(click("edit_"..case[1]),typing(case[2]),click("generate")),
    events(key(36),click("cancel"))})
  assert(UI.run(initial,generate) and #calls==before,"invalid field launched inference: "..case[1])
end
-- Cursor movement/backspace must remove a whole UTF-8 codepoint.
local before=#calls
viz=fake({events(click("edit_prompt"),typing("éà"),key(71),key(59),typing("ô"),key(58),
  {type="resize",width=480,height=700},click("generate")),{click("cancel")}})
assert(UI.run(initial,generate) and calls[before+1].prompt=="ôà")
-- Display wrapping must not impose a fixed prompt length on native Lumen.
local long_initial={}
for k,v in pairs(initial) do long_initial[k]=v end
long_initial.prompt=string.rep("un paysage éclairé ",200)
local long_before=#calls
viz=fake({{click("generate")},{click("cancel")}})
assert(UI.run(long_initial,generate) and calls[long_before+1].prompt==long_initial.prompt,
  "display preview truncated the actual prompt")
-- A generation error leaves the form usable for a subsequent explicit attempt.
local attempts=0
viz=fake({{click("generate")},{click("generate")},{click("cancel")}})
assert(UI.run(initial,function(settings)
  attempts=attempts+1
  if attempts==1 then error("prompt too long") end
  return generate(settings)
end) and attempts==2 and viz.stopped==1)
print("LUMEN_INFERENCE_UI_OK preflight, edits, UTF-8, validation, cancel, repeat, errors, fallback")
