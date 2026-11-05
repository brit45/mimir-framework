-- Lumen inference controls, using the shared SFML/Qt/GTK/Web Viz scene.
-- Generation runs only on the Lua thread after an explicit Generate click.
local M = {}
local fields = {
  {key="prompt", label="Prompt"}, {key="seed", label="Seed"},
  {key="steps", label="Étapes"}, {key="guidance", label="Guidance"},
  {key="output", label="Sortie PPM"},
}
local KEY = {A=0, H=7, Escape=36, Enter=58, Backspace=59, Tab=60, Left=71, Right=72}
local function copy(values)
  local result = {}
  for _, field in ipairs(fields) do result[field.key] = values[field.key] end
  return result
end
local function validate(key, text)
  if key == "prompt" then return text end -- An empty unconditional prompt is valid.
  if key == "output" then
    if not text:match("%S") or text:find("%z") then return nil, "Indiquer un chemin de sortie PPM." end
    return text
  end
  local value = tonumber(text)
  if not value or value ~= value or math.abs(value) == math.huge then return nil, "Nombre fini attendu : "..key end
  if key == "guidance" then
    if value < 0 or value > 3.402823466e38 then return nil, "La guidance doit être positive ou nulle et représentable en float32." end
  elseif value % 1 ~= 0 or value > 2147483647 or value < (key == "steps" and 1 or -2147483648) then
    return nil, key == "steps" and "Les étapes doivent être un entier positif (int32)." or "La seed doit être un entier int32."
  end
  return value
end
-- Truncate only the displayed preview, never the actual prompt/path.
local function preview(text, columns, lines)
  local chars = {}
  for _, code in utf8.codes(text) do chars[#chars+1] = code < 32 and " " or utf8.char(code) end
  local maximum = columns * lines
  if #chars > maximum then
    for i = #chars, maximum, -1 do chars[i] = nil end
    chars[#chars+1] = "…"
  end
  local rows = {}
  for i = 1, #chars, columns do rows[#rows+1] = table.concat(chars, "", i, math.min(#chars,i+columns-1)) end
  return table.concat(rows, "\n")
end
local function usable(viz)
  if type(viz) ~= "table" or type(viz.backend) ~= "function" then return false, "API Viz absente" end
  local backend = viz.backend()
  if backend == "NONE" then return false, "backend Viz NONE" end
  if viz.is_open() then return true end
  -- SFML can abort instead of throwing when X11 is absent. Avoid opening any
  -- desktop backend on a display-less Linux host; Web does not need a display.
  if backend ~= "WEB" then
    local linux = io.open("/proc/sys/kernel/ostype", "r")
    if linux then
      linux:close()
      local display,wayland=os.getenv("DISPLAY"),os.getenv("WAYLAND_DISPLAY")
      if (not display or display=="") and (backend=="SFML" or not wayland or wayland=="") then
        return false, "aucun affichage utilisable pour "..backend
      end
    end
  end
  local called, ok, err = pcall(viz.create, {visualization={enabled=true,
    window_width=1200, window_height=840, window_title="Mímir • Inférence Lumen", fps_limit=30}})
  if not called or not ok then return false, tostring(err or ok) end
  -- Successful creation followed by an immediate close is cancellation,
  -- never permission to fall through into an automatic CLI generation.
  return true
end

-- Returns true when the UI handled the run (including cancellation), false
-- only when no UI could be opened. The caller then keeps its one-shot CLI path.
function M.run(initial, generate, options)
  options = options or {}
  if options.disabled then return false end
  local viz = Mimir and Mimir.Viz
  local available, reason = usable(viz)
  if not available then return false, reason end
  local values, defaults = copy(initial), copy(initial)
  local width, height, running = 1200, 840, true
  local editing, buffer, cursor, selected = nil, "", 0, false
  local status = "Modifiez les réglages, puis cliquez sur Générer."
  local busy, last_caption = false, "Aucune inférence lancée"
  local help_text="Cliquer un champ : saisir. Ctrl+A : tout sélectionner. Flèches : curseur. "..
    "Entrée : valider. Tab : champ suivant. Échap : annuler le champ (ou fermer hors édition). Générer : lancer."
  local function scene()
    local area = math.max(240,width-32)
    local columns = math.max(1,math.floor((area-24)/9))
    local controls = {}
    local function control(id, label, x, y, w, h)
      controls[#controls+1] = {id=id,label=label,x=x,y=y,w=w,h=h}
    end
    control("heading", "Lumen • Réglages de génération", 16, 14, area, 36)
    local function label(index, cols, lines)
      local field = fields[index]
      local text=tostring(values[field.key])
      if editing==index then
        local before=buffer:sub(1,cursor)
        local limit=math.max(1,math.floor(cols*lines*.7))
        local length=utf8.len(before)
        if length>limit then before="…"..before:sub(utf8.offset(before,length-limit+1)) end
        text=before.."|"..buffer:sub(cursor+1)
      end
      return field.label..(editing == index and " [édition]" or " :").."\n"..preview(text,cols,lines)
    end
    local count = math.max(1,math.min(3,math.floor(area/220)))
    local cell = (area-(count-1)*12)/count
    local output_y = 186+math.ceil(3/count)*64
    local buttons_y = output_y+66
    local status_y = buttons_y+54
    local panel_y = status_y+58
    do
      control(busy and "busy_prompt" or "edit_prompt",label(1,columns,4),16,62,area,112)
      for index=2,4 do
        local slot=index-2
        control((busy and "busy_" or "edit_")..fields[index].key,label(index,math.max(8,math.floor((cell-24)/9)),1),
          16+(slot%count)*(cell+12),186+math.floor(slot/count)*64,cell,54)
      end
      control(busy and "busy_output" or "edit_output",label(5,columns,1),16,output_y,area,54)
    end
    if not busy then
      local button_w = math.min(180,(area-24)/3)
      control("generate","Générer",16,buttons_y,button_w,40)
      control("reset","Réinitialiser",28+button_w,buttons_y,button_w,40)
      control("cancel","Fermer",40+2*button_w,buttons_y,button_w,40)
    end
    control("status",preview(status,columns,2),16,status_y,area,48)
    local panels = {}
    for id=0,6 do panels[#panels+1] = {id=id,visible=id==2} end
    panels[3] = {id=2,visible=true,x=16,y=panel_y,w=area,h=math.max(100,height-panel_y-20),
      title=last_caption}
    -- Reserve keys throughout the form, including the frame between a field
    -- click and its Lua dispatch: an immediately typed N/C must stay text.
    return {panels=panels,controls=controls,events=true,capture_keys=true,
      image_size=math.max(32,math.min(512,height-panel_y-60)),
      help=help_text.." Le checkpoint, le tokenizer et la géométrie restent ceux chargés au lancement."}
  end
  local function publish() assert(viz.configure(scene())) end
  local function commit()
    if not editing then return true end
    local field = fields[editing]
    local value, err = validate(field.key,buffer)
    if value == nil then status=err;publish();return false end
    values[field.key]=value;editing=nil;selected=false
    status="Réglage validé. Cliquez sur Générer pour lancer l’inférence."
    return true
  end
  local function focus(index)
    if not commit() then return end
    editing=index;buffer=tostring(values[fields[index].key]);cursor=#buffer;selected=true
    status="Saisie : texte sélectionné. Entrée pour valider, Échap pour annuler."
    publish()
  end
  local function click(id)
    if id == "cancel" then running=false;return end
    for index,field in ipairs(fields) do if id=="edit_"..field.key then focus(index);return end end
    if id=="reset" then
      values=copy(defaults);editing=nil;selected=false;status="Réglages de lancement restaurés.";publish()
    elseif id=="generate" and commit() then
      for _,field in ipairs(fields) do
        local value,err=validate(field.key,tostring(values[field.key]))
        if value==nil then status=err;publish();return end
      end
      local snapshot=copy(values)
      busy=true;status=string.format("Génération en cours… seed=%d, étapes=%d, guidance=%g",
        snapshot.seed,snapshot.steps,snapshot.guidance);publish()
      local ok,pixels,w,h,c=pcall(generate,snapshot)
      -- Discard clicks/keystrokes issued against the busy scene, so they cannot
      -- silently start another generation with stale controls afterwards.
      viz.poll_events(0)
      busy=false
      if ok then
        last_caption=string.format("seed=%d • étapes=%d • guidance=%g • %s",
          snapshot.seed,snapshot.steps,snapshot.guidance,snapshot.prompt)
        assert(viz.clear())
        assert(viz.add_image(pixels,w,h,c,last_caption))
        status="Image enregistrée : "..snapshot.output
      else
        status="Échec : "..tostring(pixels)
        print("[lumen_text2img] "..status)
      end
      publish()
      return true
    end
  end
  local function key(event)
    if not editing then
      if event.code==KEY.H then status=help_text;publish()
      elseif event.code==KEY.Escape then running=false
      elseif event.code==KEY.Tab then focus(1) end
      return
    end
    if event.code==KEY.Escape then editing=nil;selected=false;status="Édition annulée.";publish();return end
    if event.code==KEY.Enter then if commit() then publish() end;return end
    if event.code==KEY.Tab then local next_index=editing%#fields+1;if commit() then focus(next_index) end;return end
    if event.control and event.code==KEY.A then selected=true
    elseif event.code==KEY.Backspace then
      if selected then buffer="";cursor=0
      elseif cursor>0 then
        local previous=utf8.offset(buffer,-1,cursor+1)
        buffer=buffer:sub(1,previous-1)..buffer:sub(cursor+1);cursor=previous-1
      end
      selected=false
    elseif event.code==KEY.Left then
      if cursor>0 then cursor=utf8.offset(buffer,-1,cursor+1)-1 end;selected=false
    elseif event.code==KEY.Right then
      if cursor<#buffer then cursor=(utf8.offset(buffer,2,cursor+1) or (#buffer+1))-1 end;selected=false
    end
    publish()
  end
  local function text(event)
    local code=event.unicode
    if not editing or not code or code<32 or code==127 or code>0x10ffff or (code>=0xd800 and code<=0xdfff) then return end
    if selected then buffer="";cursor=0;selected=false end
    local character=utf8.char(code)
    buffer=buffer:sub(1,cursor)..character..buffer:sub(cursor+1);cursor=cursor+#character
    publish()
  end
  local ok,err=xpcall(function()
    publish()
    while running and viz.is_open() do
      for _,event in ipairs(viz.poll_events(30)) do
        if not running then break end
        if event.type=="close" then running=false
        elseif event.type=="click" then if click(event.id) then break end
        elseif event.type=="key" then key(event)
        elseif event.type=="text" then text(event)
        elseif (event.type=="resize" or event.type=="configured") and event.width and event.height and
            (width~=event.width or height~=event.height) then
          width=event.width;height=event.height;publish()
        end
      end
    end
  end,debug.traceback)
  viz.set_enabled(false)
  if not ok then error(err,0) end
  return true
end
return M
