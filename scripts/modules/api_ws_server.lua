-- scripts/api_ws_server.lua
-- Serveur HTTP + WebSocket minimal pour piloter Mímir via REST (GET/POST/PUT/PATCH/DELETE)
-- Dépendances: LuaSocket (luasocket)
-- Usage (Lua système):   lua scripts/api_ws_server.lua
-- Usage (avec Mímir):    ./bin/mimir --lua scripts/api_ws_server.lua -- --host 127.0.0.1 --port 8088
-- Alias flags: --api-host/--api-port

local Args
local opts = {}
do
  local ok_args, mod_or_err = pcall(dofile, "scripts/modules/args.lua")
  if ok_args and type(mod_or_err) == "table" and type(mod_or_err.parse) == "function" then
    Args = mod_or_err
    opts = Args.parse(arg) or {}
  end
end

local function opt_str(key, default)
  if type(opts) ~= "table" then return default end
  local v = opts[key]
  if v == nil and key:find("-", 1, true) then
    v = opts[key:gsub("-", "_")]
  elseif v == nil then
    v = opts[key:gsub("_", "-")]
  end
  if v == nil or v == true then return default end
  return tostring(v)
end

local function opt_int(key, default)
  local v = opt_str(key, nil)
  if v == nil then return default end
  local n = tonumber(v)
  if n == nil then return default end
  return math.floor(n)
end

local function opt_bool(key, default)
  if type(opts) ~= "table" then return default end
  local v = opts[key]
  if v == nil and key:find("-", 1, true) then
    v = opts[key:gsub("-", "_")]
  elseif v == nil then
    v = opts[key:gsub("_", "-")]
  end
  if v == nil then return default end
  if type(v) == "boolean" then return v end
  local s = tostring(v):lower()
  if s == "1" or s == "true" or s == "yes" or s == "on" then return true end
  if s == "0" or s == "false" or s == "no" or s == "off" then return false end
  return default
end

local HOST = opt_str("host", opt_str("api-host", "127.0.0.1"))
local PORT = opt_int("port", opt_int("api-port", 8088))

local SERVER_NAME = opt_str("server-name", "mimir-lua-api")

local function safe_read_file(path)
  local f = io.open(path, "rb")
  if not f then return nil end
  local s = f:read("*a")
  f:close()
  if not s then return nil end
  return (s:gsub("%s+$", ""))
end

local SERVER_VERSION = safe_read_file("VERSION") or safe_read_file("./VERSION") or "unknown"

-- =========================
-- Auth / TLS options
-- =========================
local function opt_file_str(key)
  local path = opt_str(key, nil)
  if not path or path == "" then return nil end
  return safe_read_file(path)
end

-- Auth modes:
-- - off: no auth
-- - token: Authorization: Bearer <token> (or query ?token=... for WS/browser)
-- - api_key: X-API-Key: <key> (or query ?api_key=...)
-- - either: accept either token or api key
local AUTH_MODE = (opt_str("auth", "off") or "off"):lower()
local AUTH_TOKEN = opt_str("auth-token", nil) or opt_file_str("auth-token-file")
local API_KEY = opt_str("api-key", nil) or opt_file_str("api-key-file")
local AUTH_SKIP_PATHS = opt_str("auth-skip-paths", "/health,/help")

local TLS_ENABLED = opt_bool("tls", false) or opt_bool("https", false)
local TLS_CERT_FILE = opt_str("tls-cert", "")
local TLS_KEY_FILE = opt_str("tls-key", "")

-- =========================
-- CORS (pour UI web)
-- =========================
-- Par défaut, on active CORS pour simplifier l'usage depuis une page web.
-- Flags:
--   --cors true|false
--   --cors-origin "*" | "reflect" | "http://localhost:5173" | "null"
local CORS_ENABLED = opt_bool("cors", true)
local CORS_ORIGIN = opt_str("cors-origin", "*")
local CORS_ALLOW_METHODS = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
local CORS_ALLOW_HEADERS = "Content-Type, Authorization, X-API-Key, Accept"
local CORS_EXPOSE_HEADERS = "X-Response-Time-Ms"
local CORS_MAX_AGE = "600"

-- =========================
-- Dépendances (LuaSocket)
-- =========================
local ok_socket, socket = pcall(require, "socket")
if not ok_socket then
  error(
    "LuaSocket manquant: installez-le (ex: `luarocks install luasocket`) " ..
    "ou lancez ce script avec un Lua qui l'a. Détail: " .. tostring(socket)
  )
end

local ssl
do
  local ok_ssl, mod_or_err = pcall(require, "ssl")
  if ok_ssl then
    ssl = mod_or_err
  end
end

local function now_seconds()
  if type(socket) == "table" and type(socket.gettime) == "function" then
    return socket.gettime()
  end
  return os.clock()
end

local function iso_utc(ts)
  -- os.date("!...") ne gère que la seconde, donc on garde aussi la latence ms séparée.
  return os.date("!%Y-%m-%dT%H:%M:%SZ", ts or os.time())
end

-- =========================
-- JSON minimal (encode/decode)
-- =========================
local function is_array(t)
  if type(t) ~= "table" then return false end
  local n = 0
  for k, _ in pairs(t) do
    if type(k) ~= "number" then return false end
    n = math.max(n, k)
  end
  for i = 1, n do
    if rawget(t, i) == nil then return false end
  end
  return true
end

local function json_escape(s)
  return (s:gsub("[\\\"%z\1-\31]", function(c)
    local byte = string.byte(c)
    if c == "\\" then return "\\\\" end
    if c == "\"" then return "\\\"" end
    if c == "\b" then return "\\b" end
    if c == "\f" then return "\\f" end
    if c == "\n" then return "\\n" end
    if c == "\r" then return "\\r" end
    if c == "\t" then return "\\t" end
    return string.format("\\u%04x", byte)
  end))
end

local function json_encode(v)
  local tv = type(v)
  if v == nil then return "null" end
  if tv == "boolean" then return v and "true" or "false" end
  if tv == "number" then
    if v ~= v or v == math.huge or v == -math.huge then return "null" end
    return tostring(v)
  end
  if tv == "string" then return '"' .. json_escape(v) .. '"' end
  if tv == "table" then
    if is_array(v) then
      local out = {}
      for i = 1, #v do out[#out+1] = json_encode(v[i]) end
      return "[" .. table.concat(out, ",") .. "]"
    else
      local out = {}
      for k, val in pairs(v) do
        if type(k) == "string" then
          out[#out+1] = '"' .. json_escape(k) .. '":' .. json_encode(val)
        end
      end
      return "{" .. table.concat(out, ",") .. "}"
    end
  end
  return "null"
end

local function json_decode(str)
  local i = 1
  local function skip_ws()
    while true do
      local c = str:sub(i, i)
      if c == "" then return end
      if c == " " or c == "\n" or c == "\r" or c == "\t" then
        i = i + 1
      else
        return
      end
    end
  end

  local function parse_string()
    local out = {}
    i = i + 1 -- skip opening quote
    while true do
      local c = str:sub(i, i)
      if c == "" then error("JSON: string non terminée") end
      if c == '"' then
        i = i + 1
        return table.concat(out)
      end
      if c == "\\" then
        local n = str:sub(i+1, i+1)
        if n == "\"" or n == "\\" or n == "/" then out[#out+1] = n; i = i + 2
        elseif n == "b" then out[#out+1] = "\b"; i = i + 2
        elseif n == "f" then out[#out+1] = "\f"; i = i + 2
        elseif n == "n" then out[#out+1] = "\n"; i = i + 2
        elseif n == "r" then out[#out+1] = "\r"; i = i + 2
        elseif n == "t" then out[#out+1] = "\t"; i = i + 2
        elseif n == "u" then
          local hex = str:sub(i+2, i+5)
          if not hex:match("^[0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F]$") then
            error("JSON: escape \\u invalide")
          end
          local code = tonumber(hex, 16)
          if code <= 0x7F then
            out[#out+1] = string.char(code)
          elseif code <= 0x7FF then
            out[#out+1] = string.char(0xC0 + math.floor(code/0x40), 0x80 + (code % 0x40))
          else
            out[#out+1] = string.char(0xE0 + math.floor(code/0x1000), 0x80 + (math.floor(code/0x40) % 0x40), 0x80 + (code % 0x40))
          end
          i = i + 6
        else
          error("JSON: escape invalide")
        end
      else
        out[#out+1] = c
        i = i + 1
      end
    end
  end

  local function parse_number()
    local start = i
    local c = str:sub(i, i)
    if c == "-" then i = i + 1 end
    while str:sub(i, i):match("%d") do i = i + 1 end
    if str:sub(i, i) == "." then
      i = i + 1
      while str:sub(i, i):match("%d") do i = i + 1 end
    end
    local e = str:sub(i, i)
    if e == "e" or e == "E" then
      i = i + 1
      local s = str:sub(i, i)
      if s == "+" or s == "-" then i = i + 1 end
      while str:sub(i, i):match("%d") do i = i + 1 end
    end
    local num = tonumber(str:sub(start, i-1))
    if num == nil then error("JSON: nombre invalide") end
    return num
  end

  local parse_value
  local function parse_array()
    i = i + 1
    skip_ws()
    local arr = {}
    if str:sub(i, i) == "]" then i = i + 1; return arr end
    while true do
      arr[#arr+1] = parse_value()
      skip_ws()
      local c = str:sub(i, i)
      if c == "," then i = i + 1; skip_ws()
      elseif c == "]" then i = i + 1; return arr
      else error("JSON: tableau invalide") end
    end
  end

  local function parse_object()
    i = i + 1
    skip_ws()
    local obj = {}
    if str:sub(i, i) == "}" then i = i + 1; return obj end
    while true do
      if str:sub(i, i) ~= '"' then error("JSON: clé attendue") end
      local k = parse_string()
      skip_ws()
      if str:sub(i, i) ~= ":" then error("JSON: ':' attendu") end
      i = i + 1
      skip_ws()
      obj[k] = parse_value()
      skip_ws()
      local c = str:sub(i, i)
      if c == "," then i = i + 1; skip_ws()
      elseif c == "}" then i = i + 1; return obj
      else error("JSON: objet invalide") end
    end
  end

  function parse_value()
    skip_ws()
    local c = str:sub(i, i)
    if c == '"' then return parse_string() end
    if c == "{" then return parse_object() end
    if c == "[" then return parse_array() end
    if c == "-" or c:match("%d") then return parse_number() end
    if str:sub(i, i+3) == "true" then i = i + 4; return true end
    if str:sub(i, i+4) == "false" then i = i + 5; return false end
    if str:sub(i, i+3) == "null" then i = i + 4; return nil end
    error("JSON: valeur invalide")
  end

  local val = parse_value()
  skip_ws()
  if i <= #str then
    local rest = str:sub(i)
    if rest:match("%S") then error("JSON: trailing data") end
  end
  return val
end

-- =========================
-- Base64 + SHA1 (WebSocket)
-- =========================
local function base64_encode(data)
  local b = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
  return ((data:gsub('.', function(x)
    local r, byte = '', x:byte()
    for i2 = 8, 1, -1 do r = r .. (byte % 2^i2 - byte % 2^(i2-1) > 0 and '1' or '0') end
    return r
  end) .. '0000'):gsub('%d%d%d?%d?%d?%d?', function(x)
    if #x < 6 then return '' end
    local c = 0
    for i2 = 1, 6 do c = c + (x:sub(i2,i2) == '1' and 2^(6-i2) or 0) end
    return b:sub(c+1, c+1)
  end) .. ({ '', '==', '=' })[#data % 3 + 1])
end

local function sha1(msg)
  -- SHA1 pur Lua (suffisant pour handshake WebSocket)
  local function lrot(x, n)
    return ((x << n) | (x >> (32 - n))) & 0xffffffff
  end

  local bytes = { msg:byte(1, #msg) }
  local bit_len = #bytes * 8

  bytes[#bytes+1] = 0x80
  while (#bytes % 64) ~= 56 do
    bytes[#bytes+1] = 0
  end

  for shift = 56, 0, -8 do
    bytes[#bytes+1] = (bit_len >> shift) & 0xff
  end

  local h0, h1, h2, h3, h4 = 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0

  local w = {}
  for chunk = 1, #bytes, 64 do
    for j = 0, 15 do
      local i0 = chunk + j * 4
      w[j] = ((bytes[i0] << 24) | (bytes[i0+1] << 16) | (bytes[i0+2] << 8) | bytes[i0+3]) & 0xffffffff
    end
    for j = 16, 79 do
      w[j] = lrot((w[j-3] ~ w[j-8] ~ w[j-14] ~ w[j-16]) & 0xffffffff, 1)
    end

    local a, b, c, d, e = h0, h1, h2, h3, h4
    for j = 0, 79 do
      local f, k
      if j <= 19 then
        f = (b & c) | ((~b) & d)
        k = 0x5A827999
      elseif j <= 39 then
        f = b ~ c ~ d
        k = 0x6ED9EBA1
      elseif j <= 59 then
        f = (b & c) | (b & d) | (c & d)
        k = 0x8F1BBCDC
      else
        f = b ~ c ~ d
        k = 0xCA62C1D6
      end
      local temp = (lrot(a, 5) + f + e + k + w[j]) & 0xffffffff
      e, d, c, b, a = d, c, lrot(b, 30), a, temp
    end

    h0 = (h0 + a) & 0xffffffff
    h1 = (h1 + b) & 0xffffffff
    h2 = (h2 + c) & 0xffffffff
    h3 = (h3 + d) & 0xffffffff
    h4 = (h4 + e) & 0xffffffff
  end

  local function word_to_bytes(x)
    return string.char((x >> 24) & 0xff, (x >> 16) & 0xff, (x >> 8) & 0xff, x & 0xff)
  end

  return word_to_bytes(h0) .. word_to_bytes(h1) .. word_to_bytes(h2) .. word_to_bytes(h3) .. word_to_bytes(h4)
end

local function ws_accept_key(sec_key)
  local GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
  return base64_encode(sha1(sec_key .. GUID))
end

-- =========================
-- HTTP utils
-- =========================
local function trim(s)
  return (s:gsub("^%s+", ""):gsub("%s+$", ""))
end

local function parse_query(path)
  local p, q = path:match("^([^?]+)%??(.*)$")
  local query = {}
  if q and q ~= "" then
    for pair in q:gmatch("[^&]+") do
      local k, v = pair:match("^([^=]+)=?(.*)$")
      if k then
        k = k:gsub("%%(%x%x)", function(h) return string.char(tonumber(h, 16)) end)
        v = (v or ""):gsub("%%(%x%x)", function(h) return string.char(tonumber(h, 16)) end)
        query[k] = v
      end
    end
  end
  return p or path, query
end

local function read_http_request(client)
  client:settimeout(10)
  local line, err = client:receive("*l")
  if not line then return nil, err end
  local method, path, httpver = line:match("^(%S+)%s+(%S+)%s+(HTTP/%d%.%d)$")
  if not method then return nil, "bad request line" end

  local headers = {}
  while true do
    local hline, herr = client:receive("*l")
    if not hline then return nil, herr end
    if hline == "" then break end
    local k, v = hline:match("^([^:]+):%s*(.*)$")
    if k then
      headers[k:lower()] = trim(v)
    end
  end

  local body = ""
  local cl = tonumber(headers["content-length"] or "0")
  if cl and cl > 0 then
    local chunk, berr = client:receive(cl)
    if not chunk then return nil, berr end
    body = chunk
  end

  return { method = method, path = path, httpver = httpver, headers = headers, body = body }
end

local function http_response(status, headers, body)
  local reason = {
    [200] = "OK",
    [201] = "Created",
    [204] = "No Content",
    [401] = "Unauthorized",
    [403] = "Forbidden",
    [400] = "Bad Request",
    [404] = "Not Found",
    [405] = "Method Not Allowed",
    [500] = "Internal Server Error",
  }

  local lines = {}
  lines[#lines+1] = string.format("HTTP/1.1 %d %s", status, reason[status] or "")
  headers = headers or {}
  headers["Content-Length"] = tostring(#(body or ""))
  headers["Connection"] = headers["Connection"] or "close"
  for k, v in pairs(headers) do
    lines[#lines+1] = string.format("%s: %s", k, v)
  end
  lines[#lines+1] = ""
  lines[#lines+1] = body or ""
  return table.concat(lines, "\r\n")
end

-- =========================
-- Auth helpers
-- =========================
local function secure_equals(a, b)
  if type(a) ~= "string" or type(b) ~= "string" then return false end
  if #a ~= #b then return false end
  local diff = 0
  for i2 = 1, #a do
    diff = diff | (a:byte(i2) ~ b:byte(i2))
  end
  return diff == 0
end

local function csv_to_set(csv)
  local set = {}
  if type(csv) ~= "string" or csv == "" then return set end
  for item in csv:gmatch("[^,]+") do
    local v = trim(item)
    if v ~= "" then set[v] = true end
  end
  return set
end

local AUTH_SKIP_SET = csv_to_set(AUTH_SKIP_PATHS)

local function extract_bearer(headers)
  if type(headers) ~= "table" then return nil end
  local a = headers["authorization"]
  if type(a) ~= "string" or a == "" then return nil end
  local scheme, token = a:match("^(%S+)%s+(.+)$")
  if not scheme or not token then return nil end
  scheme = scheme:lower()
  if scheme ~= "bearer" and scheme ~= "token" then return nil end
  return trim(token)
end

local function extract_api_key(headers)
  if type(headers) ~= "table" then return nil end
  local k = headers["x-api-key"]
  if type(k) ~= "string" or k == "" then return nil end
  return trim(k)
end

local function auth_enabled()
  return AUTH_MODE ~= "off"
end

local function auth_mode_valid()
  return AUTH_MODE == "off" or AUTH_MODE == "token" or AUTH_MODE == "api_key" or AUTH_MODE == "either"
end

local function auth_check(method, clean_path, headers, query)
  if not auth_mode_valid() then
    return false, 500, { ["Content-Type"] = "application/json" }, json_encode({ ok = false, error = "auth mode invalide", detail = AUTH_MODE })
  end

  if not auth_enabled() then
    return true
  end

  if AUTH_SKIP_SET[clean_path] then
    return true
  end

  local want_token = AUTH_MODE == "token" or AUTH_MODE == "either"
  local want_key = AUTH_MODE == "api_key" or AUTH_MODE == "either"

  local provided_token = extract_bearer(headers) or (type(query) == "table" and (query.token or query.bearer) or nil)
  local provided_key = extract_api_key(headers) or (type(query) == "table" and (query.api_key or query.apikey or query.key) or nil)

  local token_ok = (want_token and type(AUTH_TOKEN) == "string" and AUTH_TOKEN ~= "" and type(provided_token) == "string" and secure_equals(provided_token, AUTH_TOKEN))
  local key_ok = (want_key and type(API_KEY) == "string" and API_KEY ~= "" and type(provided_key) == "string" and secure_equals(provided_key, API_KEY))

  if token_ok or key_ok then
    return true
  end

  local hdrs = {
    ["Content-Type"] = "application/json",
  }
  if want_token then
    hdrs["WWW-Authenticate"] = 'Bearer realm="mimir"'
  end
  return false, 401, hdrs, json_encode({ ok = false, error = "unauthorized" })
end

-- =========================
-- Framework dispatcher
-- =========================
local function mimir_available()
  return type(_G.Mimir) == "table" and type(_G.Mimir.Model) == "table"
end

local function framework_info()
  local available = mimir_available()
  local caps = {}
  if available then
    caps.model = type(Mimir.Model) == "table"
    caps.model_create = type(Mimir.Model.create) == "function"
    caps.model_build = type(Mimir.Model.build) == "function"
    caps.model_allocate_params = type(Mimir.Model.allocate_params) == "function"
    caps.model_infer = type(Mimir.Model.infer) == "function"
    caps.model_forward = type(Mimir.Model.forward) == "function"
    caps.model_init_weights = type(Mimir.Model.init_weights) == "function"
    caps.model_train = type(Mimir.Model.train) == "function"
    caps.serialization = type(Mimir.Serialization) == "table"
    caps.serialization_save = type(Mimir.Serialization) == "table" and type(Mimir.Serialization.save) == "function"
    caps.serialization_load = type(Mimir.Serialization) == "table" and type(Mimir.Serialization.load) == "function"
    caps.allocator = type(Mimir.Allocator) == "table"
    caps.allocator_configure = type(Mimir.Allocator) == "table" and type(Mimir.Allocator.configure) == "function"
    caps.allocator_stats = type(Mimir.Allocator) == "table" and type(Mimir.Allocator.get_stats) == "function"

    caps.architectures = type(Mimir.Architectures) == "table"
    caps.arch_available = type(Mimir.Architectures) == "table" and type(Mimir.Architectures.available) == "function"
    caps.arch_default_config = type(Mimir.Architectures) == "table" and type(Mimir.Architectures.default_config) == "function"
  end
  return {
    mimir_available = available,
    server_version = SERVER_VERSION,
    capabilities = caps,
  }
end

local function safe_call(fn, ...)
  return pcall(fn, ...)
end

local request_counter = 0
local function next_request_id()
  request_counter = request_counter + 1
  local ms = math.floor(now_seconds() * 1000)
  return string.format("%x-%x", ms, request_counter)
end

local function enrich_json_body(body_json, req_meta)
  local ok, decoded = pcall(json_decode, body_json or "")
  local payload
  if ok and type(decoded) == "table" then
    payload = decoded
  else
    payload = { ok = false, error = "Réponse interne non-JSON", raw = tostring(body_json) }
  end

  payload._meta = payload._meta or {}
  for k, v in pairs(req_meta or {}) do
    payload._meta[k] = v
  end

  return json_encode(payload)
end

local function api_handle(method, path, headers, body)
  local clean_path, query = parse_query(path)
  local content_type = (headers["content-type"] or ""):lower()

  -- CORS preflight: ne doit pas être bloqué par l'auth, sinon le navigateur échoue avant la requête réelle.
  if method == "OPTIONS" then
    return 200, { ["Content-Type"] = "application/json" }, json_encode({ ok = true })
  end

  do
    local ok_auth, status, hdrs, resp = auth_check(method, clean_path, headers, query)
    if not ok_auth then
      return status, hdrs, resp
    end
  end

  local json_body = nil
  if body and body ~= "" and content_type:find("application/json", 1, true) then
    local ok, decoded = pcall(json_decode, body)
    if ok then json_body = decoded else
      return 400, { ["Content-Type"] = "application/json" }, json_encode({ ok = false, error = "JSON invalide", detail = decoded })
    end
  end

  -- util: renvoie JSON standard
  local function ok_json(payload, status)
    return status or 200, { ["Content-Type"] = "application/json" }, json_encode(payload)
  end

  -- ====== Routes ======
  if method == "GET" and clean_path == "/help" then
    return ok_json({
      ok = true,
      server = { name = SERVER_NAME, version = SERVER_VERSION, host = HOST, port = PORT, lua = _VERSION },
      security = {
        tls = TLS_ENABLED,
        auth = {
          mode = AUTH_MODE,
          enabled = auth_enabled(),
          skip_paths = AUTH_SKIP_PATHS,
          accepted = {
            bearer = (AUTH_MODE == "token" or AUTH_MODE == "either"),
            api_key = (AUTH_MODE == "api_key" or AUTH_MODE == "either"),
            query_for_ws = true,
          },
        },
      },
      routes = {
        { method = "GET", path = "/health", desc = "Santé + disponibilité de Mimir" },
        { method = "GET", path = "/help", desc = "Liste des routes du serveur" },
        { method = "GET", path = "/architectures", desc = "Liste des architectures disponibles (registry)" },
        { method = "GET", path = "/architectures/default_config", desc = "Config par défaut", query = { model_type = "transformer" } },
        { method = "GET", path = "/hardware", desc = "Capacités hardware détectées (AVX2/FMA/F16C/BMI2)" },
        { method = "PUT", path = "/hardware", desc = "Activer/désactiver l'accélération matérielle (API actuelle)", body = { enable = true } },
        { method = "GET", path = "/allocator", desc = "Stats allocator (DynamicTensorAllocator)" },
        { method = "PATCH", path = "/allocator", desc = "Configurer allocator", body = { max_ram_gb = 10.0, enable_compression = true } },
        { method = "DELETE", path = "/allocator", desc = "Ack + GC (pas de reset allocator exposé)" },
        { method = "POST", path = "/model/create", desc = "Créer un modèle", body = { model_type = "transformer", config = { vocab_size = 32000 } } },
        { method = "POST", path = "/model/build", desc = "Build le modèle courant" },
        { method = "POST", path = "/model/init_weights", desc = "Init poids", body = { init = "xavier|he|normal|uniform|zeros", seed = 123 } },
        { method = "GET", path = "/model/params", desc = "Nombre total de paramètres" },
        { method = "POST", path = "/model/infer", desc = "Inférence", body = { input = "Hello" } },
        { method = "DELETE", path = "/model", desc = "Ack + GC (pas de reset modèle exposé)" },
        { method = "POST", path = "/serialization/load", desc = "Charger un checkpoint", body = { path = "checkpoint/", format = "raw_folder", options = { verify_checksums = true } } },
        { method = "POST", path = "/serialization/save", desc = "Sauvegarder un checkpoint", body = { path = "model.safetensors", format = "safetensors", options = { include_git_info = true } } },
      },
      websocket = {
        path = "/ws",
        desc = "WebSocket: envoyer {method, path, body} en JSON et recevoir {status, response, _meta}. Auth possible via header Authorization/X-API-Key ou query ?token=.../?api_key=...",
        example_message = { method = "GET", path = "/health" },
      }
    })
  end

  if method == "GET" and clean_path == "/architectures" then
    if not mimir_available() or type(Mimir.Architectures) ~= "table" or type(Mimir.Architectures.available) ~= "function" then
      return ok_json({ ok = false, error = "Mimir.Architectures.available indisponible" }, 500)
    end
    local list = Mimir.Architectures.available()
    return ok_json({ ok = true, architectures = list })
  end

  if method == "GET" and clean_path == "/architectures/default_config" then
    if not mimir_available() or type(Mimir.Architectures) ~= "table" or type(Mimir.Architectures.default_config) ~= "function" then
      return ok_json({ ok = false, error = "Mimir.Architectures.default_config indisponible" }, 500)
    end
    local model_type = (query.model_type)
    if not model_type or model_type == "" then
      return ok_json({ ok = false, error = "model_type requis" }, 400)
    end
    local cfg = Mimir.Architectures.default_config(model_type)
    return ok_json({ ok = type(cfg) == "table", config = cfg })
  end

  if method == "GET" and clean_path == "/health" then
    return ok_json({ ok = true, mimir = mimir_available(), time = os.date("!%Y-%m-%dT%H:%M:%SZ") })
  end

  if method == "GET" and clean_path == "/hardware" then
    if not mimir_available() or type(Mimir.Model.hardware_caps) ~= "function" then
      return ok_json({ ok = false, error = "Mimir.Model.hardware_caps indisponible" }, 500)
    end
    local caps = Mimir.Model.hardware_caps()
    return ok_json({ ok = true, caps = caps })
  end

  if method == "PUT" and clean_path == "/hardware" then
    if not mimir_available() or type(Mimir.Model.set_hardware) ~= "function" then
      return ok_json({ ok = false, error = "Mimir.Model.set_hardware indisponible" }, 500)
    end
    local function parse_bool(v)
      if type(v) == "boolean" then return v end
      if type(v) == "number" then return v ~= 0 end
      if type(v) == "string" then
        local s = v:lower()
        if s == "1" or s == "true" or s == "yes" or s == "on" then return true end
        if s == "0" or s == "false" or s == "no" or s == "off" then return false end
      end
      return nil
    end

    -- API réelle (src/LuaScripting.cpp): set_hardware(enable: bool)
    -- Compat REST: accepte aussi backend="cpu"|"auto" (map en bool), mais ne choisit pas OpenCL/Vulkan spécifiquement.
    local enable = (type(json_body) == "table") and parse_bool(json_body.enable) or nil
    if enable == nil then enable = parse_bool(query.enable) end

    if enable == nil then
      local backend = (type(json_body) == "table") and json_body.backend or query.backend
      if type(backend) == "string" and backend ~= "" then
        local b = backend:lower()
        if b == "cpu" then enable = false
        elseif b == "auto" then enable = true
        else
          return ok_json({ ok = false, error = "backend non supporté par l'API actuelle", detail = "Utilisez {enable=true|false} (set_hardware(bool))" }, 400)
        end
      end
    end

    if enable == nil then
      return ok_json({ ok = false, error = "enable requis", detail = "body: {enable:true|false} ou query ?enable=true" }, 400)
    end

    local ok_call, ok_or_err, err2 = pcall(Mimir.Model.set_hardware, enable)
    if not ok_call then
      return ok_json({ ok = false, error = tostring(ok_or_err) }, 500)
    end
    return ok_json({ ok = ok_or_err ~= false, enabled = enable, error = err2 })
  end

  if method == "PATCH" and clean_path == "/allocator" then
    if not mimir_available() or type(Mimir.Allocator) ~= "table" or type(Mimir.Allocator.configure) ~= "function" then
      return ok_json({ ok = false, error = "Mimir.Allocator.configure indisponible" }, 500)
    end
    if type(json_body) ~= "table" then
      return ok_json({ ok = false, error = "body JSON requis" }, 400)
    end
    local ok, err = Mimir.Allocator.configure(json_body)
    return ok_json({ ok = ok, error = err })
  end

  if method == "GET" and clean_path == "/allocator" then
    if not mimir_available() or type(Mimir.Allocator) ~= "table" or type(Mimir.Allocator.get_stats) ~= "function" then
      return ok_json({ ok = false, error = "Mimir.Allocator.get_stats indisponible" }, 500)
    end
    local stats = Mimir.Allocator.get_stats()
    return ok_json({ ok = true, stats = stats })
  end

  if method == "POST" and clean_path == "/model/create" then
    if not mimir_available() then
      return ok_json({ ok = false, error = "Mimir indisponible (lancer via ./bin/mimir ?)" }, 500)
    end
    if type(json_body) ~= "table" then
      return ok_json({ ok = false, error = "body JSON requis" }, 400)
    end
    local model_type = json_body.model_type
    local config = json_body.config
    if type(model_type) ~= "string" or model_type == "" then
      return ok_json({ ok = false, error = "model_type requis" }, 400)
    end

    -- Si le client n'envoie pas de config, on tente une config par défaut via le registry.
    if config == nil and type(Mimir.Architectures) == "table" and type(Mimir.Architectures.default_config) == "function" then
      local cfg = Mimir.Architectures.default_config(model_type)
      if type(cfg) == "table" then
        config = cfg
      end
    end

    local ok, err = Mimir.Model.create(model_type, config)
    return ok_json({ ok = ok, error = err })
  end

  if method == "POST" and clean_path == "/model/build" then
    if not mimir_available() then
      return ok_json({ ok = false, error = "Mimir indisponible" }, 500)
    end
    -- API actuelle: build() -> (ok, params_or_err)
    local ok, params_or_err = Mimir.Model.build()
    if ok then
      return ok_json({ ok = true, params = params_or_err })
    end
    return ok_json({ ok = false, error = params_or_err }, 500)
  end

  if method == "POST" and clean_path == "/model/init_weights" then
    if not mimir_available() then
      return ok_json({ ok = false, error = "Mimir indisponible" }, 500)
    end
    if type(json_body) ~= "table" then
      return ok_json({ ok = false, error = "body JSON requis" }, 400)
    end
    local init = json_body.init
    local seed = json_body.seed
    local ok, err = Mimir.Model.init_weights(init, seed)
    return ok_json({ ok = ok, error = err })
  end

  if method == "GET" and clean_path == "/model/params" then
    if not mimir_available() then
      return ok_json({ ok = false, error = "Mimir indisponible" }, 500)
    end
    local params = Mimir.Model.total_params()
    return ok_json({ ok = true, params = params })
  end

  if method == "POST" and clean_path == "/model/infer" then
    if not mimir_available() then
      return ok_json({ ok = false, error = "Mimir indisponible" }, 500)
    end
    if type(json_body) ~= "table" then
      return ok_json({ ok = false, error = "body JSON requis" }, 400)
    end
    local input = json_body.input
    if input == nil then
      return ok_json({ ok = false, error = "input requis" }, 400)
    end
    local out = Mimir.Model.infer(input)
    return ok_json({ ok = out ~= nil, output = out })
  end

  if method == "POST" and clean_path == "/serialization/load" then
    if not mimir_available() or type(Mimir.Serialization) ~= "table" or type(Mimir.Serialization.load) ~= "function" then
      return ok_json({ ok = false, error = "Mimir.Serialization.load indisponible" }, 500)
    end
    if type(json_body) ~= "table" then
      return ok_json({ ok = false, error = "body JSON requis" }, 400)
    end
    local path0, format, options = json_body.path, json_body.format, json_body.options
    if type(path0) ~= "string" or path0 == "" then
      return ok_json({ ok = false, error = "path requis" }, 400)
    end
    local ok, err = Mimir.Serialization.load(path0, format, options)
    return ok_json({ ok = ok, error = err })
  end

  if method == "POST" and clean_path == "/serialization/save" then
    if not mimir_available() or type(Mimir.Serialization) ~= "table" or type(Mimir.Serialization.save) ~= "function" then
      return ok_json({ ok = false, error = "Mimir.Serialization.save indisponible" }, 500)
    end
    if type(json_body) ~= "table" then
      return ok_json({ ok = false, error = "body JSON requis" }, 400)
    end
    local path0, format, options = json_body.path, json_body.format, json_body.options
    if type(path0) ~= "string" or path0 == "" then
      return ok_json({ ok = false, error = "path requis" }, 400)
    end
    local ok, err = Mimir.Serialization.save(path0, format, options)
    return ok_json({ ok = ok, error = err })
  end

  -- DELETE explicites (API complète)
  if method == "DELETE" and clean_path == "/model" then
    -- Mímir ne fournit pas (encore) de reset de modèle dans l'API publique.
    -- On renvoie un ack + GC pour libérer côté Lua.
    collectgarbage("collect")
    return ok_json({ ok = true, note = "Aucun reset de modèle exposé; GC Lua exécuté." })
  end

  if method == "DELETE" and clean_path == "/allocator" then
    collectgarbage("collect")
    return ok_json({ ok = true, note = "Allocator non réinitialisable via API; GC Lua exécuté." })
  end

  -- Méthode reconnue mais route inconnue
  return 404, { ["Content-Type"] = "application/json" }, json_encode({ ok = false, error = "route inconnue", method = method, path = clean_path })
end

-- =========================
-- WebSocket framing
-- =========================
local function recv_exact(sock, n)
  local chunks = {}
  local got = 0
  while got < n do
    local part, err, partial = sock:receive(n - got)
    part = part or partial
    if part and #part > 0 then
      chunks[#chunks+1] = part
      got = got + #part
    else
      return nil, err or "receive failed"
    end
  end
  return table.concat(chunks)
end

local function ws_send_text(sock, text)
  local payload = text or ""
  local len = #payload
  local b1 = string.char(0x81) -- FIN + text
  local header
  if len < 126 then
    header = b1 .. string.char(len)
  elseif len < 65536 then
    header = b1 .. string.char(126) .. string.char((len >> 8) & 0xff) .. string.char(len & 0xff)
  else
    error("Payload WS trop gros")
  end
  sock:send(header .. payload)
end

local function ws_read_frame(sock)
  local hdr, err = recv_exact(sock, 2)
  if not hdr then return nil, err end
  local b1, b2 = hdr:byte(1,2)
  local fin = (b1 & 0x80) ~= 0
  local opcode = b1 & 0x0f
  local masked = (b2 & 0x80) ~= 0
  local len = b2 & 0x7f
  if len == 126 then
    local ext, err2 = recv_exact(sock, 2)
    if not ext then return nil, err2 end
    len = (ext:byte(1) << 8) | ext:byte(2)
  elseif len == 127 then
    -- on limite volontairement: pas de frames > 65535 dans ce serveur minimal
    return nil, "Frame WS trop grande (len=127 non supporté)"
  end

  local mask
  if masked then
    local m, err2 = recv_exact(sock, 4)
    if not m then return nil, err2 end
    mask = m
  end

  local payload = ""
  if len > 0 then
    local p, err2 = recv_exact(sock, len)
    if not p then return nil, err2 end
    payload = p
  end

  if masked and payload and mask then
    local m1, m2, m3, m4 = mask:byte(1,4)
    local out = {}
    for i2 = 1, #payload do
      local pb = payload:byte(i2)
      local mb = ({m1,m2,m3,m4})[((i2-1) % 4) + 1]
      out[i2] = string.char(pb ~ mb)
    end
    payload = table.concat(out)
  end

  return { fin = fin, opcode = opcode, payload = payload }
end

-- =========================
-- Serveur
-- =========================
local function handle_websocket(client, req)
  local key = req.headers["sec-websocket-key"]
  if not key then
    client:send(http_response(400, { ["Content-Type"] = "text/plain" }, "Missing Sec-WebSocket-Key"))
    client:close()
    return
  end

  local clean_path, query = parse_query(req.path or "/ws")
  do
    local ok_auth, status, hdrs, resp = auth_check("GET", clean_path, req.headers or {}, query)
    if not ok_auth then
      client:send(http_response(status, hdrs, resp))
      client:close()
      return
    end
  end

  local accept = ws_accept_key(key)
  local resp = table.concat({
    "HTTP/1.1 101 Switching Protocols",
    "Upgrade: websocket",
    "Connection: Upgrade",
    "Sec-WebSocket-Accept: " .. accept,
    "\r\n",
  }, "\r\n")

  client:send(resp)
  client:settimeout(0) -- non-bloquant

  -- Boucle WS: attend des commandes JSON, renvoie une réponse JSON
  while true do
    socket.sleep(0.01)
    local frame, err = ws_read_frame(client)
    if not frame then
      if err == "timeout" then
        -- continuer
      else
        break
      end
    else
      if frame.opcode == 0x8 then -- close
        break
      elseif frame.opcode == 0x9 then -- ping
        -- pong minimal
        client:send(string.char(0x8A, 0x00))
      elseif frame.opcode == 0x1 then -- text
        local ok, msg = pcall(json_decode, frame.payload)
        if not ok or type(msg) ~= "table" then
          ws_send_text(client, json_encode({
            ok = false,
            error = "message JSON invalide",
            _meta = {
              server = { name = SERVER_NAME, version = SERVER_VERSION, host = HOST, port = PORT, lua = _VERSION },
              framework = framework_info(),
              received_at_utc = iso_utc(),
            }
          }))
        else
          local t0 = now_seconds()
          local request_id = next_request_id()
          local m = tostring(msg.method or "GET")
          local p = tostring(msg.path or "/health")
          local b = msg.body ~= nil and json_encode(msg.body) or ""
          local ct_headers = { ["content-type"] = "application/json" }
          local status, h, resp_body = api_handle(m, p, ct_headers, b)
          local t1 = now_seconds()
          local latency_ms = math.floor((t1 - t0) * 1000 + 0.5)
          local decoded_ok, decoded = pcall(json_decode, resp_body)
          if not decoded_ok then decoded = { ok = false, error = "decode JSON interne échoué", raw = tostring(resp_body) } end

          ws_send_text(client, json_encode({
            ok = true,
            status = status,
            response = decoded,
            _meta = {
              request_id = request_id,
              method = m,
              path = p,
              received_at_utc = iso_utc(),
              latency_ms = latency_ms,
              server = { name = SERVER_NAME, version = SERVER_VERSION, host = HOST, port = PORT, lua = _VERSION },
              framework = framework_info(),
            }
          }))
        end
      end
    end
  end

  client:close()
end

local function serve_forever()
  if TLS_ENABLED then
    if not ssl then
      error("TLS demandé (--tls/--https) mais LuaSec n'est pas disponible (module 'ssl').")
    end
    if TLS_CERT_FILE == "" or TLS_KEY_FILE == "" then
      error("TLS activé mais --tls-cert et --tls-key sont requis.")
    end
  end

  local server = assert(socket.bind(HOST, PORT))
  server:settimeout(0)
  local scheme = TLS_ENABLED and "https" or "http"
  print(string.format("[api] listening on %s://%s:%d (ws: /ws) name=%s version=%s lua=%s", scheme, HOST, PORT, SERVER_NAME, SERVER_VERSION, _VERSION))

  while true do
    local client = server:accept()
    if client then
      local t0 = now_seconds()
      local request_id = next_request_id()

      if TLS_ENABLED then
        client:settimeout(10)
        local wrapped, werr = ssl.wrap(client, {
          mode = "server",
          protocol = "tlsv1_2",
          key = TLS_KEY_FILE,
          certificate = TLS_CERT_FILE,
          options = { "all" },
        })
        if not wrapped then
          client:close()
          goto continue
        end
        local ok_hs, herr = wrapped:dohandshake()
        if not ok_hs then
          wrapped:close()
          goto continue
        end
        client = wrapped
      end

      local req, err = read_http_request(client)
      if not req then
        local t1 = now_seconds()
        local latency_ms = math.floor((t1 - t0) * 1000 + 0.5)
        local body = json_encode({
          ok = false,
          error = "Bad request",
          detail = tostring(err),
          _meta = {
            request_id = request_id,
            received_at_utc = iso_utc(),
            latency_ms = latency_ms,
            server = { name = SERVER_NAME, version = SERVER_VERSION, host = HOST, port = PORT, lua = _VERSION },
            framework = framework_info(),
          }
        })
        client:send(http_response(400, { ["Content-Type"] = "application/json", ["X-Response-Time-Ms"] = tostring(latency_ms) }, body))
        client:close()
      else
        local upgrade = (req.headers["upgrade"] or ""):lower()
        local connection = (req.headers["connection"] or ""):lower()
        if req.path:match("^/ws") and upgrade == "websocket" and connection:find("upgrade", 1, true) then
          handle_websocket(client, req)
        else
          local status, hdrs, resp_body = api_handle(req.method, req.path, req.headers, req.body)
          local t1 = now_seconds()
          local latency_ms = math.floor((t1 - t0) * 1000 + 0.5)
          hdrs = hdrs or {}
          hdrs["X-Response-Time-Ms"] = tostring(latency_ms)

          -- En-têtes de durcissement côté API
          hdrs["X-Content-Type-Options"] = hdrs["X-Content-Type-Options"] or "nosniff"
          hdrs["X-Frame-Options"] = hdrs["X-Frame-Options"] or "DENY"
          hdrs["Referrer-Policy"] = hdrs["Referrer-Policy"] or "no-referrer"
          hdrs["Cache-Control"] = hdrs["Cache-Control"] or "no-store"
          if TLS_ENABLED then
            hdrs["Strict-Transport-Security"] = hdrs["Strict-Transport-Security"] or "max-age=31536000"
          end

          -- CORS (utile quand la page web est sur un autre origin, y compris file://)
          if CORS_ENABLED then
            local origin = req.headers["origin"]
            local allow_origin = CORS_ORIGIN
            if allow_origin == "reflect" then
              if type(origin) == "string" and origin ~= "" then
                allow_origin = origin
                -- Indique au cache/proxy que la réponse dépend de Origin
                hdrs["Vary"] = hdrs["Vary"] or "Origin"
              else
                allow_origin = "*"
              end
            end

            hdrs["Access-Control-Allow-Origin"] = allow_origin
            hdrs["Access-Control-Allow-Methods"] = CORS_ALLOW_METHODS
            hdrs["Access-Control-Allow-Headers"] = CORS_ALLOW_HEADERS
            hdrs["Access-Control-Expose-Headers"] = CORS_EXPOSE_HEADERS
            hdrs["Access-Control-Max-Age"] = CORS_MAX_AGE
          end

          local meta = {
            request_id = request_id,
            method = req.method,
            path = req.path,
            received_at_utc = iso_utc(),
            latency_ms = latency_ms,
            server = { name = SERVER_NAME, version = SERVER_VERSION, host = HOST, port = PORT, lua = _VERSION },
            framework = framework_info(),
            client = {
              user_agent = req.headers["user-agent"],
              content_type = req.headers["content-type"],
            },
          }

          local enriched = enrich_json_body(resp_body, meta)
          client:send(http_response(status, hdrs, enriched))
          client:close()
        end
      end
    else
      socket.sleep(0.01)
    end

    ::continue::
  end
end

serve_forever()
