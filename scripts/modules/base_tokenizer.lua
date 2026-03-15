-- Base tokenizer helper
-- Objectif: partager une base commune de vocab entre tous les modèles.
--
-- Convention:
--   - Chemin par défaut: $MIMIR_BASE_TOKENIZER ou checkpoint/base_tokenizer/tokenizer.json
--   - Pour forcer la présence: $MIMIR_REQUIRE_BASE_TOKENIZER=1
--
-- Usage (dans un script):
--   local BaseTok = dofile("scripts/modules/base_tokenizer.lua")
--   BaseTok.load_base({ max_vocab = 50000, require = true })
--   cfg.vocab_size = BaseTok.vocab_size()  -- important pour Embedding

local BaseTok = {}

local function env_str(k, d)
  local v = os.getenv(k)
  if v == nil or v == "" then return d end
  return v
end

local function env_bool(k, d)
  local v = os.getenv(k)
  if v == nil or v == "" then return d end
  v = tostring(v):lower()
  if v == "1" or v == "true" or v == "yes" or v == "on" then return true end
  if v == "0" or v == "false" or v == "no" or v == "off" then return false end
  return d
end

local function shell_quote(s)
  if s == nil then return "''" end
  s = tostring(s)
  return "'" .. s:gsub("'", "'\\''") .. "'"
end

local function file_exists(path)
  if not path or #tostring(path) == 0 then return false end
  local cmd = "test -f " .. shell_quote(path) .. " >/dev/null 2>&1"
  local ok, why, code = os.execute(cmd)
  if type(ok) == "number" then return ok == 0 end
  if type(ok) == "boolean" then return ok end
  return (why == "exit" and code == 0)
end

local function ensure_parent_dir(filepath)
  local dir = tostring(filepath):match("^(.*)/[^/]*$")
  if dir and #dir > 0 then
    os.execute("mkdir -p " .. shell_quote(dir) .. " >/dev/null 2>&1")
  end
end

function BaseTok.default_path()
  return env_str("MIMIR_BASE_TOKENIZER", "checkpoint/base_tokenizer/tokenizer.json")
end

function BaseTok.load_base(opts)
  opts = opts or {}
  local path = opts.path or BaseTok.default_path()
  local require_base = opts.require
  if require_base == nil then
    require_base = env_bool("MIMIR_REQUIRE_BASE_TOKENIZER", false)
  end

  if not (Mimir and Mimir.Tokenizer) then
    return false, "Tokenizer API indisponible"
  end

  if file_exists(path) then
    local ok, err = Mimir.Tokenizer.load(path)
    if ok == false then
      if require_base then
        return false, "Tokenizer.load(base) a échoué: " .. tostring(err)
      end
      log("⚠️  Base tokenizer: load a échoué: " .. tostring(err))
      return true
    end

    -- Log de confirmation
    log("✓ Base tokenizer chargé: " .. tostring(path) .. " (vocab=" .. tostring(Mimir.Tokenizer.vocab_size()) .. ")")
    return true
  end

  if require_base then
    return false, "Base tokenizer introuvable: " .. tostring(path)
  end

  -- Fallback: créer un tokenizer vide (utile pour image-only)
  local max_vocab = tonumber(opts.max_vocab or env_str("MIMIR_BASE_TOKENIZER_MAX_VOCAB", "50000")) or 50000
  if Mimir.Tokenizer.create then
    local okc, errc = Mimir.Tokenizer.create(max_vocab)
    if okc == false then
      return false, "Tokenizer.create fallback failed: " .. tostring(errc)
    end
    if Mimir.Tokenizer.add_token then
      -- Assurer un minimum de tokens spéciaux (compat)
      pcall(Mimir.Tokenizer.add_token, "[PAD]")
      pcall(Mimir.Tokenizer.add_token, "[UNK]")
      pcall(Mimir.Tokenizer.add_token, "[BOS]")
      pcall(Mimir.Tokenizer.add_token, "[EOS]")
    end
  end

  log("⚠️  Base tokenizer manquant, fallback tokenizer vierge (vocab=" .. tostring(Mimir.Tokenizer.vocab_size()) .. ")")
  return true
end

function BaseTok.vocab_size()
  if not (Mimir and Mimir.Tokenizer and Mimir.Tokenizer.vocab_size) then return 0 end
  return tonumber(Mimir.Tokenizer.vocab_size()) or 0
end

function BaseTok.save_current_as_base(path)
  path = path or BaseTok.default_path()
  if not (Mimir and Mimir.Tokenizer and Mimir.Tokenizer.save) then
    return false, "Tokenizer API indisponible"
  end
  ensure_parent_dir(path)
  local ok, err = Mimir.Tokenizer.save(path)
  if ok == false then
    return false, tostring(err)
  end
  return true
end

return BaseTok
