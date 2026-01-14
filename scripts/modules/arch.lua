-- ==========================================================================
-- Mímir Framework - Script-side Architecture Helpers
--
-- Objectif:
-- - Ne touche PAS à l'API/bindings C++.
-- - Rend les scripts robustes face aux anciennes configs (legacy keys).
-- - Centralise l'accès au registre: Mimir.Architectures.default_config()
--
-- Usage:
--   local Arch = require("arch")
--   local cfg, warn = Arch.build_config("transformer", {embed_dim=256, max_seq_len=128})
--   assert(Mimir.Model.create("transformer", cfg))
-- ==========================================================================

local Arch = {}

local function has_mimir()
    return type(_G.Mimir) == "table" and type(Mimir.Model) == "table"
end

local function try_default_config(model_type)
    if not has_mimir() then
        return nil, "Mimir indisponible (lancer via ./bin/mimir --lua ...)"
    end
    if type(Mimir.Architectures) ~= "table" or type(Mimir.Architectures.default_config) ~= "function" then
        return nil, "Mimir.Architectures.default_config() indisponible"
    end

    local cfg, err = Mimir.Architectures.default_config(model_type)
    if type(cfg) ~= "table" then
        return nil, err or ("Aucune config par défaut pour: " .. tostring(model_type))
    end
    return cfg, nil
end

local function shallow_copy(tbl)
    local out = {}
    if type(tbl) ~= "table" then
        return out
    end
    for k, v in pairs(tbl) do
        out[k] = v
    end
    return out
end

local function merge_into(dst, src)
    if type(dst) ~= "table" or type(src) ~= "table" then
        return dst
    end
    for k, v in pairs(src) do
        dst[k] = v
    end
    return dst
end

local function infer_square_image_size(input_dim)
    if type(input_dim) ~= "number" then
        return nil
    end
    local s = math.floor(math.sqrt(input_dim) + 0.5)
    if s * s == input_dim then
        return s
    end
    return nil
end

local function apply_legacy_keys(model_type, cfg, user)
    user = user or {}

    if model_type == "transformer" or model_type == "encoder" or model_type == "decoder" then
        if user.max_seq_len ~= nil and cfg.seq_len == nil then cfg.seq_len = user.max_seq_len end
        if user.embed_dim ~= nil and cfg.d_model == nil then cfg.d_model = user.embed_dim end
        if user.d_ff ~= nil and cfg.mlp_hidden == nil then cfg.mlp_hidden = user.d_ff end
        if user.model_type == "decoder" and cfg.causal == nil then cfg.causal = true end
        if user.model_type == "encoder" and cfg.causal == nil then cfg.causal = false end
    end

    if model_type == "unet" or model_type == "vae" or model_type == "vit" or model_type == "resnet" or model_type == "mobilenet" or model_type == "diffusion" then
        if user.image_size ~= nil and (cfg.image_w == nil and cfg.image_h == nil) then
            cfg.image_w = user.image_size
            cfg.image_h = user.image_size
        end
        if user.input_channels ~= nil and cfg.image_c == nil then cfg.image_c = user.input_channels end
        if user.num_levels ~= nil and cfg.depth == nil then cfg.depth = user.num_levels end
    end

    if model_type == "vae" then
        if user.input_dim ~= nil and (cfg.image_w == nil and cfg.image_h == nil) then
            local s = infer_square_image_size(user.input_dim)
            if s then
                cfg.image_w = s
                cfg.image_h = s
                if cfg.image_c == nil then cfg.image_c = 1 end
            end
        end
        if user.encoder_hidden ~= nil and cfg.hidden_dim == nil then cfg.hidden_dim = user.encoder_hidden end
        if user.decoder_hidden ~= nil and cfg.hidden_dim == nil then cfg.hidden_dim = user.decoder_hidden end
    end

    if model_type == "vit" then
        if user.embed_dim ~= nil and cfg.d_model == nil then cfg.d_model = user.embed_dim end
        if user.d_ff ~= nil and cfg.mlp_hidden == nil then cfg.mlp_hidden = user.d_ff end
        if user.image_size ~= nil and user.patch_size ~= nil and cfg.num_tokens == nil then
            local patches = math.floor(user.image_size / user.patch_size)
            if patches > 0 then
                cfg.num_tokens = patches * patches
            end
        end
    end

    if model_type == "resnet" or model_type == "mobilenet" then
        if user.num_classes ~= nil and cfg.num_classes == nil then cfg.num_classes = user.num_classes end
        if user.base_channels ~= nil and cfg.base_channels == nil then cfg.base_channels = user.base_channels end
    end

    return cfg
end

---Construit une config compatible pour `model_type`.
---Retourne: cfg, err (err = warning ou nil).
---@param model_type string
---@param user_cfg table|nil
---@return table cfg
---@return string|nil err
function Arch.build_config(model_type, user_cfg)
    user_cfg = user_cfg or {}

    local base, err = try_default_config(model_type)
    local cfg
    if base then
        cfg = shallow_copy(base)
    else
        cfg = {}
    end

    apply_legacy_keys(model_type, cfg, user_cfg)

    -- Merge des clés user en dernier (l'utilisateur override tout)
    merge_into(cfg, user_cfg)

    return cfg, err
end

---Crée directement un modèle via Mimir.Model.create() en passant une config normalisée.
---@param model_type string
---@param user_cfg table|nil
---@return boolean ok
---@return string|nil err
function Arch.create(model_type, user_cfg)
    if not has_mimir() then
        return false, "Mimir indisponible (lancer via ./bin/mimir --lua ...)"
    end

    local cfg, warn = Arch.build_config(model_type, user_cfg)
    if type(cfg) ~= "table" then
        return false, warn
    end

    local ok, err = Mimir.Model.create(model_type, cfg)
    if not ok then
        return false, err
    end

    -- warn est informatif (ex: pas de default_config)
    return true, warn
end

return Arch
