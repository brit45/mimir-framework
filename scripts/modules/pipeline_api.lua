-- ============================================================================
-- Mímir Framework - Pipeline API
-- Système de pipeline pour piloter tous les modèles depuis Lua
-- ============================================================================

local Pipeline = {}
Pipeline.__index = Pipeline

-- ==========================================================================
-- Utilitaires (logger + config helpers)
-- ==========================================================================

local function default_log(...)
    local out = {}
    for i = 1, select("#", ...) do
        out[#out + 1] = tostring(select(i, ...))
    end
    io.stdout:write(table.concat(out, " ") .. "\n")
end

local log = (type(_G) == "table" and type(_G.log) == "function") and _G.log or default_log

local function ensure_mimir()
    if type(_G.Mimir) ~= "table" or type(_G.Mimir.Model) ~= "table" then
        error("Mimir indisponible: lancez via ./bin/mimir --lua ...")
    end
end

local function try_arch_default_config(model_type)
    if type(_G.Mimir) ~= "table" then return nil end
    if type(Mimir.Architectures) ~= "table" then return nil end
    if type(Mimir.Architectures.default_config) ~= "function" then return nil end
    local cfg = Mimir.Architectures.default_config(model_type)
    if type(cfg) ~= "table" then return nil end
    return cfg
end

local function pick_keys(src, keys)
    local out = {}
    if type(src) ~= "table" then return out end
    for _, k in ipairs(keys) do
        local v = src[k]
        if v ~= nil then out[k] = v end
    end
    return out
end

local function build_config(model_type, user_cfg, allowed_keys, fallback_cfg, legacy_mapper)
    user_cfg = user_cfg or {}
    local base = try_arch_default_config(model_type) or (fallback_cfg or {})
    local cfg = pick_keys(base, allowed_keys)
    if type(legacy_mapper) == "function" then
        legacy_mapper(cfg, user_cfg)
    end
    for _, k in ipairs(allowed_keys) do
        if user_cfg[k] ~= nil then
            cfg[k] = user_cfg[k]
        end
    end
    return cfg
end

-- ============================================================================
-- Pipeline de base
-- ============================================================================

function Pipeline:new(name, config)
    local self = setmetatable({}, Pipeline)
    self.name = name or "pipeline"
    self.config = config or {}
    self.model = nil
    self.tokenizer = nil
    self.trained = false
    self.steps = {}
    return self
end

-- ============================================================================
-- Constructeurs de pipelines spécialisés
-- ============================================================================

-- Pipeline Transformer (GPT, BERT, etc.)
function Pipeline.Transformer(config)
    config = config or {}

    local allowed = {
        -- TransformerConfig
        "seq_len", "d_model", "vocab_size", "padding_idx", "num_layers", "num_heads", "mlp_hidden", "output_dim", "causal",
        -- ModelConfig (communs)
        "dropout", "optimizer", "beta1", "beta2", "epsilon", "weight_decay", "min_lr", "decay_rate", "decay_steps", "warmup_steps", "decay_strategy",
    }

    local function legacy_map(cfg, user)
        if user.max_seq_len ~= nil and user.seq_len == nil then cfg.seq_len = user.max_seq_len end
        if user.embed_dim ~= nil and user.d_model == nil then cfg.d_model = user.embed_dim end
        if user.d_ff ~= nil and user.mlp_hidden == nil then cfg.mlp_hidden = user.d_ff end
        if user.model_type == "decoder" and user.causal == nil then cfg.causal = true end
        if user.model_type == "encoder" and user.causal == nil then cfg.causal = false end
    end

    local fallback = {
        vocab_size = 50000,
        seq_len = 512,
        d_model = 768,
        num_layers = 12,
        num_heads = 12,
        mlp_hidden = 3072,
        dropout = 0.1,
        causal = true,
    }

    local cfg = build_config("transformer", config, allowed, fallback, legacy_map)
    local pipe = Pipeline:new("transformer_pipeline", cfg)
    
    pipe.build = function(self)
        ensure_mimir()
        log("🏗️  Construction du pipeline Transformer...")
        
        -- Créer tokenizer
        Mimir.Tokenizer.create(self.config.vocab_size)
        self.tokenizer = true
        
        -- Créer modèle
        Mimir.Model.create("transformer", self.config)
        local ok, params = Mimir.Model.build()
        
        if ok then
            self.model = true
            log("✓ Modèle construit: " .. params .. " paramètres")
            return true, params
        end
        
        return false, "Échec construction"
    end
    
    pipe.train = function(self, dataset_path, epochs, lr)
        if not self.model then
            log("❌ Modèle non construit. Appelez build() d'abord.")
            return false
        end
        
        log("🎓 Entraînement du modèle...")
        
        -- Charger dataset
        Mimir.Dataset.load(dataset_path)
        if type(Mimir.Dataset) == "table" and type(Mimir.Dataset.prepare_sequences) == "function" then
            Mimir.Dataset.prepare_sequences(self.config.seq_len)
        end
        
        -- Entraîner
        Mimir.Model.train(epochs or 10, lr or 0.0003)
        
        self.trained = true
        log("✓ Entraînement terminé")
        return true
    end
    
    pipe.generate = function(self, prompt, max_length)
        if not self.trained and not self.model then
            log("⚠️  Modèle non entraîné, génération peut être aléatoire")
        end
        
        return Mimir.Model.infer(prompt)
    end
    
    pipe.save = function(self, path)
        if not self.model then
            return false, "Modèle non construit"
        end
        
        Mimir.Serialization.save(path, "safetensors")
        log("✓ Pipeline sauvegardé: " .. path)
        return true
    end
    
    return pipe
end

-- Pipeline UNet (Segmentation, génération d'images)
function Pipeline.UNet(config)
    config = config or {}

    local allowed = {
        "image_w", "image_h", "image_c", "base_channels", "depth",
        "dropout", "optimizer", "beta1", "beta2", "epsilon", "weight_decay", "min_lr", "decay_rate", "decay_steps", "warmup_steps", "decay_strategy",
    }

    local function legacy_map(cfg, user)
        if user.image_size ~= nil and (user.image_w == nil and user.image_h == nil) then
            cfg.image_w = user.image_size
            cfg.image_h = user.image_size
        end
        if user.input_channels ~= nil and user.image_c == nil then cfg.image_c = user.input_channels end
        if user.num_levels ~= nil and user.depth == nil then cfg.depth = user.num_levels end
    end

    local fallback = { image_w = 256, image_h = 256, image_c = 3, base_channels = 64, depth = 4, dropout = 0.0 }
    local cfg = build_config("unet", config, allowed, fallback, legacy_map)
    local pipe = Pipeline:new("unet_pipeline", cfg)
    
    pipe.build = function(self)
        ensure_mimir()
        log("🏗️  Construction du pipeline UNet...")
        
        Mimir.Model.create("unet", self.config)
        local ok, params = Mimir.Model.build()
        
        if ok then
            self.model = true
            log("✓ UNet construit: " .. params .. " paramètres")
            return true, params
        end
        
        return false
    end
    
    pipe.train = function(self, dataset_path, epochs, lr)
        if not self.model then
            log("❌ Modèle non construit")
            return false
        end
        
        log("🎓 Entraînement UNet...")
        Mimir.Dataset.load(dataset_path)
        Mimir.Model.train(epochs or 50, lr or 0.001)
        
        self.trained = true
        return true
    end
    
    pipe.segment = function(self, image_path)
        if not self.model then
            return nil, "Modèle non construit"
        end
        
        -- Inférence pour segmentation
        return Mimir.Model.infer(image_path)
    end
    
    pipe.save = function(self, path)
        Mimir.Serialization.save(path, "safetensors")
        return true
    end
    
    return pipe
end

-- Pipeline VAE (Variational Autoencoder)
function Pipeline.VAE(config)
    config = config or {}

    local allowed = {
        "image_w", "image_h", "image_c", "latent_dim", "hidden_dim",
        "dropout", "optimizer", "beta1", "beta2", "epsilon", "weight_decay", "min_lr", "decay_rate", "decay_steps", "warmup_steps", "decay_strategy",
    }

    local function legacy_map(cfg, user)
        -- Compat: input_dim=784 -> 28x28x1 (si carré parfait)
        if type(user.input_dim) == "number" and (user.image_w == nil and user.image_h == nil and user.image_c == nil) then
            local s = math.floor(math.sqrt(user.input_dim))
            if s * s == user.input_dim then
                cfg.image_w = s
                cfg.image_h = s
                cfg.image_c = 1
            end
        end
        if type(user.hidden_dims) == "table" and user.hidden_dim == nil then
            cfg.hidden_dim = tonumber(user.hidden_dims[1]) or cfg.hidden_dim
        end
    end

    local fallback = { image_w = 28, image_h = 28, image_c = 1, latent_dim = 128, hidden_dim = 256 }
    local cfg = build_config("vae", config, allowed, fallback, legacy_map)
    local pipe = Pipeline:new("vae_pipeline", cfg)
    
    pipe.build = function(self)
        ensure_mimir()
        log("🏗️  Construction du pipeline VAE...")
        
        Mimir.Model.create("vae", self.config)
        local ok, params = Mimir.Model.build()
        
        if ok then
            self.model = true
            log("✓ VAE construit: " .. params .. " paramètres")
            return true, params
        end
        
        return false
    end
    
    pipe.train = function(self, dataset_path, epochs, lr)
        log("🎓 Entraînement VAE...")
        Mimir.Dataset.load(dataset_path)
        Mimir.Model.train(epochs or 100, lr or 0.001)
        
        self.trained = true
        return true
    end
    
    pipe.encode = function(self, input)
        -- Encoder vers espace latent
        return Mimir.Model.forward(input)
    end
    
    pipe.decode = function(self, latent)
        -- Décoder depuis espace latent
        return Mimir.Model.infer(latent)
    end
    
    pipe.generate = function(self, num_samples)
        -- Générer depuis bruit aléatoire
        log("🎨 Génération de " .. num_samples .. " échantillons...")
        local samples = {}
        for i = 1, num_samples do
            table.insert(samples, Mimir.Model.infer(""))
        end
        return samples
    end
    
    return pipe
end

-- Pipeline ViT (Vision Transformer)
function Pipeline.ViT(config)
    config = config or {}

    local allowed = {
        "num_tokens", "d_model", "num_layers", "num_heads", "mlp_hidden", "output_dim", "causal",
        "dropout", "optimizer", "beta1", "beta2", "epsilon", "weight_decay", "min_lr", "decay_rate", "decay_steps", "warmup_steps", "decay_strategy",
    }

    local function legacy_map(cfg, user)
        if user.embed_dim ~= nil and user.d_model == nil then cfg.d_model = user.embed_dim end
        if user.d_ff ~= nil and user.mlp_hidden == nil then cfg.mlp_hidden = user.d_ff end

        -- Compat: image_size/patch_size -> num_tokens
        if user.num_tokens == nil and type(user.image_size) == "number" and type(user.patch_size) == "number" then
            local patches = user.image_size / user.patch_size
            if patches == math.floor(patches) then
                cfg.num_tokens = patches * patches
            end
        end

        -- Compat: num_classes -> output_dim
        if user.num_classes ~= nil and user.output_dim == nil then cfg.output_dim = user.num_classes end
    end

    local fallback = { num_tokens = 196, d_model = 768, num_layers = 12, num_heads = 12, mlp_hidden = 3072, output_dim = 1000, dropout = 0.1, causal = false }
    local cfg = build_config("vit", config, allowed, fallback, legacy_map)
    local pipe = Pipeline:new("vit_pipeline", cfg)
    
    pipe.build = function(self)
        ensure_mimir()
        log("🏗️  Construction du pipeline ViT...")
        
        Mimir.Model.create("vit", self.config)
        local ok, params = Mimir.Model.build()
        
        if ok then
            self.model = true
            log("✓ ViT construit: " .. params .. " paramètres")
            return true, params
        end
        
        return false
    end
    
    pipe.train = function(self, dataset_path, epochs, lr)
        log("🎓 Entraînement ViT...")
        Mimir.Dataset.load(dataset_path)
        Mimir.Model.train(epochs or 300, lr or 0.001)
        
        self.trained = true
        return true
    end
    
    pipe.classify = function(self, image_path)
        return Mimir.Model.infer(image_path)
    end
    
    return pipe
end

-- Pipeline GAN (Generative Adversarial Network)
function Pipeline.GAN(config)
    config = config or {}

    -- Le framework expose un seul "modèle courant" côté Lua (Mimir.Model.*).
    -- Un pipeline GAN nécessite au minimum 2 modèles simultanés (G + D) et
    -- n'est donc pas supporté proprement par l'API publique actuelle.
    local pipe = Pipeline:new("gan_pipeline", {})
    
    pipe.build = function(self)
        ensure_mimir()
        log("⚠️  Pipeline GAN non supporté par l'API Lua actuelle")
        return false, "GAN: nécessite 2 modèles simultanés (non supporté)"
    end
    
    pipe.train = function(self, dataset_path, epochs, lr)
        log("🎓 Entraînement GAN (adversarial)...")
        Mimir.Dataset.load(dataset_path)
        
        -- Entraînement adversarial
        for epoch = 1, epochs do
            -- Train discriminator
            -- Train generator
            log("  Epoch " .. epoch .. "/" .. epochs)
        end
        
        self.trained = true
        return true
    end
    
    pipe.generate = function(self, num_samples)
        log("🎨 Génération de " .. num_samples .. " images...")
        local images = {}
        for i = 1, num_samples do
            table.insert(images, Mimir.Model.infer(""))
        end
        return images
    end
    
    return pipe
end

-- Pipeline Diffusion (Stable Diffusion style)
function Pipeline.Diffusion(config)
    config = config or {}

    local allowed = {
        "image_w", "image_h", "image_c", "time_dim", "hidden_dim",
        "dropout", "optimizer", "beta1", "beta2", "epsilon", "weight_decay", "min_lr", "decay_rate", "decay_steps", "warmup_steps", "decay_strategy",
    }

    local function legacy_map(cfg, user)
        if user.image_size ~= nil and (user.image_w == nil and user.image_h == nil) then
            cfg.image_w = user.image_size
            cfg.image_h = user.image_size
        end
        if user.image_channels ~= nil and user.image_c == nil then cfg.image_c = user.image_channels end
        if user.model_channels ~= nil and user.hidden_dim == nil then cfg.hidden_dim = user.model_channels end
        -- NOTE: timesteps n'est pas mappé automatiquement: la config officielle utilise time_dim (dimension embedding)
    end

    local fallback = { image_w = 256, image_h = 256, image_c = 3, time_dim = 256, hidden_dim = 128 }
    local cfg = build_config("diffusion", config, allowed, fallback, legacy_map)
    local pipe = Pipeline:new("diffusion_pipeline", cfg)
    
    pipe.build = function(self)
        ensure_mimir()
        log("🏗️  Construction du pipeline Diffusion...")
        
        Mimir.Model.create("diffusion", self.config)
        local ok, params = Mimir.Model.build()
        
        if ok then
            self.model = true
            log("✓ Diffusion construit: " .. params .. " paramètres")
            return true, params
        end
        
        return false
    end
    
    pipe.train = function(self, dataset_path, epochs, lr)
        log("🎓 Entraînement Diffusion...")
        Mimir.Dataset.load(dataset_path)
        Mimir.Model.train(epochs or 1000, lr or 0.0001)
        
        self.trained = true
        return true
    end
    
    pipe.generate = function(self, prompt, num_steps)
        log("🎨 Génération via diffusion...")
        log("  Prompt: " .. (prompt or "none"))
        log("  Steps: " .. (num_steps or self.config.timesteps))
        
        return Mimir.Model.infer(prompt)
    end
    
    return pipe
end

-- Pipeline ResNet (Classification)
function Pipeline.ResNet(config)
    config = config or {}

    local allowed = {
        "image_w", "image_h", "image_c", "base_channels", "num_classes", "blocks1", "blocks2", "blocks3", "blocks4",
        "dropout", "optimizer", "beta1", "beta2", "epsilon", "weight_decay", "min_lr", "decay_rate", "decay_steps", "warmup_steps", "decay_strategy",
    }

    local function legacy_map(cfg, user)
        if type(user.layers) == "table" then
            if user.blocks1 == nil then cfg.blocks1 = tonumber(user.layers[1]) or cfg.blocks1 end
            if user.blocks2 == nil then cfg.blocks2 = tonumber(user.layers[2]) or cfg.blocks2 end
            if user.blocks3 == nil then cfg.blocks3 = tonumber(user.layers[3]) or cfg.blocks3 end
            if user.blocks4 == nil then cfg.blocks4 = tonumber(user.layers[4]) or cfg.blocks4 end
        end
        if user.input_size ~= nil and (user.image_w == nil and user.image_h == nil) then
            cfg.image_w = user.input_size
            cfg.image_h = user.input_size
        end
        if user.image_channels ~= nil and user.image_c == nil then cfg.image_c = user.image_channels end
    end

    local fallback = { image_w = 224, image_h = 224, image_c = 3, base_channels = 64, num_classes = 1000, blocks1 = 3, blocks2 = 4, blocks3 = 6, blocks4 = 3 }
    local cfg = build_config("resnet", config, allowed, fallback, legacy_map)
    local pipe = Pipeline:new("resnet_pipeline", cfg)
    
    pipe.build = function(self)
        ensure_mimir()
        log("🏗️  Construction du pipeline ResNet...")
        
        Mimir.Model.create("resnet", self.config)
        local ok, params = Mimir.Model.build()
        
        if ok then
            self.model = true
            log("✓ ResNet construit: " .. params .. " paramètres")
            return true, params
        end
        
        return false
    end
    
    pipe.train = function(self, dataset_path, epochs, lr)
        log("🎓 Entraînement ResNet...")
        Mimir.Dataset.load(dataset_path)
        Mimir.Model.train(epochs or 90, lr or 0.1)
        
        self.trained = true
        return true
    end
    
    pipe.classify = function(self, image_path)
        return Mimir.Model.infer(image_path)
    end
    
    return pipe
end

-- Pipeline MobileNet (Mobile/Edge)
function Pipeline.MobileNet(config)
    config = config or {}

    local allowed = {
        "image_w", "image_h", "image_c", "base_channels", "num_classes",
        "dropout", "optimizer", "beta1", "beta2", "epsilon", "weight_decay", "min_lr", "decay_rate", "decay_steps", "warmup_steps", "decay_strategy",
    }

    local function legacy_map(cfg, user)
        if user.input_size ~= nil and (user.image_w == nil and user.image_h == nil) then
            cfg.image_w = user.input_size
            cfg.image_h = user.input_size
        end
        if user.image_channels ~= nil and user.image_c == nil then cfg.image_c = user.image_channels end
    end

    local fallback = { image_w = 224, image_h = 224, image_c = 3, base_channels = 32, num_classes = 1000 }
    local cfg = build_config("mobilenet", config, allowed, fallback, legacy_map)
    local pipe = Pipeline:new("mobilenet_pipeline", cfg)
    
    pipe.build = function(self)
        ensure_mimir()
        log("🏗️  Construction du pipeline MobileNet...")
        
        Mimir.Model.create("mobilenet", self.config)
        local ok, params = Mimir.Model.build()
        
        if ok then
            self.model = true
            log("✓ MobileNet construit: " .. params .. " paramètres")
            log("  Optimisé pour CPU/Edge devices")
            return true, params
        end
        
        return false
    end
    
    pipe.train = function(self, dataset_path, epochs, lr)
        log("🎓 Entraînement MobileNet...")
        Mimir.Dataset.load(dataset_path)
        Mimir.Model.train(epochs or 150, lr or 0.045)
        
        self.trained = true
        return true
    end
    
    pipe.classify = function(self, image_path)
        return Mimir.Model.infer(image_path)
    end
    
    return pipe
end

-- ============================================================================
-- Pipeline Manager - Gestion de plusieurs pipelines
-- ============================================================================

local PipelineManager = {}
PipelineManager.__index = PipelineManager

function PipelineManager:new()
    local self = setmetatable({}, PipelineManager)
    self.pipelines = {}
    return self
end

function PipelineManager:add(name, pipeline)
    self.pipelines[name] = pipeline
    log("✓ Pipeline ajouté: " .. name)
end

function PipelineManager:get(name)
    return self.pipelines[name]
end

function PipelineManager:list()
    log("\n📋 Pipelines disponibles:")
    for name, pipe in pairs(self.pipelines) do
        local status = pipe.model and "✓" or "○"
        local trained = pipe.trained and " (trained)" or ""
        log("  " .. status .. " " .. name .. trained)
    end
end

function PipelineManager:save_all(base_path)
    log("\n💾 Sauvegarde de tous les pipelines...")
    for name, pipe in pairs(self.pipelines) do
        if pipe.model and pipe.save then
            local path = base_path .. "/" .. name
            pipe:save(path)
        end
    end
    log("✓ Tous les pipelines sauvegardés")
end

-- ============================================================================
-- Exports
-- ============================================================================

return {
    Pipeline = Pipeline,
    PipelineManager = PipelineManager,
    
    -- Constructeurs directs
    Transformer = Pipeline.Transformer,
    UNet = Pipeline.UNet,
    VAE = Pipeline.VAE,
    ViT = Pipeline.ViT,
    GAN = Pipeline.GAN,
    Diffusion = Pipeline.Diffusion,
    ResNet = Pipeline.ResNet,
    MobileNet = Pipeline.MobileNet
}
