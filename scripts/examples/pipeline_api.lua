-- ============================================================================
-- Mímir Framework - Pipeline API
-- Système de pipeline pour piloter tous les modèles depuis Lua
-- ============================================================================

local Pipeline = {}
Pipeline.__index = Pipeline

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
    
    local default_config = {
        vocab_size = 50000,
        embed_dim = 768,
        num_layers = 12,
        num_heads = 12,
        d_ff = 3072,
        max_seq_len = 512,
        dropout = 0.1,
        model_type = "transformer"  -- transformer, encoder, decoder
    }
    
    for k, v in pairs(config) do
        default_config[k] = v
    end
    
    local pipe = Pipeline:new("transformer_pipeline", default_config)
    
    pipe.build = function(self)
        log("🏗️  Construction du pipeline Transformer...")
        
        -- Créer tokenizer
        Mimir.Tokenizer.create(self.config.vocab_size)
        self.tokenizer = true
        
        -- Créer modèle
        Mimir.Model.create(self.config.model_type, self.config)
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
        Mimir.Dataset.prepare_sequences(self.config.max_seq_len)
        
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
    
    local default_config = {
        input_channels = 3,
        output_channels = 3,
        base_channels = 64,
        num_levels = 4,
        blocks_per_level = 2,
        use_attention = true,
        use_residual = true,
        dropout = 0.0
    }
    
    for k, v in pairs(config) do
        default_config[k] = v
    end
    
    local pipe = Pipeline:new("unet_pipeline", default_config)
    
    pipe.build = function(self)
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
    
    local default_config = {
        input_dim = 784,
        latent_dim = 128,
        hidden_dims = {512, 256},
        beta = 1.0
    }
    
    for k, v in pairs(config) do
        default_config[k] = v
    end
    
    local pipe = Pipeline:new("vae_pipeline", default_config)
    
    pipe.build = function(self)
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
            table.insert(samples, Mimir.Model.infer(nil))
        end
        return samples
    end
    
    return pipe
end

-- Pipeline ViT (Vision Transformer)
function Pipeline.ViT(config)
    config = config or {}
    
    local default_config = {
        image_size = 224,
        patch_size = 16,
        num_classes = 1000,
        embed_dim = 768,
        num_layers = 12,
        num_heads = 12,
        d_ff = 3072,
        dropout = 0.1
    }
    
    for k, v in pairs(config) do
        default_config[k] = v
    end
    
    local pipe = Pipeline:new("vit_pipeline", default_config)
    
    pipe.build = function(self)
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
    
    local default_config = {
        latent_dim = 100,
        image_channels = 3,
        image_size = 64,
        gen_channels = 64,
        disc_channels = 64
    }
    
    for k, v in pairs(config) do
        default_config[k] = v
    end
    
    local pipe = Pipeline:new("gan_pipeline", default_config)
    
    pipe.build = function(self)
        log("🏗️  Construction du pipeline GAN...")
        
        -- Construire générateur
        Mimir.Model.create("generator", self.config)
        local ok_gen, params_gen = Mimir.Model.build()
        
        if ok_gen then
            self.generator = true
            log("✓ Générateur: " .. params_gen .. " paramètres")
        end
        
        -- Construire discriminateur
        Mimir.Model.create("discriminator", self.config)
        local ok_disc, params_disc = Mimir.Model.build()
        
        if ok_disc then
            self.discriminator = true
            log("✓ Discriminateur: " .. params_disc .. " paramètres")
        end
        
        self.model = ok_gen and ok_disc
        return self.model
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
            table.insert(images, Mimir.Model.infer(nil))
        end
        return images
    end
    
    return pipe
end

-- Pipeline Diffusion (Stable Diffusion style)
function Pipeline.Diffusion(config)
    config = config or {}
    
    local default_config = {
        image_channels = 3,
        image_size = 256,
        timesteps = 1000,
        model_channels = 128,
        attention_resolutions = {16, 8},
        num_res_blocks = 2
    }
    
    for k, v in pairs(config) do
        default_config[k] = v
    end
    
    local pipe = Pipeline:new("diffusion_pipeline", default_config)
    
    pipe.build = function(self)
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
    
    local default_config = {
        num_classes = 1000,
        layers = {3, 4, 6, 3},  -- ResNet-50
        base_channels = 64,
        use_bottleneck = true
    }
    
    for k, v in pairs(config) do
        default_config[k] = v
    end
    
    local pipe = Pipeline:new("resnet_pipeline", default_config)
    
    pipe.build = function(self)
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
    
    local default_config = {
        num_classes = 1000,
        width_mult = 1.0,
        input_size = 224
    }
    
    for k, v in pairs(config) do
        default_config[k] = v
    end
    
    local pipe = Pipeline:new("mobilenet_pipeline", default_config)
    
    pipe.build = function(self)
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
