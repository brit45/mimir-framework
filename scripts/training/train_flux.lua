-- Exemple d'entraînement du modèle Flux
-- Ce script montre comment entraîner un modèle de diffusion avec VAE et text conditioning

log("========================================")
log("Flux Training Pipeline")
log("========================================")

-- Configuration
local config = {
    -- Model configuration
    model = {
        latent_channels = 4,
        latent_resolution = 64,
        image_resolution = 512,
        vae_base_channels = 128,
        vae_num_res_blocks = 2,
        text_embed_dim = 768,
        text_max_length = 77,
        vocab_size = 50000,
        num_transformer_blocks = 12,
        transformer_dim = 1024,
        num_attention_heads = 16,
        timestep_embed_dim = 256,
        num_timesteps = 1000,
        beta_start = 0.0001,
        beta_end = 0.02
    },
    
    -- Training configuration
    training = {
        epochs = 100,
        learning_rate = 1e-4,
        batch_size = 4,
        accumulation_steps = 4,  -- Gradient accumulation
        warmup_steps = 1000,
        optimizer = "adamw",
        beta1 = 0.9,
        beta2 = 0.999,
        weight_decay = 0.01,
        epsilon = 1e-8,
        min_lr = 1e-6
    },
    
    -- Dataset
    dataset = {
        path = "datasets/images_with_captions/",
        image_size = 512,
        augmentation = true
    },
    
    -- Checkpointing
    checkpoint = {
        save_dir = "checkpoints/flux_model/",
        save_every = 5000,  -- steps
        keep_last = 3
    }
}

-- 1. Initialiser le tokenizer
log("\n[1] Initialisation du tokenizer...")
tokenizer.create(config.model.vocab_size)

-- Charger ou créer le vocabulaire
local vocab_file = "vocab/flux_vocab.txt"
if tokenizer.load(vocab_file) then
    log("✓ Vocabulaire chargé: " .. vocab_file)
else
    log("Création d'un nouveau vocabulaire...")
    
    -- Tokens spéciaux
    tokenizer.add_token("[PAD]")
    tokenizer.add_token("[UNK]")
    tokenizer.add_token("[BOS]")
    tokenizer.add_token("[EOS]")
    
    -- Apprendre BPE depuis un corpus (si disponible)
    if dataset.load(config.dataset.path) then
        local corpus = {}
        for i = 1, math.min(10000, dataset.size()) do
            local item = dataset.get(i)
            if item.text then
                table.insert(corpus, item.text)
            end
        end
        
        if #corpus > 0 then
            log("Apprentissage BPE sur " .. tostring(#corpus) .. " textes...")
            tokenizer.learn_bpe(corpus, 10000)
            tokenizer.save(vocab_file)
            log("✓ Vocabulaire créé et sauvegardé")
        end
    end
end

log("Vocabulaire: " .. tostring(tokenizer.vocab_size()) .. " tokens")

-- 2. Créer et construire le modèle
log("\n[2] Création du modèle Flux...")

-- Fusionner les configs
local full_config = {}
for k, v in pairs(config.model) do
    full_config[k] = v
end
for k, v in pairs(config.training) do
    full_config[k] = v
end

model.create("flux", full_config)
architectures.flux(full_config)
flux.set_tokenizer()

local build_success, num_params = model.build()
if not build_success then
    log("✗ Erreur lors de la construction du modèle")
    return
end

log("✓ Modèle créé: " .. tostring(num_params) .. " paramètres")
log("  Mémoire estimée: ~" .. tostring(math.floor(num_params * 4 / (1024*1024))) .. " MB")

-- 3. Charger le dataset
log("\n[3] Chargement du dataset...")
local dataset_success, num_items = dataset.load(config.dataset.path)

if not dataset_success then
    log("✗ Erreur: impossible de charger le dataset")
    log("Veuillez créer un dataset avec des images et légendes dans:")
    log("  " .. config.dataset.path)
    log("")
    log("Format attendu:")
    log("  datasets/images_with_captions/")
    log("    ├── images/")
    log("    │   ├── img001.jpg")
    log("    │   ├── img002.jpg")
    log("    │   └── ...")
    log("    └── captions/")
    log("        ├── img001.txt")
    log("        ├── img002.txt")
    log("        └── ...")
    return
end

log("✓ Dataset chargé: " .. tostring(num_items) .. " exemples")

-- 4. Préparer les séquences
log("\n[4] Préparation des données...")
dataset.prepare_sequences(config.model.text_max_length)
log("✓ Données préparées")

-- 5. Configuration de la mémoire
log("\n[5] Configuration mémoire...")
memory.config({
    max_memory_mb = 8192,  -- 8 GB
    enable_pooling = true,
    enable_compression = true
})
log("✓ Mémoire configurée")

-- 6. Entraînement
log("\n[6] Démarrage de l'entraînement...")
log("Configuration:")
log("  - Epochs: " .. tostring(config.training.epochs))
log("  - Learning rate: " .. tostring(config.training.learning_rate))
log("  - Batch size: " .. tostring(config.training.batch_size))
log("  - Optimizer: " .. config.training.optimizer)
log("")
log("Entraînement en cours...")
log("(Ctrl+C pour arrêter)")
log("")

local train_success = model.train(
    config.training.epochs,
    config.training.learning_rate
)

if train_success then
    log("\n✓ Entraînement terminé")
else
    log("\n⚠ Entraînement interrompu")
end

-- 7. Sauvegarder le modèle final
log("\n[7] Sauvegarde du modèle final...")
local final_path = config.checkpoint.save_dir .. "final/"
model.save(final_path)
tokenizer.save(final_path .. "tokenizer.json")
log("✓ Modèle final sauvegardé: " .. final_path)

-- 8. Test de génération
log("\n[8] Test de génération...")
local test_prompts = {
    "a beautiful mountain landscape at sunset",
    "a cat sitting on a windowsill",
    "abstract digital art with vibrant colors",
    "realistic portrait of a person smiling"
}

for i, prompt in ipairs(test_prompts) do
    log("\nPrompt " .. tostring(i) .. ": \"" .. prompt .. "\"")
    
    -- Générer (requiert que le modèle soit entraîné)
    local result = flux.generate(prompt, 50, 7.5)
    
    if result then
        log("✓ Image générée (" .. tostring(result.resolution) .. "x" .. tostring(result.resolution) .. ")")
        -- Sauvegarder l'image (fonctionnalité à implémenter)
        -- viz.save_image("outputs/flux_" .. tostring(i) .. ".png", result)
    else
        log("✗ Erreur lors de la génération")
    end
end

-- 9. Statistiques finales
log("\n========================================")
log("STATISTIQUES FINALES")
log("========================================")
memory.print_stats()
log("")
log("Modèle entraîné et sauvegardé!")
log("Pour utiliser le modèle:")
log("  1. model.load('" .. final_path .. "')")
log("  2. flux.generate('your prompt here', 50, 7.5)")
log("========================================")
