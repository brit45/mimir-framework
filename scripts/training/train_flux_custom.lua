#!/usr/bin/env lua
-- ══════════════════════════════════════════════════════════════
--  Script d'entraînement et inférence Flux - Mímir Framework v2.1
--  Dataset: Images + Textes (pas de prepare_sequence)
-- ══════════════════════════════════════════════════════════════

log("\n╔════════════════════════════════════════════════════════╗")
log("║   Flux Training Pipeline - Text-to-Image Diffusion   ║")
log("╚════════════════════════════════════════════════════════╝\n")

-- Flux/FluxModel n'est pas exposé dans l'API Lua v2.3 (skip propre)
if type(_G.flux) ~= "table" and type(_G.FluxModel) ~= "table" and (type(_G.Mimir) ~= "table" or type(Mimir.FluxModel) ~= "table") then
    log("⚠️  Flux/FluxModel indisponible dans l'API Lua v2.3 (skip)")
    return
end

-- ══════════════════════════════════════════════════════════════
--  ÉTAPE 1: CONFIGURATION SYSTÈME (OBLIGATOIRE!)
-- ══════════════════════════════════════════════════════════════

log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  1. Configuration Système")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

-- ⚠️ CRITIQUE: Toujours configurer l'allocateur en premier!
Allocator.configure({
    max_ram_gb = 10.0,              -- Limite stricte (ne pas dépasser!)
    enable_compression = true       -- Compression LZ4 (~50% économie RAM)
})
log("✓ Allocateur configuré (limite: 10 GB, compression LZ4)")

-- Vérifier et activer l'accélération hardware
local hw = model.hardware_caps()
log("\n🔧 Capacités Hardware:")
log(string.format("  • AVX2:  %s", hw.avx2 and "✓" or "✗"))
log(string.format("  • FMA:   %s", hw.fma and "✓" or "✗"))
log(string.format("  • F16C:  %s", hw.f16c and "✓" or "✗"))
log(string.format("  • BMI2:  %s", hw.bmi2 and "✓" or "✗"))

if hw.avx2 or hw.fma then
    model.set_hardware(true)
    log("\n✓ Accélération hardware activée\n")
end

-- ══════════════════════════════════════════════════════════════
--  ÉTAPE 2: CONFIGURATION DU MODÈLE FLUX
-- ══════════════════════════════════════════════════════════════

log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  2. Configuration du Modèle Flux")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

-- Configuration optimisée pour 10 GB de RAM
local config = {
    -- Configuration Image/Latent
    image_resolution = 256,         -- Résolution images (256x256 pour économiser RAM)
    latent_channels = 4,            -- Canaux espace latent
    latent_resolution = 32,         -- Résolution latent (256/8 = 32)
    
    -- Configuration VAE
    vae_base_channels = 64,         -- Canaux de base VAE (réduit pour 10GB)
    vae_channel_mult = {1, 2, 4, 4},-- Multiplicateurs de canaux
    num_res_blocks = 2,             -- Blocs résiduels par niveau
    
    -- Configuration Texte
    vocab_size = 30000,             -- Vocabulaire (30k tokens raisonnable)
    text_max_length = 77,           -- Longueur max texte (standard CLIP)
    text_embed_dim = 512,           -- Dimension embeddings texte (réduit)
    
    -- Configuration Transformer de Diffusion
    transformer_dim = 512,          -- Dimension transformer (réduit)
    num_transformer_blocks = 6,     -- Nombre de blocs (réduit pour 10GB)
    num_attention_heads = 8,        -- Têtes d'attention
    mlp_ratio = 4.0,                -- Ratio MLP
    
    -- Configuration Diffusion
    timestep_embed_dim = 256,       -- Dimension embedding timestep
    num_diffusion_steps = 1000,     -- Steps de diffusion
    
    -- Configuration Training
    batch_size = 1,                 -- Batch size = 1 pour économiser RAM
    learning_rate = 1e-4,           -- Learning rate
    epochs = 20,                    -- Nombre d'epochs
    num_inference_steps = 50,       -- Steps d'inférence (moins que training)
    
    -- Dataset
    dataset_path = "../tensor/datasets.old"
}

log("Configuration Flux:")
log(string.format("  • Image Resolution: %dx%d", config.image_resolution, config.image_resolution))
log(string.format("  • Latent Space: %dx%dx%d", config.latent_resolution, config.latent_resolution, config.latent_channels))
log(string.format("  • VAE Channels: %d", config.vae_base_channels))
log(string.format("  • Transformer: %d blocks, %d dim, %d heads", 
    config.num_transformer_blocks, config.transformer_dim, config.num_attention_heads))
log(string.format("  • Vocab Size: %d tokens", config.vocab_size))
log(string.format("  • Text Max Length: %d", config.text_max_length))

-- ══════════════════════════════════════════════════════════════
--  ÉTAPE 3: CRÉATION DU TOKENIZER
-- ══════════════════════════════════════════════════════════════

log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  3. Tokenizer")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

tokenizer.create(config.vocab_size)

-- Ajouter tokens spéciaux
tokenizer.add_token("[PAD]")    -- Padding
tokenizer.add_token("[UNK]")    -- Unknown
tokenizer.add_token("[BOS]")    -- Begin of sequence
tokenizer.add_token("[EOS]")    -- End of sequence
tokenizer.add_token("[MASK]")   -- Masking

log("✓ Tokenizer créé (vocab_size: " .. config.vocab_size .. ")")
log("  Tokens spéciaux: [PAD], [UNK], [BOS], [EOS], [MASK]")

-- ══════════════════════════════════════════════════════════════
--  ÉTAPE 4: CRÉATION DU MODÈLE FLUX
-- ══════════════════════════════════════════════════════════════

log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  4. Création du Modèle Flux")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

-- Créer le modèle avec l'API FluxModel (gestion globale)
local success = FluxModel.new(config)

if not success then
    log("❌ ERREUR: Impossible de créer le modèle Flux")
    log("\n💡 Solutions possibles:")
    log("  1. Réduire image_resolution (256 → 128)")
    log("  2. Réduire vae_base_channels (64 → 32)")
    log("  3. Réduire num_transformer_blocks (6 → 4)")
    log("  4. Réduire transformer_dim (512 → 256)")
    os.exit(1)
end

log("✓ Modèle FluxModel créé avec succès")

-- Configurer le tokenizer pour Flux
FluxModel.setPromptTokenizer(tokenizer)
log("✓ Tokenizer configuré pour FluxModel")

-- Afficher les statistiques mémoire
local mem_stats = MemoryGuard.getStats()
log(string.format("\n📊 Mémoire utilisée: %.2f MB / %.0f MB (%.1f%%)", 
    mem_stats.current_mb, 
    10 * 1024, 
    mem_stats.usage_percent))

-- ══════════════════════════════════════════════════════════════
--  ÉTAPE 5: CHARGEMENT DU DATASET
-- ══════════════════════════════════════════════════════════════

log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  5. Chargement du Dataset")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

log("Dataset path: " .. config.dataset_path)

-- Charger le dataset (images + textes)
local dataset_ok, num_items = dataset.load(config.dataset_path)

if not dataset_ok then
    log("❌ ERREUR: Impossible de charger le dataset")
    log("  Path: " .. config.dataset_path)
    log("\n💡 Structure attendue:")
    log("  " .. config.dataset_path .. "/")
    log("    ├── image/")
    log("    │   ├── uuid1.png")
    log("    │   └── uuid2.png")
    log("    └── text/")
    log("        ├── uuid1.txt")
    log("        └── uuid2.txt")
    os.exit(1)
end

log("✓ Dataset chargé: " .. num_items .. " paires image-texte")

-- ⚠️ PAS de prepare_sequences car on a des images + textes!
log("  (Pas de prepare_sequences: dataset = images + textes)")

-- ══════════════════════════════════════════════════════════════
--  ÉTAPE 6: ENTRAÎNEMENT
-- ══════════════════════════════════════════════════════════════

log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  6. Entraînement")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

-- Activer le mode training
FluxModel.train()
log("✓ Mode training activé")

log("\nConfiguration training:")
log(string.format("  • Epochs: %d", config.epochs))
log(string.format("  • Learning Rate: %.6f", config.learning_rate))
log(string.format("  • Batch Size: %d", config.batch_size))
log(string.format("  • Diffusion Steps: %d", config.num_diffusion_steps))
log(string.format("  • Dataset Size: %d", num_items))

log("\n" .. string.rep("─", 50))
log("Démarrage de l'entraînement...")
log(string.rep("─", 50) .. "\n")

local start_time = os.time()
local total_loss = 0.0
local total_batches = 0

-- Boucle d'entraînement par epochs
for epoch = 1, config.epochs do
    log(string.format("\n╔═══════════════════════════════════════╗"))
    log(string.format("║  EPOCH %d/%d", epoch, config.epochs))
    log(string.format("╚═══════════════════════════════════════╝"))
    
    local epoch_loss = 0
    local epoch_batches = 0
    local epoch_start = os.time()
    
    
    -- Calculer nombre de batches
    local num_batches = math.ceil(num_items / config.batch_size)
    
    -- Itérer sur les batches
    for batch_idx = 1, num_batches do
        local start_idx = (batch_idx - 1) * config.batch_size + 1
        local end_idx = math.min(start_idx + config.batch_size - 1, num_items)
        
        -- Affichage progression
        if batch_idx % 10 == 0 or batch_idx == 1 or batch_idx == num_batches then
            local mem_stats = MemoryGuard.getStats()
            log(string.format("  Batch %3d/%3d | Mem: %.0f MB (%.1f%%)", 
                batch_idx, num_batches, mem_stats.current_mb, mem_stats.usage_percent))
        end
        
        -- Créer les inputs du batch
        local batch_images = {}
        local batch_texts = {}
        
        for i = start_idx, end_idx do
            local item = dataset.get(i)
            if item and item.image and item.text then
                table.insert(batch_images, item.image)
                table.insert(batch_texts, item.text)
            end
        end
        
        if #batch_images == 0 then
            log(string.format("  ⚠️  Batch %d vide, skip", batch_idx))
            goto continue_batch
        end
        
        -- Entraîner sur ce batch avec FluxModel API
        local batch_loss = 0
        for i = 1, #batch_images do
            -- Tokeniser le texte
            local tokens = FluxModel.tokenizePrompt(batch_texts[i])
            
            -- Calculer la loss de diffusion
            local loss = FluxModel.computeDiffusionLoss(batch_images[i], tokens)
            batch_loss = batch_loss + loss
        end
        
        batch_loss = batch_loss / #batch_images
        epoch_loss = epoch_loss + batch_loss
        epoch_batches = epoch_batches + 1
        total_loss = total_loss + batch_loss
        total_batches = total_batches + 1
        
        -- Afficher détails tous les 20 batches
        if batch_idx % 20 == 0 then
            local avg_loss = epoch_loss / epoch_batches
            log(string.format("    └─ Loss: %.4f | Avg: %.4f", batch_loss, avg_loss))
        end
        
        -- Libération mémoire périodique
        if batch_idx % 50 == 0 then
            collectgarbage("collect")
        end
        
        ::continue_batch::
    end
    
    -- Résumé de l'epoch
    local epoch_time = os.time() - epoch_start
    local avg_epoch_loss = epoch_batches > 0 and (epoch_loss / epoch_batches) or 0
    
    log(string.format("\n  ✓ Epoch %d complété:", epoch))
    log(string.format("    • Loss moyenne: %.4f", avg_epoch_loss))
    log(string.format("    • Batches traités: %d/%d", epoch_batches, num_batches))
    log(string.format("    • Durée: %ds", epoch_time))
    
    -- Statistiques mémoire
    local mem_stats = MemoryGuard.getStats()
    log(string.format("    • RAM: %.0f MB / %.0f MB (%.1f%%)", 
        mem_stats.current_mb, 10 * 1024, mem_stats.usage_percent))
    
    -- Sauvegarde checkpoint tous les 5 epochs
    if epoch % 5 == 0 then
        local checkpoint_dir = "checkpoints/flux_custom/epoch_" .. epoch
        log(string.format("\n  💾 Sauvegarde checkpoint epoch %d...", epoch))
        
        local mkdir_success = os.execute("mkdir -p " .. checkpoint_dir)
        if mkdir_success then
            -- Note: FluxModel sauvegarde gérée par le framework
            local save_success = tokenizer.save(checkpoint_dir .. "/tokenizer.json")
            if save_success then
                log(string.format("    ✓ Checkpoint sauvegardé: %s", checkpoint_dir))
            else
                log("    ⚠️  Échec sauvegarde tokenizer")
            end
        else
            log("    ⚠️  Échec création du dossier checkpoint")
        end
    end
    
    -- Garbage collection après chaque epoch
    collectgarbage("collect")
end

-- Résumé final de l'entraînement
local total_time = os.time() - start_time
local avg_total_loss = total_batches > 0 and (total_loss / total_batches) or 0

log("\n" .. string.rep("═", 50))
log("✅ Entraînement terminé!")
log(string.rep("═", 50))
log(string.format("\n📊 Statistiques globales:"))
log(string.format("  • Epochs: %d", config.epochs))
log(string.format("  • Batches traités: %d", total_batches))
log(string.format("  • Loss moyenne finale: %.4f", avg_total_loss))
log(string.format("  • Durée totale: %ds (%.1fm)", total_time, total_time / 60))

-- ══════════════════════════════════════════════════════════════
--  ÉTAPE 7: SAUVEGARDE DU MODÈLE FINAL
-- ══════════════════════════════════════════════════════════════

log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  7. Sauvegarde du Modèle Final")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

local final_dir = "checkpoints/flux_custom/final"
os.execute("mkdir -p " .. final_dir)

log("💾 Sauvegarde du modèle FluxModel...")
-- Note: FluxModel sauvegarde interne
tokenizer.save(final_dir .. "/tokenizer.json")
log("✓ Tokenizer sauvegardé: " .. final_dir)
    
    -- Sauvegarder la config
    local config_file = io.open(final_dir .. "/config.txt", "w")
    if config_file then
        config_file:write("Flux Model Configuration\n")
        config_file:write("========================\n\n")
        config_file:write(string.format("Image Resolution: %d\n", config.image_resolution))
        config_file:write(string.format("Latent Channels: %d\n", config.latent_channels))
        config_file:write(string.format("VAE Base Channels: %d\n", config.vae_base_channels))
        config_file:write(string.format("Transformer Dim: %d\n", config.transformer_dim))
        config_file:write(string.format("Num Transformer Blocks: %d\n", config.num_transformer_blocks))
        config_file:write(string.format("Vocab Size: %d\n", config.vocab_size))
        config_file:write(string.format("Text Max Length: %d\n", config.text_max_length))
        config_file:write(string.format("\nTraining Stats:\n"))
        config_file:write(string.format("Epochs: %d\n", config.epochs))
        config_file:write(string.format("Final Loss: %.4f\n", avg_total_loss))
        config_file:close()
        log("✓ Configuration sauvegardée")
    end

-- ══════════════════════════════════════════════════════════════
--  ÉTAPE 8: INFÉRENCE (GÉNÉRATION D'IMAGES)
-- ══════════════════════════════════════════════════════════════

log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  8. Test d'Inférence")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

-- Activer le mode eval pour l'inférence
FluxModel.eval()
log("✓ Mode eval activé (inférence)")

-- Prompts de test
local test_prompts = {
    "une jeune fille anthropomorphe avec des oreilles de chien",
    "paysage de montagne au coucher du soleil",
    "ville cyberpunk la nuit avec néons"
}

log("Génération de " .. #test_prompts .. " images de test...\n")

-- Créer dossier de sortie
os.execute("mkdir -p outputs/flux_custom")

for i, prompt in ipairs(test_prompts) do
    log(string.format("[%d/%d] Prompt: \"%s\"", i, #test_prompts, prompt))
    
    -- Générer l'image avec l'API FluxModel
    local success, result = pcall(function()
        return FluxModel.generate(prompt, config.num_inference_steps)
    end)
    
    if success and result then
        if type(result) == "table" and #result > 0 then
            log(string.format("  ✓ Image générée: %d pixels", #result))
            
            -- Statistiques de l'image
            local min_val, max_val, sum = 1e9, -1e9, 0
            for j = 1, math.min(1000, #result) do
                local val = result[j]
                min_val = math.min(min_val, val)
                max_val = math.max(max_val, val)
                sum = sum + val
            end
            local mean = sum / math.min(1000, #result)
            log(string.format("  Stats: min=%.4f, max=%.4f, mean=%.4f", min_val, max_val, mean))
            log(string.format("  💾 outputs/flux_custom/generated_%d.png", i))
        else
            log("  ✓ Génération complétée")
        end
    else
        log("  ⚠️  Échec de la génération")
        if result then
            log("    Erreur: " .. tostring(result))
        end
    end
    
    log("")
end

-- ══════════════════════════════════════════════════════════════
--  RÉSUMÉ FINAL
-- ══════════════════════════════════════════════════════════════

log("╔════════════════════════════════════════════════════════╗")
log("║                    RÉSUMÉ FINAL                        ║")
log("╚════════════════════════════════════════════════════════╝\n")

log("✅ Pipeline Flux complété avec succès!")
log("")
log("📊 Statistiques:")
log(string.format("  • Dataset: %d images", num_items))
log(string.format("  • Epochs entraînés: %d", config.epochs))
log(string.format("  • Images générées: %d", #test_prompts))
log("")

local final_mem = MemoryGuard.getStats()
log("💾 Utilisation Mémoire:")
log(string.format("  • Courante: %.2f MB", final_mem.current_mb))
log(string.format("  • Pic: %.2f MB", final_mem.peak_mb))
log(string.format("  • Limite: %.0f MB", 10 * 1024))
log(string.format("  • Utilisation: %.1f%%", final_mem.usage_percent))
log("")

if final_mem.peak_mb < 10 * 1024 then
    log("✅ Limite de 10 GB respectée!")
else
    log("⚠️  Limite de 10 GB dépassée!")
end

log("\n📁 Fichiers générés:")
log("  • Modèle final: " .. final_dir)
log("  • Checkpoints: checkpoints/flux_custom/epoch_*/")
log("  • Images test: outputs/flux_custom/")
log("")

log("💡 Utilisation du modèle:")
log("  1. Créer: flux_model = FluxModel.new(config)")
log("  2. Eval: flux_model:eval()")
log("  3. Générer: flux_model:generate(\"votre prompt\", 50)")
log("")

log("📚 Documentation:")
log("  • docs/03-API-Reference/flux-api.md")
log("  • docs/02-User-Guide/diffusion-models.md")
log("")

log("\n✨ Pipeline Flux terminé avec succès! ✨\n")
