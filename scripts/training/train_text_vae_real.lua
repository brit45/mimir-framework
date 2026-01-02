#!/usr/bin/env mimir --lua
-- ============================================================================
-- Entraînement "réel" d'un VAE sur un dataset de texte
-- - Charge le dataset depuis: ../tensor/datasets.old
-- - Représentation texte -> embedding réel via Tokenizer + Encoder (côté C++)
-- - Entraîne un VAEModel via Mimir.Model.train() (pipeline dataset réel)
-- - Sauvegarde en RawFolder dans ./checkpoints
-- ============================================================================

log("╔═══════════════════════════════════════════════════════════════╗")
log("║   Train Text VAE (Tokenizer+Encoder) + RawFolder checkpoint  ║")
log("╚═══════════════════════════════════════════════════════════════╝")

-- ---------------------------------------------------------------------------
-- 0) Runtime / mémoire (best practice: configurer tôt)
-- ---------------------------------------------------------------------------
log("\n[0/5] Runtime...")

-- Monitoring htop (progress)
if Mimir and Mimir.Htop and Mimir.Htop.create then
    -- API actuelle: create(enable_viz:boolean)
    local ok_htop = Mimir.Htop.create({
        enable_viz = false
    })
    if ok_htop then
        log("✓ Htop monitor activé")
        -- Première update (signature positionnelle dans le binding)
        pcall(function()
            Mimir.Htop.update(0, 1, 0, 1, 0.0, 0.0, 0.0)
        end)
    else
        log("⚠️  Htop: create() a échoué")
    end
end

if Mimir and Mimir.Allocator and Mimir.Allocator.configure then
    local ok = Mimir.Allocator.configure({
        max_ram_gb = 10.0,
        enable_compression = true,
        swap_strategy = "lru"
    })
    if ok then
        log("✓ Allocateur configuré (10GB + compression)")
    else
        log("⚠️  Allocateur: configure() a échoué (on continue)")
    end
end

if Mimir and Mimir.Model and Mimir.Model.set_hardware then
    -- true = auto-accel (CPU SIMD / Vulkan si dispo)
    Mimir.Model.set_hardware("auto")
end

-- ---------------------------------------------------------------------------
-- 1) Dataset
-- ---------------------------------------------------------------------------
log("\n[1/5] Dataset...")

local dataset_root = "../tensor/datasets.old"
local ok_ds, num_items_or_err = Mimir.Dataset.load(dataset_root)
if not ok_ds then
    error("dataset.load failed: " .. tostring(num_items_or_err))
end

local num_items = tonumber(num_items_or_err) or 0
log("✓ Dataset chargé: " .. tostring(num_items) .. " items")

-- ---------------------------------------------------------------------------
-- 2) Tokenizer (pour convertir le texte en tokens)
-- ---------------------------------------------------------------------------
log("\n[2/5] Tokenizer...")

local vocab_size = 50000
local ok_tok = false
if Mimir and Mimir.Tokenizer and Mimir.Tokenizer.create then
    ok_tok = Mimir.Tokenizer.create(vocab_size) == true
elseif tokenizer and tokenizer.create then
    ok_tok = tokenizer.create(vocab_size) == true
end

if not ok_tok then
    error("Tokenizer.create indisponible ou échec")
end

local tok = (Mimir and Mimir.Tokenizer) or tokenizer
log("✓ Tokenizer prêt (vocab_size cible=" .. tostring(vocab_size) .. ")")

-- Construire un minimum de vocabulaire à partir d'un échantillon
local vocab_build_items = math.min(num_items, 500)
local ensure = tok.tokenize_ensure or tok.tokenize
if not ensure then
    error("Tokenizer: tokenize/tokenize_ensure indisponible")
end

log("• Construction vocab (échantillon: " .. tostring(vocab_build_items) .. " items)...")
for i = 1, vocab_build_items do
    local item = Mimir.Dataset.get(i)
    local text = item and item.text

    if not text and item and item.text_file then
        local f = io.open(item.text_file, "r")
        if f then
            text = f:read("*a")
            f:close()
        end
    end

    if text and #text > 0 then
        ensure(text)
    end

    if i % 100 == 0 then
        log("  - vocab progress: " .. i .. "/" .. vocab_build_items)
    end
end

local actual_vocab = tok.vocab_size and tok.vocab_size() or "?"
log("✓ Vocab construit (taille actuelle=" .. tostring(actual_vocab) .. ")")

-- ---------------------------------------------------------------------------
-- 3) Modèle: VAE
-- ---------------------------------------------------------------------------
log("\n[3/5] Modèle VAE...")

-- Dimension des embeddings (doit matcher l'Encoder)
-- Par défaut, l'API crée un Encoder avec embed_dim=256 si rien n'est fourni.
local embed_dim = 256

local vae_cfg = {
    -- input_dim est l'espace d'entrée du VAE: ici on reconstruit des embeddings
    input_dim = embed_dim,
    embed_dim = embed_dim,
    latent_dim = 64,
    encoder_hidden = {512, 256},
    decoder_hidden = {256, 512},

    -- VAE specifics
    activation = "relu",
    kl_beta = 1.0,
    use_mean_in_infer = true,
    seed = 1337,

    -- Train options (consommées par Mimir.Model.train pour le VAE)
    optimizer = "adamw",
    beta1 = 0.9,
    beta2 = 0.999,
    epsilon = 1e-8,
    weight_decay = 0.01,
    max_items = 0,         -- 0 = tout le dataset; mets 2000 pour un smoke test
    max_text_chars = 8192
}

local ok_create, err_create = Mimir.Model.create("vae", vae_cfg)
if not ok_create then
    error("Model.create('vae') failed: " .. tostring(err_create))
end

local ok_build, params_or_err = Mimir.Model.build()
if not ok_build then
    error("Model.build failed: " .. tostring(params_or_err))
end

-- Optionnel: ré-init explicite (le build initialise déjà, mais on garde un seed stable)
local ok_init, err_init = Mimir.Model.init_weights("xavier", 1337)
if not ok_init then
    error("Model.init_weights failed: " .. tostring(err_init))
end

log("✓ VAEModel prêt (embed_dim=" .. embed_dim .. ", params=" .. tostring(params_or_err) .. ")")

os.execute("mkdir -p checkpoints")
local out_dir = "checkpoints/text_vae_rawfolder/"
os.execute("mkdir -p " .. out_dir)

local ok_save, err_save = Mimir.Serialization.save(out_dir.."start_debug.json", "debug_json", {
    save_tokenizer = true,
    save_encoder = true,
    include_checksums = true,
    include_gradients= true,
    include_optimizer_state = true,
    include_activations = true
})

-- ---------------------------------------------------------------------------
-- 4) Entraînement
-- ---------------------------------------------------------------------------
log("\n[4/5] Entraînement...")

local epochs = 10
local lr = 5e-4

local ok_train, err_train = Mimir.Model.train(epochs, lr)
if not ok_train then
    error("Model.train failed: " .. tostring(err_train))
end

os.execute("mkdir -p checkpoints")
local out_dir = "checkpoints/text_vae_rawfolder/"
os.execute("mkdir -p " .. out_dir)

local ok_save, err_save = Mimir.Serialization.save(out_dir.."end_debug.json", "debug_json", {
    save_tokenizer = true,
    save_encoder = true,
    include_checksums = true,
    include_gradients= true,
    include_optimizer_state = true,
    include_activations = true
})

log("✓ Entraînement terminé")

-- Mini check inference (retourne une table de floats = reconstruction)
local sample_text = "Bonjour, ceci est un test." 
local recon = Mimir.Model.infer(sample_text)
if type(recon) == "table" then
    log("• Infer OK: recon dims=" .. tostring(#recon))
    local preview = {}
    for i = 1, math.min(8, #recon) do
        preview[#preview + 1] = string.format("%.4f", tonumber(recon[i]) or 0.0)
    end
    log("  recon[1..] = [" .. table.concat(preview, ", ") .. "]")
else
    log("⚠️ infer() n'a pas renvoyé une table (type=" .. tostring(type(recon)) .. ")")
end

-- ---------------------------------------------------------------------------
-- 5) Sauvegarde (RawFolder)
-- ---------------------------------------------------------------------------
log("\n[5/5] Sauvegarde...")

os.execute("mkdir -p checkpoints")
local out_dir = "checkpoints/text_vae_rawfolder"
os.execute("mkdir -p " .. out_dir)

local ok_save, err_save = Mimir.Serialization.save(out_dir, "raw_folder", {
    save_tokenizer = true,
    save_encoder = true,
    include_checksums = true,
    include_gradients= true,
    include_optimizer_state = true,
    include_activations = true
})

if not ok_save then
    error("Serialization.save(raw_folder) failed: " .. tostring(err_save))
end

log("✓ Checkpoint RawFolder écrit: " .. out_dir)
log("\n✅ OK")
