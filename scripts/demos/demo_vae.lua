-- ============================================================================
-- Mímir Framework - Demo VAE
-- Variational Autoencoder pour génération
-- ============================================================================

log("=" .. string.rep("=", 78))
log("🎨 Demo VAE - Variational Autoencoder")
log("=" .. string.rep("=", 78))

local function _mimir_add_module_path()
    local ok, info = pcall(debug.getinfo, 1, "S")
    if not ok or type(info) ~= "table" then return end
    local src = info.source
    if type(src) ~= "string" or src:sub(1, 1) ~= "@" then return end
    local dir = src:sub(2):match("(.*/)")
    if not dir then return end
    package.path = package.path .. ";" .. dir .. "../modules/?.lua;" .. dir .. "../modules/?/init.lua"
end

_mimir_add_module_path()
local Arch = require("arch")

local Allocator = (type(_G.Mimir) == "table" and type(Mimir.Allocator) == "table") and Mimir.Allocator or _G.Allocator
local model = (type(_G.Mimir) == "table" and type(Mimir.Model) == "table") and Mimir.Model or _G.model

-- Configuration système
log("\n🔧 Configuration...")
Allocator.configure({max_ram_gb = 10.0, enable_compression = true})
model.set_hardware(true)
log("✓ Système configuré")

-- Configuration VAE
local config = {
    input_dim = 784,
    latent_dim = 128,
    encoder_hidden = {512, 256},
    decoder_hidden = {256, 512},
    use_batchnorm = false
}

-- Créer le modèle VAE
log("\n🏗️  Création du modèle VAE...")
local cfg, warn = Arch.build_config("vae", config)
if warn then
    log("⚠️  " .. tostring(warn))
end

local success, err = model.create("vae", cfg)
if not success then
    log("❌ Erreur création modèle: " .. (err or "inconnue"))
    return
end

log("✓ VAE créé via registre")

-- Allouer et initialiser
success, params = model.allocate_params()
if success then
    model.init_weights("xavier", 42)
    log("✓ Modèle construit: " .. params .. " paramètres")
    log("  Mémoire: " .. string.format("%.2f MB", params * 4 / 1024 / 1024))
else
    log("❌ Erreur allocation")
    return
end

-- Afficher config
log("\n📋 Configuration:")
log("  Input:  " .. config.input_dim .. " dimensions")
log("  Latent: " .. config.latent_dim .. " dimensions")
log("  Encoder hidden: " .. (type(config.encoder_hidden) == "table" and table.concat(config.encoder_hidden, " → ") or "n/a"))
log("  Decoder hidden: " .. (type(config.decoder_hidden) == "table" and table.concat(config.decoder_hidden, " → ") or "n/a"))

-- Estimation architecture
log("\n🏛️  Architecture:")
if type(config.encoder_hidden) == "table" and #config.encoder_hidden >= 1 then
    log("  Encoder: " .. config.input_dim .. " → " .. table.concat(config.encoder_hidden, " → ") .. " → " .. config.latent_dim)
else
    log("  Encoder: " .. config.input_dim .. " → ... → " .. config.latent_dim)
end

if type(config.decoder_hidden) == "table" and #config.decoder_hidden >= 1 then
    log("  Decoder: " .. config.latent_dim .. " → " .. table.concat(config.decoder_hidden, " → ") .. " → " .. config.input_dim)
else
    log("  Decoder: " .. config.latent_dim .. " → ... → " .. config.input_dim)
end

-- Usage pour génération
log("\n💡 Cas d'usage:")
log("  • Génération d'images (MNIST, CelebA)")
log("  • Compression latente")
log("  • Interpolation dans l'espace latent")
log("  • Détection d'anomalies")
log("  • Génération conditionnelle")

-- Modes d'utilisation
log("\n🎯 Modes d'utilisation:")
log("  1. Encode: image → latent vector")
log("  2. Decode: latent vector → image")
log("  3. Generate: random noise → image")
log("  4. Reconstruct: image → encode → decode → image")

-- Sauvegarder
log("\n💾 Sauvegarde du modèle...")
os.execute("mkdir -p checkpoints")
Mimir.Serialization.save("checkpoints/vae_demo.safetensors", "safetensors")
log("✓ Modèle sauvegardé")

log("\n✓ Demo VAE terminé!")
log("=" .. string.rep("=", 78))
