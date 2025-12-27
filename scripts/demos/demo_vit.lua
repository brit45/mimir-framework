-- ============================================================================
-- Mímir Framework - Demo ViT
-- Vision Transformer pour classification d'images
-- ============================================================================

log("=" .. string.rep("=", 78))
log("🖼️  Demo ViT - Vision Transformer")
log("=" .. string.rep("=", 78))

-- Configuration système
log("\n🔧 Configuration...")
allocator.configure({max_ram_gb = 10.0, enable_compression = true})
model.set_hardware(true)
log("✓ Système configuré")

-- Configuration ViT (Base/16)
local config = {
    image_size = 224,
    patch_size = 16,
    num_classes = 1000,
    d_model = 768,
    num_layers = 12,
    num_heads = 12,
    mlp_ratio = 4,
    dropout = 0.1,
    use_cls_token = true
}

-- Créer le modèle ViT
log("\n🏗️  Création du modèle ViT...")
local success, err = model.create("vit_model")
if not success then
    log("❌ Erreur: " .. (err or "inconnue"))
    return
end

-- Construire avec architectures API
success, err = architectures.vit(config)
if not success then
    log("❌ Erreur architecture: " .. (err or "inconnue"))
    return
end

log("✓ Architecture ViT-Base/16 construite")

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

-- Calcul du nombre de patches
local num_patches = (config.image_size / config.patch_size) ^ 2

-- Afficher config
log("\n📋 Configuration:")
log("  Image size: " .. config.image_size .. "x" .. config.image_size)
log("  Patch size: " .. config.patch_size .. "x" .. config.patch_size)
log("  Num patches: " .. num_patches)
log("  Classes: " .. config.num_classes)
log("  Embed dim: " .. config.embed_dim)
log("  Layers: " .. config.num_layers)
log("  Heads: " .. config.num_heads)

-- Estimation architecture
log("\n🏛️  Architecture:")
log("  Patch Embedding: " .. num_patches .. " patches × " .. config.embed_dim .. " dim")
log("  Transformer: " .. config.num_layers .. " layers")
log("  MLP Head: " .. config.embed_dim .. " → " .. config.num_classes)

-- Usage pour classification
log("\n💡 Cas d'usage:")
log("  • Classification d'images (ImageNet)")
log("  • Transfer learning (fine-tuning)")
log("  • Feature extraction")
log("  • Zero-shot classification")
log("  • Vision-language tasks")

-- Comparaison avec CNN
log("\n⚖️  ViT vs CNN:")
log("  ✓ Meilleure scalabilité (plus de données)")
log("  ✓ Attention globale dès le début")
log("  ✓ Architecture unifiée vision/langage")
log("  ⚠ Nécessite plus de données d'entraînement")
log("  ⚠ Plus coûteux en calcul")

-- Sauvegarder
log("\n💾 Sauvegarde du modèle...")
os.execute("mkdir -p checkpoints")
model.save("checkpoints/vit_demo")
log("✓ Modèle sauvegardé")

log("\n✓ Demo ViT terminé!")
log("=" .. string.rep("=", 78))
