# 🧠 Bonnes Pratiques de Gestion Mémoire - Mímir Framework

## 📖 Guide Complet pour une Utilisation Optimale de la Mémoire

Ce document explique comment utiliser correctement le système de gestion mémoire de Mímir pour éviter les crashs OOM et maximiser les performances.

---

## 🎯 Principes Fondamentaux

### Limite Stricte de 10 GB
Mímir impose une **limite stricte de 10 GB de RAM** pour protéger votre système contre les crashs OOM. Cette limite est appliquée par **MemoryGuard** qui refuse toute allocation dépassant le budget.

### Allocation Dynamique par Défaut
Depuis v2.0, **toutes les allocations de poids de modèle utilisent l'allocation dynamique** qui passe par Mimir.MemoryGuard. Cela garantit :
- ✅ Contrôle strict de la consommation mémoire
- ✅ Refus propre des allocations impossibles
- ✅ Panic OOM contrôlé (pas de crash OS)
- ✅ Compression LZ4 automatique des tenseurs inactifs
- ✅ Éviction LRU si nécessaire

---

## ✅ Configuration Correcte

### 1. Toujours Configurer l'Allocateur au Début

```lua
-- ✅ BON: Configuration au début du script
Mimir.Allocator.configure({
    max_ram_gb = 10.0,              -- Limite stricte (recommandé: 10 GB)
    enable_compression = true       -- Compression LZ4 pour économiser la RAM
})
```

```lua
-- ❌ MAUVAIS: Pas de configuration = limite par défaut non garantie
Mimir.Model.create("my_model")
Mimir.Model.allocate_params()  -- Peut dépasser la RAM disponible!
```

### 2. Vérifier le Hardware Disponible

```lua
-- ✅ BON: Vérifier les capacités avant de configurer
local hw = model.hardware_caps()
print("AVX2: " .. (hw.avx2 and "✓" or "✗"))
print("FMA: " .. (hw.fma and "✓" or "✗"))

if hw.avx2 or hw.fma then
    model.set_hardware(true)  -- Active l'accélération si disponible
end
```

### 3. Utiliser des Configs Raisonnables

```lua
-- ✅ BON: Configuration réaliste pour 10 GB
local config = {
    vocab_size = 50000,
    d_model = 512,           -- Pas trop grand
    num_layers = 6,          -- Nombre raisonnable de couches
    num_heads = 8,
    max_seq_len = 512
}

-- ❌ MAUVAIS: Configuration irréaliste
local bad_config = {
    vocab_size = 1000000,    -- Trop grand!
    d_model = 4096,          -- Énorme!
    num_layers = 48,         -- Trop de couches!
    num_heads = 32,
    max_seq_len = 8192       -- Séquences géantes!
}
-- Cette config dépassera facilement 10 GB!
```

---

## 🚀 Workflow Recommandé

### Script Type (Ordre Important!)

```lua
#!/usr/bin/env lua

-- ══════════════════════════════════════════════════════════
--  1. CONFIGURATION SYSTÈME (TOUJOURS EN PREMIER!)
-- ══════════════════════════════════════════════════════════

-- Configurer l'allocateur mémoire
Mimir.Allocator.configure({
    max_ram_gb = 10.0,
    enable_compression = true
})
print("✓ Allocateur configuré (limite: 10 GB)")

-- Vérifier et activer le hardware
local hw = model.hardware_caps()
if hw.avx2 or hw.fma then
    model.set_hardware(true)
    print("✓ Accélération hardware activée")
end

-- ══════════════════════════════════════════════════════════
--  2. CRÉATION DU MODÈLE
-- ══════════════════════════════════════════════════════════

local config = {
    vocab_size = 30000,
    d_model = 512,
    num_layers = 6,
    num_heads = 8,
    max_seq_len = 512
}

Mimir.Model.create("my_model", config)
print("✓ Modèle créé")

-- Ajouter des layers si nécessaire
Mimir.Model.push_layer("embed", "Embedding", config.vocab_size * config.d_model)
Mimir.Model.push_layer("fc", "Linear", config.d_model * config.d_model)
-- ...

-- ══════════════════════════════════════════════════════════
--  3. ALLOCATION DES PARAMÈTRES (VÉRIFIER LE SUCCÈS!)
-- ══════════════════════════════════════════════════════════

local success, param_count = Mimir.Model.allocate_params()

if not success then
    print("❌ ERREUR: Impossible d'allouer les paramètres!")
    print("⚠️  Limite de 10 GB atteinte")
    print("💡 Solution: Réduire la taille du modèle (d_model, num_layers, vocab_size)")
    os.exit(1)
end

print(string.format("✓ Paramètres alloués: %d", param_count))

-- ══════════════════════════════════════════════════════════
--  4. INITIALISATION
-- ══════════════════════════════════════════════════════════

success = Mimir.Model.init_weights("xavier")
if not success then
    print("❌ Échec de l'initialisation des poids")
    os.exit(1)
end
print("✓ Poids initialisés")

-- ══════════════════════════════════════════════════════════
--  5. TRAINING / INFERENCE
-- ══════════════════════════════════════════════════════════

-- Votre code d'entraînement ici...
```

---

## 🎯 Cas d'Usage Spécifiques

### Transformer / GPT

```lua
Mimir.Allocator.configure({max_ram_gb = 10.0, enable_compression = true})

local config = {
    vocab_size = 50000,       -- 50k tokens OK
    d_model = 768,            -- GPT-2 small size
    num_layers = 12,          -- 12 couches OK
    num_heads = 12,
    max_seq_len = 1024,
    dropout = 0.1
}

Mimir.Model.create("gpt")
Mimir.Architectures.transformer(config)
Mimir.Model.allocate_params()  -- ~200 M params = ~800 MB OK pour 10 GB
```

### Vision (ResNet, UNet, ViT)

```lua
Mimir.Allocator.configure({max_ram_gb = 10.0, enable_compression = true})

-- ResNet-50
local resnet_config = {
    num_classes = 1000,
    image_size = 224,
    channels = 3,
    use_pretrained = false
}
Mimir.Architectures.resnet(resnet_config)  -- ~25M params = ~100 MB OK

-- UNet pour segmentation
local unet_config = {
    in_channels = 3,
    out_channels = 1,
    features = {64, 128, 256, 512}  -- Progression raisonnable
}
Mimir.Architectures.unet(unet_config)  -- ~30M params OK
```

### Diffusion Models

```lua
Mimir.Allocator.configure({max_ram_gb = 10.0, enable_compression = true})

local diffusion_config = {
    image_size = 256,          -- 256x256 OK (512x512 = beaucoup plus!)
    channels = 3,
    time_steps = 1000,
    unet_dim = 128,            -- Dimension modérée
    num_res_blocks = 2
}

Mimir.Architectures.diffusion(diffusion_config)
```

---

## ⚠️ Erreurs Courantes à Éviter

### ❌ Erreur 1: Ne Pas Configurer l'Allocateur

```lua
-- MAUVAIS!
Mimir.Model.create("my_model")
Mimir.Model.allocate_params()  -- Pas de limite définie!
```

**Impact**: Pas de protection contre les dépassements mémoire, crash OS possible.

### ❌ Erreur 2: Ne Pas Vérifier le Succès de l'Allocation

```lua
-- MAUVAIS!
Mimir.Model.allocate_params()  -- On ignore le retour
Mimir.Model.init_weights()     -- Crash si allocation a échoué!
```

**Solution**:
```lua
-- BON!
local success = Mimir.Model.allocate_params()
if not success then
    print("Allocation impossible, modèle trop grand")
    os.exit(1)
end
```

### ❌ Erreur 3: Config Irréaliste pour 10 GB

```lua
-- MAUVAIS!
local config = {
    vocab_size = 500000,     -- 500k tokens!
    d_model = 2048,          -- Énorme!
    num_layers = 24,         -- Trop de couches!
    max_seq_len = 4096       -- Séquences géantes!
}
-- Cette config nécessite ~50+ GB de RAM!
```

**Solution**: Utilisez des valeurs réalistes (voir section "Cas d'Usage").

### ❌ Erreur 4: Allouer Plusieurs Gros Modèles Simultanément

```lua
-- MAUVAIS!
Mimir.Model.create("model1")
Mimir.Architectures.transformer({d_model = 1024, num_layers = 12})
Mimir.Model.allocate_params()

Mimir.Model.create("model2")  -- Le premier est toujours en mémoire!
Mimir.Architectures.transformer({d_model = 1024, num_layers = 12})
Mimir.Model.allocate_params()  -- Risque de dépasser 10 GB!
```

**Solution**: Libérer explicitement ou limiter la taille de chaque modèle.

---

## 📊 Estimation de la Consommation Mémoire

### Formule Générale

```
RAM (MB) ≈ nombre_params * 4 bytes / (1024 * 1024)
```

Avec compression LZ4 (ratio ~50%):
```
RAM réelle ≈ RAM * 0.5  (pour tenseurs inactifs)
```

### Exemples Concrets

| Modèle | Params | RAM sans compression | RAM avec compression |
|--------|--------|---------------------|---------------------|
| Transformer (d=512, L=6) | ~50M | ~200 MB | ~100 MB |
| Transformer (d=768, L=12) | ~200M | ~800 MB | ~400 MB |
| ResNet-50 | ~25M | ~100 MB | ~50 MB |
| UNet (256x256) | ~30M | ~120 MB | ~60 MB |
| Diffusion (256x256) | ~80M | ~320 MB | ~160 MB |

### Limite Pratique pour 10 GB

Avec compression et overhead, vous pouvez entraîner des modèles jusqu'à **~500M-1B de paramètres** selon l'architecture et le batch size.

---

## 🛠️ Debugging

### Vérifier la Consommation Actuelle

Le système affiche automatiquement les stats au démarrage :
```
📊 MemoryGuard Stats:
   • Used: 1.2 GB / 10.0 GB (12%)
   • Allocations: 45
   • Peak: 1.8 GB
```

### En Cas d'Échec d'Allocation

Mímir affiche un message clair :
```
❌ PANIC: OUT OF MEMORY ❌
⛔ Impossible d'allouer tensor de 2048 MB
⛔ MemoryGuard a refusé l'allocation - limite atteinte

📊 Stats finales:
   • Used: 9.8 GB / 10.0 GB (98%)
   • Peak: 9.8 GB

⚠️  Le programme va s'arrêter proprement pour protéger l'OS...
```

**Actions**:
1. Réduire `d_model` (ex: 1024 → 512)
2. Réduire `num_layers` (ex: 12 → 6)
3. Réduire `vocab_size` (ex: 50000 → 30000)
4. Réduire `batch_size` pendant l'entraînement

---

## 🔒 Garanties de Sécurité

### Ce Qui Est Garanti ✅

1. **Pas de crash OS**: MemoryGuard refuse les allocations dépassant 10 GB
2. **Arrêt contrôlé**: En cas d'OOM, le programme s'arrête proprement avec des logs clairs
3. **Toutes les allocations contrôlées**:
   - Poids des modèles (`tensor(size, true)`)
   - Images chargées (stb_image wrappé)
   - Allocations dynamiques (DynamicTensorAllocator)
4. **Structure legacy désactivée**: Plus de `params.resize()` incontrôlé

### Ce Qui N'Est PAS Garanti ❌

1. **Stack overflow**: Si vous créez des structures locales énormes
2. **Allocations externes**: Bibliothèques tierces non wrappées
3. **Memory leaks**: Si vous ne libérez pas les ressources

---

## 📚 Ressources Complémentaires

- [MEMORY_SAFETY_FIXES.md](../MEMORY_SAFETY_FIXES.md) - Détails techniques des correctifs
- [REBUILD_AND_TEST.md](../REBUILD_AND_TEST.md) - Instructions de compilation
- [README.md](../README.md) - Documentation générale du framework
- [scripts/validate_memory_fixes.lua](../../scripts/tests/validate_memory_fixes.lua) - Script de test complet

---

## ✨ Checklist Avant d'Exécuter un Script

- [ ] `Mimir.Allocator.configure()` appelé au début
- [ ] `max_ram_gb` défini (recommandé: 10.0)
- [ ] `enable_compression = true`
- [ ] Configuration du modèle réaliste
- [ ] Vérification du succès de `allocate_params()`
- [ ] Hardware check avec `model.hardware_caps()`

---

*Mímir Framework v2.0 - Gestion mémoire sécurisée par défaut* 🛡️
