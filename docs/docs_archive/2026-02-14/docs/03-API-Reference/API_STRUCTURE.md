# Structure de l'API Mimir v2.3.0

## Vue d'ensemble

L'API Mimir est organisée en **namespaces hiérarchiques** pour une meilleure organisation et clarté :

```
Mimir
├── Serialization.*     (Sauvegarde/chargement de modèles)
├── Dataset.*           (Gestion de datasets - future)
├── Model.*             (Manipulation de modèles - future)
└── Allocator.*         (Configuration mémoire - future)
```

---

## 📦 Mimir.Serialization

Namespace pour la sérialisation et la désérialisation de modèles.

### Mimir.Serialization.save()

Sauvegarde un modèle dans l'un des 3 formats supportés.

```lua
---@param path string Chemin du fichier/dossier
---@param format? SerializationFormat Format ("safetensors", "raw_folder", "debug_json")
---@param options? SaveOptions Options de sauvegarde
---@return boolean ok true si succès
---@return string? err Message d'erreur si échec
Mimir.Serialization.save(path, format, options)
```

**Formats supportés :**
- `"safetensors"` : Format production compatible HuggingFace/PyTorch
- `"raw_folder"` : Format debug avec checksums SHA256
- `"debug_json"` : Format inspection avec statistiques

**Exemples :**

```lua
-- SafeTensors (production)
Mimir.Serialization.save("model.safetensors", "safetensors", {
    save_optimizer = true,
    include_git_info = true
})

-- RawFolder (debug)
Mimir.Serialization.save("checkpoint/", "raw_folder", {
    save_tokenizer = true,
    save_encoder = true
})

-- DebugJson Enhanced v1.1.0 (diagnostic)
Mimir.Serialization.save("debug.json", "debug_json", {
    include_gradients = true,
    include_optimizer_state = true,
    include_weight_deltas = true,
    include_checksums = true,
    max_values_per_tensor = 20
})
```

---

### Mimir.Serialization.load()

Charge un modèle avec détection automatique du format.

```lua
---@param path string Chemin du fichier/dossier
---@param format? SerializationFormat Format (optionnel, auto-détecté)
---@param options? LoadOptions Options de chargement
---@return boolean ok true si succès
---@return string? err Message d'erreur si échec
Mimir.Serialization.load(path, format, options)
```

**Exemples :**

```lua
-- Auto-détection du format
Mimir.Serialization.load("model.safetensors")
Mimir.Serialization.load("checkpoint/")

-- Format explicite avec vérification checksums
Mimir.Serialization.load("checkpoint/", "raw_folder", {
    validate_checksums = true
})
```

---

### Mimir.Serialization.detect_format()

Détecte automatiquement le format d'un checkpoint.

```lua
---@param path string Chemin du fichier/dossier
---@return SerializationFormat? format Format détecté ou nil si inconnu
---@return string? err Message d'erreur si échec
Mimir.Serialization.detect_format(path)
```

**Exemples :**

```lua
local format = Mimir.Serialization.detect_format("model.safetensors")
print(format)  -- "SAFETENSORS"

local format2 = Mimir.Serialization.detect_format("checkpoint/")
print(format2)  -- "RAWFOLDER"
```

---

### Mimir.Serialization.save_enhanced_debug()

Fonction spécialisée pour le diagnostic d'entraînement avec Enhanced Debug JSON v1.1.0.

```lua
---@param path string Chemin du fichier JSON
---@param options? SaveOptions Options Enhanced Debug JSON v1.1
---@return boolean ok true si succès
---@return string? err Message d'erreur si échec
Mimir.Serialization.save_enhanced_debug(path, options)
```

**Exemples :**

```lua
-- Snapshot avant optimizer step
Mimir.Serialization.save_enhanced_debug("/tmp/before.json", {
    include_gradients = true,
    include_checksums = true,
    include_weight_deltas = false
})

-- Snapshot après optimizer step (avec deltas)
Mimir.Serialization.save_enhanced_debug("/tmp/after.json", {
    include_gradients = true,
    include_optimizer_state = true,
    include_checksums = true,
    include_weight_deltas = true
})
```

---

## 📊 Options de Sauvegarde

### SaveOptions

```lua
---@class SaveOptions
{
    -- Options générales
    save_tokenizer = true,         -- Sauvegarder le tokenizer
    save_encoder = true,           -- Sauvegarder l'encoder
    save_optimizer = false,        -- Sauvegarder l'état optimizer
    include_git_info = true,       -- Inclure info git
    debug_max_values = 100,        -- Nb valeurs debug (legacy)
    
    -- Enhanced Debug JSON v1.1.0
    include_gradients = false,     -- Inclure gradients
    include_optimizer_state = false, -- Inclure optimizer state
    max_values_per_tensor = 20,    -- Nb valeurs par tensor
    include_activations = false,   -- Inclure activations
    include_checksums = false,     -- Inclure checksums
    include_weight_deltas = false  -- Inclure weight deltas
}
```

### LoadOptions

```lua
---@class LoadOptions
{
    load_tokenizer = true,         -- Charger le tokenizer
    load_encoder = true,           -- Charger l'encoder
    load_optimizer = false,        -- Charger l'état optimizer
    validate_checksums = true      -- Vérifier checksums (RawFolder)
}
```

---

## 🚀 Utilisation Pratique

### Workflow de Production

```lua
-- 1. Entraînement
Mimir.Model.create("my_model")
-- ... train model ...

-- 2. Sauvegarde production
Mimir.Serialization.save("models/my_model_v1.0.safetensors", "safetensors", {
    save_optimizer = false,
    include_git_info = true
})

-- 3. Chargement en inférence
Mimir.Serialization.load("models/my_model_v1.0.safetensors")
```

### Workflow de Debug

```lua
-- 1. Snapshot avant step
Mimir.Serialization.save("debug/before.json", "debug_json", {
    include_gradients = true,
    include_checksums = true
})

-- 2. Optimizer step
model.optimizer_step(0.001)

-- 3. Snapshot après step (avec deltas)
Mimir.Serialization.save("debug/after.json", "debug_json", {
    include_gradients = true,
    include_optimizer_state = true,
    include_weight_deltas = true
})

-- 4. Analyse des changements
-- Comparer before.json et after.json
```

### Workflow de Checkpoint

```lua
-- Sauvegarde checkpoint complet avec checksums
Mimir.Serialization.save("checkpoints/epoch_10/", "raw_folder", {
    save_optimizer = true,
    save_tokenizer = true,
    save_encoder = true
})

-- Chargement avec vérification intégrité
Mimir.Serialization.load("checkpoints/epoch_10/", "raw_folder", {
    validate_checksums = true
})
```

---

## 📚 Documentation Complète

- [Enhanced Debug JSON v1.1.0](ENHANCED_DEBUG_JSON.md) - Format diagnostic détaillé
- [API Unifiée](UNIFIED_API_MIGRATION.md) - Migration et utilisation
- [Validation API](SERIALIZATION_API_VALIDATED.md) - Tests et performances

---

## ✅ Avantages de la Structure Hiérarchique

### Organisation Claire

```lua
Mimir.Serialization.save()      -- Sérialisation
Mimir.Dataset.load()            -- (future) Gestion datasets
Mimir.Model.create()            -- (future) Création modèles
Mimir.Allocator.configure()     -- (future) Configuration mémoire
```

### Namespaces Séparés

Chaque namespace regroupe des fonctionnalités liées :

- **Serialization** : Tout ce qui concerne save/load de modèles
- **Dataset** : Chargement et prétraitement de données
- **Model** : Construction et manipulation de modèles
- **Allocator** : Gestion mémoire et optimisations

### Extensibilité

Facile d'ajouter de nouvelles fonctionnalités sans polluer le namespace global :

```lua
-- Futur:
Mimir.Serialization.export_to_onnx()
Mimir.Dataset.augment()
Mimir.Model.quantize()
Mimir.Allocator.enable_hugepages()
```

### Auto-complétion IDE

Les namespaces permettent une meilleure auto-complétion :

```lua
Mimir.Serialization.  -- IDE propose: save, load, detect_format, save_enhanced_debug
Mimir.Dataset.        -- IDE propose: (futures fonctionnalités)
```

---

## 🎯 Convention de Nommage

### Fonctions

- **save** : Sauvegarder des données
- **load** : Charger des données
- **detect** : Détecter/analyser
- **configure** : Configurer un système

### Namespaces

- **PascalCase** : Serialization, Dataset, Model, Allocator
- **snake_case** pour fonctions : save(), load(), detect_format()

### Formats

- **Minuscules avec underscores** : "safetensors", "raw_folder", "debug_json"
- **MAJUSCULES en sortie** : "SAFETENSORS", "RAWFOLDER", "DEBUGJSON"

---

## 📝 Notes de Version

**v2.3.0** - Décembre 2025
- ✅ Namespace `Mimir.Serialization` avec 4 fonctions
- ✅ Enhanced Debug JSON v1.1.0 intégré
- ✅ API unifiée avec auto-détection de format
- ✅ Support de 3 formats (SafeTensors, RawFolder, DebugJson)

**Future** - v2.3.0+
- ⏳ Namespace `Mimir.Dataset`
- ⏳ Namespace `Mimir.Model`
- ⏳ Namespace `Mimir.Allocator`
- ⏳ Export ONNX/TorchScript
