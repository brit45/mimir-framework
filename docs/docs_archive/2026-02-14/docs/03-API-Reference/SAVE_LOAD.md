# Système de Sérialisation Mímir

Le système de sérialisation de Mímir offre **3 formats** pour sauvegarder et charger les modèles, chacun adapté à un cas d'usage spécifique.

---

## 📦 Formats Disponibles

### 1. SafeTensors (Production) ✅

**Format de référence** compatible avec l'écosystème Python (HuggingFace, etc).

**Caractéristiques:**
- ✅ Fichier unique `.safetensors`
- ✅ Conforme à la spécification SafeTensors
- ✅ Header JSON + données binaires contiguës
- ✅ Little-endian, lecture rapide
- ✅ Interopérable avec PyTorch/TensorFlow via safetensors
- ✅ Metadata dans `__metadata__`

**Quand l'utiliser:**
- ✅ Production
- ✅ Partage de modèles
- ✅ Interopérabilité avec Python
- ✅ Distribution publique

**Structure du fichier:**
```
[8 bytes: header_length (uint64 LE)]
[N bytes: JSON header UTF-8]
[M bytes: tensor data (contiguous, LE)]
```

**Exemple de header:**
```json
{
  "layer1/weight_0": {
    "dtype": "F32",
    "shape": [128],
    "data_offsets": [0, 512]
  },
  "layer2/weight_0": {
    "dtype": "F32",
    "shape": [256],
    "data_offsets": [512, 1536]
  },
  "__metadata__": {
    "format": "safetensors",
    "format_version": "0.3.0",
    "mimir_version": "2.3.0",
    "created_at": 1735394400,
    "total_size": 1536
  }
}
```

---

### 2. RawFolder (Debug & Development) 🔍

**Format lisible** pour le développement et le debugging.

**Caractéristiques:**
- ✅ Structure de répertoires
- ✅ Tous les configs en JSON
- ✅ Tensors en fichiers `.bin` + `.json`
- ✅ Checksums SHA256 pour validation
- ✅ Chargement partiel possible
- ✅ Inspectable manuellement

**Quand l'utiliser:**
- ✅ Développement local
- ✅ Debugging
- ✅ Inspection manuelle
- ✅ Versioning (Git friendly)
- ✅ Tests d'intégrité

**Structure:**
```
checkpoint_root/
  manifest.json              # Index général
  model/
    architecture.json        # Structure du modèle
    training.json            # État optimizer (optionnel)
  tensors/
    layer1_weight_0.bin      # Données brutes
    layer1_weight_0.json     # Config + checksum
    layer2_weight_0.bin
    layer2_weight_0.json
    ...
  tokenizer/
    Mimir.Tokenizer.json           # Vocabulaire
  encoder/
    encoder.json             # Embeddings config
  dataset/
    Mimir.Dataset.json             # Info dataset (optionnel)
```

**Exemple `manifest.json`:**
```json
{
  "format": "mimir_raw_checkpoint",
  "format_version": "1.0.0",
  "mimir_version": "2.3.0",
  "git_commit": "abc1234",
  "created_at": 1735394400,
  "components": {
    "tensors": 128,
    "model_architecture": true,
    "tokenizer": true,
    "encoder": true,
    "optimizer": false
  },
  "tensor_index": [
    {
      "name": "layer1_weight_0",
      "bin_file": "tensors/layer1_weight_0.bin",
      "json_file": "tensors/layer1_weight_0.json"
    }
  ]
}
```

**Exemple `layer1_weight_0.json`:**
```json
{
  "name": "layer1_weight_0",
  "dtype": "F32",
  "shape": [128],
  "byte_size": 512,
  "checksum": "a3b2c1d4e5f6...",
  "checksum_algo": "sha256",
  "data_file": "layer1_weight_0.bin"
}
```

---

### 3. DebugJson (Développement uniquement) 🐛

**Format JSON complet** avec statistiques et échantillons de données.

**⚠️ NE PAS utiliser pour les gros modèles!**

**Caractéristiques:**
- ✅ Fichier JSON unique
- ✅ Structure du modèle
- ✅ Statistiques par tensor (min/max/mean/std)
- ✅ Échantillons de valeurs (truncated)
- ✅ **Read-only** (pas de chargement)

**Quand l'utiliser:**
- ✅ Petits modèles de test
- ✅ Analyse rapide
- ✅ Debugging initial
- ✅ Documentation
- ❌ **JAMAIS pour les modèles de production**

**Structure:**
```json
{
  "format": "mimir_debug_dump",
  "format_version": "1.0.0",
  "warning": "This is a debug dump. NOT for production.",
  "model": {
    "name": "test_model",
    "total_params": 448,
    "num_layers": 3
  },
  "layers": [
    {"name": "layer1", "type": "Dense", "params_count": 128},
    {"name": "layer2", "type": "Dense", "params_count": 256}
  ],
  "tensors": [
    {
      "name": "layer1/weight_0",
      "dtype": "F32",
      "shape": [128],
      "total_elements": 128,
      "stats": {
        "min": -0.5,
        "max": 0.5,
        "mean": 0.01,
        "std": 0.15
      },
      "sample_values": [0.1, -0.2, 0.3, ...],
      "sample_size": 50,
      "truncated": true
    }
  ]
}
```

---

## 🔧 API C++

### Sauvegarde

```cpp
#include "Serialization/Serialization.hpp"

using namespace Mimir::Serialization;

// SafeTensors
SaveOptions opts;
opts.format = CheckpointFormat::SafeTensors;
opts.save_tokenizer = false;  // SafeTensors = weights only
std::string error;
bool ok = save_checkpoint(model, "model.safetensors", opts, &error);

// RawFolder
SaveOptions opts;
opts.format = CheckpointFormat::RawFolder;
opts.save_tokenizer = true;
opts.save_encoder = true;
bool ok = save_checkpoint(model, "checkpoints/run1", opts, &error);

// DebugJson
SaveOptions opts;
opts.format = CheckpointFormat::DebugJson;
opts.debug_max_values = 100;  // Max valeurs par tensor
bool ok = save_checkpoint(model, "debug.json", opts, &error);
```

### Chargement

```cpp
// SafeTensors
LoadOptions opts;
opts.format = CheckpointFormat::SafeTensors;
opts.strict_mode = true;
bool ok = load_checkpoint(model, "model.safetensors", opts, &error);

// RawFolder
LoadOptions opts;
opts.format = CheckpointFormat::RawFolder;
opts.load_tokenizer = true;
opts.validate_checksums = true;
bool ok = load_checkpoint(model, "checkpoints/run1", opts, &error);
```

### Auto-détection

```cpp
CheckpointFormat fmt = detect_format("model.safetensors");
// → CheckpointFormat::SafeTensors

fmt = detect_format("checkpoints/run1");
// → CheckpointFormat::RawFolder (si manifest.json existe)
```

---

## 🐍 API Lua

### Sauvegarde

```lua
-- SafeTensors (production)
local ok, err = Mimir.Serialization.save(
  "model.safetensors",
  "safetensors",
  { save_tokenizer = false }
)

-- RawFolder (debug)
local ok, err = Mimir.Serialization.save(
  "checkpoints/run1",
  "raw_folder",
  {
    save_tokenizer = true,
    save_encoder = true,
    save_optimizer = true,
    include_git_info = true
  }
)

-- DebugJson (inspection / export)
-- Note: DebugJson est un format d'export (pas de `load` via Mimir.Serialization.load)
local ok, err = Mimir.Serialization.save(
  "debug.json",
  "debug_json",
  { debug_max_values = 50 }
)

-- Format par défaut
-- Si vous omettez le format, la sauvegarde utilise "safetensors".
local ok, err = Mimir.Serialization.save("model.safetensors")
```

### Chargement

```lua
-- SafeTensors
local ok, err = Mimir.Serialization.load(
  "model.safetensors",
  "safetensors",
  { strict_mode = true }
)

-- RawFolder
local ok, err = Mimir.Serialization.load(
  "checkpoints/run1",
  "raw_folder",
  {
    load_tokenizer = true,
    load_encoder = true,
    load_optimizer = true,
    validate_checksums = true
  }
)

-- Auto-détection du format (load): omettre l'argument `format`
local ok, err = Mimir.Serialization.load("model.safetensors")
local ok, err = Mimir.Serialization.load("checkpoints/run1")
```

---

## 📊 Comparaison des Formats

| Feature               | SafeTensors | RawFolder | DebugJson |
|-----------------------|-------------|-----------|-----------|
| **Fichier unique**    | ✅          | ❌        | ✅        |
| **Lisible**           | ❌          | ✅        | ✅        |
| **Interopérable**     | ✅          | ❌        | ❌        |
| **Production**        | ✅          | ❌        | ❌        |
| **Debug**             | ❌          | ✅        | ✅        |
| **Checksums**         | ❌          | ✅        | ❌        |
| **Partial loading**   | ❌          | ✅        | ❌        |
| **Gros modèles**      | ✅          | ✅        | ❌        |
| **Git friendly**      | ❌          | ✅        | ✅        |
| **Write speed**       | ⚡⚡⚡      | ⚡        | ⚡⚡      |
| **Read speed**        | ⚡⚡⚡      | ⚡⚡      | N/A       |

---

## 🎯 Guide de Choix

### Production / Partage
→ **SafeTensors**
- Compatible écosystème ML
- Rapide
- Standard industriel

### Développement Local
→ **RawFolder**
- Inspectable
- Debuggable
- Validation par checksums

### Tests Rapides
→ **DebugJson** (petits modèles uniquement)
- Analyse immédiate
- Pas de chargement nécessaire

---

## 🔐 Sécurité & Validation

### Checksums (RawFolder)

Tous les tensors ont un checksum SHA256:
```cpp
LoadOptions opts;
opts.validate_checksums = true;  // Valide à la lecture
bool ok = load_checkpoint(model, path, opts, &error);
```

### Strict Mode

Échoue si tensors manquants:
```cpp
LoadOptions opts;
opts.strict_mode = true;  // Fail on missing tensors
bool ok = load_checkpoint(model, path, opts, &error);
```

---

## 📝 Versioning

Tous les formats incluent:
- `format_version`: Version du format
- `mimir_version`: Version de Mímir
- `git_commit`: Commit Git (optionnel)
- `created_at`: Timestamp UNIX

Exemple:
```json
{
  "format_version": "0.3.0",
  "mimir_version": "2.3.0",
  "git_commit": "abc1234",
  "created_at": 1735394400
}
```

---

## ⚠️ Limitations Actuelles

1. **DTypes (poids)**: Float32 uniquement
  - Les métadonnées (tokenizer/encoder/optimizer JSON) sont stockées en `uint8`.

2. **SafeTensors**: tokenizer/encoder/optimizer possibles
  - En option via `save_tokenizer`, `save_encoder`, `save_optimizer`.

4. **Backward compatibility**: Non garantie entre versions majeures
   - Utiliser `format_version` pour validation

---

## 🚀 Exemples Complets

### Workflow Production

```lua
-- Training
Mimir.Model.create("transformer", config)
Mimir.Model.build()
Mimir.Model.train(epochs, lr)

-- Save final model (SafeTensors)
local ok, err = Mimir.Serialization.save(
  "models/gpt_v1.safetensors",
  "safetensors",
  {
    save_tokenizer = true,
    save_encoder = true,
    save_optimizer = false
  }
)

-- Load for inference
Mimir.Model.create("transformer", config)
Mimir.Model.build()
local ok, err = Mimir.Serialization.load(
  "models/gpt_v1.safetensors",
  "safetensors",
  { load_tokenizer = true, load_encoder = true }
)
```

### Workflow Debug

```lua
-- Save checkpoint avec tout
local ok, err = Mimir.Serialization.save(
    "debug/checkpoint_epoch_10",
    "raw_folder",
    {
        save_tokenizer = true,
        save_encoder = true,
    save_optimizer = true,
        include_git_info = true
    }
)

-- Inspecter manifest
-- $ cat debug/checkpoint_epoch_10/manifest.json

-- Vérifier un tensor
-- $ xxd debug/checkpoint_epoch_10/tensors/layer1_weight_0.bin | head
-- $ cat debug/checkpoint_epoch_10/tensors/layer1_weight_0.json

-- Reload et comparer
Mimir.Model.create("transformer", config)
Mimir.Model.build()
local ok, err = Mimir.Serialization.load(
  "debug/checkpoint_epoch_10",
  "raw_folder",
  {
    load_tokenizer = true,
    load_encoder = true,
    load_optimizer = true,
    validate_checksums = true
  }
)
```

---

## 🔗 Voir Aussi

- `docs/03-API-Reference/00-API-Complete.md` - API complète Lua
- `tests/test_serialization.cpp` - Tests de validation
- `examples/save_load_demo.lua` - Exemples pratiques
- [SafeTensors Spec](https://github.com/huggingface/safetensors)

---

**Version**: 2.3.0  
**Date**: Décembre 2024  
**Status**: ✅ Production Ready
