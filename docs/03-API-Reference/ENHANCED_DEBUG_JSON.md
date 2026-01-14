# Enhanced Debug JSON Format v1.1.0

## Vue d'ensemble

Le format **Enhanced Debug JSON v1.1.0** est une amélioration majeure du format DebugJson pour le diagnostic d'entraînement. Il ajoute des informations essentielles manquantes dans la v1.0.0 :

- ✅ **Layer configs** : Hyperparamètres par layer (Conv2d stride/padding, Linear dims, etc.)
- ✅ **Real tensor shapes** : Shapes multi-dimensionnelles `[outC, inC, kH, kW]` au lieu de `{size}`
- ✅ **Gradients** : Dump optionnel des gradients pour valider backward pass
- ✅ **Weight deltas** : Détection des changements de poids avant/après optimizer step
- ✅ **Optimizer state** : Section légère avec type, learning rate, step
- ✅ **Conditional sections** : Tokenizer/encoder seulement si présents dans le modèle
- ✅ **Checksums** : Détection rapide de changements via sampling
- ✅ **Feature flags** : Indique quelles fonctionnalités sont activées

## API Lua

### Mimir.save_enhanced_debug()

```lua
local success = Mimir.save_enhanced_debug(path, options)
```

**Arguments** :
- `path` (string) : Chemin du fichier JSON de sortie
- `options` (table, optionnel) : Configuration

**Options disponibles** :

```lua
{
    include_gradients = false,         -- Inclure les gradients des tensors
    include_optimizer_state = false,   -- Inclure la section optimizer
    max_values_per_tensor = 20,        -- Nombre max de valeurs à dumper par tensor
    include_activations = false,       -- Inclure les activations (future)
    include_checksums = false,         -- Inclure les checksums pour détection de changements
    include_weight_deltas = false,     -- Inclure les deltas de poids (requires snapshot)
    include_git_info = true,           -- Inclure commit/branch git
    save_tokenizer = false,            -- Inclure la section tokenizer si présent
    save_encoder = false               -- Inclure la section encoder si présent
}
```

**Retour** :
- `true` si succès
- `false` si échec

## Exemple d'utilisation

### Diagnostic d'entraînement

```lua
-- Créer un modèle
local cfg, err = Mimir.Architectures.default_config("transformer")
assert(cfg, err)
cfg.vocab_size = 1000
cfg.d_model = 128
cfg.num_layers = 4
cfg.num_heads = 8

assert(Mimir.Model.create("transformer", cfg))

model.allocate_params()
model.init_weights("xavier")

-- Forward + backward
local input = {1, 2, 3, 4, 5}
local output = model.forward(input)
local loss_grad = model.loss_gradient(output, {1, 0, 0, 0, 0}, "mse")
model.backward(loss_grad)

-- Snapshot AVANT optimizer step
Mimir.Serialization.save_enhanced_debug("/tmp/before.json", {
    include_gradients = true,
    include_checksums = true,
    include_weight_deltas = false  -- Pas encore de snapshot précédent
})

-- Optimizer step
model.optimizer_step(0.001)

-- Snapshot APRÈS optimizer step (avec weight deltas)
Mimir.save_enhanced_debug("/tmp/after.json", {
    include_gradients = true,
    include_optimizer_state = true,
    include_checksums = true,
    include_weight_deltas = true   -- Compare avec snapshot précédent
})
```

### Vérifier si les poids ont changé

```lua
local json = read_json("/tmp/after.json")

for _, layer in ipairs(json.layers) do
    if layer.weight_delta then
        if layer.weight_delta.changed then
            print(string.format("Layer '%s' poids modifiés:", layer.name))
            print(string.format("  L2 norm delta: %.6f", layer.weight_delta.delta_l2_norm))
            print(string.format("  Max abs delta: %.6f", layer.weight_delta.delta_max_abs))
            print(string.format("  Relative change: %.2f%%", layer.weight_delta.relative_change * 100))
        else
            print(string.format("Layer '%s' UNCHANGED (problème optimizer?)", layer.name))
        end
    end
end
```

### Vérifier les gradients

```lua
local json = read_json("/tmp/debug.json")

for _, layer in ipairs(json.layers) do
    if layer.tensors then
        for _, tensor in ipairs(layer.tensors) do
            if tensor.gradients then
                if tensor.gradients.all_zero then
                    print(string.format("⚠️  WARNING: Gradients de '%s' tous à zéro!", tensor.name))
                else
                    print(string.format("✓ Gradients de '%s' OK (L2=%.6f)", 
                        tensor.name, tensor.gradients.l2_norm))
                end
            end
        end
    end
end
```

## Structure du fichier JSON v1.1.0

```json
{
    "format_version": "1.1.0",
    "timestamp": 1735123456,
    "model_name": "test_conv",
    "total_params": 1022196,
    "num_layers": 16,
    
    "features": [
        "layer_config",
        "real_shapes",
        "gradients",
        "weight_deltas",
        "optimizer_state",
        "checksums"
    ],
    
    "optimizer": {
        "type": "adam",
        "step": 42,
        "lr": 0.001
    },
    
    "layers": [
        {
            "index": 0,
            "name": "conv1",
            "type": "Conv2d",
            "params_count": 432,
            
            "config": {
                "in_channels": 3,
                "out_channels": 16,
                "kernel_h": 3,
                "kernel_w": 3,
                "stride_h": 1,
                "stride_w": 1,
                "pad_h": 1,
                "pad_w": 1,
                "dilation": 1,
                "groups": 1,
                "has_bias": true
            },
            
            "weight_checksum": "12345678901234567",
            "bias_checksum": "98765432109876543",
            
            "weight_delta": {
                "changed": true,
                "delta_l2_norm": 0.023456,
                "delta_max_abs": 0.001234,
                "relative_change": 0.000123
            },
            
            "tensors": [
                {
                    "name": "conv1.weight",
                    "dtype": "F32",
                    "shape": [16, 3, 3, 3],
                    "total_elements": 432,
                    
                    "stats": {
                        "min": -0.5,
                        "max": 0.5,
                        "mean": 0.001,
                        "std": 0.123,
                        "l2_norm": 1.234
                    },
                    
                    "sample_values": [0.1, 0.2, -0.1, 0.05, ...],
                    "sample_size": 20,
                    "truncated": true,
                    
                    "gradients": {
                        "min": -0.001,
                        "max": 0.001,
                        "mean": 0.00001,
                        "std": 0.0005,
                        "l2_norm": 0.045,
                        "all_zero": false
                    },
                    
                    "gradient_sample": [-0.0001, 0.0002, ...]
                }
            ]
        }
    ],
    
    "tokenizer": {
        "vocab_size": 50000,
        "has_vocab": true
    },
    
    "encoder": {
        "has_encoder": true,
        "architecture": "ViT"
    },
    
    "git": {
        "commit": "abc123def456",
        "branch": "main"
    }
}
```

## Cas d'usage

### 1. Debugging "poids ne changent pas"

Si après optimizer step les poids ne changent pas :

```bash
# Snapshot avant
./bin/mimir --lua train.lua  # avec save_enhanced_debug AVANT optimizer step

# Snapshot après  
./bin/mimir --lua train.lua  # avec save_enhanced_debug APRÈS optimizer step

# Vérifier
jq '.layers[].weight_delta.changed' /tmp/after.json
```

Si tous `false` → problème avec optimizer, learning rate trop petit, ou gradients à zéro.

### 2. Debugging "gradients vanishing"

```lua
local json = read_json("/tmp/debug.json")
for _, layer in ipairs(json.layers) do
    for _, tensor in ipairs(layer.tensors or {}) do
        if tensor.gradients and tensor.gradients.l2_norm < 1e-8 then
            print("⚠️  Gradients vanishing dans: " .. tensor.name)
        end
    end
end
```

### 3. Vérifier hyperparamètres d'un layer

```bash
jq '.layers[] | select(.name == "conv1") | .config' /tmp/debug.json
```

Retourne :
```json
{
  "in_channels": 3,
  "out_channels": 16,
  "kernel_h": 3,
  "kernel_w": 3,
  "stride_h": 1,
  "stride_w": 1,
  "pad_h": 1,
  "pad_w": 1,
  "dilation": 1,
  "groups": 1,
  "has_bias": true
}
```

### 4. Comparer deux checkpoints

```bash
# Extraire les shapes
jq '.layers[] | {name, tensors: [.tensors[].shape]}' before.json > shapes_before.txt
jq '.layers[] | {name, tensors: [.tensors[].shape]}' after.json > shapes_after.txt

diff shapes_before.txt shapes_after.txt
```

## Différences avec v1.0.0

| Feature | v1.0.0 | v1.1.0 |
|---------|--------|--------|
| Layer configs | ❌ | ✅ Hyperparamètres complets |
| Tensor shapes | `{size}` (1D) | `[outC, inC, kH, kW]` (multi-dim) |
| Gradients | ❌ | ✅ Optionnel avec stats |
| Weight deltas | ❌ | ✅ Avant/après optimizer |
| Optimizer state | ❌ | ✅ Type, LR, step |
| Tokenizer/encoder | Toujours | Conditionnel |
| Checksums | ❌ | ✅ Détection changements |
| Feature flags | ❌ | ✅ Indication capacités |

## Performance

- **Taille fichier** : ~5-20 KB pour petit modèle (1M params)
- **Temps sauvegarde** : < 100 ms
- **Overhead mémoire** : < 1 MB (snapshots pour deltas)

## Limites

- **Pas pour production** : Format debug uniquement
- **Truncation** : Seules les premières `max_values_per_tensor` valeurs sont dumpées
- **Snapshots** : Les weight deltas nécessitent de garder le snapshot précédent en mémoire

## API C++ (avancée)

```cpp
#include "Serialization/DebugJsonDump.hpp"

using namespace Mimir::Serialization;

DebugJsonOptions options;
options.include_gradients = true;
options.include_weight_deltas = true;
options.max_values_per_tensor = 50;

DebugJsonDump dumper;
bool success = dumper.save_enhanced(
    "/tmp/debug.json",
    model,
    options
);
```

## Références

- [SAVE_LOAD.md](SAVE_LOAD.md) : API de sérialisation générale
- [test_enhanced_minimal.lua](../../scripts/tests/test_enhanced_minimal.lua) : Test complet
- [DebugJsonDump.hpp](../../src/Serialization/DebugJsonDump.hpp) : Implémentation C++

## Changelog

### v1.1.0 (2024-12)
- ✅ Ajout layer configs avec hyperparamètres
- ✅ Shapes multi-dimensionnelles
- ✅ Support gradients optionnel
- ✅ Weight deltas (avant/après optimizer)
- ✅ Section optimizer
- ✅ Sections conditionnelles (tokenizer/encoder)
- ✅ Checksums pour détection changements
- ✅ Feature flags

### v1.0.0 (2024-11)
- Version initiale (basique)
- Tensor dumps avec truncation
- Statistics (min/max/mean/std)
