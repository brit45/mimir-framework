# Mímir Framework - Guide de Migration vers Mode Strict

**Date**: 28 décembre 2025  
**Version**: 2.3.0 (Strict Mode)  
**Status**: 🚧 En cours de refonte

---

## 📋 Objectifs de la Migration

### Avant (Mode Permissif)
- ❌ Pass-through silencieux en cas d'erreur
- ❌ Allocations mémoire non comptabilisées
- ❌ 34 layers déclarés mais non implémentés
- ❌ Dépassements RAM possibles

### Après (Mode Strict)
- ✅ Erreurs explicites + arrêt si layer non implémenté
- ✅ 100% des allocations via `RuntimeAllocator`
- ✅ 100% des layers du switch-case implémentés
- ✅ Protection stricte des limites RAM
- ✅ Tests automatiques par layer

---

## 🛠️ Changements Architecturaux

### 1. RuntimeAllocator (Nouveau)

**Fichier**: `src/RuntimeAllocator.hpp`

**Principe**:
- Toute allocation runtime passe par le gestionnaire mémoire
- RAII handles pour auto-release
- Comptabilisation automatique via MemoryGuard
- Pool de buffers réutilisables (scratchpad)

**API**:
```cpp
// Dans Model::forwardPass()
RuntimeAllocator allocator(memory_guard_, max_ram_mb_);

// Allocation d'un tensor
auto output = Mimir.Allocator.allocate_tensor(
    {batch, channels, height, width},
    "float32",
    "conv1_output"
);

// Accès aux données
float* ptr = output.ptr();
std::vector<float>& data = output.data();

// Libération automatique à la fin du scope (RAII)
```

**Migration**:
```cpp
// ❌ AVANT (non comptabilisé)
std::vector<float> layer_output(output_size, 0.0f);

// ✅ APRÈS (comptabilisé)
auto output_handle = Mimir.Allocator.allocate_tensor(
    {out_channels, out_height, out_width},
    "float32",
    layer.name + "_output"
);
std::vector<float>& layer_output = output_handle.data();
```

### 2. Mode Strict (Flag Compilateur)

**Fichier**: `src/RuntimeAllocator.hpp`

**Configuration**:
```cpp
#ifndef MIMIR_STRICT_MODE
#define MIMIR_STRICT_MODE 1  // Par défaut: strict
#endif
```

**Comportement**:
- `MIMIR_STRICT_MODE=1` (défaut): Toute erreur → exception + arrêt
- `MIMIR_STRICT_MODE=0` (debug): Warning + pass-through (legacy)

**Macros utilitaires**:
```cpp
// Vérifie une condition, lance erreur si false
RUNTIME_CHECK(inputs.size() >= 2, "Add layer requires 2 inputs");

// Lance erreur en mode strict, warning en mode permissif
RUNTIME_ERROR_STRICT("Layer not implemented");
```

### 3. LayerOpsExt (Extensions)

**Fichier**: `src/LayerOpsExt.hpp`

**Contenu**:
- Implémentations des 34 layers manquants
- Organisation par catégorie (activations, convs, etc.)
- Respect des mêmes conventions que `LayerOps.hpp`

**Layers implémentés**:
```
LeakyReLU, Mish, Softplus, HardSigmoid, HardSwish,
Subtract, Divide,
Squeeze, Unsqueeze, Chunk, Stack,
InstanceNorm2d, RMSNorm,
Conv1d, DepthwiseConv2d,
MaxPool1d, AvgPool1d,
ZeroPad2d, ReflectionPad2d, ReplicationPad2d,
UpsampleBicubic, PixelShuffle
```

---

## 📝 Plan de Migration Étape par Étape

### Phase 1: Préparation (URGENT)

#### Étape 1.1: Ajouter RuntimeAllocator dans Model.hpp
```cpp
// Dans class Model:
private:
    MemoryGuard memory_guard_;
    std::unique_ptr<RuntimeAllocator> runtime_allocator_;
    size_t max_ram_mb_ = 4096;  // Configurable
    
public:
    void set_memory_limit(size_t mb) {
        max_ram_mb_ = mb;
        memory_guard_.set_limit(mb);
    }
```

#### Étape 1.2: Initialiser RuntimeAllocator dans Model()
```cpp
Model::Model() {
    // ... initializations existantes ...
    
    // Initialiser memory guard
    memory_guard_.set_limit(max_ram_mb_);
    
    // Créer runtime allocator
    runtime_allocator_ = std::make_unique<RuntimeAllocator>(
        memory_guard_,
        max_ram_mb_
    );
}
```

#### Étape 1.3: Inclure les nouveaux headers
```cpp
// En haut de Model.cpp:
#include "RuntimeAllocator.hpp"
#include "LayerOpsExt.hpp"
```

### Phase 2: Migration des Allocations Existantes

#### Étape 2.1: Migrer Conv2d/ConvTranspose2d
**Fichier**: `src/Model.cpp`, ligne ~1980

```cpp
// ❌ AVANT
const size_t output_size = out_channels * out_height * out_width;
layer_output.resize(output_size, 0.0f);

// ✅ APRÈS
auto output_handle = runtime_allocator_->allocate_tensor(
    {out_channels, out_height, out_width},
    "float32",
    layer.name + "_output"
);
std::vector<float>& layer_output = output_handle.data();
// Note: Pas besoin de .resize(), déjà alloué
```

#### Étape 2.2: Migrer MaxPool2d
**Fichier**: `src/Model.cpp`, ligne ~2090

```cpp
// ❌ AVANT
const size_t output_size = in_channels * out_height * out_width;
layer_output.resize(output_size, -std::numeric_limits<float>::infinity());

// ✅ APRÈS
auto output_handle = runtime_allocator_->allocate_tensor(
    {in_channels, out_height, out_width},
    "float32",
    layer.name + "_output"
);
std::vector<float>& layer_output = output_handle.data();
// Initialiser à -inf
std::fill(layer_output.begin(), layer_output.end(), 
          -std::numeric_limits<float>::infinity());
```

#### Étape 2.3: Migrer BatchNorm2d
**Fichier**: `src/Model.cpp`, ligne ~2027

```cpp
// ❌ AVANT
layer_output = x;  // Copie directe

// ✅ APRÈS
auto output_handle = runtime_allocator_->allocate_tensor(
    {static_cast<int>(x.size())},
    "float32",
    layer.name + "_output"
);
std::vector<float>& layer_output = output_handle.data();
layer_output = x;  // Copie comptabilisée
```

### Phase 3: Remplacement des Pass-through

#### Étape 3.1: Identifier tous les pass-through
```bash
grep -n "pass-through" src/Model.cpp
```

Résultat attendu: **24 occurrences**

#### Étape 3.2: Remplacer par erreurs strictes

**Pattern 1**: Multi-input insuffisants
```cpp
// ❌ AVANT
case LayerType::Add: {
    if (inputs.size() < 2) {
        std::cerr << "⚠️  Add requires 2 inputs - pass-through" << std::endl;
        layer_output = x;
    } else {
        layer_output = LayerOps::add_forward(*inputs[0], *inputs[1]);
    }
    break;
}

// ✅ APRÈS
case LayerType::Add: {
    RUNTIME_CHECK(
        inputs.size() >= 2,
        "Add layer '" + layer.name + "' requires 2 inputs, got " +
        std::to_string(inputs.size())
    );
    
    layer_output = LayerOps::add_forward(*inputs[0], *inputs[1]);
    break;
}
```

**Pattern 2**: Config invalide
```cpp
// ❌ AVANT
case LayerType::Transpose: {
    if (rows <= 0 || cols <= 0) {
        std::cerr << "⚠️  Transpose: dimensions not set - pass-through" << std::endl;
        layer_output = x;
    } else {
        layer_output = LayerOps::transpose_forward(x, rows, cols);
    }
    break;
}

// ✅ APRÈS
case LayerType::Transpose: {
    RUNTIME_CHECK(
        layer.in_features > 0 && layer.out_features > 0,
        "Transpose layer '" + layer.name + "' dimensions not configured. " +
        "Set in_features and out_features."
    );
    
    layer_output = LayerOps::transpose_forward(
        x, layer.in_features, layer.out_features
    );
    break;
}
```

**Pattern 3**: Weights null
```cpp
// ❌ AVANT
case LayerType::SelfAttention: {
    if (!weights) {
        std::cerr << "⚠️  Attention: weights not initialized - pass-through" << std::endl;
        layer_output = x;
    } else {
        // ... compute attention ...
    }
    break;
}

// ✅ APRÈS
case LayerType::SelfAttention: {
    RUNTIME_CHECK(
        layer.getWeights() != nullptr,
        "SelfAttention layer '" + layer.name + "' weights not initialized. " +
        "Call allocateParams() before forward pass."
    );
    
    // ... compute attention ...
    break;
}
```

### Phase 4: Ajout des Layers Manquants

#### Étape 4.1: Ajouter les cases manquants dans switch-case

**Localisation**: `src/Model.cpp`, ligne ~1953 (switch)

**Template**:
```cpp
// ============================================================
// NOUVELLES ACTIVATIONS
// ============================================================

case LayerType::LeakyReLU: {
    float alpha = layer.leaky_relu_alpha > 0 ? layer.leaky_relu_alpha : 0.01f;
    layer_output = LayerOpsExt::leaky_relu_forward(x, alpha);
    break;
}

case LayerType::Mish: {
    layer_output = LayerOpsExt::mish_forward(x);
    break;
}

case LayerType::Softplus: {
    layer_output = LayerOpsExt::softplus_forward(x);
    break;
}

case LayerType::HardSigmoid: {
    layer_output = LayerOpsExt::hard_sigmoid_forward(x);
    break;
}

case LayerType::HardSwish: {
    layer_output = LayerOpsExt::hard_swish_forward(x);
    break;
}

// ============================================================
// ELEMENT-WISE MANQUANTS
// ============================================================

case LayerType::Subtract: {
    RUNTIME_CHECK(
        inputs.size() >= 2,
        "Subtract layer requires 2 inputs, got " + std::to_string(inputs.size())
    );
    layer_output = LayerOpsExt::subtract_forward(*inputs[0], *inputs[1]);
    break;
}

case LayerType::Divide: {
    RUNTIME_CHECK(
        inputs.size() >= 2,
        "Divide layer requires 2 inputs, got " + std::to_string(inputs.size())
    );
    layer_output = LayerOpsExt::divide_forward(*inputs[0], *inputs[1]);
    break;
}

// ============================================================
// SHAPE OPS MANQUANTS
// ============================================================

case LayerType::Squeeze: {
    std::vector<int> input_shape = {static_cast<int>(x.size())};
    std::vector<int> output_shape;
    layer_output = LayerOpsExt::squeeze_forward(
        x, input_shape, output_shape, layer.squeeze_dim
    );
    break;
}

case LayerType::Unsqueeze: {
    std::vector<int> input_shape = {static_cast<int>(x.size())};
    std::vector<int> output_shape;
    RUNTIME_CHECK(
        layer.unsqueeze_dim != -999,  // Valeur par défaut invalide
        "Unsqueeze layer '" + layer.name + "' requires unsqueeze_dim parameter"
    );
    layer_output = LayerOpsExt::unsqueeze_forward(
        x, input_shape, output_shape, layer.unsqueeze_dim
    );
    break;
}

// Et ainsi de suite pour tous les layers manquants...
```

#### Étape 4.2: Ajouter paramètres Layer si nécessaire

**Fichier**: `src/Layers.hpp`

```cpp
struct Layer {
    // ... champs existants ...
    
    // Nouveaux paramètres pour layers manquants
    float leaky_relu_alpha = 0.01f;      // LeakyReLU
    int squeeze_dim = -1;                 // Squeeze (-1 = all)
    int unsqueeze_dim = -999;             // Unsqueeze (invalid par défaut)
    // ... autres si nécessaires ...
};
```

### Phase 5: Tests Automatiques

#### Étape 5.1: Créer test runtime compute
**Fichier**: `tests/test_runtime_compute.cpp`

```cpp
#include <gtest/gtest.h>
#include "../src/Model.hpp"
#include "../src/LayerTypes.hpp"

TEST(RuntimeCompute, Conv2dForward) {
    Model model;
    model.set_memory_limit(512);  // 512 MB
    
    // Ajouter layer Conv2d
    model.push("conv1", "Conv2d", 3*64*3*3);  // 3->64, kernel 3x3
    model.layers.back().in_channels = 3;
    model.layers.back().out_channels = 64;
    model.layers.back().kernel_h = 3;
    model.layers.back().stride_h = 1;
    model.layers.back().pad_h = 1;
    model.layers.back().input_height = 32;
    model.layers.back().input_width = 32;
    
    // Allouer poids
    model.allocateParams();
    model.initializeWeights("xavier");
    
    // Input: [3, 32, 32]
    std::vector<float> input(3 * 32 * 32, 1.0f);
    
    // Forward
    std::vector<float> output = model.forwardPass(input, false);
    
    // Vérifier output shape: [64, 32, 32]
    EXPECT_EQ(output.size(), 64 * 32 * 32);
    
    // Vérifier pas de NaN/Inf
    for (float val : output) {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

// Répéter pour tous les layers...
TEST(RuntimeCompute, ReLUForward) { /* ... */ }
TEST(RuntimeCompute, LinearForward) { /* ... */ }
// etc.
```

#### Étape 5.2: Test memory leak
**Fichier**: `tests/test_memory_leak.cpp`

```cpp
TEST(MemoryLeak, MultipleForwardPasses) {
    Model model;
    model.set_memory_limit(512);
    
    // Construire modèle simple
    model.push("conv1", "Conv2d", 3*32*3*3);
    model.push("relu1", "ReLU", 0);
    model.push("linear1", "Linear", 32*10);
    
    model.allocateParams();
    
    std::vector<float> input(3 * 32 * 32, 1.0f);
    
    // Baseline
    size_t initial_usage = model.getMemoryGuard().get_current_usage_mb();
    
    // 100 forwards
    for (int i = 0; i < 100; ++i) {
        std::vector<float> output = model.forwardPass(input, false);
    }
    
    // Vérifier pas de fuite
    size_t final_usage = model.getMemoryGuard().get_current_usage_mb();
    EXPECT_EQ(initial_usage, final_usage);  // Strict equality
}
```

---

## 🎯 Checklist de Migration

### Infrastructure
- [ ] ✅ RuntimeAllocator créé (`src/RuntimeAllocator.hpp`)
- [ ] ✅ LayerOpsExt créé (`src/LayerOpsExt.hpp`)
- [ ] ✅ OP_COVERAGE.md généré (`docs/OP_COVERAGE.md`)
- [ ] Intégrer RuntimeAllocator dans Model.hpp
- [ ] Initialiser dans Model()
- [ ] Ajouter includes dans Model.cpp

### Mode Strict
- [ ] Définir `MIMIR_STRICT_MODE=1` par défaut
- [ ] Remplacer 24 pass-through par `RUNTIME_CHECK`
- [ ] Ajouter validations pré-forward (inputs, config, weights)
- [ ] Supprimer tous les `std::cerr << "⚠️ "` de fallback

### Allocations
- [ ] Migrer Conv2d/ConvTranspose2d
- [ ] Migrer MaxPool2d
- [ ] Migrer BatchNorm2d/1d
- [ ] Audit LayerOps pour allocations
- [ ] Remplacer std::vector temporaires par BufferHandle

### Layers Manquants (Priorité 1)
- [ ] LeakyReLU, Mish, Softplus, HardSigmoid, HardSwish
- [ ] Subtract, Divide
- [ ] Squeeze, Unsqueeze
- [ ] Chunk, Stack, BatchMatMul
- [ ] Conv1d, DepthwiseConv2d
- [ ] InstanceNorm2d, RMSNorm
- [ ] MaxPool1d, AvgPool1d

### Layers Manquants (Priorité 2)
- [ ] ZeroPad2d, ReflectionPad2d, ReplicationPad2d
- [ ] UpsampleBicubic, PixelShuffle
- [ ] CrossAttention (complet)
- [ ] Bilinear, EmbeddingBag, AlphaDropout

### Layers Hors Scope (À décider)
- [ ] LSTM, GRU, RNN → Implémenter OU supprimer de l'enum
- [ ] Lambda → Callbacks Lua = unsafe? Supprimer?

### Tests
- [ ] test_runtime_compute.cpp (tous les layers)
- [ ] test_memory_leak.cpp (100 forwards)
- [ ] test_gradient_numerical.cpp (optionnel)
- [ ] Intégrer dans CMakeLists.txt

### Documentation
- [ ] Mettre à jour README.md
- [ ] Guide migration dans docs/
- [ ] API reference RuntimeAllocator
- [ ] Exemples Lua avec nouveaux layers

### Validation Finale
- [ ] `grep "pass-through" src/**` → 0 résultats
- [ ] `grep "std::vector<float>" src/Model.cpp` runtime → 0 allocations sauvages
- [ ] `make test` → 100% pass
- [ ] `./scripts/generate_op_coverage.sh` → 100% coverage
- [ ] Benchmark performance (avant/après)

---

## 🚀 Exécution de la Migration

### Compilation en mode strict
```bash
cd /home/brito/Documents/lab/tensor-2
mkdir -p build && cd build

# Activer mode strict
cmake -DMIMIR_STRICT_MODE=ON ..
make -j$(nproc)

# Lancer tests
make test

# Générer coverage
../scripts/generate_op_coverage.sh
```

### Désactiver temporairement (debug)
```bash
# Dans CMakeLists.txt:
add_definitions(-DMIMIR_STRICT_MODE=0)

# Ou à la compilation:
g++ -DMIMIR_STRICT_MODE=0 ...
```

---

## 📊 Métriques de Succès

| Métrique | Avant | Cible |
|----------|-------|-------|
| Layer coverage | 50% | **100%** |
| Pass-through instances | 24 | **0** |
| Unmanaged allocations | ~10+ | **0** |
| Test coverage | 0% | **80%+** |
| Memory leaks | Unknown | **0** |
| Performance | Baseline | ≥ 95% baseline |

---

## ⚠️ Risques et Mitigations

### Risque 1: Performance Overhead (RuntimeAllocator)
- **Probabilité**: Moyenne
- **Impact**: RAII + tracking = léger overhead
- **Mitigation**: 
  - Pool de buffers réutilisables (scratchpad)
  - Benchmark avant/après
  - Optimiser chemins chauds (inlining)

### Risque 2: Breaking Changes API Lua
- **Probabilité**: Faible
- **Impact**: Scripts existants cassés
- **Mitigation**:
  - API Lua inchangée (comportements internes)
  - Nouveaux layers optionnels
  - Version 2.3.0 = major bump

### Risque 3: Complexité LSTM/GRU/RNN
- **Probabilité**: Élevée
- **Impact**: Implémentation complexe (états cachés, multi-timesteps)
- **Mitigation**:
  - Décision: Implémenter OU supprimer de l'enum
  - Si implémentation: v2.4.0 (après migration strict)
  - Si suppression: documenter et proposer alternatives

---

*Document généré le 28 décembre 2025*  
*Dernière mise à jour: Phase 1 en cours*
