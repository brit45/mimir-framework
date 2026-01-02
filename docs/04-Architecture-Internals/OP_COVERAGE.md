# Mímir Framework - Layer Operations Coverage

**Généré le**: 28 décembre 2025  
**Mode**: STRICT (aucun pass-through toléré)  
**Status**: 🚧 En refonte complète

---

## 📊 Statistiques Globales

| Catégorie | Total | Implémentés | Coverage |
|-----------|-------|-------------|----------|
| **Convolution** | 4 | 2 | ⚠️ 50% |
| **Linear/Dense** | 2 | 1 | ⚠️ 50% |
| **Embedding** | 2 | 0 | ❌ 0% |
| **Normalization** | 6 | 4 | ⚠️ 67% |
| **Activation** | 13 | 8 | ⚠️ 62% |
| **Pooling** | 6 | 4 | ⚠️ 67% |
| **Dropout** | 3 | 2 | ⚠️ 67% |
| **Shape Ops** | 7 | 3 | ⚠️ 43% |
| **Element-wise** | 4 | 2 | ⚠️ 50% |
| **Tensor Ops** | 6 | 3 | ⚠️ 50% |
| **Attention** | 3 | 2 | ⚠️ 67% |
| **Upsampling** | 4 | 2 | ⚠️ 50% |
| **Recurrent** | 3 | 0 | ❌ 0% |
| **Padding** | 3 | 0 | ❌ 0% |
| **Special** | 2 | 1 | ⚠️ 50% |
| **TOTAL** | **68** | **34** | **🔴 50%** |

---

## 🔍 Détails par Layer

### ✅ CONVOLUTION (4 layers)

| Layer | Forward | Backward | Alloc Géré | Notes |
|-------|---------|----------|------------|-------|
| Conv2d | ✅ CPU | ❌ | ❌ | std::vector non comptabilisé |
| ConvTranspose2d | ⚠️ | ❌ | ❌ | Partage code Conv2d |
| Conv1d | ❌ | ❌ | ❌ | **Pas de switch case** |
| DepthwiseConv2d | ❌ | ❌ | ❌ | **Pas de switch case** |

**Problèmes**:
- `layer_output.resize(output_size, 0.0f)` → allocation non gérée
- Conv1d et DepthwiseConv2d déclarés dans l'enum mais non implémentés

---

### ✅ LINEAR / DENSE (2 layers)

| Layer | Forward | Backward | Alloc Géré | Notes |
|-------|---------|----------|------------|-------|
| Linear | ✅ | ✅ | ❌ | Via LayerOps (vérif alloc nécessaire) |
| Bilinear | ❌ | ❌ | ❌ | **Pas de switch case** |

---

### ✅ EMBEDDING (2 layers)

| Layer | Forward | Backward | Alloc Géré | Notes |
|-------|---------|----------|------------|-------|
| Embedding | ❌ | ❌ | ❌ | Pass-through + API incompatible (needs int input) |
| EmbeddingBag | ❌ | ❌ | ❌ | **Pas de switch case** |

**Problèmes**: API actuelle est float-only, embedding needs int tokens

---

### ✅ NORMALIZATION (6 layers)

| Layer | Forward | Backward | Alloc Géré | Notes |
|-------|---------|----------|------------|-------|
| BatchNorm2d | ✅ | ❌ | ❌ | Inline naive, allocations non gérées |
| BatchNorm1d | ⚠️ | ❌ | ❌ | Partage code BatchNorm2d |
| LayerNorm | ✅ | ✅ | ❌ | Via LayerOps |
| GroupNorm | ✅ | ✅ | ❌ | Via LayerOps |
| InstanceNorm2d | ❌ | ❌ | ❌ | **Pas de switch case** |
| RMSNorm | ❌ | ❌ | ❌ | **Pas de switch case** |

---

### ✅ ACTIVATION (13 layers)

| Layer | Forward | Backward | Alloc Géré | Notes |
|-------|---------|----------|------------|-------|
| ReLU | ✅ | ✅ | ✅ | Via LayerOps (in-place possible) |
| LeakyReLU | ❌ | ❌ | ❌ | **Pas de switch case** |
| GELU | ✅ | ✅ | ✅ | Via LayerOps |
| SiLU | ✅ | ✅ | ✅ | Via LayerOps |
| Tanh | ✅ | ✅ | ✅ | Via LayerOps |
| Sigmoid | ✅ | ✅ | ✅ | Via LayerOps |
| Softmax | ✅ | ✅ | ✅ | Via LayerOps |
| LogSoftmax | ⚠️ | ❌ | ✅ | Partage code Softmax |
| Softplus | ❌ | ❌ | ❌ | **Pas de switch case** |
| Mish | ❌ | ❌ | ❌ | **Pas de switch case** |
| HardSigmoid | ❌ | ❌ | ❌ | **Pas de switch case** |
| HardSwish | ❌ | ❌ | ❌ | **Pas de switch case** |
| AlphaDropout | ❌ | ❌ | ❌ | **Pas de switch case** |

---

### ✅ POOLING (6 layers)

| Layer | Forward | Backward | Alloc Géré | Notes |
|-------|---------|----------|------------|-------|
| MaxPool2d | ✅ | ❌ | ❌ | std::vector allocation |
| AvgPool2d | ✅ | ✅ | ❌ | Via LayerOps |
| AdaptiveAvgPool2d | ✅ | ❌ | ❌ | Via global_avgpool2d |
| GlobalAvgPool2d | ✅ | ❌ | ❌ | Via LayerOps |
| MaxPool1d | ❌ | ❌ | ❌ | **Pas de switch case** |
| AvgPool1d | ❌ | ❌ | ❌ | **Pas de switch case** |

---

### ✅ DROPOUT / REGULARIZATION (3 layers)

| Layer | Forward | Backward | Alloc Géré | Notes |
|-------|---------|----------|------------|-------|
| Dropout | ✅ | ✅ | ❌ | Via LayerOps |
| Dropout2d | ⚠️ | ❌ | ❌ | Partage code Dropout |
| AlphaDropout | ❌ | ❌ | ❌ | **Pas de switch case** |

---

### ✅ SHAPE OPERATIONS (7 layers)

| Layer | Forward | Backward | Alloc Géré | Notes |
|-------|---------|----------|------------|-------|
| Flatten | ✅ | ✅ | ✅ | Via LayerOps (shallow copy) |
| Reshape | ✅ | ✅ | ✅ | Via LayerOps |
| View | ⚠️ | ✅ | ✅ | Alias de Reshape |
| Transpose | ⚠️ | ❌ | ❌ | Pass-through si dims invalides |
| Permute | ⚠️ | ❌ | ❌ | Pass-through si dims vides |
| Squeeze | ❌ | ❌ | ❌ | **Pas de switch case** |
| Unsqueeze | ❌ | ❌ | ❌ | **Pas de switch case** |

---

### ✅ ELEMENT-WISE OPERATIONS (4 layers)

| Layer | Forward | Backward | Alloc Géré | Notes |
|-------|---------|----------|------------|-------|
| Add | ✅ | ✅ | ❌ | Pass-through si < 2 inputs |
| Subtract | ❌ | ❌ | ❌ | **Pas de switch case** |
| Multiply | ✅ | ✅ | ❌ | Pass-through si < 2 inputs |
| Divide | ❌ | ❌ | ❌ | **Pas de switch case** |

---

### ✅ TENSOR OPERATIONS (6 layers)

| Layer | Forward | Backward | Alloc Géré | Notes |
|-------|---------|----------|------------|-------|
| Concat | ✅ | ✅ | ❌ | Pass-through si < 2 inputs |
| Split | ✅ | ❌ | ❌ | Pass-through si split_sizes vide |
| Chunk | ❌ | ❌ | ❌ | **Pas de switch case** |
| Stack | ❌ | ❌ | ❌ | **Pas de switch case** |
| MatMul | ✅ | ✅ | ❌ | Pass-through si dims invalides |
| BatchMatMul | ❌ | ❌ | ❌ | **Pas de switch case** |

---

### ✅ ATTENTION (3 layers)

| Layer | Forward | Backward | Alloc Géré | Notes |
|-------|---------|----------|------------|-------|
| SelfAttention | ⚠️ | ❌ | ❌ | Pass-through si weights null |
| MultiHeadAttention | ⚠️ | ❌ | ❌ | Pass-through si weights null |
| CrossAttention | ❌ | ❌ | ❌ | Pass-through (2 inputs requis) |

---

### ✅ UPSAMPLING (4 layers)

| Layer | Forward | Backward | Alloc Géré | Notes |
|-------|---------|----------|------------|-------|
| UpsampleNearest | ⚠️ | ❌ | ❌ | Pass-through si dims invalides |
| UpsampleBilinear | ⚠️ | ❌ | ❌ | Pass-through si dims invalides |
| UpsampleBicubic | ❌ | ❌ | ❌ | **Pas de switch case** |
| PixelShuffle | ❌ | ❌ | ❌ | **Pas de switch case** |

---

### ✅ RECURRENT (3 layers)

| Layer | Forward | Backward | Alloc Géré | Notes |
|-------|---------|----------|------------|-------|
| LSTM | ❌ | ❌ | ❌ | **Pas de switch case** |
| GRU | ❌ | ❌ | ❌ | **Pas de switch case** |
| RNN | ❌ | ❌ | ❌ | **Pas de switch case** |

**Note**: Si hors scope, supprimer de l'enum LayerType

---

### ✅ PADDING (3 layers)

| Layer | Forward | Backward | Alloc Géré | Notes |
|-------|---------|----------|------------|-------|
| ZeroPad2d | ❌ | ❌ | ❌ | **Pas de switch case** |
| ReflectionPad2d | ❌ | ❌ | ❌ | **Pas de switch case** |
| ReplicationPad2d | ❌ | ❌ | ❌ | **Pas de switch case** |

---

### ✅ SPECIAL (2 layers)

| Layer | Forward | Backward | Alloc Géré | Notes |
|-------|---------|----------|------------|-------|
| Identity | ✅ | ✅ | ✅ | Via LayerOps (shallow copy) |
| Lambda | ❌ | ❌ | ❌ | **Pas de switch case** - Callbacks Lua? |

---

## 🚨 Problèmes Critiques Identifiés

### 1. **Pass-through silencieux** (24 instances)
- Add, Multiply, Concat, Split, MatMul → fallback si inputs insuffisants
- Transpose, Permute → fallback si config invalide
- Attention → fallback si weights null
- Upsample → fallback si dims invalides
- **Solution**: Remplacer par erreurs + return false

### 2. **Allocations non gérées** 
```cpp
// ❌ MAUVAIS (actuel)
layer_output.resize(output_size, 0.0f);  // Pas de MemoryGuard

// ✅ BON (target)
layer_output = Mimir.Allocator.allocate_tensor(shape, dtype, "layer_N_output");
```

**Occurrences**:
- Conv2d/ConvTranspose2d (ligne 1960)
- MaxPool2d (ligne 2067)
- BatchNorm2d (ligne 2026)
- Tous les layers qui font des allocations temporaires

### 3. **Layers déclarés mais non implémentés** (34 layers)
```
Conv1d, DepthwiseConv2d, Bilinear, EmbeddingBag,
InstanceNorm2d, RMSNorm, LeakyReLU, Softplus, Mish,
HardSigmoid, HardSwish, MaxPool1d, AvgPool1d,
AlphaDropout, Squeeze, Unsqueeze, Subtract, Divide,
Chunk, Stack, BatchMatMul, UpsampleBicubic, PixelShuffle,
LSTM, GRU, RNN, ZeroPad2d, ReflectionPad2d, ReplicationPad2d,
Lambda, CrossAttention (partiel), EmbeddingBag, Bilinear
```

**Décision requise**: Implémenter OU supprimer de l'enum

---

## 📋 Plan d'Action (Ordre de Priorité)

### Phase 1: Infrastructure (URGENT)
- [ ] Créer `RuntimeAllocator` avec MemoryGuard
- [ ] API: `Tensor allocate_tensor(shape, dtype, name)`
- [ ] API: `Buffer allocate_buffer(bytes, tag)`
- [ ] RAII handles pour auto-release
- [ ] Ajouter flag `MIMIR_STRICT_MODE=1` (défaut)

### Phase 2: Suppression Pass-through (URGENT)
- [ ] Remplacer tous les `pass-through` par erreurs
- [ ] Ajouter validation pré-forward (inputs, config, weights)
- [ ] Mode strict: return false + log au lieu de fallback

### Phase 3: Migration Allocations (URGENT)
- [ ] Migrer Conv2d/ConvTranspose2d
- [ ] Migrer MaxPool2d
- [ ] Migrer BatchNorm2d/1d
- [ ] Vérifier LayerOps utilisent bien l'allocator

### Phase 4: Implémentation Layers Manquants (PAR PRIORITÉ)

**Priorité 1** (CNN essentiels):
- [ ] Conv1d
- [ ] DepthwiseConv2d
- [ ] InstanceNorm2d
- [ ] LeakyReLU
- [ ] MaxPool1d, AvgPool1d

**Priorité 2** (Tensor ops):
- [ ] Subtract, Divide
- [ ] Squeeze, Unsqueeze
- [ ] Chunk, Stack
- [ ] BatchMatMul

**Priorité 3** (Upsampling):
- [ ] UpsampleBicubic
- [ ] PixelShuffle

**Priorité 4** (Padding):
- [ ] ZeroPad2d
- [ ] ReflectionPad2d
- [ ] ReplicationPad2d

**Priorité 5** (Advanced):
- [ ] RMSNorm
- [ ] Mish, Softplus, HardSigmoid, HardSwish
- [ ] CrossAttention (complet)
- [ ] Bilinear
- [ ] EmbeddingBag
- [ ] AlphaDropout

**Hors Scope** (À SUPPRIMER si non implémentés):
- [ ] LSTM, GRU, RNN (complexe, scope?)
- [ ] Lambda (callbacks Lua = unsafe?)

### Phase 5: Tests
- [ ] Test compute par layer
- [ ] Test memory leak (100 forwards)
- [ ] Test gradient numérique
- [ ] Benchmark SIMD/AVX2

---

## 🎯 Objectif Final

**Coverage Target: 100%**
- Tous les layers de l'enum ont un case implémenté
- Aucun pass-through/fallback
- Toutes les allocations via RuntimeAllocator
- Tests automatiques pour chaque layer
- Documentation complète

**Critères de Succès**:
```bash
✅ make test_runtime_compute   # Tous les layers testés
✅ make test_memory_leak        # Aucune fuite détectée
✅ grep "pass-through" src/**   # Aucun résultat
✅ grep "std::vector<float>" src/Model.cpp  # Runtime: 0 résultats
```

---

*Mise à jour automatique à chaque commit via `scripts/generate_op_coverage.sh`*
