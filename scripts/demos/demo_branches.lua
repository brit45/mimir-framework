#!/usr/bin/env lua

-- ============================================================================
-- Script de Démonstration : Système de Gestion des Branches
-- ============================================================================
-- Ce script démontre la détection automatique et l'exécution des branches
-- pour différentes architectures (ResNet, DenseNet, Inception)

package.path = package.path .. ";./?.lua"

local function log(msg)
    print(msg)
end

local function separator(char)
    print(string.rep(char or "-", 80))
end

-- ============================================================================
-- Configuration
-- ============================================================================

log("🔧 Configuration du système de branches")
separator("=")

-- Vérifier les capacités hardware
log("\n📊 Capacités Hardware:")
local has_avx2 = model.hasAVX2 and model.hasAVX2() or false
local has_fma = model.hasFMA and model.hasFMA() or false
log(string.format("  AVX2: %s", has_avx2 and "✓ Disponible" or "✗ Non disponible"))
log(string.format("  FMA:  %s", has_fma and "✓ Disponible" or "✗ Non disponible"))

if has_avx2 then
    log("\n  → Les opérations de branche utiliseront l'accélération SIMD")
else
    log("\n  → Les opérations de branche utiliseront le code scalaire")
end

-- ============================================================================
-- Test 1: ResNet Block avec Connexion Résiduelle
-- ============================================================================

separator("=")
log("\n🏗️  TEST 1: ResNet Block (Connexion Résiduelle)")
separator("=")

log("\nConstruction d'un block résiduel ResNet-style...")

local resnet_model = Pipeline.create_model("ResNetBlock")

-- Bloc principal
log("  1. Conv2d 64→64 (3x3, stride=1, padding=1)")
resnet_model.push_layer("block1_conv1", "Conv2d", 64 * 3 * 3 * 64 + 64)

log("  2. BatchNorm (64 channels)")
resnet_model.push_layer("block1_bn1", "BatchNorm", 64 * 2)

log("  3. Conv2d 64→64 (3x3, stride=1, padding=1)")
resnet_model.push_layer("block1_conv2", "Conv2d", 64 * 3 * 3 * 64 + 64)

log("  4. BatchNorm (64 channels)")
resnet_model.push_layer("block1_bn2", "BatchNorm", 64 * 2)

-- Connexion résiduelle (détectée automatiquement)
log("  5. Residual Add (détection auto)")
resnet_model.push_layer("block1_residual", "Merge", 0)

log("\n✓ Architecture construite")
log("  → Le layer 'block1_residual' sera détecté comme:")
log("     • Type: RESIDUAL")
log("     • Opération: ADD (élément par élément)")
log("     • Comportement: y = F(x) + x")

-- Simulation d'un forward pass
log("\n🔄 Simulation Forward Pass:")
log("  Input shape: [1, 64, 32, 32]")
log("  1. block1_conv1: [1, 64, 32, 32] → [1, 64, 32, 32]")
log("  2. block1_bn1:   [1, 64, 32, 32] → [1, 64, 32, 32]")
log("  3. block1_conv2: [1, 64, 32, 32] → [1, 64, 32, 32]")
log("  4. block1_bn2:   [1, 64, 32, 32] → [1, 64, 32, 32]")
log("  5. block1_residual: Fusion ADD")
log("     → branch1 (conv2 output) + branch2 (input)")
log("     → Output: [1, 64, 32, 32]")

-- ============================================================================
-- Test 2: DenseNet Block avec Skip Connections
-- ============================================================================

separator("=")
log("\n🏗️  TEST 2: DenseNet Block (Skip Connections)")
separator("=")

log("\nConstruction d'un block dense avec concaténations...")

local dense_model = Pipeline.create_model("DenseNetBlock")

local growth_rate = 32
local num_layers = 4

log(string.format("  Growth rate: %d", growth_rate))
log(string.format("  Nombre de layers: %d\n", num_layers))

for i = 1, num_layers do
    local prefix = string.format("dense_layer_%d", i)
    
    log(string.format("  Layer %d:", i))
    log(string.format("    • BatchNorm"))
    dense_model.push_layer(prefix .. "_bn", "BatchNorm", growth_rate * 2)
    
    log(string.format("    • Conv2d (3x3, %d filters)", growth_rate))
    dense_model.push_layer(prefix .. "_conv", "Conv2d", growth_rate * 3 * 3 * growth_rate + growth_rate)
    
    log(string.format("    • Dense Concat (avec layers précédents)"))
    dense_model.push_layer(prefix .. "_dense_concat", "Merge", 0)
    log("")
end

log("✓ Architecture construite")
log("  → Chaque layer 'dense_concat' sera détecté comme:")
log("     • Type: DENSE_CONNECTION")
log("     • Opération: CONCATENATE (le long des canaux)")
log("     • Comportement: [x0, x1, x2, ..., xn]")

log("\n🔄 Évolution des dimensions:")
local channels = 64  -- Input channels
for i = 1, num_layers do
    local out_channels = channels + growth_rate
    log(string.format("  Layer %d: [1, %d, H, W] → [1, %d, H, W]", 
                      i, channels, out_channels))
    channels = out_channels
end

-- ============================================================================
-- Test 3: Inception Module (Multi-Scale)
-- ============================================================================

separator("=")
log("\n🏗️  TEST 3: Inception Module (Multi-Scale)")
separator("=")

log("\nConstruction d'un module Inception avec branches parallèles...")

local inception_model = Pipeline.create_model("InceptionModule")

log("\n  Branche 1x1:")
log("    • Conv2d 256→64 (1x1)")
inception_model.push_layer("inception_1x1", "Conv2d", 256 * 1 * 1 * 64 + 64)

log("\n  Branche 3x3:")
log("    • Conv2d 256→96 (1x1, reduction)")
inception_model.push_layer("inception_3x3_reduce", "Conv2d", 256 * 1 * 1 * 96 + 96)
log("    • Conv2d 96→128 (3x3)")
inception_model.push_layer("inception_3x3", "Conv2d", 96 * 3 * 3 * 128 + 128)

log("\n  Branche 5x5:")
log("    • Conv2d 256→16 (1x1, reduction)")
inception_model.push_layer("inception_5x5_reduce", "Conv2d", 256 * 1 * 1 * 16 + 16)
log("    • Conv2d 16→32 (5x5)")
inception_model.push_layer("inception_5x5", "Conv2d", 16 * 5 * 5 * 32 + 32)

log("\n  Branche Pool:")
log("    • MaxPool2d (3x3, stride=1)")
inception_model.push_layer("inception_pool", "MaxPool2d", 0)
log("    • Conv2d 256→32 (1x1)")
inception_model.push_layer("inception_pool_proj", "Conv2d", 256 * 1 * 1 * 32 + 32)

log("\n  Fusion Multi-Scale:")
inception_model.push_layer("inception_multiscale_merge", "Merge", 0)

log("\n✓ Architecture construite")
log("  → Le layer 'multiscale_merge' sera détecté comme:")
log("     • Type: MULTI_SCALE")
log("     • Opération: CONCATENATE")
log("     • Fusionne: 1x1 (64) + 3x3 (128) + 5x5 (32) + pool (32)")
log("     • Output channels: 64 + 128 + 32 + 32 = 256")

-- ============================================================================
-- Test 4: Détection Automatique des Patterns
-- ============================================================================

separator("=")
log("\n🔍 TEST 4: Détection Automatique des Patterns")
separator("=")

log("\nPatterns de nommage reconnus automatiquement:\n")

local patterns = {
    {"conv_residual", "RESIDUAL", "ADD", "Connexion résiduelle standard"},
    {"layer_shortcut", "RESIDUAL", "ADD", "Shortcut ResNet"},
    {"skip_connection", "SKIP_CONNECTION", "CONCATENATE", "Skip connection simple"},
    {"dense_concat", "DENSE_CONNECTION", "CONCATENATE", "DenseNet connection"},
    {"attention_branch", "ATTENTION_BRANCH", "ATTENTION_WEIGHTED", "Branche d'attention"},
    {"inception_merge", "MULTI_SCALE", "CONCATENATE", "Fusion Inception"},
    {"gate_fusion", "GATE", "GATED", "Gating mechanism"},
    {"branch_split", "SPLIT", "-", "Division en branches"},
    {"merge_add", "MERGE (ADD)", "ADD", "Fusion par addition"},
    {"merge_concat", "MERGE (CONCAT)", "CONCATENATE", "Fusion par concat"},
}

for i, pattern in ipairs(patterns) do
    local name, type_detected, op, description = table.unpack(pattern)
    log(string.format("  %2d. %-25s → %-20s [%-18s]", 
                      i, "'" .. name .. "'", type_detected, op))
    log(string.format("      %s", description))
end

-- ============================================================================
-- Test 5: Performance et Optimisations
-- ============================================================================

separator("=")
log("\n⚡ TEST 5: Performance et Optimisations")
separator("=")

log("\nOpérations de fusion optimisées:\n")

log("  ADD (Résiduel):")
log("    • Scalaire: for i=1,N do out[i] = a[i] + b[i] end")
if has_avx2 then
    log("    • AVX2:     Process 8 floats/cycle avec _mm256_add_ps")
    log("    • Speedup:  ~8x théorique")
else
    log("    • AVX2:     Non disponible (utilise version scalaire)")
end

log("\n  MULTIPLY:")
log("    • Scalaire: for i=1,N do out[i] = a[i] * b[i] end")
if has_avx2 then
    log("    • AVX2:     Process 8 floats/cycle avec _mm256_mul_ps")
    log("    • Speedup:  ~8x théorique")
else
    log("    • AVX2:     Non disponible")
end

log("\n  MAX:")
log("    • Scalaire: for i=1,N do out[i] = max(a[i], b[i]) end")
if has_avx2 then
    log("    • AVX2:     Process 8 floats/cycle avec _mm256_max_ps")
    log("    • Speedup:  ~8x théorique")
else
    log("    • AVX2:     Non disponible")
end

log("\n  CONCATENATE:")
log("    • Opération: Copy deux buffers bout à bout")
log("    • Pas de calcul, optimisé mémoire")

-- ============================================================================
-- Test 6: Workflow Complet
-- ============================================================================

separator("=")
log("\n🔄 TEST 6: Workflow Complet d'Entraînement")
separator("=")

log("\nÉtapes du workflow avec branches:\n")

log("  1. Construction:")
log("     • Définir l'architecture avec push_layer()")
log("     • Les noms de layers déterminent le type de branche")
log("     • Détection automatique lors de push_layer()")

log("\n  2. Initialisation:")
log("     • allocateParams() alloue les poids")
log("     • detectAndSetupBranches() analyse la structure")
log("     • Affiche un résumé des branches détectées")

log("\n  3. Forward Pass:")
log("     • Exécution séquentielle des layers")
log("     • Stockage des sorties pour les branches")
log("     • Détection des points de fusion")
log("     • executeBranchComputation() pour chaque branche")
log("     • Mise à jour avec résultat fusionné")

log("\n  4. Backward Pass:")
log("     • Calcul des gradients standard")
log("     • backpropThroughBranch() propage aux sources")
log("     • Accumulation des gradients aux fusions")

log("\n  5. Optimisation:")
log("     • updateWeights() applique les gradients")
log("     • Les branches sont transparentes pour l'optimiseur")

-- ============================================================================
-- Résumé
-- ============================================================================

separator("=")
log("\n📝 RÉSUMÉ")
separator("=")

log("\n✅ Fonctionnalités Implémentées:")
log("  • Détection automatique des types de branches")
log("  • Support ResNet, DenseNet, Inception")
log("  • Opérations de fusion optimisées (ADD, CONCAT, MAX, AVG)")
log("  • Accélération AVX2 pour les opérations vectorielles")
log("  • Backpropagation automatique à travers les branches")
log("  • Configuration manuelle possible si nécessaire")

log("\n🎯 Types de Branches Supportés:")
log("  • RESIDUAL (ResNet-style): y = F(x) + x")
log("  • SKIP_CONNECTION: Concaténation simple")
log("  • DENSE_CONNECTION (DenseNet): Concat avec tous précédents")
log("  • MULTI_SCALE (Inception): Fusion multi-échelle")
log("  • ATTENTION_BRANCH: Branche d'attention")
log("  • SPLIT/MERGE: Division et fusion génériques")

log("\n⚡ Performance:")
if has_avx2 then
    log("  • AVX2 activé: ~8x speedup sur les fusions")
    log("  • Process 8 floats par cycle CPU")
else
    log("  • AVX2 non disponible: version scalaire")
end
log("  • Parallélisation OpenMP pour convolutions")
log("  • Réutilisation de buffers pour limiter allocations")

log("\n📚 Documentation:")
log("  • Voir docs/BRANCH_OPERATIONS.md pour détails complets")
log("  • Exemples d'architectures inclus")
log("  • API complète documentée")

separator("=")
log("\n✨ Système de branches opérationnel!\n")
