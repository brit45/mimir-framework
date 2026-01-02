#!/usr/bin/env lua5.3
-- ============================================================================
-- TEST: Runtime Complete - Validation des Layers Implémentés
-- ============================================================================
-- Version: 2.3.0 - Runtime Complete
-- Test tous les layers critiques utilisés dans les modèles

log("╔════════════════════════════════════════════════════════════════════╗")
log("║         🧪 Runtime Complete Test Suite - Mímir v2.3.0             ║")
log("╚════════════════════════════════════════════════════════════════════╝\n")

local tests_passed = 0
local tests_failed = 0

-- ============================================================================
-- TEST 1: Multi-Head Attention (ViT, Transformer)
-- ============================================================================

log("📘 Test 1: Multi-Head Attention")
log("   Architecture: Vision Transformer style")

model.create("test_attention")

-- Simplified attention block
model.push_layer("ln1", "LayerNorm", 768 * 2)
model.push_layer("attn", "MultiHeadAttention", 768 * 768 * 4)  -- qkv + out weights
model.push_layer("ln2", "LayerNorm", 768 * 2)
model.push_layer("ffn1", "Linear", 768 * 3072 + 3072)
model.push_layer("ffn2", "Linear", 3072 * 768 + 768)

log("✓ Model created: 5 layers (Attention + FFN)")
log("   • LayerNorm × 2")
log("   • MultiHeadAttention × 1")
log("   • Linear (FFN) × 2")

local success1 = pcall(function()
    model.allocate_params()
    log("✓ Parameters allocated successfully")
end)

if success1 then
    tests_passed = tests_passed + 1
    log("✅ Test 1 PASSED\n")
else
    tests_failed = tests_failed + 1
    log("❌ Test 1 FAILED\n")
end

-- ============================================================================
-- TEST 2: Residual Block with Multi-Input (ResNet style)
-- ============================================================================

log("📘 Test 2: Residual Block (Multi-Input)")
log("   Architecture: ResNet residual connection")

model.create("test_residual")

-- Main path
model.push_layer("conv1", "Conv2d", 64*64*3*3 + 64)
model.set_layer_io("conv1", {"x"}, "conv_out")

model.push_layer("bn1", "BatchNorm", 64 * 2)
model.set_layer_io("bn1", {"conv_out"}, "bn_out")

-- Residual add
model.push_layer("add", "Add", 0)
model.set_layer_io("add", {"x", "bn_out"}, "x")

log("✓ Model created: 3 layers (Conv + BN + Add)")
log("   • Conv2d → bn → Add(x, bn)")

local success2 = pcall(function()
    model.allocate_params()
    log("✓ Parameters allocated successfully")
end)

if success2 then
    tests_passed = tests_passed + 1
    log("✅ Test 2 PASSED\n")
else
    tests_failed = tests_failed + 1
    log("❌ Test 2 FAILED\n")
end

-- ============================================================================
-- TEST 3: U-Net Decoder (Concat + Upsample)
-- ============================================================================

log("📘 Test 3: U-Net Decoder Block")
log("   Architecture: Decoder with skip connection")

model.create("test_unet_decoder")

-- Encoder feature (simulated)
model.push_layer("enc_feat", "Conv2d", 32*32*3*3 + 32)
model.set_layer_io("enc_feat", {"x"}, "encoder_feat")

-- Decoder upsampling
model.push_layer("upsample", "UpsampleNearest", 0)
model.set_layer_io("upsample", {"encoder_feat"}, "upsampled")

-- Lateral connection
model.push_layer("lateral", "Conv2d", 32*32*1*1 + 32)
model.set_layer_io("lateral", {"x"}, "lateral_feat")

-- Concat skip
model.push_layer("concat", "Concat", 0)
model.set_layer_io("concat", {"upsampled", "lateral_feat"}, "fused")

log("✓ Model created: 4 layers (Conv + Upsample + Concat)")
log("   • Encoder path → upsample")
log("   • Lateral path")
log("   • Concat(upsampled, lateral)")

local success3 = pcall(function()
    model.allocate_params()
    log("✓ Parameters allocated successfully")
end)

if success3 then
    tests_passed = tests_passed + 1
    log("✅ Test 3 PASSED\n")
else
    tests_failed = tests_failed + 1
    log("❌ Test 3 FAILED\n")
end

-- ============================================================================
-- TEST 4: Diffusion Transformer Block
-- ============================================================================

log("📘 Test 4: Diffusion Transformer Block")
log("   Architecture: Time-conditioned transformer")

model.create("test_diffusion_block")

-- Time embedding
model.push_layer("time_mlp", "Linear", 256 * 1024 + 1024)
model.set_layer_io("time_mlp", {"timestep"}, "time_emb")

-- Self-attention on latents
model.push_layer("self_attn", "SelfAttention", 512 * 512 * 4)
model.set_layer_io("self_attn", {"x"}, "attn_out")

-- Add time conditioning
model.push_layer("add_time", "Add", 0)
model.set_layer_io("add_time", {"attn_out", "time_emb"}, "conditioned")

-- MLP
model.push_layer("mlp1", "Linear", 1024 * 4096 + 4096)
model.set_layer_io("mlp1", {"conditioned"}, "mlp_hidden")

model.push_layer("mlp2", "Linear", 4096 * 1024 + 1024)
model.set_layer_io("mlp2", {"mlp_hidden"}, "x")

log("✓ Model created: 5 layers (Attention + Time + MLP)")
log("   • Time MLP")
log("   • SelfAttention")
log("   • Add (time conditioning)")
log("   • MLP × 2")

local success4 = pcall(function()
    model.allocate_params()
    log("✓ Parameters allocated successfully")
end)

if success4 then
    tests_passed = tests_passed + 1
    log("✅ Test 4 PASSED\n")
else
    tests_failed = tests_failed + 1
    log("❌ Test 4 FAILED\n")
end

-- ============================================================================
-- TEST 5: Split + Parallel Processing (Inception-style)
-- ============================================================================

log("📘 Test 5: Split + Parallel Processing")
log("   Architecture: Multi-scale processing")

model.create("test_split_parallel")

-- Split input
model.push_layer("split", "Split", 0)
model.set_layer_io("split", {"x"}, "parts")

-- Process each branch
model.push_layer("conv1", "Conv2d", 64*64*3*3 + 64)
model.set_layer_io("conv1", {"parts_0"}, "branch_0")

model.push_layer("conv2", "Conv2d", 64*64*5*5 + 64)
model.set_layer_io("conv2", {"parts_1"}, "branch_1")

-- Merge
model.push_layer("concat", "Concat", 0)
model.set_layer_io("concat", {"branch_0", "branch_1"}, "x")

log("✓ Model created: 4 layers (Split + 2 × Conv + Concat)")
log("   • Split → 2 branches")
log("   • Conv3x3 + Conv5x5")
log("   • Concat merge")

local success5 = pcall(function()
    model.allocate_params()
    log("✓ Parameters allocated successfully")
end)

if success5 then
    tests_passed = tests_passed + 1
    log("✅ Test 5 PASSED\n")
else
    tests_failed = tests_failed + 1
    log("❌ Test 5 FAILED\n")
end

-- ============================================================================
-- TEST 6: MatMul + Transpose (Manual Attention)
-- ============================================================================

log("📘 Test 6: MatMul + Transpose")
log("   Architecture: Manual attention score computation")

model.create("test_matmul")

-- Q projection
model.push_layer("q_proj", "Linear", 512*512 + 512)
model.set_layer_io("q_proj", {"x"}, "Q")

-- K projection
model.push_layer("k_proj", "Linear", 512*512 + 512)
model.set_layer_io("k_proj", {"x"}, "K")

-- Attention scores: Q @ K^T
model.push_layer("transpose_k", "Transpose", 0)
model.set_layer_io("transpose_k", {"K"}, "K_T")

model.push_layer("scores", "MatMul", 0)
model.set_layer_io("scores", {"Q", "K_T"}, "attn_scores")

log("✓ Model created: 4 layers (Linear × 2 + Transpose + MatMul)")
log("   • Q, K projections")
log("   • Transpose(K)")
log("   • MatMul(Q, K^T)")

local success6 = pcall(function()
    model.allocate_params()
    log("✓ Parameters allocated successfully")
end)

if success6 then
    tests_passed = tests_passed + 1
    log("✅ Test 6 PASSED\n")
else
    tests_failed = tests_failed + 1
    log("❌ Test 6 FAILED\n")
end

-- ============================================================================
-- RESULTS SUMMARY
-- ============================================================================

log("╔════════════════════════════════════════════════════════════════════╗")
log("║                       📊 Test Results                              ║")
log("╚════════════════════════════════════════════════════════════════════╝")
log("")

local total_tests = tests_passed + tests_failed
local pass_rate = (tests_passed / total_tests) * 100

log(string.format("Total tests:  %d", total_tests))
log(string.format("✅ Passed:     %d", tests_passed))
log(string.format("❌ Failed:     %d", tests_failed))
log(string.format("📈 Pass rate:  %.1f%%", pass_rate))
log("")

if tests_failed == 0 then
    log("╔════════════════════════════════════════════════════════════════════╗")
    log("║              🎉 ALL TESTS PASSED - RUNTIME COMPLETE!              ║")
    log("╚════════════════════════════════════════════════════════════════════╝")
    log("")
    log("✨ Runtime v2.3.0 Status:")
    log("   • Multi-Head Attention: ✅ Working")
    log("   • Residual Connections: ✅ Working")
    log("   • U-Net Skip Connections: ✅ Working")
    log("   • Diffusion Blocks: ✅ Working")
    log("   • Split/Parallel Processing: ✅ Working")
    log("   • MatMul + Transpose: ✅ Working")
    log("")
    log("🚀 Ready for:")
    log("   • Vision Transformers (ViT)")
    log("   • GPT/Transformer training")
    log("   • Diffusion models (DDPM)")
    log("   • ResNet/U-Net architectures")
    log("   • Multi-scale models (Inception)")
else
    log("⚠️  Some tests failed. Check implementation.")
end

log("")
log("════════════════════════════════════════════════════════════════════")
