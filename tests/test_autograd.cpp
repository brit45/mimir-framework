#include "../src/Autograd.hpp"
#include <gtest/gtest.h>
#include <cmath>

::testing::AssertionResult FloatEquals(float a, float b, float epsilon = 1e-5f) {
    if (std::fabs(a - b) < epsilon) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << a << " != " << b;
}

// Test 1: Gradients structure basique
TEST(AutogradTest, GradientsBasic) {
    Gradients grad;
    grad.add(0, 1.5f);
    grad.add(1, 2.5f);
    grad.add(0, 0.5f); // Accumulation
    
    EXPECT_TRUE(FloatEquals(grad.get(0), 2.0f));
    EXPECT_TRUE(FloatEquals(grad.get(1), 2.5f));
    EXPECT_TRUE(FloatEquals(grad.get(999), 0.0f)); // Non existant
}

// Test 2: Zero gradients
TEST(AutogradTest, ZeroGradients) {
    Gradients grad;
    grad.add(0, 5.0f);
    grad.add(1, 10.0f);
    
    grad.zero();
    
    EXPECT_TRUE(FloatEquals(grad.get(0), 0.0f));
    EXPECT_TRUE(FloatEquals(grad.get(1), 0.0f));
}

// Test 3: Gradient clipping
TEST(AutogradTest, GradientClipping) {
    Gradients grad;
    grad.add(0, 10.0f);
    grad.add(1, 10.0f);
    grad.add(2, 10.0f);
    
    // Norme = sqrt(10^2 + 10^2 + 10^2) = sqrt(300) ≈ 17.32
    grad.clip(5.0f);
    
    // Après clip, la norme devrait être ~5.0
    float norm = 0.0f;
    for (size_t i = 0; i < 3; ++i) {
        float g = grad.get(i);
        norm += g * g;
    }
    norm = std::sqrt(norm);
    
    EXPECT_TRUE(FloatEquals(norm, 5.0f, 0.1f));
}

// Test 4: MSE backward
TEST(AutogradTest, MSEBackward) {
    std::vector<float> pred = {1.0f, 2.0f, 3.0f};
    std::vector<float> target = {0.5f, 2.5f, 2.0f};
    
    auto grad = Autograd::mse_backward(pred, target);
    
    EXPECT_EQ(grad.size(), 3);
    
    // dL/dx = 2(x - target) / n
    EXPECT_TRUE(FloatEquals(grad[0], 2.0f * (1.0f - 0.5f) / 3.0f, 0.01f));
    EXPECT_TRUE(FloatEquals(grad[1], 2.0f * (2.0f - 2.5f) / 3.0f, 0.01f));
    EXPECT_TRUE(FloatEquals(grad[2], 2.0f * (3.0f - 2.0f) / 3.0f, 0.01f));
}

// Test 5: ComputationGraph structure
TEST(AutogradTest, ComputationGraph) {
    ComputationGraph graph;
    graph.sequence_length = 10;
    graph.latent_dim = 64;
    
    graph.token_embeddings.resize(10 * 64, 0.5f);
    graph.pos_encodings.resize(10 * 64, 0.1f);
    
    EXPECT_EQ(graph.token_embeddings.size(), 640);
    EXPECT_EQ(graph.pos_encodings.size(), 640);
    EXPECT_EQ(graph.sequence_length, 10);
    EXPECT_EQ(graph.latent_dim, 64);
}

// Test 6: Layer activations
TEST(AutogradTest, LayerActivations) {
    ComputationGraph graph;
    graph.layers.resize(2);
    
    graph.layers[0].input.resize(100, 1.0f);
    graph.layers[0].normed1.resize(100, 0.5f);
    graph.layers[0].attn_out.resize(100, 0.8f);
    
    EXPECT_EQ(graph.layers.size(), 2);
    EXPECT_EQ(graph.layers[0].input.size(), 100);
    EXPECT_TRUE(FloatEquals(graph.layers[0].input[0], 1.0f));
}

// Test 7: Gradient accumulation
TEST(AutogradTest, GradientAccumulation) {
    Gradients grad;
    
    // Simuler plusieurs mini-batches
    for (int batch = 0; batch < 5; ++batch) {
        grad.add(0, 0.1f);
        grad.add(1, 0.2f);
    }
    
    EXPECT_TRUE(FloatEquals(grad.get(0), 0.5f));
    EXPECT_TRUE(FloatEquals(grad.get(1), 1.0f));
}

// Test 8: Multiple gradient clips
TEST(AutogradTest, MultipleClips) {
    Gradients grad;
    grad.add(0, 100.0f);
    
    grad.clip(10.0f);
    float g1 = grad.get(0);
    
    grad.clip(5.0f);
    float g2 = grad.get(0);
    
    // Le second clip devrait réduire davantage
    EXPECT_LT(g2, g1);
    EXPECT_TRUE(FloatEquals(g2, 5.0f, 0.1f));
}

// Test 9: Gradient avec indices épars
TEST(AutogradTest, SparseGradients) {
    Gradients grad;
    grad.add(0, 1.0f);
    grad.add(100, 2.0f);
    grad.add(1000, 3.0f);
    
    EXPECT_TRUE(FloatEquals(grad.get(0), 1.0f));
    EXPECT_TRUE(FloatEquals(grad.get(100), 2.0f));
    EXPECT_TRUE(FloatEquals(grad.get(1000), 3.0f));
    EXPECT_TRUE(FloatEquals(grad.get(500), 0.0f));
}

// Test 10: ComputationGraph avec tokens
TEST(AutogradTest, GraphWithTokens) {
    ComputationGraph graph;
    graph.input_tokens = {1, 2, 3, 4, 5};
    graph.sequence_length = 5;
    graph.latent_dim = 32;
    
    EXPECT_EQ(graph.input_tokens.size(), 5);
    EXPECT_EQ(graph.input_tokens[0], 1);
    EXPECT_EQ(graph.input_tokens[4], 5);
}

// Test 11: MSE backward avec vecteurs identiques
TEST(AutogradTest, MSEBackwardZero) {
    std::vector<float> pred = {1.0f, 2.0f, 3.0f};
    std::vector<float> target = {1.0f, 2.0f, 3.0f};
    
    auto grad = Autograd::mse_backward(pred, target);
    
    EXPECT_EQ(grad.size(), 3);
    EXPECT_TRUE(FloatEquals(grad[0], 0.0f));
    EXPECT_TRUE(FloatEquals(grad[1], 0.0f));
    EXPECT_TRUE(FloatEquals(grad[2], 0.0f));
}

// Test 12: Clip avec norme déjà petite
TEST(AutogradTest, ClipNoEffect) {
    Gradients grad;
    grad.add(0, 0.1f);
    grad.add(1, 0.1f);
    
    float before = grad.get(0);
    grad.clip(10.0f); // Norme << 10, pas d'effet
    float after = grad.get(0);
    
    EXPECT_TRUE(FloatEquals(before, after));
}
