#include "../src/Layers.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

::testing::AssertionResult FloatEquals(float a, float b, float epsilon = 1e-5f) {
    if (std::fabs(a - b) < epsilon) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << a << " != " << b;
}

// Test 1: Layer construction
TEST(LayersTest, LayerConstruction) {
    Layer layer("test_layer", "dense", 100);
    EXPECT_EQ(layer.name, "test_layer");
    EXPECT_EQ(layer.type, "dense");
    EXPECT_EQ(layer.params_count, 100);
}

// Test 2: Activation ReLU
TEST(ActivationTest, ReLU) {
    EXPECT_TRUE(FloatEquals(Activation::relu(5.0f), 5.0f));
    EXPECT_TRUE(FloatEquals(Activation::relu(-3.0f), 0.0f));
    EXPECT_TRUE(FloatEquals(Activation::relu(0.0f), 0.0f));
    
    // Test vectoriel
    std::vector<float> data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    Activation::relu2d(data, 5, 1);
    EXPECT_TRUE(FloatEquals(data[0], 0.0f));
    EXPECT_TRUE(FloatEquals(data[1], 0.0f));
    EXPECT_TRUE(FloatEquals(data[2], 0.0f));
    EXPECT_TRUE(FloatEquals(data[3], 1.0f));
    EXPECT_TRUE(FloatEquals(data[4], 2.0f));
}

// Test 3: Activation ReLU6
TEST(ActivationTest, ReLU6) {
    EXPECT_TRUE(FloatEquals(Activation::relu6(5.0f), 5.0f));
    EXPECT_TRUE(FloatEquals(Activation::relu6(7.0f), 6.0f));
    EXPECT_TRUE(FloatEquals(Activation::relu6(-3.0f), 0.0f));
    EXPECT_TRUE(FloatEquals(Activation::relu6(6.0f), 6.0f));
}

// Test 4: Leaky ReLU
TEST(ActivationTest, LeakyReLU) {
    float alpha = 0.1f;
    EXPECT_TRUE(FloatEquals(Activation::leaky_relu(5.0f, alpha), 5.0f));
    EXPECT_TRUE(FloatEquals(Activation::leaky_relu(-10.0f, alpha), -1.0f));
    EXPECT_TRUE(FloatEquals(Activation::leaky_relu(0.0f, alpha), 0.0f));
}

// Test 5: Activation GELU
TEST(ActivationTest, GELU) {
    float result = Activation::gelu(1.0f);
    EXPECT_GT(result, 0.8f);
    EXPECT_LT(result, 0.9f); // GELU(1) ≈ 0.841
    
    result = Activation::gelu(0.0f);
    EXPECT_TRUE(FloatEquals(result, 0.0f));
    
    result = Activation::gelu(-1.0f);
    EXPECT_LT(result, 0.0f);
    EXPECT_GT(result, -0.2f); // GELU(-1) ≈ -0.159
}

// Test 6: Activation Sigmoid
TEST(ActivationTest, Sigmoid) {
    EXPECT_TRUE(FloatEquals(Activation::sigmoid(0.0f), 0.5f));
    
    float result = Activation::sigmoid(100.0f);
    EXPECT_TRUE(FloatEquals(result, 1.0f, 1e-3f)); // Proche de 1
    
    result = Activation::sigmoid(-100.0f);
    EXPECT_TRUE(FloatEquals(result, 0.0f, 1e-3f)); // Proche de 0
}

// Test 7: Activation Tanh
TEST(ActivationTest, Tanh) {
    EXPECT_TRUE(FloatEquals(Activation::tanh_act(0.0f), 0.0f));
    
    float result = Activation::tanh_act(100.0f);
    EXPECT_TRUE(FloatEquals(result, 1.0f, 1e-3f));
    
    result = Activation::tanh_act(-100.0f);
    EXPECT_TRUE(FloatEquals(result, -1.0f, 1e-3f));
}

// Test 8: Activation Swish
TEST(ActivationTest, Swish) {
    float result = Activation::swish(0.0f);
    EXPECT_TRUE(FloatEquals(result, 0.0f));
    
    result = Activation::swish(1.0f);
    EXPECT_GT(result, 0.7f);
    EXPECT_LT(result, 0.8f); // Swish(1) ≈ 0.731
}

// Test 9: Activation Softmax
TEST(ActivationTest, Softmax) {
    std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    Activation::softmax(logits);
    
    // Vérifier que la somme = 1
    float sum = 0.0f;
    for (float val : logits) {
        sum += val;
    }
    EXPECT_TRUE(FloatEquals(sum, 1.0f));
    
    // Vérifier ordre (le plus grand logit donne la plus grande proba)
    EXPECT_GT(logits[2], logits[1]);
    EXPECT_GT(logits[1], logits[0]);
}

// Test 10: Layer weights initialization
TEST(LayersTest, LayerWeights) {
    Layer layer("dense", "linear", 20);
    layer.in_features = 4;
    layer.out_features = 5;
    
    layer.weights.resize(20, 0.5f);
    layer.bias.resize(5, 0.1f);
    
    EXPECT_EQ(layer.weights.size(), 20);
    EXPECT_EQ(layer.bias.size(), 5);
    EXPECT_TRUE(FloatEquals(layer.weights[0], 0.5f));
    EXPECT_TRUE(FloatEquals(layer.bias[0], 0.1f));
}

// Test 11: Gradient storage
TEST(LayersTest, LayerGradients) {
    Layer layer("conv", "conv2d", 100);
    layer.grad_weights.resize(100, 0.0f);
    layer.grad_bias.resize(10, 0.0f);
    
    // Simuler des gradients
    layer.grad_weights[0] = 0.01f;
    layer.grad_bias[0] = 0.02f;
    
    EXPECT_TRUE(FloatEquals(layer.grad_weights[0], 0.01f));
    EXPECT_TRUE(FloatEquals(layer.grad_bias[0], 0.02f));
}

// Test 12: Batch normalization state
TEST(LayersTest, BatchNormState) {
    Layer bn("bn1", "batchnorm", 10);
    bn.running_mean.resize(10, 0.0f);
    bn.running_var.resize(10, 1.0f);
    
    // Simuler update
    bn.running_mean[0] = 0.5f;
    bn.running_var[0] = 0.8f;
    
    EXPECT_TRUE(FloatEquals(bn.running_mean[0], 0.5f));
    EXPECT_TRUE(FloatEquals(bn.running_var[0], 0.8f));
}

// Test 13: Activation type enum
TEST(LayersTest, ActivationTypes) {
    Layer layer;
    layer.activation = ActivationType::RELU;
    EXPECT_EQ(layer.activation, ActivationType::RELU);
    
    layer.activation = ActivationType::GELU;
    EXPECT_EQ(layer.activation, ActivationType::GELU);
    
    layer.activation = ActivationType::NONE;
    EXPECT_EQ(layer.activation, ActivationType::NONE);
}

// Test 14: Layer configuration
TEST(LayersTest, LayerConfig) {
    Layer conv;
    conv.kernel_size = 3;
    conv.stride = 2;
    conv.padding = 1;
    conv.dilation = 1;
    conv.groups = 1;
    conv.use_bias = true;
    
    EXPECT_EQ(conv.kernel_size, 3);
    EXPECT_EQ(conv.stride, 2);
    EXPECT_EQ(conv.padding, 1);
    EXPECT_EQ(conv.dilation, 1);
    EXPECT_EQ(conv.groups, 1);
    EXPECT_TRUE(conv.use_bias);
}
