#include "../src/Model.hpp"
#include <gtest/gtest.h>
#include <cmath>

::testing::AssertionResult FloatEquals(float a, float b, float epsilon = 1e-5f) {
    if (std::fabs(a - b) < epsilon) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << a << " != " << b;
}

// Test 1: LayerDesc structure
TEST(ModelTest, LayerDesc) {
    LayerDesc desc;
    desc.name = "layer1";
    desc.type = "linear";
    desc.paramsCount = 100;
    
    EXPECT_EQ(desc.name, "layer1");
    EXPECT_EQ(desc.type, "linear");
    EXPECT_EQ(desc.paramsCount, 100);
}

// Test 2: Optimizer construction
TEST(OptimizerTest, Basic) {
    Optimizer opt;
    opt.type = OptimizerType::ADAM;
    opt.beta1 = 0.9f;
    opt.beta2 = 0.999f;
    opt.eps = 1e-8f;
    opt.initial_lr = 1e-4f;
    
    EXPECT_EQ(opt.type, OptimizerType::ADAM);
    EXPECT_TRUE(FloatEquals(opt.beta1, 0.9f));
    EXPECT_TRUE(FloatEquals(opt.beta2, 0.999f));
}

// Test 3: Optimizer ensure
TEST(OptimizerTest, Ensure) {
    Optimizer opt;
    opt.ensure(100);
    
    EXPECT_EQ(opt.m.size(), 100);
    EXPECT_EQ(opt.v.size(), 100);
    
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_TRUE(FloatEquals(opt.m[i], 0.0f));
        EXPECT_TRUE(FloatEquals(opt.v[i], 0.0f));
    }
}

// Test 4: LR Decay - NONE
TEST(LRDecayTest, None) {
    Optimizer opt;
    opt.decay_strategy = LRDecayStrategy::NONE;
    opt.initial_lr = 1e-3f;
    opt.step = 0;
    
    EXPECT_TRUE(FloatEquals(opt.getCurrentLR(), 1e-3f));
    
    opt.step = 100;
    EXPECT_TRUE(FloatEquals(opt.getCurrentLR(), 1e-3f)); // Pas de decay
}

// Test 5: LR Decay - Warmup
TEST(LRDecayTest, Warmup) {
    Optimizer opt;
    opt.decay_strategy = LRDecayStrategy::NONE;
    opt.initial_lr = 1e-3f;
    opt.warmup_steps = 100;
    opt.step = 0;
    
    // À step=0, lr devrait être 0
    EXPECT_TRUE(FloatEquals(opt.getCurrentLR(), 0.0f));
    
    opt.step = 50;
    // À step=50 (milieu du warmup), lr = initial_lr * 0.5
    EXPECT_TRUE(FloatEquals(opt.getCurrentLR(), 5e-4f));
    
    opt.step = 100;
    // À step=100 (fin du warmup), lr = initial_lr
    EXPECT_TRUE(FloatEquals(opt.getCurrentLR(), 1e-3f));
}

// Test 6: LR Decay - COSINE
TEST(LRDecayTest, Cosine) {
    Optimizer opt;
    opt.decay_strategy = LRDecayStrategy::COSINE;
    opt.initial_lr = 1.0f;
    opt.min_lr = 0.0f;
    opt.total_steps = 1000;
    opt.warmup_steps = 0;
    opt.step = 0;
    
    float lr_start = opt.getCurrentLR();
    opt.step = 500;
    float lr_mid = opt.getCurrentLR();
    opt.step = 1000;
    float lr_end = opt.getCurrentLR();
    
    // Vérifier que le LR décroît
    EXPECT_GT(lr_start, lr_mid);
    EXPECT_GT(lr_mid, lr_end);
    EXPECT_TRUE(FloatEquals(lr_start, 1.0f));
    EXPECT_LT(lr_end, 0.1f); // Devrait être proche de min_lr
}

// Test 7: LR Decay - LINEAR
TEST(LRDecayTest, Linear) {
    Optimizer opt;
    opt.decay_strategy = LRDecayStrategy::LINEAR;
    opt.initial_lr = 1.0f;
    opt.min_lr = 0.0f;
    opt.total_steps = 100;
    opt.warmup_steps = 0;
    opt.step = 0;
    
    EXPECT_TRUE(FloatEquals(opt.getCurrentLR(), 1.0f));
    
    opt.step = 50;
    EXPECT_TRUE(FloatEquals(opt.getCurrentLR(), 0.5f, 0.01f)); // Milieu
    
    opt.step = 100;
    EXPECT_TRUE(FloatEquals(opt.getCurrentLR(), 0.0f, 0.01f)); // Fin
}

// Test 8: LR Decay - EXPONENTIAL
TEST(LRDecayTest, Exponential) {
    Optimizer opt;
    opt.decay_strategy = LRDecayStrategy::EXPONENTIAL;
    opt.initial_lr = 1.0f;
    opt.decay_rate = 0.9f;
    opt.warmup_steps = 0;
    opt.step = 0;
    
    float lr0 = opt.getCurrentLR();
    opt.step = 1;
    float lr1 = opt.getCurrentLR();
    opt.step = 2;
    float lr2 = opt.getCurrentLR();
    
    // Vérifier décroissance exponentielle
    EXPECT_GT(lr0, lr1);
    EXPECT_GT(lr1, lr2);
    EXPECT_TRUE(FloatEquals(lr0, 1.0f));
}

// Test 9: Optimizer types
TEST(OptimizerTest, Types) {
    Optimizer opt1;
    opt1.type = OptimizerType::SGD;
    EXPECT_EQ(opt1.type, OptimizerType::SGD);
    
    Optimizer opt2;
    opt2.type = OptimizerType::ADAM;
    EXPECT_EQ(opt2.type, OptimizerType::ADAM);
    
    Optimizer opt3;
    opt3.type = OptimizerType::ADAMW;
    EXPECT_EQ(opt3.type, OptimizerType::ADAMW);
}

// Test 10: AdamW weight decay
TEST(OptimizerTest, AdamWWeightDecay) {
    Optimizer opt;
    opt.type = OptimizerType::ADAMW;
    opt.weight_decay = 0.01f;
    
    EXPECT_EQ(opt.type, OptimizerType::ADAMW);
    EXPECT_TRUE(FloatEquals(opt.weight_decay, 0.01f));
}

// Test 11: Optimizer step counter
TEST(OptimizerTest, StepCounter) {
    Optimizer opt;
    EXPECT_EQ(opt.step, 0);
    
    opt.step++;
    EXPECT_EQ(opt.step, 1);
    
    opt.step += 99;
    EXPECT_EQ(opt.step, 100);
}

// Test 12: Multiple ensure calls
TEST(OptimizerTest, MultipleEnsure) {
    Optimizer opt;
    opt.ensure(50);
    EXPECT_EQ(opt.m.size(), 50);
    
    opt.ensure(100);
    EXPECT_EQ(opt.m.size(), 100);
    
    opt.ensure(75); // Ne devrait pas réduire
    EXPECT_EQ(opt.m.size(), 100);
}
