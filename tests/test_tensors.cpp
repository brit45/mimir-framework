#include "../src/tensors.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

// Test helper: float comparison avec tolerance
::testing::AssertionResult FloatEquals(float a, float b, float epsilon = 1e-5f) {
    if (std::fabs(a - b) < epsilon) {
        return ::testing::AssertionSuccess();
    } else {
        return ::testing::AssertionFailure() 
            << a << " and " << b << " differ by " << std::fabs(a - b);
    }
}

// Test 1: Construction basique
TEST(TensorTest, Construction) {
    // Default constructor
    tensor t1;
    EXPECT_TRUE(t1.data.empty());
    EXPECT_EQ(t1.Weight, 0);
    EXPECT_EQ(t1.Value, 0);
    
    // Constructor avec taille
    tensor t2(10);
    EXPECT_EQ(t2.data.size(), 10);
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(t2.data[i], 0.0f);
    }
    
    // Constructor avec données
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    tensor t3(values);
    EXPECT_EQ(t3.data.size(), 5);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(t3.data[i], values[i]);
    }
}

// Test 2: Accès aux données
TEST(TensorTest, DataAccess) {
    std::vector<float> values = {1.5f, 2.5f, 3.5f};
    tensor t(values);
    
    // getData
    float* data_ptr = t.getData();
    ASSERT_NE(data_ptr, nullptr);
    EXPECT_TRUE(FloatEquals(data_ptr[0], 1.5f));
    EXPECT_TRUE(FloatEquals(data_ptr[1], 2.5f));
    EXPECT_TRUE(FloatEquals(data_ptr[2], 3.5f));
    
    // const getData
    const tensor& t_const = t;
    const float* const_data_ptr = t_const.getData();
    ASSERT_NE(const_data_ptr, nullptr);
    EXPECT_TRUE(FloatEquals(const_data_ptr[0], 1.5f));
    
    // getSize
    EXPECT_EQ(t.getSize(), 3);
}

// Test 3: Modification des données
TEST(TensorTest, DataModification) {
    tensor t(5);
    float* data = t.getData();
    
    for (size_t i = 0; i < 5; ++i) {
        data[i] = static_cast<float>(i) * 2.0f;
    }
    
    EXPECT_TRUE(FloatEquals(t.data[0], 0.0f));
    EXPECT_TRUE(FloatEquals(t.data[1], 2.0f));
    EXPECT_TRUE(FloatEquals(t.data[2], 4.0f));
    EXPECT_TRUE(FloatEquals(t.data[3], 6.0f));
    EXPECT_TRUE(FloatEquals(t.data[4], 8.0f));
}

// Test 4: Vector4F
TEST(TensorTest, Vector4F) {
    Vector4F v1;
    EXPECT_FLOAT_EQ(v1.X, 0.0f);
    EXPECT_FLOAT_EQ(v1.Y, 0.0f);
    EXPECT_FLOAT_EQ(v1.Z, 0.0f);
    EXPECT_FLOAT_EQ(v1.W, 0.0f);
    
    Vector4F v2;
    v2.X = 1.0f;
    v2.Y = 2.0f;
    v2.Z = 3.0f;
    v2.W = 4.0f;
    
    EXPECT_TRUE(FloatEquals(v2.X, 1.0f));
    EXPECT_TRUE(FloatEquals(v2.Y, 2.0f));
    EXPECT_TRUE(FloatEquals(v2.Z, 3.0f));
    EXPECT_TRUE(FloatEquals(v2.W, 4.0f));
}

// Test 5: Copy semantics
TEST(TensorTest, CopySemantics) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f};
    tensor t1(values);
    
    // Copy constructor (implicit)
    tensor t2 = t1;
    EXPECT_EQ(t2.getSize(), 3);
    EXPECT_TRUE(FloatEquals(t2.getData()[0], 1.0f));
    EXPECT_TRUE(FloatEquals(t2.getData()[1], 2.0f));
    
    // Modification de t2 ne doit pas affecter t1
    t2.getData()[0] = 99.0f;
    EXPECT_TRUE(FloatEquals(t1.getData()[0], 1.0f));
    EXPECT_TRUE(FloatEquals(t2.getData()[0], 99.0f));
}

// Test 6: TensorSystem initialization
TEST(TensorTest, TensorSystemInit) {
    TensorSystem ts;
    bool init_result = ts.initialize();
    
    // Le test passe même si OpenCL n'est pas disponible
    // On vérifie juste que ça ne crash pas
    SUCCEED() << "TensorSystem initialization " 
              << (init_result ? "succeeded (OpenCL available)" : "failed (OpenCL not available)");
}
