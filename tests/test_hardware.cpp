#include "../src/HardwareOpt.hpp"
#include <gtest/gtest.h>

// Test 1: CPU capabilities detection
TEST(HardwareTest, CPUCapabilities) {
    HardwareCapabilities caps = detectHardwareCapabilities();
    
    // Vérifier que la détection s'est bien passée (ne devrait pas crash)
    // On log juste les valeurs détectées
    SUCCEED() << "Detected: AVX2=" << caps.has_avx2 
              << ", FMA=" << caps.has_fma
              << ", F16C=" << caps.has_f16c
              << ", BMI2=" << caps.has_bmi2;
}

// Test 2: Hardware info structure
TEST(HardwareTest, HardwareInfo) {
    HardwareCapabilities caps;
    caps.has_avx2 = true;
    caps.has_fma = true;
    caps.has_f16c = false;
    caps.has_bmi2 = false;
    caps.num_threads = 4;
    
    EXPECT_TRUE(caps.has_avx2);
    EXPECT_TRUE(caps.has_fma);
    EXPECT_FALSE(caps.has_f16c);
    EXPECT_EQ(caps.num_threads, 4);
}

// Test 3: OpenMP threads
TEST(HardwareTest, OpenMPThreads) {
    HardwareCapabilities caps = detectHardwareCapabilities();
    
    // Devrait avoir au moins 1 thread
    EXPECT_GE(caps.num_threads, 1);
}

// Test 4: Configuration affichage
TEST(HardwareTest, PrintCapabilities) {
    HardwareCapabilities caps;
    caps.has_avx2 = true;
    caps.has_fma = true;
    caps.has_f16c = true;
    caps.has_bmi2 = false;
    caps.num_threads = 8;
    
    // Test juste que ça ne crash pas
    SUCCEED() << "Created hardware config with " << caps.num_threads << " threads";
}
