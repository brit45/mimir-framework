#pragma once
#include <cmath>
#include <iostream>

inline bool nearf(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) <= eps;
}

#define TASSERT_TRUE(x) do { if (!(x)) { std::cerr << "FAIL: " #x "\n"; return 1; } } while (0)
#define TASSERT_NEAR(a,b,e) do { if (!nearf((a),(b),(e))) { std::cerr << "FAIL: " #a " ~= " #b "\n"; return 1; } } while (0)