#pragma once
#include <cmath>
#include <cstdlib>
#include <iostream>

inline bool nearf(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) <= eps;
}

inline void setTestEnvironment(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

inline void unsetTestEnvironment(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

// Native parity tests must exercise the requested backend even when the
// production placement policy measures the CPU as faster for a small layer.
inline void configureNativeRuntimeTest() {
    setTestEnvironment("MIMIR_PLANNER_MODE", "static");
    setTestEnvironment("MIMIR_RUNTIME_AUTOTUNE", "0");
}

#define TASSERT_TRUE(x) do { if (!(x)) { std::cerr << "FAIL: " #x "\n"; return 1; } } while (0)
#define TASSERT_NEAR(a,b,e) do { if (!nearf((a),(b),(e))) { std::cerr << "FAIL: " #a " ~= " #b "\n"; return 1; } } while (0)
