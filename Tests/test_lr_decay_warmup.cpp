#include "test_utils.hpp"
#include <algorithm>
#include <cstdint>

static float lr_with_warmup(int64_t step, int64_t warmup_steps, float base_lr) {
    if (warmup_steps > 0 && step <= warmup_steps) {
        float t = static_cast<float>(step) / static_cast<float>(warmup_steps);
        return base_lr * std::clamp(t, 0.0f, 1.0f);
    }
    return base_lr;
}

int main() {
    const int warmup = 4;
    const float base = 1.0f;

    TASSERT_NEAR(lr_with_warmup(0, warmup, base), 0.00f, 1e-6f);
    TASSERT_NEAR(lr_with_warmup(1, warmup, base), 0.25f, 1e-6f);
    TASSERT_NEAR(lr_with_warmup(2, warmup, base), 0.50f, 1e-6f);
    TASSERT_NEAR(lr_with_warmup(3, warmup, base), 0.75f, 1e-6f);
    TASSERT_NEAR(lr_with_warmup(4, warmup, base), 1.00f, 1e-6f);
    return 0;
}