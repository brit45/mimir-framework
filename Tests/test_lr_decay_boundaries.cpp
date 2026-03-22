#include "test_utils.hpp"
#include <cstdint>

static float lr_with_warmup(int64_t step, int64_t warmup_steps, float base_lr) {
    if (warmup_steps > 0 && step <= warmup_steps) {
        return base_lr * (static_cast<float>(step) / static_cast<float>(warmup_steps));
    }
    return base_lr;
}

int main() {
    TASSERT_NEAR(lr_with_warmup(0, 0, 0.1f), 0.1f, 1e-6f);
    TASSERT_NEAR(lr_with_warmup(5, 0, 0.1f), 0.1f, 1e-6f);
    TASSERT_NEAR(lr_with_warmup(10, 5, 0.2f), 0.2f, 1e-6f);
    return 0;
}