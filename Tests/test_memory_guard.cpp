#include "test_utils.hpp"

#include "MemoryGuard.hpp"

int main() {
    auto& g = MemoryGuard::instance();
    g.reset();
    g.setLimit(100);

    TASSERT_TRUE(g.getCurrentBytes() == 0);

    // Basic limit enforcement.
    TASSERT_TRUE(g.requestAllocation(50, "a"));
    TASSERT_TRUE(g.getCurrentBytes() == 50);
    TASSERT_TRUE(!g.requestAllocation(60, "b"));
    TASSERT_TRUE(g.getCurrentBytes() == 50);

    g.releaseAllocation(50);
    TASSERT_TRUE(g.getCurrentBytes() == 0);

    // Freeze mode blocks allocations.
    g.freezeAllocations(true);
    TASSERT_TRUE(!g.requestAllocation(1, "freeze"));
    g.freezeAllocations(false);

    // Block mode blocks allocations.
    g.blockAllocations(true);
    TASSERT_TRUE(!g.requestAllocation(1, "block"));
    g.blockAllocations(false);

    return 0;
}
