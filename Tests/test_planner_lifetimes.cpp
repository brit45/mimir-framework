#include "test_utils.hpp"

#include "Layers.hpp"
#include "Planning/Planner.hpp"

#include <unordered_map>

int main() {
    std::vector<Layer> layers;

    // l0: x -> a
    Layer l0;
    l0.name = "l0";
    l0.inputs = {"x"};
    l0.output = "a";
    layers.push_back(l0);

    // l1: a -> b
    Layer l1;
    l1.name = "l1";
    l1.inputs = {"a"};
    l1.output = "b";
    layers.push_back(l1);

    // l2: b -> a (overwrite slot)
    Layer l2;
    l2.name = "l2";
    l2.inputs = {"b"};
    l2.output = "a";
    layers.push_back(l2);

    // l3: a -> x
    Layer l3;
    l3.name = "l3";
    l3.inputs = {"a"};
    l3.output = "x";
    layers.push_back(l3);

    const auto lifetimes = Mimir::Planning::analyze_tensor_lifetimes(layers);

    auto itA = lifetimes.find("a");
    TASSERT_TRUE(itA != lifetimes.end());
    TASSERT_TRUE(itA->second.first_def == 0);
    TASSERT_TRUE(itA->second.last_use == 3);

    auto itB = lifetimes.find("b");
    TASSERT_TRUE(itB != lifetimes.end());
    TASSERT_TRUE(itB->second.first_def == 1);
    TASSERT_TRUE(itB->second.last_use == 2);

    auto itX = lifetimes.find("x");
    TASSERT_TRUE(itX != lifetimes.end());
    // x is used by l0 and defined again by l3
    TASSERT_TRUE(itX->second.first_def == 0);
    TASSERT_TRUE(itX->second.last_use == 3);

    return 0;
}
