#include "test_utils.hpp"

#include "Models/Registry/ModelArchitectures.hpp"

int main() {
    // The registry supports SD3.5 aliases.
    auto cfg1 = ModelArchitectures::defaultConfig("sd3_5");
    auto cfg2 = ModelArchitectures::defaultConfig("SD3.5");

    TASSERT_TRUE(cfg1.is_object());
    TASSERT_TRUE(cfg2.is_object());

    // Creating via alias should work and produce canonical type.
    auto m = ModelArchitectures::create("SD3.5", cfg2);
    TASSERT_TRUE(m != nullptr);
    TASSERT_TRUE(m->modelConfig.contains("type"));
    TASSERT_TRUE(m->modelConfig["type"].get<std::string>() == "sd3_5");

    return 0;
}
