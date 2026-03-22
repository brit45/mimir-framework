#include "test_utils.hpp"

#include "ConfigOverrides.hpp"

using Mimir::ConfigOverrides::json;

int main() {
    using namespace Mimir::ConfigOverrides;

    // parseOverrideValue()
    {
        TASSERT_TRUE(parseOverrideValue("true").is_boolean());
        TASSERT_TRUE(parseOverrideValue("true").get<bool>() == true);
        TASSERT_TRUE(parseOverrideValue("42").is_number_integer());
        TASSERT_TRUE(parseOverrideValue("42").get<long long>() == 42);
        TASSERT_TRUE(parseOverrideValue("3.14").is_number_float());
        TASSERT_NEAR(static_cast<float>(parseOverrideValue("3.14").get<double>()), 3.14f, 1e-6f);
        TASSERT_TRUE(parseOverrideValue("{\"a\":1}").is_object());
        TASSERT_TRUE(parseOverrideValue("[1,2]").is_array());
        // Invalid JSON should fall back to string.
        TASSERT_TRUE(parseOverrideValue("{oops").is_string());
    }

    // applyOverride()
    {
        json conf = json::object();
        std::string err;
        TASSERT_TRUE(applyOverride(conf, "a.b=1", err));
        TASSERT_TRUE(conf.contains("a"));
        TASSERT_TRUE(conf["a"].contains("b"));
        TASSERT_TRUE(conf["a"]["b"].get<long long>() == 1);

        TASSERT_TRUE(applyOverride(conf, "flag=true", err));
        TASSERT_TRUE(conf["flag"].get<bool>() == true);

        TASSERT_TRUE(applyOverride(conf, "s=\"hello\"", err));
        TASSERT_TRUE(conf["s"].is_string());
        TASSERT_TRUE(conf["s"].get<std::string>() == "hello");
    }

    return 0;
}
