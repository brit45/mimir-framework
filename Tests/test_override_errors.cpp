#include "test_utils.hpp"

#include "ConfigOverrides.hpp"

#include <string>

int main() {
    using namespace Mimir::ConfigOverrides;

    json conf = json::object();
    std::string err;

    // Missing '='
    err.clear();
    TASSERT_TRUE(!applyOverride(conf, "a.b", err));
    TASSERT_TRUE(!err.empty());

    // Empty key segment "a..b"
    err.clear();
    TASSERT_TRUE(!applyOverride(conf, "a..b=1", err));
    TASSERT_TRUE(!err.empty());

    // Empty leaf "a.b=" is invalid (value missing)
    err.clear();
    TASSERT_TRUE(!applyOverride(conf, "a.b=", err));
    TASSERT_TRUE(!err.empty());

    // Empty leaf key "a.=1"
    err.clear();
    TASSERT_TRUE(!applyOverride(conf, "a.=1", err));
    TASSERT_TRUE(!err.empty());

    return 0;
}
