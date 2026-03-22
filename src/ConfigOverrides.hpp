#pragma once

#include "include/json.hpp"

#include <string>
#include <vector>

namespace Mimir {
namespace ConfigOverrides {

using json = nlohmann::json;

std::vector<std::string> splitString(const std::string& s, char delim);

json parseOverrideValue(const std::string& raw);

// Apply an override expression of the form: path.to.key=value
// Creates intermediate objects as needed.
bool applyOverride(json& target, const std::string& expr, std::string& err);

} // namespace ConfigOverrides
} // namespace Mimir
