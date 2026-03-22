#include "ConfigOverrides.hpp"

#include <cerrno>
#include <cstdlib>

namespace Mimir {
namespace ConfigOverrides {

std::vector<std::string> splitString(const std::string& s, char delim)
{
    std::vector<std::string> parts;
    std::string cur;
    cur.reserve(s.size());
    for (char ch : s) {
        if (ch == delim) {
            parts.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(ch);
        }
    }
    parts.push_back(cur);
    return parts;
}

json parseOverrideValue(const std::string& raw)
{
    if (raw == "true") return true;
    if (raw == "false") return false;
    if (raw == "null") return nullptr;

    if (!raw.empty() && (raw.front() == '{' || raw.front() == '[' || raw.front() == '"')) {
        try {
            return json::parse(raw);
        } catch (...) {
            return raw;
        }
    }

    {
        char* end = nullptr;
        errno = 0;
        long long v = std::strtoll(raw.c_str(), &end, 10);
        if (errno == 0 && end && *end == '\0') {
            return v;
        }
    }
    {
        char* end = nullptr;
        errno = 0;
        double v = std::strtod(raw.c_str(), &end);
        if (errno == 0 && end && *end == '\0') {
            return v;
        }
    }

    return raw;
}

bool applyOverride(json& target, const std::string& expr, std::string& err)
{
    const auto eq = expr.find('=');
    if (eq == std::string::npos || eq == 0 || eq + 1 >= expr.size()) {
        err = "override invalide (attendu: path.to.key=value): " + expr;
        return false;
    }

    const std::string path = expr.substr(0, eq);
    const std::string raw_value = expr.substr(eq + 1);
    const auto keys = splitString(path, '.');
    if (keys.empty()) {
        err = "override invalide (path vide): " + expr;
        return false;
    }

    json* cur = &target;
    for (size_t i = 0; i + 1 < keys.size(); ++i) {
        const std::string& key = keys[i];
        if (key.empty()) {
            err = "override invalide (segment vide): " + expr;
            return false;
        }
        if (!cur->contains(key) || !(*cur)[key].is_object()) {
            (*cur)[key] = json::object();
        }
        cur = &(*cur)[key];
    }

    const std::string& leaf = keys.back();
    if (leaf.empty()) {
        err = "override invalide (clé finale vide): " + expr;
        return false;
    }

    (*cur)[leaf] = parseOverrideValue(raw_value);
    return true;
}

} // namespace ConfigOverrides
} // namespace Mimir
