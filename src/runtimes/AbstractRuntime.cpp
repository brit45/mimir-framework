#include "runtimes/AbstractRuntime.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cstring>

namespace {
static inline bool env_flag_true(const char* name, bool default_value) {
    const char* v = std::getenv(name);
    if (!v) return default_value;
    if (v[0] == '\0') return default_value;

    // "0" / "false" / "no" / "off" => false
    if ((v[0] == '0' && v[1] == '\0')) return false;

    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s == "false" || s == "no" || s == "off") return false;
    return true;
}

static inline int env_int(const char* name, int default_value) {
    const char* v = std::getenv(name);
    if (!v || v[0] == '\0') return default_value;

    errno = 0;
    char* end = nullptr;
    long val = std::strtol(v, &end, 10);
    if (errno != 0 || end == v) return default_value;
    if (val < INT_MIN) return INT_MIN;
    if (val > INT_MAX) return INT_MAX;
    return static_cast<int>(val);
}

static inline bool env_disabled(const char* name) {
    const char* v = std::getenv(name);
    if (!v) return false;
    if (v[0] == '\0') return false;
    return !(v[0] == '0' && v[1] == '\0');
}

static inline std::string make_env_name(const char* backend_upper, const char* suffix) {
    std::string n = "MIMIR_";
    n += backend_upper;
    n += suffix;
    return n;
}
} // namespace

RuntimeConfig RuntimeConfig::fromEnv(const char* backend_upper) {
    RuntimeConfig cfg;
    cfg.backend = backend_upper ? backend_upper : "";

    cfg.verbose = env_flag_true("MIMIR_ACCEL_VERBOSE", false);

    // Désactivation explicite
    {
        std::string disable_env = "MIMIR_DISABLE_";
        disable_env += (backend_upper ? backend_upper : "");
        cfg.disabled = env_disabled(disable_env.c_str());
    }

    // Fast-path Linear
    {
        cfg.linear_enabled = env_flag_true(make_env_name(backend_upper, "_LINEAR").c_str(), false);
        cfg.linear_min_ops = env_int(make_env_name(backend_upper, "_LINEAR_MIN_OPS").c_str(), 1 << 20);
    }

    // Fast-path Conv
    {
        cfg.conv_enabled  = env_flag_true(make_env_name(backend_upper, "_CONV").c_str(), false);
        cfg.conv_min_ops  = env_int(make_env_name(backend_upper, "_CONV_MIN_OPS").c_str(), 1 << 18);
    }

    // Fast-path Normalization
    {
        cfg.norm_enabled       = env_flag_true(make_env_name(backend_upper, "_NORM").c_str(), false);
        cfg.norm_min_elements  = env_int(make_env_name(backend_upper, "_NORM_MIN_ELEMS").c_str(), 1 << 12);
    }

    // Fast-path Attention
    {
        cfg.attention_enabled  = env_flag_true(make_env_name(backend_upper, "_ATTENTION").c_str(), false);
        cfg.attention_min_ops  = env_int(make_env_name(backend_upper, "_ATTENTION_MIN_OPS").c_str(), 1 << 18);
    }

    // Device index (optionnel)
    {
        const std::string device_env = make_env_name(backend_upper, "_DEVICE");
        cfg.device_index = env_int(device_env.c_str(), 0);
    }

    return cfg;
}
