#include "test_utils.hpp"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using TensorMap = std::map<std::string, std::vector<float>>;

static bool save_bin(const std::filesystem::path& p, const TensorMap& m) {
    std::ofstream os(p, std::ios::binary);
    if (!os) return false;
    uint32_t n = static_cast<uint32_t>(m.size());
    os.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (const auto& [k, v] : m) {
        uint32_t len = static_cast<uint32_t>(k.size());
        uint32_t cnt = static_cast<uint32_t>(v.size());
        os.write(reinterpret_cast<const char*>(&len), sizeof(len));
        os.write(k.data(), static_cast<std::streamsize>(len));
        os.write(reinterpret_cast<const char*>(&cnt), sizeof(cnt));
        os.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(cnt * sizeof(float)));
    }
    return static_cast<bool>(os);
}

static bool load_bin(const std::filesystem::path& p, TensorMap& out) {
    std::ifstream is(p, std::ios::binary);
    if (!is) return false;
    uint32_t n = 0;
    is.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (!is) return false;
    out.clear();
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t len = 0, cnt = 0;
        is.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string k(len, '\0');
        is.read(k.data(), static_cast<std::streamsize>(len));
        is.read(reinterpret_cast<char*>(&cnt), sizeof(cnt));
        std::vector<float> v(cnt);
        is.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(cnt * sizeof(float)));
        if (!is) return false;
        out.emplace(std::move(k), std::move(v));
    }
    return true;
}

int main() {
    TensorMap src{
        {"enc.b1", {1.0f, 2.0f}},
        {"enc.w1", {0.1f, 0.2f, 0.3f}},
    };

    const std::filesystem::path f = "mimir_rt.bin";
    TASSERT_TRUE(save_bin(f, src));

    TensorMap dst;
    TASSERT_TRUE(load_bin(f, dst));
    TASSERT_TRUE(dst.size() == src.size());

    for (const auto& [k, v] : src) {
        TASSERT_TRUE(dst.count(k) == 1);
        TASSERT_TRUE(dst[k].size() == v.size());
        for (size_t i = 0; i < v.size(); ++i) TASSERT_NEAR(dst[k][i], v[i], 1e-6f);
    }

    std::filesystem::remove(f);
    return 0;
}