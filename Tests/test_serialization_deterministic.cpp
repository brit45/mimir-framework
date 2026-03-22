#include "test_utils.hpp"
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>
#include <cstdint>

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

static std::vector<char> read_all(const std::filesystem::path& p) {
    std::ifstream is(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(is), std::istreambuf_iterator<char>()};
}

int main() {
    TensorMap src{
        {"b", {3.0f, 4.0f}},
        {"a", {1.0f, 2.0f}},
    };
    const auto f1 = std::filesystem::path("det1.bin");
    const auto f2 = std::filesystem::path("det2.bin");

    TASSERT_TRUE(save_bin(f1, src));
    TASSERT_TRUE(save_bin(f2, src));

    auto v1 = read_all(f1);
    auto v2 = read_all(f2);
    TASSERT_TRUE(v1 == v2);

    std::filesystem::remove(f1);
    std::filesystem::remove(f2);
    return 0;
}