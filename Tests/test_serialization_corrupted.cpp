#include "test_utils.hpp"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using TensorMap = std::map<std::string, std::vector<float>>;

static bool load_bin(const std::filesystem::path& p, TensorMap& out) {
    std::ifstream is(p, std::ios::binary);
    if (!is) return false;
    uint32_t n = 0;
    is.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (!is || n > (1u << 20)) return false;
    out.clear();
    return true;
}

int main() {
    const auto f = std::filesystem::path("corrupt.bin");
    {
        std::ofstream os(f, std::ios::binary);
        uint32_t bad = 0xFFFFFFFFu;
        os.write(reinterpret_cast<const char*>(&bad), sizeof(bad));
    }

    TensorMap out;
    TASSERT_TRUE(!load_bin(f, out));
    std::filesystem::remove(f);
    return 0;
}