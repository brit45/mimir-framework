#include "test_utils.hpp"
#include "runtimes/vulkan/VulkanCompute.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

int main() {
    const fs::path previous = fs::current_path();
    const fs::path root = fs::temp_directory_path() /
        ("mimir-vulkan-paths-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root / "cwd");
    struct Cleanup {
        fs::path previous, root;
        ~Cleanup() { fs::current_path(previous); fs::remove_all(root); }
    } cleanup{previous, root};
    fs::current_path(root / "cwd");
    try {
        using Finder = std::optional<std::string>(*)();
        const std::pair<const char*, Finder> shaders[] = {
            {"linear_forward.comp.spv", VulkanCompute::find_shader_path_linear},
            {"elementwise.comp.spv", VulkanCompute::find_shader_path_elementwise},
            {"add_forward.comp.spv", VulkanCompute::find_shader_path_add},
            {"mul_forward.comp.spv", VulkanCompute::find_shader_path_mul},
            {"relu_forward.comp.spv", VulkanCompute::find_shader_path_relu},
            {"silu_forward.comp.spv", VulkanCompute::find_shader_path_silu},
            {"gelu_forward.comp.spv", VulkanCompute::find_shader_path_gelu},
            {"sigmoid_forward.comp.spv", VulkanCompute::find_shader_path_sigmoid},
            {"tanh_forward.comp.spv", VulkanCompute::find_shader_path_tanh},
            {"conv2d_forward.comp.spv", VulkanCompute::find_shader_path_conv2d},
            {"conv_transpose2d_forward.comp.spv", VulkanCompute::find_shader_path_conv_transpose2d},
        };
        auto check = [](bool condition, const char* message) {
            if (!condition) throw std::runtime_error(message);
        };
        auto write = [](const fs::path& path) {
            fs::create_directories(path.parent_path());
            std::ofstream file(path, std::ios::binary); file << "fixture";
            if (!file) throw std::runtime_error("Cannot create shader path fixture");
        };
        setTestEnvironment("MIMIR_VULKAN_SHADER_DIR", "");
        setTestEnvironment("MIMIR_VULKAN_LINEAR_SPV", "");
#if defined(__linux__) && defined(MIMIR_TEST_SHADER_DIR)
        for (const auto& shader : shaders) {
            const auto found = shader.second();
            check(found && fs::equivalent(*found, fs::path(MIMIR_TEST_SHADER_DIR) / shader.first),
                  "Generated shaders must resolve from an unrelated cwd");
        }
        std::cout << "VULKAN_BUILD_ASSETS_OK eleven executable-relative shaders\n";
#endif
        const fs::path custom = root / "custom assets";
        for (const auto& shader : shaders) write(custom / shader.first);
        setTestEnvironment("MIMIR_VULKAN_SHADER_DIR", custom.string().c_str());
        for (const auto& shader : shaders) {
            const auto found = shader.second();
            check(found && fs::equivalent(*found, custom / shader.first), "Shader directory override ignored");
        }
        const fs::path linear = root / "custom-linear.spv";
        write(linear); setTestEnvironment("MIMIR_VULKAN_LINEAR_SPV", linear.string().c_str());
        const auto found_linear = VulkanCompute::find_shader_path_linear();
        check(found_linear && fs::equivalent(*found_linear, linear), "Linear override must keep priority");
        setTestEnvironment("MIMIR_VULKAN_SHADER_DIR", "");
        const char* legacy = "mimir-lookup-fixture.spv";
        fs::create_directories(root / "cwd/bin/shaders" / legacy); // A directory is not a shader.
        const fs::path legacy_file = root / "cwd/shaders" / legacy;
        write(legacy_file);
        const auto found_legacy = VulkanCompute::find_shader_path(legacy);
        check(found_legacy && fs::equivalent(*found_legacy, legacy_file), "Legacy relative lookup failed");
        check(!VulkanCompute::find_shader_path("mimir-lookup-missing.spv"), "Missing shader accepted");
        std::cout << "VULKAN_SHADER_PATHS_OK directory/Linear overrides, legacy paths\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
