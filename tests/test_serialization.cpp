/**
 * Test du module de sérialisation
 * 
 * Teste les 3 formats:
 * - SafeTensors
 * - RawFolder
 * - DebugJson
 */

#include "../src/Serialization/Serialization.hpp"
#include "../src/Model.hpp"
#include "../src/Tokenizer.hpp"
#include "../src/Encoder.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;
using namespace Mimir::Serialization;

// Helper: compare two float arrays
bool compare_floats(const float* a, const float* b, size_t size, float tolerance = 1e-5f) {
    for (size_t i = 0; i < size; ++i) {
        if (std::abs(a[i] - b[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": " << a[i] << " != " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Helper: create a simple test model
std::unique_ptr<Model> create_test_model() {
    auto model = std::make_unique<Model>();
    model->setName("test_model");
    
    // Add some layers
    model->push("layer1", "Linear", 128);
    model->push("layer2", "Linear", 256);
    model->push("layer3", "Linear", 64);
    
    // Allocate parameters
    model->allocateParams();
    
    // Initialize with test data
    auto& params = model->getMutableParams();
    for (size_t i = 0; i < params.size(); ++i) {
        auto& param = params[i];
        for (size_t j = 0; j < param.data.size(); ++j) {
            param.data[j] = static_cast<float>(i * 1000 + j) * 0.001f;
        }
    }
    
    // Setup tokenizer
    Tokenizer tokenizer(1000);

    model->setTokenizer(tokenizer);
    
    return model;
}

// Test 1: SafeTensors format
bool test_safetensors() {
    std::cout << "\n=== Test 1: SafeTensors Format ===" << std::endl;
    
    // Create test model
    auto model1 = create_test_model();
    
    // Save
    std::string path = "/tmp/mimir_test_safetensors.st";
    SaveOptions save_opts;
    save_opts.format = CheckpointFormat::SafeTensors;
    save_opts.save_tokenizer = false;  // SafeTensors is for weights only
    
    std::string error;
    bool saved = save_checkpoint(*model1, path, save_opts, &error);
    if (!saved) {
        std::cerr << "❌ Failed to save SafeTensors: " << error << std::endl;
        return false;
    }
    std::cout << "✓ Saved to " << path << std::endl;
    
    // Check file exists
    if (!fs::exists(path)) {
        std::cerr << "❌ File not created" << std::endl;
        return false;
    }
    
    size_t file_size = fs::file_size(path);
    std::cout << "✓ File size: " << file_size << " bytes" << std::endl;
    
    // Load
    auto model2 = create_test_model();  // Create with same structure
    LoadOptions load_opts;
    load_opts.format = CheckpointFormat::SafeTensors;
    
    bool loaded = load_checkpoint(*model2, path, load_opts, &error);
    if (!loaded) {
        std::cerr << "❌ Failed to load SafeTensors: " << error << std::endl;
        return false;
    }
    std::cout << "✓ Loaded successfully" << std::endl;
    
    // Compare parameters
    auto& params1 = model1->getMutableParams();
    auto& params2 = model2->getMutableParams();
    
    if (params1.size() != params2.size()) {
        std::cerr << "❌ Parameter count mismatch: " << params1.size() << " != " << params2.size() << std::endl;
        return false;
    }
    
    for (size_t i = 0; i < params1.size(); ++i) {
        if (params1[i].data.size() != params2[i].data.size()) {
            std::cerr << "❌ Param " << i << " size mismatch" << std::endl;
            return false;
        }
        
        if (!compare_floats(params1[i].data.data(), params2[i].data.data(), params1[i].data.size())) {
            std::cerr << "❌ Param " << i << " data mismatch" << std::endl;
            return false;
        }
    }
    
    std::cout << "✓ All parameters match" << std::endl;
    
    // Cleanup
    fs::remove(path);
    
    std::cout << "✅ SafeTensors test PASSED" << std::endl;
    return true;
}

// Test 2: RawFolder format
bool test_raw_folder() {
    std::cout << "\n=== Test 2: RawFolder Format ===" << std::endl;
    
    // Create test model
    auto model1 = create_test_model();
    
    // Save
    std::string path = "/tmp/mimir_test_raw_folder";
    SaveOptions save_opts;
    save_opts.format = CheckpointFormat::RawFolder;
    save_opts.save_tokenizer = true;
    save_opts.save_encoder = false;
    
    std::string error;
    bool saved = save_checkpoint(*model1, path, save_opts, &error);
    if (!saved) {
        std::cerr << "❌ Failed to save RawFolder: " << error << std::endl;
        return false;
    }
    std::cout << "✓ Saved to " << path << std::endl;
    
    // Check structure
    if (!fs::exists(fs::path(path) / "manifest.json")) {
        std::cerr << "❌ manifest.json not found" << std::endl;
        return false;
    }
    std::cout << "✓ manifest.json exists" << std::endl;
    
    if (!fs::exists(fs::path(path) / "model" / "architecture.json")) {
        std::cerr << "❌ architecture.json not found" << std::endl;
        return false;
    }
    std::cout << "✓ architecture.json exists" << std::endl;
    
    if (!fs::exists(fs::path(path) / "tensors")) {
        std::cerr << "❌ tensors directory not found" << std::endl;
        return false;
    }
    std::cout << "✓ tensors directory exists" << std::endl;
    
    // Count tensor files
    size_t tensor_count = 0;
    for (const auto& entry : fs::directory_iterator(fs::path(path) / "tensors")) {
        if (entry.path().extension() == ".bin") {
            tensor_count++;
        }
    }
    std::cout << "✓ Found " << tensor_count << " tensor files" << std::endl;
    
    // Load
    Model model2;  // Empty model
    LoadOptions load_opts;
    load_opts.format = CheckpointFormat::RawFolder;
    load_opts.load_tokenizer = true;
    
    bool loaded = load_checkpoint(model2, path, load_opts, &error);
    if (!loaded) {
        std::cerr << "❌ Failed to load RawFolder: " << error << std::endl;
        return false;
    }
    std::cout << "✓ Loaded successfully" << std::endl;
    
    // Compare
    if (model1->getLayers().size() != model2.getLayers().size())
    {
        std::cerr << "❌ Layer count mismatch" << std::endl;
        return false;
    }
    std::cout << "✓ Layer count matches: " << model2.getLayers().size() << std::endl;

    auto& params1 = model1->getMutableParams();
    auto& params2 = model2.getMutableParams();
    
    if (params1.size() != params2.size()) {
        std::cerr << "❌ Parameter count mismatch" << std::endl;
        return false;
    }
    
    for (size_t i = 0; i < params1.size(); ++i) {
        if (!compare_floats(params1[i].data.data(), params2[i].data.data(), params1[i].data.size())) {
            std::cerr << "❌ Param " << i << " data mismatch" << std::endl;
            return false;
        }
    }
    
    std::cout << "✓ All parameters match" << std::endl;
    
    // Cleanup
    fs::remove_all(path);
    
    std::cout << "✅ RawFolder test PASSED" << std::endl;
    return true;
}

// Test 3: DebugJson format
bool test_debug_json() {
    std::cout << "\n=== Test 3: DebugJson Format ===" << std::endl;
    
    // Create test model
    auto model = create_test_model();
    
    // Save
    std::string path = "/tmp/mimir_test_debug.json";
    SaveOptions save_opts;
    save_opts.format = CheckpointFormat::DebugJson;
    save_opts.debug_max_values = 50;
    
    std::string error;
    bool saved = save_checkpoint(*model, path, save_opts, &error);
    if (!saved) {
        std::cerr << "❌ Failed to save DebugJson: " << error << std::endl;
        return false;
    }
    std::cout << "✓ Saved to " << path << std::endl;
    
    // Check file exists and is valid JSON
    if (!fs::exists(path)) {
        std::cerr << "❌ File not created" << std::endl;
        return false;
    }
    
    size_t file_size = fs::file_size(path);
    std::cout << "✓ File size: " << file_size << " bytes" << std::endl;
    
    // Try to parse JSON
    std::ifstream file(path);
    json debug_json;
    try {
        file >> debug_json;
        std::cout << "✓ Valid JSON" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "❌ Invalid JSON: " << e.what() << std::endl;
        return false;
    }
    
    // Check structure
    if (!debug_json.contains("format")) {
        std::cerr << "❌ Missing 'format' field" << std::endl;
        return false;
    }
    std::cout << "✓ Format: " << debug_json["format"] << std::endl;
    
    if (!debug_json.contains("model")) {
        std::cerr << "❌ Missing 'model' field" << std::endl;
        return false;
    }
    std::cout << "✓ Model info present" << std::endl;
    
    if (!debug_json.contains("tensors")) {
        std::cerr << "❌ Missing 'tensors' field" << std::endl;
        return false;
    }
    size_t tensor_count = debug_json["tensors"].size();
    std::cout << "✓ Tensor count: " << tensor_count << std::endl;
    
    // Check tensor has stats
    if (tensor_count > 0) {
        const auto& first_tensor = debug_json["tensors"][0];
        if (!first_tensor.contains("stats")) {
            std::cerr << "❌ Missing tensor stats" << std::endl;
            return false;
        }
        std::cout << "✓ Tensor stats present" << std::endl;
        
        if (!first_tensor.contains("sample_values")) {
            std::cerr << "❌ Missing sample values" << std::endl;
            return false;
        }
        size_t sample_size = first_tensor["sample_values"].size();
        std::cout << "✓ Sample values: " << sample_size << std::endl;
    }
    
    // Cleanup
    fs::remove(path);
    
    std::cout << "✅ DebugJson test PASSED" << std::endl;
    return true;
}

// Test 4: Format detection
bool test_format_detection() {
    std::cout << "\n=== Test 4: Format Detection ===" << std::endl;
    
    CheckpointFormat fmt;
    
    // Test .safetensors extension
    fmt = detect_format("/tmp/test.safetensors");
    assert(fmt == CheckpointFormat::SafeTensors);
    std::cout << "✓ .safetensors detected" << std::endl;
    
    // Test .json extension (must check if file exists for proper detection)
    // For now, test with extension-based detection
    fmt = detect_format("/tmp/test.json");
    assert(fmt == CheckpointFormat::DebugJson || fmt == CheckpointFormat::SafeTensors);
    std::cout << "✓ .json handled" << std::endl;
    
    // Test directory with manifest.json
    std::string test_dir = "/tmp/mimir_test_detect";
    fs::create_directories(test_dir);
    std::ofstream(fs::path(test_dir) / "manifest.json") << "{}";
    
    fmt = detect_format(test_dir);
    assert(fmt == CheckpointFormat::RawFolder);
    std::cout << "✓ RawFolder directory detected" << std::endl;
    
    // Cleanup
    fs::remove_all(test_dir);
    
    std::cout << "✅ Format detection test PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "╔════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     Test de Sérialisation Mímir                       ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════╝" << std::endl;
    
    bool all_passed = true;
    
    // Run tests
    all_passed &= test_safetensors();
    all_passed &= test_raw_folder();
    all_passed &= test_debug_json();
    all_passed &= test_format_detection();
    
    // Summary
    std::cout << "\n╔════════════════════════════════════════════════════════╗" << std::endl;
    if (all_passed) {
        std::cout << "║  ✅ ALL TESTS PASSED                                   ║" << std::endl;
    } else {
        std::cout << "║  ❌ SOME TESTS FAILED                                  ║" << std::endl;
    }
    std::cout << "╚════════════════════════════════════════════════════════╝" << std::endl;
    
    return all_passed ? 0 : 1;
}
