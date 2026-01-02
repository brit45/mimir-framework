#include "Serialization.hpp"
#include "SafeTensorsWriter.hpp"
#include "SafeTensorsReader.hpp"
#include "RawCheckpointWriter.hpp"
#include "RawCheckpointReader.hpp"
#include "DebugJsonDump.hpp"
#include "../Model.hpp"
#include <fstream>
#include <sstream>
#include <cstdlib>

namespace Mimir {
namespace Serialization {

// ============================================================================
// High-Level API Implementation
// ============================================================================

bool save_checkpoint(
    Model& model,
    const std::string& path,
    const SaveOptions& options,
    std::string* error
) {
    try {
        switch (options.format) {
            case CheckpointFormat::SafeTensors: {
                SafeTensorsWriter writer;
                return writer.save(model, path, options, error);
            }
            
            case CheckpointFormat::RawFolder: {
                RawCheckpointWriter writer;
                return writer.save(model, path, options, error);
            }
            
            case CheckpointFormat::DebugJson: {
                DebugJsonDump dumper;
                
                // Convert SaveOptions to DebugJsonOptions
                DebugJsonOptions debug_opts;
                debug_opts.include_gradients = options.include_gradients;
                debug_opts.include_optimizer_state = options.include_optimizer_state;
                debug_opts.max_values_per_tensor = options.max_values_per_tensor;
                debug_opts.include_activations = options.include_activations;
                debug_opts.include_checksums = options.include_checksums;
                debug_opts.include_weight_deltas = options.include_weight_deltas;
                debug_opts.include_git_info = options.include_git_info;
                debug_opts.save_tokenizer = options.save_tokenizer;
                debug_opts.save_encoder = options.save_encoder;
                
                return dumper.save_enhanced(path, model, debug_opts, error);
            }
            
            default:
                if (error) {
                    *error = "Unknown checkpoint format";
                }
                return false;
        }
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Exception during save: ") + e.what();
        }
        return false;
    }
}

bool load_checkpoint(
    Model& model,
    const std::string& path,
    const LoadOptions& options,
    std::string* error
) {
    try {
        switch (options.format) {
            case CheckpointFormat::SafeTensors: {
                SafeTensorsReader reader;
                return reader.load(model, path, options, error);
            }
            
            case CheckpointFormat::RawFolder: {
                RawCheckpointReader reader;
                return reader.load(model, path, options, error);
            }
            
            case CheckpointFormat::DebugJson: {
                if (error) {
                    *error = "DebugJson format is write-only (for debugging)";
                }
                return false;
            }
            
            default:
                if (error) {
                    *error = "Unknown checkpoint format";
                }
                return false;
        }
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Exception during load: ") + e.what();
        }
        return false;
    }
}

// ============================================================================
// Utility Functions Implementation
// ============================================================================

std::string dtype_to_string(DType dtype) {
    switch (dtype) {
        case DType::Float32: return "F32";
        case DType::Float16: return "F16";
        case DType::Int32: return "I32";
        case DType::Int16: return "I16";
        case DType::Uint16: return "U16";
        case DType::Uint8: return "U8";
        default: return "UNKNOWN";
    }
}

DType string_to_dtype(const std::string& str) {
    if (str == "F32" || str == "float32") return DType::Float32;
    if (str == "F16" || str == "float16") return DType::Float16;
    if (str == "I32" || str == "int32") return DType::Int32;
    if (str == "I16" || str == "int16") return DType::Int16;
    if (str == "U16" || str == "uint16") return DType::Uint16;
    if (str == "U8" || str == "uint8") return DType::Uint8;
    throw std::runtime_error("Unknown dtype string: " + str);
}

size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return 4;
        case DType::Float16: return 2;
        case DType::Int32: return 4;
        case DType::Int16: return 2;
        case DType::Uint16: return 2;
        case DType::Uint8: return 1;
        default: return 0;
    }
}

CheckpointFormat detect_format(const std::string& path) {
    fs::path p(path);
    
    if (fs::is_directory(p)) {
        // Check for manifest.json to distinguish RawFolder
        if (fs::exists(p / "manifest.json")) {
            return CheckpointFormat::RawFolder;
        }
    } else if (fs::is_regular_file(p)) {
        // Check extension
        std::string ext = p.extension().string();
        if (ext == ".safetensors" || ext == ".st") {
            return CheckpointFormat::SafeTensors;
        } else if (ext == ".json") {
            return CheckpointFormat::DebugJson;
        }
    }
    
    // Default to SafeTensors
    return CheckpointFormat::SafeTensors;
}

std::string get_mimir_version() {
    // Try to read from VERSION file
    fs::path version_file = fs::current_path() / "VERSION";
    if (fs::exists(version_file)) {
        std::ifstream ifs(version_file);
        std::string version;
        if (std::getline(ifs, version)) {
            return version;
        }
    }
    return "2.3.0";  // Default version
}

std::string get_git_commit() {
    // Try to get git commit hash
    const char* cmd = "git rev-parse --short HEAD 2>/dev/null";
    FILE* pipe = popen(cmd, "r");
    if (!pipe) return "";
    
    char buffer[128];
    std::string result;
    if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result = buffer;
        // Remove trailing newline
        if (!result.empty() && result.back() == '\n') {
            result.pop_back();
        }
    }
    pclose(pipe);
    return result;
}

} // namespace Serialization
} // namespace Mimir
