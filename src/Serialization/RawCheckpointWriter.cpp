#include "RawCheckpointWriter.hpp"
#include "../Model.hpp"
#include "../Tokenizer.hpp"
#include "../Encoder.hpp"
#include "../Models/VAEModel.hpp"
#include "../Sha256.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <ctime>

namespace Mimir {
namespace Serialization {

RawCheckpointWriter::RawCheckpointWriter() {
}

RawCheckpointWriter::~RawCheckpointWriter() {
}

bool RawCheckpointWriter::save(
    Model& model,
    const std::string& path,
    const SaveOptions& options,
    std::string* error
) {
    try {
        fs::path root(path);
        
        // Create directory structure
        if (!create_structure(root.string(), error)) {
            return false;
        }
        
        // Collect tensors
        std::vector<TensorData> tensors = collect_tensors(model, options);
        
        // Save each tensor
        for (const auto& tensor : tensors) {
            if (!save_tensor(root.string(), tensor, error)) {
                return false;
            }
        }
        
        // Save architecture
        if (!save_architecture(root.string(), model, error)) {
            return false;
        }

        // Save training state (optimizer) if requested and available
        if (options.save_optimizer) {
            if (!save_training(root.string(), model, error)) {
                return false;
            }
        }
        
        // Save tokenizer if requested
        if (options.save_tokenizer) {
            if (!save_tokenizer(root.string(), model, error)) {
                return false;
            }
        }
        
        // Save encoder if requested
        if (options.save_encoder && model.getHasEncoder()) {
            if (!save_encoder(root.string(), model, error)) {
                return false;
            }
        }
        
        // Save manifest
        if (!save_manifest(root.string(), tensors, options, error)) {
            return false;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Raw checkpoint write error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointWriter::create_structure(
    const std::string& root,
    std::string* error
) {
    try {
        fs::path root_path(root);
        
        // Create directories
        fs::create_directories(root_path);
        fs::create_directories(root_path / "tensors");
        fs::create_directories(root_path / "model");
        fs::create_directories(root_path / "tokenizer");
        fs::create_directories(root_path / "encoder");
        fs::create_directories(root_path / "dataset");
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Failed to create directory structure: ") + e.what();
        }
        return false;
    }
}

std::vector<RawCheckpointWriter::TensorData> RawCheckpointWriter::collect_tensors(
    Model& model,
    const SaveOptions& options
) {
    std::vector<TensorData> tensors;
    
    const auto& layers = model.getLayers();
    
    // Collect layer weight blocks (modern allocation)
    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        const auto& layer = layers[layer_idx];
        
        if (!layer.weight_block) {
            continue;  // Layer has no weights
        }
        
        const float* data_ptr = layer.weight_block->getData();
        size_t size = layer.weight_block->getSize();
        
        if (data_ptr == nullptr || size == 0) {
            continue;  // Skip empty
        }
        
        TensorData td;
        td.name = layer.name + "_weights";
        td.dtype = DType::Float32;
        td.shape = {size};  // 1D tensor
        td.byte_size = size * sizeof(float);
        td.data_ptr = data_ptr;
        
        tensors.push_back(td);
    }
    
    // Add encoder embeddings
    if (options.save_encoder && model.getHasEncoder()) {
        const auto& enc = model.getEncoder();
        if (!enc.token_embeddings.empty()) {
            TensorData td;
            td.name = "encoder_token_embeddings";
            td.dtype = DType::Float32;
            td.shape = {static_cast<size_t>(enc.vocab_size), static_cast<size_t>(enc.dim)};
            td.byte_size = enc.token_embeddings.size() * sizeof(float);
            td.data_ptr = enc.token_embeddings.data();
            tensors.push_back(td);
        }
    }

    // Optimizer state tensors (debug/resume): m and v
    if (options.save_optimizer) {
        if (const Optimizer* opt = model.getSerializedOptimizer()) {
            if (!opt->m.empty()) {
                TensorData td;
                td.name = "optimizer/m";
                td.dtype = DType::Float32;
                td.shape = {opt->m.size()};
                td.byte_size = opt->m.size() * sizeof(float);
                td.data_ptr = opt->m.data();
                tensors.push_back(td);
            }
            if (!opt->v.empty()) {
                TensorData td;
                td.name = "optimizer/v";
                td.dtype = DType::Float32;
                td.shape = {opt->v.size()};
                td.byte_size = opt->v.size() * sizeof(float);
                td.data_ptr = opt->v.data();
                tensors.push_back(td);
            }
        }
    }

    // Gradient snapshot tensors (debug only). For now, implemented for VAEModel.
    if (options.include_gradients) {
        if (auto* vae = dynamic_cast<VAEModel*>(&model)) {
            const auto& grads = vae->getLastGradientsByLayer();
            for (const auto& kv : grads) {
                if (kv.second.empty()) continue;
                TensorData td;
                td.name = "grads/" + kv.first;
                td.dtype = DType::Float32;
                td.shape = {kv.second.size()};
                td.byte_size = kv.second.size() * sizeof(float);
                td.data_ptr = kv.second.data();
                tensors.push_back(td);
            }
        }
    }
    
    return tensors;
}

bool RawCheckpointWriter::save_training(
    const std::string& root,
    Model& model,
    std::string* error
) {
    try {
        fs::path training_path = fs::path(root) / "model" / "training.json";

        json j;
        j["has_optimizer"] = false;

        if (const Optimizer* opt = model.getSerializedOptimizer()) {
            j["has_optimizer"] = true;
            j["type"] = static_cast<int>(opt->type);
            j["step"] = opt->step;
            j["lr_current"] = opt->getCurrentLR();
            j["beta1"] = opt->beta1;
            j["beta2"] = opt->beta2;
            j["eps"] = opt->eps;
            j["weight_decay"] = opt->weight_decay;
            j["decay_strategy"] = static_cast<int>(opt->decay_strategy);
            j["initial_lr"] = opt->initial_lr;
            j["min_lr"] = opt->min_lr;
            j["decay_rate"] = opt->decay_rate;
            j["decay_steps"] = opt->decay_steps;
            j["total_steps"] = opt->total_steps;
            j["warmup_steps"] = opt->warmup_steps;
            j["state_sizes"] = {
                {"m", opt->m.size()},
                {"v", opt->v.size()}
            };
        }

        std::ofstream file(training_path);
        if (!file) {
            if (error) {
                *error = "Failed to create training.json";
            }
            return false;
        }

        file << std::setw(2) << j;
        file.close();
        return true;
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Training save error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointWriter::save_tensor(
    const std::string& root,
    const TensorData& tensor,
    std::string* error
) {
    try {
        fs::path root_path(root);
        fs::path bin_path = root_path / "tensors" / (tensor.name + ".bin");
        fs::path json_path = root_path / "tensors" / (tensor.name + ".json");

        // Ensure directories exist (tensor.name may contain subpaths like "vae/enc_fc0_weights")
        fs::create_directories(bin_path.parent_path());
        fs::create_directories(json_path.parent_path());
        
        // Write binary data
        std::ofstream bin_file(bin_path, std::ios::binary);
        if (!bin_file) {
            if (error) {
                *error = "Failed to create tensor file: " + bin_path.string();
            }
            return false;
        }
        
        bin_file.write(static_cast<const char*>(tensor.data_ptr), tensor.byte_size);
        bin_file.close();
        
        // Calculate checksum
        std::string checksum = calculate_checksum(tensor.data_ptr, tensor.byte_size);
        
        // Write JSON metadata
        json tensor_json;
        tensor_json["name"] = tensor.name;
        tensor_json["dtype"] = dtype_to_string(tensor.dtype);
        tensor_json["shape"] = tensor.shape;
        tensor_json["byte_size"] = tensor.byte_size;
        tensor_json["checksum"] = checksum;
        tensor_json["checksum_algo"] = "sha256";
        tensor_json["data_file"] = tensor.name + ".bin";
        
        std::ofstream json_file(json_path);
        if (!json_file) {
            if (error) {
                *error = "Failed to create tensor JSON: " + json_path.string();
            }
            return false;
        }
        
        json_file << std::setw(2) << tensor_json;
        json_file.close();
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Tensor save error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointWriter::save_architecture(
    const std::string& root,
    Model& model,
    std::string* error
) {
    try {
        fs::path arch_path = fs::path(root) / "model" / "architecture.json";
        
        json arch;
        arch["model_name"] = model.getModelName();
        arch["total_params"] = model.totalParamCount();
        arch["num_layers"] = model.getLayers().size();
        
        // Save layer info
        json layers_array = json::array();
        for (const auto& layer : model.getLayers()) {
            json layer_obj;
            layer_obj["name"] = layer.name;
            layer_obj["type"] = layer.type;
            layer_obj["params_count"] = layer.params_count;
            layers_array.push_back(layer_obj);
        }
        arch["layers"] = layers_array;
        
        // Save I/O dimensions if known
        if (model.width() > 0 && model.height() > 0) {
            arch["image_width"] = model.width();
            arch["image_height"] = model.height();
        }
        
        std::ofstream file(arch_path);
        if (!file) {
            if (error) {
                *error = "Failed to create architecture.json";
            }
            return false;
        }
        
        file << std::setw(2) << arch;
        file.close();
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Architecture save error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointWriter::save_tokenizer(
    const std::string& root,
    Model& model,
    std::string* error
) {
    try {
        fs::path tok_path = fs::path(root) / "tokenizer" / "tokenizer.json";
        
        const auto& tokenizer = model.getTokenizer();
        json tok_json = tokenizer.to_json();
        
        std::ofstream file(tok_path);
        if (!file) {
            if (error) {
                *error = "Failed to create tokenizer.json";
            }
            return false;
        }
        
        file << std::setw(2) << tok_json;
        file.close();
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Tokenizer save error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointWriter::save_encoder(
    const std::string& root,
    Model& model,
    std::string* error
) {
    try {
        fs::path enc_path = fs::path(root) / "encoder" / "encoder.json";
        
        const auto& encoder = model.getEncoder();
        json enc_json = encoder.to_json();
        
        std::ofstream file(enc_path);
        if (!file) {
            if (error) {
                *error = "Failed to create encoder.json";
            }
            return false;
        }
        
        file << std::setw(2) << enc_json;
        file.close();
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Encoder save error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointWriter::save_manifest(
    const std::string& root,
    const std::vector<TensorData>& tensors,
    const SaveOptions& options,
    std::string* error
) {
    try {
        fs::path manifest_path = fs::path(root) / "manifest.json";
        
        json manifest;
        manifest["format"] = "mimir_raw_checkpoint";
        manifest["format_version"] = "1.0.0";
        manifest["mimir_version"] = get_mimir_version();
        
        if (options.include_git_info) {
            std::string git_commit = get_git_commit();
            if (!git_commit.empty()) {
                manifest["git_commit"] = git_commit;
            }
        }
        
        auto now = std::time(nullptr);
        manifest["created_at"] = static_cast<long long>(now);
        
        // List all components
        json components;
        components["tensors"] = tensors.size();
        components["model_architecture"] = true;
        components["tokenizer"] = options.save_tokenizer;
        components["encoder"] = options.save_encoder;
        components["optimizer"] = options.save_optimizer;
        manifest["components"] = components;
        
        // Tensor index
        json tensor_index = json::array();
        for (const auto& tensor : tensors) {
            json t_entry;
            t_entry["name"] = tensor.name;
            t_entry["bin_file"] = "tensors/" + tensor.name + ".bin";
            t_entry["json_file"] = "tensors/" + tensor.name + ".json";
            tensor_index.push_back(t_entry);
        }
        manifest["tensor_index"] = tensor_index;
        
        std::ofstream file(manifest_path);
        if (!file) {
            if (error) {
                *error = "Failed to create manifest.json";
            }
            return false;
        }
        
        file << std::setw(2) << manifest;
        file.close();
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Manifest save error: ") + e.what();
        }
        return false;
    }
}

std::string RawCheckpointWriter::calculate_checksum(
    const void* data,
    size_t size
) {
    // Use SHA256 for checksums
    std::string input(static_cast<const char*>(data), size);
    return sha256(input);
}

} // namespace Serialization
} // namespace Mimir
