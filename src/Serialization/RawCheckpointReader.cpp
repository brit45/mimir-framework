#include "RawCheckpointReader.hpp"
#include "../Model.hpp"
#include "../Tokenizer.hpp"
#include "../Encoder.hpp"
#include "../Sha256.hpp"
#include <fstream>
#include <algorithm>

namespace Mimir {
namespace Serialization {

RawCheckpointReader::RawCheckpointReader() {
}

RawCheckpointReader::~RawCheckpointReader() {
}

bool RawCheckpointReader::load(
    Model& model,
    const std::string& path,
    const LoadOptions& options,
    std::string* error
) {
    try {
        fs::path root(path);
        
        // Check if directory exists
        if (!fs::exists(root) || !fs::is_directory(root)) {
            if (error) {
                *error = "Checkpoint directory not found: " + path;
            }
            return false;
        }
        
        // Load manifest
        json manifest;
        if (!load_manifest(root.string(), manifest, error)) {
            return false;
        }
        
        // Load architecture
        if (!load_architecture(root.string(), model, options, error)) {
            return false;
        }
        
        // Load tokenizer if requested
        if (options.load_tokenizer) {
            if (!load_tokenizer(root.string(), model, error)) {
                // Non-fatal if tokenizer missing in non-strict mode
                if (options.strict_mode) {
                    return false;
                }
            }
        }
        
        // Load encoder if requested
        if (options.load_encoder) {
            if (!load_encoder(root.string(), model, error)) {
                // Non-fatal if encoder missing in non-strict mode
                if (options.strict_mode) {
                    return false;
                }
            }
        }

        // Load training/optimizer state if requested
        if (options.load_optimizer) {
            // Non-fatal if absent unless strict_mode
            if (!load_training(root.string(), model, error)) {
                if (options.strict_mode) {
                    return false;
                }
            }
        }
        
        // Apply tensors to model
        return apply_tensors_to_model(root.string(), model, manifest, options, error);
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Raw checkpoint load error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointReader::load_training(
    const std::string& root,
    Model& model,
    std::string* error
) {
    try {
        fs::path training_path = fs::path(root) / "model" / "training.json";
        if (!fs::exists(training_path)) {
            if (error) {
                *error = "training.json not found";
            }
            return false;
        }

        std::ifstream file(training_path);
        if (!file) {
            if (error) {
                *error = "Failed to open training.json";
            }
            return false;
        }

        json j;
        file >> j;

        if (!j.value("has_optimizer", false)) {
            model.clearSerializedOptimizer();
            return true;
        }

        Optimizer opt;
        opt.type = static_cast<OptimizerType>(j.value("type", static_cast<int>(OptimizerType::ADAMW)));
        opt.step = static_cast<size_t>(j.value("step", 0));
        opt.beta1 = j.value("beta1", opt.beta1);
        opt.beta2 = j.value("beta2", opt.beta2);
        opt.eps = j.value("eps", opt.eps);
        opt.weight_decay = j.value("weight_decay", opt.weight_decay);
        opt.decay_strategy = static_cast<LRDecayStrategy>(j.value("decay_strategy", static_cast<int>(opt.decay_strategy)));
        opt.initial_lr = j.value("initial_lr", opt.initial_lr);
        opt.min_lr = j.value("min_lr", opt.min_lr);
        opt.decay_rate = j.value("decay_rate", opt.decay_rate);
        opt.decay_steps = j.value("decay_steps", opt.decay_steps);
        opt.total_steps = j.value("total_steps", opt.total_steps);
        opt.warmup_steps = j.value("warmup_steps", opt.warmup_steps);

        model.setSerializedOptimizer(opt);
        return true;
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Training load error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointReader::load_manifest(
    const std::string& root,
    json& manifest_out,
    std::string* error
) {
    try {
        fs::path manifest_path = fs::path(root) / "manifest.json";
        
        if (!fs::exists(manifest_path)) {
            if (error) {
                *error = "manifest.json not found";
            }
            return false;
        }
        
        std::ifstream file(manifest_path);
        if (!file) {
            if (error) {
                *error = "Failed to open manifest.json";
            }
            return false;
        }
        
        file >> manifest_out;
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Manifest load error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointReader::load_architecture(
    const std::string& root,
    Model& model,
    const LoadOptions& options,
    std::string* error
) {
    try {
        fs::path arch_path = fs::path(root) / "model" / "architecture.json";
        
        if (!fs::exists(arch_path)) {
            if (error) {
                *error = "architecture.json not found";
            }
            return false;
        }
        
        std::ifstream file(arch_path);
        if (!file) {
            if (error) {
                *error = "Failed to open architecture.json";
            }
            return false;
        }
        
        json arch;
        file >> arch;
        
        // Load basic info
        if (options.apply_model_name && arch.contains("model_name")) {
            model.setModelName(arch["model_name"].get<std::string>());
        }

        // Load model config (critical for downstream consumers relying on model.modelConfig)
        if (options.apply_model_config && arch.contains("model_config")) {
            try {
                model.modelConfig = arch["model_config"];
            } catch (...) {
                if (error) {
                    *error = "Invalid model_config in architecture.json";
                }
                return false;
            }
        }
        
        // Load layers structure
        // IMPORTANT: l'architecture sauvegardée ne contient pas toutes les métadonnées
        // (in_features/out_features, kernels, etc.). Si le caller a déjà construit
        // l'architecture via un builder (recommandé), on ne l'écrase pas.
        if (arch.contains("layers") && model.getLayers().empty()) {
            model.getMutableLayers().clear();
            for (const auto& layer_obj : arch["layers"]) {
                const std::string name = layer_obj.value("name", "");
                const std::string type = layer_obj.value("type", "");
                const size_t params_count = layer_obj.value("params_count", 0);
                Layer layer(name, type, params_count); // initialise type_enum + normalise le type
                model.getMutableLayers().push_back(std::move(layer));
            }

            // Allocate parameters based on architecture (si on vient de reconstruire)
            model.allocateParams();
        }
        
        // Load I/O dimensions
        // Note: tw and th are read-only via width()/height(), set during model construction
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Architecture load error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointReader::load_tokenizer(
    const std::string& root,
    Model& model,
    std::string* error
) {
    try {
        fs::path tok_path = fs::path(root) / "tokenizer" / "tokenizer.json";
        
        if (!fs::exists(tok_path)) {
            if (error) {
                *error = "tokenizer.json not found";
            }
            return false;
        }
        
        std::ifstream file(tok_path);
        if (!file) {
            if (error) {
                *error = "Failed to open tokenizer.json";
            }
            return false;
        }
        
        json tok_json;
        file >> tok_json;
        
        auto& tokenizer = model.getMutableTokenizer();
        tokenizer.from_json(tok_json);
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Tokenizer load error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointReader::load_encoder(
    const std::string& root,
    Model& model,
    std::string* error
) {
    try {
        fs::path enc_path = fs::path(root) / "encoder" / "encoder.json";
        
        if (!fs::exists(enc_path)) {
            if (error) {
                *error = "encoder.json not found";
            }
            return false;
        }
        
        std::ifstream file(enc_path);
        if (!file) {
            if (error) {
                *error = "Failed to open encoder.json";
            }
            return false;
        }
        
        json enc_json;
        file >> enc_json;
        
        auto& encoder = model.getMutableEncoder();
        encoder.from_json(enc_json);
        
        model.setHasEncoder(true);
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Encoder load error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointReader::load_tensor_metadata(
    const std::string& json_path,
    TensorMetadata& metadata_out,
    std::string* error
) {
    try {
        if (!fs::exists(json_path)) {
            if (error) {
                *error = "Tensor JSON not found: " + json_path;
            }
            return false;
        }
        
        std::ifstream file(json_path);
        if (!file) {
            if (error) {
                *error = "Failed to open tensor JSON: " + json_path;
            }
            return false;
        }
        
        json tensor_json;
        file >> tensor_json;
        
        metadata_out.name = tensor_json.value("name", "");
        metadata_out.dtype = string_to_dtype(tensor_json.value("dtype", "F32"));
        metadata_out.shape = tensor_json.value("shape", std::vector<size_t>{});
        metadata_out.byte_size = tensor_json.value("byte_size", 0);
        metadata_out.checksum = tensor_json.value("checksum", "");
        metadata_out.checksum_algo = tensor_json.value("checksum_algo", "sha256");
        metadata_out.data_file = tensor_json.value("data_file", "");
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Tensor metadata load error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointReader::load_tensor_data(
    const std::string& bin_path,
    const TensorMetadata& metadata,
    void* dest_buffer,
    size_t dest_size,
    std::string* error
) {
    try {
        if (!fs::exists(bin_path)) {
            if (error) {
                *error = "Tensor data file not found: " + bin_path;
            }
            return false;
        }
        
        std::ifstream file(bin_path, std::ios::binary);
        if (!file) {
            if (error) {
                *error = "Failed to open tensor data: " + bin_path;
            }
            return false;
        }
        
        // Check size
        if (metadata.byte_size > dest_size) {
            if (error) {
                *error = "Buffer too small for tensor: " + metadata.name;
            }
            return false;
        }
        
        // Read data
        file.read(static_cast<char*>(dest_buffer), metadata.byte_size);
        if (!file) {
            if (error) {
                *error = "Failed to read tensor data: " + metadata.name;
            }
            return false;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Tensor data load error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointReader::verify_checksum(
    const void* data,
    size_t size,
    const std::string& expected_checksum,
    const std::string& algo
) {
    if (algo == "sha256") {
        std::string input(static_cast<const char*>(data), size);
        std::string computed = sha256(input);
        return computed == expected_checksum;
    }
    // Unknown algorithm - skip verification
    return true;
}

bool RawCheckpointReader::apply_tensors_to_model(
    const std::string& root,
    Model& model,
    const json& manifest,
    const LoadOptions& options,
    std::string* error
) {
    try {
        fs::path root_path(root);
        
        if (!manifest.contains("tensor_index")) {
            if (error) {
                *error = "Missing tensor_index in manifest";
            }
            return false;
        }
        
        const auto& layers = model.getLayers();
        
        // Build map: tensor name -> layer index
        std::unordered_map<std::string, size_t> layer_map;
        for (size_t i = 0; i < layers.size(); ++i) {
            std::string tensor_name = layers[i].name + "_weights";
            layer_map[tensor_name] = i;
        }
        
        // Load each tensor
        for (const auto& t_entry : manifest["tensor_index"]) {
            std::string tensor_name = t_entry.value("name", "");
            std::string json_file = t_entry.value("json_file", "");
            
            fs::path json_path = root_path / json_file;
            
            // Load metadata
            TensorMetadata t_metadata;
            if (!load_tensor_metadata(json_path.string(), t_metadata, error)) {
                if (options.strict_mode) {
                    return false;
                }
                continue;
            }
            
            // Find corresponding layer
            auto it = layer_map.find(tensor_name);
            if (it == layer_map.end()) {
                // Optimizer tensors (optional)
                if ((tensor_name == "optimizer/m" || tensor_name == "optimizer/v")) {
                    if (!options.load_optimizer) {
                        // Known-but-ignored tensor
                        continue;
                    }

                    Optimizer* opt = model.getMutableSerializedOptimizer();
                    if (!opt) {
                        // If training.json wasn't loaded, create a default optimizer container
                        Optimizer tmp;
                        model.setSerializedOptimizer(tmp);
                        opt = model.getMutableSerializedOptimizer();
                    }

                    size_t expected_size = 1;
                    for (size_t dim : t_metadata.shape) expected_size *= dim;

                    std::vector<float>* dst = (tensor_name == "optimizer/m") ? &opt->m : &opt->v;
                    dst->resize(expected_size);

                    fs::path bin_path = root_path / "tensors" / t_metadata.data_file;
                    if (!load_tensor_data(
                        bin_path.string(), t_metadata,
                        dst->data(),
                        dst->size() * sizeof(float),
                        error
                    )) {
                        return false;
                    }
                    continue;
                }

                // Gradient snapshots (debug-only) - ignore on load
                if (tensor_name.rfind("grads/", 0) == 0) {
                    continue;
                }

                // Check if it's encoder embeddings
                if (tensor_name == "encoder_token_embeddings" && options.load_encoder) {
                    auto& enc = model.getMutableEncoder();
                    
                    size_t expected_size = 1;
                    for (size_t dim : t_metadata.shape) {
                        expected_size *= dim;
                    }
                    
                    if (enc.token_embeddings.size() < expected_size) {
                        enc.token_embeddings.resize(expected_size);
                    }
                    
                    fs::path bin_path = root_path / "tensors" / t_metadata.data_file;
                    if (!load_tensor_data(
                        bin_path.string(), t_metadata,
                        enc.token_embeddings.data(),
                        enc.token_embeddings.size() * sizeof(float),
                        error
                    )) {
                        return false;
                    }
                    
                    // Verify checksum if requested
                    if (options.validate_checksums && !t_metadata.checksum.empty()) {
                        if (!verify_checksum(
                            enc.token_embeddings.data(),
                            t_metadata.byte_size,
                            t_metadata.checksum,
                            t_metadata.checksum_algo
                        )) {
                            if (error) {
                                *error = "Checksum mismatch for: " + tensor_name;
                            }
                            return false;
                        }
                    }
                    
                    continue;
                }
                
                if (options.strict_mode) {
                    if (error) {
                        *error = "Unknown tensor in strict mode: " + tensor_name;
                    }
                    return false;
                }
                continue;
            }
            
            // Load tensor into layer weight block
            size_t layer_idx = it->second;
            auto& layer = const_cast<Layer&>(layers[layer_idx]);
            
            if (!layer.weight_block) {
                if (error) {
                    *error = "Layer has no weight block: " + layer.name;
                }
                return false;
            }
            
            // Get expected size
            size_t expected_size = 1;
            for (size_t dim : t_metadata.shape) {
                expected_size *= dim;
            }
            
            // Verify size matches
            size_t actual_size = layer.weight_block->getSize();
            if (actual_size != expected_size) {
                if (error) {
                    *error = "Size mismatch for " + tensor_name + ": expected " + 
                             std::to_string(expected_size) + " got " + std::to_string(actual_size);
                }
                return false;
            }
            
            // Load data
            float* data_ptr = layer.weight_block->getData();
            if (!data_ptr) {
                if (error) {
                    *error = "Failed to get data pointer for: " + tensor_name;
                }
                return false;
            }
            
            fs::path bin_path = root_path / "tensors" / t_metadata.data_file;
            if (!load_tensor_data(
                bin_path.string(), t_metadata,
                data_ptr,
                t_metadata.byte_size,
                error
            )) {
                return false;
            }
            
            // Verify checksum if requested
            if (options.validate_checksums && !t_metadata.checksum.empty()) {
                if (!verify_checksum(
                    data_ptr,
                    t_metadata.byte_size,
                    t_metadata.checksum,
                    t_metadata.checksum_algo
                )) {
                    if (error) {
                        *error = "Checksum mismatch for: " + tensor_name;
                    }
                    return false;
                }
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Apply tensors error: ") + e.what();
        }
        return false;
    }
}

} // namespace Serialization
} // namespace Mimir
