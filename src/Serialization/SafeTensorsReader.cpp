#include "SafeTensorsReader.hpp"
#include "../Model.hpp"
#include "../Encoder.hpp"
#include <fstream>
#include <algorithm>
#include <cstring>

namespace Mimir {
namespace Serialization {

SafeTensorsReader::SafeTensorsReader() {
}

SafeTensorsReader::~SafeTensorsReader() {
}

bool SafeTensorsReader::load(
    Model& model,
    const std::string& path,
    const LoadOptions& options,
    std::string* error
) {
    try {
        // Check if file exists
        if (!fs::exists(path)) {
            if (error) {
                *error = "File not found: " + path;
            }
            return false;
        }
        
        // Parse header
        json header;
        std::vector<ParsedTensor> tensors;
        size_t data_offset;
        
        if (!parse_header(path, header, tensors, data_offset, error)) {
            return false;
        }
        
        // Apply tensors to model
        return apply_tensors_to_model(model, tensors, path, data_offset, options, error);
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("SafeTensors load error: ") + e.what();
        }
        return false;
    }
}

bool SafeTensorsReader::parse_header(
    const std::string& path,
    json& header_out,
    std::vector<ParsedTensor>& tensors_out,
    size_t& data_offset_out,
    std::string* error
) {
    try {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            if (error) {
                *error = "Failed to open file: " + path;
            }
            return false;
        }
        
        // Read header length (8 bytes, little-endian)
        uint64_t header_len = read_u64_le(file);
        if (!file || header_len == 0 || header_len > 100 * 1024 * 1024) {  // Max 100MB header
            if (error) {
                *error = "Invalid header length";
            }
            return false;
        }
        
        // Read header JSON
        std::vector<char> header_data(header_len);
        file.read(header_data.data(), header_len);
        if (!file) {
            if (error) {
                *error = "Failed to read header";
            }
            return false;
        }
        
        // Parse JSON
        std::string header_str(header_data.begin(), header_data.end());
        header_out = json::parse(header_str);
        
        // Data starts after 8-byte length + header
        data_offset_out = 8 + header_len;
        
        // Extract tensor information
        for (auto it = header_out.begin(); it != header_out.end(); ++it) {
            if (it.key() == "__metadata__") {
                continue;  // Skip metadata
            }
            
            const json& tensor_entry = it.value();
            
            ParsedTensor tensor;
            tensor.name = it.key();
            
            // Parse dtype
            if (!tensor_entry.contains("dtype")) {
                if (error) {
                    *error = "Missing dtype for tensor: " + tensor.name;
                }
                return false;
            }
            std::string dtype_str = tensor_entry["dtype"];
            tensor.dtype = string_to_dtype(dtype_str);
            
            // Parse shape
            if (!tensor_entry.contains("shape")) {
                if (error) {
                    *error = "Missing shape for tensor: " + tensor.name;
                }
                return false;
            }
            tensor.shape = tensor_entry["shape"].get<std::vector<size_t>>();
            
            // Parse data_offsets
            if (!tensor_entry.contains("data_offsets")) {
                if (error) {
                    *error = "Missing data_offsets for tensor: " + tensor.name;
                }
                return false;
            }
            auto offsets = tensor_entry["data_offsets"];
            if (!offsets.is_array() || offsets.size() != 2) {
                if (error) {
                    *error = "Invalid data_offsets for tensor: " + tensor.name;
                }
                return false;
            }
            tensor.data_begin = offsets[0].get<size_t>();
            tensor.data_end = offsets[1].get<size_t>();
            
            // Validate size
            size_t expected_size = 1;
            for (size_t dim : tensor.shape) {
                expected_size *= dim;
            }
            expected_size *= dtype_size(tensor.dtype);
            
            if (tensor.data_end - tensor.data_begin != expected_size) {
                if (error) {
                    *error = "Size mismatch for tensor: " + tensor.name;
                }
                return false;
            }
            
            tensors_out.push_back(tensor);
        }
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Header parse error: ") + e.what();
        }
        return false;
    }
}

bool SafeTensorsReader::load_tensor_data(
    const std::string& path,
    size_t data_offset,
    const ParsedTensor& tensor,
    void* dest_buffer,
    size_t dest_size,
    std::string* error
) {
    try {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            if (error) {
                *error = "Failed to open file: " + path;
            }
            return false;
        }
        
        // Seek to tensor data
        size_t absolute_offset = data_offset + tensor.data_begin;
        file.seekg(absolute_offset);
        if (!file) {
            if (error) {
                *error = "Failed to seek to tensor data: " + tensor.name;
            }
            return false;
        }
        
        // Read tensor data
        size_t bytes_to_read = tensor.data_end - tensor.data_begin;
        if (bytes_to_read > dest_size) {
            if (error) {
                *error = "Buffer too small for tensor: " + tensor.name;
            }
            return false;
        }
        
        file.read(static_cast<char*>(dest_buffer), bytes_to_read);
        if (!file) {
            if (error) {
                *error = "Failed to read tensor data: " + tensor.name;
            }
            return false;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Tensor load error: ") + e.what();
        }
        return false;
    }
}

bool SafeTensorsReader::apply_tensors_to_model(
    Model& model,
    const std::vector<ParsedTensor>& tensors,
    const std::string& path,
    size_t data_offset,
    const LoadOptions& options,
    std::string* error
) {
    try {
        const auto& layers = model.getLayers();
        
        // Create a map of tensor names for quick lookup
        std::unordered_map<std::string, const ParsedTensor*> tensor_map;
        for (const auto& tensor : tensors) {
            tensor_map[tensor.name] = &tensor;
        }
        
        // Load layer weight blocks
        for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
            auto& layer = const_cast<Layer&>(layers[layer_idx]);
            
            if (!layer.weight_block) {
                continue;  // Layer has no weights
            }
            
            // Try to find tensor with name "layerN/weights"
            std::string tensor_name = layer.name + "/weights";
            
            auto it = tensor_map.find(tensor_name);
            if (it != tensor_map.end()) {
                const ParsedTensor* tensor = it->second;
                
                // Get expected size
                size_t expected_size = 1;
                for (size_t dim : tensor->shape) {
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
                
                // Get data pointer
                float* data_ptr = layer.weight_block->getData();
                if (!data_ptr) {
                    if (error) {
                        *error = "Failed to get data pointer for: " + tensor_name;
                    }
                    return false;
                }
                
                // Load data
                if (!load_tensor_data(
                    path, data_offset, *tensor,
                    data_ptr, actual_size * sizeof(float),
                    error
                )) {
                    return false;
                }
            } else if (options.strict_mode) {
                if (error) {
                    *error = "Missing tensor in strict mode: " + tensor_name;
                }
                return false;
            }
        }

        // Load tokenizer JSON (as raw bytes)
        if (options.load_tokenizer) {
            auto it = tensor_map.find("tokenizer/json");
            if (it != tensor_map.end()) {
                const ParsedTensor* tensor = it->second;
                if (tensor->dtype == DType::Uint8) {
                    const size_t n = tensor->data_end - tensor->data_begin;
                    std::vector<uint8_t> buf(n);
                    if (!load_tensor_data(path, data_offset, *tensor, buf.data(), buf.size(), error)) {
                        return false;
                    }
                    try {
                        std::string s(reinterpret_cast<const char*>(buf.data()), buf.size());
                        json j = json::parse(s);
                        model.getMutableTokenizer().from_json(j);
                    } catch (...) {
                        // Ignore invalid tokenizer JSON for backward compatibility
                    }
                }
            } else if (options.strict_mode) {
                if (error) {
                    *error = "Missing tensor in strict mode: tokenizer/json";
                }
                return false;
            }
        }

        // Load encoder JSON (preferred for full encoder state)
        if (options.load_encoder) {
            auto it = tensor_map.find("encoder/json");
            if (it != tensor_map.end()) {
                const ParsedTensor* tensor = it->second;
                if (tensor->dtype == DType::Uint8) {
                    const size_t n = tensor->data_end - tensor->data_begin;
                    std::vector<uint8_t> buf(n);
                    if (!load_tensor_data(path, data_offset, *tensor, buf.data(), buf.size(), error)) {
                        return false;
                    }
                    try {
                        std::string s(reinterpret_cast<const char*>(buf.data()), buf.size());
                        json j = json::parse(s);
                        model.getMutableEncoder().from_json(j);
                        model.setHasEncoder(true);
                    } catch (...) {
                        // Ignore invalid encoder JSON and fall back to embeddings tensor
                    }
                }
            } else if (options.strict_mode) {
                if (error) {
                    *error = "Missing tensor in strict mode: encoder/json";
                }
                return false;
            }
        }
        
        // Load encoder embeddings
        if (options.load_encoder) {
            auto it = tensor_map.find("encoder/token_embeddings");
            if (it != tensor_map.end()) {
                const ParsedTensor* tensor = it->second;
                
                auto& enc = model.getMutableEncoder();
                if (tensor->shape.size() == 2) {
                    enc.vocab_size = static_cast<int>(tensor->shape[0]);
                    enc.dim = static_cast<int>(tensor->shape[1]);
                }
                size_t expected_size = 1;
                for (size_t dim : tensor->shape) {
                    expected_size *= dim;
                }
                
                if (enc.token_embeddings.size() < expected_size) {
                    enc.token_embeddings.resize(expected_size);
                }
                
                if (!load_tensor_data(
                    path, data_offset, *tensor,
                    enc.token_embeddings.data(),
                    enc.token_embeddings.size() * sizeof(float),
                    error
                )) {
                    return false;
                }

                model.setHasEncoder(true);
            }
        }

        // Load optimizer (json + state vectors) if requested
        if (options.load_optimizer) {
            // Parse optimizer/json
            auto itj = tensor_map.find("optimizer/json");
            if (itj != tensor_map.end()) {
                const ParsedTensor* tensor = itj->second;
                if (tensor->dtype == DType::Uint8) {
                    const size_t n = tensor->data_end - tensor->data_begin;
                    std::vector<uint8_t> buf(n);
                    if (!load_tensor_data(path, data_offset, *tensor, buf.data(), buf.size(), error)) {
                        return false;
                    }
                    try {
                        std::string s(reinterpret_cast<const char*>(buf.data()), buf.size());
                        json j = json::parse(s);
                        Optimizer opt;
                        opt.type = static_cast<OptimizerType>(j.value("type", static_cast<int>(opt.type)));
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
                    } catch (...) {
                        // Invalid optimizer JSON
                        if (options.strict_mode) {
                            if (error) {
                                *error = "Invalid optimizer/json";
                            }
                            return false;
                        }
                    }
                }
            } else if (options.strict_mode) {
                if (error) {
                    *error = "Missing tensor in strict mode: optimizer/json";
                }
                return false;
            }

            // Load optimizer state vectors (optional; if present without json, create default container)
            auto load_opt_vec = [&](const std::string& name, std::vector<float>& dst) -> bool {
                auto it = tensor_map.find(name);
                if (it == tensor_map.end()) return true;
                const ParsedTensor* t = it->second;
                size_t expected = 1;
                for (size_t dim : t->shape) expected *= dim;
                dst.resize(expected);
                return load_tensor_data(path, data_offset, *t, dst.data(), dst.size() * sizeof(float), error);
            };

            Optimizer* optp = model.getMutableSerializedOptimizer();
            if (!optp) {
                Optimizer tmp;
                model.setSerializedOptimizer(tmp);
                optp = model.getMutableSerializedOptimizer();
            }

            if (!load_opt_vec("optimizer/m", optp->m)) return false;
            if (!load_opt_vec("optimizer/v", optp->v)) return false;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Model apply error: ") + e.what();
        }
        return false;
    }
}

uint64_t SafeTensorsReader::read_u64_le(std::ifstream& f) {
    uint8_t bytes[8];
    f.read(reinterpret_cast<char*>(bytes), 8);
    
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= (static_cast<uint64_t>(bytes[i]) << (i * 8));
    }
    return value;
}

} // namespace Serialization
} // namespace Mimir
