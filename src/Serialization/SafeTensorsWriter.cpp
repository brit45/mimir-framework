#include "SafeTensorsWriter.hpp"
#include "../Model.hpp"
#include "../Tokenizer.hpp"
#include "../Encoder.hpp"
#include "../Models/VAEModel.hpp"
#include <fstream>
#include <cstring>
#include <iomanip>
#include <ctime>

namespace Mimir {
namespace Serialization {

SafeTensorsWriter::SafeTensorsWriter() {
}

SafeTensorsWriter::~SafeTensorsWriter() {
}

bool SafeTensorsWriter::save(
    Model& model,
    const std::string& path,
    const SaveOptions& options,
    std::string* error
) {
    try {
        // Collect all tensors
        std::vector<TensorData> tensors = collect_tensors(model, options);
        
        if (tensors.empty()) {
            if (error) {
                *error = "No tensors to save";
            }
            return false;
        }
        
        // Build header
        json header = build_header(tensors, options);
        
        // Write file
        return write_file(path, header, tensors, error);
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("SafeTensors write error: ") + e.what();
        }
        return false;
    }
}

std::vector<SafeTensorsWriter::TensorData> SafeTensorsWriter::collect_tensors(
    Model& model,
    const SaveOptions& options
) {
    std::vector<TensorData> tensors;

    // Reset owned buffers for this save call
    owned_buffers_.clear();
    
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
        td.name = layer.name + "/weights";
        td.dtype = DType::Float32;
        td.shape = {size};  // 1D tensor
        td.byte_size = size * sizeof(float);
        td.data_ptr = data_ptr;
        
        tensors.push_back(td);
    }
    
    // Add encoder embeddings if present
    if (options.save_encoder && model.getHasEncoder()) {
        const auto& enc = model.getEncoder();
        if (!enc.token_embeddings.empty()) {
            TensorData td;
            td.name = "encoder/token_embeddings";
            td.dtype = DType::Float32;
            td.shape = {static_cast<size_t>(enc.vocab_size), static_cast<size_t>(enc.dim)};
            td.byte_size = enc.token_embeddings.size() * sizeof(float);
            td.data_ptr = enc.token_embeddings.data();
            tensors.push_back(td);
        }
    }

    // Add tokenizer JSON (as raw bytes)
    if (options.save_tokenizer) {
        const auto& tok = model.getTokenizer();
        std::string tok_str = tok.to_json().dump();
        owned_buffers_.push_back(std::vector<uint8_t>(tok_str.begin(), tok_str.end()));
        TensorData td;
        td.name = "tokenizer/json";
        td.dtype = DType::Uint8;
        td.shape = {owned_buffers_.back().size()};
        td.byte_size = owned_buffers_.back().size();
        td.data_ptr = owned_buffers_.back().data();
        tensors.push_back(td);
    }

    // Add encoder JSON (dim/vocab + special embeddings) if requested
    if (options.save_encoder && model.getHasEncoder()) {
        const auto& enc = model.getEncoder();
        std::string enc_str = enc.to_json().dump();
        owned_buffers_.push_back(std::vector<uint8_t>(enc_str.begin(), enc_str.end()));
        TensorData td;
        td.name = "encoder/json";
        td.dtype = DType::Uint8;
        td.shape = {owned_buffers_.back().size()};
        td.byte_size = owned_buffers_.back().size();
        td.data_ptr = owned_buffers_.back().data();
        tensors.push_back(td);
    }

    // Add optimizer state (if requested and available)
    if (options.save_optimizer) {
        if (const Optimizer* opt = model.getSerializedOptimizer()) {
            // JSON meta
            {
                json j;
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

                std::string s = j.dump();
                owned_buffers_.push_back(std::vector<uint8_t>(s.begin(), s.end()));
                TensorData td;
                td.name = "optimizer/json";
                td.dtype = DType::Uint8;
                td.shape = {owned_buffers_.back().size()};
                td.byte_size = owned_buffers_.back().size();
                td.data_ptr = owned_buffers_.back().data();
                tensors.push_back(td);
            }

            // State vectors
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

    // Gradient snapshot tensors (debug-only). For now, implemented for VAEModel.
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

json SafeTensorsWriter::build_header(
    const std::vector<TensorData>& tensors,
    const SaveOptions& options
) {
    json header;
    
    // Calculate offsets
    size_t current_offset = 0;
    
    for (const auto& tensor : tensors) {
        json tensor_entry;
        tensor_entry["dtype"] = dtype_to_string(tensor.dtype);
        tensor_entry["shape"] = tensor.shape;
        
        // data_offsets: [begin, end) relative to start of data section
        json offsets = json::array();
        offsets.push_back(current_offset);
        offsets.push_back(current_offset + tensor.byte_size);
        tensor_entry["data_offsets"] = offsets;
        
        header[tensor.name] = tensor_entry;
        current_offset += tensor.byte_size;
    }
    
    // Add metadata
    json metadata;
    metadata["format"] = "safetensors";
    metadata["format_version"] = "0.3.0";
    metadata["mimir_version"] = get_mimir_version();
    
    if (options.include_git_info) {
        std::string git_commit = get_git_commit();
        if (!git_commit.empty()) {
            metadata["git_commit"] = git_commit;
        }
    }
    
    // Timestamp
    auto now = std::time(nullptr);
    metadata["created_at"] = static_cast<long long>(now);
    
    // Total size
    metadata["total_size"] = current_offset;
    
    // Custom metadata
    if (!options.custom_metadata.empty()) {
        try {
            json custom = json::parse(options.custom_metadata);
            metadata["custom"] = custom;
        } catch (...) {
            // Ignore invalid JSON
        }
    }
    
    header["__metadata__"] = metadata;
    
    return header;
}

bool SafeTensorsWriter::write_file(
    const std::string& path,
    const json& header,
    const std::vector<TensorData>& tensors,
    std::string* error
) {
    try {
        // Ensure directory exists
        fs::path file_path(path);
        if (file_path.has_parent_path()) {
            fs::create_directories(file_path.parent_path());
        }
        
        // Open file for binary writing
        std::ofstream file(path, std::ios::binary);
        if (!file) {
            if (error) {
                *error = "Failed to open file for writing: " + path;
            }
            return false;
        }
        
        // Serialize header to JSON string
        std::string header_str = header.dump();
        uint64_t header_len = static_cast<uint64_t>(header_str.size());
        
        // Write header length (8 bytes, little-endian)
        write_u64_le(file, header_len);
        
        // Write header JSON
        file.write(header_str.c_str(), header_len);
        if (!file) {
            if (error) {
                *error = "Failed to write header";
            }
            return false;
        }
        
        // Write tensor data in order
        for (const auto& tensor : tensors) {
            if (tensor.data_ptr == nullptr || tensor.byte_size == 0) {
                if (error) {
                    *error = "Invalid tensor data for: " + tensor.name;
                }
                return false;
            }
            
            // Write raw bytes (already in little-endian on x86/ARM)
            file.write(static_cast<const char*>(tensor.data_ptr), tensor.byte_size);
            if (!file) {
                if (error) {
                    *error = "Failed to write tensor data: " + tensor.name;
                }
                return false;
            }
        }
        
        file.close();
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("File write exception: ") + e.what();
        }
        return false;
    }
}

void SafeTensorsWriter::write_u64_le(std::ofstream& f, uint64_t value) {
    uint8_t bytes[8];
    for (int i = 0; i < 8; ++i) {
        bytes[i] = static_cast<uint8_t>((value >> (i * 8)) & 0xFF);
    }
    f.write(reinterpret_cast<const char*>(bytes), 8);
}

size_t SafeTensorsWriter::calculate_total_size(
    const std::vector<TensorData>& tensors
) const {
    size_t total = 0;
    for (const auto& tensor : tensors) {
        total += tensor.byte_size;
    }
    return total;
}

} // namespace Serialization
} // namespace Mimir
