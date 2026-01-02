#pragma once

#include "Serialization.hpp"
#include "../include/json.hpp"
#include <vector>
#include <unordered_map>

class Model;
struct Layer;  // Forward declaration

namespace Mimir {
namespace Serialization {

using json = nlohmann::json;

/**
 * Enhanced DebugJson options (v1.1.0)
 */
struct DebugJsonOptions {
    bool include_gradients = false;
    bool include_optimizer_state = false;
    size_t max_values_per_tensor = 20;
    bool include_activations = false;
    bool include_checksums = false;
    bool include_weight_deltas = false;
    bool include_git_info = true;
    bool save_tokenizer = false;
    bool save_encoder = false;
};

/**
 * DebugJsonDump - Create debug JSON dumps of models
 * 
 * v1.1.0: Enhanced with layer configs, real shapes, gradients, weight deltas.
 * For development and debugging only. NOT for production use.
 * Saves model structure + truncated tensor data + statistics.
 */
class DebugJsonDump {
public:
    DebugJsonDump();
    ~DebugJsonDump();
    
    /**
     * Save model as debug JSON dump.
     */
    bool save(
        Model& model,
        const std::string& path,
        const SaveOptions& options,
        std::string* error = nullptr
    );
    
    /**
     * Save model as debug JSON dump (enhanced v1.1.0).
     */
    bool save_enhanced(
        const std::string& path,
        const Model& model,
        const DebugJsonOptions& options,
        std::string* error = nullptr
    );
    
private:
    struct TensorStats {
        float min;
        float max;
        float mean;
        float std;
        size_t total_elements;
        double l2_norm;
    };
    
    struct WeightDelta {
        double delta_l2_norm;
        double delta_max_abs;
        bool changed;
        double relative_change;
    };
    
    /**
     * Calculate tensor statistics.
     */
    TensorStats calculate_stats(const float* data, size_t size);
    
    /**
     * Calculate weight delta between snapshots.
     */
    WeightDelta calculate_delta(const float* before, const float* after, size_t size);
    
    /**
     * Compute checksum for tensor (simple hash for change detection).
     */
    uint64_t compute_checksum(const float* data, size_t size);
    
    /**
     * Build JSON representation.
     */
    json build_json(
        Model& model,
        const SaveOptions& options
    );
    
    /**
     * Build enhanced JSON representation (v1.1.0).
     */
    json build_json_enhanced(
        const Model& model,
        const DebugJsonOptions& options
    );
    
    /**
     * Extract layer config as JSON.
     */
    json extract_layer_config(const Layer& layer);
    
    /**
     * Get real tensor shape (multi-dimensional).
     */
    std::vector<size_t> get_tensor_shape(const Layer& layer, bool is_bias = false);
    
    /**
     * Add tensor info to JSON (with truncation).
     */
    void add_tensor_info(
        json& parent,
        const std::string& name,
        const float* data,
        size_t size,
        size_t max_values
    );
    
    /**
     * Add tensor with shape and gradients.
     */
    void add_tensor_enhanced(
        json& parent,
        const std::string& name,
        const float* data,
        const std::vector<size_t>& shape,
        const float* grad_data,
        size_t max_values,
        bool include_grads
    );
    
    // Weight snapshots for delta computation
    std::unordered_map<std::string, std::vector<float>> weight_snapshots_;
};

} // namespace Serialization
} // namespace Mimir
