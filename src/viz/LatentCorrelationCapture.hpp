#pragma once

#include "LatentCorrelation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace mimir::viz::correlation_capture {

// These helpers summarize immutable host activations for diagnostics only.
// They never run a layer, update a parameter, or change a model tensor.
enum class Layout { FeatureLast, ChannelFirst, Flat };

struct Source {
    const std::vector<float>* values = nullptr;
    std::string key, label;
    int layer = -1;
    Layout layout = Layout::Flat;
    size_t width = 0;
    size_t plane = 1;
    bool sequence = false;
    int priority = 4;
};

inline size_t sampleIndex(size_t count, size_t selected, size_t index) {
    if (selected <= 1) return (count - 1) / 2;
    const size_t denominator = selected - 1;
    return index * ((count - 1) / denominator) +
        index * ((count - 1) % denominator) / denominator;
}

// A bounded ordinal sample among VALID positions prevents padding from
// changing a contextual mean. No complete activation tensor is copied.
inline bool appendFeatures(const Source& source, size_t quota,
                           const std::vector<int>* token_ids, int padding_id,
                           const std::vector<int>* attention_mask,
                           LatentCorrelationObservation& observation,
                           const std::vector<float>* float_token_ids = nullptr) {
    constexpr size_t position_budget = 256;
    if (!source.values || source.values->empty() || !source.width || !quota) return false;
    const auto& values = *source.values;
    size_t positions = 1;
    if (source.layout == Layout::FeatureLast) {
        if (values.size() % source.width) return false;
        positions = values.size() / source.width;
    } else if (source.layout == Layout::ChannelFirst) {
        if (!source.plane || source.width > values.size() / source.plane ||
            values.size() % (source.width * source.plane)) return false;
        positions = values.size() / source.width; // batch * spatial plane
    } else if (source.width != values.size()) return false;

    const bool ids_match = source.sequence && token_ids && token_ids->size() == positions && padding_id >= 0;
    const bool float_ids_match = source.sequence && !ids_match && float_token_ids &&
        float_token_ids->size() == positions && padding_id >= 0;
    const bool mask_matches = source.sequence && attention_mask && attention_mask->size() == positions;
    if (float_ids_match) for (float value : *float_token_ids) {
        // The legacy float Embedding path uses llround then int conversion.
        // Reject invalid IDs rather than invoking that conversion out of range
        // or treating a nonfinite token as an ordinary padding position.
        if (!std::isfinite(value) || static_cast<double>(value) < std::numeric_limits<int>::min() ||
            static_cast<double>(value) > std::numeric_limits<int>::max()) return false;
    }
    auto valid = [&](size_t position) {
        return (!ids_match || (*token_ids)[position] != padding_id) &&
            (!float_ids_match || std::llround(static_cast<double>((*float_token_ids)[position])) != padding_id) &&
            (!mask_matches || (*attention_mask)[position] != 0);
    };
    size_t valid_count = positions;
    if (ids_match || float_ids_match || mask_matches) {
        valid_count = 0;
        for (size_t position = 0; position < positions; ++position) valid_count += valid(position);
    }
    if (!valid_count) return false;
    const size_t selected_positions = std::min(valid_count, position_budget);
    std::vector<size_t> sampled_positions;
    sampled_positions.reserve(selected_positions);
    if (!ids_match && !float_ids_match && !mask_matches) {
        for (size_t index = 0; index < selected_positions; ++index)
            sampled_positions.push_back(sampleIndex(positions, selected_positions, index));
    } else {
        size_t ordinal = 0, selected = 0;
        for (size_t position = 0; position < positions && selected < selected_positions; ++position) {
            if (!valid(position)) continue;
            if (ordinal == sampleIndex(valid_count, selected_positions, selected)) {
                sampled_positions.push_back(position);
                ++selected;
            }
            ++ordinal;
        }
    }

    const size_t features = std::min(source.width, quota);
    const char* pooling = source.layout == Layout::Flat ? "composante plate (sans moyenne)" :
        source.layout == Layout::ChannelFirst ? "moyenne par canal (batch et positions spatiales)" :
        source.sequence ? "moyenne par feature des positions valides" : "moyenne par feature des lignes";
    // Stage the source so a nonfinite coordinate cannot produce a partial
    // schema that changes silently from observation to observation.
    std::vector<LatentCorrelationFeature> descriptors;
    std::vector<float> pooled;
    descriptors.reserve(features); pooled.reserve(features);
    for (size_t selected = 0; selected < features; ++selected) {
        const size_t feature = sampleIndex(source.width, features, selected);
        double sum = 0.0;
        for (size_t position : sampled_positions) {
            size_t offset = feature;
            if (source.layout == Layout::FeatureLast) offset = position * source.width + feature;
            else if (source.layout == Layout::ChannelFirst) {
                const size_t batch = position / source.plane;
                offset = (batch * source.width + feature) * source.plane + position % source.plane;
            }
            const float value = values[offset];
            if (!std::isfinite(value)) return false;
            sum += static_cast<double>(value);
        }
        const double mean = sum / sampled_positions.size();
        if (!std::isfinite(mean) || std::abs(mean) > std::numeric_limits<float>::max()) return false;
        LatentCorrelationFeature descriptor;
        descriptor.key = source.key + "/" + std::to_string(static_cast<int>(source.layout)) +
            "/" + std::to_string(source.width) + "/" + std::to_string(feature);
        descriptor.label = source.label + " · " +
            (source.layout == Layout::ChannelFirst ? "canal " : "feature ") + std::to_string(feature);
        descriptor.pooling = pooling;
        descriptor.layer = source.layer;
        descriptor.feature_index = feature;
        descriptor.original_dimension = source.width;
        descriptors.push_back(std::move(descriptor));
        pooled.push_back(static_cast<float>(mean));
    }
    observation.features.insert(observation.features.end(), descriptors.begin(), descriptors.end());
    observation.values.insert(observation.values.end(), pooled.begin(), pooled.end());
    return true;
}

inline void hashValue(uint64_t& hash, uint64_t value) {
    for (unsigned byte = 0; byte < 8; ++byte) {
        hash ^= static_cast<uint8_t>(value >> (byte * 8));
        hash *= UINT64_C(1099511628211);
    }
}

} // namespace mimir::viz::correlation_capture
