#pragma once

#include "NeuronNetwork.hpp"

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace mimir::viz {

inline constexpr size_t liveWeightCoefficientLimit = 2048;
inline constexpr size_t liveWeightNeuronPointLimit = 2048;
inline constexpr size_t liveWeightStructuralLinkLimit = 1024;
inline constexpr size_t liveWeightNoLayer = std::numeric_limits<size_t>::max();

// One exact scalar from resident model parameter storage. A mapping describes
// the real consuming/producing graph units only when their axes are verified.
// Packed parameters with unknown geometry remain useful scalar glyphs; they
// never acquire an invented dense synapse. Biases are not directed edges.
struct LiveWeightSample {
    std::string key;
    std::string label;
    int layer = -1;
    size_t parameter_index = 0;
    float value = 0.f;
    bool is_bias = false;
    bool mapped = false;
    size_t source_layer = liveWeightNoLayer;
    size_t target_layer = liveWeightNoLayer;
    size_t input_unit = 0;
    size_t output_unit = 0;
};

struct LiveWeightNetworkSnapshot {
    NeuronNetworkSnapshot network;
    std::vector<LiveWeightSample> weights;
    size_t total_parameters = 0;
    size_t unavailable_layers = 0;
    size_t nonfinite_parameters = 0; // Among selected coefficients, not full blocks.
    size_t shared_layers = 0;
    bool sampled = false;
};

} // namespace mimir::viz
