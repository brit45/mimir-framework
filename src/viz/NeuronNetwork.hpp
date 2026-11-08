#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace mimir::viz {

// These are graph units, not physical neuron coordinates. A convolutional
// channel and an embedding component are explicitly distinguished from a
// dense output unit. Unknown geometry remains an operation node.
struct NeuronLayer {
    int layer = -1;
    std::string name, type, output, kind;
    std::vector<std::string> inputs;
    size_t units = 0;
    size_t depth = 0;
    bool is_operation = true;
    bool is_input = false;
};

struct NeuronConnection {
    size_t source_layer = 0, target_layer = 0;
    std::string tensor;
};

struct NeuronNetworkSnapshot {
    std::vector<NeuronLayer> layers;
    std::vector<NeuronConnection> connections;
};

struct NeuronNetworkPoint {
    std::array<float, 3> position{{0.f, 0.f, 0.f}};
    std::string label, kind;
    int layer = -1;
    size_t node_index = 0;
    size_t unit_index = 0, layer_units = 0;
    bool is_operation = false;
};

// Links represent the layer-to-layer tensor routing. They are intentionally
// aggregated, not a fabricated list of individual learned synapses.
struct NeuronNetworkLink {
    size_t source = 0, target = 0;
    float weight = 0.f;
    bool weighted = false;
    bool aggregated = true;
};

struct NeuronNetworkLayout {
    std::vector<NeuronNetworkPoint> points;
    std::vector<NeuronNetworkLink> links;
    size_t total_units = 0, total_layers = 0, unknown_layers = 0;
    size_t total_connections = 0, omitted_layers = 0;
    bool sampled = false;
};

inline constexpr size_t neuronNetworkPointLimit = 4096;
inline constexpr size_t neuronNetworkLinkLimit = 8192;

inline size_t neuronSampleIndex(size_t count, size_t selected, size_t index) {
    if (!count) return 0;
    if (selected <= 1) return (count - 1) / 2;
    const size_t divisor = selected - 1;
    return index * ((count - 1) / divisor) + index * ((count - 1) % divisor) / divisor;
}

inline size_t neuronSaturatingAdd(size_t a, size_t b) {
    return b > std::numeric_limits<size_t>::max() - a ? std::numeric_limits<size_t>::max() : a + b;
}

// X is graph depth. Concurrent branches receive distinct Y lanes; each
// layer's sampled units occupy a small Y/Z grid. No PCA or invented spatial
// model geometry is involved, and no tensor/parameter storage is read here.
inline NeuronNetworkLayout layoutNeuronNetwork3D(const NeuronNetworkSnapshot& snapshot,
                                                size_t max_points = neuronNetworkPointLimit,
                                                size_t max_links = neuronNetworkLinkLimit) {
    NeuronNetworkLayout result;
    result.total_layers = snapshot.layers.size();
    result.total_connections = snapshot.connections.size();
    max_points = std::min(max_points, neuronNetworkPointLimit);
    max_links = std::min(max_links, neuronNetworkLinkLimit);
    for (const auto& layer : snapshot.layers) {
        if (layer.is_operation || !layer.units) ++result.unknown_layers;
        else result.total_units = neuronSaturatingAdd(result.total_units, layer.units);
    }
    if (snapshot.layers.empty() || !max_points) {
        result.omitted_layers = snapshot.layers.size();
        result.sampled = !snapshot.layers.empty();
        return result;
    }
    const size_t included = std::min(snapshot.layers.size(), max_points);
    result.omitted_layers = snapshot.layers.size() - included;
    std::vector<size_t> selected_layers(included), quotas(included, 1);
    for (size_t i = 0; i < included; ++i)
        selected_layers[i] = neuronSampleIndex(snapshot.layers.size(), included, i);
    size_t used = included;
    bool progress = true;
    while (used < max_points && progress) {
        progress = false;
        for (size_t i = 0; i < included && used < max_points; ++i) {
            const auto& layer = snapshot.layers[selected_layers[i]];
            const size_t desired = layer.is_operation ? 1 : std::max<size_t>(1, layer.units);
            if (quotas[i] < desired) { ++quotas[i]; ++used; progress = true; }
        }
    }
    std::vector<std::vector<size_t>> point_indices(snapshot.layers.size());
    std::vector<std::pair<size_t, size_t>> depth_lanes;
    size_t max_depth = 0;
    for (size_t i = 0; i < included; ++i) {
        const size_t depth = snapshot.layers[selected_layers[i]].depth;
        max_depth = std::max(max_depth, depth);
        auto found = std::find_if(depth_lanes.begin(), depth_lanes.end(),
            [&](const auto& item) { return item.first == depth; });
        if (found == depth_lanes.end()) depth_lanes.emplace_back(depth, 1);
        else ++found->second;
    }
    std::vector<std::pair<size_t, size_t>> used_lanes;
    for (size_t i = 0; i < included; ++i) {
        const auto& layer = snapshot.layers[selected_layers[i]];
        const auto total = std::find_if(depth_lanes.begin(), depth_lanes.end(),
            [&](const auto& item) { return item.first == layer.depth; })->second;
        auto lane = std::find_if(used_lanes.begin(), used_lanes.end(),
            [&](const auto& item) { return item.first == layer.depth; });
        size_t lane_index = 0;
        if (lane == used_lanes.end()) used_lanes.emplace_back(layer.depth, 1);
        else { lane_index = lane->second; ++lane->second; }
        const size_t count = quotas[i];
        const size_t side = static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(count))));
        const size_t rows = (count + side - 1) / side;
        for (size_t unit = 0; unit < count; ++unit) {
            NeuronNetworkPoint point;
            point.layer = layer.layer;
            point.node_index = selected_layers[i];
            point.kind = layer.kind.empty() ? (layer.is_operation ? "opération" : "composante") : layer.kind;
            point.is_operation = layer.is_operation || !layer.units;
            point.layer_units = layer.units;
            point.unit_index = point.is_operation ? 0 : neuronSampleIndex(layer.units, count, unit);
            const float grid_y = (float(unit / side) - float(rows - 1) * .5f) / float(std::max<size_t>(1, side - 1));
            const float grid_z = (float(unit % side) - float(side - 1) * .5f) / float(std::max<size_t>(1, side - 1));
            point.position = {{max_depth ? float(layer.depth) / float(max_depth) * 4.4f : 0.f,
                (float(lane_index) - float(total - 1) * .5f) * 1.5f + grid_y,
                grid_z}};
            point.label = (layer.is_input ? "Entrée " : "L" + std::to_string(layer.layer) + " ") +
                layer.name + " [" + layer.type + "]";
            if (point.is_operation) point.label += " · opération, largeur indéterminée";
            else point.label += " · " + point.kind + " " + std::to_string(point.unit_index) +
                " / " + std::to_string(layer.units);
            point.label += " · profondeur " + std::to_string(layer.depth);
            if (!layer.output.empty()) point.label += " · sortie " + layer.output;
            point_indices[selected_layers[i]].push_back(result.points.size());
            result.points.push_back(std::move(point));
        }
        if (!layer.is_operation && count < layer.units) result.sampled = true;
    }
    // Center and apply one global scale, preserving relative X/Y/Z distances.
    std::array<float, 3> minimum = result.points.front().position, maximum = minimum;
    for (const auto& point : result.points) for (size_t axis = 0; axis < 3; ++axis) {
        minimum[axis] = std::min(minimum[axis], point.position[axis]);
        maximum[axis] = std::max(maximum[axis], point.position[axis]);
    }
    float extent = 0.f;
    for (size_t axis = 0; axis < 3; ++axis) extent = std::max(extent, maximum[axis] - minimum[axis]);
    const float scale = extent > 0.f ? 4.4f / extent : 1.f;
    for (auto& point : result.points) for (size_t axis = 0; axis < 3; ++axis)
        point.position[axis] = (point.position[axis] - (minimum[axis] + maximum[axis]) * .5f) * scale;

    std::vector<size_t> drawable;
    for (size_t index = 0; index < snapshot.connections.size(); ++index) {
        const auto& link = snapshot.connections[index];
        if (link.source_layer < point_indices.size() && link.target_layer < point_indices.size() &&
            !point_indices[link.source_layer].empty() && !point_indices[link.target_layer].empty())
            drawable.push_back(index);
    }
    const size_t link_count = std::min(drawable.size(), max_links);
    for (size_t index = 0; index < link_count; ++index) {
        const auto& connection = snapshot.connections[drawable[neuronSampleIndex(drawable.size(), link_count, index)]];
        const auto& sources = point_indices[connection.source_layer];
        const auto& targets = point_indices[connection.target_layer];
        result.links.push_back({sources[(sources.size() - 1) / 2], targets[(targets.size() - 1) / 2], 0.f, false, true});
    }
    result.sampled = result.sampled || result.omitted_layers || result.links.size() < snapshot.connections.size();
    return result;
}

} // namespace mimir::viz
