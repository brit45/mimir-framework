#include "Model.hpp"
#include "viz/NeuronNetwork.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using namespace mimir::viz;
void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
const NeuronLayer& capturedLayer(const NeuronNetworkSnapshot& snapshot, int layer) {
    const auto found = std::find_if(snapshot.layers.begin(), snapshot.layers.end(),
        [&](const auto& value) { return value.layer == layer && !value.is_input; });
    if (found == snapshot.layers.end()) throw std::runtime_error("missing model layer");
    return *found;
}
size_t capturedIndex(const NeuronNetworkSnapshot& snapshot, int layer) {
    return static_cast<size_t>(&capturedLayer(snapshot, layer) - snapshot.layers.data());
}
bool linked(const NeuronNetworkSnapshot& snapshot, int source, int target) {
    const size_t from = capturedIndex(snapshot, source), to = capturedIndex(snapshot, target);
    return std::any_of(snapshot.connections.begin(), snapshot.connections.end(),
        [&](const auto& link) { return link.source_layer == from && link.target_layer == to; });
}
Layer& push(Model& model, const char* name, const char* type,
            std::vector<std::string> inputs, const char* output) {
    model.push(name, type, 0);
    auto& layer = model.getMutableLayers().back();
    layer.inputs = std::move(inputs); layer.output = output;
    return layer;
}
void exactMLPAndSkip() {
    Model model;
    auto& first = push(model, "dense", "Linear", {"__input__"}, "hidden");
    first.in_features = 3; first.out_features = 4;
    push(model, "activation", "ReLU", {"hidden"}, "activated");
    auto& branch = push(model, "branch", "Linear", {"__input__"}, "branch_out");
    branch.in_features = 3; branch.out_features = 4;
    push(model, "merge", "Add", {"activated", "branch_out"}, "merged");
    push(model, "dropout", "Dropout", {"merged"}, "x");
    const auto config = model.modelConfig;
    const auto snapshot = model.captureNeuronNetworkSnapshot();
    check(snapshot.layers.size() == 6, "external input or real layer missing");
    check(snapshot.connections.size() == 6, "routed edge count mismatch");
    check(capturedLayer(snapshot, 0).units == 4 && capturedLayer(snapshot, 1).units == 4,
          "dense or propagated activation width is wrong");
    check(capturedLayer(snapshot, 4).units == 4, "dropout lost feature geometry");
    check(capturedLayer(snapshot, 0).depth == 1 && capturedLayer(snapshot, 2).depth == 1 &&
          capturedLayer(snapshot, 3).depth == 3, "graph depth followed execution adjacency");
    check(linked(snapshot, 1, 3) && linked(snapshot, 2, 3) && !linked(snapshot, 1, 2),
          "skip routing was replaced by sequential adjacency");
    const auto layout = layoutNeuronNetwork3D(snapshot);
    check(layout.total_units == 23 && layout.points.size() == 23 && !layout.sampled,
          "small known graph should expose every unit");
    check(layout.total_layers == 6 && layout.unknown_layers == 0, "known MLP marked unknown");
    check(layout.links.size() == snapshot.connections.size(), "small routing links omitted");
    for (const auto& link : layout.links) {
        check(!link.weighted && link.aggregated && link.weight == 0.f, "structural links claim learned weights");
        check(link.source < layout.points.size() && link.target < layout.points.size(), "link out of bounds");
    }
    for (const auto& point : layout.points) for (float value : point.position)
        check(std::isfinite(value) && std::abs(value) <= 2.21f, "layout out of bounds");
    check(config == model.modelConfig && model.tensor_store.empty(), "metadata capture changed model state");
    // This capture is available before parameter allocation, so it cannot
    // accidentally depend on initialized weights or execute a neural forward.
    for (const auto& layer : model.getLayers())
        check(!layer.weight_block && layer.weights.empty() && layer.grad_weights.empty(), "capture allocated numerical state");
}
void geometryAndUnknown() {
    Model model;
    auto& conv3 = push(model, "volume", "Conv3d", {"image"}, "volume");
    conv3.in_channels = 2; conv3.out_channels = 7;
    auto& conv4 = push(model, "space_time", "Conv4d", {"volume"}, "spacetime");
    conv4.in_channels = 7; conv4.out_channels = 5;
    auto& embedding = push(model, "embedding", "Embedding", {"ids"}, "embeddings");
    embedding.vocab_size = 100000; embedding.embed_dim = 11;
    push(model, "unknown_geometry", "Reshape", {"embeddings"}, "x");
    const auto snapshot = model.captureNeuronNetworkSnapshot();
    check(capturedLayer(snapshot, 0).units == 7 && capturedLayer(snapshot, 0).kind == "canal",
          "Conv3d must expose channels, not pretend voxel counts");
    check(capturedLayer(snapshot, 1).units == 5 && capturedLayer(snapshot, 1).kind == "canal",
          "Conv4d channel count wrong");
    check(capturedLayer(snapshot, 2).units == 11, "embedding confused vocabulary rows with features");
    check(capturedLayer(snapshot, 3).is_operation && !capturedLayer(snapshot, 3).units,
          "unknown geometry fabricated neurons");
    const auto layout = layoutNeuronNetwork3D(snapshot);
    check(layout.unknown_layers == 2, "unknown operation and token IDs should be explicit");
    for (const auto& point : layout.points) if (point.layer == 3)
        check(point.is_operation && !point.layer_units && point.label.find("indéterminée") != std::string::npos,
              "unknown operation is not visibly distinguished");
}
void backwardWidthConstraints() {
    Model model;
    push(model, "input", "Identity", {"__input__"}, "x");
    push(model, "input_activation", "GELU", {"x"}, "prepared");
    auto& dense = push(model, "dense", "Linear", {"prepared"}, "hidden");
    dense.in_features = 4; dense.out_features = 7;
    push(model, "activation", "GELU", {"hidden"}, "x");
    const auto snapshot = model.captureNeuronNetworkSnapshot();
    check(capturedLayer(snapshot, 0).units == 4 && capturedLayer(snapshot, 1).units == 4,
          "consumer dimensions did not resolve preceding unary operations");
    check(snapshot.layers.front().is_input && snapshot.layers.front().units == 4,
          "unary input geometry did not reach external input");
    const auto layout = layoutNeuronNetwork3D(snapshot);
    check(layout.unknown_layers == 0 && layout.total_units == 26 && layout.points.size() == 26,
          "known unary input graph still has unknown dimensions");

    // Two consumers with incompatible declarations must remain a conflict;
    // the second pass may not silently pick one inferred input width.
    auto& incompatible = push(model, "incompatible", "Linear", {"prepared"}, "other");
    incompatible.in_features = 9; incompatible.out_features = 2;
    const auto conflict = model.captureNeuronNetworkSnapshot();
    check(capturedLayer(conflict, 0).is_operation && capturedLayer(conflict, 1).is_operation &&
          conflict.layers.front().is_operation, "bidirectional propagation concealed a width conflict");

    Model broadcast;
    push(broadcast, "sum", "Add", {"a", "b"}, "sum");
    auto& consumer = push(broadcast, "consumer", "Linear", {"sum"}, "x");
    consumer.in_features = 4; consumer.out_features = 2;
    const auto added = broadcast.captureNeuronNetworkSnapshot();
    check(capturedLayer(added, 0).units == 4, "consumer-proven output width was discarded");
    for (const auto& node : added.layers) if (node.is_input)
        check(node.is_operation && node.units == 0, "output width was incorrectly propagated to broadcast operands");

    Model spatial;
    auto& conv = push(spatial, "map", "Conv2d", {"image"}, "map");
    conv.in_channels = 1; conv.out_channels = 2;
    conv.input_height = 1; conv.input_width = 2;
    auto& norm = push(spatial, "flat_norm", "LayerNorm", {"map"}, "x");
    norm.in_features = 4;
    const auto axes = spatial.captureNeuronNetworkSnapshot();
    check(capturedLayer(axes, 0).units == 2 && capturedLayer(axes, 0).kind == "canal" &&
          capturedLayer(axes, 1).units == 4 && capturedLayer(axes, 1).kind == "composante",
          "normalization incorrectly equated channel and flattened-component counts");
}
void overwrittenAndAmbiguous() {
    Model model;
    auto& a = push(model, "a", "Linear", {}, "x"); a.in_features = 2; a.out_features = 3;
    auto& b = push(model, "b", "Linear", {}, "x"); b.in_features = 3; b.out_features = 4;
    auto& c = push(model, "c", "Linear", {"external"}, "branch"); c.in_features = 2; c.out_features = 4;
    auto& d = push(model, "d", "Linear", {"external"}, "end"); d.in_features = 7; d.out_features = 4;
    const auto snapshot = model.captureNeuronNetworkSnapshot();
    check(linked(snapshot, 0, 1), "overwritten x did not resolve most recent producer");
    const auto input = std::find_if(snapshot.layers.begin(), snapshot.layers.end(), [](const auto& layer) {
        return layer.is_input && layer.name == "external";
    });
    check(input != snapshot.layers.end() && input->is_operation && input->units == 0,
          "conflicting input dimensions presented as known");
}
void boundsAndDeterminism() {
    NeuronNetworkSnapshot snapshot;
    for (size_t layer = 0; layer < 32; ++layer) {
        NeuronLayer value;
        value.layer = static_cast<int>(layer); value.name = "wide" + std::to_string(layer);
        value.type = "Linear"; value.units = 1000000; value.depth = layer;
        value.kind = "neurone"; value.is_operation = false;
        snapshot.layers.push_back(value);
        if (layer) snapshot.connections.push_back({layer - 1, layer, "x"});
    }
    const auto first = layoutNeuronNetwork3D(snapshot), second = layoutNeuronNetwork3D(snapshot);
    check(first.points.size() == neuronNetworkPointLimit && first.total_units == 32000000 && first.sampled,
          "wide graph exceeds budget or lost exact count");
    check(first.points.size() == second.points.size() && first.links.size() == second.links.size(), "layout nondeterministic size");
    for (size_t index = 0; index < first.points.size(); ++index)
        check(first.points[index].position == second.points[index].position &&
              first.points[index].label == second.points[index].label, "layout nondeterministic placement");
    for (int layer = 0; layer < 32; ++layer) {
        size_t low = std::numeric_limits<size_t>::max(), high = 0, count = 0;
        for (const auto& point : first.points) if (point.layer == layer) {
            low = std::min(low, point.unit_index); high = std::max(high, point.unit_index); ++count;
        }
        check(count == 128 && low == 0 && high == 999999, "unit sampling is unfair or omits width endpoints");
    }
    snapshot.layers.resize(2);
    snapshot.connections.clear();
    for (size_t index = 0; index < 10000; ++index) snapshot.connections.push_back({0, 1, "route"});
    check(layoutNeuronNetwork3D(snapshot).links.size() == neuronNetworkLinkLimit, "link budget exceeded");
    snapshot.layers.resize(5000);
    const auto many_layers = layoutNeuronNetwork3D(snapshot);
    check(many_layers.points.size() == neuronNetworkPointLimit && many_layers.omitted_layers == 904,
          "layer budget not reported");
    check(layoutNeuronNetwork3D(snapshot, 0).points.empty(), "zero budget ignored");
}
} // namespace

int main() {
    try {
        exactMLPAndSkip(); geometryAndUnknown(); backwardWidthConstraints(); overwrittenAndAmbiguous(); boundsAndDeterminism();
        std::cout << "NEURON_NETWORK_CAPTURE_LAYOUT_OK\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "NEURON_NETWORK_FAILED: " << error.what() << '\n';
        return 1;
    }
}
