#include "Model.hpp"
#include "DynamicTensorAllocator.hpp"
#include "viz/LiveWeightNetwork.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace {
using namespace mimir::viz;
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
Layer& push(Model& model, const char* name, const char* type,
            std::vector<std::string> inputs, const char* output) {
    model.push(name, type, 0);
    auto& layer = model.getMutableLayers().back();
    layer.inputs = std::move(inputs); layer.output = output;
    return layer;
}
void denseGoldenAndRouting() {
    Model model;
    auto& dense = push(model, "dense", "Linear", {"input"}, "hidden");
    dense.in_features = 3; dense.out_features = 2;
    dense.weights = {-3.f,-2.f,-1.f,0.f,1.f,2.f,3.f,4.f};
    const auto before = dense.weights;
    const auto config = model.modelConfig;
    const auto snapshot = model.captureLiveWeightNetworkSnapshot();
    check(snapshot.weights.size() == 8 && snapshot.total_parameters == 8 && !snapshot.sampled,
          "small dense parameters were omitted");
    for (const auto& sample : snapshot.weights) {
        check(sample.value == before[sample.parameter_index], "capture changed a scalar coefficient");
        check(!sample.key.empty() && sample.target_layer < snapshot.network.layers.size(), "coefficient lacks provenance");
        if (sample.parameter_index < 6) {
            check(sample.mapped && !sample.is_bias && sample.input_unit == sample.parameter_index % 3 &&
                  sample.output_unit == sample.parameter_index / 3, "dense O,I storage was transposed");
            check(snapshot.network.layers[sample.source_layer].is_input, "dense does not connect its actual input");
        } else check(sample.is_bias && !sample.mapped && sample.output_unit == sample.parameter_index - 6,
                     "bias was fabricated into a synapse");
    }
    check(model.getLayers()[0].weights == before && model.modelConfig == config && model.tensor_store.empty(),
          "capture mutated the model");
    push(model, "activation", "GELU", {"hidden"}, "active");
    auto& unrelated = push(model, "unrelated", "Linear", {"input"}, "other");
    unrelated.in_features = 3; unrelated.out_features = 2; unrelated.use_bias = false;
    unrelated.weights.assign(6, 2.f);
    auto& output = push(model, "output", "Linear", {"active"}, "x");
    output.in_features = 2; output.out_features = 2; output.use_bias = false;
    output.weights.assign(4, 5.f);
    const auto routed = model.captureLiveWeightNetworkSnapshot();
    for (const auto& sample : routed.weights) if (sample.layer == 3)
        check(sample.mapped && routed.network.layers[sample.source_layer].layer == 1,
              "weighted edge followed execution adjacency instead of real routing");
}
void convolutionGeometry() {
    for (const char* type : {"Conv3d", "Conv4d"}) {
        Model model;
        auto& layer = push(model, "grouped", type, {"image"}, "x");
        layer.in_channels = layer.out_channels = 4; layer.groups = 2;
        layer.input_spatial_shape = std::string(type) == "Conv3d" ? std::vector<int>{2,2,2} : std::vector<int>{1,2,2,2};
        layer.kernel_shape = std::string(type) == "Conv3d" ? std::vector<int>{1,1,2} : std::vector<int>{1,1,1,2};
        layer.weights.resize(20);
        for (size_t i = 0; i < layer.weights.size(); ++i) layer.weights[i] = static_cast<float>(i) - 9.f;
        const auto snapshot = model.captureLiveWeightNetworkSnapshot();
        check(snapshot.weights.size() == 20, "ConvND coefficient count incorrect");
        for (const auto& sample : snapshot.weights) {
            check(sample.value == static_cast<float>(sample.parameter_index) - 9.f, "ConvND coefficient value changed");
            if (sample.parameter_index < 16) {
                const size_t output = sample.parameter_index / 4;
                const size_t input = (output / 2) * 2 + (sample.parameter_index / 2) % 2;
                check(sample.mapped && sample.output_unit == output && sample.input_unit == input &&
                      sample.output_unit / 2 == sample.input_unit / 2, "grouped ConvND mapped across groups");
                check(sample.label.find("tap=") != std::string::npos, "kernel tap provenance absent");
            } else check(sample.is_bias && !sample.mapped, "ConvND bias is not identified");
        }
    }
    Model causal;
    auto& depthwise = push(causal, "causal", "CausalDepthwiseConv1d", {"features"}, "x");
    depthwise.in_features = depthwise.out_features = 3; depthwise.kernel_size = 2; depthwise.seq_len = 0;
    depthwise.weights.assign(9, .5f);
    const auto snapshot = causal.captureLiveWeightNetworkSnapshot();
    check(snapshot.weights.size() == 9, "causal convolution needs no live context to expose parameters");
    for (const auto& sample : snapshot.weights) if (!sample.is_bias)
        check(sample.mapped && sample.input_unit == sample.output_unit, "depthwise mapping crossed channels");
}
void packedSharedColdAndNonfinite() {
    Model packed;
    auto& attention = push(packed, "attention", "MultiHeadAttention", {"input"}, "x");
    attention.embed_dim = 4; attention.weights.assign(19, -.25f);
    const auto unknown = packed.captureLiveWeightNetworkSnapshot();
    check(unknown.weights.size() == 19, "packed parameters were hidden");
    for (const auto& sample : unknown.weights)
        check(!sample.mapped && !sample.is_bias && sample.value == -.25f,
              "packed attention storage was fabricated into a dense matrix");

    tensor shared(std::vector<float>{1.f,2.f,3.f,4.f});
    Model tied;
    auto& owner = push(tied, "owner", "Linear", {"input"}, "first");
    owner.in_features = owner.out_features = 2; owner.use_bias = false; owner.weight_block = &shared;
    auto& alias = push(tied, "alias", "Linear", {"first"}, "x");
    alias.in_features = alias.out_features = 2; alias.use_bias = false; alias.weight_block = &shared;
    const auto duplicate = tied.captureLiveWeightNetworkSnapshot();
    check(duplicate.weights.size() == 4 && duplicate.total_parameters == 4 && duplicate.shared_layers == 1,
          "shared parameter storage counted twice");
    tied.getMutableLayers()[1].weight_block = nullptr;
    tied.getMutableLayers()[1].weights.assign(4, 99.f);
    tied.getMutableLayers()[1].shared_weights_from = "owner";
    const auto declared = tied.captureLiveWeightNetworkSnapshot();
    check(declared.weights.size() == 4 && declared.shared_layers == 1, "declared alias provisional values leaked");

    tensor cold(32, true);
    const auto* handle = static_cast<const DynamicTensorAllocator::TensorHandle*>(cold.dynamic_handle);
    check(handle && !handle->is_loaded && !handle->data_ptr, "cold fixture was hydrated before capture");
    Model unloaded;
    auto& dormant = push(unloaded, "cold", "Linear", {"input"}, "x");
    dormant.in_features = 4; dormant.out_features = 8; dormant.use_bias = false; dormant.weight_block = &cold;
    const auto missing = unloaded.captureLiveWeightNetworkSnapshot();
    check(missing.weights.empty() && missing.total_parameters == 32 && missing.unavailable_layers == 1 &&
          !handle->is_loaded && !handle->data_ptr, "diagnostic hydrated cold parameter storage");

    Model invalid;
    auto& finite = push(invalid, "finite", "Linear", {"input"}, "x");
    finite.in_features = finite.out_features = 2; finite.use_bias = false;
    finite.weights = {std::numeric_limits<float>::quiet_NaN(),0.f,std::numeric_limits<float>::infinity(),-1.f};
    const auto filtered = invalid.captureLiveWeightNetworkSnapshot();
    check(filtered.weights.size() == 2 && filtered.nonfinite_parameters == 2 && filtered.total_parameters == 4,
          "nonfinite selected weights were published or miscounted");
    check(filtered.weights[0].value == 0.f, "zero weight disappeared from the diagnostic");
}
void boundedVisibleStableSampling() {
    Model model;
    auto& layer = push(model, "wide", "Linear", {"input"}, "x");
    layer.in_features = 4096; layer.out_features = 2; layer.use_bias = false;
    layer.weights.resize(8192);
    for (size_t i = 0; i < layer.weights.size(); ++i) layer.weights[i] = static_cast<float>(i % 67) * .01f;
    const auto first = model.captureLiveWeightNetworkSnapshot();
    check(first.weights.size() == liveWeightCoefficientLimit && first.sampled && first.total_parameters == 8192,
          "wide layer exceeded sampling budget");
    check(first.weights.front().parameter_index == 0 && first.weights.back().parameter_index == 8191,
          "bounded sampling omitted physical endpoint coefficients");
    const auto nodes = layoutNeuronNetwork3D(first.network, liveWeightNeuronPointLimit, liveWeightStructuralLinkLimit);
    std::unordered_set<size_t> shown_inputs;
    for (const auto& point : nodes.points) if (point.layer < 0) shown_inputs.insert(point.unit_index);
    for (const auto& sample : first.weights)
        check(sample.mapped && shown_inputs.count(sample.input_unit), "sampled synapse points to an invisible source unit");
    for (auto& value : model.getMutableLayers()[0].weights) value += .25f;
    const auto second = model.captureLiveWeightNetworkSnapshot();
    check(first.weights.size() == second.weights.size(), "sampling changed its size after an update");
    for (size_t i = 0; i < first.weights.size(); ++i) {
        check(first.weights[i].key == second.weights[i].key &&
              first.weights[i].parameter_index == second.weights[i].parameter_index,
              "weight magnitude changed sampled parameter identities");
        check(std::abs((second.weights[i].value - first.weights[i].value) - .25f) < 1e-6f,
              "updated coefficient was not captured exactly");
    }
}
}
int main() {
    try {
        denseGoldenAndRouting(); convolutionGeometry(); packedSharedColdAndNonfinite(); boundedVisibleStableSampling();
        std::cout << "LIVE_WEIGHT_NETWORK_CAPTURE_OK\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "LIVE_WEIGHT_NETWORK_CAPTURE_FAILED: " << error.what() << '\n';
        return 1;
    }
}
