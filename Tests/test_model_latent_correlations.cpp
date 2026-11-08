#include "Model.hpp"
#include "viz/LatentCorrelationCapture.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace mimir::viz;
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}
std::vector<std::vector<float>> parameterState(const Model& model) {
    std::vector<std::vector<float>> state;
    for (const auto& layer : model.getLayers()) {
        const auto* values = layer.getWeights();
        state.push_back(values ? std::vector<float>(values, values + layer.getWeightsSize()) : std::vector<float>{});
        state.push_back(layer.grad_weights);
        state.push_back(layer.grad_bias);
    }
    return state;
}
void configureSequence(Model& model) {
    model.setModelName("Actual contextual activations");
    model.modelConfig = {{"type", "correlation_sequence_fixture"}, {"padding_idx", 0}};
    model.push("tokens", "Embedding", 16);
    auto* embedding = model.getLayerByName("tokens");
    embedding->inputs = {"__input__"}; embedding->output = "embeddings";
    embedding->vocab_size = 8; embedding->embed_dim = 2; embedding->padding_idx = 0;
    model.push("bottleneck/latent_down", "Linear", 4);
    auto* hidden = model.getLayerByName("bottleneck/latent_down");
    hidden->inputs = {"embeddings"}; hidden->output = "latent_code";
    hidden->in_features = hidden->out_features = 2; hidden->seq_len = 4; hidden->use_bias = false;
    model.push("latent_activation", "Identity", 0);
    auto* activation = model.getLayerByName("latent_activation");
    activation->inputs = {"latent_code"}; activation->output = "latent_activated";
    model.push("head", "Linear", 2);
    auto* head = model.getLayerByName("head");
    head->inputs = {"latent_activated"}; head->output = "x";
    head->in_features = 2; head->out_features = 1; head->seq_len = 4; head->use_bias = false;
    model.allocateParams(); model.initializeWeights("xavier", 19);
    embedding = model.getLayerByName("tokens");
    const float table[]{1000, 2000, 1, 3, 3, 7, 5, 11, 7, 15, 9, 19, 11, 23, 13, 27};
    std::copy(std::begin(table), std::end(table), embedding->getWeights());
    hidden = model.getLayerByName("bottleneck/latent_down");
    const float identity[]{1, 0, 0, 1}; std::copy(std::begin(identity), std::end(identity), hidden->getWeights());
    head = model.getLayerByName("head"); head->getWeights()[0] = .3f; head->getWeights()[1] = .4f;
}
float featureValue(const LatentCorrelationObservation& observation, int layer, size_t feature) {
    for (size_t index = 0; index < observation.features.size(); ++index)
        if (observation.features[index].layer == layer && observation.features[index].feature_index == feature)
            return observation.values[index];
    throw std::runtime_error("expected activation feature absent");
}
void sequenceForward(Model& model, const std::vector<int>& ids, bool training) {
    // Native sequence architectures update this runtime geometry for each
    // prompt. The captured feature schema must remain independent of it.
    for (auto& layer : model.getMutableLayers())
        if (layer.type_enum == LayerType::Linear) layer.seq_len = static_cast<int>(ids.size());
    model.forwardPass(ids, training);
}
void sequenceCase() {
    Model model; configureSequence(model);
    check(model.captureLatentCorrelationObservation(0, 1, true).values.empty(), "captured buffers before any forward");
    sequenceForward(model, {1, 0, 2, 0}, true);
    const auto parameters = parameterState(model);
    const auto config = model.modelConfig;
    const auto floats = model.tensor_store;
    const auto integers = model.tensor_store_int;
    const auto first = model.captureLatentCorrelationObservation(17, 1, true);
    check(first.values.size() == first.features.size() && !first.values.empty(), "empty/misaligned activation observation");
    check(first.values.size() <= 128, "exceeded feature budget");
    check(first.optimizer_step == 17 && first.forward_serial == 1 && first.training, "lost forward provenance");
    check(first.architecture == "correlation_sequence_fixture", "lost model architecture");
    check(first.features.front().layer == 1, "contextual latent source was not prioritized");
    check(std::abs(featureValue(first, 1, 0) - 2.f) < 1e-6f &&
          std::abs(featureValue(first, 1, 1) - 5.f) < 1e-6f, "padding contaminated activation means");
    check(first.context_fingerprint != 0, "small actual context has no fingerprint");
    const auto repeated = model.captureLatentCorrelationObservation(17, 1, true);
    check(first.values == repeated.values && first.context_fingerprint == repeated.context_fingerprint,
          "read-only observation is nondeterministic");
    const auto after = parameterState(model);
    check(parameters.size() == after.size(), "capture changed parameter topology");
    for (size_t index = 0; index < after.size(); ++index)
        check(same(parameters[index], after[index]), "capture changed parameters or gradients");
    check(config == model.modelConfig && integers == model.tensor_store_int, "capture changed configuration or token IDs");
    check(floats.size() == model.tensor_store.size(), "capture changed tensor count");
    for (const auto& entry : floats) check(same(entry.second, model.tensor_store.at(entry.first)), "capture changed activations");

    sequenceForward(model, {3, 1, 2}, true);
    const auto second = model.captureLatentCorrelationObservation(18, 2, true);
    check(first.features.size() == second.features.size(), "sequence length changed feature schema");
    for (size_t index = 0; index < first.features.size(); ++index)
        check(first.features[index].key == second.features[index].key, "sequence length changed feature identity");
    check(std::abs(featureValue(second, 1, 0) - 3.f) < 1e-6f &&
          std::abs(featureValue(second, 1, 1) - 7.f) < 1e-6f, "observation did not follow actual context");
    check(first.context_fingerprint != second.context_fingerprint, "different actual contexts share fingerprint");

    sequenceForward(model, {0, 0}, true);
    check(model.captureLatentCorrelationObservation(18, 3, true).values.empty(), "captured an all-padding context");
    sequenceForward(model, {2, 3}, false);
    // The old training forward_state is still valid. Remove inference internals
    // deliberately: the diagnostic must use the real current output fallback.
    model.tensor_store.erase("embeddings"); model.tensor_store.erase("latent_code");
    model.tensor_store.erase("latent_activated");
    const auto inference = model.captureLatentCorrelationObservation(18, 4, false);
    check(!inference.values.empty(), "no explicit final-output fallback");
    for (const auto& feature : inference.features) {
        check(feature.layer == 3, "inference consumed a stale training activation");
        check(feature.label.find("sortie finale") != std::string::npos, "fallback was presented as a hidden activation");
    }
    model.getTensorMutable("x")[0] = std::numeric_limits<float>::quiet_NaN();
    check(model.captureLatentCorrelationObservation(18, 4, false).values.empty(), "nonfinite source produced a partial feature schema");
}
void overwrittenNamesCase() {
    Model model; model.modelConfig["type"] = "overwritten_names";
    model.push("latent", "Linear", 4);
    auto& hidden = model.getMutableLayers().back();
    hidden.inputs = {"__input__"}; hidden.output = "x";
    hidden.in_features = hidden.out_features = 2; hidden.use_bias = false;
    model.push("out", "Linear", 2);
    auto& head = model.getMutableLayers().back();
    head.inputs = {"x"}; head.output = "x"; head.in_features = 2; head.out_features = 1; head.use_bias = false;
    model.allocateParams(); model.initializeWeights("xavier", 11);
    const float identity[]{1, 0, 0, 1}; std::copy(std::begin(identity), std::end(identity), model.getLayerByName("latent")->getWeights());
    model.forwardPass(std::vector<float>{3, 7}, true);
    const auto observation = model.captureLatentCorrelationObservation(0, 1, true);
    check(std::abs(featureValue(observation, 0, 0) - 3.f) < 1e-6f &&
          std::abs(featureValue(observation, 0, 1) - 7.f) < 1e-6f, "overwritten output was not recovered from its consumer");
    check(observation.features.front().label.find("entrée conservée") != std::string::npos,
          "recovered input provenance is missing");
}
void floatIdInferenceCase() {
    Model model; configureSequence(model);
    model.setVizTapsEnabled(true); // The live Viz route retains each real output.
    sequenceForward(model, {1, 0, 2, 0}, false);
    model.consumeVizTaps();
    const auto observation = model.captureLatentCorrelationObservation(0, 1, false);
    check(std::abs(featureValue(observation, 1, 0) - 2.f) < 1e-6f &&
          std::abs(featureValue(observation, 1, 1) - 5.f) < 1e-6f,
          "inference float-ID route included padding without a training cache");
}
void spatialCase(int rank) {
    Model model; model.modelConfig["type"] = "spatial_correlation";
    model.push("latent/mu", rank == 3 ? "Conv3d" : "Conv4d", 4);
    auto& conv = model.getMutableLayers().back();
    conv.inputs = {"__input__"}; conv.output = "mu"; conv.in_channels = conv.out_channels = 2;
    conv.kernel_size = 1; conv.use_bias = false;
    conv.input_spatial_shape.assign(size_t(rank), 2);
    conv.shape = {2, 2}; conv.shape.insert(conv.shape.end(), size_t(rank), 2);
    model.push("out", "Identity", 0);
    model.getMutableLayers().back().inputs = {"mu"}; model.getMutableLayers().back().output = "x";
    model.allocateParams(); model.initializeWeights("xavier", 8);
    const float identity[]{1, 0, 0, 1}; std::copy(std::begin(identity), std::end(identity), model.getLayerByName("latent/mu")->getWeights());
    const size_t plane = size_t{1} << rank;
    std::vector<float> input(4 * plane);
    for (size_t batch = 0; batch < 2; ++batch) for (size_t position = 0; position < plane; ++position) {
        const float value = float(batch * 1000 + position + 1);
        input[(batch * 2) * plane + position] = value;
        input[(batch * 2 + 1) * plane + position] = -2.f * value;
    }
    model.forwardPass(input, true);
    const auto observation = model.captureLatentCorrelationObservation(0, 1, true);
    const float expected = 500.f + float(plane + 1) / 2.f;
    check(std::abs(featureValue(observation, 0, 0) - expected) < 1e-6f &&
          std::abs(featureValue(observation, 0, 1) + 2.f * expected) < 1e-6f,
          "spatial pooling mixed feature channels or batches");
    check(observation.features.front().original_dimension == 2 &&
          observation.features.front().pooling.find("canal") != std::string::npos,
          "spatial feature geometry was invented");
}
void legacySpatialOutputCase() {
    Model model; model.modelConfig["type"] = "legacy_spatial_correlation";
    model.push("latent/conv", "Conv2d", 6);
    auto& conv = model.getMutableLayers().back();
    conv.inputs = {"__input__"}; conv.output = "mu";
    conv.in_channels = 1; conv.out_channels = 2; conv.input_height = 2; conv.input_width = 4;
    conv.kernel_h = 1; conv.kernel_w = 3; conv.stride_h = 1; conv.stride_w = 2; conv.use_bias = false;
    model.push("out", "Identity", 0);
    model.getMutableLayers().back().inputs = {"mu"}; model.getMutableLayers().back().output = "x";
    model.allocateParams(); model.initializeWeights("xavier", 31);
    auto* weights = model.getLayerByName("latent/conv")->getWeights(); weights[0] = .5f; weights[1] = 2.f;
    model.forwardPass(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, true);
    const auto& actual = model.getTensor("mu");
    check(!actual.empty() && actual.size() % 2 == 0, "invalid native legacy convolution output");
    const size_t plane = actual.size() / 2;
    const auto observation = model.captureLatentCorrelationObservation(0, 1, true);
    for (size_t channel = 0; channel < 2; ++channel) {
        double expected = 0;
        for (size_t position = 0; position < plane; ++position) expected += actual[channel * plane + position];
        expected /= plane;
        check(std::abs(double(featureValue(observation, 0, channel)) - expected) < 1e-6,
              "legacy diagnostic mixed channels using theoretical spatial geometry");
    }
    check(observation.features.front().original_dimension == 2 &&
          observation.features.front().pooling.find("canal") != std::string::npos,
          "actual legacy channels were presented as invented flat geometry");
    // A contradictory batch declaration has no reliable legacy C/spatial
    // interpretation. Expose actual flat components, clearly labeled.
    model.getLayerByName("latent/conv")->shape = {2, 1, 1, 4};
    const auto ambiguous = model.captureLatentCorrelationObservation(0, 1, true);
    check(!ambiguous.features.empty() && ambiguous.features.front().original_dimension == actual.size() &&
          ambiguous.features.front().pooling.find("plate") != std::string::npos,
          "ambiguous legacy batch metadata was assigned invented channels");
}
class CountingModel : public Model {
public:
    mutable size_t captures = 0;
    LatentCorrelationObservation captureLatentCorrelationObservation(size_t step, uint64_t serial, bool training) const override {
        ++captures;
        return Model::captureLatentCorrelationObservation(step, serial, training);
    }
};
void controlGateCase() {
    CountingModel model; configureSequence(model);
    auto control = std::make_shared<LatentSpaceControl>();
    control->resetSession("Actual contextual activations", "correlation_sequence_fixture");
    model.bindLatentSpaceControl(control);
    sequenceForward(model, {1, 2}, true);
    check(model.captures == 0, "disabled mode scanned model activations");
    control->setCorrelationsEnabled(true);
    sequenceForward(model, {1, 2}, true);
    check(model.captures == 0, "closed view scanned activations");
    control->setViewActive(true);
    control->setRealtimeEnabled(false);
    sequenceForward(model, {1, 2}, true);
    check(model.captures == 1, "manual-render mode did not collect corpus observations");
    model.captureRequestedLatentSpace(1, true);
    check(!control->takeRawSnapshot(), "correlation mode also captured structural weight vectors");
    sequenceForward(model, {2, 3}, false);
    check(model.captures == 2, "inference boundary did not collect activation observations");
    control->setViewActive(false);
    sequenceForward(model, {1, 3}, true);
    check(model.captures == 2, "hidden view kept scanning the model");
    control->setViewActive(true);
    control->resetSession("Replacement", "new_model");
    sequenceForward(model, {1, 3}, true);
    check(model.captures == 2, "old model captured into replacement session");
    model.bindLatentSpaceControl(control);
    sequenceForward(model, {1, 3}, true);
    check(model.captures == 3, "rebound session did not collect");
    control->setCorrelationsEnabled(false);
    sequenceForward(model, {1, 3}, true);
    check(model.captures == 3, "disabled mode kept scanning");
}
void boundedSampleCase() {
    using namespace mimir::viz::correlation_capture;
    std::vector<float> values(1000 * 257);
    for (size_t position = 0; position < 1000; ++position)
        for (size_t feature = 0; feature < 257; ++feature) values[position * 257 + feature] = float(feature);
    Source source; source.values = &values; source.key = "wide"; source.label = "wide activation";
    source.layout = Layout::FeatureLast; source.width = 257;
    LatentCorrelationObservation observation;
    check(appendFeatures(source, 128, nullptr, -1, nullptr, observation), "bounded sampling failed");
    check(observation.values.size() == 128 && observation.features.front().feature_index == 0 &&
          observation.features.back().feature_index == 256, "bounded sample omitted feature endpoints");
    for (size_t index = 0; index < observation.values.size(); ++index)
        check(observation.values[index] == float(observation.features[index].feature_index), "pooling altered a constant feature");
    std::vector<float> sequence_values{100, 100, 1, 3, 200, 200, 3, 7};
    std::vector<float> float_ids{.49f, 1.49f, 0.f, 2.49f};
    source.values = &sequence_values; source.width = 2; source.sequence = true;
    LatentCorrelationObservation rounded;
    check(appendFeatures(source, 2, nullptr, 0, nullptr, rounded, &float_ids) &&
          rounded.values == std::vector<float>({2.f, 5.f}), "float-ID mask differs from CPU Embedding rounding");
    for (float invalid : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::max()}) {
        float_ids[0] = invalid;
        LatentCorrelationObservation rejected;
        check(!appendFeatures(source, 2, nullptr, 0, nullptr, rejected, &float_ids) && rejected.values.empty(),
              "invalid float ID was silently reinterpreted as padding");
    }
}
} // namespace

int main() {
    try {
        sequenceCase(); overwrittenNamesCase(); floatIdInferenceCase(); spatialCase(3); spatialCase(4); legacySpatialOutputCase(); controlGateCase(); boundedSampleCase();
        std::cout << "MODEL_LATENT_CORRELATIONS_OK\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
