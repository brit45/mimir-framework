#include "AsyncMonitor.hpp"
#include "Models/Registry/ModelArchitectures.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

namespace {
using namespace mimir::viz;
void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template<class Predicate> void waitFor(Predicate predicate, const char* why) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
    while (!predicate() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    check(predicate(), why);
}
std::shared_ptr<Model> model() {
    auto value = ModelArchitectures::create("basic_mlp", json{{"input_dim",4},{"hidden_dim",8},
        {"output_dim",3},{"hidden_layers",1},{"dropout",0.0}});
    value->allocateParams(); value->initializeWeights("xavier", 419);
    return value;
}
LatentSpaceRawSnapshot snapshot(Model& value, uint64_t generation = 1, uint64_t epoch = 1) {
    LatentSpaceRawSnapshot raw;
    raw.generation = generation; raw.live_weight_epoch = epoch;
    raw.live_weight_mode = true; raw.model_name = "Native MLP"; raw.architecture = "basic_mlp";
    raw.live_weights = value.captureLiveWeightNetworkSnapshot();
    return raw;
}
void projectionAndEvolution() {
    auto native = model();
    auto first = snapshot(*native);
    const auto initial = projectLiveWeightNetwork3D(first);
    check(initial.live_weight_mode && !initial.neuron_mode && !initial.correlation_mode,
          "live projection has conflicting modes");
    check(!initial.live_weight_baseline_available && !initial.live_weight_links.empty(),
          "initial frame invented a delta or lost actual dense edges");
    check(initial.live_weight_coefficients.size() == first.live_weights.weights.size(), "coefficients lost");
    auto next = first; next.optimizer_step = 3;
    check(!next.live_weights.weights.empty(), "native fixture has no coefficients");
    next.live_weights.weights[0].value = -.125f;
    const auto changed = projectLiveWeightNetwork3D(next, &first);
    check(changed.live_weight_baseline_available && changed.live_weight_previous_step == 0,
          "valid baseline lost its actual optimizer step");
    check(changed.points.size() == initial.points.size(), "values changed topology size");
    for (size_t i = 0; i < changed.points.size(); ++i)
        check(changed.points[i].position == initial.points[i].position, "weight values moved the topology");
    for (size_t i = 0; i < changed.live_weight_coefficients.size(); ++i) {
        const auto& coefficient = changed.live_weight_coefficients[i];
        check(coefficient.has_delta && coefficient.delta ==
            double(next.live_weights.weights[i].value) - first.live_weights.weights[i].value,
            "coefficient delta does not match actual scalar difference");
        check(!coefficient.is_bias || (!coefficient.mapped && coefficient.point_index < changed.points.size()),
              "bias was drawn as a directed synapse");
    }
    for (const auto& link : changed.live_weight_links)
        check(link.source < changed.points.size() && link.target < changed.points.size() &&
              link.coefficient_index < changed.live_weight_coefficients.size(), "live edge out of bounds");
    const auto still = projectLiveWeightNetwork3D(next, &next);
    for (const auto& coefficient : still.live_weight_coefficients)
        check(coefficient.has_delta && coefficient.delta == 0., "unchanged weights invented motion");
    auto reset = next; ++reset.live_weight_epoch;
    check(!projectLiveWeightNetwork3D(reset, &next).live_weight_baseline_available, "mode reset retained baseline");
    reset = next; ++reset.generation;
    check(!projectLiveWeightNetwork3D(reset, &next).live_weight_baseline_available, "model reset retained baseline");
    reset = next; ++reset.live_weights.network.layers.front().units;
    check(!projectLiveWeightNetwork3D(reset, &next).live_weight_baseline_available, "schema change retained baseline");
    reset = next; reset.optimizer_step = 2;
    check(!projectLiveWeightNetwork3D(reset, &next).live_weight_baseline_available, "step rewind retained baseline");
    auto extreme = first;
    first.live_weights.weights[0].value = -std::numeric_limits<float>::max();
    extreme.live_weights.weights[0].value = std::numeric_limits<float>::max();
    const auto difference = projectLiveWeightNetwork3D(extreme, &first);
    check(std::isfinite(difference.live_weight_coefficients[0].delta) &&
          difference.live_weight_coefficients[0].delta > std::numeric_limits<float>::max(),
          "FP32 coefficient difference overflowed");
}
void modes() {
    auto native = model();
    LatentSpaceControl control;
    const auto generation = control.resetSession("Native MLP", "basic_mlp");
    check(!control.live_weights_enabled.load() && !control.realtime_enabled.load(), "live mode defaults on");
    control.setViewActive(true); control.setLiveWeightsEnabled(true);
    check(control.realtime_enabled.load() && !control.neurons_enabled.load() && !control.correlations_enabled.load(),
          "enabling live mode did not select realtime/exclusive mode");
    uint64_t token = 0, revision = 0; bool neurons = true, live = false;
    check(control.beginCapture(token, false, generation, LatentSpaceControl::Clock::now(),
          &revision, &neurons, &live) && live && !neurons, "capture reserved the wrong mode");
    auto raw = snapshot(*native, token, 0); raw.revision = revision;
    auto wrong = raw; wrong.live_weight_mode = false;
    check(!control.submitRawSnapshot(wrong), "live mode accepted ordinary vectors");
    check(control.submitRawSnapshot(raw), "live raw rejected");
    const auto owned = control.takeRawSnapshot();
    check(owned && owned->live_weight_epoch == control.correlation_epoch.load(), "capture epoch not stamped atomically");
    check(!control.beginProjection(token, revision) && control.beginProjection(token, revision, false, true),
          "projection admitted a mismatched mode");
    const auto projection = projectLiveWeightNetwork3D(*owned);
    check(control.publishProjection(projection), "live publication rejected");
    control.setCorrelationsEnabled(true);
    check(!control.live_weights_enabled.load() && !control.projectionSnapshot(), "old live graph survived correlation switch");
    check(!control.publishProjection(projection) && !control.submitRawSnapshot(*owned), "stale live worker accepted");
    control.setLiveWeightsEnabled(true);
    check(!control.correlations_enabled.load(), "live switch did not stop corpus collection");
    control.setNeuronsEnabled(true);
    check(!control.live_weights_enabled.load() && !control.projectionSnapshot(), "old live graph survived neuron switch");
    control.setLiveWeightsEnabled(true);
    control.resetSession("Another model", "other");
    check(control.live_weights_enabled.load() && !control.publishProjection(projection), "model reset retained old weights");
}
void nativeTrainingWorker() {
    auto observed = model(), reference = model();
    AsyncMonitor monitor;
    monitor.configureLatentSpace(observed->modelConfig);
    const auto control = monitor.latentSpaceControl();
    observed->bindLatentSpaceControl(control);
    control->setViewActive(true); control->setLiveWeightsEnabled(true);
    control->setRealtimeEnabled(false); // Manual requests make the oracle independent of wall time.
    const std::vector<float> input{-.4f,.2f,.7f,.1f};
    observed->forwardPass(input, false);
    waitFor([&] { const auto p = control->projectionSnapshot(); return p && p->live_weight_mode; },
            "native safe point did not publish the initial live frame");
    check(!control->projectionSnapshot()->live_weight_baseline_available, "initial worker frame invented history");
    Optimizer optimizer, oracle_optimizer; optimizer.type = oracle_optimizer.type = OptimizerType::SGD;
    for (size_t step = 1; step <= 2; ++step) {
        observed->zeroGradients(); reference->zeroGradients();
        const auto actual = observed->forwardPass(input, true), expected = reference->forwardPass(input, true);
        check(actual == expected, "live monitoring changed the forward output");
        observed->backwardPass(std::vector<float>(3,.1f));
        reference->backwardPass(std::vector<float>(3,.1f));
        control->requestSnapshot();
        observed->optimizerStep(optimizer,.001f); reference->optimizerStep(oracle_optimizer,.001f);
        waitFor([&] { const auto p = control->projectionSnapshot(); return p && p->live_weight_mode && p->optimizer_step == step; },
                "optimizer did not publish post-update live weights");
        const auto p = control->projectionSnapshot();
        check(p->live_weight_baseline_available && p->live_weight_previous_step == step - 1,
              "worker baseline does not match the preceding published capture");
        bool changed = false;
        for (const auto& coefficient : p->live_weight_coefficients)
            changed = changed || (coefficient.has_delta && coefficient.delta != 0.);
        check(changed, "actual training updates did not appear in the live frame");
        const auto actual_weights = observed->captureLiveWeightNetworkSnapshot();
        const auto expected_weights = reference->captureLiveWeightNetworkSnapshot();
        check(actual_weights.weights.size() == expected_weights.weights.size(), "observer changed sampled schema");
        for (size_t i = 0; i < actual_weights.weights.size(); ++i)
            check(actual_weights.weights[i].key == expected_weights.weights[i].key &&
                  actual_weights.weights[i].value == expected_weights.weights[i].value &&
                  actual_weights.weights[i].value == p->live_weight_coefficients[i].value,
                  "post-step visual values differ from the unobserved native model");
    }
    const auto old_revision = control->projectionSnapshot()->revision;
    control->requestSnapshot(); observed->forwardPass(input, false);
    waitFor([&] { const auto p = control->projectionSnapshot(); return p && p->revision > old_revision; },
            "manual inference refresh did not publish");
    const auto unchanged = control->projectionSnapshot();
    check(unchanged->optimizer_step == 2 && unchanged->live_weight_previous_step == 2, "inference invented an optimizer step");
    for (const auto& coefficient : unchanged->live_weight_coefficients)
        check(coefficient.has_delta && coefficient.delta == 0., "static inference weights appear active");
    control->setCorrelationsEnabled(true); control->setLiveWeightsEnabled(true);
    control->setRealtimeEnabled(false); observed->forwardPass(input,false);
    waitFor([&] { const auto p = control->projectionSnapshot(); return p && p->live_weight_mode; },
            "mode reset failed to publish a new live frame");
    check(!control->projectionSnapshot()->live_weight_baseline_available, "mode reset retained stale worker baseline");
    monitor.stop();
}
}
int main() {
    try { projectionAndEvolution(); modes(); nativeTrainingWorker();
        std::cout << "LIVE_WEIGHT_CONTROL_TRAINING_WORKER_OK\n"; return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
