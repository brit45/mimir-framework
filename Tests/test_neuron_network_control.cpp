#include "AsyncMonitor.hpp"
#include "Models/Registry/ModelArchitectures.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
using namespace mimir::viz;
void check(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }
template<class Predicate> void waitUntil(Predicate predicate, const char* reason) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
    while (!predicate() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    check(predicate(), reason);
}
std::shared_ptr<Model> smallModel() {
    return ModelArchitectures::create("basic_mlp", json{{"input_dim",4},{"hidden_dim",8},
        {"output_dim",3},{"hidden_layers",1},{"dropout",0.0}});
}
LatentSpaceRawSnapshot rawNetwork(Model& model, uint64_t generation, uint64_t revision) {
    LatentSpaceRawSnapshot raw;
    raw.generation = generation; raw.revision = revision;
    raw.model_name = "Native MLP"; raw.architecture = "basic_mlp";
    raw.neuron_mode = true;
    raw.neuron_network = model.captureNeuronNetworkSnapshot();
    return raw;
}
void modesAndStaleResults() {
    auto model = smallModel(); // Metadata capture also works before allocation.
    LatentSpaceControl control;
    const auto generation = control.resetSession("Native MLP", "basic_mlp");
    check(!control.neurons_enabled.load(), "neuron mode defaults on");
    control.setViewActive(true);
    control.setNeuronsEnabled(true);
    check(!control.correlations_enabled.load(), "modes are not exclusive");
    uint64_t token = 0, revision = 0;
    bool neurons = false;
    check(control.beginCapture(token, false, generation, LatentSpaceControl::Clock::now(), &revision, &neurons)
          && neurons, "capture did not reserve the neuron mode atomically");
    auto raw = rawNetwork(*model, token, revision);
    auto wrong = raw; wrong.neuron_mode = false;
    check(!control.submitRawSnapshot(wrong), "neuron mode accepted a weight capture");
    check(control.submitRawSnapshot(raw), "neuron capture refused");
    const auto owned = control.takeRawSnapshot();
    check(bool(owned), "neuron snapshot lost");
    check(!control.beginProjection(token, revision), "weight worker entered neuron mode");
    check(control.beginProjection(token, revision, true), "neuron layout declined");
    const auto projection = projectNeuronNetwork3D(*owned);
    check(projection.neuron_mode && !projection.correlation_mode && !projection.points.empty(),
          "adapter returned an incorrect projection mode");
    check(control.publishProjection(projection), "neuron graph publication refused");
    check(control.projectionSnapshot()->neuron_links.size() > 0, "real graph routing disappeared");
    control.setCorrelationsEnabled(true);
    check(!control.neurons_enabled.load() && !control.projectionSnapshot(), "correlation switch kept old neurons");
    check(!control.publishProjection(projection) && !control.submitRawSnapshot(raw), "stale neuron result survived mode switch");
    uint64_t epoch = 0;
    check(control.beginCorrelationObservation(generation, 1, epoch), "correlation collection did not resume");
    control.setNeuronsEnabled(true);
    check(!control.correlations_enabled.load() && control.correlation_observations.load() == 0,
          "neuron switch retained the correlation window");
    check(!control.beginCorrelationObservation(generation, 2, epoch), "correlation capture ran in neuron mode");
    check(control.beginCapture(token, false, generation, LatentSpaceControl::Clock::now(), &revision, &neurons),
          "new neuron capture refused");
    raw = rawNetwork(*model, token, revision);
    control.setNeuronsEnabled(false);
    check(!control.submitRawSnapshot(raw), "toggle accepted stale pending neuron data");
    control.fail("stale worker", token, revision);
    check(control.state.load() != LatentSpaceControl::State::Failed, "stale failure replaced new mode state");
    control.setNeuronsEnabled(true);
    check(control.beginCapture(token, false, generation, LatentSpaceControl::Clock::now(), &revision, &neurons),
          "reset fixture capture refused");
    raw = rawNetwork(*model, token, revision);
    const auto next_generation = control.resetSession("Another model", "custom");
    check(next_generation != generation && control.neurons_enabled.load(), "model reset lost mode preference");
    check(!control.submitRawSnapshot(raw) && !control.publishProjection(projection), "new model accepted an old graph");
}
void nativeWorker() {
    auto model = smallModel();
    model->allocateParams(); model->initializeWeights("xavier", 381);
    AsyncMonitor monitor;
    monitor.configureLatentSpace(model->modelConfig);
    const auto control = monitor.latentSpaceControl();
    model->bindLatentSpaceControl(control);
    control->setViewActive(true); control->setNeuronsEnabled(true);
    const std::vector<float> input{-.4f, .2f, .7f, .1f};
    const auto output = model->forwardPass(input, false);
    waitUntil([&] { const auto p = control->projectionSnapshot(); return p && p->neuron_mode; },
              "native inference did not publish its neuron layout");
    auto graph = control->projectionSnapshot();
    check(graph->architecture == "basic_mlp" &&
          graph->neuron_total_layers == model->captureNeuronNetworkSnapshot().layers.size(),
          "worker used another model's topology");
    check(!graph->neuron_sampling && graph->points.size() == graph->neuron_total_units,
          "small known MLP was unnecessarily sampled or counted as unknown");
    for (const auto& point : graph->points) {
        check(!point.neuron_kind.empty() && !point.neuron_is_operation, "known MLP unit is undocumented");
        for (float value : point.position) check(std::isfinite(value), "nonfinite neuron layout");
    }
    for (const auto& link : graph->neuron_links)
        check(link.source < graph->points.size() && link.target < graph->points.size() &&
              !link.weighted && link.aggregated, "structural link was relabelled as an exact synapse");
    check(model->forwardPass(input, false) == output, "neuron capture changed model output");
    const auto first_revision = graph->revision;
    model->forwardPass(input, true);
    model->backwardPass(std::vector<float>(3, .1f));
    Optimizer optimizer; optimizer.type = OptimizerType::SGD;
    control->requestSnapshot();
    model->optimizerStep(optimizer, .001f);
    waitUntil([&] { const auto p = control->projectionSnapshot();
        return p && p->neuron_mode && p->revision > first_revision && p->optimizer_step == size_t(optimizer.step); },
        "optimizer safe point did not update neuron graph step");
    control->setNeuronsEnabled(false);
    model->forwardPass(input, false);
    waitUntil([&] { const auto p = control->projectionSnapshot(); return p && !p->neuron_mode && !p->correlation_mode; },
              "return to weight PCA did not publish ordinary vectors");
    control->setNeuronsEnabled(true); control->setRealtimeEnabled(true);
    model->forwardPass(input, false);
    waitUntil([&] { const auto p = control->projectionSnapshot(); return p && p->neuron_mode; },
              "realtime mode did not restore neuron layout");
    monitor.configureLatentSpace(json{{"type", "new_model"}});
    check(!control->projectionSnapshot(), "model switch retained old neuron positions");
    monitor.stop();
}
}
int main() {
    try {
        modesAndStaleResults(); nativeWorker();
        std::cout << "NEURON_NETWORK_CONTROL_WORKER_OK\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
