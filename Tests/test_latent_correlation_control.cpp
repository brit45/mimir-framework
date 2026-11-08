#include "AsyncMonitor.hpp"
#include "viz/LatentSpaceProjection.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
using namespace mimir::viz;
void check(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }
LatentCorrelationObservation fixture(uint64_t generation, uint64_t epoch, uint64_t serial) {
    LatentCorrelationObservation row;
    row.generation = generation; row.epoch = epoch; row.forward_serial = serial;
    row.optimizer_step = serial; row.model_name = "corpus fixture"; row.architecture = "native_test";
    row.training = serial % 2;
    const float x = static_cast<float>(serial % 17) - 8.f;
    row.values = {x, 2*x+17.f, -3*x+4.f, 1.f};
    for (size_t i = 0; i < row.values.size(); ++i)
        row.features.push_back({"hidden:"+std::to_string(i), "Hidden component "+std::to_string(i),
                                "moyenne par canal", 1, i, 4});
    return row;
}
void controls() {
    LatentSpaceControl control;
    const auto generation = control.resetSession("Fixture", "native_test");
    control.setViewActive(true);
    uint64_t epoch = 0, revision = 0;
    check(!control.correlations_enabled.load(), "correlations must default off");
    check(!control.beginCorrelationObservation(generation, 1, epoch), "disabled capture ran");
    control.setCorrelationsEnabled(true);
    check(control.beginCorrelationObservation(generation, 1, epoch), "enabled capture declined");
    check(!control.beginCorrelationObservation(generation, 1, epoch), "same forward captured twice");
    auto old = fixture(generation, epoch, 1);
    check(control.submitCorrelationObservation(old), "observation submission failed");
    check(control.takeCorrelationObservations().size() == 1, "observation mailbox lost data");
    check(control.beginCorrelationProjection(generation, epoch, 1, false, revision), "initial accumulating graph declined");
    LatentCorrelationWindow window; window.append(old);
    auto projection = projectLatentCorrelations3D(window.analyze(), old, revision);
    check(projection.correlation_mode && !projection.correlation_ready && projection.points.empty(),
          "accumulating state claims correlations");
    control.clearCorrelations();
    check(!control.publishProjection(projection), "clear accepted old projection");
    check(!control.submitCorrelationObservation(old), "clear accepted old data");
    check(control.correlation_observations.load() == 0, "clear kept count");
    uint64_t fresh_epoch = 0;
    check(control.beginCorrelationObservation(generation, 2, fresh_epoch) && fresh_epoch != epoch,
          "clear did not renew observation epoch");
    for (uint64_t i = 2; i < 80; ++i)
        check(control.submitCorrelationObservation(fixture(generation, fresh_epoch, i)), "bounded mailbox rejected input");
    const auto pending = control.takeCorrelationObservations();
    check(pending.size() == 32 && pending.front().forward_serial == 48, "mailbox is not latest bounded window");
    control.setCorrelationsEnabled(false);
    check(!control.beginCorrelationObservation(generation, 80, epoch), "capture continued after disabling");
    check(!control.submitCorrelationObservation(pending.back()), "disabled mailbox accepted old observation");
    control.fail("stale correlation failure", generation, 0, fresh_epoch);
    check(control.state.load() != LatentSpaceControl::State::Failed, "old worker changed new mode state");
    check(!control.projectionSnapshot(), "disabling left correlation graph visible");
    control.requestSnapshot(); uint64_t token = 0;
    check(control.beginCapture(token) && token == generation, "ordinary capture did not resume");
    control.resetSession("Next model", "next");
    control.setCorrelationsEnabled(true);
    check(!control.submitCorrelationObservation(old), "new model accepted old data");
}
template<class Predicate> void waitUntil(Predicate predicate, const char* reason) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
    while (!predicate() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    check(predicate(), reason);
}
void worker() {
    AsyncMonitor monitor;
    monitor.configureLatentSpace({{"type", "native_test"}, {"model_name", "Corpus fixture"}});
    const auto control = monitor.latentSpaceControl();
    const auto generation = control->generation.load();
    control->setViewActive(true);
    control->setCorrelationsEnabled(true);
    uint64_t epoch = 0;
    for (uint64_t serial = 1; serial <= 8; ++serial) {
        check(control->beginCorrelationObservation(generation, serial, epoch), "worker capture eligibility failed");
        check(control->submitCorrelationObservation(fixture(generation, epoch, serial)), "worker input refused");
    }
    waitUntil([&] { const auto p = control->projectionSnapshot(); return p && p->correlation_ready; },
              "async worker did not publish first ready graph with realtime off");
    auto projected = control->projectionSnapshot();
    check(projected->correlation_mode && projected->correlation_observations == 8, "wrong corpus count/mode");
    check(projected->points.size() == 3 && projected->correlation_constant_features == 1, "wrong variable features");
    check(projected->correlation_edges.size() == 3, "worker dropped Pearson edges");
    check(projected->correlation_training_observations == 4 && projected->correlation_inference_observations == 4,
          "worker lost train/inference provenance");
    check(projected->correlation_first_step == 1 && projected->correlation_last_step == 8, "worker step range wrong");
    bool positive = false, negative = false;
    for (const auto& edge : projected->correlation_edges) {
        positive = positive || edge.correlation > .999f;
        negative = negative || edge.correlation < -.999f;
    }
    check(positive && negative, "worker computed wrong signed Pearson relationships");
    const auto frozen_revision = projected->revision;
    for (uint64_t serial = 9; serial <= 12; ++serial) {
        check(control->beginCorrelationObservation(generation, serial, epoch), "collection stopped while display frozen");
        check(control->submitCorrelationObservation(fixture(generation, epoch, serial)), "frozen input refused");
    }
    waitUntil([&] { return control->correlation_observations.load() == 12; }, "collection did not continue while frozen");
    check(control->projectionSnapshot()->revision == frozen_revision, "realtime-off graph changed without request");
    control->requestSnapshot();
    waitUntil([&] { const auto p = control->projectionSnapshot(); return p && p->correlation_observations == 12; },
              "manual refresh did not use accumulated corpus window");
    control->setRealtimeEnabled(true);
    check(control->beginCorrelationObservation(generation, 13, epoch), "realtime observation refused");
    check(control->submitCorrelationObservation(fixture(generation, epoch, 13)), "realtime mailbox refused");
    waitUntil([&] { const auto p = control->projectionSnapshot(); return p && p->correlation_observations == 13; },
              "realtime refresh did not publish new observations");
    // A changed schema invalidates the displayed graph even with realtime off.
    control->setRealtimeEnabled(false);
    auto changed = fixture(generation, epoch, 14); changed.features[0].pooling = "nouvelle géométrie";
    check(control->submitCorrelationObservation(changed), "changed schema input refused");
    waitUntil([&] { const auto p = control->projectionSnapshot(); return p && !p->correlation_ready && p->correlation_observations == 1; },
              "schema change kept stale correlations visible");
    monitor.configureLatentSpace({{"type", "new_model"}});
    const auto new_projection = control->projectionSnapshot();
    check((!new_projection || (new_projection->architecture == "new_model" &&
                              new_projection->correlation_observations == 0 && new_projection->points.empty())) &&
          control->correlation_observations.load() == 0,
          "model switch retained previous corpus window");
    check(!control->submitCorrelationObservation(changed), "model switch accepted old observation");
    control->setCorrelationsEnabled(false);
    monitor.stop();
}
}
int main() {
    try { controls(); worker(); std::cout << "LATENT_CORRELATION_CONTROL_WORKER_OK\n"; return 0; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
