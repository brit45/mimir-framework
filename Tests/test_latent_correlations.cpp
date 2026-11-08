#include "viz/LatentSpaceProjection.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace mimir::viz;
void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
LatentCorrelationObservation observation(uint64_t serial, std::vector<float> values) {
    LatentCorrelationObservation result;
    result.generation = 1;
    result.epoch = 1;
    result.forward_serial = serial;
    result.optimizer_step = size_t(serial);
    result.values = std::move(values);
    for (size_t i = 0; i < result.values.size(); ++i) {
        LatentCorrelationFeature feature;
        feature.key = "layer0/feature" + std::to_string(i);
        feature.label = "Feature " + std::to_string(i);
        feature.layer = 0;
        feature.feature_index = i;
        feature.original_dimension = result.values.size();
        feature.pooling = "mean";
        result.features.push_back(std::move(feature));
    }
    return result;
}
double dot(const std::vector<float>& left, const std::vector<float>& right) {
    double result = 0.0;
    for (size_t i = 0; i < left.size(); ++i) result += double(left[i]) * right[i];
    return result;
}
float edge(const LatentCorrelationAnalysis& result, size_t left, size_t right) {
    for (const auto& candidate : result.edges)
        if (candidate.source == left && candidate.target == right) return candidate.correlation;
    throw std::runtime_error("expected Pearson edge absent");
}
void oracleCase() {
    LatentCorrelationWindow window;
    const std::vector<float> x{-7, -5, -3, -1, 1, 3, 5, 7};
    const std::vector<float> orthogonal{1, -1, -1, 1, 1, -1, -1, 1};
    for (size_t i = 0; i < x.size(); ++i) {
        auto row = observation(i + 1, {x[i], 2 * x[i] + 17, -3 * x[i] + 4, orthogonal[i], 11});
        row.training = i % 2 == 0;
        // Context identity is provenance, not a claim of independent samples.
        row.context_fingerprint = 77;
        row.optimizer_step = 100 - i;
        check(window.append(row), "oracle observation rejected");
        if (i < 7) check(!window.analyze().ready, "correlations published before minimum observations");
    }
    const auto result = window.analyze();
    check(result.ready && result.observations == 8, "oracle analysis not ready");
    check(result.features.size() == 4 && result.constant_features == 1, "constant feature was not excluded");
    check(result.profiles.size() == result.features.size() && result.means.size() == result.features.size() &&
          result.stddevs.size() == result.features.size(), "feature statistics are misaligned");
    check(std::abs(edge(result, 0, 1) - 1.0f) < 1e-6, "positive affine Pearson oracle failed");
    check(std::abs(edge(result, 0, 2) + 1.0f) < 1e-6, "negative affine Pearson oracle failed");
    check(result.edge_candidates == 3 && result.edges.size() == 3, "uncorrelated feature produced an edge");
    check(result.candidate_edges.size() == result.edge_candidates, "valid candidates lost in oracle case");
    check(std::abs(dot(result.profiles[0], result.profiles[3])) < 1e-7, "orthogonal Pearson oracle failed");
    check(result.means[0] == 0 && result.means[1] == 17 && result.means[2] == 4,
          "means differ from independent oracle");
    check(std::abs(result.stddevs[0] - std::sqrt(21.0)) < 1e-12, "population standard deviation incorrect");
    check(result.first_optimizer_step == 93 && result.last_optimizer_step == 100, "step range is not min/max");
    check(result.training_observations == 4 && result.inference_observations == 4, "observation provenance lost");
    for (const auto& profile : result.profiles) {
        double sum = 0;
        for (float value : profile) sum += value;
        check(std::abs(sum) < 1e-6 && std::abs(dot(profile, profile) - 1) < 1e-6,
              "profile is not centered and L2 normalized");
    }
}
void smallVarianceCase() {
    LatentCorrelationWindow window;
    for (size_t i = 0; i < 8; ++i) {
        const float tiny = (float(i) - 3.5f) * 1e-30f;
        check(window.append(observation(i + 1, {tiny, -tiny * 2})), "small observation rejected");
    }
    const auto result = window.analyze();
    check(result.ready && result.constant_features == 0, "small finite variance was arbitrarily discarded");
    check(result.stddevs[0] > 0 && result.stddevs[0] < 1e-28, "small standard deviation lost");
    check(std::abs(edge(result, 0, 1) + 1) < 1e-6, "small-variance Pearson oracle failed");
    LatentCorrelationWindow constant;
    for (size_t i = 0; i < 8; ++i) check(constant.append(observation(i + 1, {4, 4})), "constant row rejected");
    const auto empty = constant.analyze();
    check(!empty.ready && empty.features.empty() && empty.constant_features == 2 && empty.edges.empty(),
          "constant-only window published correlations");
    check(empty.candidate_edges.empty(), "constant-only window retained candidates");
}
void rejectionAndSessionCases() {
    LatentCorrelationWindow window;
    auto row = observation(10, {1, 2});
    check(window.append(row), "initial observation rejected");
    const auto revision = window.revision();
    check(!window.append(row) && window.revision() == revision && window.size() == 1,
          "duplicate serial changed window");
    row.forward_serial = 9;
    check(!window.append(row), "out-of-order serial accepted");
    row.forward_serial = 11;
    row.values[0] = std::numeric_limits<float>::quiet_NaN();
    check(!window.append(row) && window.size() == 1, "NaN observation was not rejected atomically");
    row.values[0] = std::numeric_limits<float>::infinity();
    check(!window.append(row), "infinite observation accepted");
    row = observation(11, {1, 2});
    row.values.pop_back();
    check(!window.append(row), "mismatched values/features accepted");
    check(!window.append(observation(11, {})), "empty observation accepted");
    row = observation(11, {1, 2});
    row.features[1].key = row.features[0].key;
    check(!window.append(row), "duplicate feature identity accepted");
    row.features[1].key.clear();
    check(!window.append(row), "empty feature identity accepted");
    check(!window.append(observation(11, std::vector<float>(LatentCorrelationWindow::max_features + 1, 1))),
          "oversized feature observation accepted");
    check(window.revision() == revision, "rejected observation changed revision");

    auto changed = observation(11, {3, 4});
    changed.features[0].label = "Renamed label";
    check(window.append(changed) && window.size() == 2, "display label reset feature schema");
    changed.forward_serial = 12;
    changed.features[0].pooling = "channel_mean";
    check(window.append(changed) && window.size() == 1, "pooling schema change did not reset window");
    changed.forward_serial = 13;
    changed.features[0].original_dimension = 42;
    check(window.append(changed) && window.size() == 1, "dimension schema change did not reset window");
    changed.forward_serial = 14;
    changed.features[0].feature_index = 1;
    check(window.append(changed) && window.size() == 1, "component layout change did not reset window");
    changed.forward_serial = 15;
    changed.features[0].key = "replacement/feature0";
    check(window.append(changed) && window.size() == 1, "feature identity change did not reset window");
    changed.forward_serial = 16;
    changed.features[0].layer = 2;
    check(window.append(changed) && window.size() == 1, "feature layer change did not reset window");
    changed.forward_serial = 1;
    changed.generation = 2;
    check(window.append(changed) && window.size() == 1, "generation reset did not admit a fresh serial");
    changed.epoch = 2;
    check(window.append(changed) && window.size() == 1, "epoch reset did not admit a fresh serial");
    const auto before_reset = window.revision();
    window.reset(3, 4);
    check(window.size() == 0 && window.revision() > before_reset && !window.analyze().ready,
          "explicit reset kept observations or revision");
}
void boundedWindowCase() {
    LatentCorrelationWindow window;
    for (size_t i = 0; i < LatentCorrelationWindow::max_observations + 40; ++i)
        check(window.append(observation(i + 1, {float(i), -float(i)})), "bounded window append rejected");
    const auto result = window.analyze();
    check(window.size() == LatentCorrelationWindow::max_observations &&
          result.observations == LatentCorrelationWindow::max_observations &&
          result.window_capacity == LatentCorrelationWindow::max_observations, "window exceeded capacity");
    check(result.first_optimizer_step == 41 && result.last_optimizer_step == 168,
          "window did not evict oldest observations");
    check(std::abs(result.means[0] - 103.5) < 1e-12, "window means contain evicted observations");
    check(std::abs(edge(result, 0, 1) + 1) < 1e-6, "rolling window lost correlation");
}
void deterministicEdgeCapCase() {
    LatentCorrelationWindow window;
    for (size_t i = 0; i < 8; ++i) {
        // Identical features give exact equal strengths and a known tie order.
        std::vector<float> values(LatentCorrelationWindow::max_features, float(i));
        check(window.append(observation(i + 1, std::move(values))), "maximum feature count rejected");
    }
    const auto first = window.analyze(), second = window.analyze();
    check(first.features.size() == LatentCorrelationWindow::max_features &&
          first.edge_candidates == LatentCorrelationWindow::max_features * (LatentCorrelationWindow::max_features - 1) / 2,
          "candidate edge count incorrect");
    check(first.edges.size() == LatentCorrelationWindow::max_edges && first.edges.size() == second.edges.size(),
          "edge count exceeded bound");
    check(first.candidate_edges.size() == first.edge_candidates &&
          first.candidate_edges.size() == LatentCorrelationWindow::max_edge_candidates &&
          first.candidate_edges.size() == second.candidate_edges.size(),
          "exhaustive candidates were truncated at the default display cap");
    size_t expected_source = 0, expected_target = 1;
    for (size_t i = 0; i < first.candidate_edges.size(); ++i) {
        const auto& a = first.candidate_edges[i];
        const auto& b = second.candidate_edges[i];
        check(a.source == expected_source && a.target == expected_target, "equal-strength edge order is not deterministic");
        check(a.source == b.source && a.target == b.target && a.correlation == b.correlation &&
              std::abs(a.correlation - 1) < 1e-6, "repeated analysis changed edges");
        if (i < first.edges.size())
            check(first.edges[i].source == a.source && first.edges[i].target == a.target &&
                  first.edges[i].correlation == a.correlation, "default subset is not the strongest candidate prefix");
        if (++expected_target == first.features.size()) {
            ++expected_source;
            expected_target = expected_source + 1;
        }
    }
    const auto latest = observation(8, std::vector<float>(LatentCorrelationWindow::max_features, 7.f));
    const auto projection = projectLatentCorrelations3D(first, latest, 1);
    check(projection.correlation_ready && projection.points.size() == first.features.size(),
          "candidate projection unavailable");
    check(projection.correlation_edges.size() == LatentCorrelationWindow::max_edges &&
          projection.correlation_candidate_edges.size() == LatentCorrelationWindow::max_edge_candidates,
          "projection dropped exhaustive candidates or changed default cap");
    check(&correlationEdgesForDisplay(projection, false) == &projection.correlation_edges &&
          &correlationEdgesForDisplay(projection, true) == &projection.correlation_candidate_edges,
          "display toggle did not select the corresponding immutable snapshot edges");
}
void thresholdAndStrengthCase() {
    const std::vector<float> x{-7, -5, -3, -1, 1, 3, 5, 7};
    const std::vector<float> orthogonal{1, -1, -1, 1, 1, -1, -1, 1};
    LatentCorrelationWindow threshold;
    LatentCorrelationWindow strongest;
    for (size_t i = 0; i < 8; ++i) {
        const double unit_x = double(x[i]) / std::sqrt(168.0);
        const double unit_orthogonal = double(orthogonal[i]) / std::sqrt(8.0);
        const auto mixed = [&](double correlation) {
            return float(correlation * unit_x + std::sqrt(1 - correlation * correlation) * unit_orthogonal);
        };
        check(threshold.append(observation(i + 1, {float(unit_x), mixed(.25), mixed(.5)})),
              "threshold fixture rejected");
        std::vector<float> values(LatentCorrelationWindow::max_features, mixed(.5));
        values[0] = float(unit_x);
        check(strongest.append(observation(i + 1, std::move(values))), "edge strength fixture rejected");
    }
    const auto result = threshold.analyze();
    for (const auto& candidate : result.edges)
        check(!(candidate.source == 0 && candidate.target == 1), "below-threshold correlation retained");
    check(std::abs(edge(result, 0, 2) - .5f) < 1e-6, "above-threshold correlation lost or incorrect");
    const auto limited = strongest.analyze();
    check(limited.edges.size() == LatentCorrelationWindow::max_edges, "strongest edge fixture did not reach cap");
    for (const auto& candidate : limited.edges)
        check(candidate.source != 0 && std::abs(candidate.correlation - 1) < 1e-6,
              "weaker first candidates displaced strongest edges at cap");
    bool weak_candidate_visible = false;
    for (const auto& candidate : limited.candidate_edges) {
        check(std::isfinite(candidate.correlation), "non-finite candidate retained");
        if (candidate.source == 0) {
            check(std::abs(candidate.correlation - .5f) < 1e-6 &&
                  !correlationEdgeVisible(candidate, false, .65f) &&
                  correlationEdgeVisible(candidate, true, .65f),
                  "all candidates did not bypass display threshold for a valid weaker link");
            weak_candidate_visible = true;
        }
    }
    check(weak_candidate_visible, "weaker valid candidates disappeared after strongest-edge selection");
}
void displaySelectionCase() {
    LatentSpaceProjection legacy;
    legacy.correlation_edges = {{0, 1, -.5f}};
    check(&correlationEdgesForDisplay(legacy, true) == &legacy.correlation_edges,
          "legacy snapshots without candidate list lost their edges");
    const auto visible = [](float correlation, bool all, float threshold = .65f) {
        return correlationEdgeVisible({0, 1, correlation}, all, threshold);
    };
    check(visible(.3f, true) && visible(-.3f, true) && !visible(.25f, true),
          "candidate validity floor was removed with exhaustive display");
    check(!visible(std::numeric_limits<float>::quiet_NaN(), true) &&
          !visible(std::numeric_limits<float>::infinity(), true) && !visible(1.5f, true),
          "invalid Pearson values accepted for display");
    check(visible(.9f, false) && visible(-.9f, false) && !visible(.5f, false) &&
          !visible(.9f, false, std::numeric_limits<float>::quiet_NaN()),
          "normal display threshold or finite validation changed");
}
} // namespace

int main() {
    try {
        oracleCase();
        smallVarianceCase();
        rejectionAndSessionCases();
        boundedWindowCase();
        deterministicEdgeCapCase();
        thresholdAndStrengthCase();
        displaySelectionCase();
        std::cout << "LATENT_CORRELATION_MATH_OK Pearson, finite window, schema, sessions, exhaustive candidates\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
