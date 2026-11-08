#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace mimir::viz {

// Stable feature identities describe a component of a real layer activation.
// Pooling is part of the schema: differently pooled activations are not mixed.
struct LatentCorrelationFeature {
    std::string key;
    std::string label;
    std::string pooling;
    int layer = -1;
    size_t feature_index = 0;
    size_t original_dimension = 0;
};

struct LatentCorrelationObservation {
    uint64_t generation = 0;
    uint64_t epoch = 0;
    uint64_t forward_serial = 0;
    uint64_t context_fingerprint = 0;
    size_t optimizer_step = 0;
    bool training = false;
    std::string model_name;
    std::string architecture;
    std::string context_label;
    std::vector<LatentCorrelationFeature> features;
    std::vector<float> values;
};

struct LatentCorrelationEdge {
    size_t source = 0;
    size_t target = 0;
    float correlation = 0.0f;
};

struct LatentCorrelationAnalysis {
    std::vector<LatentCorrelationFeature> features;
    // One centered, L2-normalized response profile per retained feature.
    // These profiles are suitable for PCA; the edges retain Pearson r before
    // projection, so a rendered distance is never used as a correlation.
    std::vector<std::vector<float>> profiles;
    std::vector<double> means;
    std::vector<double> stddevs;
    // Full valid candidates remain available for the optional exhaustive view;
    // edges keeps the strongest default subset for existing callers.
    std::vector<LatentCorrelationEdge> candidate_edges;
    std::vector<LatentCorrelationEdge> edges;
    size_t observations = 0;
    size_t window_capacity = 128;
    size_t first_optimizer_step = 0;
    size_t last_optimizer_step = 0;
    size_t training_observations = 0;
    size_t inference_observations = 0;
    size_t constant_features = 0;
    size_t edge_candidates = 0;
    bool ready = false;
    std::string status;
};

// This bounded window belongs to the capture/control path, not to the model's
// numerical forward/backward state. All observations have the same feature
// schema and finite values, giving every feature pair the same sample count.
class LatentCorrelationWindow {
public:
    static constexpr size_t max_features = 128;
    static constexpr size_t max_observations = 128;
    static constexpr size_t min_observations = 8;
    static constexpr size_t max_edges = 512;
    static constexpr size_t max_edge_candidates = max_features * (max_features - 1) / 2;
    static constexpr double min_edge_correlation = 0.30;

    void reset(uint64_t generation, uint64_t epoch) {
        generation_ = generation;
        epoch_ = epoch;
        initialized_ = true;
        clearSchema();
        ++revision_;
    }

    bool append(const LatentCorrelationObservation& observation) {
        const size_t feature_count = observation.features.size();
        if (feature_count == 0 || feature_count > max_features ||
            observation.values.size() != feature_count) return false;
        for (size_t i = 0; i < feature_count; ++i) {
            if (!std::isfinite(observation.values[i]) || observation.features[i].key.empty()) return false;
            // Duplicate keys would make the identity of a component ambiguous.
            for (size_t j = 0; j < i; ++j)
                if (observation.features[i].key == observation.features[j].key) return false;
        }

        if (!initialized_ || observation.generation != generation_ || observation.epoch != epoch_)
            reset(observation.generation, observation.epoch);
        if (!schema_.empty() && !sameSchema(schema_, observation.features)) {
            clearSchema();
            ++revision_;
        }
        // Model forward serials are monotone within a capture session. Reject
        // repeated or out-of-order passes without an unbounded set of serials.
        if (have_serial_ && observation.forward_serial <= last_serial_) return false;
        if (schema_.empty()) schema_ = observation.features;
        Row row;
        row.values = observation.values;
        row.optimizer_step = observation.optimizer_step;
        row.training = observation.training;
        rows_.push_back(std::move(row));
        if (rows_.size() > max_observations) rows_.pop_front();
        last_serial_ = observation.forward_serial;
        have_serial_ = true;
        ++revision_;
        return true;
    }

    size_t size() const { return rows_.size(); }
    uint64_t revision() const { return revision_; }

    LatentCorrelationAnalysis analyze() const {
        LatentCorrelationAnalysis result;
        result.observations = rows_.size();
        result.window_capacity = max_observations;
        if (rows_.empty()) {
            result.status = "En attente d'activations du modèle";
            return result;
        }
        result.first_optimizer_step = std::numeric_limits<size_t>::max();
        for (const auto& row : rows_) {
            result.first_optimizer_step = std::min(result.first_optimizer_step, row.optimizer_step);
            result.last_optimizer_step = std::max(result.last_optimizer_step, row.optimizer_step);
            if (row.training) ++result.training_observations;
            else ++result.inference_observations;
        }

        const size_t count = rows_.size();
        std::vector<std::vector<double>> normalized;
        normalized.reserve(schema_.size());
        for (size_t feature = 0; feature < schema_.size(); ++feature) {
            double mean = 0.0;
            for (const auto& row : rows_) mean += static_cast<double>(row.values[feature]);
            mean /= static_cast<double>(count);
            // Double, two-pass centering avoids cancellation in E[x²]-E[x]².
            // There is no absolute variance threshold: tiny but variable
            // finite activations contain valid correlation information too.
            double sum_squares = 0.0;
            std::vector<double> profile;
            profile.reserve(count);
            for (const auto& row : rows_) {
                const double centered = static_cast<double>(row.values[feature]) - mean;
                profile.push_back(centered);
                sum_squares += centered * centered;
            }
            if (!(sum_squares > 0.0)) {
                ++result.constant_features;
                continue;
            }
            const double norm = std::sqrt(sum_squares);
            std::vector<float> display_profile;
            display_profile.reserve(count);
            for (double& value : profile) {
                value /= norm;
                display_profile.push_back(static_cast<float>(value));
            }
            result.features.push_back(schema_[feature]);
            result.profiles.push_back(std::move(display_profile));
            result.means.push_back(mean);
            // Population standard deviation describes the observed window.
            result.stddevs.push_back(std::sqrt(sum_squares / static_cast<double>(count)));
            normalized.push_back(std::move(profile));
        }

        if (count < min_observations) {
            result.status = "Accumulation des activations : " + std::to_string(count) + "/" +
                std::to_string(min_observations) + " observations";
            return result;
        }
        if (normalized.size() < 2) {
            result.status = "Au moins deux composantes variables sont nécessaires";
            return result;
        }
        result.ready = true;
        std::vector<LatentCorrelationEdge> candidates;
        for (size_t source = 0; source < normalized.size(); ++source) {
            for (size_t target = source + 1; target < normalized.size(); ++target) {
                double correlation = 0.0;
                for (size_t observation = 0; observation < count; ++observation)
                    correlation += normalized[source][observation] * normalized[target][observation];
                correlation = std::max(-1.0, std::min(1.0, correlation));
                if (std::abs(correlation) >= min_edge_correlation)
                    candidates.push_back({source, target, static_cast<float>(correlation)});
            }
        }
        result.edge_candidates = candidates.size();
        std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
            const float left_strength = std::abs(left.correlation), right_strength = std::abs(right.correlation);
            if (left_strength != right_strength) return left_strength > right_strength;
            if (left.source != right.source) return left.source < right.source;
            return left.target < right.target;
        });
        result.candidate_edges = candidates;
        if (candidates.size() > max_edges) candidates.resize(max_edges);
        result.edges = std::move(candidates);
        result.status = "Corrélations de Pearson sur " + std::to_string(count) + " observations";
        return result;
    }

private:
    struct Row {
        std::vector<float> values;
        size_t optimizer_step = 0;
        bool training = false;
    };
    static bool sameSchema(const std::vector<LatentCorrelationFeature>& left,
                           const std::vector<LatentCorrelationFeature>& right) {
        if (left.size() != right.size()) return false;
        for (size_t i = 0; i < left.size(); ++i)
            if (left[i].key != right[i].key || left[i].pooling != right[i].pooling ||
                left[i].layer != right[i].layer || left[i].feature_index != right[i].feature_index ||
                left[i].original_dimension != right[i].original_dimension) return false;
        return true;
    }
    void clearSchema() {
        schema_.clear();
        rows_.clear();
        have_serial_ = false;
        last_serial_ = 0;
    }
    uint64_t generation_ = 0, epoch_ = 0, revision_ = 0, last_serial_ = 0;
    bool initialized_ = false, have_serial_ = false;
    std::vector<LatentCorrelationFeature> schema_;
    std::deque<Row> rows_;
};

} // namespace mimir::viz
