#pragma once

#include "LatentCorrelation.hpp"
#include "NeuronNetwork.hpp"
#include "LiveWeightNetwork.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace mimir::viz {

enum class LatentVectorGroup : uint8_t {
    TokenEmbedding,
    AttentionWeight,
    FeedForwardWeight,
    LatentDownWeight,
    LatentUpWeight,
    NormWeight,
    SequenceActivation,
    ParameterWeight,
    ConvolutionWeight,
    RecurrentWeight,
    OutputWeight,
    LayerActivation,
    LatentActivation,
    Activation = LayerActivation
};

inline const char* latentVectorGroupName(LatentVectorGroup group) {
    switch (group) {
        case LatentVectorGroup::TokenEmbedding: return "Embeddings de tokens";
        case LatentVectorGroup::AttentionWeight: return "Poids d'attention";
        case LatentVectorGroup::FeedForwardWeight: return "Poids du MLP";
        case LatentVectorGroup::LatentDownWeight: return "Poids latent-down";
        case LatentVectorGroup::LatentUpWeight: return "Poids latent-up";
        case LatentVectorGroup::NormWeight: return "Poids de normalisation";
        case LatentVectorGroup::SequenceActivation: return "Activations par position";
        case LatentVectorGroup::ParameterWeight: return "Paramètres";
        case LatentVectorGroup::ConvolutionWeight: return "Poids de convolution";
        case LatentVectorGroup::RecurrentWeight: return "Poids récurrents";
        case LatentVectorGroup::OutputWeight: return "Poids de sortie";
        case LatentVectorGroup::LayerActivation: return "Activations des couches";
        case LatentVectorGroup::LatentActivation: return "Activations latentes";
    }
    return "Vecteurs";
}

struct LatentVectorSample {
    std::string label;
    LatentVectorGroup group = LatentVectorGroup::TokenEmbedding;
    std::vector<float> values;
    int layer = -1;
    int sequence_position = -1;
    float norm = 0.0f;
    size_t original_dimension = 0;
    size_t feature_offset = 0;
};

struct LatentSpaceRawSnapshot {
    uint64_t generation = 0;
    uint64_t revision = 0;
    std::string model_name;
    std::string architecture;
    size_t feature_dimension = 0;
    size_t optimizer_step = 0;
    int d_model = 0;
    int latent_dim = 0;
    int sequence_length = 0;
    std::string position_encoding;
    std::vector<LatentVectorSample> vectors;
    bool neuron_mode = false;
    NeuronNetworkSnapshot neuron_network;
    bool live_weight_mode = false;
    uint64_t live_weight_epoch = 0;
    LiveWeightNetworkSnapshot live_weights;
};

struct LatentSpacePoint {
    std::array<float, 3> position{{0.0f, 0.0f, 0.0f}};
    std::string label;
    LatentVectorGroup group = LatentVectorGroup::TokenEmbedding;
    int layer = -1;
    int sequence_position = -1;
    float vector_norm = 0.0f;
    size_t original_dimension = 0;
    size_t feature_offset = 0;
    size_t sampled_dimension = 0;
    float activation_mean = 0.0f;
    float activation_stddev = 0.0f;
    size_t feature_index = 0;
    std::string pooling;
    std::string neuron_kind;
    size_t neuron_unit_index = 0;
    size_t neuron_node_index = 0;
    size_t neuron_layer_units = 0;
    bool neuron_is_operation = false;
    size_t live_weight_coefficient = std::numeric_limits<size_t>::max();
};

struct LiveWeightCoefficient {
    std::string key, label;
    size_t parameter_index = 0;
    float value = 0.f;
    double delta = 0.;
    bool has_delta = false;
    size_t point_index = std::numeric_limits<size_t>::max();
    bool is_bias = false, mapped = false;
};
struct LiveWeightLink {
    size_t source = 0, target = 0, coefficient_index = 0;
};

struct LatentSpaceProjection {
    uint64_t generation = 0;
    uint64_t revision = 0;
    std::string model_name;
    std::string architecture;
    size_t feature_dimension = 0;
    size_t effective_rank = 0;
    size_t optimizer_step = 0;
    int d_model = 0;
    int latent_dim = 0;
    int sequence_length = 0;
    std::string position_encoding;
    std::array<float, 3> explained_variance_percent{{0.0f, 0.0f, 0.0f}};
    std::vector<LatentSpacePoint> points;
    bool correlation_mode = false;
    bool correlation_ready = false;
    uint64_t correlation_epoch = 0;
    size_t correlation_observations = 0;
    size_t correlation_window = 128;
    size_t correlation_training_observations = 0;
    size_t correlation_inference_observations = 0;
    size_t correlation_first_step = 0;
    size_t correlation_last_step = 0;
    size_t correlation_constant_features = 0;
    size_t correlation_edge_candidates = 0;
    std::string correlation_status;
    std::vector<LatentCorrelationEdge> correlation_edges;
    std::vector<LatentCorrelationEdge> correlation_candidate_edges;
    bool neuron_mode = false;
    size_t neuron_total_units = 0;
    size_t neuron_total_layers = 0;
    size_t neuron_unknown_layers = 0;
    bool neuron_sampling = false;
    std::vector<NeuronNetworkLink> neuron_links;
    bool live_weight_mode = false;
    uint64_t live_weight_epoch = 0;
    size_t live_weight_previous_step = 0;
    bool live_weight_baseline_available = false;
    size_t live_weight_total_parameters = 0;
    size_t live_weight_nonfinite = 0;
    size_t live_weight_skipped_layers = 0;
    size_t live_weight_shared_layers = 0;
    bool live_weight_sampled = false;
    std::vector<LiveWeightCoefficient> live_weight_coefficients;
    std::vector<LiveWeightLink> live_weight_links;
};

inline const std::vector<LatentCorrelationEdge>& correlationEdgesForDisplay(
    const LatentSpaceProjection& projection, bool show_all_candidates)
{
    // The fallback keeps manually built/older snapshots useful even when they
    // only carry the strongest subset.
    return show_all_candidates && !projection.correlation_candidate_edges.empty()
        ? projection.correlation_candidate_edges : projection.correlation_edges;
}

inline bool correlationEdgeVisible(const LatentCorrelationEdge& edge,
                                  bool show_all_candidates, float threshold)
{
    if (!std::isfinite(edge.correlation)) return false;
    constexpr float epsilon = 1e-6f;
    const float strength = std::abs(edge.correlation);
    if (strength > 1.f + epsilon ||
        strength + epsilon < static_cast<float>(LatentCorrelationWindow::min_edge_correlation)) return false;
    return show_all_candidates || (std::isfinite(threshold) && strength + epsilon >= threshold);
}

// Center and L2-normalize each sampled model vector, then project it onto the
// three leading principal components. PCA uses the sample Gram matrix, so
// temporary memory scales with the bounded sample count rather than width^2.
inline LatentSpaceProjection projectLatentSpace3D(const LatentSpaceRawSnapshot& raw) {
    LatentSpaceProjection result;
    result.generation = raw.generation;
    result.revision = raw.revision;
    result.model_name = raw.model_name;
    result.architecture = raw.architecture;
    result.optimizer_step = raw.optimizer_step;
    result.d_model = raw.d_model;
    result.latent_dim = raw.latent_dim;
    result.sequence_length = raw.sequence_length;
    result.position_encoding = raw.position_encoding;

    // Heterogeneous architectures expose real sampled vector widths. Complete
    // shorter vectors with zero solely for this structural display; these axes
    // and distances do not claim a shared semantic latent space across layers.
    size_t dimension = 0;
    for(const auto& sample:raw.vectors)
        if(!sample.values.empty() && sample.values.size()<=2048)
            dimension=std::max(dimension,sample.values.size());
    result.feature_dimension = dimension;
    if (dimension == 0 || raw.vectors.empty()) return result;

    std::vector<const LatentVectorSample*> valid;
    valid.reserve(std::min<size_t>(raw.vectors.size(), 512));
    for (const auto& sample : raw.vectors) {
        if (sample.values.empty() || sample.values.size() > dimension) continue;
        bool finite = true;
        for (float value : sample.values) {
            if (!std::isfinite(value)) { finite = false; break; }
        }
        if (finite) valid.push_back(&sample);
        if (valid.size() >= 512) break;
    }
    if (valid.empty()) return result;

    const size_t count = valid.size();
    std::vector<double> rows(count * dimension, 0.0);
    std::vector<double> mean(dimension, 0.0);
    std::vector<double> norms(count, 0.0);
    for (size_t row = 0; row < count; ++row) {
        const auto& values = valid[row]->values;
        double norm = 0.0;
        for (size_t column = 0; column < values.size(); ++column)
            norm += static_cast<double>(values[column]) * values[column];
        norms[row] = std::sqrt(norm);
        for (size_t column = 0; column < values.size(); ++column) {
            const double normalized = norms[row] > 1e-20 ? static_cast<double>(values[column]) / norms[row] : 0.0;
            rows[row * dimension + column] = normalized;
            mean[column] += normalized;
        }
    }
    for (double& value : mean) value /= static_cast<double>(count);

    std::vector<double> gram(count * count, 0.0);
    for (size_t row = 0; row < count; ++row) {
        double* vector = rows.data() + row * dimension;
        for (size_t column = 0; column < dimension; ++column)
            vector[column] -= mean[column];
    }
    double trace = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double* left = rows.data() + i * dimension;
        for (size_t j = i; j < count; ++j) {
            const double* right = rows.data() + j * dimension;
            double dot = 0.0;
            for (size_t k = 0; k < dimension; ++k) dot += left[k] * right[k];
            gram[i * count + j] = dot;
            gram[j * count + i] = dot;
        }
        trace += gram[i * count + i];
    }
    if (!std::isfinite(trace)) return result;
    if (!(trace > 1e-20)) {
        // A singleton, zero weights, or identical vectors have rank zero and
        // still describe a valid model snapshot. Show their points at origin.
        for(size_t row=0;row<count;++row){
            const auto& sample=*valid[row];
            LatentSpacePoint point;
            point.label=sample.label;point.group=sample.group;
            point.layer=sample.layer;point.sequence_position=sample.sequence_position;
            point.vector_norm=static_cast<float>(norms[row]);
            point.original_dimension=sample.original_dimension?sample.original_dimension:sample.values.size();
            point.feature_offset=sample.feature_offset;point.sampled_dimension=sample.values.size();
            result.points.push_back(std::move(point));
        }
        return result;
    }

    std::array<std::vector<double>, 3> axes;
    std::array<double, 3> eigenvalues{{0.0, 0.0, 0.0}};
    for(auto& axis:axes)axis.assign(count,0.0);
    const size_t components=std::min<size_t>(axes.size(),count-1);
    for (size_t component = 0; component < components; ++component) {
        auto& axis = axes[component];
        axis.resize(count);
        uint32_t seed = static_cast<uint32_t>(0x9e3779b9u + component * 0x85ebca6bu);
        for (double& value : axis) {
            seed = seed * 1664525u + 1013904223u;
            value = (static_cast<double>(seed) / 2147483648.0) - 1.0;
        }
        auto orthogonalize = [&]() {
            for (size_t previous = 0; previous < component; ++previous) {
                double dot = 0.0;
                for (size_t i = 0; i < count; ++i) dot += axis[i] * axes[previous][i];
                for (size_t i = 0; i < count; ++i) axis[i] -= dot * axes[previous][i];
            }
            double norm = 0.0;
            for (double value : axis) norm += value * value;
            norm = std::sqrt(norm);
            if (!(norm > 1e-20)) return false;
            for (double& value : axis) value /= norm;
            return true;
        };
        if (!orthogonalize()) {
            bool found = false;
            for (size_t candidate = 0; candidate < count && !found; ++candidate) {
                std::fill(axis.begin(), axis.end(), 0.0);
                axis[candidate] = 1.0;
                found = orthogonalize();
            }
        }
        for (int iteration = 0; iteration < 40; ++iteration) {
            std::vector<double> next(count, 0.0);
            for (size_t i = 0; i < count; ++i) {
                const double* row = gram.data() + i * count;
                double sum = 0.0;
                for (size_t j = 0; j < count; ++j) sum += row[j] * axis[j];
                next[i] = sum;
            }
            axis.swap(next);
            if (!orthogonalize()) break;
        }
        double eigenvalue = 0.0;
        for (size_t i = 0; i < count; ++i) {
            double projected = 0.0;
            for (size_t j = 0; j < count; ++j)
                projected += gram[i * count + j] * axis[j];
            eigenvalue += axis[i] * projected;
        }
        eigenvalues[component] = eigenvalue > trace*1e-12 ? eigenvalue : 0.0;
        if(eigenvalues[component]>0.0)++result.effective_rank;
    }

    for (size_t component = 0; component < axes.size(); ++component)
        result.explained_variance_percent[component] = static_cast<float>(
            100.0 * eigenvalues[component] / trace);

    std::vector<std::array<double, 3>> coordinates(count);
    for (size_t component = 0; component < axes.size(); ++component) {
        const double scale = std::sqrt(eigenvalues[component]);
        for (size_t row = 0; row < count; ++row) {
            coordinates[row][component] = scale * axes[component][row];
        }
    }
    // One scale preserves PCA geometry and relative variance across all axes.
    // Per-axis normalization would visually inflate low-variance components.
    double max_radius=1e-12;
    for(const auto& point:coordinates){
        double squared_radius=0.0;
        for(double value:point)squared_radius+=value*value;
        max_radius=std::max(max_radius,std::sqrt(squared_radius));
    }
    const double display_scale=2.5/max_radius;
    result.points.reserve(count);
    for (size_t row = 0; row < count; ++row) {
        const auto& sample = *valid[row];
        LatentSpacePoint point;
        for (size_t component = 0; component < 3; ++component)
            point.position[component] = static_cast<float>(
                display_scale * coordinates[row][component]);
        point.label = sample.label;
        point.group = sample.group;
        point.layer = sample.layer;
        point.sequence_position = sample.sequence_position;
        point.vector_norm = static_cast<float>(norms[row]);
        point.original_dimension = sample.original_dimension ? sample.original_dimension : sample.values.size();
        point.feature_offset = sample.feature_offset;
        point.sampled_dimension = sample.values.size();
        result.points.push_back(std::move(point));
    }
    return result;
}

// Features are points here: their centered, unit-length activation profiles
// across observations have dot product equal to Pearson r. Edges were computed
// in that original space, before the structural PCA changes display geometry.
inline LatentSpaceProjection projectLatentCorrelations3D(
    const LatentCorrelationAnalysis& analysis,
    const LatentCorrelationObservation& latest,
    uint64_t revision)
{
    LatentSpaceRawSnapshot raw;
    raw.generation = latest.generation;
    raw.revision = revision;
    raw.model_name = latest.model_name;
    raw.architecture = latest.architecture;
    raw.optimizer_step = latest.optimizer_step;
    if (analysis.ready) {
        raw.vectors.reserve(analysis.features.size());
        for (size_t index = 0; index < analysis.features.size(); ++index) {
            const auto& feature = analysis.features[index];
            LatentVectorSample sample;
            sample.label = feature.label;
            sample.group = LatentVectorGroup::LatentActivation;
            sample.layer = feature.layer;
            sample.original_dimension = analysis.observations;
            sample.values = analysis.profiles[index];
            sample.norm = 1.f;
            raw.vectors.push_back(std::move(sample));
        }
    }
    auto result = projectLatentSpace3D(raw);
    result.correlation_mode = true;
    result.correlation_ready = analysis.ready;
    result.correlation_epoch = latest.epoch;
    result.correlation_observations = analysis.observations;
    result.correlation_window = analysis.window_capacity;
    result.correlation_training_observations = analysis.training_observations;
    result.correlation_inference_observations = analysis.inference_observations;
    result.correlation_first_step = analysis.first_optimizer_step;
    result.correlation_last_step = analysis.last_optimizer_step;
    result.correlation_constant_features = analysis.constant_features;
    result.correlation_edge_candidates = analysis.edge_candidates;
    result.correlation_status = analysis.status;
    if (result.points.size() == analysis.features.size() && analysis.ready) {
        result.correlation_edges = analysis.edges;
        result.correlation_candidate_edges = analysis.candidate_edges;
        for (size_t index = 0; index < result.points.size(); ++index) {
            auto& point = result.points[index];
            point.activation_mean = static_cast<float>(analysis.means[index]);
            point.activation_stddev = static_cast<float>(analysis.stddevs[index]);
            point.feature_index = analysis.features[index].feature_index;
            point.pooling = analysis.features[index].pooling;
        }
    } else if (analysis.ready) {
        result.correlation_ready = false;
        result.correlation_status = "Profils d’activations incompatibles avec la projection";
    }
    return result;
}

// A topology layout has its own axes. It must never pass through the PCA used
// for weights or activation profiles, which would lose the graph's depth.
inline LatentSpaceProjection projectNeuronLayout3D(const LatentSpaceRawSnapshot& raw,
        const NeuronNetworkSnapshot& network, size_t max_points = neuronNetworkPointLimit,
        size_t max_links = neuronNetworkLinkLimit) {
    LatentSpaceProjection result;
    result.generation = raw.generation;
    result.revision = raw.revision;
    result.model_name = raw.model_name;
    result.architecture = raw.architecture;
    result.optimizer_step = raw.optimizer_step;
    result.neuron_mode = true;
    auto layout = layoutNeuronNetwork3D(network, max_points, max_links);
    result.neuron_total_units = layout.total_units;
    result.neuron_total_layers = layout.total_layers;
    result.neuron_unknown_layers = layout.unknown_layers;
    result.neuron_sampling = layout.sampled;
    result.neuron_links = std::move(layout.links);
    result.points.reserve(layout.points.size());
    for (auto& unit : layout.points) {
        LatentSpacePoint point;
        point.position = unit.position;
        point.label = std::move(unit.label);
        point.layer = unit.layer;
        point.group = LatentVectorGroup::LayerActivation;
        point.neuron_kind = std::move(unit.kind);
        point.neuron_unit_index = unit.unit_index;
        point.neuron_node_index = unit.node_index;
        point.neuron_layer_units = unit.layer_units;
        point.neuron_is_operation = unit.is_operation;
        result.points.push_back(std::move(point));
    }
    return result;
}

inline LatentSpaceProjection projectNeuronNetwork3D(const LatentSpaceRawSnapshot& raw) {
    return projectNeuronLayout3D(raw, raw.neuron_network);
}

inline bool liveWeightSchemaEqual(const LiveWeightNetworkSnapshot& a,
                                 const LiveWeightNetworkSnapshot& b) {
    if (a.total_parameters != b.total_parameters || a.weights.size() != b.weights.size() ||
        a.network.layers.size() != b.network.layers.size() ||
        a.network.connections.size() != b.network.connections.size()) return false;
    for (size_t i = 0; i < a.network.layers.size(); ++i) {
        const auto& x = a.network.layers[i]; const auto& y = b.network.layers[i];
        if (x.layer != y.layer || x.name != y.name || x.type != y.type || x.output != y.output ||
            x.kind != y.kind || x.inputs != y.inputs || x.units != y.units || x.depth != y.depth ||
            x.is_operation != y.is_operation || x.is_input != y.is_input) return false;
    }
    for (size_t i = 0; i < a.network.connections.size(); ++i) {
        const auto& x = a.network.connections[i]; const auto& y = b.network.connections[i];
        if (x.source_layer != y.source_layer || x.target_layer != y.target_layer || x.tensor != y.tensor) return false;
    }
    for (size_t i = 0; i < a.weights.size(); ++i) {
        const auto& x = a.weights[i]; const auto& y = b.weights[i];
        if (x.key != y.key || x.parameter_index != y.parameter_index || x.layer != y.layer ||
            x.source_layer != y.source_layer || x.target_layer != y.target_layer ||
            x.input_unit != y.input_unit || x.output_unit != y.output_unit ||
            x.is_bias != y.is_bias || x.mapped != y.mapped) return false;
    }
    return true;
}

// Keep the topology fixed and visualize scalar coefficients in their declared
// layer. Only proven mappings create synaptic edges; packed/unknown storage
// remains an exact labelled coefficient glyph. Values never move the layout.
inline LatentSpaceProjection projectLiveWeightNetwork3D(const LatentSpaceRawSnapshot& raw,
                                                       const LatentSpaceRawSnapshot* previous = nullptr) {
    const auto& captured = raw.live_weights;
    auto result = projectNeuronLayout3D(raw, captured.network,
        liveWeightNeuronPointLimit, liveWeightStructuralLinkLimit);
    result.neuron_mode = false;
    result.live_weight_mode = true;
    result.live_weight_epoch = raw.live_weight_epoch;
    result.live_weight_total_parameters = captured.total_parameters;
    result.live_weight_nonfinite = captured.nonfinite_parameters;
    result.live_weight_skipped_layers = captured.unavailable_layers;
    result.live_weight_shared_layers = captured.shared_layers;
    result.live_weight_sampled = captured.sampled || result.neuron_sampling;
    const bool baseline = previous && previous->live_weight_mode &&
        previous->generation == raw.generation && previous->live_weight_epoch == raw.live_weight_epoch &&
        previous->model_name == raw.model_name && previous->architecture == raw.architecture &&
        previous->optimizer_step <= raw.optimizer_step && liveWeightSchemaEqual(captured, previous->live_weights);
    result.live_weight_baseline_available = baseline;
    result.live_weight_previous_step = baseline ? previous->optimizer_step : raw.optimizer_step;
    std::map<std::pair<size_t, size_t>, size_t> units;
    std::map<size_t, std::array<double, 4>> centers;
    for (size_t index = 0; index < result.points.size(); ++index) {
        const auto& point = result.points[index];
        if (!point.neuron_is_operation)
            units.emplace(std::make_pair(point.neuron_node_index, point.neuron_unit_index), index);
        auto& center = centers[point.neuron_node_index];
        for (size_t axis = 0; axis < 3; ++axis) center[axis] += point.position[axis];
        center[3] += 1.;
    }
    const size_t count = std::min(captured.weights.size(), liveWeightCoefficientLimit);
    result.live_weight_coefficients.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        const auto& sample = captured.weights[index];
        if (!std::isfinite(sample.value)) continue;
        LiveWeightCoefficient coefficient;
        coefficient.key = sample.key; coefficient.label = sample.label;
        coefficient.parameter_index = sample.parameter_index; coefficient.value = sample.value;
        coefficient.is_bias = sample.is_bias;
        coefficient.has_delta = baseline && std::isfinite(previous->live_weights.weights[index].value);
        if (coefficient.has_delta)
            coefficient.delta = static_cast<double>(sample.value) - previous->live_weights.weights[index].value;
        const auto source = units.find({sample.source_layer, sample.input_unit});
        const auto target = units.find({sample.target_layer, sample.output_unit});
        coefficient.mapped = sample.mapped && !sample.is_bias && source != units.end() && target != units.end();
        const size_t coefficient_index = result.live_weight_coefficients.size();
        if (coefficient.mapped) {
            result.live_weight_links.push_back({source->second, target->second, coefficient_index});
        } else {
            const auto anchor = centers.find(sample.target_layer);
            if (anchor == centers.end() || anchor->second[3] <= 0.) {
                result.live_weight_sampled = true;
                continue; // Omitted layers must not acquire fictitious coordinates.
            }
            LatentSpacePoint glyph;
            glyph.label = sample.label;
            glyph.layer = sample.layer;
            glyph.neuron_kind = sample.is_bias ? "biais" : "coefficient";
            glyph.neuron_node_index = sample.target_layer;
            glyph.live_weight_coefficient = coefficient_index;
            for (size_t axis = 0; axis < 3; ++axis)
                glyph.position[axis] = static_cast<float>(anchor->second[axis] / anchor->second[3]);
            // Hash only the immutable scalar identity, never its current value.
            uint64_t hash = UINT64_C(14695981039346656037);
            for (unsigned char byte : sample.key) { hash ^= byte; hash *= UINT64_C(1099511628211); }
            for (size_t axis = 0; axis < 3; ++axis) {
                glyph.position[axis] += (static_cast<float>((hash >> (axis * 16)) & 1023) / 1023.f - .5f) * .32f;
                glyph.position[axis] = std::clamp(glyph.position[axis], -2.5f, 2.5f);
            }
            coefficient.point_index = result.points.size();
            result.points.push_back(std::move(glyph));
        }
        result.live_weight_coefficients.push_back(std::move(coefficient));
    }
    return result;
}

struct LatentSpaceControl {
    enum class State : int { Unavailable, Ready, Requested, Capturing, Projecting, Complete, Failed };
    using Clock = std::chrono::steady_clock;
    static constexpr std::chrono::milliseconds realtime_interval{250};

    std::atomic<bool> available{false};
    std::atomic<bool> training_active{false};
    std::atomic<bool> realtime_enabled{false};
    std::atomic<bool> correlations_enabled{false};
    std::atomic<bool> neurons_enabled{false};
    std::atomic<bool> live_weights_enabled{false};
    std::atomic<bool> correlation_ready{false};
    std::atomic<uint64_t> correlation_epoch{0};
    std::atomic<size_t> correlation_observations{0};
    std::atomic<bool> view_active{false};
    std::atomic<bool> requested{false};
    std::atomic<bool> raw_ready{false};
    std::atomic<State> state{State::Unavailable};
    std::atomic<uint64_t> generation{0};

    void setMessage(std::string value) {
        std::lock_guard<std::mutex> lock(message_mutex_);
        message_ = std::move(value);
    }
    std::string messageSnapshot() const {
        std::lock_guard<std::mutex> lock(message_mutex_);
        return message_;
    }
    void setRealtimeEnabled(bool enabled) {
        std::lock_guard<std::mutex> lock(session_mutex_);
        if(realtime_enabled.exchange(enabled,std::memory_order_acq_rel)!=enabled && enabled)
            have_capture_time_=false;
    }
    void setViewActive(bool active) {
        std::lock_guard<std::mutex> lock(session_mutex_);
        if(view_active.exchange(active,std::memory_order_acq_rel)!=active && active)
            have_capture_time_=false;
    }
    void setCorrelationsEnabled(bool enabled) {
        std::lock_guard<std::mutex> lock(session_mutex_);
        if (correlations_enabled.exchange(enabled, std::memory_order_acq_rel) == enabled) return;
        if (enabled) neurons_enabled.store(false, std::memory_order_release);
        if (enabled) live_weights_enabled.store(false, std::memory_order_release);
        resetCorrelationHistory(true);
        setMessage(enabled ? "Collecte des activations sur les prochaines entrées du modèle…"
                           : "Retour à la projection des paramètres et activations…");
    }
    void setNeuronsEnabled(bool enabled) {
        std::lock_guard<std::mutex> lock(session_mutex_);
        if (neurons_enabled.exchange(enabled, std::memory_order_acq_rel) == enabled) return;
        if (enabled) correlations_enabled.store(false, std::memory_order_release);
        if (enabled) live_weights_enabled.store(false, std::memory_order_release);
        resetCorrelationHistory(true);
        setMessage(enabled ? "Capture de la répartition des neurones au prochain point sûr…"
                           : "Retour à la projection des paramètres et activations…");
    }
    void setLiveWeightsEnabled(bool enabled) {
        std::lock_guard<std::mutex> lock(session_mutex_);
        if (live_weights_enabled.exchange(enabled, std::memory_order_acq_rel) == enabled) return;
        if (enabled) {
            correlations_enabled.store(false, std::memory_order_release);
            neurons_enabled.store(false, std::memory_order_release);
            realtime_enabled.store(true, std::memory_order_release);
        }
        resetCorrelationHistory(true);
        setMessage(enabled ? "Capture des poids appris après les mises à jour du modèle…"
                           : "Retour à la projection des paramètres et activations…");
    }
    void clearCorrelations() {
        std::lock_guard<std::mutex> lock(session_mutex_);
        resetCorrelationHistory(correlations_enabled.load(std::memory_order_acquire));
        if (correlations_enabled.load(std::memory_order_acquire))
            setMessage("Historique effacé : nouvelles observations attendues…");
    }
    // Called only at the end of a successful model forward. Collection is
    // independent of refresh cadence: the bounded window follows actual inputs,
    // whereas Temps réel controls how often the displayed graph is recomputed.
    bool beginCorrelationObservation(uint64_t expected_generation, uint64_t forward_serial,
                                     uint64_t& epoch) {
        std::lock_guard<std::mutex> lock(session_mutex_);
        if (!available.load(std::memory_order_acquire) ||
            !correlations_enabled.load(std::memory_order_acquire) ||
            !view_active.load(std::memory_order_acquire) || !forward_serial ||
            generation.load(std::memory_order_acquire) != expected_generation ||
            forward_serial <= last_correlation_forward_serial_) return false;
        last_correlation_forward_serial_ = forward_serial;
        epoch = correlation_epoch.load(std::memory_order_acquire);
        return true;
    }
    bool submitCorrelationObservation(LatentCorrelationObservation observation) {
        std::lock_guard<std::mutex> lock(session_mutex_);
        if (!available.load(std::memory_order_acquire) ||
            !correlations_enabled.load(std::memory_order_acquire) ||
            !view_active.load(std::memory_order_acquire) ||
            observation.generation != generation.load(std::memory_order_acquire) ||
            observation.epoch != correlation_epoch.load(std::memory_order_acquire) ||
            !observation.forward_serial || observation.values.empty() ||
            observation.values.size() > LatentCorrelationWindow::max_features ||
            observation.features.size() != observation.values.size()) return false;
        for (float value : observation.values) if (!std::isfinite(value)) return false;
        if (observation.model_name.empty()) observation.model_name = model_name_;
        if (observation.architecture.empty()) observation.architecture = architecture_;
        constexpr size_t pending_limit = 32;
        if (pending_correlations_.size() == pending_limit)
            pending_correlations_.erase(pending_correlations_.begin());
        pending_correlations_.push_back(std::move(observation));
        correlation_ready.store(true, std::memory_order_release);
        return true;
    }
    std::vector<LatentCorrelationObservation> takeCorrelationObservations() {
        std::lock_guard<std::mutex> lock(session_mutex_);
        std::vector<LatentCorrelationObservation> result;
        result.swap(pending_correlations_);
        correlation_ready.store(false, std::memory_order_release);
        return result;
    }
    bool updateCorrelationCount(uint64_t token, uint64_t epoch, size_t count) {
        std::lock_guard<std::mutex> lock(session_mutex_);
        if (!correlations_enabled.load(std::memory_order_acquire) ||
            token != generation.load(std::memory_order_acquire) ||
            epoch != correlation_epoch.load(std::memory_order_acquire)) return false;
        correlation_observations.store(count, std::memory_order_release);
        return true;
    }
    bool beginCorrelationProjection(uint64_t token, uint64_t epoch, uint64_t data_revision,
                                    bool ready, uint64_t& revision,
                                    Clock::time_point now = Clock::now()) {
        std::lock_guard<std::mutex> lock(session_mutex_);
        if (!available.load(std::memory_order_acquire) ||
            !correlations_enabled.load(std::memory_order_acquire) ||
            !view_active.load(std::memory_order_acquire) ||
            token != generation.load(std::memory_order_acquire) ||
            epoch != correlation_epoch.load(std::memory_order_acquire) || capture_inflight_) return false;
        const auto previous = std::atomic_load_explicit(&projection_, std::memory_order_acquire);
        const bool first_ready = ready && (!previous || !previous->correlation_ready);
        const bool accumulating = (!previous || !previous->correlation_ready) &&
                                  data_revision != last_correlation_data_revision_;
        const bool manual = requested.load(std::memory_order_acquire);
        const bool due = !have_capture_time_ || (now >= last_capture_time_ && now-last_capture_time_ >= realtime_interval);
        const bool realtime = realtime_enabled.load(std::memory_order_acquire) && due &&
                              data_revision != last_correlation_data_revision_;
        if (!manual && !first_ready && !accumulating && !realtime) return false;
        requested.store(false, std::memory_order_release);
        inflight_generation_ = token;
        inflight_revision_ = ++latest_revision_;
        revision = inflight_revision_;
        capture_inflight_ = true;
        awaiting_raw_ = false;
        last_capture_time_ = now;
        have_capture_time_ = true;
        last_correlation_data_revision_ = data_revision;
        state.store(State::Projecting, std::memory_order_release);
        setMessage(ready ? "Corrélations de Pearson et ACP des profils d’activations…"
                         : "Accumulation des observations du corpus…");
        return true;
    }
    uint64_t resetSession(std::string model_name,std::string architecture,bool active=true) {
        std::lock_guard<std::mutex> lock(session_mutex_);
        const uint64_t token=generation.fetch_add(1,std::memory_order_acq_rel)+1;
        latest_revision_=0;
        capture_inflight_=false;
        awaiting_raw_=false;
        inflight_generation_=inflight_revision_=0;
        have_capture_time_=false;
        resetCorrelationHistory(false);
        model_name_=std::move(model_name);architecture_=std::move(architecture);
        const bool configured=!model_name_.empty() || !architecture_.empty();
        requested.store(false,std::memory_order_release);
        raw_ready.store(false,std::memory_order_release);
        std::atomic_store_explicit(&pending_raw_,std::shared_ptr<const LatentSpaceRawSnapshot>{},std::memory_order_release);
        std::atomic_store_explicit(&projection_,std::shared_ptr<const LatentSpaceProjection>{},std::memory_order_release);
        available.store(configured,std::memory_order_release);
        training_active.store(configured && active,std::memory_order_release);
        state.store(configured?State::Ready:State::Unavailable,std::memory_order_release);
        setMessage(configured?"Capture prête pour "+(model_name_.empty()?architecture_:model_name_):"Aucun modèle configuré pour la projection 3D");
        return token;
    }
    std::pair<std::string,std::string> modelIdentitySnapshot() const {
        std::lock_guard<std::mutex> lock(session_mutex_);
        return {model_name_,architecture_};
    }
    void requestSnapshot() {
        std::lock_guard<std::mutex> lock(session_mutex_);
        if(!available.load(std::memory_order_acquire))return;
        requested.store(true,std::memory_order_release);
        const auto previous=state.load(std::memory_order_acquire);
        if(previous!=State::Capturing && previous!=State::Projecting)
            state.store(State::Requested,std::memory_order_release);
        setMessage(correlations_enabled.load(std::memory_order_acquire)
            ? "Actualisation des corrélations sur les activations observées…"
            : live_weights_enabled.load(std::memory_order_acquire)
                ? "Capture des poids actifs au prochain point sûr du modèle…"
            : neurons_enabled.load(std::memory_order_acquire)
                ? "Capture de la répartition des neurones à la prochaine opération du modèle…"
                : "Capture des vecteurs à la prochaine opération du modèle…");
    }
    // Explicit time keeps cadence tests deterministic. Production callers use
    // steady_clock; no timer/UI thread ever reads a live model to capture it.
    bool beginCapture(uint64_t& token,bool force=false,uint64_t expected_generation=0,
                      Clock::time_point now=Clock::now(),uint64_t* revision=nullptr,
                      bool* neuron_capture=nullptr,bool* live_weight_capture=nullptr) {
        std::lock_guard<std::mutex> lock(session_mutex_);
        if (correlations_enabled.load(std::memory_order_acquire)) return false;
        if(!available.load(std::memory_order_acquire))return false;
        if(expected_generation && expected_generation!=generation.load(std::memory_order_acquire))return false;
        const bool manual=force || requested.load(std::memory_order_acquire);
        const bool realtime=realtime_enabled.load(std::memory_order_acquire) && view_active.load(std::memory_order_acquire);
        const bool due=!have_capture_time_ || (now>=last_capture_time_ && now-last_capture_time_>=realtime_interval);
        // Automatic refresh must let the current capture, mailbox and PCA
        // complete. Otherwise fast optimizer steps could starve publication.
        if(!manual && (!realtime || !due || capture_inflight_ || raw_ready.load(std::memory_order_acquire)))return false;
        requested.store(false,std::memory_order_release);
        token=generation.load(std::memory_order_acquire);
        // Reserve the revision now: even a manual refresh still being captured
        // supersedes an older worker result, before submitting its raw data.
        inflight_generation_=token;
        inflight_revision_=++latest_revision_;
        capture_inflight_=true;
        awaiting_raw_=true;
        last_capture_time_=now;
        have_capture_time_=true;
        if(revision)*revision=inflight_revision_;
        const bool neurons = neurons_enabled.load(std::memory_order_acquire);
        const bool live_weights = live_weights_enabled.load(std::memory_order_acquire);
        if(neuron_capture)*neuron_capture=neurons;
        if(live_weight_capture)*live_weight_capture=live_weights;
        state.store(State::Capturing,std::memory_order_release);
        setMessage(live_weights ? "Échantillonnage des coefficients appris du modèle…"
                   : neurons ? "Capture des couches, unités et flux du modèle…"
                           : "Échantillonnage des paramètres et des activations du modèle…");
        return true;
    }
    bool submitRawSnapshot(LatentSpaceRawSnapshot snapshot) {
        std::lock_guard<std::mutex> lock(session_mutex_);
        if (correlations_enabled.load(std::memory_order_acquire)) return false;
        if(!available.load(std::memory_order_acquire))return false;
        if(snapshot.neuron_mode != neurons_enabled.load(std::memory_order_acquire))return false;
        if(snapshot.live_weight_mode != live_weights_enabled.load(std::memory_order_acquire))return false;
        if(snapshot.live_weight_mode && snapshot.neuron_mode)return false;
        if(snapshot.live_weight_mode) {
            const auto epoch = correlation_epoch.load(std::memory_order_acquire);
            if(snapshot.live_weight_epoch && snapshot.live_weight_epoch != epoch)return false;
            snapshot.live_weight_epoch = epoch;
        }
        const auto token=generation.load(std::memory_order_acquire);
        if(snapshot.generation!=token)return false;
        if(snapshot.revision) {
            if(!capture_inflight_ || !awaiting_raw_ || snapshot.revision!=inflight_revision_ ||
               token!=inflight_generation_ || snapshot.revision!=latest_revision_)return false;
        } else if(capture_inflight_ && awaiting_raw_ && token==inflight_generation_) {
            snapshot.revision=inflight_revision_;
        } else {
            // Preserve direct/manual submissions used by monitors and clients
            // that did not explicitly call beginCapture first.
            snapshot.revision=++latest_revision_;
            inflight_generation_=token;
            inflight_revision_=snapshot.revision;
            capture_inflight_=true;
            last_capture_time_=Clock::now();
            have_capture_time_=true;
        }
        awaiting_raw_=false;
        if(snapshot.model_name.empty())snapshot.model_name=model_name_;
        if(snapshot.architecture.empty())snapshot.architecture=architecture_;
        std::atomic_store_explicit(&pending_raw_,
            std::make_shared<const LatentSpaceRawSnapshot>(std::move(snapshot)),std::memory_order_release);
        raw_ready.store(true,std::memory_order_release);
        return true;
    }
    std::shared_ptr<const LatentSpaceRawSnapshot> takeRawSnapshot() {
        std::lock_guard<std::mutex> lock(session_mutex_);
        auto snapshot=std::atomic_exchange_explicit(&pending_raw_,
            std::shared_ptr<const LatentSpaceRawSnapshot>{},std::memory_order_acq_rel);
        raw_ready.store(false,std::memory_order_release);
        if(snapshot && (snapshot->generation!=generation.load(std::memory_order_acquire) || snapshot->revision!=latest_revision_))return {};
        return snapshot;
    }
    bool beginProjection(uint64_t token,uint64_t revision=0,bool neuron_mode=false,bool live_weight_mode=false) {
        std::lock_guard<std::mutex> lock(session_mutex_);
        if (correlations_enabled.load(std::memory_order_acquire)) return false;
        if (neuron_mode != neurons_enabled.load(std::memory_order_acquire)) return false;
        if (live_weight_mode != live_weights_enabled.load(std::memory_order_acquire)) return false;
        if (neuron_mode && live_weight_mode) return false;
        if(!available.load(std::memory_order_acquire) || token!=generation.load(std::memory_order_acquire))return false;
        if(revision && revision!=latest_revision_)return false;
        if(capture_inflight_ && (token!=inflight_generation_ || (revision && revision!=inflight_revision_) || awaiting_raw_))return false;
        state.store(State::Projecting,std::memory_order_release);
        setMessage(live_weight_mode ? "Disposition DL des poids et calcul de leurs variations…"
                   : neuron_mode ? "Disposition 3D des neurones selon le graphe…"
                               : "ACP en cours sur les vecteurs capturés…");
        return true;
    }
    bool publishProjection(LatentSpaceProjection projection,
            std::shared_ptr<const LatentSpaceProjection>* published = nullptr) {
        std::lock_guard<std::mutex> lock(session_mutex_);
        if(!available.load(std::memory_order_acquire) || projection.generation!=generation.load(std::memory_order_acquire))return false;
        if (projection.correlation_mode != correlations_enabled.load(std::memory_order_acquire)) return false;
        if (projection.neuron_mode != neurons_enabled.load(std::memory_order_acquire)) return false;
        if (projection.live_weight_mode != live_weights_enabled.load(std::memory_order_acquire)) return false;
        if (projection.live_weight_mode && (projection.neuron_mode || projection.correlation_mode)) return false;
        if (projection.live_weight_mode && projection.live_weight_epoch != correlation_epoch.load(std::memory_order_acquire)) return false;
        if (projection.neuron_mode && projection.correlation_mode) return false;
        if (projection.correlation_mode && projection.correlation_epoch != correlation_epoch.load(std::memory_order_acquire)) return false;
        if(projection.revision!=latest_revision_)return false;
        if(capture_inflight_ && (projection.generation!=inflight_generation_ || projection.revision!=inflight_revision_ || awaiting_raw_))return false;
        const bool correlations = projection.correlation_mode;
        const bool neurons = projection.neuron_mode;
        const bool live_weights = projection.live_weight_mode;
        const std::string correlation_message = projection.correlation_status;
        auto immutable = std::make_shared<const LatentSpaceProjection>(std::move(projection));
        if (published) *published = immutable;
        std::atomic_store_explicit(&projection_, std::move(immutable), std::memory_order_release);
        finishInflight();
        const bool pending=requested.load(std::memory_order_acquire);
        state.store(pending?State::Requested:State::Complete,std::memory_order_release);
        setMessage(pending ? (correlations ? "Corrélations prêtes ; actualisation en attente"
                                          : "Projection prête; nouvelle capture en attente")
                           : (correlations ? correlation_message : live_weights ? "Poids actifs DL prêts"
                                           : neurons ? "Répartition des neurones prête" : "Projection 3D prête"));
        return true;
    }
    std::shared_ptr<const LatentSpaceProjection> projectionSnapshot() const {
        return std::atomic_load_explicit(&projection_,std::memory_order_acquire);
    }
    void fail(std::string message,uint64_t token=0,uint64_t revision=0,
              uint64_t expected_correlation_epoch=0) {
        std::lock_guard<std::mutex> lock(session_mutex_);
        if (expected_correlation_epoch && expected_correlation_epoch != correlation_epoch.load(std::memory_order_acquire)) return;
        if(token && token!=generation.load(std::memory_order_acquire))return;
        if(revision && revision!=latest_revision_)return;
        if(capture_inflight_ && ((token && token!=inflight_generation_) || (revision && revision!=inflight_revision_)))return;
        finishInflight();
        requested.store(false,std::memory_order_release);
        state.store(State::Failed,std::memory_order_release);
        setMessage(std::move(message));
    }

private:
    // session_mutex_ is held; toggles/reset invalidate workers without changing
    // the model generation to which the model-thread capture hook is bound.
    void resetCorrelationHistory(bool invalidate_projection) {
        correlation_epoch.fetch_add(1, std::memory_order_acq_rel);
        pending_correlations_.clear();
        correlation_ready.store(false, std::memory_order_release);
        correlation_observations.store(0, std::memory_order_release);
        last_correlation_forward_serial_ = 0;
        last_correlation_data_revision_ = 0;
        if (!invalidate_projection) return;
        ++latest_revision_;
        finishInflight();
        std::atomic_store_explicit(&projection_, std::shared_ptr<const LatentSpaceProjection>{}, std::memory_order_release);
        const bool configured = available.load(std::memory_order_acquire);
        requested.store(configured, std::memory_order_release);
        state.store(configured ? State::Requested : State::Unavailable, std::memory_order_release);
        have_capture_time_ = false;
    }
    // session_mutex_ is held. A stale completion/failure never reaches here.
    void finishInflight() {
        capture_inflight_=false;
        awaiting_raw_=false;
        inflight_generation_=inflight_revision_=0;
        raw_ready.store(false,std::memory_order_release);
        std::atomic_store_explicit(&pending_raw_,std::shared_ptr<const LatentSpaceRawSnapshot>{},std::memory_order_release);
    }
    mutable std::mutex session_mutex_;
    std::string model_name_,architecture_;
    uint64_t latest_revision_=0;
    uint64_t inflight_generation_=0,inflight_revision_=0;
    bool capture_inflight_=false,awaiting_raw_=false,have_capture_time_=false;
    Clock::time_point last_capture_time_{};
    mutable std::mutex message_mutex_;
    std::string message_;
    std::shared_ptr<const LatentSpaceRawSnapshot> pending_raw_;
    std::shared_ptr<const LatentSpaceProjection> projection_;
    std::vector<LatentCorrelationObservation> pending_correlations_;
    uint64_t last_correlation_forward_serial_ = 0;
    uint64_t last_correlation_data_revision_ = 0;
};

} // namespace mimir::viz
