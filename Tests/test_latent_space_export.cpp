#include "viz/LatentSpaceExport.hpp"
#include "viz/TrainingCapture.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using namespace mimir::viz;
using Json = nlohmann::json;

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

LatentSpaceProjection fixture() {
    LatentSpaceProjection result;
    result.generation = 9007199254741001ULL;
    result.revision = 19;
    result.model_name = "Modèle générique";
    result.architecture = "branched_conv4d";
    result.optimizer_step = 321;
    result.d_model = 64;
    result.latent_dim = 17;
    result.sequence_length = 113;
    result.position_encoding = "rope";
    result.feature_dimension = 97;
    result.effective_rank = 2;
    result.explained_variance_percent = {{72.5f, 23.25f, 4.25f}};
    for (size_t index = 0; index < 3; ++index) {
        LatentSpacePoint point;
        point.position = {{float(index) - .75f, float(index) * -.375f, .125f}};
        point.label = "L2/latent canal " + std::to_string(index);
        point.group = LatentVectorGroup::SequenceActivation;
        point.layer = 2;
        point.sequence_position = int(index) + 7;
        point.vector_norm = 3.25f + float(index);
        point.original_dimension = 113;
        point.feature_offset = 7;
        point.sampled_dimension = 31;
        point.activation_mean = -.125f;
        point.activation_stddev = .375f;
        point.feature_index = index + 13;
        point.pooling = "moyenne spatiale";
        point.neuron_kind = index == 2 ? "opération" : "canal";
        point.neuron_unit_index = index + 5;
        point.neuron_node_index = index + 9;
        point.neuron_layer_units = index == 2 ? 0 : 37;
        point.neuron_is_operation = index == 2;
        result.points.push_back(point);
    }
    result.points[2].live_weight_coefficient = 1;
    result.correlation_ready = true;
    result.correlation_epoch = 4;
    result.correlation_observations = 12;
    result.correlation_window = 128;
    result.correlation_training_observations = 9;
    result.correlation_inference_observations = 3;
    result.correlation_first_step = 301;
    result.correlation_last_step = 321;
    result.correlation_constant_features = 5;
    result.correlation_edge_candidates = 3;
    result.correlation_status = "12 observations, dont 3 en inférence";
    result.correlation_edges = {{0, 1, .875f}, {1, 2, -.75f}, {0, 2, .375f}};
    result.neuron_total_units = 4099;
    result.neuron_total_layers = 11;
    result.neuron_unknown_layers = 1;
    result.neuron_sampling = true;
    result.neuron_links = {{0, 1, 0.f, false, true}, {1, 2, -.25f, true, false}};
    result.live_weight_epoch = 8;
    result.live_weight_previous_step = 320;
    result.live_weight_baseline_available = true;
    result.live_weight_total_parameters = 85001;
    result.live_weight_nonfinite = 2;
    result.live_weight_skipped_layers = 3;
    result.live_weight_shared_layers = 1;
    result.live_weight_sampled = true;
    LiveWeightCoefficient mapped;
    mapped.key = "L2:85";
    mapped.label = "latent/weight[85]";
    mapped.parameter_index = 85;
    mapped.value = -.875f;
    mapped.delta = -1.234567890123456e-40;
    mapped.has_delta = true;
    mapped.mapped = true;
    LiveWeightCoefficient glyph;
    glyph.key = "L2:86";
    glyph.label = "latent/bias[86]";
    glyph.parameter_index = 86;
    glyph.value = .125f;
    glyph.delta = .23456789012345678;
    glyph.has_delta = true;
    glyph.point_index = 2;
    glyph.is_bias = true;
    result.live_weight_coefficients = {mapped, glyph};
    result.live_weight_links = {{0, 1, 0}};
    return result;
}

LatentSpaceExportView viewFixture() {
    LatentSpaceExportView view;
    view.yaw = -.6875;
    view.pitch = .3125;
    view.zoom = 2.25;
    view.correlation_threshold = .8125;
    view.width = 931;
    view.height = 617;
    view.correlation_collected_observations = 23;
    return view;
}

void documentCases() {
    auto projection = fixture();
    const auto view = viewFixture();
    const std::array<std::string, 4> modes{{"structural", "correlations", "neurons", "live_weights"}};
    for (size_t mode = 0; mode < modes.size(); ++mode) {
        projection.correlation_mode = mode == 1;
        projection.neuron_mode = mode == 2;
        projection.live_weight_mode = mode == 3;
        const auto result = latentSpaceExportDocument(projection, view);
        check(result.at("format") == "mimir.latent-space-3d" && result.at("version") == 1,
              "export format/version missing");
        check(result.at("mode") == modes[mode], "wrong exported 3D mode");
        const auto& model = result.at("model");
        check(model.at("name") == projection.model_name && model.at("architecture") == projection.architecture,
              "model identity lost");
        check(model.at("generation").get<uint64_t>() == projection.generation &&
              model.at("revision") == 19 && model.at("optimizer_step") == 321,
              "immutable snapshot provenance lost or rounded");
        check(model.at("generation_decimal") == "9007199254741001",
              "HTML identifier cannot retain a generation beyond JavaScript's safe integers");
        check(model.at("d_model") == 64 && model.at("latent_dim") == 17 &&
              model.at("sequence_length") == 113 && model.at("position_encoding") == "rope",
              "model projection configuration lost");
        check(result.at("view") == Json{{"organic", false}, {"heatspots", false}, {"yaw", -.6875}, {"pitch", .3125}, {"zoom", 2.25},
              {"correlation_threshold", .8125}, {"correlation_show_all_candidates", false},
              {"width", 931}, {"height", 617}},
              "camera, displayed dimensions or correlation threshold changed");
        check(result.at("projection").at("feature_dimension") == 97 &&
              result.at("projection").at("effective_rank") == 2 &&
              result.at("projection").at("explained_variance_percent") == Json::array({72.5, 23.25, 4.25}),
              "PCA metadata lost");
        check(result.at("points").size() == 3, "sampled graph points lost");
        const auto& point = result.at("points").at(1);
        check(point.at("index") == 1 && point.at("position") == Json::array({.25, -.375, .125}) &&
              point.at("label") == projection.points[1].label,
              "point coordinates or identity changed");
        check(point.at("layer") == 2 && point.at("sequence_position") == 8 &&
              point.at("vector_norm") == 4.25 && point.at("original_dimension") == 113 &&
              point.at("feature_offset") == 7 && point.at("sampled_dimension") == 31 &&
              point.at("activation_mean") == -.125 && point.at("activation_stddev") == .375 &&
              point.at("feature_index") == 14 && point.at("pooling") == "moyenne spatiale",
              "point sampling or activation provenance lost");
        check(point.at("neuron_unit_index") == 6 && point.at("neuron_node_index") == 10 &&
              point.at("neuron_layer_units") == 37 && point.at("neuron_kind") == "canal" &&
              point.at("neuron_is_operation") == false && !point.at("group_name").get<std::string>().empty(),
              "neuron/operation/group metadata lost");
        check(result.at("points").at(2).at("neuron_is_operation") == true &&
              result.at("points").at(0).at("live_weight_coefficient").is_null() &&
              result.at("points").at(2).at("live_weight_coefficient") == 1,
              "unmapped coefficient sentinel is not null");
        const auto& correlations = result.at("correlations");
        check(correlations.at("observations") == 12 && correlations.at("window") == 128 &&
              correlations.at("collected_observations") == 23 &&
              correlations.at("training_observations") == 9 && correlations.at("inference_observations") == 3 &&
              correlations.at("first_step") == 301 && correlations.at("last_step") == 321 &&
              correlations.at("constant_features") == 5 && correlations.at("edge_candidates") == 3 &&
              correlations.at("epoch") == 4 && correlations.at("ready") == true &&
              correlations.at("status") == projection.correlation_status,
              "correlation observations and excluded constant features lost");
        const auto& correlation_links = result.at("links").at("correlations");
        check(correlation_links.size() == 3 && correlation_links.at(1).at("source") == 1 &&
              correlation_links.at(1).at("target") == 2 && correlation_links.at(1).at("correlation") == -.75 &&
              correlation_links.at(2).at("correlation") == .375,
              "signed correlations or below-threshold raw correlations lost");
        const auto& neurons = result.at("neurons");
        check(neurons.at("total_units") == 4099 && neurons.at("total_layers") == 11 &&
              neurons.at("unknown_layers") == 1 && neurons.at("sampling") == true,
              "neuron sampling or unknown-layer coverage lost");
        const auto& neuron_links = result.at("links").at("neurons");
        check(neuron_links.size() == 2 && neuron_links.at(0).at("weighted") == false &&
              neuron_links.at(0).at("aggregated") == true && neuron_links.at(1).at("weight") == -.25 &&
              neuron_links.at(1).at("weighted") == true && neuron_links.at(1).at("aggregated") == false,
              "aggregated topology links were confused with learned weights");
        const auto& live = result.at("live_weights");
        check(live.at("epoch") == 8 && live.at("previous_step") == 320 &&
              live.at("baseline_available") == true && live.at("total_parameters") == 85001 &&
              live.at("nonfinite") == 2 && live.at("skipped_layers") == 3 &&
              live.at("shared_layers") == 1 && live.at("sampled") == true,
              "live-weight baseline or omitted coverage lost");
        const auto& coefficients = live.at("coefficients");
        check(coefficients.size() == 2 && coefficients.at(0).at("parameter_index") == 85 &&
              coefficients.at(0).at("key") == "L2:85" && coefficients.at(0).at("value") == -.875 &&
              coefficients.at(0).at("point_index").is_null() && coefficients.at(0).at("mapped") == true,
              "signed mapped coefficient or unmapped point sentinel lost");
        check(coefficients.at(0).at("delta").get<double>() == projection.live_weight_coefficients[0].delta &&
              coefficients.at(1).at("delta").get<double>() == projection.live_weight_coefficients[1].delta &&
              coefficients.at(1).at("point_index") == 2 && coefficients.at(1).at("is_bias") == true &&
              coefficients.at(1).at("has_delta") == true,
              "coefficient delta was narrowed to float or bias identity lost");
        const auto& live_links = result.at("links").at("live_weights");
        check(live_links.size() == 1 && live_links.at(0).at("source") == 0 &&
              live_links.at(0).at("target") == 1 && live_links.at(0).at("coefficient_index") == 0,
              "exact learned-weight endpoints changed");
        check(Json::parse(result.dump()) == result, "JSON export did not round-trip exactly");
        check(result.at("sampling").at("exported_points") == 3 &&
              result.at("sampling").at("exported_coefficients") == 2 && !result.at("notes").empty(),
              "export omits displayed sampling coverage and interpretation limits");
        check(projection.points[0].position[0] == -.75f && projection.points[0].live_weight_coefficient ==
              std::numeric_limits<size_t>::max() && projection.correlation_edges[1].correlation == -.75f,
              "export mutated its source projection");
    }
}

void allCorrelationCandidatesCase() {
    auto projection = fixture();
    projection.correlation_mode = true;
    projection.points.resize(37, projection.points.front());
    projection.correlation_candidate_edges.clear();
    for (size_t source = 0; source < projection.points.size(); ++source)
        for (size_t target = source + 1; target < projection.points.size(); ++target)
            projection.correlation_candidate_edges.push_back({source, target, -.375f});
    projection.correlation_edge_candidates = projection.correlation_candidate_edges.size();
    projection.correlation_edges.assign(projection.correlation_candidate_edges.begin(),
        projection.correlation_candidate_edges.begin() + LatentCorrelationWindow::max_edges);
    auto view = viewFixture();
    view.correlation_show_all_candidates = true;
    const auto result = latentSpaceExportDocument(projection, view);
    const auto& links = result.at("links").at("correlations");
    check(links.size() == 666 && links.size() > projection.correlation_edges.size(),
          "export discarded candidates beyond the default link limit");
    check(links.at(665).at("source") == 35 && links.at(665).at("target") == 36 &&
          links.at(665).at("correlation") == -.375,
          "additional candidate endpoints or signed values changed");
    check(result.at("view").at("correlation_show_all_candidates") == true &&
          result.at("view").at("correlation_threshold") == .8125,
          "all-candidates view lost its state or replaced the normal threshold");
    check(result.at("sampling").at("exported_correlation_links") == 666 &&
          result.at("sampling").at("correlation_default_link_limit") == 512 &&
          result.at("sampling").at("correlation_candidate_link_limit") == 8128 &&
          result.at("sampling").at("correlation_link_limit") == 8128,
          "exported link limits no longer describe the retained candidates");
    view.correlation_show_all_candidates = false;
    const auto normal = latentSpaceExportDocument(projection, view);
    check(normal.at("links").at("correlations") == links &&
          normal.at("view").at("correlation_show_all_candidates") == false,
          "normal export discarded candidates needed by its interactive option");
    const auto html = latentSpaceExportHtml(result);
    check(html.find("id=\"all-candidates\"") != std::string::npos &&
          html.find("Tous les candidats valides") != std::string::npos,
          "offline correlation candidate control missing");
    check(Json::parse(result.dump()) == result, "complete candidate export failed to round-trip");
}

void sequencePathsCase() {
    LatentSpaceProjection projection;
    check(latentSpaceExportDocument(projection, LatentSpaceExportView{}).at("correlations")
              .at("collected_observations").is_null(),
          "unknown displayed observation count is not null");
    const std::array<std::string, 5> labels{{"block/A position 3", "block/A position 1",
        "block/B position 0", "block/A position 2", "block/A position 4"}};
    const int positions[]{3, 1, 0, 2, 4};
    for (size_t index = 0; index < labels.size(); ++index) {
        LatentSpacePoint point;
        point.label = labels[index];
        point.group = LatentVectorGroup::SequenceActivation;
        point.layer = index == 4 ? 3 : 2;
        point.sequence_position = positions[index];
        projection.points.push_back(point);
    }
    const auto result = latentSpaceExportDocument(projection, viewFixture());
    const auto& paths = result.at("links").at("sequences");
    check(paths.size() == 2 && paths.at(0).at("source") == 1 && paths.at(0).at("target") == 3 &&
          paths.at(1).at("source") == 3 && paths.at(1).at("target") == 0,
          "sequence paths are not ordered within the same tensor and layer");
    projection.correlation_mode = true;
    check(latentSpaceExportDocument(projection, viewFixture()).at("links").at("sequences").empty(),
          "sequence links were invented for Pearson mode");
    projection.correlation_mode = false;
    projection.neuron_mode = true;
    check(latentSpaceExportDocument(projection, viewFixture()).at("links").at("sequences").empty(),
          "sequence links were invented for topology mode");
}

void htmlCase() {
    auto projection = fixture();
    const std::string hostile = "</script><img src=x onerror=alert(1)> & \"é\"\xe2\x80\xa8\xe2\x80\xa9";
    projection.model_name = hostile;
    projection.points[0].label = hostile;
    projection.live_weight_coefficients[0].label = hostile;
    const auto document = latentSpaceExportDocument(projection, viewFixture());
    const auto html = latentSpaceExportHtml(document);
    check(html.find("<canvas") != std::string::npos, "interactive offline canvas missing");
    check(html.find("textContent") != std::string::npos && html.find("innerHTML") == std::string::npos,
          "model labels are not inserted as text");
    check(html.find("<script src=") == std::string::npos && html.find("fetch(") == std::string::npos &&
          html.find("https://") == std::string::npos && html.find("http://") == std::string::npos,
          "HTML export needs external resources");
    check(html.find(hostile) == std::string::npos && html.find("<img src=x") == std::string::npos &&
          html.find("\xe2\x80\xa8") == std::string::npos && html.find("\xe2\x80\xa9") == std::string::npos,
          "hostile labels escaped out of the embedded JSON");
    const auto id = html.find("id=\"graph-data\"");
    check(id != std::string::npos, "embedded graph document missing");
    const auto start = html.find('>', id);
    const auto end = html.find("</script>", start);
    check(start != std::string::npos && end != std::string::npos, "embedded graph document is truncated");
    check(Json::parse(html.substr(start + 1, end - start - 1)) == document,
          "HTML embedding changed model labels, graph data or double precision");
}

void captureCase() {
    const auto root = std::filesystem::temp_directory_path() / ("mimir-capture-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup { std::filesystem::path path; ~Cleanup() { std::filesystem::remove_all(path); } } cleanup{root};
    TrainingCapture capture;
    auto projection = fixture();
    const auto csv = (root / "loss.csv").string();
    capture.append(projection, csv, Json{{"loss",.25},{"optimizer_step",321}});
    const auto path = capture.path();
    projection.optimizer_step++;
    projection.points[0].vector_norm = 7.25f;
    capture.append(projection, csv, Json{{"loss",.125},{"optimizer_step",322}});
    std::ifstream input(path);
    std::string line;
    std::getline(input,line); const auto first = Json::parse(line);
    std::getline(input,line); const auto second = Json::parse(line);
    check(!std::getline(input,line), "unexpected capture frames");
    check(first.at("format") == "mimir.training-capture" && first.at("version") == 1,
          "capture schema missing");
    check(first.at("metrics").at("loss") == .25 && second.at("metrics").at("loss") == .125,
          "capture changed training measurements");
    check(first.at("graph").at("model").at("optimizer_step") == 321 &&
          second.at("graph").at("model").at("optimizer_step") == 322 &&
          second.at("graph").at("points").at(0).at("vector_norm") == 7.25,
          "capture lost chronological step or exact values");
    check(second.at("capture").at("elapsed_ms").get<double>() >= first.at("capture").at("elapsed_ms").get<double>(),
          "nonmonotonic capture timestamps");
    check(std::filesystem::file_size(path.string()+".html") > 1000, "offline replay viewer missing");
    LatentSpaceExportView view; view.organic = true;
    check(latentSpaceExportDocument(projection,view).at("view").at("organic") == true,
          "organic export style lost");
    TrainingCapture separate;
    separate.append(projection,csv,Json::object());
    check(separate.path() != path, "new training session overwrote the previous capture");
}

void nonfiniteCase() {
    auto projection = fixture();
    auto view = viewFixture();
    projection.points[0].position[1] = std::numeric_limits<float>::infinity();
    projection.points[0].vector_norm = std::numeric_limits<float>::quiet_NaN();
    projection.live_weight_coefficients[0].delta = -std::numeric_limits<double>::infinity();
    view.yaw = std::numeric_limits<double>::quiet_NaN();
    const auto result = latentSpaceExportDocument(projection, view);
    check(result.at("points").at(0).at("position").at(1).is_null() &&
          result.at("points").at(0).at("vector_norm").is_null() &&
          result.at("live_weights").at("coefficients").at(0).at("delta").is_null() &&
          result.at("view").at("yaw").is_null(), "nonfinite numeric values are not explicit nulls");
    check(result.at("numeric_policy").at("nonfinite_values") == 4,
          "nonfinite export accounting is incorrect");
    check(Json::parse(result.dump()) == result, "nonfinite values produced invalid JSON");
}
} // namespace

int main() {
    try {
        documentCases();
        allCorrelationCandidatesCase();
        sequencePathsCase();
        htmlCase();
        nonfiniteCase();
        captureCase();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "PASS 3D export modes, snapshot provenance, exact values and offline HTML escaping\n";
}
