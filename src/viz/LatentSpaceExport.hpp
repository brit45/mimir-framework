#pragma once

#include "LatentSpaceProjection.hpp"
#include "OrganicNetwork.hpp"
#include "../include/json.hpp"

#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace mimir::viz {

// Export only an already displayed, immutable projection. This helper never
// reads model tensors, requests a capture, or advances the training runtime.
struct LatentSpaceExportView {
    bool organic = false;
    bool heatspots = false;
    double yaw = -0.55;
    double pitch = 0.25;
    double zoom = 1.0;
    double correlation_threshold = 0.65;
    bool correlation_show_all_candidates = false;
    unsigned width = 1120;
    unsigned height = 820;
    size_t correlation_collected_observations = std::numeric_limits<size_t>::max();
};

namespace latent_export_detail {
inline nlohmann::json index(size_t value) {
    return value == std::numeric_limits<size_t>::max()
        ? nlohmann::json(nullptr) : nlohmann::json(value);
}

inline nlohmann::json number(double value, size_t& nonfinite) {
    if (std::isfinite(value)) return value;
    ++nonfinite;
    return nullptr;
}

// A JSON string is not safe inside an HTML script element until its HTML
// delimiters are escaped. Escaping the UTF-8 line separators also keeps this
// document compatible with older JavaScript parsers. JSON.parse restores the
// original text; no label is interpolated into executable JavaScript or HTML.
inline std::string scriptJson(const nlohmann::json& document) {
    const std::string input = document.dump();
    std::string output;
    output.reserve(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        switch (input[i]) {
            case '<': output += "\\u003c"; break;
            case '>': output += "\\u003e"; break;
            case '&': output += "\\u0026"; break;
            default:
                if (i + 2 < input.size() && static_cast<unsigned char>(input[i]) == 0xe2 &&
                    static_cast<unsigned char>(input[i + 1]) == 0x80 &&
                    (static_cast<unsigned char>(input[i + 2]) == 0xa8 ||
                     static_cast<unsigned char>(input[i + 2]) == 0xa9)) {
                    output += static_cast<unsigned char>(input[i + 2]) == 0xa8 ? "\\u2028" : "\\u2029";
                    i += 2;
                } else output += input[i];
        }
    }
    return output;
}
} // namespace latent_export_detail

inline nlohmann::json latentSpaceExportDocument(const LatentSpaceProjection& p,
                                               const LatentSpaceExportView& view) {
    using nlohmann::json;
    using latent_export_detail::index;
    size_t nonfinite = 0;
    const auto number = [&](double value) { return latent_export_detail::number(value, nonfinite); };
    const bool topology = p.neuron_mode || p.live_weight_mode;
    const auto& correlation_candidates = correlationEdgesForDisplay(p, true);
    const char* mode = p.live_weight_mode ? "live_weights" : p.neuron_mode ? "neurons"
        : p.correlation_mode ? "correlations" : "structural";
    json result = {
        {"format", "mimir.latent-space-3d"}, {"version", 1}, {"mode", mode},
        {"model", {
            {"name", p.model_name}, {"architecture", p.architecture},
            {"generation", p.generation}, {"revision", p.revision}, {"optimizer_step", p.optimizer_step},
            {"generation_decimal", std::to_string(p.generation)},
            {"revision_decimal", std::to_string(p.revision)},
            {"optimizer_step_decimal", std::to_string(p.optimizer_step)},
            {"d_model", p.d_model}, {"latent_dim", p.latent_dim},
            {"sequence_length", p.sequence_length}, {"position_encoding", p.position_encoding}}},
        {"view", {{"organic", view.organic}, {"heatspots", view.heatspots}, {"yaw", number(view.yaw)}, {"pitch", number(view.pitch)}, {"zoom", number(view.zoom)},
            {"correlation_threshold", number(view.correlation_threshold)},
            {"correlation_show_all_candidates", view.correlation_show_all_candidates},
            {"width", view.width}, {"height", view.height}}},
        {"projection", {{"coordinate_system", topology ? "graph_depth_unit_grid" : "principal_components"},
            {"axes", topology ? json::array({"Profondeur", "Unités Y", "Unités Z"})
                              : json::array({"PC1", "PC2", "PC3"})},
            {"feature_dimension", p.feature_dimension}, {"effective_rank", p.effective_rank},
            {"explained_variance_percent", json::array({number(p.explained_variance_percent[0]),
                number(p.explained_variance_percent[1]), number(p.explained_variance_percent[2])})}}},
        {"correlations", {{"enabled", p.correlation_mode}, {"ready", p.correlation_ready},
            {"epoch", p.correlation_epoch}, {"epoch_decimal", std::to_string(p.correlation_epoch)},
            {"observations", p.correlation_observations}, {"window", p.correlation_window},
            {"collected_observations", index(view.correlation_collected_observations)},
            {"training_observations", p.correlation_training_observations},
            {"inference_observations", p.correlation_inference_observations},
            {"first_step", p.correlation_first_step}, {"last_step", p.correlation_last_step},
            {"constant_features", p.correlation_constant_features},
            {"edge_candidates", p.correlation_edge_candidates}, {"status", p.correlation_status}}},
        {"neurons", {{"enabled", p.neuron_mode}, {"total_units", p.neuron_total_units},
            {"total_layers", p.neuron_total_layers}, {"unknown_layers", p.neuron_unknown_layers},
            {"sampling", p.neuron_sampling}}},
        {"live_weights", {{"enabled", p.live_weight_mode}, {"epoch", p.live_weight_epoch},
            {"epoch_decimal", std::to_string(p.live_weight_epoch)},
            {"previous_step", p.live_weight_previous_step},
            {"previous_step_decimal", std::to_string(p.live_weight_previous_step)},
            {"baseline_available", p.live_weight_baseline_available},
            {"total_parameters", p.live_weight_total_parameters}, {"nonfinite", p.live_weight_nonfinite},
            {"skipped_layers", p.live_weight_skipped_layers}, {"shared_layers", p.live_weight_shared_layers},
            {"sampled", p.live_weight_sampled}, {"coefficients", json::array()}}},
        {"sampling", {{"exported_points", p.points.size()},
            {"exported_correlation_links", correlation_candidates.size()},
            {"exported_neuron_links", p.neuron_links.size()},
            {"exported_live_weight_links", p.live_weight_links.size()},
            {"exported_coefficients", p.live_weight_coefficients.size()},
            {"structural_point_limit", 512}, {"structural_feature_limit", 2048},
            {"correlation_feature_limit", LatentCorrelationWindow::max_features},
            {"correlation_observation_limit", LatentCorrelationWindow::max_observations},
            {"correlation_min_observations", LatentCorrelationWindow::min_observations},
            {"correlation_link_limit", LatentCorrelationWindow::max_edge_candidates},
            {"correlation_default_link_limit", LatentCorrelationWindow::max_edges},
            {"correlation_candidate_link_limit", LatentCorrelationWindow::max_edge_candidates},
            {"correlation_capture_min_abs", LatentCorrelationWindow::min_edge_correlation},
            {"neuron_point_limit", neuronNetworkPointLimit}, {"neuron_link_limit", neuronNetworkLinkLimit},
            {"live_weight_coefficient_limit", liveWeightCoefficientLimit},
            {"live_weight_neuron_point_limit", liveWeightNeuronPointLimit},
            {"live_weight_structural_link_limit", liveWeightStructuralLinkLimit}}},
        {"notes", json::array({
            "Instantané de la projection affichée ; aucune nouvelle inférence ni capture des tenseurs.",
            "Vue diagnostique échantillonnée, non exhaustive ; elle ne mesure pas la compréhension humaine.",
            "ACP : vecteurs normalisés et centrés ; les largeurs hétérogènes sont complétées par zéro pour la projection.",
            "Pearson : corrélations de composantes d'activation avant ACP ; une corrélation n'établit pas une causalité. Les poids et les entrées peuvent varier dans la fenêtre.",
            "Tous les candidats valides : les liens Pearson capturés sont conservés dans le JSON ; la case contourne le seuil visuel et la limite habituelle de 512 liens, sans étendre les composantes ni les observations capturées.",
            "Topologie : X est la profondeur du graphe, Y/Z une grille d'unités. Les unités répétées entre couches ne sont pas un décompte de neurones indépendants.",
            "Liens gris : routage structurel agrégé, pas des synapses apprises individuelles.",
            "Organique : volumes corticaux symboliques à profondeur de graphe conservée, ramifications décoratives ; halos = variations mesurées des poids. Les coordonnées originales sont conservées séparément. Aucune activité biologique n’est simulée.",
            "Poids actifs : coefficients réellement capturés ; seules les correspondances vérifiées sont reliées. Biais et coefficients non mappés restent des glyphes.",
            "Delta : différence entre captures publiées compatibles ; il peut couvrir plusieurs pas. La première capture n'a pas de référence.",
            "Le fichier HTML est autonome et hors ligne ; il ne suit pas les mises à jour ultérieures du modèle."})},
        {"points", json::array()},
        {"links", {{"correlations", json::array()}, {"neurons", json::array()},
            {"live_weights", json::array()}, {"sequences", json::array()}}}
    };
    result["organic_network"]={{"layout",topology ? "folded_lobes_graph_depth" : "principal_components_unchanged"},
        {"activity","normalized_absolute_parameter_delta"},
        {"dendrites","decorative; not additional model connections"}};
    const auto organic_scene=buildOrganicNetworkScene(p);
    const auto derived_number=[](float value)->json {return std::isfinite(value) ? json(value) : json(nullptr);};
    auto& points = result["points"];
    for (size_t i = 0; i < p.points.size(); ++i) {
        const auto& point = p.points[i];
        points.push_back({{"index", index(i)},
            {"position", json::array({number(point.position[0]), number(point.position[1]), number(point.position[2])})},
            {"organic_position", json::array({derived_number(organic_scene.positions[i][0]),derived_number(organic_scene.positions[i][1]),derived_number(organic_scene.positions[i][2])})},
            {"plasticity", number(organic_scene.plasticity[i])},
            {"organic_seed", organicIdentity(point)},
            {"label", point.label}, {"group", static_cast<unsigned>(point.group)},
            {"group_name", latentVectorGroupName(point.group)}, {"layer", point.layer},
            {"sequence_position", point.sequence_position}, {"vector_norm", number(point.vector_norm)},
            {"original_dimension", index(point.original_dimension)}, {"feature_offset", index(point.feature_offset)},
            {"sampled_dimension", index(point.sampled_dimension)},
            {"activation_mean", number(point.activation_mean)}, {"activation_stddev", number(point.activation_stddev)},
            {"feature_index", index(point.feature_index)}, {"pooling", point.pooling},
            {"neuron_kind", point.neuron_kind}, {"neuron_unit_index", index(point.neuron_unit_index)},
            {"neuron_node_index", index(point.neuron_node_index)}, {"neuron_layer_units", index(point.neuron_layer_units)},
            {"neuron_is_operation", point.neuron_is_operation},
            {"live_weight_coefficient", index(point.live_weight_coefficient)}});
    }
    for (size_t i = 0; i < correlation_candidates.size(); ++i) {
        const auto& edge = correlation_candidates[i];
        result["links"]["correlations"].push_back({{"index", index(i)}, {"source", index(edge.source)},
            {"target", index(edge.target)}, {"correlation", number(edge.correlation)}});
    }
    for (size_t i = 0; i < p.neuron_links.size(); ++i) {
        const auto& edge = p.neuron_links[i];
        result["links"]["neurons"].push_back({{"index", index(i)}, {"source", index(edge.source)},
            {"target", index(edge.target)}, {"weight", number(edge.weight)},
            {"weighted", edge.weighted}, {"aggregated", edge.aggregated}});
    }
    for (size_t i = 0; i < p.live_weight_coefficients.size(); ++i) {
        const auto& coefficient = p.live_weight_coefficients[i];
        result["live_weights"]["coefficients"].push_back({{"index", index(i)},
            {"key", coefficient.key}, {"label", coefficient.label}, {"parameter_index", index(coefficient.parameter_index)},
            {"parameter_index_decimal", coefficient.parameter_index == std::numeric_limits<size_t>::max()
                ? json(nullptr) : json(std::to_string(coefficient.parameter_index))},
            {"value", number(coefficient.value)}, {"delta", number(coefficient.delta)},
            {"has_delta", coefficient.has_delta}, {"point_index", index(coefficient.point_index)},
            {"is_bias", coefficient.is_bias}, {"mapped", coefficient.mapped}});
    }
    for (size_t i = 0; i < p.live_weight_links.size(); ++i) {
        const auto& edge = p.live_weight_links[i];
        result["links"]["live_weights"].push_back({{"index", index(i)}, {"source", index(edge.source)},
            {"target", index(edge.target)}, {"coefficient_index", index(edge.coefficient_index)}});
    }
    // Match the view's per-tensor sequence paths. Never connect positions from
    // different layers or tensors, and never invent a path for topology/Pearson.
    if (!topology && !p.correlation_mode) {
        std::map<std::string, std::vector<size_t>> paths;
        for (size_t i = 0; i < p.points.size(); ++i) {
            const auto& point = p.points[i];
            if (point.group != LatentVectorGroup::SequenceActivation || point.sequence_position < 0) continue;
            const size_t suffix = point.label.rfind(" position ");
            if (suffix == std::string::npos) continue;
            paths[std::to_string(point.layer) + ":" + point.label.substr(0, suffix)].push_back(i);
        }
        for (auto& path : paths) {
            auto& ordered = path.second;
            std::sort(ordered.begin(), ordered.end(), [&](size_t a, size_t b) {
                return p.points[a].sequence_position < p.points[b].sequence_position;
            });
            for (size_t i = 1; i < ordered.size(); ++i)
                result["links"]["sequences"].push_back({{"index", result["links"]["sequences"].size()},
                    {"source", ordered[i - 1]}, {"target", ordered[i]}, {"tensor_path", path.first}});
        }
    }
    result["sampling"]["exported_sequence_links"] = result["links"]["sequences"].size();
    result["numeric_policy"] = {{"nonfinite", "null"}, {"nonfinite_values", nonfinite},
        {"unavailable_index", "null"},
        {"integers", "JSON integer values are exact; *_decimal preserves displayed identifiers beyond JavaScript's safe integer range."}};
    return result;
}

inline std::string latentSpaceExportHtml(const nlohmann::json& document) {
    std::string html = R"MIMIR(<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>Mímir · Graphe 3D</title>
<style>
:root{color-scheme:dark;font:14px system-ui,sans-serif;background:#10141d;color:#dde3ef}*{box-sizing:border-box}body{margin:0;padding:18px;max-width:1600px;margin-inline:auto}h1{font-size:21px;margin:0 0 9px}p{line-height:1.5}button,input{font:inherit}button{background:#303c53;border:1px solid #647491;color:inherit;border-radius:5px;padding:8px 12px;cursor:pointer}button:focus-visible,canvas:focus-visible{outline:2px solid #b99bea;outline-offset:3px}.toolbar{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin:12px 0}.threshold{display:flex;gap:8px;align-items:center}.threshold input{width:150px}#canvas{display:block;width:100%;height:540px;background:#0d1119;border:1px solid #536078;border-radius:4px;touch-action:none}#summary,#axes,#legend,#camera{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5}#summary{color:#b9cbe5}#axes,#camera{font-size:12px;color:#acbbd2;margin-top:8px}#legend{font-size:13px;margin:10px 0}#hover{min-height:7em;max-height:24em;overflow:auto;padding:12px;background:#1c2433;border-left:3px solid #b69ce3;white-space:pre-wrap;overflow-wrap:anywhere;font:13px ui-monospace,monospace;line-height:1.5}details{margin:15px 0;border:1px solid #3f4b61;padding:10px;border-radius:5px}summary{cursor:pointer;font-weight:600}#metadata{white-space:pre-wrap;overflow-wrap:anywhere;max-height:35em;overflow:auto}#notes p{margin:8px 0}small{color:#afbad0}
:root{overflow-y:scroll}[hidden]{display:none!important}
</style></head><body>
<h1 id="title">Mímir · Graphe 3D</h1><div id="summary"></div>
<div class="toolbar"><button type="button" id="reset">Réinitialiser la vue</button><button type="button" id="zoom-in" aria-label="Agrandir">Zoom +</button><button type="button" id="zoom-out" aria-label="Réduire">Zoom −</button><label class="threshold" id="threshold-control">Seuil |Pearson| <input type="range" id="threshold" min="0" max="1" step="0.01"><output id="threshold-value"></output></label><label id="all-candidates-control"><input type="checkbox" id="all-candidates"> Tous les candidats valides</label><label><input type="checkbox" id="organic"> Organique</label><label><input type="checkbox" id="heatspots"> Heatspots |Δw|</label><small>Glisser : rotation · Molette : zoom · Flèches : rotation · 0 : réinitialiser</small></div>
<canvas id="canvas" tabindex="0" aria-label="Projection 3D interactive du modèle"></canvas>
<div id="axes"></div><div id="camera"></div><div id="legend"></div>
<pre id="hover" aria-live="polite">Survolez un point ou un lien pour afficher son identité et ses valeurs.</pre>
<details><summary>Contexte, provenance et limites</summary><div id="notes"></div><pre id="metadata"></pre></details>
<script type="application/json" id="graph-data">)MIMIR";
    html += latent_export_detail::scriptJson(document);
    html += R"MIMIR(</script>
<script>
'use strict';
(() => {
const doc=JSON.parse(document.getElementById('graph-data').textContent);
const byId=id=>document.getElementById(id), text=(id,value)=>{byId(id).textContent=value;};
const finite=value=>typeof value==='number'&&Number.isFinite(value);
const fmt=value=>value===null||value===undefined?'indisponible':String(value);
const idx=value=>Number.isSafeInteger(value)&&value>=0;
const model=doc.model||{}, initial=doc.view||{}, projection=doc.projection||{};
const heatspotsInput=byId('heatspots');heatspotsInput.checked=initial.heatspots===true;heatspotsInput.disabled=doc.mode!=='live_weights';heatspotsInput.onchange=()=>schedule();
const organicInput=byId('organic');organicInput.checked=initial.organic===true;organicInput.onchange=()=>schedule();
const corr=doc.correlations||{}, neurons=doc.neurons||{}, live=doc.live_weights||{};
const points=doc.points||[], links=doc.links||{}, coeffs=live.coefficients||[];
const topology=doc.mode==='neurons'||doc.mode==='live_weights', isLive=doc.mode==='live_weights', isCorr=doc.mode==='correlations';
const names={structural:'Vecteurs de poids et activations · ACP',correlations:'Corrélations internes · Pearson',neurons:'Répartition des unités · Topologie',live_weights:'Poids actifs · Coefficients et deltas'};
text('title','Mímir · '+(names[doc.mode]||doc.mode));
let summary='Modèle : '+(model.name||'—')+' · Architecture : '+(model.architecture||'—')+'\nGénération '+(model.generation_decimal??fmt(model.generation))+' · Révision '+(model.revision_decimal??fmt(model.revision))+' · Pas optimiseur '+(model.optimizer_step_decimal??fmt(model.optimizer_step))+' · '+points.length+' points exportés.';
if(isCorr)summary+='\n'+fmt(corr.observations)+'/'+fmt(corr.window)+' observations · entraînement '+fmt(corr.training_observations)+' · inférence '+fmt(corr.inference_observations)+' · pas '+fmt(corr.first_step)+' → '+fmt(corr.last_step)+'\n'+(corr.status||'');
if(isCorr&&corr.collected_observations!==null&&corr.collected_observations!==undefined)summary+='\nCollecte au moment de la capture : '+fmt(corr.collected_observations)+' observations.';
if(topology)summary+='\nUnités du graphe : '+fmt(neurons.total_units)+' · couches '+fmt(neurons.total_layers)+' · géométrie inconnue '+fmt(neurons.unknown_layers)+' · échantillonnage : '+(neurons.sampling?'oui':'non');
if(isLive)summary+='\n'+coeffs.length+' coefficients / '+fmt(live.total_parameters)+' paramètres · non finis exclus à la capture '+fmt(live.nonfinite)+' · couches ignorées '+fmt(live.skipped_layers)+' · partagées '+fmt(live.shared_layers)+'\nRéférence delta : '+(live.baseline_available?'pas '+(live.previous_step_decimal??fmt(live.previous_step))+' → '+(model.optimizer_step_decimal??fmt(model.optimizer_step)):'indisponible (première capture)')+' · échantillonnage : '+(live.sampled?'oui':'non');
summary+='\nInstantané interactif hors ligne. Le modèle peut continuer à évoluer indépendamment de ce fichier.';
text('summary',summary);
text('axes',topology?'X : profondeur du graphe · Y/Z : grille des unités par couche (coordonnées de présentation).':'ACP : '+(projection.explained_variance_percent||[]).map((v,i)=>'PC'+(i+1)+' '+fmt(v)+' %').join(' · ')+' · Les distances projetées ne constituent pas une mesure de compréhension.');
text('legend',isLive?'Cyan : poids positif · Orange : négatif · Gris : nul ou routage structurel · Épaisseur/luminosité : amplitude · Halo jaune : delta non nul · Losange : biais · Carré : coefficient local non mappé.':isCorr?'Cyan : corrélation positive · Orange : négative · Mode normal : seuil sur |r| et liens les plus forts. Tous les candidats valides : aucun filtre visuel supplémentaire. Le signe et la valeur sont calculés avant ACP.':topology?'Couleur : couche · Entrées : cyan clair · Bord jaune : canal · Gris encadré : opération de géométrie inconnue · Liens gris : routage agrégé.':'Couleur : famille de vecteurs (détails au survol) · Chemin vert : positions d’un même tenseur d’activation.');
for(const note of doc.notes||[]){const p=document.createElement('p');p.textContent=note;byId('notes').appendChild(p);}
const metadata={format:doc.format,version:doc.version,mode:doc.mode,model:doc.model,view:doc.view,projection:doc.projection,correlations:doc.correlations,neurons:doc.neurons,live_weights:{...live,coefficients:'Voir les '+coeffs.length+' coefficients dans les données JSON embarquées.'},sampling:doc.sampling,numeric_policy:doc.numeric_policy};
text('metadata',JSON.stringify(metadata,null,2));
const canvas=byId('canvas'), ctx=canvas.getContext('2d');
if(!ctx){text('hover','Le rendu Canvas 2D n’est pas disponible dans ce navigateur. Les données JSON sont conservées dans ce fichier.');return;}
let yaw=finite(initial.yaw)?initial.yaw:-.55, pitch=finite(initial.pitch)?initial.pitch:.25, zoom=finite(initial.zoom)&&initial.zoom>0?initial.zoom:1;
let threshold=finite(initial.correlation_threshold)?Math.max(0,Math.min(1,initial.correlation_threshold)):.65;
const thresholdInput=byId('threshold');thresholdInput.value=String(threshold);byId('threshold-control').hidden=!isCorr;
let showAllCandidates=initial.correlation_show_all_candidates===true;
const allCandidatesInput=byId('all-candidates');allCandidatesInput.checked=showAllCandidates;byId('all-candidates-control').hidden=!isCorr;thresholdInput.disabled=showAllCandidates;
const defaultLinkLimit=idx((doc.sampling||{}).correlation_default_link_limit)?doc.sampling.correlation_default_link_limit:512;
const captureMinAbs=finite((doc.sampling||{}).correlation_capture_min_abs)?doc.sampling.correlation_capture_min_abs:.30;
function visibleCorrelation(e,index){return finite(e.correlation)&&Math.abs(e.correlation)<=1+1e-6&&Math.abs(e.correlation)+1e-6>=captureMinAbs&&(showAllCandidates||(index<defaultLinkLimit&&Math.abs(e.correlation)+1e-6>=threshold));}
let w=1,h=1,screen=[],segments=[],mouse=null,drag=null,drawPending=false;
const groupColors=['#43cadc','#f2aa4e','#eb7061','#ae78e7','#e36bbe','#88a6f5','#67da85','#bec1d6','#f1bf55','#70b9e3','#ea88be','#7bd5a5','#b995ef'];
const layerColors=['#70b9e3','#c18fe8','#7bd5a5','#eb9f6f','#ea8ab7','#99aaf0','#dacb70','#66d3d0'];
let maxWeight=0,maxDelta=0;
for(const c of coeffs){if(finite(c.value))maxWeight=Math.max(maxWeight,Math.abs(c.value));if(c.has_delta&&finite(c.delta))maxDelta=Math.max(maxDelta,Math.abs(c.delta));}
const relative=(value,max)=>max>0?Math.sqrt(Math.min(1,Math.abs(value)/max)):0;
const weightColor=value=>{const alpha=(65+175*relative(value,maxWeight))/255;return value>0?'rgba(75,208,235,'+alpha+')':value<0?'rgba(246,150,76,'+alpha+')':'rgba(142,151,167,.47)';};
const coefficient=p=>idx(p.live_weight_coefficient)?coeffs[p.live_weight_coefficient]:undefined;
function project(pos){if(!Array.isArray(pos)||pos.length!==3||!pos.every(finite))return null;const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);const x=pos[0]*cy-pos[2]*sy,z=pos[0]*sy+pos[2]*cy,y=pos[1]*cp-z*sp,depth=pos[1]*sp+z*cp;const scale=Math.min(w,h)*.21*zoom*4.5/Math.max(1.4,6+depth);const result={x:w*.5+x*scale,y:h*.5-y*scale,depth};return finite(result.x)&&finite(result.y)?result:null;}
function inside(p){return p&&p.x>=0&&p.x<=w&&p.y>=0&&p.y<=h;}
function segmentDistance(m,a,b){const dx=b.x-a.x,dy=b.y-a.y,length=dx*dx+dy*dy,t=length>0?Math.max(0,Math.min(1,((m.x-a.x)*dx+(m.y-a.y)*dy)/length)):0;return Math.hypot(m.x-a.x-t*dx,m.y-a.y-t*dy);}
function clipped(a,b){let enter=0,leave=1;const dx=b.x-a.x,dy=b.y-a.y;for(const [p,q] of [[-dx,a.x],[dx,w-a.x],[-dy,a.y],[dy,h-a.y]]){if(Math.abs(p)<1e-12){if(q<0)return null;}else{const r=q/p;if(p<0){if(r>leave)return null;enter=Math.max(enter,r);}else{if(r<enter)return null;leave=Math.min(leave,r);}}}return [{x:a.x+enter*dx,y:a.y+enter*dy},{x:a.x+leave*dx,y:a.y+leave*dy}];}
function line(edge,kind,color,width=1,glow=0){if(!idx(edge.source)||!idx(edge.target))return;const a=screen[edge.source],b=screen[edge.target];if(!a||!b)return;const clippedLine=clipped(a,b);if(!clippedLine)return;const [start,end]=clippedLine;ctx.beginPath();ctx.moveTo(start.x,start.y);if(organicInput.checked){const dx=end.x-start.x,dy=end.y-start.y,len=Math.hypot(dx,dy)||1,bend=Math.min(24,len*.14);ctx.quadraticCurveTo((start.x+end.x)/2-dy/len*bend*2,(start.y+end.y)/2+dx/len*bend*2,end.x,end.y);}else ctx.lineTo(end.x,end.y);if(glow>0){ctx.strokeStyle='rgba(255,224,111,'+(.27+.35*glow)+')';ctx.lineWidth=width+3;ctx.stroke();}ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();if(organicInput.checked){const dx=end.x-start.x,dy=end.y-start.y,len=Math.hypot(dx,dy)||1,bend=Math.min(24,len*.14);let previous=start;for(let i=1;i<=12;i++){const t=i/12,curve=4*t*(1-t)*bend,current={x:start.x+dx*t-dy/len*curve,y:start.y+dy*t+dx/len*curve};segments.push({edge,kind,a:previous,b:current});previous=current;}}else segments.push({edge,kind,a:start,b:end});}
function axis(position,color,label){const a=project([0,0,0]),b=project(position);if(!a||!b)return;ctx.strokeStyle=color;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();ctx.fillStyle=color;ctx.font='12px system-ui,sans-serif';ctx.fillText(label,b.x+5,b.y-7);}
function marker(p,s){const c=isLive?coefficient(p):undefined;if(c&&!finite(c.value))return;const size=topology?(p.neuron_is_operation?9:6):p.group===6?9:7;let color=groupColors[p.group]||'#dcdcdc',outline=null;
if(c){color=weightColor(c.value);if(c.has_delta&&finite(c.delta)&&c.delta!==0)outline='#ffe06f';}else if(topology){color=p.neuron_is_operation?'#434b5b':p.layer<0?'#afe6ef':layerColors[p.layer%layerColors.length];outline=p.neuron_is_operation?'#b9bfcd':(/canal|canaux/.test(p.neuron_kind||'')?'#f1bf55':null);}
if(organicInput.checked&&topology&&finite(p.plasticity)&&p.plasticity>0){ctx.beginPath();ctx.arc(s.x,s.y,size*(1.1+.5*p.plasticity),0,Math.PI*2);ctx.fillStyle='rgba(255,210,95,'+(.1+.2*p.plasticity)+')';ctx.fill();}
ctx.fillStyle=color;ctx.lineWidth=outline?1.5:1;ctx.strokeStyle=outline||color;ctx.beginPath();if(organicInput.checked){ctx.arc(s.x,s.y,size*.65,0,Math.PI*2);}else if(c&&c.is_bias){ctx.moveTo(s.x,s.y-size*.75);ctx.lineTo(s.x+size*.75,s.y);ctx.lineTo(s.x,s.y+size*.75);ctx.lineTo(s.x-size*.75,s.y);ctx.closePath();}else ctx.rect(s.x-size*.5,s.y-size*.5,size,size);ctx.fill();if(outline)ctx.stroke();if(organicInput.checked){ctx.beginPath();ctx.arc(s.x,s.y,size*.2,0,Math.PI*2);ctx.fillStyle='rgba(235,245,255,.75)';ctx.fill();}}
function hover(){if(!mouse||(isCorr&&!corr.ready)){text('hover','Survolez un point ou un lien pour afficher son identité et ses valeurs.');return;}let selected=-1,distance=10;for(let i=0;i<screen.length;i++){const s=screen[i],c=isLive?coefficient(points[i]):undefined;if(!inside(s)||(c&&!finite(c.value)))continue;const d=Math.hypot(mouse.x-s.x,mouse.y-s.y);if(d<=distance){distance=d;selected=i;}}
if(selected>=0){const p=points[selected],c=coefficient(p);let value='POINT '+selected+' · '+p.label+'\n'+p.group_name+' · couche '+fmt(p.layer)+'\nCoordonnées : '+(p.position||[]).map(fmt).join(', ');if(topology)value+='\n'+p.neuron_kind+' · unité '+fmt(p.neuron_unit_index)+' / '+fmt(p.neuron_layer_units)+' · nœud '+fmt(p.neuron_node_index)+(p.neuron_is_operation?' · opération sans géométrie connue':'');else value+='\nNorme : '+fmt(p.vector_norm)+' · dimension initiale '+fmt(p.original_dimension)+' · offset '+fmt(p.feature_offset)+' · dimensions échantillonnées '+fmt(p.sampled_dimension)+'\nPosition séquence '+fmt(p.sequence_position);if(isCorr)value+='\nFeature '+fmt(p.feature_index)+' · pooling '+p.pooling+' · moyenne '+fmt(p.activation_mean)+' · écart type '+fmt(p.activation_stddev);if(isLive&&c)value+='\n'+c.label+'\nCoefficient '+c.key+' · offset paramètre '+(c.parameter_index_decimal??fmt(c.parameter_index))+'\nValeur '+fmt(c.value)+' · '+(c.has_delta?'delta '+fmt(c.delta):'delta indisponible')+' · '+(c.is_bias?'biais':c.mapped?'mappé sur une liaison vérifiée':'coefficient local non mappé')+'\nPas '+(live.previous_step_decimal??fmt(live.previous_step))+' → '+(model.optimizer_step_decimal??fmt(model.optimizer_step));text('hover',value);return;}
let nearest=null;distance=6;for(const segment of segments){const d=segmentDistance(mouse,segment.a,segment.b);if(d<=distance){nearest=segment;distance=d;}}if(nearest){const {edge,kind}=nearest;let value='LIEN '+kind+' #'+fmt(edge.index)+'\n'+points[edge.source].label+'\n→ '+points[edge.target].label;if(kind==='Pearson')value+='\nr = '+fmt(edge.correlation)+' (avant ACP) · '+fmt(corr.observations)+' observations';else if(kind==='poids'){const c=coeffs[edge.coefficient_index];if(c)value+='\n'+c.label+' · '+c.key+'\nValeur '+fmt(c.value)+' · '+(c.has_delta?'delta '+fmt(c.delta):'delta indisponible')+' · pas '+(live.previous_step_decimal??fmt(live.previous_step))+' → '+(model.optimizer_step_decimal??fmt(model.optimizer_step));}else if(kind==='routage')value+='\n'+(edge.aggregated?'Routage agrégé':'Routage')+' · '+(edge.weighted?'poids '+fmt(edge.weight):'sans poids scalaire individuel');else value+='\nPositions consécutives du même tenseur : '+edge.tensor_path;text('hover',value);}else text('hover','Survolez un point ou un lien pour afficher son identité et ses valeurs.');}
function draw(){drawPending=false;ctx.clearRect(0,0,w,h);segments=[];screen=points.map(p=>project(organicInput.checked&&topology?(p.organic_position||p.position):p.position));ctx.save();ctx.beginPath();ctx.rect(1,1,w-2,h-2);ctx.clip();const axes=projection.axes||(topology?['Profondeur','Unités Y','Unités Z']:['PC1','PC2','PC3']);axis([2.8,0,0],'#e66969',axes[0]);axis([0,2.8,0],'#69d787',axes[1]);axis([0,0,2.8],'#69a0eb',axes[2]);
if(heatspotsInput.checked&&isLive){const gw=64,gh=48,values=new Float32Array(gw*gh);for(let i=0;i<points.length;i++){const p=points[i],s=screen[i],v=p.plasticity;if(!s||!finite(s.x)||!finite(s.y)||!finite(v)||v<=0)continue;const px=(s.x-1)/(w-2),py=(s.y-1)/(h-2);if(px<-.15||px>1.15||py<-.15||py>1.15)continue;const cx=Math.round(px*(gw-1)),cy=Math.round(py*(gh-1));for(let y=Math.max(0,cy-6);y<=Math.min(gh-1,cy+6);y++)for(let x=Math.max(0,cx-6);x<=Math.min(gw-1,cx+6);x++)values[y*gw+x]=Math.min(1,values[y*gw+x]+Math.min(1,v)*Math.exp(-((x-cx)**2+(y-cy)**2)/8));}const layer=document.createElement('canvas');layer.width=gw;layer.height=gh;const lc=layer.getContext('2d'),image=lc.createImageData(gw,gh);for(let i=0;i<values.length;i++){const v=values[i];if(v<=.005)continue;image.data[i*4]=Math.floor(255*Math.min(1,v*3));image.data[i*4+1]=Math.floor(255*Math.max(0,Math.min(1,2-v*2)));image.data[i*4+2]=Math.floor(255*Math.max(0,1-v*3));image.data[i*4+3]=Math.floor(145*Math.sqrt(v));}lc.putImageData(image,0,0);ctx.imageSmoothingEnabled=true;ctx.drawImage(layer,1,1,w-2,h-2);}
if(!isCorr||corr.ready){if(isCorr){for(const [index,e] of (links.correlations||[]).entries())if(visibleCorrelation(e,index))line(e,'Pearson',e.correlation>=0?'rgba(75,208,222,'+((90+150*Math.min(1,Math.abs(e.correlation)))/255)+')':'rgba(244,161,85,'+((90+150*Math.min(1,Math.abs(e.correlation)))/255)+')');}else if(topology){for(const e of links.neurons||[]){if(e.weighted&&!finite(e.weight))continue;line(e,'routage',!e.weighted?'rgba(144,160,182,.39)':e.weight>=0?'rgba(75,208,222,.69)':'rgba(244,161,85,.69)');}if(isLive)for(const e of links.live_weights||[]){const c=idx(e.coefficient_index)?coeffs[e.coefficient_index]:undefined;if(!c||!finite(c.value))continue;line(e,'poids',weightColor(c.value),1+3*relative(c.value,maxWeight),c.has_delta&&finite(c.delta)&&c.delta!==0?relative(c.delta,maxDelta):0);}}else for(const e of links.sequences||[])line(e,'séquence','rgba(93,183,122,.69)');if(organicInput.checked&&topology){const stride=Math.max(1,Math.ceil(points.length/1024));ctx.lineWidth=1;for(let i=0;i<points.length;i+=stride){const p=points[i];if(p.neuron_is_operation||p.live_weight_coefficient!==null)continue;const center=p.organic_position||p.position,seed=p.organic_seed>>>0;for(let branch=0;branch<4;branch++){const phase=(seed%4096)*.0015339808+branch*2.39996323,reach=.045+.025*((seed>>>(branch%16))&3),tip=project([center[0]+Math.cos(phase)*reach,center[1]+Math.sin(phase)*reach,center[2]+Math.sin(phase*1.7)*reach]),a=screen[i];if(!a||!tip)continue;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(tip.x,tip.y);ctx.strokeStyle='rgba(128,190,220,.32)';ctx.stroke();}}}for(let i=0;i<points.length;i++)if(inside(screen[i]))marker(points[i],screen[i]);}
ctx.restore();if(topology)text('axes',organicInput.checked?'X : profondeur du graphe · Y/Z : volumes corticaux symboliques · ramifications décoratives · halos : variation des poids.':'X : profondeur du graphe · Y/Z : grille des unités par couche.');text('threshold-value',showAllCandidates?'désactivé':fmt(threshold));text('camera','Caméra : yaw '+yaw.toFixed(4)+' · pitch '+pitch.toFixed(4)+' · zoom '+zoom.toFixed(3)+' · '+segments.length+' liens affichés · capture '+fmt(initial.width)+' × '+fmt(initial.height));hover();}
function schedule(){if(!drawPending){drawPending=true;requestAnimationFrame(draw);}}
function resize(){const rect=canvas.getBoundingClientRect();w=Math.max(1,rect.width);const aspect=finite(initial.width)&&initial.width>0&&finite(initial.height)&&initial.height>0?initial.width/initial.height:1120/820;h=Math.max(1,w/aspect);canvas.style.height=h+'px';const ratio=window.devicePixelRatio||1;canvas.width=Math.ceil(w*ratio);canvas.height=Math.ceil(h*ratio);ctx.setTransform(ratio,0,0,ratio,0,0);schedule();}
function reset(){yaw=finite(initial.yaw)?initial.yaw:-.55;pitch=finite(initial.pitch)?initial.pitch:.25;zoom=finite(initial.zoom)&&initial.zoom>0?initial.zoom:1;threshold=finite(initial.correlation_threshold)?Math.max(0,Math.min(1,initial.correlation_threshold)):.65;thresholdInput.value=String(threshold);showAllCandidates=initial.correlation_show_all_candidates===true;allCandidatesInput.checked=showAllCandidates;thresholdInput.disabled=showAllCandidates;schedule();}
function changeZoom(factor){zoom=Math.min(50,Math.max(.05,zoom*factor));schedule();}
function position(event){const r=canvas.getBoundingClientRect();return {x:event.clientX-r.left,y:event.clientY-r.top};}
canvas.addEventListener('pointerdown',event=>{if(event.button!==0)return;canvas.focus();drag={id:event.pointerId,x:event.clientX,y:event.clientY};canvas.setPointerCapture(event.pointerId);});
canvas.addEventListener('pointermove',event=>{mouse=position(event);if(drag&&drag.id===event.pointerId){yaw+=(event.clientX-drag.x)*.01;pitch=Math.max(-1.5,Math.min(1.5,pitch+(event.clientY-drag.y)*.01));drag.x=event.clientX;drag.y=event.clientY;schedule();}else hover();});
const endDrag=event=>{if(drag&&drag.id===event.pointerId){drag=null;if(canvas.hasPointerCapture(event.pointerId))canvas.releasePointerCapture(event.pointerId);}};
canvas.addEventListener('pointerup',endDrag);canvas.addEventListener('pointercancel',endDrag);canvas.addEventListener('lostpointercapture',()=>{drag=null;});canvas.addEventListener('pointerleave',()=>{if(!drag){mouse=null;hover();}});
canvas.addEventListener('wheel',event=>{event.preventDefault();changeZoom(Math.exp(-Math.max(-250,Math.min(250,event.deltaY))*.0015));},{passive:false});
canvas.addEventListener('keydown',event=>{if(event.key==='ArrowLeft')yaw-=.08;else if(event.key==='ArrowRight')yaw+=.08;else if(event.key==='ArrowUp')pitch=Math.min(1.5,pitch+.08);else if(event.key==='ArrowDown')pitch=Math.max(-1.5,pitch-.08);else if(event.key==='+'||event.key==='=')changeZoom(1.2);else if(event.key==='-')changeZoom(1/1.2);else if(event.key==='0')reset();else return;event.preventDefault();schedule();});
byId('reset').addEventListener('click',reset);byId('zoom-in').addEventListener('click',()=>changeZoom(1.2));byId('zoom-out').addEventListener('click',()=>changeZoom(1/1.2));thresholdInput.addEventListener('input',()=>{threshold=Number(thresholdInput.value);schedule();});allCandidatesInput.addEventListener('change',()=>{showAllCandidates=allCandidatesInput.checked;thresholdInput.disabled=showAllCandidates;schedule();});window.addEventListener('resize',resize);resize();
})();
</script></body></html>
)MIMIR";
    return html;
}

} // namespace mimir::viz
