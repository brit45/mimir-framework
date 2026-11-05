#pragma once
#include "Planning/Planner.hpp"
#include "Planning/FusionCalibration.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "include/json.hpp"
#include <unordered_map>

namespace Mimir::Viz {
// Called on the model thread only. Tensor names are resolved in execution order,
// so in-place writes connect to the previous producer rather than to themselves.
inline nlohmann::json modelGraphSnapshot(const std::vector<Layer>& layers,
    const Planning::ExecutionPlan* plan, const std::string& mode,
    bool training, bool fusion, bool reuse) {
    using Json = nlohmann::json;
    Json graph={{"mode",mode},{"planned",plan!=nullptr},{"training",training},
        {"fusion",fusion},{"buffer_reuse",reuse},{"nodes",Json::array()},
        {"edges",Json::array()},{"external_inputs",Json::array()}};
    if(plan) {
        graph["plan"]=Json::parse(Planning::dump_execution_plan_json(*plan));
        graph["fusion_groups"]=graph["plan"]["fusion_groups"];
    } else graph["fusion_groups"]=Json::array();
    namespace Policy=Planning::ExecutionPolicy;
    graph["execution_policy"]={{"selection","measured_latency"},{"unmeasured_cpu_workers",1},
        {"graph_requires_backward_snapshots",training},{"calibration_budget_bytes",Policy::calibrationBudget()},
        {"detected_cpu_cache_bytes",Policy::cacheBytes()},
        {"portable_serial_elements_limit",Policy::Limits::portable_serial_elements},
        {"portable_serial_work_limit",Policy::Limits::portable_serial_work},
        {"portable_parallel_elements_limit",Policy::Limits::portable_parallel_elements},
        {"portable_parallel_work_limit",Policy::Limits::portable_parallel_work},
        {"explicit_runtime_minima",Json::array()}};
    for(const auto* runtime:RuntimeRouter::instance().configuredRuntimes()) {
        if(!runtime)continue;
        const auto& config=runtime->config();
        graph["execution_policy"]["explicit_runtime_minima"].push_back({{"runtime",runtime->name()},
            {"initialized",runtime->isInitialized()},{"disabled",config.disabled},{"device_index",config.device_index},
            {"linear",config.linear_min_ops},{"convolution",config.conv_min_ops},
            {"normalization",config.norm_min_elements},{"attention",config.attention_min_ops}});
    }
    std::unordered_map<std::string,size_t> producers;
    std::vector<size_t> depths(layers.size(),0), lanes(layers.size()+1,0);
    for(size_t i=0;i<layers.size();++i) {
        const auto& layer=layers[i];
        for(const auto& input:Planning::planner_inputs_for(layer)) {
            auto found=producers.find(input);
            if(found!=producers.end()) {
                graph["edges"].push_back({{"from",found->second},{"to",i},{"tensor",input}});
                depths[i]=std::max(depths[i],depths[found->second]+1);
            } else graph["external_inputs"].push_back({{"to",i},{"tensor",input}});
        }
        Json node={{"index",i},{"name",layer.name},{"type",layer.type},
            {"inputs",Planning::planner_inputs_for(layer)},
            {"output",Planning::planner_output_name_for(layer)},
            {"parameters",layer.getWeightsSize()},{"depth",depths[i]},
            {"lane",lanes[depths[i]]++},{"runtime","non planifie"},
            {"capability","inconnue"},{"graph_fused",false},{"kernel_fused",false},{"skip",false}};
        if(plan && i<plan->layers.size()) {
            const auto& p=plan->layers[i];
            // Legacy plans do not run the placement pass; don't label their
            // default CPU fields as actual runtime decisions.
            if(mode!="legacy") {
                node["runtime"]=Planning::runtime_kind_name(p.runtime);
                node["capability"]=Planning::capability_level_name(p.capability);
            } else node["runtime"]="dispatcher (legacy)";
            node["graph_fused"]=p.graph_fused;node["kernel_fused"]=p.kernel_fused;
            node["skip"]=p.skip;node["scratch_bytes"]=p.scratch_bytes;
            if(p.fused_into)node["fused_into"]=*p.fused_into;
            const auto tensor=plan->tensors.find(p.output);
            if(tensor!=plan->tensors.end()) {
                node["shape"]=tensor->second.shape;node["bytes"]=tensor->second.bytes;
                node["generation"]=tensor->second.generation;
            }
        }
        for(bool backward:{false,true}) {
            const auto& profiles=backward?Planning::Calibration::backward_profiles:Planning::Calibration::forward_profiles;
            const auto choices=profiles.find(&layer);
            if(choices!=profiles.end()) {
                Json candidates=Json::array();
                for(const auto& choice:choices->second)candidates.push_back({
                    {"runtime",choice.runtime->name()},{"seconds",choice.seconds},{"cpu_workers",choice.cpu_workers},
                    {"cpu_algorithm",choice.cpu_algorithm==Policy::CpuAlgorithm::Reference?"reference":"pointwise_convolution"}});
                node[backward?"backward_execution_choices":"forward_execution_choices"]=std::move(candidates);
            }
            if(const auto* decision=Planning::FusionCalibration::profile(&layer,backward)) {
                Json alternatives=Json::array();
                for(const auto& candidate:decision->alternatives)
                    alternatives.push_back({{"runtime",candidate.runtime},{"seconds",candidate.seconds},{"bit_exact",candidate.exact},{"cpu_workers",candidate.cpu_workers}});
                node[backward?"fusion_backward_decision":"fusion_forward_decision"]={
                    {"measured",decision->measured},{"fused",decision->fused},
                    {"runtime",decision->selected_runtime},{"reason",decision->reason},{"cpu_workers",decision->cpu_workers},
                    {"unfused_seconds",decision->unfused_seconds},{"alternatives",std::move(alternatives)}};
            }
        }
        graph["nodes"].push_back(std::move(node));
        producers[Planning::planner_output_name_for(layer)]=i;
    }
    return graph;
}
}
