#include "viz/ModelGraph.hpp"
#include "LiveModelConfig.hpp"
#include <iostream>
#include <stdexcept>
#define CHECK(c) do { if(!(c)) throw std::runtime_error(#c); } while(0)
int main() {
    std::vector<Layer> layers;
    layers.emplace_back("first","Linear",6);
    layers.back().inputs={"x"};layers.back().output="a";
    layers.emplace_back("rewrite","ReLU",0);
    layers.back().inputs={"a"};layers.back().output="a";
    layers.emplace_back("branch","Add",0);
    layers.back().inputs={"a","x"};layers.back().output="b";
    auto graph=Mimir::Viz::modelGraphSnapshot(layers,nullptr,"desactive",false,false,false);
    CHECK(graph["nodes"].size()==3);CHECK(graph["edges"].size()==2);
    CHECK(graph["edges"][0]["from"]==0 && graph["edges"][0]["to"]==1);
    CHECK(graph["edges"][1]["from"]==1 && graph["edges"][1]["to"]==2);
    CHECK(graph["nodes"][2]["depth"]==2);CHECK(!graph["planned"].get<bool>());
    auto plan=Mimir::Planning::build_execution_plan_global(layers,false,Mimir::Planning::PlannerMode::Static,false,false);
    graph=Mimir::Viz::modelGraphSnapshot(layers,&plan,"static",false,false,false);
    CHECK(graph["planned"].get<bool>());CHECK(graph.contains("plan"));
    CHECK(graph["nodes"][0]["runtime"]=="CPU");
    auto legacy=Mimir::Viz::modelGraphSnapshot(layers,&plan,"legacy",false,false,false);
    CHECK(legacy["nodes"][0]["runtime"]=="dispatcher (legacy)");
    layers.emplace_back("parallel","Identity",0);
    layers.back().inputs={"x"};layers.back().output="parallel";
    layers.emplace_back("join","Add",0);
    layers.back().inputs={"b","parallel"};layers.back().output="result";
    auto branch=Mimir::Viz::modelGraphSnapshot(layers,nullptr,"desactive",false,false,false);
    CHECK(branch["nodes"][3]["depth"]==0 && branch["nodes"][3]["lane"]==1);
    CHECK(branch["nodes"][4]["depth"]==3 && branch["edges"].size()==4);
    auto empty=Mimir::Viz::modelGraphSnapshot({},nullptr,"en attente",false,false,false);
    CHECK(empty["nodes"].empty() && empty["edges"].empty());
    // An existing fusion can be consumed by another fusion, recursively.
    std::vector<Layer> nested;
    nested.emplace_back("conv", "Conv2d", 0);
    nested[0].activation=ActivationType::RELU;
    nested[0].inputs={"input"}; nested[0].output="conv_out";
    for(const char* type:{"Tanh","Identity","Sigmoid"}) {
        const auto previous=nested.back().output;
        nested.emplace_back(type,type,0);
        nested.back().inputs={previous};nested.back().output=std::string(type)+"_out";
    }
    for(bool training:{false,true}) {
        auto fused=Mimir::Planning::build_execution_plan_global(nested,training);
        auto snapshot=Mimir::Viz::modelGraphSnapshot(nested,&fused,"static",training,true,false);
        const auto& groups=snapshot["fusion_groups"];
        CHECK(groups.size()==4);
        CHECK(groups[0]["layers"]==nlohmann::json::array({0}));
        CHECK(groups[0]["parent"]==1);
        CHECK(groups[3]["parent"].is_null());
        CHECK(groups[3]["children"]==nlohmann::json::array({2}));
        CHECK(snapshot["nodes"][3]["fused_into"]==0);
        CHECK(snapshot["nodes"][0]["graph_fused"]==true);
        CHECK(snapshot["nodes"][0]["kernel_fused"]==false);
        std::vector<size_t> visits(nested.size());
        for(size_t id=0;id<groups.size();++id) {
            for(const auto& member:groups[id]["layers"]) ++visits[member.get<size_t>()];
            for(const auto& child:groups[id]["children"]) CHECK(child.get<size_t>()<id);
        }
        CHECK(visits==std::vector<size_t>(nested.size(),1));
        auto disabled=Mimir::Planning::build_execution_plan_global(nested,training,
            Mimir::Planning::PlannerMode::Static,false);
        CHECK(disabled.fusion_groups.empty());
    }
    // A live side consumer prevents fusing across the shared tensor edge.
    auto side=nested;
    side.emplace_back("side","Identity",0);
    side.back().inputs={"conv_out"};side.back().output="side_out";
    auto side_plan=Mimir::Planning::build_execution_plan_global(side,true);
    CHECK(side_plan.fuse_chain_next[0]==-1);

    // Deep compositions store each layer once and avoid recursive stack growth.
    std::vector<Layer> deep;
    for(size_t i=0;i<512;++i) {
        deep.emplace_back(std::to_string(i),"Identity",0);
        deep.back().inputs={i?std::to_string(i-1):"input"};
        deep.back().output=std::to_string(i);
    }
    auto deep_plan=Mimir::Planning::build_execution_plan_global(deep,true);
    CHECK(deep_plan.fusion_groups.size()==511);
    size_t direct_members=0;
    for(const auto& group:deep_plan.fusion_groups) direct_members+=group.layers.size();
    CHECK(direct_members==deep.size());
    LiveModelConfig config;config.publishModelGraph(graph);auto old=config.modelGraph();
    config.publishModelGraph(legacy);CHECK((*old)["mode"]=="static");
    CHECK((*config.modelGraph())["mode"]=="legacy");
    config.requestModelGraph(true);CHECK(config.modelGraphRequested());
    std::cout<<"MODEL_GRAPH_SNAPSHOT_OK\n";
}
