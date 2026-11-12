#include "viz/OrganicNetwork.hpp"
#include "viz/Heatspots.hpp"
#include "viz/LatentSpaceExport.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <stdexcept>
using namespace mimir::viz;
void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(int argc,char** argv){
    const auto empty=buildHeatspots({});
    for(float v:empty.values)check(v==0.f,"heat without measurements");
    const auto heat=buildHeatspots({{.5f,.5f,1.f},{.5f,.5f,std::numeric_limits<float>::infinity()},{2.f,2.f,1.f}});
    check(heat.values[24*64+32]==1.f && heat.values[0]==0.f,"heatspot peak/locality");
    check(heatspotColor(0.f)[3]==0 && heatspotColor(1.f)[0]==255 && heatspotColor(1.f)[1]==0,"heat color scale");
    LatentSpaceProjection p;p.live_weight_mode=true;
    for(size_t i=0;i<128;++i){LatentSpacePoint n;n.label="layer/unit/"+std::to_string(i);n.layer=int(i/32);n.neuron_unit_index=i;n.neuron_node_index=i/32;n.neuron_layer_units=32;n.position={{float(i/32),0.f,0.f}};p.points.push_back(n);}
    LiveWeightCoefficient c;c.value=.5f;c.has_delta=true;c.delta=.125;p.live_weight_coefficients.push_back(c);
    p.live_weight_links.push_back({0,32,0});p.live_weight_links.push_back({9999,32,0});
    const auto before=p.points;const auto scene=buildOrganicNetworkScene(p);
    check(scene.positions.size()==128,"missing units");
    check(scene.plasticity[0]==1.f && scene.plasticity[32]==1.f && scene.plasticity[1]==0.f,"delta mapping");
    for(size_t i=0;i<p.points.size();++i){
        check(scene.positions[i][0]==before[i].position[0],"graph depth changed");
        check(p.points[i].position==before[i].position,"snapshot mutated");
        check(std::isfinite(scene.positions[i][1])&&std::isfinite(scene.positions[i][2]),"nonfinite geometry");
        check((i%2 ? scene.positions[i][2]>0.f : scene.positions[i][2]<0.f),"lobes missing");
    }
    std::reverse(p.points.begin(),p.points.end());
    const auto reversed=buildOrganicNetworkScene(p);
    for(size_t i=0;i<128;++i)check(reversed.positions[127-i]==scene.positions[i],"unstable layout after reordering");
    p.points=before;p.live_weight_coefficients[0].delta=0.;
    auto quiet=buildOrganicNetworkScene(p);check(quiet.positions==scene.positions,"weights moved topology");
    for(float value:quiet.plasticity)check(value==0.f,"invented activity without delta");
    p.live_weight_coefficients[0].delta=std::numeric_limits<double>::infinity();
    quiet=buildOrganicNetworkScene(p);for(float value:quiet.plasticity)check(value==0.f,"nonfinite activity");
    p.live_weight_coefficients[0].delta=.125;p.live_weight_coefficients[0].has_delta=false;
    quiet=buildOrganicNetworkScene(p);for(float value:quiet.plasticity)check(value==0.f,"activity on first capture");
    p.live_weight_coefficients[0].has_delta=true;
    LatentSpaceExportView view;view.organic=true;view.heatspots=true;
    const auto document=latentSpaceExportDocument(p,view);
    check(document["points"][0]["organic_position"][0]==scene.positions[0][0],"export geometry mismatch");
    check(document["points"][0]["plasticity"]==1.f,"export delta mismatch");
    p.live_weight_mode=false;const auto structural=buildOrganicNetworkScene(p);
    for(size_t i=0;i<128;++i)check(structural.positions[i]==p.points[i].position,"PCA coordinates deformed");
    if(argc>1) {
        p.live_weight_mode=true;p.optimizer_step=32;p.model_name="Démonstration du rendu organique";
        p.architecture="fixture graphique dense, valeurs synthétiques";p.neuron_total_units=128;p.neuron_total_layers=4;
        p.live_weight_baseline_available=true;p.live_weight_previous_step=24;
        for(auto& point:p.points)point.position[0]-=1.5f;
        p.live_weight_coefficients.clear();p.live_weight_links.clear();
        for(size_t i=0;i<96;++i)for(size_t branch=0;branch<3;++branch){
            LiveWeightCoefficient coefficient;coefficient.key="fixture/"+std::to_string(i)+"/"+std::to_string(branch);
            coefficient.label=coefficient.key;coefficient.value=std::sin(float(i*3+branch));
            coefficient.has_delta=true;coefficient.delta=i%5==0?.01*std::cos(float(i)):0.;coefficient.mapped=true;
            const size_t target=(i/32+1)*32+(i+branch*7)%32;
            p.live_weight_links.push_back({i,target,p.live_weight_coefficients.size()});p.live_weight_coefficients.push_back(coefficient);
        }
        p.live_weight_total_parameters=p.live_weight_coefficients.size();
        const auto root=std::filesystem::path(argv[1]);std::filesystem::create_directories(root);
        const auto exported=latentSpaceExportDocument(p,view);
        std::ofstream(root/"reseau-organique.json")<<exported.dump(2);
        std::ofstream(root/"reseau-organique.html")<<latentSpaceExportHtml(exported);
    }
    std::cout<<"PASS organic graph: stable geometry, exact identities, delta activity, immutable snapshots and export\n";
}
