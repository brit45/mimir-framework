#include "Model.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "viz/LatentSpaceProjection.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
using mimir::viz::LatentSpaceRawSnapshot;
void check(bool ok,const std::string& message) {if(!ok)throw std::runtime_error(message);}
bool same(const std::vector<float>& a,const std::vector<float>& b) {
    return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0);
}
std::vector<std::vector<float>> weights(const Model& model) {
    std::vector<std::vector<float>> result;
    for(const auto& layer:model.getLayers()) {
        const auto* p=layer.getWeights();const auto count=layer.getWeightsSize();
        result.push_back(p && count?std::vector<float>(p,p+count):std::vector<float>{});
        result.push_back(layer.grad_weights);result.push_back(layer.grad_bias);
    }
    return result;
}
void bounds(const LatentSpaceRawSnapshot& raw) {
    check(raw.vectors.size()<=512,"capture exceeded 512 points");
    size_t values=0;
    for(const auto& sample:raw.vectors) {
        check(!sample.label.empty() && !sample.values.empty(),"unlabelled/empty vector");
        check(sample.values.size()<=2048,"capture exceeded feature window");
        check(sample.original_dimension>=sample.values.size(),"invalid original feature dimension");
        check(sample.feature_offset<=sample.original_dimension-sample.values.size(),"invalid feature window offset");
        check(std::isfinite(sample.norm) && sample.norm>=0,"invalid vector norm");
        double squared=0;
        for(float value:sample.values){check(std::isfinite(value),"non-finite vector captured");squared+=double(value)*value;}
        check(std::abs(double(sample.norm)-std::sqrt(squared))<=2e-5*(1+std::sqrt(squared)),"norm does not describe captured window");
        values+=sample.values.size();
    }
    check(values<=256*1024,"capture exceeded aggregate float budget");
}
void verify(Model& model,const std::string& architecture) {
    const auto before=weights(model);
    const auto configuration=model.modelConfig;
    const auto tensors=model.tensor_store;
    const auto integer_tensors=model.tensor_store_int;
    const auto raw=model.captureLatentSpaceSnapshot(17,8,4,3);
    const auto repeated=model.captureLatentSpaceSnapshot(17,8,4,3);
    bounds(raw);
    check(raw.optimizer_step==17,"capture lost optimizer step");
    check(!raw.model_name.empty(),"capture lost model name");
    check(raw.architecture==architecture,"capture lost architecture: "+architecture);
    check(raw.vectors.size()>=3,"insufficient generic vectors: "+architecture);
    check(repeated.vectors.size()==raw.vectors.size(),"non-deterministic sample count");
    for(size_t i=0;i<raw.vectors.size();++i) {
        const auto& a=raw.vectors[i];const auto& b=repeated.vectors[i];
        check(a.label==b.label && a.layer==b.layer && a.sequence_position==b.sequence_position &&
              a.original_dimension==b.original_dimension && a.feature_offset==b.feature_offset && same(a.values,b.values),
              "non-deterministic sample selection");
    }
    const auto projected=mimir::viz::projectLatentSpace3D(raw);
    check(projected.points.size()>=3 && projected.points.size()<=raw.vectors.size(),"projection failed: "+architecture);
    check(projected.optimizer_step==17 && projected.model_name==raw.model_name && projected.architecture==architecture,
          "projection lost source model metadata");
    for(const auto& point:projected.points)for(float coordinate:point.position)
        check(std::isfinite(coordinate),"non-finite PCA coordinate");
    for(float variance:projected.explained_variance_percent)
        check(std::isfinite(variance) && variance>=0 && variance<=100.001f,"invalid explained variance");
    const auto after=weights(model);
    check(before.size()==after.size(),"capture changed parameter topology");
    for(size_t i=0;i<before.size();++i)check(same(before[i],after[i]),"capture/PCA mutated parameters or gradients");
    check(configuration==model.modelConfig && integer_tensors==model.tensor_store_int,"capture mutated model config/IDs");
    check(tensors.size()==model.tensor_store.size(),"capture mutated tensor store");
    for(const auto& entry:tensors)check(same(entry.second,model.tensor_store.at(entry.first)),"capture mutated activation");
    std::cout<<"GENERIC_LATENT_CAPTURE_OK "<<architecture<<" vectors="<<raw.vectors.size()<<'\n';
}
void registryCases() {
    struct Case {const char* name;json config;size_t input_size;bool integer;};
    const std::vector<Case> cases{
        {"basic_mlp",{{"input_dim",4},{"hidden_dim",8},{"output_dim",3},{"hidden_layers",1},{"dropout",0.0}},4,false},
        {"vae",{{"image_w",4},{"image_h",4},{"image_c",1},{"latent_dim",3},{"hidden_dim",8},{"stochastic_latent",false}},16,false},
        {"vae_conv",{{"image_w",4},{"image_h",4},{"image_c",1},{"latent_w",2},{"latent_h",2},{"latent_c",2},
             {"base_channels",2},{"resnet",false},{"attention",false},{"enc_norm","none"},{"dec_norm","none"},
             {"decoder_upsample","nearest_conv"},{"stochastic_latent",false}},16,false},
        {"transformer",{{"seq_len",3},{"d_model",4},{"vocab_size",16},{"num_layers",1},{"num_heads",2},{"mlp_hidden",6},{"output_dim",3}},3,true},
        {"causal_lm",{{"seq_len",4},{"d_model",8},{"vocab_size",16},{"num_layers",1},{"num_heads",2},{"num_kv_heads",1},
             {"mlp_hidden",12},{"latent_dim",4}},4,true}
    };
    for(const auto& fixture:cases) {
        auto model=ModelArchitectures::create(fixture.name,fixture.config);
        model->allocateParams();model->initializeWeights("xavier",812);
        if(fixture.integer) {
            std::vector<int> ids(fixture.input_size);for(size_t i=0;i<ids.size();++i)ids[i]=int(i)+2;
            std::unordered_map<std::string,std::vector<float>> conditioning;
            if(std::string(fixture.name)=="transformer") {
                // This native encoder graph explicitly consumes its MAG/MOD
                // conditioning vectors in addition to integer token ids.
                conditioning["mag"]=model->getEncoder().getMagEmbedding();
                conditioning["mod"]=model->getEncoder().getModEmbedding();
                check(conditioning["mag"].size()==4 && conditioning["mod"].size()==4,
                      "invalid native Transformer conditioning fixture");
            }
            model->forwardPassNamedView(conditioning,{{"__input__",ids}},true);
        } else {
            std::vector<float> input(fixture.input_size);
            for(size_t i=0;i<input.size();++i)input[i]=.2f*std::sin(float(i)*.31f)+.1f;
            model->forwardPass(input,true);
        }
        verify(*model,fixture.name);
    }
}
void convolutionCase(int rank) {
    Model model;model.modelConfig["type"]=rank==3?"custom_conv3d":"custom_conv4d";
    model.setModelName(rank==3?"Custom Conv3d":"Custom Conv4d");
    model.push("spatial",rank==3?"Conv3d":"Conv4d",0);
    auto& layer=model.getMutableLayers().back();
    layer.in_channels=2;layer.out_channels=3;layer.kernel_size=1;layer.use_bias=true;
    layer.input_spatial_shape.assign(size_t(rank),2);
    layer.shape={1,2};layer.shape.insert(layer.shape.end(),layer.input_spatial_shape.begin(),layer.input_spatial_shape.end());
    layer.inputs={"__input__"};layer.output="x";
    model.allocateParams();model.initializeWeights("xavier",91);
    std::vector<float> input(size_t(2)<<rank);
    for(size_t i=0;i<input.size();++i)input[i]=std::sin(float(i)*.31f)*.2f;
    model.forwardPass(input,true);
    verify(model,rank==3?"custom_conv3d":"custom_conv4d");
    const auto raw=model.captureLatentSpaceSnapshot(1,8,4,3);
    bool real_filter=false;
    for(const auto& sample:raw.vectors)if(sample.group==mimir::viz::LatentVectorGroup::ConvolutionWeight) {
        check(sample.values.size()==2,"ND filter vector has an invented feature width");
        bool matches=false;
        for(size_t row=0;row<3;++row)
            matches=matches || std::equal(sample.values.begin(),sample.values.end(),layer.getWeights()+row*2);
        check(matches,"ND vector does not contain a real learned filter");real_filter=true;
    }
    check(real_filter,"ND filters absent from generic capture");
}
void mixedAndSharedCase() {
    Model model;model.modelConfig["type"]="custom_mixed";
    model.push("owner","Linear",12);model.push("alias","Linear",12);
    model.push("frozen","Constant",4);model.push("empty","Identity",0);model.push("nonfinite","Linear",12);
    auto& ls=model.getMutableLayers();
    for(size_t index:{size_t(0),size_t(1),size_t(4)}){ls[index].in_features=4;ls[index].out_features=3;ls[index].use_bias=false;}
    ls[1].shared_weights_from="owner";ls[2].shape={2,2};ls[2].trainable_parameter=false;
    model.allocateParams();
    for(auto& layer:ls)for(size_t i=0;i<layer.getWeightsSize();++i)layer.getWeights()[i]=.1f+std::sin(float(i)*.3f);
    // The declared alias must already be deduplicated before the first forward
    // resolves its physical block pointer.
    const auto declared_alias=model.captureLatentSpaceSnapshot(4,8,4,3);bounds(declared_alias);
    bool captured_owner=false;
    for(const auto& sample:declared_alias.vectors) {
        check(sample.layer!=1,"declared shared weights sampled before alias resolution");
        captured_owner=captured_owner || sample.layer==0;
    }
    check(captured_owner,"declared shared owner missing before alias resolution");
    // Also cover aliases that already share the actual tensor storage.
    ls[1].weight_block=ls[0].weight_block;
    std::fill_n(ls[4].getWeights(),ls[4].getWeightsSize(),std::numeric_limits<float>::quiet_NaN());
    ls[0].getWeights()[0]=std::numeric_limits<float>::infinity();
    const auto before=weights(model);
    const auto raw=model.captureLatentSpaceSnapshot(5,8,4,3);bounds(raw);
    bool frozen=false;
    for(const auto& sample:raw.vectors) {
        check(sample.layer!=1,"shared weights sampled twice");
        check(sample.layer!=3 && sample.layer!=4,"empty/nonfinite layer sampled");
        frozen=frozen || sample.layer==2;
    }
    check(frozen,"fixed nontrainable parameters were omitted");
    const auto after=weights(model);
    for(size_t i=0;i<before.size();++i)check(same(before[i],after[i]),"capture mutated shared/nonfinite weights");
}
void minimalModelCases() {
    Model model;model.push("unallocated","Linear",12);
    auto& layer=model.getMutableLayers().back();layer.in_features=4;layer.out_features=3;layer.use_bias=false;
    check(model.captureLatentSpaceSnapshot(0).vectors.empty(),"capture allocated or invented uninitialized weights");
    check(!layer.weight_block,"capture allocated model parameters");
    Model zero;zero.modelConfig["type"]="zero_constant";zero.push("fixed","Constant",1);
    zero.getMutableLayers().back().shape={1};zero.allocateParams();
    zero.getMutableLayers().back().getWeights()[0]=0;
    const auto raw=zero.captureLatentSpaceSnapshot(0);bounds(raw);
    check(raw.vectors.size()==1 && raw.vectors[0].norm==0,"single fixed zero parameter was omitted");
    const auto projected=mimir::viz::projectLatentSpace3D(raw);
    check(projected.points.size()==1,"single zero model cannot be displayed");
}
void budgetCases() {
    for(size_t width:{size_t(4),size_t(2048),size_t(3000)}) {
        Model model;model.modelConfig["type"]="custom_budget";
        const size_t rows=width==3000?4:800;
        model.push("wide","Linear",rows*width);
        auto& layer=model.getMutableLayers().back();layer.in_features=int(width);layer.out_features=int(rows);layer.use_bias=false;
        model.allocateParams();
        for(size_t i=0;i<layer.getWeightsSize();++i)layer.getWeights()[i]=.1f+std::sin(float(i%997)*.071f);
        const auto raw=model.captureLatentSpaceSnapshot(0,std::numeric_limits<size_t>::max(),
            std::numeric_limits<size_t>::max(),std::numeric_limits<size_t>::max());
        bounds(raw);check(!raw.vectors.empty(),"budget discarded all vectors");
        if(width==3000) {
            bool window=false;
            for(const auto& sample:raw.vectors)window=window || (sample.original_dimension==3000 && sample.values.size()<=2048);
            check(window,"long vectors lost their original dimension");
        }
    }
}
}
int main() {
    try {
        Model::setFrameworkLogsSuppressed(true);
        registryCases();convolutionCase(3);convolutionCase(4);mixedAndSharedCase();minimalModelCases();budgetCases();
        std::cout<<"GENERIC_LATENT_SPACE_MODELS_OK registry, spatial, shared, nonfinite, budgets, no mutation\n";
        return 0;
    } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
