#include "runtimes/cpu/RuntimeLayerDispatch.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
void equal(const std::vector<float>& a,const std::vector<float>& b) {
    if(a.size()!=b.size() || (!a.empty() && std::memcmp(a.data(),b.data(),a.size()*sizeof(float))))
        throw std::runtime_error("attention layer backward changed its ordered values");
}
void fill(std::vector<float>& values,float scale) {
    for(size_t i=0;i<values.size();++i)values[i]=float(int((i*13)%37)-18)*scale;
}
struct Acceleration {
    bool saved=RuntimeLayerOps::hardwareAccelerationEnabled();
    ~Acceleration(){RuntimeLayerOps::setHardwareAcceleration(saved);}
};
void run(const char* type,int rows,int dim,int heads,bool bias,bool causal,int workers) {
    Acceleration restore;
    Layer actual("attention",type,0);
    actual.embed_dim=actual.in_features=dim;actual.seq_len=rows;actual.num_heads=heads;
    actual.causal=causal;actual.use_bias=bias;
    const bool self=actual.type_enum==LayerType::SelfAttention;
    actual.weights.resize(4*size_t(dim)*dim+(bias?(self?4:1)*dim:0));
    fill(actual.weights,.007f);
    actual.grad_weights.assign(actual.weights.size(),.125f);
    actual.grad_bias.assign(dim,.25f);
    Layer reference=actual;
    std::vector<float> x(size_t(rows)*dim),kv(size_t(rows+3)*dim),dy(x.size());
    fill(x,.031f);fill(kv,.023f);fill(dy,.011f);
    std::vector<const std::vector<float>*> inputs{&x};if(!self)inputs.push_back(&kv);
    std::vector<std::vector<float>> want,got;
    Mimir::Planning::ExecutionPolicy::CpuScope scope(workers);
    for(int repeat=0;repeat<2;++repeat) {
        RuntimeLayerOps::setHardwareAcceleration(false);
        if(!RuntimeLayerDispatch::cpu_backward_layer(inputs,{&dy},want,reference,true))
            throw std::runtime_error("attention scalar layer refused valid geometry");
        RuntimeLayerOps::setHardwareAcceleration(true);
        if(!RuntimeLayerDispatch::cpu_backward_layer(inputs,{&dy},got,actual,true))
            throw std::runtime_error("attention tiled layer refused valid geometry");
        if(want.size()!=got.size())throw std::runtime_error("attention gradient count");
        for(size_t i=0;i<want.size();++i)equal(got[i],want[i]);
        equal(actual.grad_weights,reference.grad_weights);equal(actual.grad_bias,reference.grad_bias);
        // Next call sees the same addresses with new weights and source values.
        actual.weights[0]+=.17f;reference.weights[0]+=.17f;x.back()-=.13f;kv.front()+=.11f;
    }
}
}
int main() {
    try {
        for(const char* type: {"SelfAttention","CrossAttention"})for(bool causal:{false,true})
            for(bool bias:{false,true})for(int workers:{1,3})
                for(int rows:{1,7,65,129})run(type,rows,15,3,bias,causal,workers);
        std::cout<<"ATTENTION_LAYER_ORDERED_PARITY_OK scalar/tiled, causal, bias, threads, accumulation, mutations\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
