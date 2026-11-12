// Global new/delete intentionally use matched malloc/free for instrumentation.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif

// CPU-kernel allocation/parity probe. Build against a source snapshot to compare
// the binary value stream before/after; timings are not whole-model latency.
#include "runtimes/cpu/RuntimeLayerDispatch.hpp"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <new>

namespace {
thread_local bool counting=false;
thread_local size_t allocations=0,allocated_bytes=0;
}
void* operator new(size_t size) {
    if(void* p=std::malloc(size?size:1)) {if(counting){++allocations;allocated_bytes+=size;}return p;}
    throw std::bad_alloc();
}
void* operator new[](size_t n) {return ::operator new(n);}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete[](void* p) noexcept {std::free(p);}
void operator delete(void* p,size_t) noexcept {std::free(p);}
void operator delete[](void* p,size_t) noexcept {std::free(p);}

namespace {
void fill(std::vector<float>& values,float scale=.1f) {
    for(size_t i=0;i<values.size();++i)values[i]=float(int(i%19)-9)*scale;
}
void write(std::ofstream& out,const std::vector<float>& values) {
    const uint64_t size=values.size();out.write(reinterpret_cast<const char*>(&size),sizeof(size));
    out.write(reinterpret_cast<const char*>(values.data()),std::streamsize(values.size()*sizeof(float)));
}
template<class F> void measure(const std::string& name,F&& run) {
    run();run();
    allocations=allocated_bytes=0;
    const auto start=std::chrono::steady_clock::now();counting=true;
    for(int i=0;i<100;++i)run();
    counting=false;
    const auto us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/100.;
    std::cout<<name<<','<<allocations/100.<<','<<allocated_bytes/100.<<','<<us<<'\n';
}
void layerCase(std::ofstream& dump,LayerType type,const std::string& name,int tokens) {
    Layer layer;layer.type=name;layer.type_enum=type;layer.seq_len=tokens;
    layer.embed_dim=32;layer.num_heads=4;layer.in_features=16;layer.out_features=24;
    layer.use_bias=true;
    const bool attention=type==LayerType::SelfAttention||type==LayerType::CrossAttention;
    const int gates=type==LayerType::LSTM?4:type==LayerType::GRU?3:1;
    size_t nw=attention?4*32*32+4*32:size_t(gates)*(24*16+24*24+2*24);
    if(type==LayerType::CrossAttention){layer.in_features=32;nw=4*32*32+32;}
    layer.weights.resize(nw);fill(layer.weights,.01f);
    std::vector<float> x(size_t(tokens)*(attention?32:16)),kv(size_t(tokens+3)*32);fill(x);fill(kv,.07f);
    std::vector<const std::vector<float>*> inputs{&x};
    if(type==LayerType::CrossAttention)inputs.push_back(&kv);
    std::vector<std::vector<float>> output,grad;
    auto forward=[&]{if(!RuntimeLayerDispatch::cpu_forward_layer(inputs,output,layer,true))throw std::runtime_error("forward "+name);};
    forward();std::vector<float> dy(output[0].size());fill(dy,.03f);
    const std::vector<const std::vector<float>*> upstream{&dy};
    auto backward=[&]{
        std::fill(layer.grad_weights.begin(),layer.grad_weights.end(),0.f);
        std::fill(layer.grad_bias.begin(),layer.grad_bias.end(),0.f);
        if(!RuntimeLayerDispatch::cpu_backward_layer(inputs,upstream,grad,layer,true))throw std::runtime_error("backward "+name);
    };
    measure(name+"_fwd_"+std::to_string(tokens),forward);
    write(dump,output[0]);
    measure(name+"_bwd_"+std::to_string(tokens),backward);
    for(const auto& g:grad)write(dump,g);
    write(dump,layer.grad_weights);write(dump,layer.grad_bias);
}
void causalCase(std::ofstream& dump,int seq,bool causal) {
    const int model=32,heads=4,kv_heads=2;
    std::vector<float> x(size_t(seq)*model),dy(x.size()),w(2*32*32+2*16*32),dw(w.size()),y,dx;
    fill(x);fill(dy,.03f);fill(w,.01f);
    auto run=[&]{std::fill(dw.begin(),dw.end(),0.f);
        if(!CausalAttentionOps::run(x,dy,seq,model,heads,kv_heads,causal,10000.f,w.data(),dw.data(),y,&dx))throw std::runtime_error("causal");};
    measure("gqa_bwd_"+std::to_string(seq)+(causal?"_causal":"_full"),run);
    write(dump,y);write(dump,dx);write(dump,dw);
}
void convCase(std::ofstream& dump,int dilation,int stride) {
    Layer layer;layer.use_bias=true;layer.weights.resize(3*2*9+3);fill(layer.weights,.01f);
    std::vector<float> x(2*7*9),y,dx;fill(x);
    auto forward=[&]{CpuConvolution::forward(x,y,layer.weights.data(),layer.weights.data()+54,7,9,2,3,3,stride,2,dilation);};
    forward();std::vector<float> dy(y.size());fill(dy,.03f);
    auto backward=[&]{std::fill(layer.grad_weights.begin(),layer.grad_weights.end(),0.f);std::fill(layer.grad_bias.begin(),layer.grad_bias.end(),0.f);
        CpuConvolution::backward(x,dy,dx,layer,7,9,2,3,3,stride,2,dilation);};
    const std::string suffix=std::to_string(dilation)+"_"+std::to_string(stride);
    measure("conv_fwd_"+suffix,forward);write(dump,y);
    measure("conv_bwd_"+suffix,backward);write(dump,dx);write(dump,layer.grad_weights);write(dump,layer.grad_bias);
}
}
int main(int argc,char** argv) {
    try {
        std::ofstream dump(argc>1?argv[1]:"/tmp/mimir-buffer-values.bin",std::ios::binary);
        std::cout<<"case,allocations_per_call,allocated_bytes_per_call,microseconds_per_call\n";
        for(int tokens:{1,7,13}) {
            for(const auto& pair: {std::pair{LayerType::SelfAttention,"SelfAttention"}, {LayerType::CrossAttention,"CrossAttention"},
                {LayerType::RNN,"RNN"},{LayerType::GRU,"GRU"},{LayerType::LSTM,"LSTM"}})
                layerCase(dump,pair.first,pair.second,tokens);
            causalCase(dump,tokens,true);causalCase(dump,tokens,false);
        }
        for(int dilation:{1,2})for(int stride:{1,2})convCase(dump,dilation,stride);
        if(!dump)throw std::runtime_error("value dump failed");
    }catch(const std::exception& error){counting=false;std::cerr<<error.what()<<'\n';return 1;}
}
