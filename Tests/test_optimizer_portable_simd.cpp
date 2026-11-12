#include "Model.hpp"
#include "runtimes/cpu/SimdDispatch.hpp"
#include "runtimes/optimizers/OptimizerDispatch.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>

struct State {
    std::vector<float> w,g,m,v;
    explicit State(size_t count):w(count),g(count),m(count),v(count) {
        for(size_t i=0;i<count;++i){w[i]=float(int(i%23)-11)*.017f;g[i]=float(int(i%19)-9)*.013f;
            m[i]=float(int(i%7)-3)*.00017f;v[i]=float(i%13)*.000013f;}
    }
    RuntimeOptimizerRequest request(const Optimizer& optimizer) {
        return {&optimizer,.0013f,w.data(),g.data(),m.data(),v.data(),w.size()};
    }
};
void exact(const State& a,const State& b) {
    for(auto pair:{std::pair{&a.w,&b.w},std::pair{&a.m,&b.m},std::pair{&a.v,&b.v}})
        if(std::memcmp(pair.first->data(),pair.second->data(),pair.first->size()*sizeof(float)))
            throw std::runtime_error("portable Adam SIMD changed parameter/moment bits");
}
int main(){
    struct Reset{~Reset(){RuntimeSimd::clearIsaRequest();}} reset;
    const bool have_avx2=RuntimeSimd::supported(RuntimeSimd::Isa::AVX2);
    for(auto type:{OptimizerType::ADAM,OptimizerType::ADAMW})for(size_t count:{1,7,8,9,33,257,4097}) {
        Optimizer optimizer;optimizer.type=type;optimizer.weight_decay=.031f;optimizer.eps=1e-8f;
        State scalar(count),vector=scalar,parallel=scalar;
        for(int step:{1,2,17,1000}) {
            optimizer.step=step;
            RuntimeSimd::requestIsa(RuntimeSimd::Isa::Scalar);
            {Mimir::Planning::ExecutionPolicy::CpuScope scope(1);optimizerHostStep(scalar.request(optimizer));}
            if(!RuntimeSimd::requestIsa(RuntimeSimd::Isa::AVX2))continue;
            {Mimir::Planning::ExecutionPolicy::CpuScope scope(1);optimizerHostStep(vector.request(optimizer));}
            {Mimir::Planning::ExecutionPolicy::CpuScope scope(3);optimizerHostStep(parallel.request(optimizer));}
            exact(scalar,vector);exact(scalar,parallel);
        }
    }
    std::cout<<(have_avx2?"PASS Adam/AdamW scalar versus runtime AVX2: exact weights/moments, tails and parallel partitions\n":"PASS scalar optimizer; AVX2 unavailable on this host\n");
}
