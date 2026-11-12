#include "runtimes/cpu/AttentionBackward.hpp"
#include "runtimes/cpu/ReusableScratch.hpp"
#include "runtimes/cpu/SimdDispatch.hpp"
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void equal(const std::vector<float>& actual, const std::vector<float>& expected, const char* label) {
    check(actual.size()==expected.size(),"attention output size");
    for (size_t i=0;i<actual.size();++i)if (std::memcmp(&actual[i],&expected[i],sizeof(float))) {
        std::cerr<<label<<" at "<<i<<": "<<actual[i]<<" expected "<<expected[i]<<'\n';
        throw std::runtime_error("attention ordered parity");
    }
}
struct Result { std::vector<float> attended, q, k, v; };

// Deliberately simple full-matrix reference, with the original scalar
// reduction order, softmax epsilon, finite mask and initial maximum.
Result reference(const std::vector<float>& q,const std::vector<float>& k,const std::vector<float>& v,
    const std::vector<float>& da,size_t qs,size_t ks,size_t nq,size_t nk,size_t dim,size_t heads,bool causal) {
    Result r{std::vector<float>(nq*dim),std::vector<float>(nq*dim),
             std::vector<float>(nk*dim),std::vector<float>(nk*dim)};
    const size_t hd=dim/heads;
    const float scale=1.f/std::sqrt(float(hd));
    std::vector<float> p(nq*nk),dp(p.size()),ds(p.size());
    for (size_t h=0;h<heads;++h) {
        const size_t off=h*hd;
        for (size_t i=0;i<nq;++i) {
            float maximum=-1e30f;
            for (size_t j=0;j<nk;++j) {
                float score=-1e9f;
                if (!(causal && j>i)) {
                    float dot=0.f;
                    for (size_t d=0;d<hd;++d) dot+=q[i*qs+off+d]*k[j*ks+off+d];
                    score=dot*scale;
                }
                p[i*nk+j]=score;
                if (score>maximum) maximum=score;
            }
            float total=0.f;
            for (size_t j=0;j<nk;++j) {
                p[i*nk+j]=std::exp(p[i*nk+j]-maximum);
                total+=p[i*nk+j];
            }
            const float inverse=1.f/(total+1e-9f);
            for (size_t j=0;j<nk;++j) p[i*nk+j]*=inverse;
            for (size_t d=0;d<hd;++d) {
                float sum=0.f;
                for (size_t j=0;j<nk;++j) sum+=p[i*nk+j]*v[j*ks+off+d];
                r.attended[i*dim+off+d]=sum;
            }
        }
        for (size_t j=0;j<nk;++j)for (size_t d=0;d<hd;++d) {
            float sum=0.f;
            for (size_t i=0;i<nq;++i) sum+=p[i*nk+j]*da[i*dim+off+d];
            r.v[j*dim+off+d]+=sum;
        }
        for (size_t i=0;i<nq;++i)for (size_t j=0;j<nk;++j) {
            float sum=0.f;
            for (size_t d=0;d<hd;++d) sum+=da[i*dim+off+d]*v[j*ks+off+d];
            dp[i*nk+j]=sum;
        }
        for (size_t i=0;i<nq;++i) {
            float dot=0.f;
            for (size_t j=0;j<nk;++j) dot+=dp[i*nk+j]*p[i*nk+j];
            for (size_t j=0;j<nk;++j) ds[i*nk+j]=causal && j>i?0.f:(dp[i*nk+j]-dot)*p[i*nk+j];
        }
        for (size_t i=0;i<nq;++i)for (size_t d=0;d<hd;++d) {
            float sum=0.f;
            for (size_t j=0;j<nk;++j) sum+=ds[i*nk+j]*k[j*ks+off+d];
            r.q[i*dim+off+d]+=sum*scale;
        }
        for (size_t j=0;j<nk;++j)for (size_t d=0;d<hd;++d) {
            float sum=0.f;
            for (size_t i=0;i<nq;++i) sum+=ds[i*nk+j]*q[i*qs+off+d];
            r.k[j*dim+off+d]+=sum*scale;
        }
    }
    return r;
}

void caseOf(size_t nq,size_t nk,size_t dim,size_t heads,bool causal,bool extreme=false) {
    const size_t qs=3*dim,ks=2*dim+5;
    const float nan=std::numeric_limits<float>::quiet_NaN();
    std::vector<float> q(nq*qs,nan),k(nk*ks,nan),v(k.size(),nan),da(nq*dim);
    for (size_t i=0;i<nq;++i)for (size_t d=0;d<dim;++d) q[i*qs+d]=extreme?1e4f:float(int((i*13+d*7)%37)-18)*.023f;
    for (size_t i=0;i<nk;++i)for (size_t d=0;d<dim;++d) {
        k[i*ks+d]=extreme?-1e6f:float(int((i*11+d*3)%41)-20)*.017f;
        v[i*ks+d]=float(int((i*7+d*13)%31)-15)*.013f;
    }
    for (size_t i=0;i<da.size();++i) da[i]=float(int(i%19)-9)*.011f;
    const auto want=reference(q,k,v,da,qs,ks,nq,nk,dim,heads,causal);
    for (int workers:{1,3}) {
        Result got{std::vector<float>(nq*dim,17.f),std::vector<float>(nq*dim,17.f),
                   std::vector<float>(nk*dim,17.f),std::vector<float>(nk*dim,17.f)};
        CpuAttention::backward(q.data(),k.data(),v.data(),qs,ks,da.data(),
            got.attended.data(),got.q.data(),got.k.data(),got.v.data(),nq,nk,dim,heads,causal,workers);
        equal(got.attended,want.attended,"attended");equal(got.q,want.q,"dQ");
        equal(got.k,want.k,"dK");equal(got.v,want.v,"dV");
    }
}
}

int main() {
    try {
        const auto original=RuntimeSimd::selectedIsa();
        for (auto isa:{RuntimeSimd::Isa::Scalar,RuntimeSimd::Isa::SSE2,RuntimeSimd::Isa::AVX,
                       RuntimeSimd::Isa::AVX2,RuntimeSimd::Isa::AVX512}) {
            if (!RuntimeSimd::supported(isa)) continue;
            check(RuntimeSimd::requestIsa(isa),"ISA request");
            for (auto shape:{std::array<size_t,4>{1,1,3,1},{7,11,15,3},{33,65,32,4},
                             {65,33,16,2},{129,193,16,2},{193,129,21,3},{65,7,1032,1},
                             {16,16,16,1},{17,16,16,1},{16,17,16,1},{16,16,17,1}})
                for (bool causal:{false,true})caseOf(shape[0],shape[1],shape[2],shape[3],causal);
            caseOf(67,69,8,2,true,true); // Finite causal mask may have nonzero probabilities.
        }
        check(RuntimeSimd::requestIsa(original),"restore ISA");
        auto& pool=RuntimeScratch::Pool::local();pool.clear();
        caseOf(129,193,16,2,false);
        const size_t allocations=pool.allocations();
        caseOf(129,193,16,2,true);
        check(pool.allocations()==allocations,"warm attention allocated another scratch buffer");
#ifdef _OPENMP
        std::atomic<bool> nested{true};
        #pragma omp parallel num_threads(2)
        {
            try {caseOf(65,33,16,2,true);}catch(...){nested=false;}
        }
        check(nested,"nested attention execution");
#endif
        pool.clear();
        auto& guard=MemoryGuard::instance();const size_t limit=guard.getLimit();
        std::vector<float> q(65*4,.1f),result(65*4,-77.f),dq(result.size(),-77.f),dk(4*4,-77.f),dv(dk.size(),-77.f);
        guard.setLimit(1);
        bool rejected=false;
        try {CpuAttention::backward(q.data(),q.data(),q.data(),4,4,q.data(),
            result.data(),dq.data(),dk.data(),dv.data(),65,4,4,2,false,3);}
        catch(const std::exception&){rejected=true;}
        guard.setLimit(limit);
        check(rejected,"attention exceeded MemoryGuard");
        for (const auto* values:{&result,&dq,&dk,&dv})
            for (float f:*values)check(f==-77.f,"failed allocation changed output");
        std::cout<<"ATTENTION_TILED_ORDERED_PARITY_OK tails, causal, padding, workers, nested execution, reuse, budget\n";
    } catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
