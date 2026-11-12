#include "runtimes/cpu/SimdDispatch.hpp"
#include "runtimes/cpu/SIMD_Ops.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(x) do {if(!(x)){std::cerr<<"SIMD CHECK failed: " #x <<" at "<<__LINE__<<'\n';return 1;}}while(0)
static bool equal(const std::vector<float>& a,const std::vector<float>& b) {
    return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0);
}
static int verify() {
    using namespace RuntimeSimd;
    const Isa original=selectedIsa();
    std::cout<<"CPU SIMD selected="<<isaName(original)<<" fma="<<features().fma<<" f16c="<<features().f16c<<'\n';
    for(bool fused:{false,true})for(std::size_t n:{1u,3u,4u,7u,8u,15u,16u,23u,24u,31u,47u,48u,67u,193u,257u,1024u,2048u}) {
        const std::size_t m=n>67?17:3,k=n==2048?64:129;
        std::vector<float> a(m*k),b(k*n),bt(n*k),reference(m*n),actual(m*n),leftReference(k*n),leftActual(k*n);
        for(std::size_t i=0;i<a.size();++i)a[i]=std::sin(float(i%137)*.13f)*.1f;
        for(std::size_t i=0;i<b.size();++i)b[i]=std::cos(float(i%173)*.17f)*.1f;
        for(std::size_t i=0;i<k;++i)for(std::size_t j=0;j<n;++j)bt[j*k+i]=b[i*n+j];
        CHECK(requestIsa(Isa::Scalar));
        matmul(reference.data(),a.data(),b.data(),m,n,k,false,fused);
        for(std::size_t i=0;i<m;++i)for(std::size_t j=0;j<n;++j){
            double exact=0;for(std::size_t r=0;r<k;++r)exact+=double(a[i*k+r])*double(b[r*n+j]);
            CHECK(std::abs(double(reference[i*n+j])-exact)<1e-6);
        }
        std::vector<float> right(m*n);
        for(std::size_t i=0;i<right.size();++i)right[i]=std::sin(float(i)*.7f);
        matmulLeftTranspose(leftReference.data(),a.data(),right.data(),m,n,k,fused);
        for(Isa isa:{Isa::Scalar,Isa::SSE2,Isa::AVX,Isa::AVX2,Isa::AVX512}) {
            if(!supported(isa)){CHECK(!requestIsa(isa));continue;}
            CHECK(requestIsa(isa));
            matmul(actual.data(),a.data(),b.data(),m,n,k,false,fused);
            CHECK(equal(reference,actual));
            matmulTranspose(actual.data(),a.data(),bt.data(),m,n,k,fused,2);
            CHECK(equal(reference,actual));
            matmulLeftTranspose(leftActual.data(),a.data(),right.data(),m,n,k,fused);
            CHECK(equal(leftReference,leftActual));
            matmul(actual.data(),a.data(),b.data(),m,n,k,true,fused);
            for(std::size_t i=0;i<actual.size();++i)CHECK(actual[i]==reference[i]+reference[i]);
            std::vector<float> batchA=a,batchB=b,batchC(2*m*n);
            batchA.insert(batchA.end(),a.begin(),a.end());batchB.insert(batchB.end(),b.begin(),b.end());
            matmulBatched(batchC.data(),batchA.data(),batchB.data(),2,m,n,k,fused,2);
            CHECK(std::memcmp(batchC.data(),reference.data(),reference.size()*sizeof(float))==0);
            CHECK(std::memcmp(batchC.data()+m*n,reference.data(),reference.size()*sizeof(float))==0);
        }
    }
    // Independent ordered reference with padded and transposed views. Exercise
    // all panel boundaries, output padding, nonzero gradient accumulation and
    // zero reductions (which may legally receive null input pointers).
    for (const auto shape : {std::array<size_t,3>{1,7,0}, {1,193,33},
            {9,197,65}, {17,257,129}, {9,1024,129}, {3,19,2048}}) {
        const auto m=shape[0], n=shape[1], k=shape[2];
        for (bool transposed : {false,true}) for (bool fused : {false,true}) {
            const size_t ar=k*2+3, ak=2, bc=transposed?k+3:1, bk=transposed?1:n+5, cr=n+7;
            std::vector<float> a(m*ar), b(k*bk+n*bc+1), expected(m*cr,17.f), actual;
            for(size_t i=0;i<a.size();++i)a[i]=std::sin(float(i%137)*.13f)*.1f;
            for(size_t i=0;i<b.size();++i)b[i]=std::cos(float(i%173)*.17f)*.1f;
            for(bool accumulate : {false,true}) {
                std::fill(expected.begin(),expected.end(),17.f);
                for(size_t i=0;i<m;++i)for(size_t j=0;j<n;++j) {
                    float sum=0;
                    for(size_t r=0;r<k;++r) {
                        if(fused)sum=std::fma(a[i*ar+r*ak],b[r*bk+j*bc],sum);
                        else sum+=a[i*ar+r*ak]*b[r*bk+j*bc];
                    }
                    expected[i*cr+j]=accumulate?17.f+sum:sum;
                }
                for(Isa isa:{Isa::Scalar,Isa::SSE2,Isa::AVX,Isa::AVX2,Isa::AVX512})if(supported(isa)) {
                    CHECK(requestIsa(isa));
                    for(int workers:{1,3}) {
                        actual.assign(m*cr,17.f);
                        matmulStrided(actual.data(),k?a.data():nullptr,k?b.data():nullptr,
                            m,n,k,ar,ak,bk,bc,cr,accumulate,fused,workers);
                        CHECK(equal(expected,actual));
                    }
                }
            }
            std::fill(expected.begin(),expected.end(),17.f);
            for(size_t i=0;i<m;++i)for(size_t j=0;j<n;++j)for(size_t r=0;r<k;++r) {
                float& value=expected[i*cr+j];
                if(fused)value=std::fma(a[i*ar+r*ak],b[r*bk+j*bc],value);
                else value+=a[i*ar+r*ak]*b[r*bk+j*bc];
            }
            for(Isa isa:{Isa::Scalar,Isa::SSE2,Isa::AVX,Isa::AVX2,Isa::AVX512})if(supported(isa)) {
                CHECK(requestIsa(isa));
                for(int workers:{1,3}) {
                    actual.assign(m*cr,17.f);
                    const size_t split=std::min(size_t(31),k);
                    matmulStridedContinue(actual.data(),k?a.data():nullptr,k?b.data():nullptr,
                        m,n,split,ar,ak,bk,bc,cr,fused,workers);
                    matmulStridedContinue(actual.data(),k?a.data()+split*ak:nullptr,k?b.data()+split*bk:nullptr,
                        m,n,k-split,ar,ak,bk,bc,cr,fused,workers);
                    CHECK(equal(expected,actual));
                }
            }
        }
        std::vector<float> a(m*k), bt(n*k), bias(n), expected(m*n), actual(m*n);
        for(size_t i=0;i<a.size();++i)a[i]=std::sin(float(i)*.13f)*.1f;
        for(size_t i=0;i<bt.size();++i)bt[i]=std::cos(float(i)*.17f)*.1f;
        for(size_t j=0;j<n;++j)bias[j]=std::cos(float(j)*.07f);
        for(size_t i=0;i<m;++i)for(size_t j=0;j<n;++j) {
            float sum=bias[j]; // Bias must seed the reduction, not be added after.
            for(size_t r=0;r<k;++r)sum+=a[i*k+r]*bt[j*k+r];
            expected[i*n+j]=sum;
        }
        for(Isa isa:{Isa::Scalar,Isa::SSE2,Isa::AVX,Isa::AVX2,Isa::AVX512})if(supported(isa)) {
            CHECK(requestIsa(isa));
            matmulTransposeBiased(actual.data(),a.data(),bt.data(),bias.data(),m,n,k,3);
            CHECK(equal(expected,actual));
        }
    }
    for(std::size_t count:{0u,1u,3u,4u,7u,8u,15u,16u,17u,31u,67u,1027u}) {
        std::vector<float> a(count),b(count),reference(count),actual(count);
        for(std::size_t i=0;i<count;++i){a[i]=std::sin(float(i)*.11f);b[i]=std::cos(float(i)*.13f);}
        for(Isa isa:{Isa::Scalar,Isa::SSE2,Isa::AVX,Isa::AVX2,Isa::AVX512}) {
            if(!supported(isa))continue;
            CHECK(requestIsa(Isa::Scalar));add(reference.data(),a.data(),b.data(),count);
            CHECK(requestIsa(isa));add(actual.data(),a.data(),b.data(),count);CHECK(equal(reference,actual));
            CHECK(requestIsa(Isa::Scalar));mul(reference.data(),a.data(),b.data(),count);
            CHECK(requestIsa(isa));mul(actual.data(),a.data(),b.data(),count);CHECK(equal(reference,actual));
            for(bool fused:{false,true}){
                reference=b;actual=b;
                CHECK(requestIsa(Isa::Scalar));axpy(reference.data(),a.data(),.137f,count,fused);
                CHECK(requestIsa(isa));axpy(actual.data(),a.data(),.137f,count,fused);CHECK(equal(reference,actual));
            }
            // Exact input/output aliasing and misaligned subviews, including tails.
            reference=a;actual=a;
            for(std::size_t i=0;i<count;++i)reference[i]+=b[i];
            add(actual.data(),actual.data(),b.data(),count);CHECK(equal(reference,actual));
            if(count>1){add(actual.data()+1,a.data()+1,b.data()+1,count-1);CHECK(std::memcmp(actual.data()+1,reference.data()+1,(count-1)*sizeof(float))==0);}
        }
    }
    // A width-dependent polynomial/clamped-tanh activation is not GELU.
    std::vector<float> x(35),gelu(35);
    for(std::size_t i=0;i<x.size();++i)x[i]=(float(i)-17.f)*.21f;
    SIMD::gelu_forward_avx2(gelu.data(),x.data(),x.size());
    for(std::size_t i=0;i<x.size();++i)CHECK(gelu[i]==x[i]*.5f*(1.f+std::tanh(.7978845608f*(x[i]+.044715f*x[i]*x[i]*x[i]))));
    SIMD::softmax_avx2(nullptr,nullptr,0);
    std::vector<float> zeroK(2*67,5.f);
    matmul(zeroK.data(),nullptr,nullptr,2,67,0);for(float v:zeroK)CHECK(v==0);
    matmul(zeroK.data(),nullptr,nullptr,2,67,0,true);for(float v:zeroK)CHECK(v==0);
    matmulTranspose(zeroK.data(),nullptr,nullptr,2,67,0);for(float v:zeroK)CHECK(v==0);
    // Compare hardware F16C and software encoding for finite values, including
    // values at half-way boundaries, subnormals, infinities, and vector tails.
    std::vector<float> halfValues;
    for(unsigned bits=0;bits<65536;++bits) {
        if((bits&0x7c00)==0x7c00)continue;
        const std::uint16_t h=std::uint16_t(bits);float value;
        CHECK(requestIsa(Isa::Scalar));fp16ToFp32(&value,&h,1);halfValues.push_back(value);
    }
    halfValues.push_back(std::ldexp(1.f,-25));halfValues.push_back(1.f+std::ldexp(1.f,-11));
    halfValues.push_back(std::numeric_limits<float>::infinity());
    std::vector<std::uint16_t> encoded(halfValues.size()),hardware(encoded.size());
    CHECK(requestIsa(Isa::Scalar));fp32ToFp16(encoded.data(),halfValues.data(),halfValues.size());
    for(Isa isa:{Isa::Scalar,Isa::SSE2,Isa::AVX,Isa::AVX2,Isa::AVX512})if(supported(isa)){
        CHECK(requestIsa(isa));fp32ToFp16(hardware.data(),halfValues.data(),halfValues.size());CHECK(encoded==hardware);
    }
    CHECK(!requestIsa(static_cast<Isa>(1000)));
    CHECK(requestIsa(original));
    std::cout<<"SIMD_ORDERED_PARITY_OK (scalar/SSE2/AVX/AVX2/AVX512 where available, fused and separate)\n";
    return 0;
}
static int benchmark() {
    using namespace RuntimeSimd;
    const Isa fastest=selectedIsa();
    if(fastest==Isa::Scalar){std::cout<<"SIMD benchmark: scalar-only build/selection\n";return 0;}
    constexpr std::size_t m=128,n=256,k=256;
    std::vector<float> a(m*k),b(k*n),c(m*n),reference(m*n);
    for(std::size_t i=0;i<a.size();++i)a[i]=std::sin(float(i)*.13f);
    for(std::size_t i=0;i<b.size();++i)b[i]=std::cos(float(i)*.17f);
    for(bool fused:{false,true}) {
        double scalar=0,accelerated=0;
        for(Isa isa:{Isa::Scalar,fastest}){
            CHECK(requestIsa(isa));
            matmul(c.data(),a.data(),b.data(),m,n,k,false,fused);
            const auto before=std::chrono::steady_clock::now();
            for(int iteration=0;iteration<5;++iteration)matmul(c.data(),a.data(),b.data(),m,n,k,false,fused);
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count()/5;
            if(isa==Isa::Scalar){scalar=ms;reference=c;}else{accelerated=ms;CHECK(equal(reference,c));}
        }
        std::cout<<"SIMD benchmark 128x256x256 serial fused="<<fused<<" scalar_ms="<<scalar<<" selected="<<isaName(fastest)<<" ms="<<accelerated<<" speedup="<<scalar/accelerated<<'\n';
    }
    return 0;
}
int main(int argc,char** argv) {
    if(argc>1 && std::string(argv[1])=="--expect-env-rejection") {
        try{(void)RuntimeSimd::selectedIsa();}catch(const std::invalid_argument& e){std::cout<<e.what()<<'\n';return 0;}
        return 1;
    }
    if(argc>1 && std::string(argv[1])=="--environment") {
        std::cout<<RuntimeSimd::isaName(RuntimeSimd::selectedIsa())<<'\n';return 0;
    }
    if(argc>1 && std::string(argv[1])=="--benchmark")return benchmark();
    return verify();
}
