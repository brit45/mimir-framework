#include "runtimes/cpu/FiniteCheckDispatch.hpp"
#include "runtimes/cpu/SimdDispatch.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

static void check(bool ok) {if(!ok)throw std::runtime_error("finite classification changed");}
static bool reference(const float* values,size_t count) {
    if(!values)return count==0;
    for(size_t i=0;i<count;++i)if(!std::isfinite(values[i]))return false;
    return true;
}
static float value(std::uint32_t bits) {float result;std::memcpy(&result,&bits,sizeof(result));return result;}
int main() {
    try {
        using Mimir::RuntimeChecks::allFiniteDispatched;
        check(allFiniteDispatched(nullptr,0));check(!allFiniteDispatched(nullptr,1));
        const auto original=RuntimeSimd::selectedIsa();
        for(auto isa:{RuntimeSimd::Isa::Scalar,RuntimeSimd::Isa::SSE2,RuntimeSimd::Isa::AVX,
                     RuntimeSimd::Isa::AVX2,RuntimeSimd::Isa::AVX512}) {
            if(!RuntimeSimd::supported(isa))continue;
            check(RuntimeSimd::requestIsa(isa));
            for(size_t count:{0u,1u,3u,4u,7u,8u,15u,16u,31u,32u,33u,63u,64u,65u,127u,1027u,65537u})
                for(size_t offset:{0u,1u,3u}) {
                    // Invalid sentinels outside the view expose reads past a
                    // SIMD tail even when those bytes are still allocated.
                    std::vector<float> storage(count+offset+8,std::numeric_limits<float>::quiet_NaN());
                    float* data=storage.data()+offset;
                    for(size_t i=0;i<count;++i) {
                        const float limits[]={0.f,-0.f,std::numeric_limits<float>::max(),-std::numeric_limits<float>::max(),
                            std::numeric_limits<float>::min(),std::numeric_limits<float>::denorm_min(),-std::numeric_limits<float>::denorm_min()};
                        data[i]=limits[i%7];
                    }
                    check(allFiniteDispatched(data,count)==reference(data,count));
                    if(!count)continue;
                    std::vector<size_t> indices;
                    if(count<=127)for(size_t i=0;i<count;++i)indices.push_back(i);
                    else indices={0,3,7,8,15,16,23,31,32,63,64,count-1};
                    for(size_t index:indices)for(std::uint32_t bits:{0x7f800000u,0xff800000u,0x7f800001u,0x7fc00000u,0xffc00001u}) {
                        const float saved=data[index];data[index]=value(bits);
                        check(!reference(data,count));check(!allFiniteDispatched(data,count));data[index]=saved;
                    }
                }
            // Independent std::isfinite classification across every exponent,
            // both signs and normal/subnormal/NaN mantissa boundaries.
            std::vector<float> data(65,.5f);size_t index=0;
            for(std::uint32_t sign:{0u,0x80000000u})for(std::uint32_t exponent=0;exponent<256;++exponent)
                for(std::uint32_t mantissa:{0u,1u,0x400000u,0x7fffffu}) {
                    index=(index+13)%data.size();data[index]=value(sign|(exponent<<23)|mantissa);
                    check(allFiniteDispatched(data.data(),data.size())==reference(data.data(),data.size()));
                    data[index]=.5f;
                }
        }
        check(RuntimeSimd::requestIsa(original));
        std::cout<<"FINITE_DISPATCH_OK exact IEEE classification, null, signs, subnormals, infinities, NaNs, misalignment and tails\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
