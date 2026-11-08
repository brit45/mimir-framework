#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace PortablePrimerValidation {
// Device float controls do not universally preserve FP32 subnormals. Reject
// their inputs explicitly instead of silently changing signs, domains or ties.
// This only classifies storage bits; no tensor result is computed on the host.
inline bool supportedValues(const std::vector<float>& values){
    static_assert(sizeof(float)==sizeof(uint32_t) && std::numeric_limits<float>::is_iec559,
                  "Portable Primer kernels require IEEE-754 binary32");
    for(float value:values){
        uint32_t bits=0;
        std::memcpy(&bits,&value,sizeof(bits));
        const uint32_t exponent=bits&0x7f800000u;
        const uint32_t fraction=bits&0x007fffffu;
        if(exponent==0x7f800000u || (exponent==0 && fraction!=0))return false;
    }
    return true;
}
inline bool supportedInputs(const std::vector<const std::vector<float>*>& inputs){
    for(const auto* input:inputs)if(!input || !supportedValues(*input))return false;
    return true;
}
} // namespace PortablePrimerValidation
