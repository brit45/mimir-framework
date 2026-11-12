#pragma once
#include "OptimizerDispatch.hpp"
#include "runtimes/portable/PortableDevice.hpp"
#include <map>
#include <memory>
#include <string>
inline bool portableOptimizer(const RuntimeOptimizerRequest& request,const char* backend,int index,
                              PortableStaging::DeviceCache& contexts) {
    if((!optimizerDeviceSizeFits(request) || !optimizerRequestValid(request)))return false;
    auto& device=contexts[{backend,index}];
    if(!device)device=PortableStaging::createOptimizerDevice(backend,index);
    if(!device)return false;
    auto memory=optimizerPack(request, true);const auto arguments=optimizerParameters(request, true);
    if(!device->run(memory,arguments)){device.reset();return false;}
    return optimizerCommit(request,memory,true);
}
