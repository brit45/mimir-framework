#pragma once
#include "runtimes/GpuPointwiseFusion.hpp"
#include "PortableDevice.hpp"
#include "PortableWorkBudget.hpp"
#include "PointwiseFusionIds.hpp"
#include <mutex>

namespace Mimir::GpuFusion {
class PortableExecutor {
    std::unique_ptr<PortableStaging::Device> device;
    std::mutex mutex;
public:
    bool run(Program& program,const char* backend,int index) {
        if(!program.n)return true;
        std::lock_guard<std::mutex> lock(mutex);
        if(!device)device=PortableStaging::createDevice(backend,index);
        if(!device)return false;
        const auto& p=program;
        const int blocks=(p.n+255)/256;
        std::vector<int> args=p.backward?
            std::vector<int>{backward_kernel_id,256,blocks,p.input,p.snapshots,p.operations,p.output,p.n,p.stages}:
            std::vector<int>{forward_kernel_id,256,blocks,p.input,p.operations,p.output,p.n,p.stages,int(p.retain)};
        if(!PortableStaging::parallelDispatchFits(p.memory.size(),args))return false;
        return device->runBatch(program.memory,{args},{{size_t(p.output),p.output_count}});
    }
};
}
