#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include "Planning/ExecutionPolicy.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace RuntimeParallel {
// An OpenMP `if(false)` region still calls into the runtime. Dispatch outside
// the region so small operations in a graph retain a plain SIMD loop.
template<class Function>
inline void forEach(size_t count, int thread_count, const Function& operation) {
    thread_count=std::max(1,std::min(thread_count,int(std::min(count,size_t(std::numeric_limits<int>::max())))));
    if (thread_count == 1) {
        #pragma omp simd
        for (size_t i = 0; i < count; ++i) operation(i);
    } else {
        #pragma omp parallel for simd num_threads(thread_count) schedule(static)
        for (size_t i = 0; i < count; ++i) operation(i);
    }
}
// The planner supplies a measured choice. Outside a planned dispatch, the
// safe reference is serial; workload sizes do not invent parallel thresholds.
inline int workers(size_t work) {
    return Mimir::Planning::ExecutionPolicy::cpuWorkers(work);
}
}
