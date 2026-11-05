#include "Planning/ExecutionPolicy.hpp"
#include "MemoryGuard.hpp"
#include <cstdlib>
#include <limits>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__unix__)
#include <unistd.h>
#endif
namespace Mimir::Planning::ExecutionPolicy {
namespace {thread_local CpuChoice active;}
CpuChoice cpuChoice(){return active;}
CpuScope::CpuScope(int workers,CpuAlgorithm algorithm):previous_(active){active={std::max(1,workers),algorithm};}
CpuScope::~CpuScope(){active=previous_;}
int cpuWorkers(size_t work){
    if(work<=1 || active.workers<=1)return 1;
#ifdef _OPENMP
    if(omp_in_parallel())return 1;
    return int(std::min({work,size_t(active.workers),size_t(std::max(1,omp_get_max_threads()))}));
#else
    return 1;
#endif
}
std::vector<int> cpuCandidates(bool ordered_reduction){
    std::vector<int> result{1};
    if(ordered_reduction)return result;
#ifdef _OPENMP
    if(omp_in_parallel())return result;
    const int maximum=std::max(1,omp_get_max_threads());
    for(int n=2;n<maximum && n>0;){result.push_back(n);if(n>maximum/2)break;n*=2;}
    if(maximum>1 && result.back()!=maximum)result.push_back(maximum);
#endif
    return result;
}
std::vector<CpuAlgorithm> cpuAlgorithms(bool pointwise){
    return pointwise?std::vector<CpuAlgorithm>{CpuAlgorithm::Reference,CpuAlgorithm::PointwiseConvolution}:
        std::vector<CpuAlgorithm>{CpuAlgorithm::Reference};
}
int configuredMinimum(const char* backend,Work kind){
    const char* suffix=kind==Work::Linear?"_LINEAR_MIN_OPS":kind==Work::Convolution?"_CONV_MIN_OPS":
        kind==Work::Normalization?"_NORM_MIN_ELEMS":"_ATTENTION_MIN_OPS";
    const std::string name=std::string("MIMIR_")+(backend?backend:"")+suffix;
    const char* value=std::getenv(name.c_str());
    if(!value)return automatic_minimum;
    char* end=nullptr;const long long parsed=std::strtoll(value,&end,10);
    if(end==value || *end || parsed<=0)return automatic_minimum;
    return int(std::min(parsed,static_cast<long long>(std::numeric_limits<int>::max())));
}
size_t calibrationBudget(){
    const auto& guard=MemoryGuard::instance();
    const size_t limit=guard.getLimit(),used=guard.getCurrentBytes();
    size_t budget=limit?(used<limit?limit-used:0):std::numeric_limits<size_t>::max();
    bool known=limit!=0;
#if defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
    const long pages=sysconf(_SC_AVPHYS_PAGES),page_size=sysconf(_SC_PAGESIZE);
    if(pages>=0 && page_size>0 && uint64_t(pages)<=std::numeric_limits<size_t>::max()/uint64_t(page_size)) {
        budget=std::min(budget,size_t(pages)*size_t(page_size));known=true;
    }
#endif
    if(const char* value=std::getenv("MIMIR_PLANNER_CALIBRATION_MAX_BYTES")) {
        budget=std::min(budget,size_t(std::strtoull(value,nullptr,10)));known=true;
    }
    // No guessed RAM capacity on unknown platforms: calibration stays off
    // until the application supplies a MemoryGuard or an explicit budget.
    return known?budget:0;
}

size_t cacheBytes(){
    static const size_t size=[]()->size_t {
#if defined(_SC_LEVEL2_CACHE_SIZE)
        const long value=sysconf(_SC_LEVEL2_CACHE_SIZE);if(value>0)return size_t(value);
#endif
        return 0; // Unknown hardware: no invented cache capacity.
    }();return size;
}
int convolutionTileRows(size_t reduction,size_t channels,int spatial){
    if(spatial<=0 || reduction>std::numeric_limits<size_t>::max()-channels)return 1;
    if(channels>(std::numeric_limits<size_t>::max()-reduction)/2)return 1;
    const size_t row=reduction+2*channels;
    if(!row || row>std::numeric_limits<size_t>::max()/sizeof(float))return 1;
    size_t bytes=cacheBytes();
    if(const char* value=std::getenv("MIMIR_PLANNER_SCRATCH_BYTES"))bytes=std::strtoull(value,nullptr,10);
    const auto& guard=MemoryGuard::instance();const size_t limit=guard.getLimit(),used=guard.getCurrentBytes();
    if(limit)bytes=std::min(bytes,used<limit?limit-used:size_t(0));
    return int(std::min(size_t(spatial),std::max(size_t(1),(bytes>128?bytes-128:0)/(row*sizeof(float)))));
}
}
