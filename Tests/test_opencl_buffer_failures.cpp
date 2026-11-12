#include "runtimes/opencl/OpenCLCompute.hpp"
#include <iostream>
#include <stdexcept>
#ifdef ENABLE_OPENCL
namespace {int fail_after=0;bool keep_failing=false,fail_now=false;}
extern "C" cl_mem __real_clCreateBuffer(cl_context,cl_mem_flags,size_t,void*,cl_int*);
extern "C" cl_mem __wrap_clCreateBuffer(cl_context context,cl_mem_flags flags,size_t size,void* host,cl_int* error) {
    if(fail_now || (fail_after>0 && --fail_after==0)) {
        fail_now=keep_failing;
        if(error)*error=CL_MEM_OBJECT_ALLOCATION_FAILURE;
        return nullptr;
    }
    return __real_clCreateBuffer(context,flags,size,host,error);
}
#endif
int main(){
#ifndef ENABLE_OPENCL
    return 77;
#else
    OpenCLCompute::ComputeEngine engine;
    if(!engine.initialize())return 77;
    auto run=[&](int m,int k,int n,bool expected) {
        std::vector<float> a(size_t(m)*k,1.f),b(size_t(k)*n,1.f),out(size_t(m)*n,-5.f);
        if(engine.matmulForward(a.data(),b.data(),out.data(),m,k,n)!=expected)
            throw std::runtime_error("unexpected OpenCL allocation recovery result");
        for(float value:out)if(value!=(expected?float(k):-5.f))
            throw std::runtime_error("OpenCL failure/recovery corrupted inputs or output");
    };
    run(2,2,2,true);
    fail_after=1;run(9,17,19,true); // Evict idle slots, retry allocation.
    fail_after=2;run(13,23,31,true); // Preserve an earlier borrowed argument.
    run(13,23,31,true);
    fail_after=3;keep_failing=true;run(31,29,37,false); // Drop all borrowed references on failure.
    if(engine.cachedBufferBytes()!=0)throw std::runtime_error("failed allocation retained idle cache");
    fail_now=keep_failing=false;fail_after=0;
    run(31,29,37,true);
    const auto count=engine.bufferAllocationCount();run(31,29,37,true);
    if(engine.bufferAllocationCount()!=count)throw std::runtime_error("recovered cache did not warm up");
    engine.cleanup();
    if(engine.cachedBufferBytes())throw std::runtime_error("cleanup leaked cached device storage");
    std::cout<<"PASS OpenCL injected allocation failures: eviction, live arguments, unchanged failed output, retry and cleanup\n";
#endif
}
