// Nonblocking uploads must complete before run() returns, including failures.
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include "runtimes/portable/PortableDevice.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace {
enum class Failure { None, FirstUpload, SecondUpload, Argument, Read };
Failure failure = Failure::None;
int writes = 0, nonblocking = 0, drains = 0;
void require(bool ok) { if (!ok) throw std::runtime_error("OpenCL upload lifetime regression"); }
}

extern "C" {
cl_int __real_clEnqueueWriteBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t,
    const void*, cl_uint, const cl_event*, cl_event*);
cl_int __wrap_clEnqueueWriteBuffer(cl_command_queue q, cl_mem b, cl_bool blocking,
    size_t offset, size_t size, const void* data, cl_uint n, const cl_event* events, cl_event* out) {
    ++writes;
    nonblocking += blocking == CL_FALSE;
    if ((failure == Failure::FirstUpload && writes == 1) ||
        (failure == Failure::SecondUpload && writes == 2)) return CL_OUT_OF_RESOURCES;
    return __real_clEnqueueWriteBuffer(q,b,blocking,offset,size,data,n,events,out);
}
cl_int __real_clSetKernelArg(cl_kernel, cl_uint, size_t, const void*);
cl_int __wrap_clSetKernelArg(cl_kernel k, cl_uint i, size_t size, const void* value) {
    if (failure == Failure::Argument) return CL_INVALID_ARG_VALUE;
    return __real_clSetKernelArg(k,i,size,value);
}
cl_int __real_clEnqueueReadBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t,
    void*, cl_uint, const cl_event*, cl_event*);
cl_int __wrap_clEnqueueReadBuffer(cl_command_queue q, cl_mem b, cl_bool blocking,
    size_t offset, size_t size, void* data, cl_uint n, const cl_event* events, cl_event* out) {
    if (failure == Failure::Read) return CL_OUT_OF_RESOURCES;
    return __real_clEnqueueReadBuffer(q,b,blocking,offset,size,data,n,events,out);
}
cl_int __real_clFinish(cl_command_queue);
cl_int __wrap_clFinish(cl_command_queue q) { ++drains; return __real_clFinish(q); }
}

int main() {
    try {
        auto device = PortableStaging::createDevice("OPENCL",0);
        if (!device) { std::cout << "SKIP OpenCL unavailable\n"; return 77; }
        constexpr int count = 1027;
        std::vector<float> memory(1+2*count,0.f);
        // cuda_unary_kernel(in, out, n, ReLU, alpha), with disjoint arena views.
        const std::vector<int> arguments{34,5,256,1,1+count,count,0,0};
        auto reset = [&] {
            for (int i=0;i<count;++i) memory[1+i] = float(i%19)-9.f;
            std::fill(memory.begin()+1+count,memory.end(),-123.f);
            writes = nonblocking = drains = 0;
        };
        auto check = [&] {
            for (int i=0;i<count;++i)
                require(memory[1+count+i] == std::max(0.f,memory[1+i]));
        };
        reset(); require(device->run(memory,arguments)); check();
        require(writes == 2 && nonblocking == 2);
        for (auto fault : {Failure::FirstUpload,Failure::SecondUpload,Failure::Argument,Failure::Read}) {
            reset(); failure = fault;
            require(!device->run(memory,arguments));
            require(drains == 1);
            // Once run() returns, callers may overwrite/free both host arrays.
            std::fill(memory.begin(),memory.end(),42.f);
            failure = Failure::None;
            reset(); require(device->run(memory,arguments)); check();
            require(writes == 2 && nonblocking == 2 && drains == 0);
        }
        std::cout << "OPENCL_ASYNC_UPLOAD_LIFETIME_OK failures=4 recovery=4\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
