#include "runtimes/opencl/OpenCLCompute.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc,char** argv) {
#ifndef ENABLE_OPENCL
    return 77;
#else
    OpenCLCompute::ComputeEngine engine;
    if(!engine.initialize()){std::cout<<"SKIP OpenCL unavailable\n";return 77;}
    std::ofstream values(argc>1?argv[1]:"/tmp/mimir-opencl-values.bin",std::ios::binary);
    std::cout<<"backend=OPENCL native kernels; no framework fallback\n";
#ifdef MIMIR_VERIFY_OPENCL_CACHE
    std::cout<<"device="<<engine.deviceName()<<" gpu="<<engine.deviceIsGpu()<<'\n';
#endif
    for(int shape:{7,16,33,7}) {
        const int m=shape,k=17,n=19;
        std::vector<float> a(size_t(m)*k),b(size_t(k)*n),out(size_t(m)*n),bias(n);
        for(size_t i=0;i<a.size();++i)a[i]=float(int(i%13)-6)*.03f;
        for(size_t i=0;i<b.size();++i)b[i]=float(int(i%17)-8)*.02f;
        for(size_t i=0;i<bias.size();++i)bias[i]=float(i)*.001f;
        auto measure=[&](const char* name,auto run) {
            if(!run()||!run())throw std::runtime_error(name);
#ifdef MIMIR_VERIFY_OPENCL_CACHE
            const auto allocations=engine.bufferAllocationCount();
#endif
            const auto start=std::chrono::steady_clock::now();
            for(int step=0;step<100;++step)if(!run())throw std::runtime_error(name);
            const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/100.;
#ifdef MIMIR_VERIFY_OPENCL_CACHE
            if(engine.bufferAllocationCount()!=allocations)throw std::runtime_error("warm OpenCL buffers allocated again");
#endif
            for(float value:out)if(!std::isfinite(value))throw std::runtime_error("nonfinite output");
            values.write(reinterpret_cast<const char*>(out.data()),std::streamsize(out.size()*sizeof(float)));
            std::cout<<name<<','<<shape<<','<<us<<'\n';
        };
        measure("matmul",[&]{return engine.matmulForward(a.data(),b.data(),out.data(),m,k,n);});
        std::vector<float> weights(size_t(n)*k);
        for(size_t i=0;i<weights.size();++i)weights[i]=float(int(i%11)-5)*.02f;
        measure("linear",[&]{return engine.linearForward(a.data(),weights.data(),bias.data(),out.data(),m,k,n);});
        // Same parameter address with new values must be uploaded again.
        weights[0]+=.375f;
        measure("linear_changed_weights",[&]{return engine.linearForward(a.data(),weights.data(),nullptr,out.data(),m,k,n);});
        std::vector<float> lhs(out.size(),.5f),rhs(out.size(),.25f);
        measure("unary",[&]{return engine.unaryForward(lhs.data(),out.data(),int(out.size()),0,.01f);});
        measure("binary",[&]{return engine.binaryForward(lhs.data(),rhs.data(),out.data(),int(out.size()),2,.01f);});
        std::vector<float> ba(2*a.size()),bb(2*b.size());
        std::copy(a.begin(),a.end(),ba.begin());std::copy(a.begin(),a.end(),ba.begin()+a.size());
        std::copy(b.begin(),b.end(),bb.begin());std::copy(b.begin(),b.end(),bb.begin()+b.size());
        out.resize(2*size_t(m)*n);
        measure("batch_matmul",[&]{return engine.batchMatMulForward(ba.data(),bb.data(),out.data(),2,m,k,n);});
    }
    if(!values)throw std::runtime_error("OpenCL value dump failed");
#ifdef MIMIR_VERIFY_OPENCL_CACHE
    std::cout<<"OPENCL_REUSE_OK allocations="<<engine.bufferAllocationCount()<<" cached_bytes="<<engine.cachedBufferBytes()<<'\n';
    engine.cleanup();
    if(engine.cachedBufferBytes()!=0)throw std::runtime_error("OpenCL cleanup retained device buffers");
    if(!engine.initialize())throw std::runtime_error("OpenCL context restart failed");
    float input[3]={-1.f,0.f,1.f},output[3]={};
    if(!engine.unaryForward(input,output,3,0,.01f)||output[0]!=0.f||output[2]!=1.f)
        throw std::runtime_error("OpenCL restart reused stale context storage");
#endif
#endif
}
