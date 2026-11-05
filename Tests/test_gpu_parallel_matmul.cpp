#include "test_utils.hpp"
#include "Layers.hpp"
#include "runtimes/cuda/CudaRuntime.hpp"
#include "runtimes/rocm/RocmRuntime.hpp"
#include "runtimes/opencl/OpenCLRuntime.hpp"
#include "runtimes/vulkan/VulkanRuntime.hpp"
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

static int check(AbstractRuntime& runtime, int batches, int m, int k, int n) {
    Layer layer;
    layer.type_enum = batches == 1 ? LayerType::MatMul : LayerType::BatchMatMul;
    layer.seq_len = batches;
    layer.in_features = m;
    layer.out_features = k;
    layer.embed_dim = n;
    std::vector<float> a(size_t(batches)*m*k), b(size_t(batches)*k*n);
    for (size_t i=0; i<a.size(); ++i) a[i]=float(int(i%13)-6)*.125f;
    for (size_t i=0; i<b.size(); ++i) b[i]=float(int(i%17)-8)*.0625f;
    const std::vector<const std::vector<float>*> inputs{&a,&b};
    TASSERT_TRUE(runtimeCapabilityIsNative(runtime.queryForwardOperationCapability(layer, inputs, false)));
    std::vector<std::vector<float>> result;
    TASSERT_TRUE(runtime.forwardLayer(inputs, result, layer, false));
    TASSERT_TRUE(result.size()==1 && result[0].size()==size_t(batches)*m*n);
    for (int z=0; z<batches; ++z) for (int row=0; row<m; ++row) for (int col=0; col<n; ++col) {
        double expected=0;
        for (int f=0; f<k; ++f) expected+=double(a[(size_t(z)*m+row)*k+f])*b[(size_t(z)*k+f)*n+col];
        TASSERT_NEAR(result[0][(size_t(z)*m+row)*n+col],expected,1e-5);
    }
    // Each batch has different operands; compare all three gradient axes too.
    std::vector<float> dy(size_t(batches)*m*n, .125f);
    const std::vector<const std::vector<float>*> grads{&dy};
    TASSERT_TRUE(runtimeCapabilityIsNative(runtime.queryBackwardOperationCapability(layer, inputs, grads, false)));
    {
        std::vector<std::vector<float>> dx;
        TASSERT_TRUE(runtime.backwardLayer(inputs, grads, dx, layer, false));
        TASSERT_TRUE(dx.size()==2 && dx[0].size()==a.size() && dx[1].size()==b.size());
        for (int z=0; z<batches; ++z) {
            for (int row=0; row<m; ++row) for (int f=0; f<k; ++f) {
                double expected=0;
                for (int col=0; col<n; ++col) expected+=double(dy[(size_t(z)*m+row)*n+col])*b[(size_t(z)*k+f)*n+col];
                TASSERT_NEAR(dx[0][(size_t(z)*m+row)*k+f],expected,1e-5);
            }
            for (int f=0; f<k; ++f) for (int col=0; col<n; ++col) {
                double expected=0;
                for (int row=0; row<m; ++row) expected+=double(a[(size_t(z)*m+row)*k+f])*dy[(size_t(z)*m+row)*n+col];
                TASSERT_NEAR(dx[1][(size_t(z)*k+f)*n+col],expected,1e-5);
            }
        }
    }
    dy.pop_back();
    std::vector<std::vector<float>> invalid_gradients;
    TASSERT_TRUE(!runtime.backwardLayer(inputs, grads, invalid_gradients, layer, false));
    b.pop_back();
    TASSERT_TRUE(!runtime.forwardLayer(inputs, result, layer, false));
    return 0;
}
int main(int argc,char** argv) {
    try {
        const std::string backend=argc>1?argv[1]:"VULKAN";
        std::unique_ptr<AbstractRuntime> runtime;
#ifdef ENABLE_VULKAN
        if (backend=="VULKAN") runtime=std::make_unique<VulkanRuntime>();
#endif
#ifdef ENABLE_OPENCL
        if (backend=="OPENCL") runtime=std::make_unique<OpenCLRuntime>();
#endif
#ifdef ENABLE_CUDA
        if (backend=="CUDA") runtime=std::make_unique<CudaRuntime>();
#endif
#ifdef ENABLE_ROCM
        if (backend=="ROCM") runtime=std::make_unique<RocmRuntime>();
#endif
        if (!runtime) return 77;
        RuntimeConfig config;
        config.linear_enabled=true;
        config.linear_min_ops=0;
        if (!runtime->initialize(config)) return 77;
        TASSERT_TRUE(check(*runtime,1,1,7,65533)==0);
        TASSERT_TRUE(check(*runtime,8,1,7,1027)==0);
        TASSERT_TRUE(check(*runtime,3,33,17,1)==0);
        TASSERT_TRUE(check(*runtime,2,15,19,17)==0);
        TASSERT_TRUE(check(*runtime,2,16,19,17)==0);
        TASSERT_TRUE(check(*runtime,3,17,23,33)==0);
        std::cout<<"GPU_PARALLEL_MATMUL_OK "<<backend<<'\n';
    } catch (const std::exception& error) {
        std::cerr<<error.what()<<'\n';
        return 1;
    }
}
