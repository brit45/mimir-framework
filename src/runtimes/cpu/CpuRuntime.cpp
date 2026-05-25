#include "runtimes/cpu/CpuRuntime.hpp"

#include "SIMD_Ops.hpp"
#include "runtimes/cpu/RuntimeLayerDispatch.hpp"

#include <cstddef>

bool CpuRuntime::initialize(const RuntimeConfig& cfg) {
    // CPU est toujours disponible; on conserve cfg pour homogénéité.
    config_ = cfg;
    initialized_ = true;
    return true;
}

void CpuRuntime::shutdown() {
    initialized_ = false;
}

bool CpuRuntime::linearForward(
    const float* input,
    const float* weights,
    const float* bias_or_null,
    float* output,
    int batch,
    int in_f,
    int out_f
) {
    if (!initialized_) return false;
    if (!input || !weights || !output) return false;
    if (batch <= 0 || in_f <= 0 || out_f <= 0) return false;

    // output[batch x out_f] = input[batch x in_f] @ weights[out_f x in_f]^T
    SIMD::matmul_transpose_avx2(
        output,
        input,
        weights,
        static_cast<size_t>(batch),
        static_cast<size_t>(out_f),
        static_cast<size_t>(in_f)
    );

    if (bias_or_null) {
        for (int b = 0; b < batch; ++b) {
            float* row = output + static_cast<size_t>(b) * static_cast<size_t>(out_f);
            for (int o = 0; o < out_f; ++o) {
                row[o] += bias_or_null[o];
            }
        }
    }

    return true;
}

bool CpuRuntime::forwardLayer(
    const std::vector<const std::vector<float>*>& inputs,
    std::vector<std::vector<float>>& outputs,
    const Layer& layer,
    bool training
) {
    if (!initialized_) return false;
    return RuntimeLayerDispatch::cpu_forward_layer(inputs, outputs, layer, training);
}