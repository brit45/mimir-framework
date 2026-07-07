#include "runtimes/opencl/OpenCLRuntime.hpp"

#ifdef ENABLE_OPENCL
#include "runtimes/opencl/OpenCLCompute.hpp"
#endif

#include "Layers.hpp"

#include <algorithm>
#include <cstddef>
#include <new>
#include <vector>

struct OpenCLRuntime::Impl {
#ifdef ENABLE_OPENCL
    OpenCLCompute::ComputeEngine engine;
#endif
    bool initialized = false;
};

bool OpenCLRuntime::initialize(const RuntimeConfig& cfg) {
    config_ = cfg;

#ifndef ENABLE_OPENCL
    (void)cfg;
    shutdown();
    return false;
#else
    if (impl_ && impl_->initialized) return true;

    if (config_.disabled) {
        shutdown();
        return false;
    }

    if (!impl_) {
        impl_ = new (std::nothrow) Impl();
        if (!impl_) return false;
    }

    impl_->initialized = impl_->engine.initialize();
    if (!impl_->initialized) {
        shutdown();
        return false;
    }

    return true;
#endif
}

void OpenCLRuntime::shutdown() {
#ifdef ENABLE_OPENCL
    if (impl_) {
        impl_->engine.cleanup();
        impl_->initialized = false;
    }
#endif
    delete impl_;
    impl_ = nullptr;
}

bool OpenCLRuntime::isInitialized() const {
    return impl_ && impl_->initialized;
}

bool OpenCLRuntime::linearForward(
    const float* input,
    const float* weights,
    const float* bias_or_null,
    float* output,
    int batch,
    int in_f,
    int out_f
) {
#ifndef ENABLE_OPENCL
    (void)input; (void)weights; (void)bias_or_null; (void)output; (void)batch; (void)in_f; (void)out_f;
    return false;
#else
    if (!isInitialized() || !impl_) return false;
    return impl_->engine.linearForward(input, weights, bias_or_null, output, batch, in_f, out_f);
#endif
}

bool OpenCLRuntime::forwardLayer(
    const std::vector<const std::vector<float>*>& inputs,
    std::vector<std::vector<float>>& outputs,
    const Layer& layer,
    bool training
) {
    (void)training;
#ifndef ENABLE_OPENCL
    (void)inputs; (void)outputs; (void)layer;
    return false;
#else
    if (!isInitialized() || !impl_) return false;
    if (config_.disabled || !config_.linear_enabled) return false;

    switch (layer.type_enum) {
        case LayerType::Linear: {
            if (inputs.empty() || !inputs[0]) return false;
            const std::vector<float>& x = *inputs[0];

            const int in_f = layer.in_features;
            const int out_f = layer.out_features;
            const int seq_len = layer.seq_len;
            const int batch = (seq_len > 0) ? seq_len : (in_f > 0 ? static_cast<int>(x.size()) / in_f : 0);
            if (batch <= 0 || in_f <= 0 || out_f <= 0) return false;

            const size_t expected_in = static_cast<size_t>(batch) * static_cast<size_t>(in_f);
            const size_t expected_w = static_cast<size_t>(out_f) * static_cast<size_t>(in_f);
            const size_t expected_total_w = expected_w + (layer.use_bias ? static_cast<size_t>(out_f) : 0ULL);
            const long long ops = static_cast<long long>(batch) * static_cast<long long>(in_f) * static_cast<long long>(out_f);

            if (ops < std::max(0, config_.linear_min_ops) || x.size() != expected_in) return false;
            const float* w = layer.getWeights();
            if (!w || layer.getWeightsSize() < expected_total_w) return false;

            const float* b = layer.use_bias ? (w + expected_w) : nullptr;
            outputs.resize(1);
            outputs[0].assign(static_cast<size_t>(batch) * static_cast<size_t>(out_f), 0.0f);
            return linearForward(x.data(), w, b, outputs[0].data(), batch, in_f, out_f);
        }
        case LayerType::MatMul: {
            if (inputs.size() < 2 || !inputs[0] || !inputs[1]) return false;
            const std::vector<float>& A = *inputs[0];
            const std::vector<float>& B = *inputs[1];

            const int M = layer.in_features;
            const int K = layer.out_features;
            const int N = layer.embed_dim;
            if (M <= 0 || K <= 0 || N <= 0) return false;

            if (A.size() != static_cast<size_t>(M) * static_cast<size_t>(K)) return false;
            if (B.size() != static_cast<size_t>(K) * static_cast<size_t>(N)) return false;

            const long long ops = static_cast<long long>(M) * static_cast<long long>(K) * static_cast<long long>(N);
            if (ops < std::max(0, config_.linear_min_ops)) return false;

            outputs.resize(1);
            outputs[0].assign(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
            return impl_->engine.matmulForward(A.data(), B.data(), outputs[0].data(), M, K, N);
        }
        case LayerType::BatchMatMul: {
            if (inputs.size() < 2 || !inputs[0] || !inputs[1]) return false;
            const std::vector<float>& A = *inputs[0];
            const std::vector<float>& B = *inputs[1];

            const int batches = layer.seq_len;
            const int M = layer.in_features;
            const int K = layer.out_features;
            const int N = layer.embed_dim;
            if (batches <= 0 || M <= 0 || K <= 0 || N <= 0) return false;

            if (A.size() != static_cast<size_t>(batches) * static_cast<size_t>(M) * static_cast<size_t>(K)) return false;
            if (B.size() != static_cast<size_t>(batches) * static_cast<size_t>(K) * static_cast<size_t>(N)) return false;

            const long long ops = static_cast<long long>(batches) * static_cast<long long>(M) * static_cast<long long>(K) * static_cast<long long>(N);
            if (ops < std::max(0, config_.linear_min_ops)) return false;

            outputs.resize(1);
            outputs[0].assign(static_cast<size_t>(batches) * static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
            return impl_->engine.batchMatMulForward(A.data(), B.data(), outputs[0].data(), batches, M, K, N);
        }
        default:
            return false;
    }
#endif
}
