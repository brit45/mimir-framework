#include "runtimes/rocm/RocmRuntime.hpp"

#include "runtimes/cpu/RuntimeLayerDispatch.hpp"

#ifdef ENABLE_ROCM
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#endif

#include <cstddef>
#include <new>
#include <vector>

struct RocmRuntime::Impl {
#ifdef ENABLE_ROCM
    rocblas_handle handle = nullptr;

    struct DeviceBuf {
        void* ptr = nullptr;
        size_t bytes = 0;

        ~DeviceBuf() { reset(); }
        DeviceBuf() = default;
        DeviceBuf(const DeviceBuf&) = delete;
        DeviceBuf& operator=(const DeviceBuf&) = delete;

        void reset() {
            if (ptr) {
                hipFree(ptr);
                ptr = nullptr;
                bytes = 0;
            }
        }

        bool alloc(size_t nbytes) {
            reset();
            if (nbytes == 0) return true;
            if (hipMalloc(&ptr, nbytes) != hipSuccess) {
                ptr = nullptr;
                bytes = 0;
                return false;
            }
            bytes = nbytes;
            return true;
        }

        bool copyFromHost(const void* src, size_t nbytes) {
            if (!ptr || nbytes > bytes) return false;
            return hipMemcpy(ptr, src, nbytes, hipMemcpyHostToDevice) == hipSuccess;
        }

        bool copyToHost(void* dst, size_t nbytes) {
            if (!ptr || nbytes > bytes) return false;
            return hipMemcpy(dst, ptr, nbytes, hipMemcpyDeviceToHost) == hipSuccess;
        }
    };

    struct CachedBuf {
        DeviceBuf buf;
        size_t count = 0;

        bool ensure(size_t n) {
            if (count == n && buf.ptr) return true;
            if (!buf.alloc(n * sizeof(float))) return false;
            count = n;
            return true;
        }
    };

    CachedBuf device_bias;
    CachedBuf device_ones;
    std::vector<float> host_ones;
#endif

    bool initialized = false;
};

bool RocmRuntime::initialize(const RuntimeConfig& cfg) {
    config_ = cfg;

#ifndef ENABLE_ROCM
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

    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count <= 0) {
        shutdown();
        return false;
    }

    int device_index = config_.device_index;
    if (device_index < 0 || device_index >= device_count) {
        device_index = 0;
    }

    if (hipSetDevice(device_index) != hipSuccess) {
        shutdown();
        return false;
    }

    if (rocblas_create_handle(&impl_->handle) != rocblas_status_success) {
        shutdown();
        return false;
    }

    impl_->initialized = true;
    return true;
#endif
}

void RocmRuntime::shutdown() {
#ifdef ENABLE_ROCM
    if (impl_) {
        if (impl_->handle) {
            rocblas_destroy_handle(impl_->handle);
            impl_->handle = nullptr;
        }
        impl_->device_bias.buf.reset();
        impl_->device_ones.buf.reset();
        impl_->host_ones.clear();
        impl_->initialized = false;
    }
#endif

    delete impl_;
    impl_ = nullptr;
}

bool RocmRuntime::isInitialized() const {
    return impl_ && impl_->initialized;
}

bool RocmRuntime::linearForward(
    const float* input,
    const float* weights,
    const float* bias_or_null,
    float* output,
    int batch,
    int in_f,
    int out_f
) {
#ifndef ENABLE_ROCM
    (void)input; (void)weights; (void)bias_or_null; (void)output; (void)batch; (void)in_f; (void)out_f;
    return false;
#else
    if (!isInitialized() || !impl_->handle) return false;
    if (!input || !weights || !output) return false;
    if (batch <= 0 || in_f <= 0 || out_f <= 0) return false;

    const size_t bytes_in = static_cast<size_t>(batch) * static_cast<size_t>(in_f) * sizeof(float);
    const size_t bytes_w = static_cast<size_t>(out_f) * static_cast<size_t>(in_f) * sizeof(float);
    const size_t bytes_out = static_cast<size_t>(batch) * static_cast<size_t>(out_f) * sizeof(float);

    Impl::DeviceBuf d_in;
    Impl::DeviceBuf d_w;
    Impl::DeviceBuf d_out;

    if (!d_in.alloc(bytes_in)) return false;
    if (!d_w.alloc(bytes_w)) return false;
    if (!d_out.alloc(bytes_out)) return false;

    if (!d_in.copyFromHost(input, bytes_in)) return false;
    if (!d_w.copyFromHost(weights, bytes_w)) return false;

    // rocBLAS column-major: C(out_f x batch) = A^T(out_f x in_f) * B(in_f x batch)
    const float alpha = 1.0f;
    const float beta0 = 0.0f;

    if (rocblas_sgemm(
            impl_->handle,
            rocblas_operation_transpose,
            rocblas_operation_none,
            out_f,
            batch,
            in_f,
            &alpha,
            reinterpret_cast<const float*>(d_w.ptr),
            in_f,
            reinterpret_cast<const float*>(d_in.ptr),
            in_f,
            &beta0,
            reinterpret_cast<float*>(d_out.ptr),
            out_f
        ) != rocblas_status_success) {
        return false;
    }

    if (bias_or_null) {
        if (!impl_->device_bias.ensure(static_cast<size_t>(out_f))) return false;
        if (!impl_->device_bias.buf.copyFromHost(bias_or_null, static_cast<size_t>(out_f) * sizeof(float))) return false;

        if (!impl_->device_ones.ensure(static_cast<size_t>(batch))) return false;
        if (static_cast<int>(impl_->host_ones.size()) != batch) {
            impl_->host_ones.assign(static_cast<size_t>(batch), 1.0f);
        }
        if (!impl_->device_ones.buf.copyFromHost(impl_->host_ones.data(), static_cast<size_t>(batch) * sizeof(float))) return false;

        const float beta1 = 1.0f;
        const int k = 1;
        if (rocblas_sgemm(
                impl_->handle,
                rocblas_operation_none,
                rocblas_operation_none,
                out_f,
                batch,
                k,
                &alpha,
                reinterpret_cast<const float*>(impl_->device_bias.buf.ptr),
                out_f,
                reinterpret_cast<const float*>(impl_->device_ones.buf.ptr),
                1,
                &beta1,
                reinterpret_cast<float*>(d_out.ptr),
                out_f
            ) != rocblas_status_success) {
            return false;
        }
    }

    if (hipDeviceSynchronize() != hipSuccess) {
        return false;
    }

    return d_out.copyToHost(output, bytes_out);
#endif
}

bool RocmRuntime::forwardLayer(
    const std::vector<const std::vector<float>*>& inputs,
    std::vector<std::vector<float>>& outputs,
    const Layer& layer,
    bool training
) {
    if (!isInitialized()) return false;
    if (config_.disabled) return false;

    if (layer.type_enum == LayerType::Linear && config_.linear_enabled && !training) {
        if (!inputs.empty() && inputs[0]) {
            const std::vector<float>& x = *inputs[0];
            const int in_f = layer.in_features;
            const int out_f = layer.out_features;
            const int seq_len = layer.seq_len;
            const int batch = (seq_len > 0) ? seq_len : (in_f > 0 ? static_cast<int>(x.size()) / in_f : 0);

            if (batch > 0 && in_f > 0 && out_f > 0) {
                const size_t expected_in = static_cast<size_t>(batch) * static_cast<size_t>(in_f);
                const size_t expected_w = static_cast<size_t>(out_f) * static_cast<size_t>(in_f);
                const size_t expected_total_w = expected_w + (layer.use_bias ? static_cast<size_t>(out_f) : 0ULL);

                const long long ops = static_cast<long long>(batch) * static_cast<long long>(in_f) * static_cast<long long>(out_f);
                if (ops >= std::max(0, config_.linear_min_ops) && x.size() == expected_in) {
                    const float* w = layer.getWeights();
                    if (w && layer.getWeightsSize() >= expected_total_w) {
                        const float* b = layer.use_bias ? (w + expected_w) : nullptr;
                        outputs.resize(1);
                        outputs[0].assign(static_cast<size_t>(batch) * static_cast<size_t>(out_f), 0.0f);
                        if (linearForward(x.data(), w, b, outputs[0].data(), batch, in_f, out_f)) {
                            return true;
                        }
                    }
                }
            }
        }
    }

    return RuntimeLayerDispatch::cpu_forward_layer(inputs, outputs, layer, training);
}
