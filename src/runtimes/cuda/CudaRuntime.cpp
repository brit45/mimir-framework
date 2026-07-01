#include "runtimes/cuda/CudaRuntime.hpp"

#include "runtimes/cpu/RuntimeLayerDispatch.hpp"

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <new>
#include <vector>

#ifdef ENABLE_CUDA
// ─────────────────────────────────────────────────────────────────────────────
// Im2col: transforms input [C, H, W] → col [C*k*k, H_out*W_out] (row-major)
// ─────────────────────────────────────────────────────────────────────────────
static void im2col_cpu(
    const float* __restrict__ src, float* __restrict__ col,
    int C, int H, int W,
    int k, int stride, int pad, int dil,
    int H_out, int W_out)
{
    const int HW = H_out * W_out;
    for (int c = 0; c < C; ++c) {
        const float* src_c = src + c * H * W;
        for (int kh = 0; kh < k; ++kh) {
            for (int kw = 0; kw < k; ++kw) {
                float* col_row = col + (c * k * k + kh * k + kw) * HW;
                for (int oh = 0; oh < H_out; ++oh) {
                    const int ih = oh * stride + kh * dil - pad;
                    const bool ih_ok = (ih >= 0 && ih < H);
                    for (int ow = 0; ow < W_out; ++ow) {
                        const int iw = ow * stride + kw * dil - pad;
                        col_row[oh * W_out + ow] =
                            (ih_ok && iw >= 0 && iw < W) ? src_c[ih * W + iw] : 0.0f;
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Row-wise in-place softmax over A[rows, cols]
// ─────────────────────────────────────────────────────────────────────────────
static void softmax_rows(float* A, int rows, int cols)
{
    for (int i = 0; i < rows; ++i) {
        float* row = A + i * cols;
        float mx = row[0];
        for (int j = 1; j < cols; ++j) if (row[j] > mx) mx = row[j];
        float s = 0.0f;
        for (int j = 0; j < cols; ++j) { row[j] = std::exp(row[j] - mx); s += row[j]; }
        const float inv = 1.0f / (s > 0.0f ? s : 1.0f);
        for (int j = 0; j < cols; ++j) row[j] *= inv;
    }
}
#endif // ENABLE_CUDA

struct CudaRuntime::Impl {
#ifdef ENABLE_CUDA
    cublasHandle_t handle = nullptr;

    struct DeviceBuf {
        void* ptr = nullptr;
        size_t bytes = 0;

        ~DeviceBuf() { reset(); }
        DeviceBuf() = default;
        DeviceBuf(const DeviceBuf&) = delete;
        DeviceBuf& operator=(const DeviceBuf&) = delete;

        void reset() {
            if (ptr) {
                cudaFree(ptr);
                ptr = nullptr;
                bytes = 0;
            }
        }

        bool alloc(size_t nbytes) {
            reset();
            if (nbytes == 0) return true;
            if (cudaMalloc(&ptr, nbytes) != cudaSuccess) {
                ptr = nullptr;
                bytes = 0;
                return false;
            }
            bytes = nbytes;
            return true;
        }

        bool copyFromHost(const void* src, size_t nbytes) {
            if (!ptr || nbytes > bytes) return false;
            return cudaMemcpy(ptr, src, nbytes, cudaMemcpyHostToDevice) == cudaSuccess;
        }

        bool copyToHost(void* dst, size_t nbytes) {
            if (!ptr || nbytes > bytes) return false;
            return cudaMemcpy(dst, ptr, nbytes, cudaMemcpyDeviceToHost) == cudaSuccess;
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

bool CudaRuntime::initialize(const RuntimeConfig& cfg) {
    config_ = cfg;

#ifndef ENABLE_CUDA
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
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        shutdown();
        return false;
    }

    int device_index = config_.device_index;
    if (device_index < 0 || device_index >= device_count) {
        device_index = 0;
    }

    if (cudaSetDevice(device_index) != cudaSuccess) {
        shutdown();
        return false;
    }

    if (cublasCreate(&impl_->handle) != CUBLAS_STATUS_SUCCESS) {
        shutdown();
        return false;
    }

    (void)cublasSetStream(impl_->handle, nullptr);

    impl_->initialized = true;
    return true;
#endif
}

void CudaRuntime::shutdown() {
#ifdef ENABLE_CUDA
    if (impl_) {
        if (impl_->handle) {
            cublasDestroy(impl_->handle);
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

bool CudaRuntime::isInitialized() const {
    return impl_ && impl_->initialized;
}

bool CudaRuntime::linearForward(
    const float* input,
    const float* weights,
    const float* bias_or_null,
    float* output,
    int batch,
    int in_f,
    int out_f
) {
#ifndef ENABLE_CUDA
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

    // GEMM 1: output = input * weights^T
    // cublas column-major: C(out_f x batch) = A^T(out_f x in_f) * B(in_f x batch)
    const float alpha = 1.0f;
    const float beta0 = 0.0f;

    if (cublasSgemm(
            impl_->handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
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
        ) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }

    if (bias_or_null) {
        // output += bias[:,1] * ones[1,batch]
        if (!impl_->device_bias.ensure(static_cast<size_t>(out_f))) return false;
        if (!impl_->device_bias.buf.copyFromHost(bias_or_null, static_cast<size_t>(out_f) * sizeof(float))) return false;

        if (!impl_->device_ones.ensure(static_cast<size_t>(batch))) return false;
        if (static_cast<int>(impl_->host_ones.size()) != batch) {
            impl_->host_ones.assign(static_cast<size_t>(batch), 1.0f);
        }
        if (!impl_->device_ones.buf.copyFromHost(impl_->host_ones.data(), static_cast<size_t>(batch) * sizeof(float))) return false;

        const float beta1 = 1.0f;
        const int k = 1;
        if (cublasSgemm(
                impl_->handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
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
            ) != CUBLAS_STATUS_SUCCESS) {
            return false;
        }
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        return false;
    }

    return d_out.copyToHost(output, bytes_out);
#endif
}

bool CudaRuntime::forwardLayer(
    const std::vector<const std::vector<float>*>& inputs,
    std::vector<std::vector<float>>& outputs,
    const Layer& layer,
    bool training
) {
    if (!isInitialized()) return false;
    if (config_.disabled) return false;

    // Fast-path: Linear sur GPU (si activé et non-training).
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

#ifdef ENABLE_CUDA
    // ─────────────────────────────────────────────────────────────────────────
    // Fast-path: Conv2d  (im2col + cuBLAS SGEMM)
    // ─────────────────────────────────────────────────────────────────────────
    if (layer.type_enum == LayerType::Conv2d && config_.conv_enabled && !training) {
        do {
            if (inputs.empty() || !inputs[0]) break;
            const std::vector<float>& xc = *inputs[0];
            const int C_in  = std::max(1, layer.in_channels);
            const int C_out = std::max(1, layer.out_channels);
            const int k     = std::max(1, layer.kernel_h > 0 ? layer.kernel_h : layer.get_kernel_h());
            const int str   = std::max(1, layer.stride_h > 0 ? layer.stride_h : layer.get_stride_h());
            const int pad   = std::max(0, layer.pad_h >= 0 ? layer.pad_h : layer.get_pad_h());
            const int dil   = std::max(1, layer.dilation_h > 0 ? layer.dilation_h : 1);
            const int H     = std::max(1, layer.input_height);
            const int W     = std::max(1, layer.input_width);
            const int H_out = (H + 2*pad - dil*(k-1) - 1) / str + 1;
            const int W_out = (W + 2*pad - dil*(k-1) - 1) / str + 1;
            if (H_out <= 0 || W_out <= 0) break;

            const int K  = C_in * k * k;
            const int HW = H_out * W_out;
            const long long ops = static_cast<long long>(C_out) * K * HW;
            if (ops < static_cast<long long>(config_.conv_min_ops)) break;
            if (static_cast<int>(xc.size()) != C_in * H * W) break;

            const size_t w_k = static_cast<size_t>(C_out) * K;
            const float* wc = layer.getWeights();
            if (!wc || layer.getWeightsSize() < w_k + (layer.use_bias ? C_out : 0u)) break;

            // Im2col on host: col[K, HW]
            const size_t col_sz = static_cast<size_t>(K) * HW;
            std::vector<float> col_buf(col_sz);
            im2col_cpu(xc.data(), col_buf.data(), C_in, H, W, k, str, pad, dil, H_out, W_out);

            Impl::DeviceBuf d_col, d_w, d_out;
            if (!d_col.alloc(col_sz * sizeof(float))) break;
            if (!d_w.alloc(w_k * sizeof(float))) break;
            if (!d_out.alloc(static_cast<size_t>(C_out) * HW * sizeof(float))) break;
            if (!d_col.copyFromHost(col_buf.data(), col_sz * sizeof(float))) break;
            if (!d_w.copyFromHost(wc, w_k * sizeof(float))) break;

            // SGEMM: output_rm[C_out,HW] = weights_rm[C_out,K] × col_rm[K,HW]
            // cuBLAS (col-major): C[HW,C_out] = B[HW,K] × A[K,C_out]
            // → cublasSgemm(OP_N,OP_N, HW, C_out, K, 1, col, HW, w, K, 0, out, HW)
            const float cv_a = 1.0f, cv_b0 = 0.0f;
            if (cublasSgemm(impl_->handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    HW, C_out, K, &cv_a,
                    reinterpret_cast<const float*>(d_col.ptr), HW,
                    reinterpret_cast<const float*>(d_w.ptr), K,
                    &cv_b0,
                    reinterpret_cast<float*>(d_out.ptr), HW) != CUBLAS_STATUS_SUCCESS) break;

            // Bias: d_out_cm[HW,C_out] += ones_HW[HW] × bias[C_out]^T  (cublasSger)
            if (layer.use_bias) {
                std::vector<float> ones_hw(static_cast<size_t>(HW), 1.0f);
                Impl::DeviceBuf d_bias, d_ones_hw;
                if (!d_bias.alloc(static_cast<size_t>(C_out) * sizeof(float))) break;
                if (!d_ones_hw.alloc(static_cast<size_t>(HW) * sizeof(float))) break;
                if (!d_bias.copyFromHost(wc + w_k, static_cast<size_t>(C_out) * sizeof(float))) break;
                if (!d_ones_hw.copyFromHost(ones_hw.data(), static_cast<size_t>(HW) * sizeof(float))) break;
                if (cublasSger(impl_->handle, HW, C_out, &cv_a,
                        reinterpret_cast<const float*>(d_ones_hw.ptr), 1,
                        reinterpret_cast<const float*>(d_bias.ptr), 1,
                        reinterpret_cast<float*>(d_out.ptr), HW) != CUBLAS_STATUS_SUCCESS) break;
            }

            if (cudaDeviceSynchronize() != cudaSuccess) break;
            outputs.resize(1);
            outputs[0].resize(static_cast<size_t>(C_out) * HW);
            if (!d_out.copyToHost(outputs[0].data(), static_cast<size_t>(C_out) * HW * sizeof(float))) break;
            return true;
        } while (false);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Fast-path: LayerNorm / RMSNorm  (stats sur host, affine sur GPU)
    // ─────────────────────────────────────────────────────────────────────────
    const bool is_norm = (layer.type_enum == LayerType::LayerNorm ||
                          layer.type_enum == LayerType::RMSNorm);
    if (is_norm && config_.norm_enabled) {
        do {
            if (inputs.empty() || !inputs[0]) break;
            const std::vector<float>& xn = *inputs[0];
            const int N = static_cast<int>(xn.size());
            if (N < config_.norm_min_elements) break;
            if (!layer.affine) break;

            const int norm_sz = (layer.in_features > 0) ? layer.in_features : N;
            if (norm_sz <= 0 || N % norm_sz != 0) break;
            const int groups = N / norm_sz;

            const float* gw = layer.getWeights();
            if (!gw || static_cast<int>(layer.getWeightsSize()) < norm_sz) break;

            // Normalisation sur host
            const float eps = (layer.eps > 0.0f) ? layer.eps : 1e-5f;
            std::vector<float> x_hat(static_cast<size_t>(N));

            if (layer.type_enum == LayerType::RMSNorm) {
                for (int g = 0; g < groups; ++g) {
                    const int base = g * norm_sz;
                    float ss = 0.0f;
                    for (int i = 0; i < norm_sz; ++i) { float v = xn[base+i]; ss += v*v; }
                    const float inv = 1.0f / std::sqrt(ss / norm_sz + eps);
                    for (int i = 0; i < norm_sz; ++i) x_hat[base+i] = xn[base+i] * inv;
                }
            } else { // LayerNorm
                for (int g = 0; g < groups; ++g) {
                    const int base = g * norm_sz;
                    float mean = 0.0f;
                    for (int i = 0; i < norm_sz; ++i) mean += xn[base+i];
                    mean /= norm_sz;
                    float var = 0.0f;
                    for (int i = 0; i < norm_sz; ++i) { float d = xn[base+i]-mean; var += d*d; }
                    const float inv = 1.0f / std::sqrt(var/norm_sz + eps);
                    for (int i = 0; i < norm_sz; ++i) x_hat[base+i] = (xn[base+i]-mean)*inv;
                }
            }

            // Affine sur GPU: y = gamma ⊙ x_hat  (+beta si use_bias)
            Impl::DeviceBuf d_xh, d_gamma, d_out;
            if (!d_xh.alloc(static_cast<size_t>(N) * sizeof(float))) break;
            if (!d_gamma.alloc(static_cast<size_t>(norm_sz) * sizeof(float))) break;
            if (!d_out.alloc(static_cast<size_t>(N) * sizeof(float))) break;
            if (!d_xh.copyFromHost(x_hat.data(), static_cast<size_t>(N) * sizeof(float))) break;
            if (!d_gamma.copyFromHost(gw, static_cast<size_t>(norm_sz) * sizeof(float))) break;

            bool ok = true;
            for (int g = 0; g < groups && ok; ++g) {
                const int base = g * norm_sz;
                if (cublasSdgmm(impl_->handle, CUBLAS_SIDE_LEFT,
                        norm_sz, 1,
                        reinterpret_cast<const float*>(d_xh.ptr) + base, norm_sz,
                        reinterpret_cast<const float*>(d_gamma.ptr), 1,
                        reinterpret_cast<float*>(d_out.ptr) + base, norm_sz
                    ) != CUBLAS_STATUS_SUCCESS) ok = false;
            }
            if (!ok) break;

            if (layer.use_bias && static_cast<int>(layer.getWeightsSize()) >= 2 * norm_sz) {
                Impl::DeviceBuf d_beta;
                if (!d_beta.alloc(static_cast<size_t>(norm_sz) * sizeof(float))) break;
                if (!d_beta.copyFromHost(gw + norm_sz, static_cast<size_t>(norm_sz) * sizeof(float))) break;
                const float na = 1.0f;
                for (int g = 0; g < groups && ok; ++g) {
                    const int base = g * norm_sz;
                    if (cublasSaxpy(impl_->handle, norm_sz, &na,
                            reinterpret_cast<const float*>(d_beta.ptr), 1,
                            reinterpret_cast<float*>(d_out.ptr) + base, 1
                        ) != CUBLAS_STATUS_SUCCESS) ok = false;
                }
                if (!ok) break;
            }

            if (cudaDeviceSynchronize() != cudaSuccess) break;
            outputs.resize(1);
            outputs[0].resize(static_cast<size_t>(N));
            if (!d_out.copyToHost(outputs[0].data(), static_cast<size_t>(N) * sizeof(float))) break;
            return true;
        } while (false);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Fast-path: SelfAttention / MultiHeadAttention
    // ─────────────────────────────────────────────────────────────────────────
    if ((layer.type_enum == LayerType::SelfAttention ||
         layer.type_enum == LayerType::MultiHeadAttention) &&
        config_.attention_enabled && !training)
    {
        do {
            if (inputs.empty() || !inputs[0]) break;
            const std::vector<float>& xa = *inputs[0];

            const int seq = layer.seq_len > 0 ? layer.seq_len : 1;
            const int ed  = layer.embed_dim > 0 ? layer.embed_dim : static_cast<int>(xa.size());
            const int nh  = layer.num_heads > 0 ? layer.num_heads : 1;
            if (ed <= 0 || nh <= 0 || ed % nh != 0) break;
            const int hd = ed / nh;
            const long long attn_ops = 4LL * seq * ed * ed;
            if (attn_ops < static_cast<long long>(config_.attention_min_ops)) break;
            if (static_cast<int>(xa.size()) != seq * ed) break;

            const float* wa = layer.getWeights();
            if (!wa || static_cast<int>(layer.getWeightsSize()) < ed * ed * 4) break;
            const int qkv_d = ed * 3;
            const float* wqkv = wa;
            const float* wout = wa + ed * ed * 3;

            const float aa = 1.0f, ab0 = 0.0f;

            // Step 1: QKV[seq, 3*ed] = X[seq,ed] × W_QKV[ed,3*ed]
            // cublasSgemm(OP_N,OP_N, qkv_d, seq, ed, 1, wqkv, qkv_d, x, ed, 0, qkv, qkv_d)
            Impl::DeviceBuf d_xa, d_wqkv, d_qkv;
            if (!d_xa.alloc(static_cast<size_t>(seq) * ed * sizeof(float))) break;
            if (!d_wqkv.alloc(static_cast<size_t>(ed) * qkv_d * sizeof(float))) break;
            if (!d_qkv.alloc(static_cast<size_t>(seq) * qkv_d * sizeof(float))) break;
            if (!d_xa.copyFromHost(xa.data(), static_cast<size_t>(seq) * ed * sizeof(float))) break;
            if (!d_wqkv.copyFromHost(wqkv, static_cast<size_t>(ed) * qkv_d * sizeof(float))) break;

            if (cublasSgemm(impl_->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    qkv_d, seq, ed, &aa,
                    reinterpret_cast<const float*>(d_wqkv.ptr), qkv_d,
                    reinterpret_cast<const float*>(d_xa.ptr), ed,
                    &ab0,
                    reinterpret_cast<float*>(d_qkv.ptr), qkv_d) != CUBLAS_STATUS_SUCCESS) break;

            if (cudaDeviceSynchronize() != cudaSuccess) break;
            std::vector<float> qkv_h(static_cast<size_t>(seq) * qkv_d);
            if (!d_qkv.copyToHost(qkv_h.data(), static_cast<size_t>(seq) * qkv_d * sizeof(float))) break;

            // Split Q, K, V sur host
            std::vector<float> Q(seq * ed), Km(seq * ed), Vm(seq * ed);
            for (int t = 0; t < seq; ++t) {
                const float* src = &qkv_h[static_cast<size_t>(t) * qkv_d];
                std::copy(src,        src + ed,    &Q[t*ed]);
                std::copy(src + ed,   src + 2*ed,  &Km[t*ed]);
                std::copy(src + 2*ed, src + qkv_d, &Vm[t*ed]);
            }

            // Step 2: attention par tête
            std::vector<float> attended(static_cast<size_t>(seq) * ed, 0.0f);
            const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
            bool aok = true;

            for (int h = 0; h < nh && aok; ++h) {
                const int hoff = h * hd;
                std::vector<float> Qh(seq*hd), Kh(seq*hd), Vh(seq*hd);
                for (int t = 0; t < seq; ++t) {
                    std::copy(&Q[t*ed+hoff],  &Q[t*ed+hoff+hd],  &Qh[t*hd]);
                    std::copy(&Km[t*ed+hoff], &Km[t*ed+hoff+hd], &Kh[t*hd]);
                    std::copy(&Vm[t*ed+hoff], &Vm[t*ed+hoff+hd], &Vh[t*hd]);
                }

                Impl::DeviceBuf d_qh, d_kh, d_vh, d_sc, d_ctx;
                if (!d_qh.alloc(seq*hd*sizeof(float)) ||
                    !d_kh.alloc(seq*hd*sizeof(float)) ||
                    !d_vh.alloc(seq*hd*sizeof(float)) ||
                    !d_sc.alloc(seq*seq*sizeof(float)) ||
                    !d_ctx.alloc(seq*hd*sizeof(float))) { aok = false; break; }
                if (!d_qh.copyFromHost(Qh.data(), seq*hd*sizeof(float)) ||
                    !d_kh.copyFromHost(Kh.data(), seq*hd*sizeof(float)) ||
                    !d_vh.copyFromHost(Vh.data(), seq*hd*sizeof(float))) { aok = false; break; }

                // scores[seq,seq] = Qh[seq,hd] × Kh[seq,hd]^T
                // → cublasSgemm(OP_T,OP_N, seq, seq, hd, scale, Kh, hd, Qh, hd, 0, sc, seq)
                if (cublasSgemm(impl_->handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        seq, seq, hd, &scale,
                        reinterpret_cast<const float*>(d_kh.ptr), hd,
                        reinterpret_cast<const float*>(d_qh.ptr), hd,
                        &ab0,
                        reinterpret_cast<float*>(d_sc.ptr), seq) != CUBLAS_STATUS_SUCCESS) { aok=false; break; }

                if (cudaDeviceSynchronize() != cudaSuccess) { aok = false; break; }
                std::vector<float> sc_h(seq * seq);
                if (!d_sc.copyToHost(sc_h.data(), seq*seq*sizeof(float))) { aok = false; break; }

                if (layer.causal) {
                    for (int i = 0; i < seq; ++i) {
                        float* row = &sc_h[i * seq];
                        for (int j = i+1; j < seq; ++j) row[j] = -1e9f;
                    }
                }
                softmax_rows(sc_h.data(), seq, seq);
                if (!d_sc.copyFromHost(sc_h.data(), seq*seq*sizeof(float))) { aok = false; break; }

                // ctx[seq,hd] = scores[seq,seq] × Vh[seq,hd]
                // → cublasSgemm(OP_N,OP_N, hd, seq, seq, 1, Vh, hd, sc, seq, 0, ctx, hd)
                if (cublasSgemm(impl_->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        hd, seq, seq, &aa,
                        reinterpret_cast<const float*>(d_vh.ptr), hd,
                        reinterpret_cast<const float*>(d_sc.ptr), seq,
                        &ab0,
                        reinterpret_cast<float*>(d_ctx.ptr), hd) != CUBLAS_STATUS_SUCCESS) { aok=false; break; }

                if (cudaDeviceSynchronize() != cudaSuccess) { aok = false; break; }
                std::vector<float> ctx_h(seq * hd);
                if (!d_ctx.copyToHost(ctx_h.data(), seq*hd*sizeof(float))) { aok = false; break; }

                for (int t = 0; t < seq; ++t)
                    std::copy(&ctx_h[t*hd], &ctx_h[(t+1)*hd], &attended[t*ed+hoff]);
            }
            if (!aok) break;

            // Step 3: output[seq,ed] = attended[seq,ed] × W_out[ed,ed]
            // → cublasSgemm(OP_N,OP_N, ed, seq, ed, 1, wout, ed, attended, ed, 0, out, ed)
            Impl::DeviceBuf d_att, d_wo, d_final;
            if (!d_att.alloc(static_cast<size_t>(seq)*ed*sizeof(float))) break;
            if (!d_wo.alloc(static_cast<size_t>(ed)*ed*sizeof(float))) break;
            if (!d_final.alloc(static_cast<size_t>(seq)*ed*sizeof(float))) break;
            if (!d_att.copyFromHost(attended.data(), static_cast<size_t>(seq)*ed*sizeof(float))) break;
            if (!d_wo.copyFromHost(wout, static_cast<size_t>(ed)*ed*sizeof(float))) break;

            if (cublasSgemm(impl_->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    ed, seq, ed, &aa,
                    reinterpret_cast<const float*>(d_wo.ptr), ed,
                    reinterpret_cast<const float*>(d_att.ptr), ed,
                    &ab0,
                    reinterpret_cast<float*>(d_final.ptr), ed) != CUBLAS_STATUS_SUCCESS) break;

            if (cudaDeviceSynchronize() != cudaSuccess) break;
            outputs.resize(1);
            outputs[0].resize(static_cast<size_t>(seq) * ed);
            if (!d_final.copyToHost(outputs[0].data(), static_cast<size_t>(seq)*ed*sizeof(float))) break;
            return true;
        } while (false);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Fast-path: CrossAttention
    // ─────────────────────────────────────────────────────────────────────────
    if (layer.type_enum == LayerType::CrossAttention &&
        config_.attention_enabled && !training &&
        inputs.size() >= 2 && inputs[1] != nullptr)
    {
        do {
            const std::vector<float>& q_in  = *inputs[0];
            const std::vector<float>& kv_in = *inputs[1];

            const int nh = layer.num_heads > 0 ? layer.num_heads : 1;
            int ed = layer.embed_dim;
            if (ed <= 0 && layer.head_dim > 0) ed = layer.head_dim * nh;
            if (ed <= 0 || nh <= 0 || ed % nh != 0) break;
            const int hd = ed / nh;

            if (q_in.size() % static_cast<size_t>(ed) != 0) break;
            if (kv_in.size() % static_cast<size_t>(ed) != 0) break;
            const int qlen  = static_cast<int>(q_in.size())  / ed;
            const int kvlen = static_cast<int>(kv_in.size()) / ed;

            const long long cattn_ops = 2LL * qlen * kvlen * ed;
            if (cattn_ops < static_cast<long long>(config_.attention_min_ops)) break;

            const float* wca = layer.getWeights();
            const int q_sz  = ed * ed;
            const int kv_sz = ed * 2 * ed;
            const int o_sz  = ed * ed;
            if (!wca || static_cast<int>(layer.getWeightsSize()) < q_sz + kv_sz + o_sz) break;
            const float* wq  = wca;
            const float* wkv = wca + q_sz;
            const float* wo  = wca + q_sz + kv_sz;

            const float ca = 1.0f, cb0 = 0.0f;

            // Step 1: Q[qlen,ed] = q_in[qlen,ed] × wq[ed,ed]
            Impl::DeviceBuf d_qin, d_wq, d_Q;
            if (!d_qin.alloc(qlen*ed*sizeof(float))) break;
            if (!d_wq.alloc(q_sz*sizeof(float))) break;
            if (!d_Q.alloc(qlen*ed*sizeof(float))) break;
            if (!d_qin.copyFromHost(q_in.data(), qlen*ed*sizeof(float))) break;
            if (!d_wq.copyFromHost(wq, q_sz*sizeof(float))) break;
            if (cublasSgemm(impl_->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    ed, qlen, ed, &ca,
                    reinterpret_cast<const float*>(d_wq.ptr), ed,
                    reinterpret_cast<const float*>(d_qin.ptr), ed,
                    &cb0,
                    reinterpret_cast<float*>(d_Q.ptr), ed) != CUBLAS_STATUS_SUCCESS) break;

            // Step 2: KV[kvlen,2*ed] = kv_in[kvlen,ed] × wkv[ed,2*ed]
            Impl::DeviceBuf d_kvin, d_wkv, d_KV;
            if (!d_kvin.alloc(kvlen*ed*sizeof(float))) break;
            if (!d_wkv.alloc(kv_sz*sizeof(float))) break;
            if (!d_KV.alloc(kvlen*2*ed*sizeof(float))) break;
            if (!d_kvin.copyFromHost(kv_in.data(), kvlen*ed*sizeof(float))) break;
            if (!d_wkv.copyFromHost(wkv, kv_sz*sizeof(float))) break;
            if (cublasSgemm(impl_->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    2*ed, kvlen, ed, &ca,
                    reinterpret_cast<const float*>(d_wkv.ptr), 2*ed,
                    reinterpret_cast<const float*>(d_kvin.ptr), ed,
                    &cb0,
                    reinterpret_cast<float*>(d_KV.ptr), 2*ed) != CUBLAS_STATUS_SUCCESS) break;

            if (cudaDeviceSynchronize() != cudaSuccess) break;
            std::vector<float> Q_h(qlen*ed), KV_h(kvlen*2*ed);
            if (!d_Q.copyToHost(Q_h.data(), qlen*ed*sizeof(float))) break;
            if (!d_KV.copyToHost(KV_h.data(), kvlen*2*ed*sizeof(float))) break;

            std::vector<float> K_h(kvlen*ed), V_h(kvlen*ed);
            for (int t = 0; t < kvlen; ++t) {
                std::copy(&KV_h[t*2*ed],      &KV_h[t*2*ed+ed],   &K_h[t*ed]);
                std::copy(&KV_h[t*2*ed+ed],   &KV_h[(t+1)*2*ed],  &V_h[t*ed]);
            }

            // Step 3: attention par tête
            std::vector<float> attended_c(qlen*ed, 0.0f);
            const float cscale = 1.0f / std::sqrt(static_cast<float>(hd));
            bool cok = true;

            for (int h = 0; h < nh && cok; ++h) {
                const int hoff = h * hd;
                std::vector<float> Qh(qlen*hd), Kh(kvlen*hd), Vh(kvlen*hd);
                for (int t = 0; t < qlen;  ++t) std::copy(&Q_h[t*ed+hoff], &Q_h[t*ed+hoff+hd], &Qh[t*hd]);
                for (int t = 0; t < kvlen; ++t) std::copy(&K_h[t*ed+hoff], &K_h[t*ed+hoff+hd], &Kh[t*hd]);
                for (int t = 0; t < kvlen; ++t) std::copy(&V_h[t*ed+hoff], &V_h[t*ed+hoff+hd], &Vh[t*hd]);

                Impl::DeviceBuf d_qh, d_kh, d_vh, d_sc, d_ctx;
                if (!d_qh.alloc(qlen*hd*sizeof(float))  || !d_kh.alloc(kvlen*hd*sizeof(float)) ||
                    !d_vh.alloc(kvlen*hd*sizeof(float))  || !d_sc.alloc(qlen*kvlen*sizeof(float)) ||
                    !d_ctx.alloc(qlen*hd*sizeof(float))) { cok=false; break; }
                if (!d_qh.copyFromHost(Qh.data(), qlen*hd*sizeof(float))   ||
                    !d_kh.copyFromHost(Kh.data(), kvlen*hd*sizeof(float))  ||
                    !d_vh.copyFromHost(Vh.data(), kvlen*hd*sizeof(float)))  { cok=false; break; }

                // scores[qlen,kvlen] = Qh[qlen,hd] × Kh[kvlen,hd]^T
                // → cublasSgemm(OP_T,OP_N, kvlen, qlen, hd, scale, Kh, hd, Qh, hd, 0, sc, kvlen)
                if (cublasSgemm(impl_->handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        kvlen, qlen, hd, &cscale,
                        reinterpret_cast<const float*>(d_kh.ptr), hd,
                        reinterpret_cast<const float*>(d_qh.ptr), hd,
                        &cb0,
                        reinterpret_cast<float*>(d_sc.ptr), kvlen) != CUBLAS_STATUS_SUCCESS) { cok=false; break; }

                if (cudaDeviceSynchronize() != cudaSuccess) { cok=false; break; }
                std::vector<float> sc_h(qlen * kvlen);
                if (!d_sc.copyToHost(sc_h.data(), qlen*kvlen*sizeof(float))) { cok=false; break; }
                softmax_rows(sc_h.data(), qlen, kvlen);
                if (!d_sc.copyFromHost(sc_h.data(), qlen*kvlen*sizeof(float))) { cok=false; break; }

                // ctx[qlen,hd] = scores[qlen,kvlen] × Vh[kvlen,hd]
                // → cublasSgemm(OP_N,OP_N, hd, qlen, kvlen, 1, Vh, hd, sc, kvlen, 0, ctx, hd)
                if (cublasSgemm(impl_->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        hd, qlen, kvlen, &ca,
                        reinterpret_cast<const float*>(d_vh.ptr), hd,
                        reinterpret_cast<const float*>(d_sc.ptr), kvlen,
                        &cb0,
                        reinterpret_cast<float*>(d_ctx.ptr), hd) != CUBLAS_STATUS_SUCCESS) { cok=false; break; }

                if (cudaDeviceSynchronize() != cudaSuccess) { cok=false; break; }
                std::vector<float> ctx_h(qlen * hd);
                if (!d_ctx.copyToHost(ctx_h.data(), qlen*hd*sizeof(float))) { cok=false; break; }
                for (int t = 0; t < qlen; ++t)
                    std::copy(&ctx_h[t*hd], &ctx_h[(t+1)*hd], &attended_c[t*ed+hoff]);
            }
            if (!cok) break;

            // Step 4: output[qlen,ed] = attended_c[qlen,ed] × wo[ed,ed]
            Impl::DeviceBuf d_att_c, d_wo_c, d_final_c;
            if (!d_att_c.alloc(qlen*ed*sizeof(float))) break;
            if (!d_wo_c.alloc(o_sz*sizeof(float))) break;
            if (!d_final_c.alloc(qlen*ed*sizeof(float))) break;
            if (!d_att_c.copyFromHost(attended_c.data(), qlen*ed*sizeof(float))) break;
            if (!d_wo_c.copyFromHost(wo, o_sz*sizeof(float))) break;
            if (cublasSgemm(impl_->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    ed, qlen, ed, &ca,
                    reinterpret_cast<const float*>(d_wo_c.ptr), ed,
                    reinterpret_cast<const float*>(d_att_c.ptr), ed,
                    &cb0,
                    reinterpret_cast<float*>(d_final_c.ptr), ed) != CUBLAS_STATUS_SUCCESS) break;

            if (cudaDeviceSynchronize() != cudaSuccess) break;
            outputs.resize(1);
            outputs[0].resize(static_cast<size_t>(qlen) * ed);
            if (!d_final_c.copyToHost(outputs[0].data(), static_cast<size_t>(qlen)*ed*sizeof(float))) break;
            return true;
        } while (false);
    }
#endif // ENABLE_CUDA

    // Fallback: CPU (implémentation complète des LayerType).
    return RuntimeLayerDispatch::cpu_forward_layer(inputs, outputs, layer, training);
}
