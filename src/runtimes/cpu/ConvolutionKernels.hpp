#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include "SIMD_Ops.hpp"
#include "RuntimeAllocator.hpp"
#include "runtimes/LayerOps.hpp"

namespace CpuConvolution {
// Bounded im2col tiles keep the model's GEMM path inside the CPU runtime.
inline void multiply(float* out,const float* a,const float* b,int m,int n,int k,bool transpose_b) {
    if (RuntimeLayerOps::hardwareAccelerationEnabled()) {
        if (transpose_b) SIMD::matmul_transpose_avx2(out,a,b,m,n,k);
        else SIMD::matmul_avx2(out,a,b,m,n,k);
    } else {
        for(int i=0;i<m;++i) for(int j=0;j<n;++j) {
            float sum=0;
            for(int t=0;t<k;++t) sum=std::fma(a[static_cast<size_t>(i)*k+t],b[transpose_b?static_cast<size_t>(j)*k+t:static_cast<size_t>(t)*n+j],sum);
            out[static_cast<size_t>(i)*n+j]=sum;
        }
    }
}
// C += A * B without a full m*n intermediate, using the original GEMM
// reduction order and bounded register accumulators.
inline void multiply_add(float* out,const float* a,const float* b,int m,int n,int k) {
    if(!RuntimeLayerOps::hardwareAccelerationEnabled()) {
        for(int row=0;row<m;++row) for(int column=0;column<n;++column) {
            float sum=0;
            for(int t=0;t<k;++t) sum=std::fma(a[static_cast<size_t>(row)*k+t],b[static_cast<size_t>(t)*n+column],sum);
            out[static_cast<size_t>(row)*n+column]+=sum;
        }
        return;
    }
    HardwareOpt::matmul_add_fma_saturated(out,a,b,m,n,k);
}
inline void pack(const float* x,float* col,int start,int count,int h,int w,int channels,int kernel,int stride,int pad,int dilation,int out_w) {
    const int k=channels*kernel*kernel;
    for(int r=0;r<count;++r) {
        const int oh=(start+r)/out_w,ow=(start+r)%out_w;
        for(int c=0;c<channels;++c) for(int kh=0;kh<kernel;++kh) for(int kw=0;kw<kernel;++kw) {
            const int ih=oh*stride-pad+kh*dilation,iw=ow*stride-pad+kw*dilation;
            col[static_cast<size_t>(r)*k+(c*kernel+kh)*kernel+kw]=
                ih>=0&&ih<h&&iw>=0&&iw<w?x[(static_cast<size_t>(c)*h+ih)*w+iw]:0.0f;
        }
    }
}
inline int tile_size(int k,int out_c,int spatial) {
    return Mimir::Planning::ExecutionPolicy::convolutionTileRows(size_t(k),size_t(out_c),spatial);
}
inline void forward(const std::vector<float>& x,std::vector<float>& y,const float* weights,const float* bias,
                    int h,int w,int in_c,int out_c,int kernel,int stride,int pad,int dilation) {
    const int oh=(h+2*pad-dilation*(kernel-1)-1)/stride+1;
    const int ow=(w+2*pad-dilation*(kernel-1)-1)/stride+1;
    if(oh<=0||ow<=0) throw std::invalid_argument("Conv2d: invalid output dimensions");
    const int spatial=oh*ow,k=in_c*kernel*kernel,tile=tile_size(k,out_c,spatial);
    // CHW 1x1 is already a matrix: no im2col, temporary rows or layout copy.
    // The planner compares this variant against the packed reference; shape
    // eligibility is mathematical, not a size-based performance preference.
    if(kernel==1 && stride==1 && pad==0 && (Mimir::Planning::ExecutionPolicy::cpuChoice().algorithm==Mimir::Planning::ExecutionPolicy::CpuAlgorithm::PointwiseConvolution)) {
        y.resize(static_cast<size_t>(out_c)*spatial);
        multiply(y.data(),weights,x.data(),out_c,spatial,in_c,false);
        if(bias) for(int c=0;c<out_c;++c) {
            float* row=y.data()+static_cast<size_t>(c)*spatial;
            for(int i=0;i<spatial;++i) row[i]+=bias[c];
        }
        return;
    }
    auto& guard=MemoryGuard::instance();RuntimeAllocator allocator(guard,guard.getLimit()/(1024*1024));
    auto scratch=allocator.get_scratchpad_uninitialized((static_cast<size_t>(tile)*(size_t(k)+out_c)+16)*sizeof(float),"conv/forward");
    float* col=scratch.data();
    // Offset scratch regions by one cache line to avoid power-of-two aliases.
    float* rows=col+static_cast<size_t>(tile)*k+16;
    y.resize(static_cast<size_t>(out_c)*spatial);
    for(int start=0;start<spatial;start+=tile) {
        const int count=std::min(tile,spatial-start);
        pack(x.data(),col,start,count,h,w,in_c,kernel,stride,pad,dilation,ow);
        multiply(rows,col,weights,count,out_c,k,true);
        for(int r=0;r<count;++r) for(int c=0;c<out_c;++c)
            y[static_cast<size_t>(c)*spatial+start+r]=rows[static_cast<size_t>(r)*out_c+c]+(bias?bias[c]:0.0f);
    }
}

// CHW transposed convolution with weights laid out [out_c, in_c, kernel, kernel].
// Keep the scatter accumulation order used by the original CPU implementation.
inline void transposeForward(const std::vector<float>& input, std::vector<float>& output,
                             const float* weights, const float* bias,
                             int h, int w, int in_c, int out_c, int kernel,
                             int stride, int pad) {
    if (h <= 0 || w <= 0 || in_c <= 0 || out_c <= 0 || kernel <= 0 ||
        stride <= 0 || pad < 0 || !weights) {
        throw std::invalid_argument("ConvTranspose2d: invalid dimensions or weights");
    }
    const int64_t out_h64 = (static_cast<int64_t>(h) - 1) * stride -
        2 * static_cast<int64_t>(pad) + kernel;
    const int64_t out_w64 = (static_cast<int64_t>(w) - 1) * stride -
        2 * static_cast<int64_t>(pad) + kernel;
    if (out_h64 <= 0 || out_w64 <= 0 ||
        out_h64 > std::numeric_limits<int>::max() ||
        out_w64 > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("ConvTranspose2d: invalid output dimensions");
    }
    const auto checked_product = [](size_t a, size_t b) {
        if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
            throw std::invalid_argument("ConvTranspose2d: dimensions overflow");
        return a * b;
    };
    const size_t input_plane = checked_product(static_cast<size_t>(h), static_cast<size_t>(w));
    const size_t input_size = checked_product(input_plane, static_cast<size_t>(in_c));
    if (input.size() < input_size)
        throw std::invalid_argument("ConvTranspose2d: input smaller than configured dimensions");
    const size_t kernel_plane = checked_product(static_cast<size_t>(kernel), static_cast<size_t>(kernel));
    (void)checked_product(checked_product(static_cast<size_t>(out_c), static_cast<size_t>(in_c)),
                          kernel_plane);
    const int out_h = static_cast<int>(out_h64);
    const int out_w = static_cast<int>(out_w64);
    const size_t output_plane = checked_product(static_cast<size_t>(out_h), static_cast<size_t>(out_w));
    const size_t output_size = checked_product(output_plane, static_cast<size_t>(out_c));
    if (output_size > output.max_size())
        throw std::invalid_argument("ConvTranspose2d: output dimensions exceed vector capacity");
    output.assign(output_size, 0.0f);
    if (bias) {
        for (int oc = 0; oc < out_c; ++oc) {
            for (size_t i = 0; i < output_plane; ++i)
                output[static_cast<size_t>(oc) * output_plane + i] = bias[oc];
        }
    }
    for (int oc = 0; oc < out_c; ++oc) {
        for (int ic = 0; ic < in_c; ++ic) {
            for (int ih = 0; ih < h; ++ih) {
                for (int iw = 0; iw < w; ++iw) {
                    const float value = input[(static_cast<size_t>(ic) * h + ih) * w + iw];
                    for (int kh = 0; kh < kernel; ++kh) {
                        for (int kw = 0; kw < kernel; ++kw) {
                            const int64_t oh = static_cast<int64_t>(ih) * stride + kh - pad;
                            const int64_t ow = static_cast<int64_t>(iw) * stride + kw - pad;
                            if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                                const size_t kernel_index =
                                    ((static_cast<size_t>(oc) * in_c + ic) * kernel + kh) * kernel + kw;
                                const size_t output_index =
                                    (static_cast<size_t>(oc) * out_h + static_cast<size_t>(oh)) * out_w +
                                    static_cast<size_t>(ow);
                                output[output_index] += value * weights[kernel_index];
                            }
                        }
                    }
                }
            }
        }
    }
}

inline void backward(const std::vector<float>& x,const std::vector<float>& go,std::vector<float>& dx,Layer& layer,
                     int h,int w,int in_c,int out_c,int kernel,int stride,int pad,int dilation) {
    const int oh=(h+2*pad-dilation*(kernel-1)-1)/stride+1;
    const int ow=(w+2*pad-dilation*(kernel-1)-1)/stride+1;
    const int spatial=oh*ow,k=in_c*kernel*kernel,tile=tile_size(k,out_c,spatial);
    auto& guard=MemoryGuard::instance();RuntimeAllocator allocator(guard,guard.getLimit()/(1024*1024));
    auto scratch=allocator.get_scratchpad_uninitialized((static_cast<size_t>(tile)*(size_t(k)+2ULL*out_c)+32)*sizeof(float),"conv/backward");
    float* col=scratch.data();
    float* dy=col+static_cast<size_t>(tile)*k+16;
    float* dy_t=dy+static_cast<size_t>(tile)*out_c+16;
    dx.assign(x.size(),0.0f);
    if(layer.grad_weights.size()!=layer.getWeightsSize()) layer.grad_weights.assign(layer.getWeightsSize(),0.0f);
    if(layer.use_bias && layer.grad_bias.size()!=static_cast<size_t>(out_c)) layer.grad_bias.assign(out_c,0.0f);
    for(int start=0;start<spatial;start+=tile) {
        const int count=std::min(tile,spatial-start);
        pack(x.data(),col,start,count,h,w,in_c,kernel,stride,pad,dilation,ow);
        for(int r=0;r<count;++r) for(int c=0;c<out_c;++c) {
            const float g=go[static_cast<size_t>(c)*spatial+start+r];
            dy[static_cast<size_t>(r)*out_c+c]=g;
            dy_t[static_cast<size_t>(c)*count+r]=g;
            if(layer.use_bias) { layer.grad_weights[static_cast<size_t>(out_c)*k+c]+=g;layer.grad_bias[c]+=g; }
        }
        multiply_add(layer.grad_weights.data(),dy_t,col,out_c,k,count);
        // The packed input is dead after dW. Its allocation now holds dCol.
        multiply(col,dy,layer.getWeights(),count,k,out_c,false);
        for(int r=0;r<count;++r) {
            const int row=(start+r)/ow,column=(start+r)%ow;
            for(int c=0;c<in_c;++c) for(int kh=0;kh<kernel;++kh) for(int kw=0;kw<kernel;++kw) {
                const int ih=row*stride-pad+kh*dilation,iw=column*stride-pad+kw*dilation;
                if(ih>=0&&ih<h&&iw>=0&&iw<w)
                    dx[(static_cast<size_t>(c)*h+ih)*w+iw]+=col[static_cast<size_t>(r)*k+(c*kernel+kh)*kernel+kw];
            }
        }
    }
}
} // namespace CpuConvolution
