#ifndef __HARDWARE_OPT_HPP__
#define __HARDWARE_OPT_HPP__

#include "SimdDispatch.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include "Parallel.hpp"
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#if !defined(_WIN32)
#include <sys/mman.h>   // madvise, HugePages
#include <unistd.h>
#endif

// ===== FP16 STORAGE + F16C =====

namespace HardwareOpt {

// Respect the planner's measured choice and the number of independent row
// tiles. Nested calls never create another OpenMP team.
inline int matmul_workers(size_t rows,size_t columns,size_t reduction) {
    if(!rows || !columns || !reduction)return 1;
    return RuntimeParallel::workers(rows);
}

#if defined(_MSC_VER)
#define MIMIR_RESTRICT __restrict
#else
#define MIMIR_RESTRICT __restrict__
#endif

// Legacy names remain source compatible; ISA-specific code lives in the
// multiversioned dispatcher, never in a globally AVX-compiled header.
inline void fp32_to_fp16_f16c(uint16_t* dst,const float* src,size_t count) {
    RuntimeSimd::fp32ToFp16(dst,src,count);
}
inline void fp16_to_fp32_f16c(float* dst,const uint16_t* src,size_t count) {
    RuntimeSimd::fp16ToFp32(dst,src,count);
}

// ===== BF16 STORAGE (software, portable) =====

inline void fp32_to_bf16(uint16_t* dst, const float* src, size_t count) {
    // Rounding-to-nearest-even on truncation of lower 16 bits.
    #pragma omp simd
    for (size_t i = 0; i < count; ++i) {
        union { float f; uint32_t u; } v;
        v.f = src[i];
        const uint32_t lsb = (v.u >> 16) & 1u;
        const uint32_t rounding_bias = 0x7FFFu + lsb;
        dst[i] = static_cast<uint16_t>((v.u + rounding_bias) >> 16);
    }
}

inline void bf16_to_fp32(float* dst, const uint16_t* src, size_t count) {
    #pragma omp simd
    for (size_t i = 0; i < count; ++i) {
        union { float f; uint32_t u; } v;
        v.u = (static_cast<uint32_t>(src[i]) << 16);
        dst[i] = v.f;
    }
}

// Ordered reductions use the same correctly rounded FMA for vector blocks
// and scalar tails. The planner still controls independent-row parallelism.
template<bool Accumulate>
inline void matmul_fma_saturated_impl(float* C,const float* A,const float* B,
                                      size_t M,size_t N,size_t K) {
    RuntimeSimd::matmul(C,A,B,M,N,K,Accumulate,true,matmul_workers(M*((N+23)/24),24,K));
}
inline void matmul_fma_saturated(float* C,const float* A,const float* B,
                                size_t M,size_t N,size_t K) {
    matmul_fma_saturated_impl<false>(C,A,B,M,N,K);
}
inline void matmul_add_fma_saturated(float* C,const float* A,const float* B,
                                    size_t M,size_t N,size_t K) {
    matmul_fma_saturated_impl<true>(C,A,B,M,N,K);
}

// Legacy single-channel convolution. Each kernel coefficient contributes
// exactly once, with complete checks at the right and bottom padded borders.
inline void conv2d_fma_saturated(float* output,const float* input,const float* kernel,
                                int in_h,int in_w,int out_h,int out_w,
                                int kernel_size,int stride,int padding) {
    const int workers=RuntimeParallel::workers(size_t(out_h)*out_w);
    RuntimeParallel::forEach(size_t(out_h)*out_w,workers,[&](size_t pos) {
        const int oh=int(pos/size_t(out_w)),ow=int(pos%size_t(out_w));
        float sum=0;
        for(int kh=0;kh<kernel_size;++kh) {
            const int ih=oh*stride+kh-padding;
            if(ih<0 || ih>=in_h)continue;
            for(int kw=0;kw<kernel_size;++kw) {
                const int iw=ow*stride+kw-padding;
                if(iw>=0 && iw<in_w)sum=std::fma(input[ih*in_w+iw],kernel[kh*kernel_size+kw],sum);
            }
        }
        output[pos]=sum;
    });
}

// Quantization does not require BMI2: these are arithmetic conversions, not
// PEXT/PDEP operations. Preserve complete tails on every CPU architecture.
inline void quantize_int8_bmi(int8_t* dst,const float* src,size_t count,float scale,float zero_point) {
    const float inv_scale=1.0f/scale;
    for(size_t i=0;i<count;++i)dst[i]=static_cast<int8_t>(std::clamp(std::nearbyint(src[i]*inv_scale+zero_point),-128.0f,127.0f));
}
inline void dequantize_int8_bmi(float* dst,const int8_t* src,size_t count,float scale,float zero_point) {
    for(size_t i=0;i<count;++i)dst[i]=(float(src[i])-zero_point)*scale;
}
inline void quantize_int4_bmi(uint8_t* dst,const float* src,size_t count,float scale,float zero_point) {
    const float inv_scale=1.0f/scale;
    for(size_t i=0;i<count;i+=2) {
        const auto low=static_cast<uint8_t>(std::clamp(std::nearbyint(src[i]*inv_scale+zero_point),0.0f,15.0f));
        const auto high=i+1<count?static_cast<uint8_t>(std::clamp(std::nearbyint(src[i+1]*inv_scale+zero_point),0.0f,15.0f)):uint8_t(0);
        dst[i/2]=uint8_t(low|(high<<4));
    }
}

// ===== HUGEPAGES + MADVISE =====

class HugePageAllocator {
public:
    static constexpr size_t HUGEPAGE_SIZE = 2 * 1024 * 1024; // 2MB
    
    // Allouer avec HugePages
    static void* allocate_huge(size_t size) {
        // Round up to hugepage boundary
        size_t aligned_size = ((size + HUGEPAGE_SIZE - 1) / HUGEPAGE_SIZE) * HUGEPAGE_SIZE;

#if defined(_WIN32)
        // Windows fallback: standard heap allocation in cross-builds.
        return std::malloc(aligned_size);
#else
        
        void* ptr = mmap(nullptr, aligned_size,
                        PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                        -1, 0);
        
        if (ptr == MAP_FAILED) {
            // Fallback to normal pages
            ptr = mmap(nullptr, aligned_size,
                      PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS,
                      -1, 0);
        }
        
        if (ptr != MAP_FAILED) {
            // Advise kernel about usage pattern
            madvise(ptr, aligned_size, MADV_HUGEPAGE);    // Prefer huge pages
            madvise(ptr, aligned_size, MADV_SEQUENTIAL);  // Sequential access
            madvise(ptr, aligned_size, MADV_WILLNEED);    // Prefetch
            
            // Prefetch parallèle par blocs de 4KB (page size) pour grandes allocations
            if (aligned_size > 16 * HUGEPAGE_SIZE) {
                for (size_t offset = 0; offset < aligned_size; offset += 4096) {
                    __builtin_prefetch(static_cast<char*>(ptr) + offset, 1, 3);
                }
            }
        }
        
        return ptr;
#endif
    }
    
    // Libérer
    static void deallocate_huge(void* ptr, size_t size) {
#if defined(_WIN32)
        (void)size;
        std::free(ptr);
#else
        if (ptr) {
            size_t aligned_size = ((size + HUGEPAGE_SIZE - 1) / HUGEPAGE_SIZE) * HUGEPAGE_SIZE;
            munmap(ptr, aligned_size);
        }
#endif
    }
    
    // Allouer buffer avec HugePages
    template<typename T>
    static T* allocate_buffer(size_t count) {
        size_t bytes = count * sizeof(T);
        return static_cast<T*>(allocate_huge(bytes));
    }
    
    // Libérer buffer
    template<typename T>
    static void deallocate_buffer(T* ptr, size_t count) {
        deallocate_huge(ptr, count * sizeof(T));
    }
};

// Wrapper pour std::vector avec HugePages
template<typename T>
class HugePageVector {
private:
    T* data_;
    size_t size_;
    size_t capacity_;
    
public:
    HugePageVector() : data_(nullptr), size_(0), capacity_(0) {}
    
    ~HugePageVector() {
        if (data_) {
            HugePageAllocator::deallocate_buffer(data_, capacity_);
        }
    }
    
    void reserve(size_t new_capacity) {
        if (new_capacity > capacity_) {
            T* new_data = HugePageAllocator::allocate_buffer<T>(new_capacity);
            if (data_) {
                std::memcpy(new_data, data_, size_ * sizeof(T));
                HugePageAllocator::deallocate_buffer(data_, capacity_);
            }
            data_ = new_data;
            capacity_ = new_capacity;
        }
    }
    
    void resize(size_t new_size) {
        if (new_size > capacity_) {
            reserve(new_size * 2); // Growth factor 2x
        }
        size_ = new_size;
    }
    
    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }
};

} // namespace HardwareOpt

#undef MIMIR_RESTRICT

#endif // __HARDWARE_OPT_HPP__
