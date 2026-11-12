#pragma once

#include "RuntimeAllocator.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <vector>

namespace RuntimeScratch {
// Scratch never escapes a kernel. Slots contain only idle buffers; nested calls
// and concurrently live temporaries always own distinct storage. No global lock.
class Pool {
    static constexpr size_t slots = 32;
    std::array<std::optional<BufferHandle>, slots> idle_;
    size_t bytes_ = 0, used_ = 0;
    size_t hits_ = 0, allocations_ = 0;
public:
    static Pool& local() { thread_local Pool pool; return pool; }
    static size_t budget() {
        return std::min(size_t(64)*1024*1024, MemoryGuard::instance().getLimit()/16);
    }
    void clear() noexcept {
        for (size_t i=0;i<used_;++i) idle_[i].reset();
        bytes_ = used_ = 0;
    }
    size_t cachedBytes() const { return bytes_; }
    size_t hits() const { return hits_; }
    size_t allocations() const { return allocations_; }
    BufferHandle acquire(size_t bytes) {
        if (bytes_ > budget()) clear();
        size_t best = used_;
        for (size_t i=0;i<used_;++i)
            if (idle_[i] && idle_[i]->size_bytes() >= bytes &&
                (best==used_ || idle_[i]->size_bytes() < idle_[best]->size_bytes())) best=i;
        if (best != used_) {
            auto result = std::move(*idle_[best]);
            bytes_ -= result.size_bytes();
            --used_;
            if (best != used_) idle_[best] = std::move(idle_[used_]);
            idle_[used_].reset(); ++hits_;
            return result;
        }
        // No idle block fits: discard obsolete smaller working sets before
        // asking MemoryGuard for a larger one. Live leases remain untouched.
        clear();
        auto& guard = MemoryGuard::instance();
        if (bytes > guard.getLimit()) throw std::length_error("scratch request exceeds MemoryGuard limit");
        if (guard.getCurrentBytes() > guard.getLimit() ||
            bytes > guard.getLimit()-std::min(guard.getLimit(),guard.getCurrentBytes())) clear();
        RuntimeAllocator allocator(guard,0);
        auto result = allocator.allocate_buffer_uninitialized(bytes,"cpu/scratch");
        ++allocations_;
        return result;
    }
    void release(BufferHandle&& buffer) noexcept {
        const size_t cap=budget(), bytes=buffer.size_bytes();
        if (bytes_ > cap) clear();
        if (!bytes || bytes > cap) return;
        // Evict idle storage before retaining a new working set. Live leases
        // are never invalidated, including when the global limit is reduced.
        if (bytes > cap-bytes_) clear();
        if (used_ < slots) {
            bytes_ += bytes; idle_[used_++].emplace(std::move(buffer)); return;
        }
        size_t smallest=0;
        for(size_t i=1;i<used_;++i)
            if(idle_[i]->size_bytes()<idle_[smallest]->size_bytes())smallest=i;
        if(bytes>idle_[smallest]->size_bytes()) {
            bytes_-=idle_[smallest]->size_bytes(); bytes_+=bytes;
            idle_[smallest].emplace(std::move(buffer));
        }
    }
};

class FloatBuffer {
    std::optional<BufferHandle> buffer_;
    size_t size_ = 0;
    float* data_ = nullptr;
    static size_t byteCount(size_t count) {
        if (count > std::numeric_limits<size_t>::max()/sizeof(float))
            throw std::length_error("scratch buffer size overflow");
        return count*sizeof(float);
    }
    void reserve(size_t count) {
        const size_t bytes=byteCount(count);
        if (!count || (buffer_ && buffer_->size_bytes() >= bytes)) return;
        auto replacement=Pool::local().acquire(bytes);
        if (size_) std::copy_n(data(),size_,replacement.data());
        if (buffer_) Pool::local().release(std::move(*buffer_));
        buffer_.emplace(std::move(replacement));
        data_=buffer_->data();
    }
public:
    FloatBuffer() = default;
    explicit FloatBuffer(size_t count) { resize(count); }
    FloatBuffer(size_t count,float value) { assign(count,value); }
    ~FloatBuffer() { if (buffer_) Pool::local().release(std::move(*buffer_)); }
    FloatBuffer(const FloatBuffer&) = delete;
    FloatBuffer& operator=(const FloatBuffer&) = delete;
    // Leases are scoped to the creating thread and cannot be transferred.
    FloatBuffer(FloatBuffer&&) = delete;
    FloatBuffer& operator=(FloatBuffer&&) = delete;
    size_t size() const { return size_; }
    bool empty() const { return size_==0; }
    float* data() { return data_; }
    const float* data() const { return data_; }
    float* begin() { return data(); }
    float* end() { return size_ ? data()+size_ : data(); }
    const float* begin() const { return data(); }
    const float* end() const { return size_ ? data()+size_ : data(); }
    float& operator[](size_t i) { return data()[i]; }
    const float& operator[](size_t i) const { return data()[i]; }
    void resize(size_t count) {
        reserve(count);
        if (count > size_) std::fill(data()+size_,data()+count,0.f);
        size_=count;
    }
    // Only use when every element will be written before any read.
    void resizeUninitialized(size_t count) { reserve(count); size_=count; }
    void assign(size_t count,float value) {
        reserve(count); size_=count;
        if (count) std::fill_n(data(),count,value);
    }
    void swap(FloatBuffer& other) noexcept {
        buffer_.swap(other.buffer_); std::swap(size_,other.size_); std::swap(data_,other.data_);
    }
};

// Disjoint slices of one leased arena. Spans are internal to the kernel and
// must never be retained by a model, output tensor, worker, or asynchronous GPU.
class FloatSpan {
    float* data_;
    size_t size_;
public:
    FloatSpan(float* data,size_t size):data_(data),size_(size) {}
    float* data() {return data_;}
    const float* data() const {return data_;}
    size_t size() const {return size_;}
    float* begin() {return data_;}
    float* end() {return size_?data_+size_:data_;}
    float& operator[](size_t i) {return data_[i];}
    const float& operator[](size_t i) const {return data_[i];}
    void fill(float value) {if(size_)std::fill_n(data_,size_,value);}
    // Kernels writing to a fixed slice may validate its planned output shape.
    void resize(size_t size) {if(size!=size_)throw std::length_error("scratch slice shape mismatch");}
    void swap(FloatSpan& other) noexcept {std::swap(data_,other.data_);std::swap(size_,other.size_);}
};
class FloatArena {
    FloatBuffer storage_;
    size_t used_=0;
public:
    explicit FloatArena(std::initializer_list<size_t> sizes) {
        size_t total=0;
        for(size_t size:sizes) {
            const size_t padding=size?16:0;
            if(padding>std::numeric_limits<size_t>::max()-total || size>std::numeric_limits<size_t>::max()-total-padding)
                throw std::length_error("scratch arena size overflow");
            total+=size+padding;
        }
        storage_.resizeUninitialized(total);
    }
    FloatSpan take(size_t size) {
        const size_t available=storage_.size()-used_,padding=size?16:0;
        if(available<padding || size>available-padding)throw std::length_error("scratch arena exhausted");
        float* ptr=storage_.data();
        if(used_)ptr+=used_;
        used_+=size+(size?16:0);
        return {ptr,size};
    }
};

// Read-only borrowed weights. The model owns the underlying parameter block.
class ConstFloatView {
    const float* data_ = nullptr;
    size_t size_ = 0;
public:
    ConstFloatView() = default;
    ConstFloatView(const std::vector<float>& values) : data_(values.data()),size_(values.size()) {}
    ConstFloatView(const float* first,const float* last) { assign(first,last); }
    void assign(const float* first,const float* last) { data_=first; size_=first==last?0:size_t(last-first); }
    size_t size() const { return size_; }
    bool empty() const { return !size_; }
    const float* data() const { return data_; }
    const float& operator[](size_t i) const { return data_[i]; }
};
} // namespace RuntimeScratch
