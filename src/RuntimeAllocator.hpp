#pragma once

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <stdexcept>
#include "MemoryGuard.hpp"
#include "DynamicTensorAllocator.hpp"
#include "tensors.hpp"

// ============================================================================
// RuntimeAllocator - Gestionnaire mémoire strict pour Mímir
// ============================================================================
// 
// Objectif : TOUTE allocation runtime (activations, buffers temporaires, 
// intermédiaires) doit passer par ce gestionnaire pour respecter la limite RAM.
//
// Fonctionnalités :
// - Comptabilisation automatique via MemoryGuard
// - RAII handles pour auto-release
// - Pool de buffers réutilisables (scratchpad)
// - Tracking des allocations (debug)
// - Interdiction des allocations sauvages (std::vector)
//
// Usage :
//   RuntimeAllocator alloc(memory_guard, max_ram_mb);
//   auto tensor = alloc.allocate_tensor({batch, channels, h, w}, "conv1_out");
//   auto buffer = alloc.allocate_buffer(1024 * 1024, "temp_gemm");
//   // Auto-release via RAII
// ============================================================================

// Forward declarations
class MemoryGuard;
class DynamicTensorAllocator;

// Tensor descriptor pour shape + metadata
struct TensorDescriptor {
    std::vector<int> shape;         // [N, C, H, W] ou autre
    std::string dtype = "float32";  // Pour future support FP16/INT8
    size_t size_bytes = 0;
    std::string name;
    
    TensorDescriptor() = default;
    TensorDescriptor(std::vector<int> s, const std::string& dt = "float32", const std::string& n = "")
        : shape(std::move(s)), dtype(dt), name(n) {
        size_t numel = 1;
        for (int dim : shape) numel *= dim;
        
        if (dtype == "float32") {
            size_bytes = numel * sizeof(float);
        } else if (dtype == "float16") {
            size_bytes = numel * 2;
        } else if (dtype == "int8") {
            size_bytes = numel;
        } else {
            throw std::runtime_error("Unsupported dtype: " + dtype);
        }
    }
    
    size_t numel() const {
        size_t n = 1;
        for (int dim : shape) n *= dim;
        return n;
    }
};

// RAII Handle pour buffer temporaire
class BufferHandle {
public:
    BufferHandle(float* ptr, size_t bytes, const std::string& tag, MemoryGuard* guard)
        : data_(ptr), size_bytes_(bytes), tag_(tag), guard_(guard) {}
    
    ~BufferHandle() {
        if (guard_ && data_) {
            // Libération automatique
            guard_->releaseAllocation(size_bytes_);
        }
    }
    
    // Non-copyable, movable
    BufferHandle(const BufferHandle&) = delete;
    BufferHandle& operator=(const BufferHandle&) = delete;
    
    BufferHandle(BufferHandle&& other) noexcept
        : data_(other.data_), size_bytes_(other.size_bytes_),
          tag_(other.tag_), guard_(other.guard_) {
        other.data_ = nullptr;
        other.guard_ = nullptr;
    }
    
    BufferHandle& operator=(BufferHandle&& other) noexcept {
        if (this != &other) {
            // Release current
            if (guard_ && data_) guard_->releaseAllocation(size_bytes_);
            
            data_ = other.data_;
            size_bytes_ = other.size_bytes_;
            tag_ = other.tag_;
            guard_ = other.guard_;
            
            other.data_ = nullptr;
            other.guard_ = nullptr;
        }
        return *this;
    }
    
    float* data() { return data_; }
    const float* data() const { return data_; }
    size_t size() const { return size_bytes_ / sizeof(float); }
    size_t size_bytes() const { return size_bytes_; }
    const std::string& tag() const { return tag_; }
    
private:
    float* data_;
    size_t size_bytes_;
    std::string tag_;
    MemoryGuard* guard_;
};

// RAII Handle pour Tensor (wrapper autour de std::vector mais comptabilisé)
class TensorHandle {
public:
    TensorHandle(std::vector<float>&& data, const TensorDescriptor& desc, MemoryGuard* guard)
        : data_(std::move(data)), descriptor_(desc), guard_(guard) {}
    
    ~TensorHandle() {
        if (guard_ && !data_.empty()) {
            guard_->releaseAllocation(descriptor_.size_bytes);
        }
    }
    
    // Non-copyable, movable
    TensorHandle(const TensorHandle&) = delete;
    TensorHandle& operator=(const TensorHandle&) = delete;
    
    TensorHandle(TensorHandle&& other) noexcept
        : data_(std::move(other.data_)), descriptor_(other.descriptor_), guard_(other.guard_) {
        other.guard_ = nullptr;
    }
    
    TensorHandle& operator=(TensorHandle&& other) noexcept {
        if (this != &other) {
            if (guard_ && !data_.empty()) guard_->releaseAllocation(descriptor_.size_bytes);
            
            data_ = std::move(other.data_);
            descriptor_ = other.descriptor_;
            guard_ = other.guard_;
            
            other.guard_ = nullptr;
        }
        return *this;
    }
    
    std::vector<float>& data() { return data_; }
    const std::vector<float>& data() const { return data_; }
    const TensorDescriptor& descriptor() const { return descriptor_; }
    
    size_t size() const { return data_.size(); }
    float* ptr() { return data_.data(); }
    const float* ptr() const { return data_.data(); }
    
private:
    std::vector<float> data_;
    TensorDescriptor descriptor_;
    MemoryGuard* guard_;
};

// ============================================================================
// RuntimeAllocator - Main class
// ============================================================================

class RuntimeAllocator {
public:
    RuntimeAllocator(MemoryGuard& guard, size_t max_ram_mb = 4096)
        : memory_guard_(guard), max_ram_bytes_(max_ram_mb * 1024ULL * 1024ULL) {
        // Initialiser avec MemoryGuard configuré
        if (memory_guard_.getLimit() == 10ULL * 1024 * 1024 * 1024) {  // Valeur par défaut
            memory_guard_.setLimit(max_ram_mb * 1024ULL * 1024ULL);
        }
    }
    
    // Allocation d'un Tensor avec shape
    TensorHandle allocate_tensor(const std::vector<int>& shape,
                                  const std::string& dtype = "float32",
                                  const std::string& name = "") {
        TensorDescriptor desc(shape, dtype, name);
        
        // Vérifier limite avant allocation
        if (!memory_guard_.requestAllocation(desc.size_bytes, name)) {
            throw std::runtime_error(
                "RuntimeAllocator: Cannot allocate tensor '" + name + "' (" +
                std::to_string(desc.size_bytes / (1024*1024)) + " MB) - " +
                "would exceed RAM limit. Current: " +
                std::to_string(memory_guard_.getCurrentBytes() / (1024*1024)) + " MB / " +
                std::to_string(memory_guard_.getLimit() / (1024*1024)) + " MB"
            );
        }
        
        // Allouer (comptabilisé par MemoryGuard via reserve)
        std::vector<float> data(desc.numel(), 0.0f);
        
        // Tracking (optionnel, debug)
        total_allocated_bytes_ += desc.size_bytes;
        num_allocations_++;
        
        return TensorHandle(std::move(data), desc, &memory_guard_);
    }
    
    // Allocation d'un buffer brut (pour calculs temporaires)
    BufferHandle allocate_buffer(size_t bytes, const std::string& tag = "") {
        if (!memory_guard_.requestAllocation(bytes, tag)) {
            throw std::runtime_error(
                "RuntimeAllocator: Cannot allocate buffer '" + tag + "' (" +
                std::to_string(bytes / (1024*1024)) + " MB) - " +
                "would exceed RAM limit. Current: " +
                std::to_string(memory_guard_.getCurrentBytes() / (1024*1024)) + " MB / " +
                std::to_string(memory_guard_.getLimit() / (1024*1024)) + " MB"
            );
        }
        
        // Allouer via new[] (ou aligned_alloc pour SIMD)
        float* ptr = new float[bytes / sizeof(float)];
        std::memset(ptr, 0, bytes);
        
        total_allocated_bytes_ += bytes;
        num_allocations_++;
        
        return BufferHandle(ptr, bytes, tag, &memory_guard_);
    }
    
    // Pool de buffers réutilisables (scratchpad)
    // Pour éviter alloc/free répétitifs dans les boucles
    BufferHandle get_scratchpad(size_t min_bytes, const std::string& tag = "scratchpad") {
        // Chercher buffer existant de taille suffisante
        auto it = scratchpad_pool_.find(tag);
        if (it != scratchpad_pool_.end() && it->second.size_bytes() >= min_bytes) {
            // Réutiliser
            auto buffer = std::move(it->second);
            scratchpad_pool_.erase(it);
            return buffer;
        }
        
        // Allouer nouveau
        return allocate_buffer(min_bytes, tag);
    }
    
    // Retourner buffer au pool pour réutilisation
    void return_scratchpad(BufferHandle&& buffer) {
        std::string tag = buffer.tag();
        scratchpad_pool_.emplace(tag, std::move(buffer));
    }
    
    // Stats (debug)
    size_t get_total_allocated() const { return total_allocated_bytes_; }
    size_t get_num_allocations() const { return num_allocations_; }
    size_t get_peak_usage() const { return memory_guard_.getPeakBytes(); }
    size_t get_current_usage() const { return memory_guard_.getCurrentBytes(); }
    
    // Vérification post-forward
    bool check_no_leaks() const {
        size_t current = memory_guard_.getCurrentBytes() / (1024 * 1024);
        // Après forward, current devrait être proche de 0 (ou weights only)
        // Tolérance: 10 MB pour les poids/états persistants
        return current < 10;
    }
    
    void reset_stats() {
        total_allocated_bytes_ = 0;
        num_allocations_ = 0;
    }
    
    // Vidage du pool (fin de batch)
    void clear_scratchpad_pool() {
        scratchpad_pool_.clear();
    }
    
private:
    MemoryGuard& memory_guard_;
    size_t max_ram_bytes_;
    
    // Stats
    size_t total_allocated_bytes_ = 0;
    size_t num_allocations_ = 0;
    
    // Pool de buffers réutilisables
    std::unordered_map<std::string, BufferHandle> scratchpad_pool_;
};

// ============================================================================
// Helper pour créer des Tensors depuis std::vector existants
// ============================================================================

inline TensorHandle wrap_tensor(std::vector<float>&& data,
                                 const std::vector<int>& shape,
                                 const std::string& name,
                                 MemoryGuard& guard) {
    TensorDescriptor desc(shape, "float32", name);
    
    // data déjà alloué, mais comptabiliser quand même
    if (!guard.requestAllocation(desc.size_bytes)) {
        throw std::runtime_error("wrap_tensor: would exceed memory limit");
    }
    
    return TensorHandle(std::move(data), desc, &guard);
}

// ============================================================================
// Helper macro pour mode strict
// ============================================================================

#ifndef MIMIR_STRICT_MODE
#define MIMIR_STRICT_MODE 1  // Par défaut: mode strict
#endif

#define RUNTIME_ERROR_STRICT(msg) \
    do { \
        if (MIMIR_STRICT_MODE) { \
            throw std::runtime_error(msg); \
        } else { \
            std::cerr << "⚠️  [PERMISSIVE MODE] " << msg << std::endl; \
        } \
    } while(0)

#define RUNTIME_CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            RUNTIME_ERROR_STRICT(msg); \
        } \
    } while(0)
