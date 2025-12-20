#ifndef __MEMORY_GUARD_HPP__
#define __MEMORY_GUARD_HPP__

#include <cstddef>
#include <atomic>
#include <iostream>
#include <stdexcept>

/**
 * MemoryGuard - Garde-fou strict pour limiter l'allocation RAM
 * Contrôle TOUTES les allocations de tenseurs et refuse si dépassement
 */
class MemoryGuard {
public:
    static MemoryGuard& instance() {
        static MemoryGuard guard;
        return guard;
    }
    
    // Configuration
    void setLimit(size_t bytes) {
        max_bytes_ = bytes;
        std::cout << "🛡️  MemoryGuard: Limite stricte définie à " 
                  << (bytes / 1024 / 1024 / 1024) << " GB" << std::endl;
    }
    
    size_t getLimit() const { return max_bytes_; }
    
    // Vérifier et enregistrer une allocation
    bool requestAllocation(size_t bytes, const std::string& tag = "") {
        size_t current = current_bytes_.load();
        size_t new_total = current + bytes;
        
        if (new_total > max_bytes_) {
            std::cerr << "❌ MemoryGuard: ALLOCATION REFUSÉE!" << std::endl;
            std::cerr << "   Demandé: " << (bytes / 1024 / 1024) << " MB" << std::endl;
            std::cerr << "   Actuel: " << (current / 1024 / 1024) << " MB" << std::endl;
            std::cerr << "   Nouveau total: " << (new_total / 1024 / 1024) << " MB" << std::endl;
            std::cerr << "   Limite: " << (max_bytes_ / 1024 / 1024) << " MB" << std::endl;
            if (!tag.empty()) {
                std::cerr << "   Tag: " << tag << std::endl;
            }
            return false;
        }
        
        current_bytes_ += bytes;
        if (current_bytes_ > peak_bytes_) {
            peak_bytes_ = current_bytes_.load();
        }
        
        allocations_count_++;
        
        if (!tag.empty() && bytes > 100 * 1024 * 1024) { // Log si > 100 MB
            std::cout << "📊 Allocation: " << (bytes / 1024 / 1024) << " MB"
                      << " (" << tag << ")"
                      << " - Total: " << (current_bytes_ / 1024 / 1024) << " MB"
                      << " / " << (max_bytes_ / 1024 / 1024) << " MB" << std::endl;
        }
        
        return true;
    }
    
    // Libérer une allocation
    void releaseAllocation(size_t bytes) {
        if (current_bytes_ >= bytes) {
            current_bytes_ -= bytes;
        } else {
            current_bytes_ = 0;
        }
        deallocations_count_++;
    }
    
    // Statistiques
    size_t getCurrentBytes() const { return current_bytes_.load(); }
    size_t getPeakBytes() const { return peak_bytes_.load(); }
    float getUsagePercent() const {
        return 100.0f * static_cast<float>(current_bytes_) / static_cast<float>(max_bytes_);
    }
    
    void printStats() const {
        std::cout << "\n╔═══════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║           MEMORY GUARD - STATISTIQUES                ║" << std::endl;
        std::cout << "╠═══════════════════════════════════════════════════════╣" << std::endl;
        std::cout << "║ Limite:     " << std::setw(10) << (max_bytes_ / 1024 / 1024) 
                  << " MB                         ║" << std::endl;
        std::cout << "║ Actuel:     " << std::setw(10) << (current_bytes_ / 1024 / 1024) 
                  << " MB                         ║" << std::endl;
        std::cout << "║ Pic:        " << std::setw(10) << (peak_bytes_ / 1024 / 1024) 
                  << " MB                         ║" << std::endl;
        std::cout << "║ Utilisation:" << std::setw(9) << std::fixed << std::setprecision(1)
                  << getUsagePercent() << " %                          ║" << std::endl;
        std::cout << "║ Allocations:" << std::setw(9) << allocations_count_ 
                  << "                              ║" << std::endl;
        std::cout << "║ Libérations:" << std::setw(9) << deallocations_count_ 
                  << "                              ║" << std::endl;
        std::cout << "╚═══════════════════════════════════════════════════════╝" << std::endl;
    }
    
    void reset() {
        current_bytes_ = 0;
        peak_bytes_ = 0;
        allocations_count_ = 0;
        deallocations_count_ = 0;
    }
    
private:
    MemoryGuard() = default;
    ~MemoryGuard() = default;
    
    std::atomic<size_t> current_bytes_{0};
    std::atomic<size_t> peak_bytes_{0};
    std::atomic<size_t> max_bytes_{10ULL * 1024 * 1024 * 1024}; // 10 GB par défaut
    std::atomic<size_t> allocations_count_{0};
    std::atomic<size_t> deallocations_count_{0};
};

#endif // __MEMORY_GUARD_HPP__
