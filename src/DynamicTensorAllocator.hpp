#ifndef __DYNAMIC_TENSOR_ALLOCATOR_HPP__
#define __DYNAMIC_TENSOR_ALLOCATOR_HPP__

#include "AdvancedRAMManager.hpp"
#include "MemoryGuard.hpp"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <cstring>

/**
 * DynamicTensorAllocator - Allocation RAM dynamique pour tenseurs
 * 
 * Combine AdvancedRAMManager (compression, éviction) + MemoryGuard (limite stricte)
 * pour gérer intelligemment la mémoire des tenseurs avec:
 * - Allocation à la demande (lazy loading)
 * - Compression automatique des tenseurs inactifs
 * - Éviction LRU si limite atteinte
 * - Rechargement transparent depuis disque/mémoire compressée
 */
class DynamicTensorAllocator {
public:
    static DynamicTensorAllocator& instance() {
        static DynamicTensorAllocator alloc;
        return alloc;
    }
    
    struct TensorHandle {
        size_t size = 0;
        bool is_loaded = false;
        bool is_compressed = false;
        std::string cache_key;
        float* data_ptr = nullptr;  // Pointeur vers données actives
    };
    
    // Configuration
    void configure(size_t max_ram_gb, bool enable_compression = true) {
        max_ram_bytes_ = max_ram_gb * 1024ULL * 1024ULL * 1024ULL;
        compression_enabled_ = enable_compression;
        
        // Configurer MemoryGuard
        auto& guard = MemoryGuard::instance();
        guard.setLimit(max_ram_bytes_);
        
        // Configurer AdvancedRAMManager
        AdvancedRAMManager::Config ram_config;
        ram_config.max_ram_bytes = max_ram_bytes_;
        ram_config.enable_compression = enable_compression;
        ram_config.enable_async_loading = false;  // Synchrone pour contrôle strict
        ram_config.enable_prediction = false;
        ram_config.enable_statistics = true;
        ram_config.worker_threads = 2;
        
        auto& ram_mgr = AdvancedRAMManager::instance();
        ram_mgr.configure(ram_config);
        
        std::cout << "🚀 DynamicTensorAllocator configuré:" << std::endl;
        std::cout << "   - Limite RAM: " << max_ram_gb << " GB" << std::endl;
        std::cout << "   - Compression: " << (enable_compression ? "activée" : "désactivée") << std::endl;
    }
    
    // Allouer un tenseur (retourne un handle)
    TensorHandle* allocateTensor(size_t num_elements, const std::string& tag = "") {
        std::lock_guard<std::mutex> lock(mutex_);
        
        size_t bytes_needed = num_elements * sizeof(float);
        auto& guard = MemoryGuard::instance();
        
        // Vérifier si allocation possible
        if (!guard.requestAllocation(bytes_needed, tag)) {
            // Tenter éviction LRU via AdvancedRAMManager
            std::cout << "⚠️  Mémoire insuffisante, tentative d'éviction..." << std::endl;
            evictLRU(bytes_needed);
            
            // Réessayer
            if (!guard.requestAllocation(bytes_needed, tag)) {
                std::cerr << "❌ Impossible d'allouer tenseur même après éviction!" << std::endl;
                return nullptr;
            }
        }
        
        // Créer le handle
        auto handle = std::make_unique<TensorHandle>();
        handle->size = num_elements;
        handle->is_loaded = false;
        handle->is_compressed = false;
        handle->cache_key = tag + "_" + std::to_string(next_id_++);
        handle->data_ptr = nullptr;
        
        handles_[handle->cache_key] = std::move(handle);
        return handles_[handle->cache_key].get();
    }
    
    // Obtenir les données d'un tenseur (chargement à la demande)
    float* getTensorData(TensorHandle* handle) {
        if (!handle) return nullptr;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Si déjà chargé, retourner
        if (handle->is_loaded && handle->data_ptr) {
            access_order_[handle->cache_key] = ++access_counter_;
            return handle->data_ptr;
        }
        
        // Charger depuis AdvancedRAMManager ou allouer
        auto& ram_mgr = AdvancedRAMManager::instance();
        
        if (handle->is_compressed) {
            // Décompresser depuis RAMManager
            auto data = ram_mgr.get(handle->cache_key);
            if (data.has_value()) {
                handle->data_ptr = reinterpret_cast<float*>(
                    malloc(handle->size * sizeof(float)));
                if (handle->data_ptr) {
                    memcpy(handle->data_ptr, data->data(), 
                           handle->size * sizeof(float));
                    handle->is_loaded = true;
                    handle->is_compressed = false;
                    access_order_[handle->cache_key] = ++access_counter_;
                    return handle->data_ptr;
                }
            }
        } else {
            // Allocation fraîche
            handle->data_ptr = reinterpret_cast<float*>(
                malloc(handle->size * sizeof(float)));
            if (handle->data_ptr) {
                // Initialiser à 0
                memset(handle->data_ptr, 0, handle->size * sizeof(float));
                handle->is_loaded = true;
                access_order_[handle->cache_key] = ++access_counter_;
                return handle->data_ptr;
            }
        }
        
        return nullptr;
    }
    
    // Compresser un tenseur (libère RAM active, stocke compressé)
    void compressTensor(TensorHandle* handle) {
        if (!handle || !handle->is_loaded || !handle->data_ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (!compression_enabled_) return;
        
        auto& ram_mgr = AdvancedRAMManager::instance();
        
        // Copier vers vector pour RAMManager
        std::vector<uint8_t> data(handle->size * sizeof(float));
        memcpy(data.data(), handle->data_ptr, handle->size * sizeof(float));
        
        // Stocker compressé
        if (ram_mgr.allocate(handle->cache_key, data, true)) {
            // Libérer mémoire active
            free(handle->data_ptr);
            handle->data_ptr = nullptr;
            handle->is_loaded = false;
            handle->is_compressed = true;
            
            // Mettre à jour MemoryGuard
            auto& guard = MemoryGuard::instance();
            guard.releaseAllocation(handle->size * sizeof(float));
        }
    }
    
    // Libérer complètement un tenseur
    void freeTensor(TensorHandle* handle) {
        if (!handle) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto& ram_mgr = AdvancedRAMManager::instance();
        auto& guard = MemoryGuard::instance();
        
        // Libérer données actives
        if (handle->data_ptr) {
            free(handle->data_ptr);
            handle->data_ptr = nullptr;
            guard.releaseAllocation(handle->size * sizeof(float));
        }
        
        // Libérer depuis RAMManager
        if (handle->is_compressed) {
            ram_mgr.deallocate(handle->cache_key);
        }
        
        // Supprimer handle
        access_order_.erase(handle->cache_key);
        handles_.erase(handle->cache_key);
    }
    
    // Statistiques
    void printStats() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::cout << "\n╔═══════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║      DYNAMIC TENSOR ALLOCATOR - STATISTIQUES         ║" << std::endl;
        std::cout << "╠═══════════════════════════════════════════════════════╣" << std::endl;
        
        size_t loaded_count = 0;
        size_t compressed_count = 0;
        size_t total_size = 0;
        
        for (const auto& [key, handle] : handles_) {
            if (handle->is_loaded) loaded_count++;
            if (handle->is_compressed) compressed_count++;
            total_size += handle->size * sizeof(float);
        }
        
        std::cout << "║ Tenseurs totaux:  " << handles_.size() << std::endl;
        std::cout << "║ Chargés:          " << loaded_count << std::endl;
        std::cout << "║ Compressés:       " << compressed_count << std::endl;
        std::cout << "║ Taille totale:    " << (total_size / 1024 / 1024) << " MB" << std::endl;
        std::cout << "╚═══════════════════════════════════════════════════════╝" << std::endl;
        
        // Afficher stats MemoryGuard
        MemoryGuard::instance().printStats();
        
        // Afficher stats RAMManager
        AdvancedRAMManager::instance().printDetailedStats();
    }
    
    size_t getTensorCount() const { return handles_.size(); }
    size_t getLoadedCount() const {
        size_t count = 0;
        for (const auto& [k, h] : handles_) {
            if (h->is_loaded) count++;
        }
        return count;
    }
    
private:
    DynamicTensorAllocator() = default;
    
    void evictLRU(size_t bytes_needed) {
        // Trier par ordre d'accès (LRU)
        std::vector<std::pair<std::string, uint64_t>> items;
        for (const auto& [key, timestamp] : access_order_) {
            items.push_back({key, timestamp});
        }
        
        std::sort(items.begin(), items.end(),
                 [](const auto& a, const auto& b) { return a.second < b.second; });
        
        size_t freed = 0;
        for (const auto& [key, _] : items) {
            if (freed >= bytes_needed) break;
            
            auto it = handles_.find(key);
            if (it != handles_.end() && it->second->is_loaded) {
                compressTensor(it->second.get());
                freed += it->second->size * sizeof(float);
            }
        }
        
        std::cout << "⟳ Éviction LRU: " << (freed / 1024 / 1024) << " MB libérés" << std::endl;
    }
    
    std::mutex mutex_;
    size_t max_ram_bytes_ = 10ULL * 1024 * 1024 * 1024;
    bool compression_enabled_ = true;
    size_t next_id_ = 0;
    uint64_t access_counter_ = 0;
    
    std::unordered_map<std::string, std::unique_ptr<TensorHandle>> handles_;
    std::unordered_map<std::string, uint64_t> access_order_;
};

#endif // __DYNAMIC_TENSOR_ALLOCATOR_HPP__
