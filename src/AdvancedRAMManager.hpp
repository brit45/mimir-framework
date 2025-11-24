#ifndef __ADVANCED_RAM_MANAGER_HPP__
#define __ADVANCED_RAM_MANAGER_HPP__

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <optional>
#include <functional>

// Compression LZ4 (header-only lightweight implementation)
// Pour une vraie implémentation, utilisez: #include <lz4.h>
namespace SimpleLZ4 {
    // Version simplifiée pour démo - remplacer par vraie LZ4 en production
    inline std::vector<uint8_t> compress(const std::vector<uint8_t>& data) {
        // Simulation: compression ~50%
        std::vector<uint8_t> compressed;
        compressed.reserve(data.size() / 2 + 100);
        
        // Header: taille originale (4 bytes)
        uint32_t original_size = static_cast<uint32_t>(data.size());
        compressed.push_back((original_size >> 0) & 0xFF);
        compressed.push_back((original_size >> 8) & 0xFF);
        compressed.push_back((original_size >> 16) & 0xFF);
        compressed.push_back((original_size >> 24) & 0xFF);
        
        // Compression RLE simple (Run-Length Encoding)
        for (size_t i = 0; i < data.size(); ) {
            uint8_t value = data[i];
            size_t count = 1;
            
            while (i + count < data.size() && 
                   data[i + count] == value && 
                   count < 255) {
                count++;
            }
            
            if (count > 3) {
                // Run: marker(255) + count + value
                compressed.push_back(255);
                compressed.push_back(static_cast<uint8_t>(count));
                compressed.push_back(value);
            } else {
                // Literal: copier tel quel
                for (size_t j = 0; j < count; j++) {
                    compressed.push_back(value);
                }
            }
            
            i += count;
        }
        
        return compressed;
    }
    
    inline std::vector<uint8_t> decompress(const std::vector<uint8_t>& compressed) {
        if (compressed.size() < 4) return {};
        
        // Lire la taille originale
        uint32_t original_size = 
            (uint32_t)compressed[0] |
            ((uint32_t)compressed[1] << 8) |
            ((uint32_t)compressed[2] << 16) |
            ((uint32_t)compressed[3] << 24);
        
        std::vector<uint8_t> decompressed;
        decompressed.reserve(original_size);
        
        for (size_t i = 4; i < compressed.size(); ) {
            if (compressed[i] == 255 && i + 2 < compressed.size()) {
                // Run
                uint8_t count = compressed[i + 1];
                uint8_t value = compressed[i + 2];
                for (int j = 0; j < count; j++) {
                    decompressed.push_back(value);
                }
                i += 3;
            } else {
                // Literal
                decompressed.push_back(compressed[i]);
                i++;
            }
        }
        
        return decompressed;
    }
    
    inline float compressionRatio(size_t original, size_t compressed) {
        return original > 0 ? (float)compressed / (float)original : 1.0f;
    }
}

// Structure pour les statistiques d'accès
struct AccessStats {
    size_t access_count = 0;
    uint64_t last_access = 0;
    uint64_t first_access = 0;
    double avg_access_interval = 0.0;
    
    void recordAccess(uint64_t timestamp) {
        if (access_count == 0) {
            first_access = timestamp;
        } else {
            double interval = static_cast<double>(timestamp - last_access);
            avg_access_interval = (avg_access_interval * access_count + interval) / (access_count + 1);
        }
        last_access = timestamp;
        access_count++;
    }
    
    // Prédire si cet item sera utilisé bientôt
    bool predictWillBeUsedSoon(uint64_t current_time, double threshold = 2.0) const {
        if (access_count < 2) return false;
        
        double expected_next = last_access + avg_access_interval;
        double time_since = static_cast<double>(current_time - last_access);
        
        return time_since < (avg_access_interval * threshold);
    }
};

// Gestionnaire avancé de RAM avec toutes les optimisations
class AdvancedRAMManager {
public:
    static AdvancedRAMManager& instance() {
        static AdvancedRAMManager mgr;
        return mgr;
    }
    
    // Configuration
    struct Config {
        size_t max_ram_bytes = 10ULL * 1024 * 1024 * 1024; // 10 GB
        bool enable_compression = true;
        bool enable_async_loading = true;
        bool enable_prediction = true;
        bool enable_statistics = true;
        size_t preload_queue_size = 128; // Nombre d'items à précharger
        size_t worker_threads = 4;
    };
    
    void configure(const Config& cfg) {
        std::lock_guard<std::mutex> lock(mutex_);
        config_ = cfg;
        max_ram_bytes_ = cfg.max_ram_bytes;
        
        // Démarrer les threads si async activé
        if (cfg.enable_async_loading && !async_thread_running_) {
            startAsyncWorkers();
        }
    }
    
    // Allocation avec compression optionnelle
    bool allocate(const std::string& key, const std::vector<uint8_t>& data, bool compress = true) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Vérifier si déjà alloué
        if (allocations_.find(key) != allocations_.end()) {
            return true;
        }
        
        std::vector<uint8_t> storage_data = data;
        bool is_compressed = false;
        
        // Compression si activée et bénéfique
        if (compress && config_.enable_compression && data.size() > 1024) {
            auto compressed = SimpleLZ4::compress(data);
            float ratio = SimpleLZ4::compressionRatio(data.size(), compressed.size());
            
            if (ratio < 0.9f) { // Au moins 10% de gain
                storage_data = std::move(compressed);
                is_compressed = true;
                compression_savings_ += (data.size() - storage_data.size());
            }
        }
        
        size_t required = storage_data.size();
        
        // Vérifier la capacité
        if (!canAllocate(required)) {
            // Éviction LRU
            evictLRU(required);
            
            if (!canAllocate(required)) {
                return false; // Impossible même après éviction
            }
        }
        
        // Allouer
        AllocationInfo info;
        info.data = std::move(storage_data);
        info.original_size = data.size();
        info.is_compressed = is_compressed;
        info.access_count = 0;
        info.last_access = getCurrentTimestamp();
        
        allocations_[key] = std::move(info);
        current_ram_bytes_ += required;
        peak_ram_bytes_ = std::max(peak_ram_bytes_, current_ram_bytes_);
        
        // Statistiques
        if (config_.enable_statistics) {
            recordAllocation(key);
        }
        
        return true;
    }
    
    // Récupérer les données (décompression automatique)
    std::optional<std::vector<uint8_t>> get(const std::string& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = allocations_.find(key);
        if (it == allocations_.end()) {
            return std::nullopt;
        }
        
        auto& info = it->second;
        info.last_access = getCurrentTimestamp();
        info.access_count++;
        
        // Statistiques d'accès
        if (config_.enable_statistics) {
            access_stats_[key].recordAccess(info.last_access);
        }
        
        // Décompression si nécessaire
        if (info.is_compressed) {
            cache_hits_compressed_++;
            return SimpleLZ4::decompress(info.data);
        } else {
            cache_hits_uncompressed_++;
            return info.data;
        }
    }
    
    // Libérer une allocation
    void deallocate(const std::string& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = allocations_.find(key);
        if (it != allocations_.end()) {
            current_ram_bytes_ -= it->second.data.size();
            allocations_.erase(it);
        }
    }
    
    // Préchargement asynchrone
    void preloadAsync(const std::string& key, std::function<std::vector<uint8_t>()> loader) {
        if (!config_.enable_async_loading) {
            // Mode synchrone
            auto data = loader();
            allocate(key, data);
            return;
        }
        
        // Ajouter à la queue de préchargement
        {
            std::lock_guard<std::mutex> lock(preload_mutex_);
            preload_queue_.push({key, loader});
        }
        preload_cv_.notify_one();
    }
    
    // Prédire et précharger le working set
    void predictAndPreload(const std::vector<std::string>& recent_keys, 
                          std::function<std::vector<uint8_t>(const std::string&)> loader) {
        if (!config_.enable_prediction) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        uint64_t now = getCurrentTimestamp();
        
        // Identifier les items susceptibles d'être utilisés bientôt
        std::vector<std::string> to_preload;
        
        for (const auto& [key, stats] : access_stats_) {
            if (stats.predictWillBeUsedSoon(now)) {
                // Vérifier si pas déjà chargé
                if (allocations_.find(key) == allocations_.end()) {
                    to_preload.push_back(key);
                }
            }
        }
        
        // Précharger de manière asynchrone
        for (const auto& key : to_preload) {
            if (to_preload.size() > config_.preload_queue_size) break;
            
            preloadAsync(key, [&loader, key]() {
                return loader(key);
            });
        }
        
        if (!to_preload.empty()) {
            std::cout << "🔮 Prédiction: préchargement de " << to_preload.size() 
                     << " items" << std::endl;
        }
    }
    
    // Partitionnement par modalité
    struct ModalityStats {
        size_t text_bytes = 0;
        size_t image_bytes = 0;
        size_t audio_bytes = 0;
        size_t video_bytes = 0;
        size_t count_text = 0;
        size_t count_image = 0;
        size_t count_audio = 0;
        size_t count_video = 0;
    };
    
    ModalityStats getModalityStats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        ModalityStats stats;
        for (const auto& [key, info] : allocations_) {
            // Détecter la modalité par préfixe
            if (key.find("text_") == 0) {
                stats.text_bytes += info.data.size();
                stats.count_text++;
            } else if (key.find("img_") == 0) {
                stats.image_bytes += info.data.size();
                stats.count_image++;
            } else if (key.find("audio_") == 0) {
                stats.audio_bytes += info.data.size();
                stats.count_audio++;
            } else if (key.find("video_") == 0) {
                stats.video_bytes += info.data.size();
                stats.count_video++;
            }
        }
        
        return stats;
    }
    
    // Statistiques détaillées
    void printDetailedStats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::cout << "\n╔═══════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║        ADVANCED RAM MANAGER - STATISTIQUES           ║" << std::endl;
        std::cout << "╠═══════════════════════════════════════════════════════╣" << std::endl;
        
        // RAM
        std::cout << "║ 💾 RAM:                                               ║" << std::endl;
        std::cout << "║   Current:     " << std::setw(8) << (current_ram_bytes_ / 1024 / 1024) 
                  << " MB                         ║" << std::endl;
        std::cout << "║   Peak:        " << std::setw(8) << (peak_ram_bytes_ / 1024 / 1024) 
                  << " MB                         ║" << std::endl;
        std::cout << "║   Max:         " << std::setw(8) << (max_ram_bytes_ / 1024 / 1024) 
                  << " MB                         ║" << std::endl;
        std::cout << "║   Usage:       " << std::setw(6) << std::fixed << std::setprecision(1)
                  << getUsagePercent() << " %                           ║" << std::endl;
        
        // Allocations
        std::cout << "║ 📊 Allocations:                                       ║" << std::endl;
        std::cout << "║   Active:      " << std::setw(8) << allocations_.size() 
                  << "                            ║" << std::endl;
        std::cout << "║   Total:       " << std::setw(8) << total_allocations_ 
                  << "                            ║" << std::endl;
        std::cout << "║   Evictions:   " << std::setw(8) << total_evictions_ 
                  << "                            ║" << std::endl;
        
        // Compression
        if (config_.enable_compression) {
            std::cout << "║ 🗜️  Compression:                                       ║" << std::endl;
            std::cout << "║   Savings:     " << std::setw(8) << (compression_savings_ / 1024 / 1024) 
                      << " MB                         ║" << std::endl;
            
            size_t compressed_count = 0;
            for (const auto& [k, v] : allocations_) {
                if (v.is_compressed) compressed_count++;
            }
            std::cout << "║   Compressed:  " << std::setw(8) << compressed_count 
                      << " / " << allocations_.size() << "                     ║" << std::endl;
        }
        
        // Cache
        std::cout << "║ 💨 Cache:                                             ║" << std::endl;
        size_t total_hits = cache_hits_compressed_ + cache_hits_uncompressed_;
        std::cout << "║   Hits:        " << std::setw(8) << total_hits 
                  << "                            ║" << std::endl;
        std::cout << "║   Misses:      " << std::setw(8) << cache_misses_ 
                  << "                            ║" << std::endl;
        if (total_hits + cache_misses_ > 0) {
            float hit_rate = 100.0f * total_hits / (total_hits + cache_misses_);
            std::cout << "║   Hit rate:    " << std::setw(6) << std::fixed << std::setprecision(1)
                      << hit_rate << " %                           ║" << std::endl;
        }
        
        // Modalités
        auto modal_stats = getModalityStats();
        std::cout << "║ 🎨 Modalités:                                         ║" << std::endl;
        std::cout << "║   Text:        " << std::setw(8) << modal_stats.count_text 
                  << " items (" << (modal_stats.text_bytes / 1024) << " KB)       ║" << std::endl;
        std::cout << "║   Images:      " << std::setw(8) << modal_stats.count_image 
                  << " items (" << (modal_stats.image_bytes / 1024 / 1024) << " MB)       ║" << std::endl;
        std::cout << "║   Audio:       " << std::setw(8) << modal_stats.count_audio 
                  << " items (" << (modal_stats.audio_bytes / 1024) << " KB)       ║" << std::endl;
        std::cout << "║   Video:       " << std::setw(8) << modal_stats.count_video 
                  << " items (" << (modal_stats.video_bytes / 1024 / 1024) << " MB)       ║" << std::endl;
        
        std::cout << "╚═══════════════════════════════════════════════════════╝" << std::endl;
    }
    
    // Getters
    size_t getCurrentRAM() const { 
        std::lock_guard<std::mutex> lock(mutex_);
        return current_ram_bytes_; 
    }
    
    size_t getPeakRAM() const { 
        std::lock_guard<std::mutex> lock(mutex_);
        return peak_ram_bytes_; 
    }
    
    float getUsagePercent() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return 100.0f * static_cast<float>(current_ram_bytes_) / static_cast<float>(max_ram_bytes_);
    }
    
    // Nettoyage
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        allocations_.clear();
        access_stats_.clear();
        current_ram_bytes_ = 0;
    }
    
    // Arrêt propre des workers (à appeler avant la fin du programme)
    void shutdown() {
        stopAsyncWorkers();
    }
    
    ~AdvancedRAMManager() {
        stopAsyncWorkers();
    }
    
private:
    AdvancedRAMManager() = default;
    
    struct AllocationInfo {
        std::vector<uint8_t> data;
        size_t original_size = 0;
        bool is_compressed = false;
        uint64_t last_access = 0;
        size_t access_count = 0;
    };
    
    struct PreloadTask {
        std::string key;
        std::function<std::vector<uint8_t>()> loader;
    };
    
    mutable std::mutex mutex_;
    mutable std::mutex preload_mutex_;
    std::condition_variable preload_cv_;
    
    Config config_;
    size_t max_ram_bytes_ = 10ULL * 1024 * 1024 * 1024;
    size_t current_ram_bytes_ = 0;
    size_t peak_ram_bytes_ = 0;
    size_t compression_savings_ = 0;
    size_t total_allocations_ = 0;
    size_t total_evictions_ = 0;
    size_t cache_hits_compressed_ = 0;
    size_t cache_hits_uncompressed_ = 0;
    size_t cache_misses_ = 0;
    
    std::unordered_map<std::string, AllocationInfo> allocations_;
    std::unordered_map<std::string, AccessStats> access_stats_;
    std::queue<PreloadTask> preload_queue_;
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> async_thread_running_{false};
    std::atomic<bool> stop_workers_{false};
    
    bool canAllocate(size_t bytes) const {
        return (current_ram_bytes_ + bytes) <= max_ram_bytes_;
    }
    
    void evictLRU(size_t bytes_needed) {
        // Trier par last_access (LRU)
        std::vector<std::pair<std::string, uint64_t>> items;
        for (const auto& [key, info] : allocations_) {
            items.push_back({key, info.last_access});
        }
        
        std::sort(items.begin(), items.end(), 
                 [](const auto& a, const auto& b) { return a.second < b.second; });
        
        size_t freed = 0;
        size_t count = 0;
        
        for (const auto& [key, _] : items) {
            if (freed >= bytes_needed) break;
            
            auto it = allocations_.find(key);
            if (it != allocations_.end()) {
                freed += it->second.data.size();
                current_ram_bytes_ -= it->second.data.size();
                allocations_.erase(it);
                count++;
                total_evictions_++;
            }
        }
        
        if (count > 0) {
            std::cout << "⟳ LRU éviction: " << count << " items, libéré " 
                     << (freed / 1024 / 1024) << " MB" << std::endl;
        }
    }
    
    void recordAllocation(const std::string& key) {
        total_allocations_++;
    }
    
    uint64_t getCurrentTimestamp() const {
        static uint64_t counter = 0;
        return ++counter;
    }
    
    void startAsyncWorkers() {
        stop_workers_ = false;
        async_thread_running_ = true;
        
        for (size_t i = 0; i < config_.worker_threads; i++) {
            worker_threads_.emplace_back([this]() {
                asyncWorkerLoop();
            });
        }
    }
    
    void stopAsyncWorkers() {
        if (async_thread_running_) {
            stop_workers_ = true;
            preload_cv_.notify_all();
            
            // Timeout pour éviter les blocages infinis
            auto start = std::chrono::steady_clock::now();
            for (auto& thread : worker_threads_) {
                if (thread.joinable()) {
                    // Essayer de joindre avec timeout
                    auto elapsed = std::chrono::steady_clock::now() - start;
                    if (elapsed < std::chrono::seconds(2)) {
                        thread.join();
                    } else {
                        // Détacher si timeout dépassé
                        thread.detach();
                    }
                }
            }
            
            worker_threads_.clear();
            async_thread_running_ = false;
        }
    }
    
    void asyncWorkerLoop() {
        while (!stop_workers_) {
            PreloadTask task;
            
            {
                std::unique_lock<std::mutex> lock(preload_mutex_);
                preload_cv_.wait(lock, [this]() {
                    return !preload_queue_.empty() || stop_workers_;
                });
                
                if (stop_workers_) break;
                
                if (!preload_queue_.empty()) {
                    task = std::move(preload_queue_.front());
                    preload_queue_.pop();
                } else {
                    continue;
                }
            }
            
            // Charger et allouer
            try {
                auto data = task.loader();
                allocate(task.key, data);
            } catch (const std::exception& e) {
                std::cerr << "⚠️  Erreur préchargement async: " << e.what() << std::endl;
            }
        }
    }
};

#endif // __ADVANCED_RAM_MANAGER_HPP__
