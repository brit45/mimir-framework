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

#include <filesystem>
#include <fstream>

// Compression LZ4 (optionnelle)
#ifdef ENABLE_LZ4
#include <lz4.h>
#endif

namespace LZ4Compression {
#ifdef ENABLE_LZ4
    inline std::vector<uint8_t> compress(const std::vector<uint8_t>& data) {
        if (data.empty()) return {};

        int src_size = static_cast<int>(data.size());
        int max_dst_size = LZ4_compressBound(src_size);

        std::vector<uint8_t> compressed(max_dst_size + 4); // +4 pour le header

        compressed[0] = (src_size >> 0) & 0xFF;
        compressed[1] = (src_size >> 8) & 0xFF;
        compressed[2] = (src_size >> 16) & 0xFF;
        compressed[3] = (src_size >> 24) & 0xFF;

        int compressed_size = LZ4_compress_default(
            reinterpret_cast<const char*>(data.data()),
            reinterpret_cast<char*>(compressed.data() + 4),
            src_size,
            max_dst_size
        );

        if (compressed_size <= 0) {
            return data;
        }

        compressed.resize(compressed_size + 4);
        return compressed;
    }

    inline std::vector<uint8_t> decompress(const std::vector<uint8_t>& compressed) {
        if (compressed.size() < 4) return {};

        int original_size =
            (int)compressed[0] |
            ((int)compressed[1] << 8) |
            ((int)compressed[2] << 16) |
            ((int)compressed[3] << 24);

        if (original_size <= 0 || original_size > 1000000000) {
            return {};
        }

        std::vector<uint8_t> decompressed(original_size);

        int decompressed_size = LZ4_decompress_safe(
            reinterpret_cast<const char*>(compressed.data() + 4),
            reinterpret_cast<char*>(decompressed.data()),
            static_cast<int>(compressed.size() - 4),
            original_size
        );

        if (decompressed_size != original_size) {
            std::cerr << "⚠️  Erreur décompression LZ4: " << decompressed_size << " vs " << original_size << std::endl;
            return {};
        }

        return decompressed;
    }
#else
    inline std::vector<uint8_t> compress(const std::vector<uint8_t>& data) {
        return data;
    }

    inline std::vector<uint8_t> decompress(const std::vector<uint8_t>& data) {
        return data;
    }
#endif

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
        bool enable_disk_spill = true;                 // Eviction non-destructive (sur disque)
        std::string spill_dir = ".mimir_spill";        // Dossier spill
        size_t preload_queue_size = 128; // Nombre d'items à précharger
        size_t worker_threads = 4;
    };
    
    void configure(const Config& cfg) {
        std::lock_guard<std::mutex> lock(mutex_);
        config_ = cfg;
        max_ram_bytes_ = cfg.max_ram_bytes;

        // Préparer le dossier de spill
        if (config_.enable_disk_spill) {
            try {
                std::filesystem::create_directories(config_.spill_dir);
            } catch (...) {
                std::cerr << "⚠️  AdvancedRAMManager: impossible de créer spill_dir='" << config_.spill_dir << "'" << std::endl;
                config_.enable_disk_spill = false;
            }
        }
        
        // Démarrer les threads si async activé
        if (cfg.enable_async_loading && !async_thread_running_) {
            startAsyncWorkers();
        }
    }
    
    // =============================
    // Système de Blocage d'Allocation
    // =============================
    
    void blockAllocations(bool block = true) {
        allocations_blocked_ = block;
        if (block) {
            std::cout << "🔒 AdvancedRAMManager: Allocations BLOQUÉES" << std::endl;
        } else {
            std::cout << "🔓 AdvancedRAMManager: Allocations DÉBLOQUÉES" << std::endl;
        }
    }
    
    bool isBlocked() const { return allocations_blocked_.load(); }
    
    void freezeAllocations(bool freeze = true) {
        freeze_mode_ = freeze;
        if (freeze) {
            std::cout << "❄️  AdvancedRAMManager: Mode FREEZE activé" << std::endl;
        } else {
            std::cout << "☀️  AdvancedRAMManager: Mode FREEZE désactivé" << std::endl;
        }
    }
    
    bool isFrozen() const { return freeze_mode_.load(); }
    
    // Allocation avec compression optionnelle
    bool allocate(const std::string& key, const std::vector<uint8_t>& data, bool compress = true) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Vérifier si les allocations sont bloquées
        if (allocations_blocked_.load()) {
            blocked_attempts_++;
            std::cerr << "🔒 AdvancedRAMManager: Allocation bloquée pour '" << key << "'" << std::endl;
            return false;
        }
        
        // Vérifier si en mode freeze
        if (freeze_mode_.load()) {
            frozen_attempts_++;
            std::cerr << "❄️  AdvancedRAMManager: Allocation gelée pour '" << key << "'" << std::endl;
            return false;
        }
        
        // Vérifier si déjà alloué
        if (allocations_.find(key) != allocations_.end()) {
            return true;
        }
        
        std::vector<uint8_t> storage_data = data;
        bool is_compressed = false;
        
        // Compression si activée et bénéfique
        if (compress && config_.enable_compression && data.size() > 1024) {
            auto compressed = LZ4Compression::compress(data);
            float ratio = LZ4Compression::compressionRatio(data.size(), compressed.size());
            
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
            evictLRU(required, &key);
            
            if (!canAllocate(required)) {
                return false; // Impossible même après éviction
            }
        }
        
        // Allouer
        AllocationInfo info;
        info.data = std::move(storage_data);
        info.original_size = data.size();
        info.is_compressed = is_compressed;
        info.on_disk = false;
        info.disk_path.clear();
        info.stored_bytes = required;
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

        // Reload depuis disque si nécessaire
        if (info.on_disk && info.data.empty()) {
            const size_t bytes_needed = info.stored_bytes;
            if (!canAllocate(bytes_needed)) {
                evictLRU(bytes_needed, &key);
            }
            if (!canAllocate(bytes_needed)) {
                return std::nullopt;
            }

            std::vector<uint8_t> loaded;
            if (!readSpillFile(info.disk_path, loaded)) {
                return std::nullopt;
            }
            info.data = std::move(loaded);
            current_ram_bytes_ += info.data.size();
            peak_ram_bytes_ = std::max(peak_ram_bytes_, current_ram_bytes_);
        }
        
        // Statistiques d'accès
        if (config_.enable_statistics) {
            access_stats_[key].recordAccess(info.last_access);
        }
        
        // Décompression si nécessaire
        if (info.is_compressed) {
            cache_hits_compressed_++;
            return LZ4Compression::decompress(info.data);
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
            if (!it->second.data.empty()) {
                current_ram_bytes_ -= it->second.data.size();
            }
            if (it->second.on_disk && !it->second.disk_path.empty()) {
                safeRemoveSpillFile(it->second.disk_path);
                if (current_disk_bytes_ >= it->second.stored_bytes) {
                    current_disk_bytes_ -= it->second.stored_bytes;
                } else {
                    current_disk_bytes_ = 0;
                }
            }
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
        for (auto& [_, info] : allocations_) {
            if (info.on_disk && !info.disk_path.empty()) {
                safeRemoveSpillFile(info.disk_path);
            }
        }
        allocations_.clear();
        access_stats_.clear();
        current_ram_bytes_ = 0;
        current_disk_bytes_ = 0;
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
        bool on_disk = false;
        std::string disk_path;
        size_t stored_bytes = 0; // bytes stockés (data.size() si en RAM, sinon taille du fichier spill)
        uint64_t last_access = 0;
        size_t access_count = 0;
    };

    static inline std::string hex_u64(uint64_t v) {
        static const char* hex = "0123456789abcdef";
        std::string out(16, '0');
        for (int i = 15; i >= 0; --i) {
            out[static_cast<size_t>(i)] = hex[v & 0xFULL];
            v >>= 4;
        }
        return out;
    }

    std::string makeSpillPath(const std::string& key) const {
        const uint64_t h = static_cast<uint64_t>(std::hash<std::string>{}(key));
        std::filesystem::path dir(config_.spill_dir);
        std::filesystem::path name("mimir_spill_" + hex_u64(h) + ".bin");
        return (dir / name).string();
    }

    static inline bool writeSpillFile(const std::string& path, const std::vector<uint8_t>& data) {
        try {
            const std::filesystem::path p(path);
            const std::filesystem::path tmp = p.string() + ".tmp";
            {
                std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
                if (!f) return false;
                if (!data.empty()) {
                    f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
                }
                f.flush();
                if (!f) return false;
            }
            std::error_code ec;
            std::filesystem::rename(tmp, p, ec);
            if (ec) {
                std::filesystem::remove(tmp, ec);
                return false;
            }
            return true;
        } catch (...) {
            return false;
        }
    }

    static inline bool readSpillFile(const std::string& path, std::vector<uint8_t>& out) {
        try {
            std::ifstream f(path, std::ios::binary);
            if (!f) return false;
            f.seekg(0, std::ios::end);
            std::streamsize sz = f.tellg();
            if (sz < 0) return false;
            f.seekg(0, std::ios::beg);
            out.assign(static_cast<size_t>(sz), 0);
            if (sz > 0) {
                f.read(reinterpret_cast<char*>(out.data()), sz);
                if (!f) return false;
            }
            return true;
        } catch (...) {
            return false;
        }
    }

    static inline void safeRemoveSpillFile(const std::string& path) {
        try {
            std::error_code ec;
            std::filesystem::remove(path, ec);
            (void)ec;
        } catch (...) {
        }
    }

    bool spillToDisk(const std::string& key, AllocationInfo& info) {
        if (!config_.enable_disk_spill) return false;
        if (info.data.empty()) {
            // Rien à libérer en RAM. On considère que c'est déjà "spillé".
            if (!info.on_disk) {
                info.on_disk = true;
                info.disk_path = makeSpillPath(key);
            }
            return true;
        }

        if (!info.on_disk || info.disk_path.empty()) {
            info.disk_path = makeSpillPath(key);
        }

        if (!writeSpillFile(info.disk_path, info.data)) {
            return false;
        }

        // Mise à jour comptage disque
        if (!info.on_disk) {
            current_disk_bytes_ += info.data.size();
        } else {
            // déjà sur disque: on ne sait pas si la taille a changé; best-effort
            if (current_disk_bytes_ >= info.stored_bytes) current_disk_bytes_ -= info.stored_bytes;
            current_disk_bytes_ += info.data.size();
        }
        info.on_disk = true;
        info.stored_bytes = info.data.size();

        // Libérer RAM
        current_ram_bytes_ -= info.data.size();
        info.data.clear();
        info.data.shrink_to_fit();
        return true;
    }
    
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

    size_t current_disk_bytes_ = 0;
    
    std::unordered_map<std::string, AllocationInfo> allocations_;
    std::unordered_map<std::string, AccessStats> access_stats_;
    std::queue<PreloadTask> preload_queue_;
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> async_thread_running_{false};
    std::atomic<bool> stop_workers_{false};
    
    // Système de blocage
    std::atomic<bool> allocations_blocked_{false};
    std::atomic<bool> freeze_mode_{false};
    std::atomic<size_t> blocked_attempts_{0};
    std::atomic<size_t> frozen_attempts_{0};
    
    bool canAllocate(size_t bytes) const {
        return (current_ram_bytes_ + bytes) <= max_ram_bytes_;
    }
    
    void evictLRU(size_t bytes_needed, const std::string* exclude_key = nullptr) {
        // Trier par last_access (LRU)
        std::vector<std::pair<std::string, uint64_t>> items;
        for (const auto& [key, info] : allocations_) {
            if (exclude_key && key == *exclude_key) continue;
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
                auto& info = it->second;

                // Si déjà spillé mais encore résident en RAM, on peut juste dropper data.
                if (info.on_disk && !info.data.empty()) {
                    freed += info.data.size();
                    current_ram_bytes_ -= info.data.size();
                    info.data.clear();
                    info.data.shrink_to_fit();
                    count++;
                    total_evictions_++;
                    continue;
                }

                // Sinon tenter spill-to-disk si activé
                if (!info.on_disk && config_.enable_disk_spill) {
                    const size_t before = info.data.size();
                    if (spillToDisk(key, info)) {
                        freed += before;
                        count++;
                        total_evictions_++;
                        continue;
                    }
                }

                // Fallback destructif
                freed += info.data.size();
                current_ram_bytes_ -= info.data.size();
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
