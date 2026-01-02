#ifndef __ASYNC_MONITOR_HPP__
#define __ASYNC_MONITOR_HPP__

#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <memory>
#include "HtopDisplay.hpp"
#include "Visualizer.hpp"

/**
 * AsyncMonitor - Gestion asynchrone de HtopDisplay et Visualizer
 * 
 * Exécute les moniteurs dans des threads séparés pour ne pas bloquer
 * le processus principal. Synchronisation périodique des métriques.
 */
class AsyncMonitor {
public:
    struct Metrics {
        int epoch = 0;
        int total_epochs = 0;
        int batch = 0;
        int total_batches = 0;
        float loss = 0.0f;
        float avg_loss = 0.0f;
        float lr = 0.0f;
        int batch_time_ms = 0;
        size_t memory_mb = 0;
        size_t memory_freed = 0;
        float bps = 0.0f;
        size_t params = 0;
        float timestep = 0.0f;
        float kl = 0.0f;
        float wass = 0.0f;
        float ent = 0.0f;
        float mom = 0.0f;
        float spat = 0.0f;
        float temp = 0.0f;
        float mse = 0.0f;
        float grad_norm = 0.0f;
        float grad_max = 0.0f;

        // Optimizer (for display/debug)
        int opt_type = 0;          // 0=SGD, 1=ADAM, 2=ADAMW
        int opt_step = 0;
        float opt_beta1 = 0.0f;
        float opt_beta2 = 0.0f;
        float opt_eps = 0.0f;
        float opt_weight_decay = 0.0f;
    };
    
    AsyncMonitor() : running_(false), update_interval_ms_(100) {}
    
    ~AsyncMonitor() {
        stop();
    }
    
    // Démarrer les moniteurs
    void start(bool enable_htop = true, bool enable_viz = false, 
               const json& viz_config = json()) {
        if (running_) return;
        
        running_ = true;
        
        if (enable_htop) {
            htop_ = std::make_shared<HtopDisplay>();
            htop_->hideCursor();
            htop_->clearScreen();
            
            htop_thread_ = std::thread([this]() {
                htopLoop();
            });
        }
        
        if (enable_viz) {
            viz_ = std::make_shared<Visualizer>(viz_config);
            if (viz_->initialize()) {
                viz_thread_ = std::thread([this]() {
                    vizLoop();
                });
            }
        }
    }
    
    // Arrêter les moniteurs
    void stop() {
        if (!running_) return;
        
        running_ = false;
        cv_.notify_all();
        
        if (htop_thread_.joinable()) {
            htop_thread_.join();
        }
        
        if (viz_thread_.joinable()) {
            viz_thread_.join();
        }
        
        if (htop_) {
            htop_->showCursor();
        }
    }
    
    // Mettre à jour les métriques (thread-safe)
    void updateMetrics(const Metrics& metrics) {
        std::lock_guard<std::mutex> lock(mutex_);
        metrics_ = metrics;
        has_update_ = true;
    }
    
    // Définir l'intervalle de mise à jour (ms)
    void setUpdateInterval(int ms) {
        update_interval_ms_ = ms;
    }
    
    // Ajouter une image au visualiseur
    void addImage(const std::vector<uint8_t>& pixels, const std::string& prompt) {
        if (!viz_) return;
        
        std::lock_guard<std::mutex> lock(viz_mutex_);
        pending_images_.push_back({pixels, prompt});
    }
    
    // Accesseurs
    std::shared_ptr<HtopDisplay> getHtop() { return htop_; }
    std::shared_ptr<Visualizer> getViz() { return viz_; }
    bool isRunning() const { return running_; }
    
private:
    void htopLoop() {
        while (running_) {
            Metrics local_metrics;
            bool has_data = false;
            
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (has_update_) {
                    local_metrics = metrics_;
                    has_data = true;
                    has_update_ = false;
                }
            }
            
            if (has_data && htop_) {
                htop_->updateStats(
                    local_metrics.epoch, local_metrics.total_epochs,
                    local_metrics.batch, local_metrics.total_batches,
                    local_metrics.loss, local_metrics.avg_loss,
                    local_metrics.lr, local_metrics.batch_time_ms,
                    local_metrics.memory_mb, local_metrics.memory_freed,
                    local_metrics.bps, local_metrics.params,
                    local_metrics.timestep, local_metrics.kl,
                    local_metrics.wass, local_metrics.ent,
                    local_metrics.mom, local_metrics.spat,
                    local_metrics.temp, local_metrics.mse,
                    local_metrics.grad_norm, local_metrics.grad_max,
                    local_metrics.opt_type, local_metrics.opt_step,
                    local_metrics.opt_beta1, local_metrics.opt_beta2,
                    local_metrics.opt_eps, local_metrics.opt_weight_decay
                );
                htop_->render();
            }
            
            std::this_thread::sleep_for(
                std::chrono::milliseconds(update_interval_ms_));
        }
    }
    
    void vizLoop() {
        while (running_ && viz_ && viz_->isOpen()) {
            viz_->processEvents();
            
            // Mettre à jour métriques
            Metrics local_metrics;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                local_metrics = metrics_;
            }
            
            viz_->updateMetrics(
                local_metrics.epoch, local_metrics.batch,
                local_metrics.loss, local_metrics.lr,
                local_metrics.mse, local_metrics.kl,
                local_metrics.wass, local_metrics.ent,
                local_metrics.mom, local_metrics.spat,
                local_metrics.temp
            );
            viz_->addLossPoint(local_metrics.loss);
            
            // Ajouter images en attente
            {
                std::lock_guard<std::mutex> lock(viz_mutex_);
                for (const auto& img : pending_images_) {
                    viz_->addGeneratedImage(img.pixels, img.prompt);
                }
                pending_images_.clear();
            }
            
            viz_->update();
            
            std::this_thread::sleep_for(
                std::chrono::milliseconds(update_interval_ms_));
        }
    }
    
    std::shared_ptr<HtopDisplay> htop_;
    std::shared_ptr<Visualizer> viz_;
    
    std::thread htop_thread_;
    std::thread viz_thread_;
    
    std::mutex mutex_;
    std::mutex viz_mutex_;
    std::condition_variable cv_;
    
    std::atomic<bool> running_;
    std::atomic<bool> has_update_;
    int update_interval_ms_;
    
    Metrics metrics_;
    
    struct PendingImage {
        std::vector<uint8_t> pixels;
        std::string prompt;
    };
    std::vector<PendingImage> pending_images_;
};

#endif // __ASYNC_MONITOR_HPP__
