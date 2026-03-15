#ifndef __ASYNC_MONITOR_HPP__
#define __ASYNC_MONITOR_HPP__

#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <cstdint>
#include <vector>
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
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

        // Warmup KL: beta effectif appliqué (après warmup)
        float kl_beta_effective = 0.0f;

        // Nom du type de loss de reconstruction utilisée par le modèle (ex: "MSE", "L1").
        // Laisser vide si non applicable.
        std::string recon_loss_type;

        // Validation (dernières métriques connues)
        bool val_has = false;
        bool val_ok = false;
        bool val_in_progress = false;
        int val_step = 0;
        int val_items = 0;
        int val_done = 0;
        int val_total = 0;
        float val_recon = 0.0f;
        float val_kl = 0.0f;
        float val_align = 0.0f;

        // Optimizer (for display/debug)
        int opt_type = 0;          // 0=SGD, 1=ADAM, 2=ADAMW
        int opt_step = 0;
        float opt_beta1 = 0.0f;
        float opt_beta2 = 0.0f;
        float opt_eps = 0.0f;
        float opt_weight_decay = 0.0f;
    };
    
    AsyncMonitor()
        : running_(false)
        , has_update_(false)
        , update_interval_ms_(100)
        , htop_update_interval_ms_(100)
        , viz_update_interval_ms_(16) {}
    
    ~AsyncMonitor() {
        stop();
    }
    
    // Démarrer les moniteurs
    void start(bool enable_htop = true, bool enable_viz = false, 
               const json& viz_config = json()) {
        // NOTE: start() doit être idempotent.
        // On autorise le cas "htop déjà démarré" puis "activation de la viz" plus tard.
        if (!running_) {
            running_ = true;
        }

        if (enable_htop && !htop_) {
            // UI sur un fd dédié (tty) pour que stdout/stderr puissent être redirigés
            // sans casser le rendu.
            if (ui_fd_ < 0) {
                ui_fd_ = ::open("/dev/tty", O_WRONLY | O_CLOEXEC);
                if (ui_fd_ < 0) {
                    // fallback: dupliquer stdout AVANT redirection
                    ui_fd_ = ::dup(STDOUT_FILENO);
                }
            }

            if (ui_fd_ < 0) {
                // Dernier recours: écrire sur stdout (peut être capturé, mais évite fd invalide)
                ui_fd_ = STDOUT_FILENO;
            }

            htop_ = std::make_shared<HtopDisplay>(ui_fd_);
            // Par défaut, Visualizer exporte déjà loss_history.csv.
            // Si Visualizer est actif, on coupe le CSV côté Htop pour éviter des écritures concurrentes.
            htop_->setCsvEnabled(!enable_viz);
            htop_->enterAltScreen();
            htop_->hideCursor();
            htop_->clearScreen();

            // IMPORTANT: capturer stdout/stderr vers un buffer de logs quand le TUI est actif.
            // Cela évite que des printf cassent le rendu et permet de les afficher dans l'UI.
            startOutputCapture();

            htop_thread_ = std::thread([this]() {
                htopLoop();
            });
        }

        if (enable_viz && !viz_) {
            viz_ = std::make_shared<Visualizer>(viz_config);

            // Best-effort: permettre de configurer la cadence Viz depuis config.
            // Exemple: {"visualization": {"update_interval_ms": 16}}
            try {
                if (viz_config.contains("visualization")) {
                    const auto& v = viz_config["visualization"];
                    const int ms = v.value("update_interval_ms", static_cast<int>(viz_update_interval_ms_.load()));
                    if (ms > 0) {
                        viz_update_interval_ms_ = ms;
                    }
                }
            } catch (...) {
                // ignore
            }

            // IMPORTANT (SFML): la fenêtre / contexte OpenGL doivent être créés et
            // utilisés dans le même thread. On initialise donc la fenêtre DANS le
            // thread viz avant d'entrer dans la boucle de rendu.
            {
                std::lock_guard<std::mutex> lk(viz_init_mutex_);
                viz_init_done_ = false;
                viz_init_ok_ = false;
                viz_init_err_.clear();
            }

            viz_thread_ = std::thread([this]() {
                bool ok = false;
                std::string err;
                try {
                    ok = (viz_ && viz_->initialize());
                    if (!ok) {
                        err = "Visualizer::initialize() a échoué";
                    }
                } catch (const std::exception& e) {
                    ok = false;
                    err = e.what();
                }

                {
                    std::lock_guard<std::mutex> lk(viz_init_mutex_);
                    viz_init_ok_ = ok;
                    viz_init_done_ = true;
                    viz_init_err_ = err;
                }
                viz_init_cv_.notify_all();

                if (!ok) {
                    return;
                }

                vizLoop();

                // IMPORTANT: détruire la fenêtre SFML dans ce thread.
                if (viz_) {
                    viz_->shutdown();
                }
            });

            // Attendre que l'init viz soit terminée (succès ou échec).
            {
                std::unique_lock<std::mutex> lk(viz_init_mutex_);
                viz_init_cv_.wait_for(lk, std::chrono::seconds(2), [&]() { return viz_init_done_; });
            }
        }
    }
    
    // Arrêter les moniteurs
    void stop() {
        if (!running_) return;
        
        running_ = false;
        cv_.notify_all();
        
        if (htop_thread_.joinable() && std::this_thread::get_id() != htop_thread_.get_id()) {
            htop_thread_.join();
        }
        
        if (viz_thread_.joinable() && std::this_thread::get_id() != viz_thread_.get_id()) {
            viz_thread_.join();
        }

        // Arrêter la capture stdout/stderr (si active) AVANT de détruire htop_.
        stopOutputCapture();
        
        if (htop_) {
            htop_->leaveAltScreen();
            htop_->showCursor();
        }

        if (ui_fd_ >= 0 && ui_fd_ != STDOUT_FILENO && ui_fd_ != STDERR_FILENO) {
            ::close(ui_fd_);
        }
        ui_fd_ = -1;

        // Reset pour permettre un start() ultérieur.
        htop_.reset();
        viz_.reset();
    }

    // Statut init viz (utile pour bindings)
    bool vizInitOk() const {
        std::lock_guard<std::mutex> lk(viz_init_mutex_);
        return viz_init_done_ && viz_init_ok_;
    }
    std::string vizInitError() const {
        std::lock_guard<std::mutex> lk(viz_init_mutex_);
        return viz_init_err_;
    }
    
    // Mettre à jour les métriques (thread-safe)
    void updateMetrics(const Metrics& metrics) {
        std::lock_guard<std::mutex> lock(mutex_);
        metrics_ = metrics;
        has_update_ = true;
        metrics_version_.fetch_add(1, std::memory_order_relaxed);
    }

    // Mettre à jour uniquement l'état de validation (thread-safe) sans écraser
    // les métriques d'entraînement déjà poussées par le C++.
    void updateValidation(bool in_progress,
                          int step,
                          int done,
                          int total,
                          bool has,
                          bool ok,
                          float recon,
                          float kl,
                          float align) {
        std::lock_guard<std::mutex> lock(mutex_);
        metrics_.val_in_progress = in_progress;
        metrics_.val_step = step;
        metrics_.val_done = std::max(0, done);
        metrics_.val_total = std::max(0, total);

        // Pendant une validation, on peut afficher une progression même si "has" n'est pas encore final.
        if (has || in_progress) {
            metrics_.val_has = true;
        }
        metrics_.val_ok = ok;

        // Afficher les résultats partiels/finals si fournis.
        metrics_.val_recon = recon;
        metrics_.val_kl = kl;
        metrics_.val_align = align;

        // Compat UI: val_items représente le volume de validation (total si connu, sinon done).
        if (metrics_.val_total > 0) metrics_.val_items = metrics_.val_total;
        else if (metrics_.val_done > 0) metrics_.val_items = metrics_.val_done;

        has_update_ = true;
        metrics_version_.fetch_add(1, std::memory_order_relaxed);
    }
    
    // Définir l'intervalle de mise à jour (ms)
    void setUpdateInterval(int ms) {
        update_interval_ms_ = ms;
        htop_update_interval_ms_ = ms;
        viz_update_interval_ms_ = ms;
    }

    // Intervalles séparés (ms)
    void setHtopUpdateInterval(int ms) {
        htop_update_interval_ms_ = ms;
    }
    void setVizUpdateInterval(int ms) {
        viz_update_interval_ms_ = ms;
    }
    
    // Ajouter une image au visualiseur (file "generation")
    void addImage(const std::vector<uint8_t>& pixels, const std::string& prompt) {
        addImage(pixels, 0, 0, 0, prompt);
    }

    void addImage(const std::vector<uint8_t>& pixels, int w, int h, int channels, const std::string& prompt) {
        if (!viz_) return;

        std::lock_guard<std::mutex> lock(viz_mutex_);
        PendingImage img;
        img.pixels = pixels;
        img.w = w;
        img.h = h;
        img.channels = channels;
        img.prompt = prompt;
        pending_images_.push_back(std::move(img));
    }

    // Définir l'image du dataset utilisée (RGB/grayscale)
    void setDatasetImage(const std::vector<uint8_t>& pixels, int w, int h, int channels, const std::string& label) {
        if (!viz_) return;
        if (w <= 0 || h <= 0) return;
        if (channels != 1 && channels != 3 && channels != 4) return;

        std::lock_guard<std::mutex> lock(viz_mutex_);
        pending_dataset_image_ = PendingFrame{pixels, w, h, channels, label};
    }

    // Définir en une seule opération le sample dataset (image + texte) afin d'éviter
    // les désynchronisations visuelles (image d'un item et texte d'un autre).
    void setDatasetSample(
        const std::vector<uint8_t>& pixels,
        int w,
        int h,
        int channels,
        const std::string& label,
        const std::string& raw_text,
        const std::string& tokenized,
        const std::string& encoded
    ) {
        if (!viz_) return;
        if (w <= 0 || h <= 0) return;
        if (channels != 1 && channels != 3 && channels != 4) return;

        std::lock_guard<std::mutex> lock(viz_mutex_);
        PendingDatasetSample s;
        s.frame = PendingFrame{pixels, w, h, channels, label};
        s.text = PendingText{raw_text, tokenized, encoded};
        pending_dataset_sample_ = std::move(s);
    }

    // Définir le texte associé à l'item dataset (si modèle texte)
    void setDatasetText(const std::string& raw_text, const std::string& tokenized, const std::string& encoded) {
        if (!viz_) return;
        std::lock_guard<std::mutex> lock(viz_mutex_);
        pending_dataset_text_ = PendingText{raw_text, tokenized, encoded};
    }

    // Définir l'image de projection (souvent une heatmap)
    void setProjectionImage(const std::vector<uint8_t>& pixels, int w, int h, int channels, const std::string& label) {
        if (!viz_) return;
        if (w <= 0 || h <= 0) return;
        if (channels != 1 && channels != 3 && channels != 4) return;

        std::lock_guard<std::mutex> lock(viz_mutex_);
        pending_projection_image_ = PendingFrame{pixels, w, h, channels, label};
    }

    void setUnderstandingImage(const std::vector<uint8_t>& pixels, int w, int h, int channels, const std::string& label) {
        if (!viz_) return;
        if (w <= 0 || h <= 0) return;
        if (channels != 1 && channels != 3 && channels != 4) return;

        std::lock_guard<std::mutex> lock(viz_mutex_);
        pending_understanding_image_ = PendingFrame{pixels, w, h, channels, label};
    }

    void setLayerBlockImages(const std::vector<Visualizer::BlockFrame>& frames) {
        if (!viz_) return;
        std::lock_guard<std::mutex> lock(viz_mutex_);
        pending_layer_blocks_ = frames;
    }
    
    // Accesseurs
    std::shared_ptr<HtopDisplay> getHtop() { return htop_; }
    std::shared_ptr<Visualizer> getViz() { return viz_; }
    bool isRunning() const { return running_; }

    // UI -> training thread: arrêt propre demandé via bouton Viz.
    bool consumeStopTrainingRequested() {
        if (!viz_) return false;
        return viz_->consumeStopTrainingRequested();
    }
    
private:
    void startOutputCapture()
    {
        if (capture_running_.load()) return;
        if (!htop_) return;

        // Sauver stdout/stderr actuels.
        saved_stdout_fd_ = ::dup(STDOUT_FILENO);
        saved_stderr_fd_ = ::dup(STDERR_FILENO);
        if (saved_stdout_fd_ < 0 || saved_stderr_fd_ < 0) {
            // Best-effort: si dup échoue, ne pas capturer.
            if (saved_stdout_fd_ >= 0) { ::close(saved_stdout_fd_); saved_stdout_fd_ = -1; }
            if (saved_stderr_fd_ >= 0) { ::close(saved_stderr_fd_); saved_stderr_fd_ = -1; }
            return;
        }

        if (::pipe(pipe_fds_) != 0) {
            ::close(saved_stdout_fd_);
            ::close(saved_stderr_fd_);
            saved_stdout_fd_ = -1;
            saved_stderr_fd_ = -1;
            return;
        }

        // Rediriger stdout + stderr vers le pipe.
        ::fflush(stdout);
        ::fflush(stderr);
        ::dup2(pipe_fds_[1], STDOUT_FILENO);
        ::dup2(pipe_fds_[1], STDERR_FILENO);

        capture_running_ = true;
        log_thread_ = std::thread([this]() {
            std::string pending;
            pending.reserve(8192);
            std::vector<char> buf;
            buf.resize(4096);
            while (capture_running_.load()) {
                const ssize_t n = ::read(pipe_fds_[0], buf.data(), buf.size());
                if (n > 0) {
                    if (htop_) {
                        htop_->appendLogChunk(std::string(buf.data(), (size_t)n));
                    }
                    continue;
                }
                if (n == 0) {
                    break; // EOF
                }
                if (errno == EINTR) continue;
                break;
            }
        });
    }

    void stopOutputCapture()
    {
        if (!capture_running_.load()) {
            // Même si capture inactive, s'assurer de fermer des fd restants.
            if (pipe_fds_[0] >= 0) { ::close(pipe_fds_[0]); pipe_fds_[0] = -1; }
            if (pipe_fds_[1] >= 0) { ::close(pipe_fds_[1]); pipe_fds_[1] = -1; }
            if (saved_stdout_fd_ >= 0) { ::close(saved_stdout_fd_); saved_stdout_fd_ = -1; }
            if (saved_stderr_fd_ >= 0) { ::close(saved_stderr_fd_); saved_stderr_fd_ = -1; }
            return;
        }

        ::fflush(stdout);
        ::fflush(stderr);

        // Restaurer stdout/stderr (cela ferme implicitement les dup2 sur pipe_fds_[1]).
        if (saved_stdout_fd_ >= 0) {
            ::dup2(saved_stdout_fd_, STDOUT_FILENO);
            ::close(saved_stdout_fd_);
            saved_stdout_fd_ = -1;
        }
        if (saved_stderr_fd_ >= 0) {
            ::dup2(saved_stderr_fd_, STDERR_FILENO);
            ::close(saved_stderr_fd_);
            saved_stderr_fd_ = -1;
        }

        capture_running_ = false;

        // Fermer l'écriture du pipe pour débloquer read().
        if (pipe_fds_[1] >= 0) {
            ::close(pipe_fds_[1]);
            pipe_fds_[1] = -1;
        }

        if (log_thread_.joinable() && std::this_thread::get_id() != log_thread_.get_id()) {
            log_thread_.join();
        }

        if (pipe_fds_[0] >= 0) {
            ::close(pipe_fds_[0]);
            pipe_fds_[0] = -1;
        }
    }

    void htopLoop() {
        uint64_t last_ver = 0;
        while (running_) {
            Metrics local_metrics;
            bool has_data = false;
            uint64_t ver = 0;
            
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ver = metrics_version_.load(std::memory_order_relaxed);
                if (ver != last_ver) {
                    local_metrics = metrics_;
                    has_data = true;
                    last_ver = ver;
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
                    local_metrics.recon_loss_type,
                    local_metrics.grad_norm, local_metrics.grad_max,
                    local_metrics.opt_type, local_metrics.opt_step,
                    local_metrics.opt_beta1, local_metrics.opt_beta2,
                    local_metrics.opt_eps, local_metrics.opt_weight_decay
                );
                htop_->render();
            }
            
            std::this_thread::sleep_for(
                std::chrono::milliseconds(htop_update_interval_ms_.load()));
        }
    }
    
    void vizLoop() {
        uint64_t last_ver = 0;
        while (running_ && viz_ && viz_->isOpen()) {
            viz_->processEvents();
            
            // Mettre à jour métriques
            Metrics local_metrics;
            bool has_new_metrics = false;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                const uint64_t ver = metrics_version_.load(std::memory_order_relaxed);
                if (ver != last_ver) {
                    local_metrics = metrics_;
                    has_new_metrics = true;
                    last_ver = ver;
                }
            }

            if (has_new_metrics) {
                viz_->updateMetrics(
                    local_metrics.epoch, local_metrics.batch,
                    local_metrics.loss, local_metrics.lr,
                    local_metrics.mse, local_metrics.kl,
                    local_metrics.wass, local_metrics.ent,
                    local_metrics.mom, local_metrics.spat,
                    local_metrics.temp,
                    local_metrics.timestep,
                    local_metrics.total_epochs, local_metrics.total_batches, local_metrics.avg_loss,
                    local_metrics.batch_time_ms,
                    local_metrics.memory_mb,
                    local_metrics.bps,
                    local_metrics.params,
                    local_metrics.grad_norm,
                    local_metrics.grad_max,
                    local_metrics.opt_type,
                    local_metrics.opt_step,
                    local_metrics.opt_beta1,
                    local_metrics.opt_beta2,
                    local_metrics.opt_eps,
                    local_metrics.opt_weight_decay,
                    local_metrics.val_has,
                    local_metrics.val_ok,
                    local_metrics.val_step,
                    local_metrics.val_items,
                    local_metrics.val_recon,
                    local_metrics.val_kl,
                    local_metrics.val_align,
                    local_metrics.recon_loss_type,
                    local_metrics.val_in_progress,
                    local_metrics.val_done,
                    local_metrics.val_total,
                    local_metrics.kl_beta_effective
                );
                viz_->addLossPoint(local_metrics.loss);
            }
            
            // Ajouter images en attente
            {
                std::lock_guard<std::mutex> lock(viz_mutex_);
                for (const auto& img : pending_images_) {
                    viz_->addGeneratedImage(img.pixels, img.w, img.h, img.channels, img.prompt);
                }
                pending_images_.clear();

                // Appliquer d'abord l'update atomique (image+texte) si présent.
                if (pending_dataset_sample_.has_value()) {
                    const auto& s = pending_dataset_sample_.value();
                    viz_->setDatasetImage(s.frame.pixels, s.frame.w, s.frame.h, s.frame.channels, s.frame.label);
                    viz_->setDatasetText(s.text.raw, s.text.tokens, s.text.encoded);
                    pending_dataset_sample_.reset();
                    pending_dataset_image_.reset();
                    pending_dataset_text_.reset();
                }

                if (pending_dataset_image_.has_value()) {
                    const auto& f = pending_dataset_image_.value();
                    viz_->setDatasetImage(f.pixels, f.w, f.h, f.channels, f.label);
                    pending_dataset_image_.reset();
                }

                if (pending_dataset_text_.has_value()) {
                    const auto& t = pending_dataset_text_.value();
                    viz_->setDatasetText(t.raw, t.tokens, t.encoded);
                    pending_dataset_text_.reset();
                }
                if (pending_projection_image_.has_value()) {
                    const auto& f = pending_projection_image_.value();
                    viz_->setProjectionImage(f.pixels, f.w, f.h, f.channels, f.label);
                    pending_projection_image_.reset();
                }
                if (pending_understanding_image_.has_value()) {
                    const auto& f = pending_understanding_image_.value();
                    viz_->setUnderstandingImage(f.pixels, f.w, f.h, f.channels, f.label);
                    pending_understanding_image_.reset();
                }

                if (pending_layer_blocks_.has_value()) {
                    viz_->setLayerBlockImages(pending_layer_blocks_.value());
                    pending_layer_blocks_.reset();
                }
            }
            
            viz_->update();
            
            std::this_thread::sleep_for(
                std::chrono::milliseconds(viz_update_interval_ms_.load()));
        }
    }
    
    std::shared_ptr<HtopDisplay> htop_;
    std::shared_ptr<Visualizer> viz_;
    
    std::thread htop_thread_;
    std::thread viz_thread_;

    // UI fd (tty) + capture stdout/stderr
    int ui_fd_ = -1;
    int saved_stdout_fd_ = -1;
    int saved_stderr_fd_ = -1;
    int pipe_fds_[2] = {-1, -1};
    std::atomic<bool> capture_running_{false};
    std::thread log_thread_;
    
    std::mutex mutex_;
    std::mutex viz_mutex_;
    std::condition_variable cv_;

    // Synchronisation init SFML
    mutable std::mutex viz_init_mutex_;
    std::condition_variable viz_init_cv_;
    bool viz_init_done_ = false;
    bool viz_init_ok_ = false;
    std::string viz_init_err_;
    
    std::atomic<bool> running_;
    std::atomic<bool> has_update_;
    int update_interval_ms_;
    std::atomic<int> htop_update_interval_ms_;
    std::atomic<int> viz_update_interval_ms_;
    std::atomic<uint64_t> metrics_version_{0};
    
    Metrics metrics_;
    
    struct PendingImage {
        std::vector<uint8_t> pixels;
        int w = 0;
        int h = 0;
        int channels = 0;
        std::string prompt;
    };
    std::vector<PendingImage> pending_images_;

    struct PendingFrame {
        std::vector<uint8_t> pixels;
        int w = 0;
        int h = 0;
        int channels = 1;
        std::string label;
    };
    std::optional<PendingFrame> pending_dataset_image_;
    struct PendingText {
        std::string raw;
        std::string tokens;
        std::string encoded;
    };
    std::optional<PendingText> pending_dataset_text_;

    struct PendingDatasetSample {
        PendingFrame frame;
        PendingText text;
    };
    std::optional<PendingDatasetSample> pending_dataset_sample_;
    std::optional<PendingFrame> pending_projection_image_;
    std::optional<PendingFrame> pending_understanding_image_;

    std::optional<std::vector<Visualizer::BlockFrame>> pending_layer_blocks_;
};

#endif // __ASYNC_MONITOR_HPP__
