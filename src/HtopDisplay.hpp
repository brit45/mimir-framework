#pragma once

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <chrono>
#include <algorithm>
#include <sys/ioctl.h>
#include <unistd.h>

// ============================================================================
// Interface Htop-like pour monitoring de l'entraînement
// ============================================================================
class HtopDisplay
{
private:
    struct winsize terminal_size;
    int width, height;
    bool display_enabled;
    std::chrono::steady_clock::time_point start_time;

    // Statistiques
    struct Stats
    {
        int current_epoch;
        int total_epochs;
        int current_batch;
        int total_batches;
        float current_loss;
        float avg_loss;
        float kl_divergence;
        float wasserstein;
        float entropy_diff;
        float moment_mismatch;
        float spatial_coherence;
        float temporal_consistency;
        float mse_loss;
        float learning_rate;
        int batch_time_ms;
        size_t memory_used_mb;
        size_t memory_freed_mb;
        float batches_per_sec;
        int eta_seconds;
        size_t total_params;
        float timestep;
    } stats;

public:
    HtopDisplay() : display_enabled(true)
    {
        getTerminalSize();
        start_time = std::chrono::steady_clock::now();
        stats = {};
    }

    ~HtopDisplay()
    {
        showCursor();
    }

    void getTerminalSize()
    {
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &terminal_size);
        width = terminal_size.ws_col;
        height = terminal_size.ws_row;
    }

    void clearScreen()
    {
        std::cout << "\033[2J\033[H" << std::flush;
    }

    void resetCursor()
    {
        // Retourner au début sans effacer (plus rapide)
        std::cout << "\033[H" << std::flush;
    }

    void hideCursor()
    {
        std::cout << "\033[?25l" << std::flush;
    }

    void showCursor()
    {
        std::cout << "\033[?25h" << std::flush;
    }

    void moveCursor(int row, int col)
    {
        std::cout << "\033[" << row << ";" << col << "H";
    }

    void clearLine()
    {
        std::cout << "\033[2K";
    }

    std::string colorText(const std::string &text, int color)
    {
        return "\033[" + std::to_string(color) + "m" + text + "\033[0m";
    }

    std::string progressBar(float percent, int bar_width = 40, bool show_percent = true)
    {
        percent = std::clamp(percent, 0.0f, 100.0f);
        int filled = static_cast<int>(percent / 100.0f * bar_width);
        std::stringstream ss;
        ss << "[";
        for (int i = 0; i < bar_width; ++i)
        {
            if (i < filled)
            {
                if (percent > 80.0f)
                    ss << colorText("█", 32); // Vert
                else if (percent > 50.0f)
                    ss << colorText("█", 33); // Jaune
                else
                    ss << "█";
            }
            else if (i == filled)
                ss << "▓";
            else
                ss << "░";
        }
        ss << "]";
        if (show_percent)
        {
            ss << " " << std::fixed << std::setprecision(1) << percent << "%";
        }
        return ss.str();
    }

    std::string memoryBar(size_t used_mb, size_t total_mb, int bar_width = 30)
    {
        float percent = (float)used_mb / total_mb * 100.0f;
        int filled = static_cast<int>(std::min(percent, 100.0f) / 100.0f * bar_width);
        std::stringstream ss;
        ss << "[";
        for (int i = 0; i < bar_width; ++i)
        {
            if (i < filled)
            {
                if (percent > 100.0f)
                    ss << colorText("█", 91); // Rouge clignotant
                else if (percent > 80.0f)
                    ss << colorText("█", 31); // Rouge
                else if (percent > 60.0f)
                    ss << colorText("█", 33); // Jaune
                else
                    ss << colorText("█", 32); // Vert
            }
            else
            {
                ss << "░";
            }
        }
        ss << "] " << used_mb << "/" << total_mb << " MB";
        if (percent > 100.0f)
        {
            ss << colorText(" ⚠ OVERFILL!", 91);
        }
        return ss.str();
    }

    std::string metricBar(const std::string &label, float value, float scale = 1.0f, int bar_width = 25)
    {
        float percent = std::min(value * scale * 100.0f, 100.0f);
        int filled = static_cast<int>(percent / 100.0f * bar_width);
        std::stringstream ss;
        ss << std::left << std::setw(18) << label << " [";
        for (int i = 0; i < bar_width; ++i)
        {
            if (i < filled)
            {
                if (percent > 75.0f)
                    ss << colorText("█", 31); // Rouge = élevé
                else if (percent > 50.0f)
                    ss << colorText("█", 33); // Jaune
                else if (percent > 25.0f)
                    ss << colorText("█", 32); // Vert
                else
                    ss << "▓";
            }
            else
            {
                ss << "░";
            }
        }
        ss << "] " << std::fixed << std::setprecision(6) << value;
        return ss.str();
    }

    std::string formatTime(int seconds)
    {
        int hours = seconds / 3600;
        int mins = (seconds % 3600) / 60;
        int secs = seconds % 60;
        std::stringstream ss;
        if (hours > 0)
            ss << hours << "h ";
        ss << std::setw(2) << std::setfill('0') << mins << ":"
           << std::setw(2) << std::setfill('0') << secs;
        return ss.str();
    }

    void updateStats(int epoch, int total_epochs, int batch, int total_batches,
                     float loss, float avg_loss, float lr, int batch_time_ms,
                     size_t memory_mb, size_t memory_freed, float bps, size_t params,
                     float t, float kl, float wass, float ent, float mom,
                     float spat, float temp, float mse)
    {
        stats.current_epoch = epoch;
        stats.total_epochs = total_epochs;
        stats.current_batch = batch;
        stats.total_batches = total_batches;
        stats.current_loss = loss;
        stats.avg_loss = avg_loss;
        stats.learning_rate = lr;
        stats.batch_time_ms = batch_time_ms;
        stats.memory_used_mb = memory_mb;
        stats.memory_freed_mb = memory_freed;
        stats.batches_per_sec = bps;
        stats.total_params = params;
        stats.timestep = t;
        stats.kl_divergence = kl;
        stats.wasserstein = wass;
        stats.entropy_diff = ent;
        stats.moment_mismatch = mom;
        stats.spatial_coherence = spat;
        stats.temporal_consistency = temp;
        stats.mse_loss = mse;

        // Calculer ETA
        int batches_remaining = (total_epochs - epoch) * total_batches + (total_batches - batch);
        if (bps > 0)
        {
            stats.eta_seconds = static_cast<int>(batches_remaining / bps);
        }
    }

    void render()
    {
        if (!display_enabled)
            return;

        getTerminalSize();

        // Retourner au début sans effacer l'écran entier (optimisation)
        resetCursor();
        hideCursor();

        int row = 1;

        // Header
        std::string header = "=  Mímir Framework  =";
        int padding = (width - header.length()) / 2;
        moveCursor(row++, padding);
        clearLine();
        std::cout << colorText(header, 96);

        auto now = std::chrono::steady_clock::now();
        int elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

        moveCursor(row++, 2);
        clearLine();
        std::cout << colorText("Uptime:", 36) << " " << formatTime(elapsed)
                  << "  |  " << colorText("ETA:", 36) << " " << formatTime(stats.eta_seconds);

        moveCursor(row++, 1);
        clearLine();
        std::cout << std::string(width - 2, '_');

        // Epoch Progress
        moveCursor(row++, 2);
        clearLine();
        std::cout << colorText("EPOCH", 33) << " " << stats.current_epoch << "/" << stats.total_epochs;
        float epoch_percent = (float)stats.current_epoch / stats.total_epochs * 100.0f;
        moveCursor(row++, 2);
        clearLine();
        std::cout << progressBar(epoch_percent, width - 15);

        // Batch Progress
        moveCursor(row++, 2);
        clearLine();
        std::cout << colorText("BATCH", 33) << " " << stats.current_batch << "/" << stats.total_batches;
        float batch_percent = (float)stats.current_batch / stats.total_batches * 100.0f;
        moveCursor(row++, 2);
        clearLine();
        std::cout << progressBar(batch_percent, width - 15);

        moveCursor(row++, 1);
        clearLine();
        std::cout << std::string(width - 2, '_');

        // Loss Metrics
        moveCursor(row++, 2);
        clearLine();
        std::cout << colorText("● LOSS METRICS", 95);

        moveCursor(row++, 4);
        clearLine();
        std::cout << "Current: " << colorText(std::to_string(stats.current_loss).substr(0, 10), 93)
                  << "  Avg: " << colorText(std::to_string(stats.avg_loss).substr(0, 10), 92);

        // Particle Density Loss Components
        moveCursor(row++, 4);
        clearLine();
        std::cout << "KL Divergence    " << progressBar(std::min(stats.kl_divergence * 10, 100.0f), 25, false)
                  << " " << std::fixed << std::setprecision(6) << stats.kl_divergence;

        moveCursor(row++, 4);
        clearLine();
        std::cout << "Wasserstein      " << progressBar(std::min(stats.wasserstein * 100, 100.0f), 25, false)
                  << " " << stats.wasserstein;

        moveCursor(row++, 4);
        clearLine();
        std::cout << "Entropy Δ        " << progressBar(std::min(stats.entropy_diff * 10, 100.0f), 25, false)
                  << " " << stats.entropy_diff;

        moveCursor(row++, 4);
        clearLine();
        std::cout << "Moment Mismatch  " << progressBar(std::min(stats.moment_mismatch * 20, 100.0f), 25, false)
                  << " " << stats.moment_mismatch;

        moveCursor(row++, 4);
        clearLine();
        std::cout << "Spatial Coher.   " << progressBar(std::min(stats.spatial_coherence * 100, 100.0f), 25, false)
                  << " " << stats.spatial_coherence;

        moveCursor(row++, 4);
        clearLine();
        std::cout << "Temporal Cons.   " << progressBar(std::min(stats.temporal_consistency * 100, 100.0f), 25, false)
                  << " " << stats.temporal_consistency;

        moveCursor(row++, 4);
        clearLine();
        std::cout << "MSE (0.1×)       " << progressBar(std::min(stats.mse_loss * 100, 100.0f), 25, false)
                  << " " << stats.mse_loss;

        moveCursor(row++, 1);
        clearLine();
        std::cout << std::string(width - 2, '_');

        // System Resources
        moveCursor(row++, 2);
        clearLine();
        std::cout << colorText("● SYSTEM RESOURCES", 95);

        moveCursor(row++, 4);
        clearLine();
        std::cout << "Memory: " << memoryBar(stats.memory_used_mb, 10240, 30);

        moveCursor(row++, 4);
        clearLine();
        std::cout << "Freed this batch: " << colorText(std::to_string(stats.memory_freed_mb) + " MB", 92);

        moveCursor(row++, 4);
        clearLine();
        std::cout << "Parameters: " << colorText(std::to_string(stats.total_params), 96)
                  << " (~" << std::fixed << std::setprecision(2)
                  << (stats.total_params * 2.0f / 1024.0f / 1024.0f / 1024.0f) << " GB)";

        moveCursor(row++, 1);
        clearLine();
        std::cout << std::string(width - 2, '_');

        // Training Info
        moveCursor(row++, 2);
        clearLine();
        std::cout << colorText("● TRAINING INFO", 95);

        moveCursor(row++, 4);
        clearLine();
        std::cout << "Learning Rate: " << colorText(std::to_string(stats.learning_rate), 93);

        moveCursor(row++, 4);
        clearLine();
        std::cout << "Timestep (t): " << colorText(std::to_string(stats.timestep).substr(0, 6), 96);

        moveCursor(row++, 4);
        clearLine();
        std::cout << "Batch Time: " << colorText(std::to_string(stats.batch_time_ms) + " ms", 92);

        moveCursor(row++, 4);
        clearLine();
        std::cout << "Speed: " << colorText(std::to_string(stats.batches_per_sec).substr(0, 5) + " batches/s", 92);

        moveCursor(row++, 1);
        clearLine();
        std::cout << std::string(width - 2, '_');

        // Footer
        moveCursor(row++, 2);
        clearLine();
        std::cout << colorText("Press Ctrl+C to stop", 90);

        // Effacer les lignes restantes pour éviter les artefacts
        for (int i = row; i < height; ++i)
        {
            moveCursor(i, 1);
            clearLine();
        }

        std::cout << std::flush;
    }
};
