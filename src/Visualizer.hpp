#ifndef __TENSOR_VISUALIZER_HPP__
#define __TENSOR_VISUALIZER_HPP__

#include <SFML/Graphics.hpp>
#include <vector>
#include <string>
#include <deque>
#include <memory>
#include "include/json.hpp"

using json = nlohmann::json;

class Visualizer {
public:
    Visualizer(const json& config);
    ~Visualizer();

    // Initialisation de la fenêtre
    bool initialize();

    // Vérifier si la fenêtre est ouverte
    bool isOpen() const;

    // Traiter les événements (fermeture, etc.)
    void processEvents();

    // Mettre à jour l'affichage
    void update();

    // Ajouter une image générée (64x64 grayscale)
    void addGeneratedImage(const std::vector<uint8_t>& image, const std::string& prompt);

    // Mettre à jour les métriques d'entraînement
    void updateMetrics(int epoch, int batch, float loss, float lr, float mse = 0.0f,
                      float kl = 0.0f, float wass = 0.0f, float ent = 0.0f,
                      float mom = 0.0f, float spat = 0.0f, float temp = 0.0f);

    // Ajouter un point au graphique de loss
    void addLossPoint(float loss);

    // Effacer toutes les images
    void clearImages();

    // Activer/désactiver la visualisation
    void setEnabled(bool enabled);
    bool isEnabled() const;

    // Sauvegarder l'historique de loss dans un fichier CSV
    void saveLossHistory(const std::string& filepath) const;

private:
    std::string loss_log_file;  // Chemin du fichier de log
    // Configuration
    bool enabled;
    int window_width;
    int window_height;
    std::string window_title;
    int fps_limit;
    bool show_generated_images;
    bool show_training_progress;
    bool show_loss_graph;
    int graph_history_size;
    int image_grid_cols;
    int image_grid_rows;

    // SFML
    std::unique_ptr<sf::RenderWindow> window;
    sf::Font font;

    // Données de visualisation
    struct ImageData {
        std::vector<uint8_t> pixels;
        std::string prompt;
        sf::Texture texture;
        sf::Sprite sprite;
    };
    std::vector<ImageData> generated_images;

    // Métriques
    int current_epoch;
    int current_batch;
    float current_loss;
    float current_lr;
    float current_mse;
    std::deque<float> loss_history;
    
    // Historique complet pour CSV (epoch, batch, loss, lr, mse, etc.)
    struct LossRecord {
        int step;
        int epoch;
        int batch;
        float loss;
        float lr;
        float mse;
        float kl_divergence;
        float wasserstein;
        float entropy_diff;
        float moment_mismatch;
        float spatial_coherence;
        float temporal_consistency;
    };
    std::vector<LossRecord> full_loss_history;

    // Méthodes de rendu
    void renderBackground();
    void renderGeneratedImages();
    void renderTrainingProgress();
    void renderLossGraph();
    void renderMetrics();

    // Helpers
    void createImageTexture(ImageData& img_data, int img_size);
    sf::Color getLossColor(float loss);

    std::stringstream log_history;
};

#endif // __TENSOR_VISUALIZER_HPP__
