#include "Visualizer.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>

Visualizer::Visualizer(const json& config)
    : enabled(false)
    , window_width(1280)
    , window_height(720)
    , window_title("FLUX Model Visualization")
    , fps_limit(60)
    , show_generated_images(true)
    , show_training_progress(true)
    , show_loss_graph(true)
    , graph_history_size(200)
    , image_grid_cols(3)
    , image_grid_rows(1)
    , current_epoch(0)
    , current_batch(0)
    , current_loss(0.0f)
    , current_lr(0.0f)
    , current_mse(0.0f)
    , loss_log_file("checkpoints/loss_history.csv")
{
    if (config.contains("visualization")) {
        auto viz = config["visualization"];
        enabled = viz.value("enabled", false);
        window_width = viz.value("window_width", 1280);
        window_height = viz.value("window_height", 720);
        window_title = viz.value("window_title", "FLUX Model Visualization");
        fps_limit = viz.value("fps_limit", 60);
        show_generated_images = viz.value("show_generated_images", true);
        show_training_progress = viz.value("show_training_progress", true);
        show_loss_graph = viz.value("show_loss_graph", true);
        graph_history_size = viz.value("graph_history_size", 200);
        image_grid_cols = viz.value("image_grid_cols", 3);
        image_grid_rows = viz.value("image_grid_rows", 1);
    }
}

Visualizer::~Visualizer() {
    if (window && window->isOpen()) {
        window->close();
    }
}

bool Visualizer::initialize() {
    if (!enabled) {
        return false;
    }

    try {
        window = std::make_unique<sf::RenderWindow>(
            sf::VideoMode(window_width, window_height),
            window_title,
            sf::Style::Titlebar | sf::Style::Close
        );
        window->setFramerateLimit(fps_limit);

        // Charger une police par défaut (requiert un fichier .ttf)
        // Pour simplifier, on utilisera les formes géométriques SFML
        // Si vous avez une police TTF, décommentez:
        // if (!font.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")) {
        //     std::cerr << "Erreur: impossible de charger la police" << std::endl;
        //     return false;
        // }

        std::cout << "✓ Fenêtre de visualisation SFML initialisée (" 
                  << window_width << "x" << window_height << ")" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Erreur lors de l'initialisation SFML: " << e.what() << std::endl;
        return false;
    }
}

bool Visualizer::isOpen() const {
    return window && window->isOpen();
}

void Visualizer::processEvents() {
    if (!window) return;

    sf::Event event;
    while (window->pollEvent(event)) {
        if (event.type == sf::Event::Closed) {
            window->close();
        }
    }
}

void Visualizer::update() {
    if (!window || !window->isOpen()) return;

    window->clear(sf::Color(30, 30, 35)); // Fond sombre

    renderBackground();
    if (show_generated_images) renderGeneratedImages();
    if (show_training_progress) renderTrainingProgress();
    if (show_loss_graph) renderLossGraph();
    renderMetrics();

    window->display();
}

void Visualizer::addGeneratedImage(const std::vector<uint8_t>& image, const std::string& prompt) {
    if (!enabled) return;

    ImageData img_data;
    img_data.pixels = image;
    img_data.prompt = prompt;
    
    int img_size = static_cast<int>(std::sqrt(image.size()));
    createImageTexture(img_data, img_size);

    generated_images.push_back(std::move(img_data));

    // Limiter le nombre d'images affichées
    int max_images = image_grid_cols * image_grid_rows;
    if (generated_images.size() > static_cast<size_t>(max_images)) {
        generated_images.erase(generated_images.begin());
    }
}

void Visualizer::updateMetrics(int epoch, int batch, float loss, float lr, float mse,
                              float kl, float wass, float ent, float mom, float spat, float temp) {
    current_epoch = epoch;
    current_batch = batch;
    current_loss = loss;
    current_lr = lr;
    current_mse = mse;
    
    // Créer un record complet avec toutes les métriques
    LossRecord record;
    record.step = full_loss_history.size();
    record.epoch = epoch;
    record.batch = batch;
    record.loss = loss;
    record.lr = lr;
    record.mse = mse;
    record.kl_divergence = kl;
    record.wasserstein = wass;
    record.entropy_diff = ent;
    record.moment_mismatch = mom;
    record.spatial_coherence = spat;
    record.temporal_consistency = temp;
    full_loss_history.push_back(record);
    
    // Sauvegarder automatiquement l'historique après chaque ajout
    saveLossHistory(loss_log_file);
}

void Visualizer::addLossPoint(float loss) {
    loss_history.push_back(loss);
    if (loss_history.size() > static_cast<size_t>(graph_history_size)) {
        loss_history.pop_front();
    }
    
    // Ajouter à l'historique complet
    LossRecord record;
    record.step = static_cast<int>(full_loss_history.size());
    record.epoch = current_epoch;
    record.batch = current_batch;
    record.loss = loss;
    record.lr = current_lr;
    record.mse = current_mse;
    full_loss_history.push_back(record);
    
    // Sauvegarder automatiquement l'historique après chaque ajout
    saveLossHistory(loss_log_file);
}

void Visualizer::clearImages() {
    generated_images.clear();
}

void Visualizer::setEnabled(bool en) {
    enabled = en;
}

bool Visualizer::isEnabled() const {
    return enabled;
}

// Méthodes privées de rendu

void Visualizer::renderBackground() {
    // Grille de fond subtile
    sf::RectangleShape line(sf::Vector2f(1, window_height));
    line.setFillColor(sf::Color(40, 40, 45, 100));
    
    for (int x = 0; x < window_width; x += 50) {
        line.setPosition(x, 0);
        window->draw(line);
    }

    line.setSize(sf::Vector2f(window_width, 1));
    for (int y = 0; y < window_height; y += 50) {
        line.setPosition(0, y);
        window->draw(line);
    }
}

void Visualizer::renderGeneratedImages() {
    if (generated_images.empty()) return;

    int img_display_size = 200; // Taille d'affichage
    int margin = 20;
    int start_x = margin;
    int start_y = margin;

    for (size_t i = 0; i < generated_images.size(); ++i) {
        auto& img = generated_images[i];
        
        int col = i % image_grid_cols;
        int row = i / image_grid_cols;
        
        int x = start_x + col * (img_display_size + margin);
        int y = start_y + row * (img_display_size + margin);

        // Cadre
        sf::RectangleShape frame(sf::Vector2f(img_display_size + 4, img_display_size + 4));
        frame.setPosition(x - 2, y - 2);
        frame.setFillColor(sf::Color::Transparent);
        frame.setOutlineColor(sf::Color(100, 150, 200));
        frame.setOutlineThickness(2);
        window->draw(frame);

        // Image
        img.sprite.setPosition(x, y);
        window->draw(img.sprite);

        // Titre (simulé avec rectangle - nécessite police pour texte)
        sf::RectangleShape label(sf::Vector2f(img_display_size, 20));
        label.setPosition(x, y + img_display_size + 5);
        label.setFillColor(sf::Color(50, 50, 60, 200));
        window->draw(label);
    }
}

void Visualizer::renderTrainingProgress() {
    int bar_x = window_width - 300;
    int bar_y = 20;
    int bar_width = 280;
    int bar_height = 30;

    // Barre de progression epoch (simulée)
    sf::RectangleShape progress_bg(sf::Vector2f(bar_width, bar_height));
    progress_bg.setPosition(bar_x, bar_y);
    progress_bg.setFillColor(sf::Color(50, 50, 60));
    progress_bg.setOutlineColor(sf::Color(80, 80, 90));
    progress_bg.setOutlineThickness(2);
    window->draw(progress_bg);

    // Barre remplie (exemple: 50% d'avancement)
    float progress = std::min(1.0f, current_batch / 100.0f);
    sf::RectangleShape progress_bar(sf::Vector2f(bar_width * progress, bar_height));
    progress_bar.setPosition(bar_x, bar_y);
    progress_bar.setFillColor(sf::Color(100, 180, 100));
    window->draw(progress_bar);
}

void Visualizer::renderLossGraph() {
    if (loss_history.empty()) return;

    int graph_x = window_width - 400;
    int graph_y = window_height - 250;
    int graph_width = 380;
    int graph_height = 200;

    // Fond du graphique
    sf::RectangleShape graph_bg(sf::Vector2f(graph_width, graph_height));
    graph_bg.setPosition(graph_x, graph_y);
    graph_bg.setFillColor(sf::Color(20, 20, 25, 220));
    graph_bg.setOutlineColor(sf::Color(100, 100, 120));
    graph_bg.setOutlineThickness(2);
    window->draw(graph_bg);

    // Trouver min/max pour normalisation
    float min_loss = *std::min_element(loss_history.begin(), loss_history.end());
    float max_loss = *std::max_element(loss_history.begin(), loss_history.end());
    float range = max_loss - min_loss;
    if (range < 0.001f) range = 1.0f;

    // Dessiner la courbe
    sf::VertexArray line(sf::LineStrip, loss_history.size());
    for (size_t i = 0; i < loss_history.size(); ++i) {
        float x = graph_x + (i * graph_width) / static_cast<float>(graph_history_size);
        float normalized = (loss_history[i] - min_loss) / range;
        float y = graph_y + graph_height - (normalized * graph_height * 0.9f) - 10;
        
        line[i].position = sf::Vector2f(x, y);
        line[i].color = getLossColor(loss_history[i]);
    }
    window->draw(line);

    // Points
    for (size_t i = 0; i < loss_history.size(); ++i) {
        float x = graph_x + (i * graph_width) / static_cast<float>(graph_history_size);
        float normalized = (loss_history[i] - min_loss) / range;
        float y = graph_y + graph_height - (normalized * graph_height * 0.9f) - 10;
        
        sf::CircleShape point(3);
        point.setPosition(x - 3, y - 3);
        point.setFillColor(getLossColor(loss_history[i]));
        window->draw(point);
    }
}

void Visualizer::renderMetrics() {
    // Affichage des métriques textuelles (simulé avec rectangles colorés)
    int metrics_x = window_width - 300;
    int metrics_y = 80;
    int line_height = 25;

    auto drawMetricBox = [&](int y_offset, sf::Color color) {
        sf::RectangleShape box(sf::Vector2f(280, 20));
        box.setPosition(metrics_x, metrics_y + y_offset);
        box.setFillColor(color);
        window->draw(box);
    };

    // Epoch (bleu)
    drawMetricBox(0, sf::Color(50, 100, 180, 200));
    
    // Batch (vert)
    drawMetricBox(line_height, sf::Color(50, 150, 80, 200));
    
    // Loss (orange/rouge selon valeur)
    sf::Color loss_color = current_loss > 100.0f ? sf::Color(200, 80, 50, 200) : sf::Color(180, 140, 50, 200);
    drawMetricBox(line_height * 2, loss_color);
    
    // Learning rate (violet)
    drawMetricBox(line_height * 3, sf::Color(140, 80, 180, 200));
}

void Visualizer::createImageTexture(ImageData& img_data, int img_size) {
    // Créer une image SFML à partir des pixels grayscale
    sf::Image sfml_image;
    sfml_image.create(img_size, img_size);

    for (int y = 0; y < img_size; ++y) {
        for (int x = 0; x < img_size; ++x) {
            uint8_t gray = img_data.pixels[y * img_size + x];
            sfml_image.setPixel(x, y, sf::Color(gray, gray, gray));
        }
    }

    img_data.texture.loadFromImage(sfml_image);
    img_data.sprite.setTexture(img_data.texture);
    
    // Mettre à l'échelle pour affichage
    int display_size = 200;
    float scale = static_cast<float>(display_size) / img_size;
    img_data.sprite.setScale(scale, scale);
}

sf::Color Visualizer::getLossColor(float loss) {
    // Gradient vert -> jaune -> rouge basé sur la loss
    if (loss < 50.0f) {
        return sf::Color(100, 200, 100); // Vert
    } else if (loss < 150.0f) {
        return sf::Color(200, 200, 100); // Jaune
    } else {
        return sf::Color(200, 100, 100); // Rouge
    }
}

void Visualizer::saveLossHistory(const std::string& filepath) const {
    
    std::ofstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "Erreur: impossible d'ouvrir " << filepath << " pour écriture" << std::endl;
        return;
    }
    
    // En-tête CSV avec toutes les métriques LOSS METRICS
    file << "step,epoch,batch,loss,learning_rate,mse,kl_divergence,wasserstein,entropy_diff,moment_mismatch,spatial_coherence,temporal_consistency" << std::endl;
    
    // Écrire tout l'historique complet (toutes les epochs et tous les steps)
    for (const auto& record : full_loss_history) {
        file << record.step << "," 
             << record.epoch << "," 
             << record.batch << "," 
             << std::fixed << std::setprecision(6) << record.loss << "," 
             << std::scientific << record.lr << ","
             << std::fixed << std::setprecision(6) << record.mse << ","
             << record.kl_divergence << ","
             << record.wasserstein << ","
             << record.entropy_diff << ","
             << record.moment_mismatch << ","
             << record.spatial_coherence << ","
             << record.temporal_consistency << std::endl;
    }
    
    file.close();
}
