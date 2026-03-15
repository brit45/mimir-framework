#ifndef __TENSOR_VISUALIZER_HPP__
#define __TENSOR_VISUALIZER_HPP__

#ifdef ENABLE_SFML
#include <SFML/Graphics.hpp>
#endif
#include <vector>
#include <string>
#include <deque>
#include <array>
#include <memory>
#include <sstream>
#include <unordered_set>
#include <optional>
#include <atomic>
#include <cstdint>
#include "include/json.hpp"

using json = nlohmann::json;

#ifdef ENABLE_SFML
class Visualizer {
public:
    Visualizer(const json& config);
    ~Visualizer();

    // Ferme la fenêtre et libère les ressources SFML.
    // À appeler idéalement depuis le thread qui a créé la fenêtre.
    void shutdown();

    // Initialisation de la fenêtre
    bool initialize();

    // Vérifier si la fenêtre est ouverte
    bool isOpen() const;

    // Traiter les événements (fermeture, etc.)
    void processEvents();

    // Requête de resynchronisation (thread-safe): utilisée pour que le thread
    // d'entraînement puisse détecter que l'utilisateur a pressé 'R' dans la Viz.
    void requestResync();
    bool consumeResyncRequested();

    // Requête d'arrêt de l'entraînement depuis l'UI (bouton dans le panneau Metrics).
    void requestStopTraining();
    bool consumeStopTrainingRequested();

    // Mettre à jour l'affichage
    void update();

    // Ajouter une image générée.
    // Compat legacy: l'ancienne API supposait une image carrée en 1 canal (w/h inférés via sqrt).
    void addGeneratedImage(const std::vector<uint8_t>& image, const std::string& prompt);
    // Nouvelle API: dimensions explicites (supporte RGB).
    void addGeneratedImage(const std::vector<uint8_t>& image, int w, int h, int channels, const std::string& prompt);

    // Définir l'image du dataset actuellement utilisée (RGB ou grayscale).
    void setDatasetImage(const std::vector<uint8_t>& pixels, int w, int h, int channels, const std::string& label);

    // Si le modèle utilise du texte provenant du dataset: afficher le texte brut,
    // une forme tokenisée (ids) et un résumé de l'encodage (stats/shape).
    void setDatasetText(const std::string& raw_text, const std::string& tokenized, const std::string& encoded);

    // Définir la sortie de projection (visualisée en image/grille, généralement grayscale).
    void setProjectionImage(const std::vector<uint8_t>& pixels, int w, int h, int channels, const std::string& label);

    // Définir une visualisation de “compréhension” (feature/projection) en heatmap/grille.
    void setUnderstandingImage(const std::vector<uint8_t>& pixels, int w, int h, int channels, const std::string& label);

    // Définir les images intermédiaires (traitement par blocs/layers)
    struct BlockFrame {
        std::vector<uint8_t> pixels;
        int w = 0;
        int h = 0;
        int channels = 1;
        std::string label;
    };
    void setLayerBlockImages(const std::vector<BlockFrame>& frames);

    // Mettre à jour les métriques d'entraînement
    void updateMetrics(int epoch, int batch, float loss, float lr, float mse = 0.0f,
                      float kl = 0.0f, float wass = 0.0f, float ent = 0.0f,
                      float mom = 0.0f, float spat = 0.0f, float temp = 0.0f,
                      float timestep = 0.0f,
                      int total_epochs = 0, int total_batches = 0, float avg_loss = 0.0f,
                      int batch_time_ms = 0, size_t memory_mb = 0, float bps = 0.0f, size_t params = 0,
                      float grad_norm = 0.0f, float grad_max = 0.0f,
                      int opt_type = 0, int opt_step = 0,
                      float opt_beta1 = 0.0f, float opt_beta2 = 0.0f,
                      float opt_eps = 0.0f, float opt_weight_decay = 0.0f,
                      bool val_has = false, bool val_ok = false, int val_step = 0, int val_items = 0,
                      float val_recon = 0.0f, float val_kl = 0.0f, float val_align = 0.0f,
                      const std::string& recon_loss_type = std::string(),
                      bool val_in_progress = false,
                      int val_done = 0,
                      int val_total = 0,
                      float kl_beta_effective = 0.0f);

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

    // Flag thread-safe: mis à true quand l'utilisateur demande une resync (touche R).
    std::atomic<bool> resync_requested_{false};

    // Flag thread-safe: mis à true quand l'utilisateur demande d'arrêter l'entraînement.
    std::atomic<bool> stop_training_requested_{false};

    // Label parsing / architecture hints
    bool hide_activation_blocks = false;
    std::string architecture_path;
    bool architecture_loaded = false;
    std::unordered_set<std::string> arch_layer_names;
    std::unordered_set<std::string> arch_tensor_outputs;
    std::unordered_set<std::string> arch_tensor_sinks;
    uint64_t arch_last_check_ms = 0;
    int graph_history_size;
    int image_grid_cols;
    int image_grid_rows;

    // SFML
    std::unique_ptr<sf::RenderWindow> window;
    sf::Font font;
    bool font_loaded = false;

    // Logo framework/programme (chargé depuis ./logo.png)
    sf::Texture logo_texture_;
    sf::Sprite logo_sprite_;
    bool logo_loaded_ = false;
    sf::Clock logo_clock_;
    float logo_splash_seconds_ = 2.5f;

    // Données de visualisation
    struct ImageData {
        std::vector<uint8_t> pixels;
        std::string prompt;
        int w = 0;
        int h = 0;
        int channels = 1;
        int display_size = 0;
        sf::Texture texture;
        sf::Sprite sprite;
    };
    std::vector<ImageData> generated_images;

    // Images contextuelles (dataset + projection)
    ImageData dataset_image;
    bool has_dataset_image = false;
    std::string dataset_label;

    ImageData projection_image;
    bool has_projection_image = false;
    std::string projection_label;

    // Thumbnails (120px) réutilisés dans le panneau "Blocks / Layers"
    ImageData projection_thumb_;
    bool has_projection_thumb_ = false;

    ImageData understanding_image;
    bool has_understanding_image = false;
    std::string understanding_label;

    // Dernière "sortie" (on réutilise la dernière image générée)
    ImageData output_thumb_;
    bool has_output_thumb_ = false;
    std::string output_thumb_label_;

    // Texte dataset (optionnel)
    bool has_dataset_text = false;
    std::string dataset_text_raw;
    std::string dataset_text_tokens;
    std::string dataset_text_encoded;

    std::vector<ImageData> layer_block_images;
    std::vector<std::string> layer_block_labels;
    bool has_layer_blocks = false;

    // Métriques
    int current_epoch;
    int current_total_epochs;
    int current_batch;
    int current_total_batches;
    float current_loss;
    float current_avg_loss;
    float current_lr;
    float current_mse;
    std::string current_recon_loss_type;
    float current_kl;
    float current_wass;
    float current_ent;
    float current_mom;
    float current_spat;
    float current_temp;
    float current_timestep;
    int current_batch_time_ms;
    size_t current_memory_mb;
    float current_bps;
    size_t current_params;
    float current_grad_norm;
    float current_grad_max;
    int current_opt_type;
    int current_opt_step;
    float current_opt_beta1;
    float current_opt_beta2;
    float current_opt_eps;
    float current_opt_weight_decay;

    // Warmup KL: beta effectif (après warmup)
    float current_kl_beta_effective = 0.0f;

    // Validation (dernières valeurs connues)
    bool current_val_has;
    bool current_val_ok;
    bool current_val_in_progress;
    int current_val_step;
    int current_val_items;
    int current_val_done;
    int current_val_total;
    float current_val_recon;
    float current_val_kl;
    float current_val_align;
    std::deque<float> loss_history;

    // Stats pour l'échelle du graph (évite un scan O(n) à chaque frame)
    bool has_loss_stats_ = false;
    float loss_min_ = 0.0f;
    float loss_max_ = 0.0f;
    
    // Historique complet pour CSV (epoch, batch, loss, lr, mse, etc.)
    struct LossRecord {
        int step;
        int epoch;
        int total_epochs;
        int batch;
        int total_batches;
        float loss;
        float avg_loss;
        float lr;
        int batch_time_ms;
        float bps;
        size_t memory_mb;
        size_t params;
        float mse;
        float kl_divergence;
        float wasserstein;
        float entropy_diff;
        float moment_mismatch;
        float spatial_coherence;
        float temporal_consistency;
        float timestep;
        float grad_norm;
        float grad_max;
        int opt_type;
        int opt_step;
        float opt_beta1;
        float opt_beta2;
        float opt_eps;
        float opt_weight_decay;
    };
    std::vector<LossRecord> full_loss_history;

    // Méthodes de rendu
    void renderBackground();
    void renderContextImages();
    void renderLayerBlocks();
    void renderGeneratedImages();
    void renderTrainingProgress();
    void renderLossGraph();
    void renderMetrics();

    // Helpers
    void createImageTexture(ImageData& img_data, int w, int h, int channels, int display_size);
    sf::Color getLossColor(float loss);

    // UI
    bool show_help_overlay_ = false;
    bool show_prompt_text_ = true;
    bool zoom_active_ = false;
    enum class FocusTarget { Dataset, Projection, Understanding, LayerBlock, Generated };
    FocusTarget focus_target_ = FocusTarget::Dataset;
    int focus_block_index_ = 0;
    int focus_generated_index_ = 0;

    // Layout: panneaux déplaçables (drag & drop)
    enum class PanelId { Context = 0, Blocks = 1, Generated = 2, Training = 3, Metrics = 4, Graph = 5, Count = 6 };
    struct Panel {
        sf::Vector2f pos{0.f, 0.f};
        sf::Vector2f size{0.f, 0.f};
        std::string title;
        bool visible = true;
        bool allow_drag = true;
    };
    std::array<Panel, static_cast<size_t>(PanelId::Count)> panels_{};
    bool panels_initialized_ = false;
    bool dragging_panel_ = false;
    PanelId dragged_panel_ = PanelId::Context;
    sf::Vector2f drag_grab_offset_{0.f, 0.f};

    bool resizing_panel_ = false;
    PanelId resized_panel_ = PanelId::Context;
    sf::Vector2f resize_start_mouse_{0.f, 0.f};
    sf::Vector2f resize_start_size_{0.f, 0.f};

    static constexpr float kPanelResizeHandle = 14.f;
    static constexpr float kPanelMinW = 220.f;
    static constexpr float kPanelMinH = 140.f;

    static constexpr float kPanelTitleH = 22.f;
    static constexpr float kPanelPad = 10.f;

    void initDefaultPanelsIfNeeded();
    void clampPanelsToWindow();
    void syncUIView();
    bool isPanelVisible(PanelId id) const;
    sf::FloatRect panelRect(PanelId id) const;
    sf::FloatRect panelTitleRect(PanelId id) const;
    sf::FloatRect panelContentRect(PanelId id) const;
    sf::FloatRect panelResizeHandleRect(PanelId id) const;
    sf::FloatRect panelCloseButtonRect(PanelId id) const;
    std::optional<PanelId> hitTestPanelTitle(const sf::Vector2f& mouse) const;
    std::optional<PanelId> hitTestPanelResizeHandle(const sf::Vector2f& mouse) const;
    std::optional<PanelId> hitTestPanelCloseButton(const sf::Vector2f& mouse) const;
    void drawPanelChrome(PanelId id);
    sf::Color panelAccent(PanelId id) const;

    // Cursor
    bool cursors_loaded_ = false;
    bool cursor_ok_arrow_ = false;
    bool cursor_ok_hand_ = false;
    bool cursor_ok_cross_ = false;
    bool cursor_ok_resize_ = false;
    sf::Cursor cursor_arrow_;
    sf::Cursor cursor_hand_;
    sf::Cursor cursor_cross_;
    sf::Cursor cursor_resize_;
    enum class CursorKind { Arrow, Hand, Cross, Resize };
    CursorKind cursor_kind_ = CursorKind::Arrow;
    void initCursorsIfNeeded();
    void setCursor(CursorKind kind);

    void rebuildAllTextures();
    void renderHelpOverlay();
    void renderZoomOverlay();

    // Focus outline
    std::optional<sf::FloatRect> last_dataset_rect_;
    std::optional<sf::FloatRect> last_projection_rect_;
    std::optional<sf::FloatRect> last_understanding_rect_;
    std::vector<sf::FloatRect> last_block_rects_;
    std::vector<sf::FloatRect> last_generated_rects_;
    std::vector<int> last_generated_indices_;
    void renderFocusOutline();

    // Stop button (Metrics panel)
    std::optional<sf::FloatRect> last_stop_button_rect_;

    // Scroll (Blocks/Layers panel)
    float blocks_scroll_y_ = 0.0f;
    float blocks_scroll_max_ = 0.0f;
    static constexpr float kBlocksScrollSpeed = 48.0f;

    // Scrollbar (slicer) cliquable pour Blocks/Layers
    bool dragging_blocks_scrollbar_ = false;
    float blocks_scroll_drag_grab_y_ = 0.0f;
    sf::FloatRect last_blocks_scroll_track_rect_{0.f, 0.f, 0.f, 0.f};
    sf::FloatRect last_blocks_scroll_thumb_rect_{0.f, 0.f, 0.f, 0.f};

    // Refresh auto textures (évite le reload UI)
    uint64_t last_auto_texture_refresh_ms_ = 0;
    static constexpr uint64_t kAutoTextureRefreshPeriodMs = 2000;

    // Architecture awareness (optional)
    void maybeLoadArchitecture();

    // UI layout settings (persisted to JSON)
    json serializeUILayout() const;
    bool applyUILayout(const json& layout);
    bool loadUISettings(json& out) const;
    bool saveUISettings(const json& s) const;
    void saveUILayoutToLast();
    void saveUILayoutToSlot(int slot);
    void loadUILayoutFromSlot(int slot);

    bool save_chord_armed_ = false;
    uint64_t save_chord_armed_ms_ = 0;
    bool save_chord_consumed_ = false;

    std::stringstream log_history;
};

#else

// Version headless (sans SFML) : API compatible, pas de rendu.
class Visualizer {
public:
    explicit Visualizer(const json&) {}
    ~Visualizer() = default;

    bool initialize() { return false; }
    bool isOpen() const { return false; }
    void processEvents() {}
    void update() {}

    void shutdown() {}

    void requestResync() {}
    bool consumeResyncRequested() { return false; }

    void requestStopTraining() {}
    bool consumeStopTrainingRequested() { return false; }

    void addGeneratedImage(const std::vector<uint8_t>&, const std::string&) {}

    void setDatasetImage(const std::vector<uint8_t>&, int, int, int, const std::string&) {}
    void setDatasetText(const std::string&, const std::string&, const std::string&) {}
    void setProjectionImage(const std::vector<uint8_t>&, int, int, int, const std::string&) {}
    void setUnderstandingImage(const std::vector<uint8_t>&, int, int, int, const std::string&) {}

    struct BlockFrame {
        std::vector<uint8_t> pixels;
        int w = 0;
        int h = 0;
        int channels = 1;
        std::string label;
    };
    void setLayerBlockImages(const std::vector<BlockFrame>&) {}

    void updateMetrics(int, int, float, float, float = 0.0f,
                      float = 0.0f, float = 0.0f, float = 0.0f,
                      float = 0.0f, float = 0.0f, float = 0.0f,
                      float = 0.0f,
                      int = 0, int = 0, float = 0.0f,
                      int = 0, size_t = 0, float = 0.0f, size_t = 0,
                      float = 0.0f, float = 0.0f,
                      int = 0, int = 0,
                      float = 0.0f, float = 0.0f,
                      float = 0.0f, float = 0.0f,
                      bool = false, bool = false, int = 0, int = 0,
                      float = 0.0f, float = 0.0f, float = 0.0f,
                      const std::string& = std::string(),
                      bool = false,
                      int = 0,
                      int = 0,
                      float = 0.0f) {}

    void addLossPoint(float) {}
    void clearImages() {}
    void setEnabled(bool) {}
    bool isEnabled() const { return false; }
    void saveLossHistory(const std::string&) const {}
};

#endif

#endif // __TENSOR_VISUALIZER_HPP__
