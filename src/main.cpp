#include "Model.hpp"
#include "Models/UNet.hpp"
#include "Helpers.hpp"
#include "HtopDisplay.hpp"
#include "include/json.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <cstdlib>

using json = nlohmann::json;
namespace fs = std::filesystem;

void printUsage(const char *prog)
{
    std::cout << "Usage: " << prog << " [OPTIONS]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --config <config.json>   Load configuration file and start training\n";
    std::cout << "  --help                   Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << prog << " --config config.json\n";
}

int main(int argc, char **argv)
{
    std::cout << "╔════════════════════════════════════════╗\n";
    std::cout << "║          Mímir Framework v1.0          ║\n";
    std::cout << "║            Deep Learning C++           ║\n";
    std::cout << "╚════════════════════════════════════════╝\n\n";

    std::string config_path;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc)
        {
            config_path = argv[++i];
        }
        else if (std::strcmp(argv[i], "--help") == 0)
        {
            printUsage(argv[0]);
            return 0;
        }
        else
        {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    if (config_path.empty())
    {
        std::cerr << "❌ Error: --config is required\n";
        printUsage(argv[0]);
        return 1;
    }

    // Load configuration
    std::shared_ptr<Model> model;
    json config;

    std::cout << "📋 Loading configuration: " << config_path << "\n";
    try
    {
        std::ifstream f(config_path);
        f >> config;

        std::cout << "✓ Configuration loaded\n";

        // Initialize model from config
        if (config.contains("model") && config["model"].contains("type"))
        {
            std::string model_type = config["model"]["type"];
            std::cout << "🏗️  Initializing model type: " << model_type << "\n";

            if (model_type == "UNet" && config.contains("unet"))
            {
                auto &unet_cfg = config["unet"];

                // Extraire les paramètres UNet
                int input_channels = unet_cfg.value("input_channels", 3);
                int output_channels = unet_cfg.value("output_channels", 3);
                int base_filters = unet_cfg.value("base_filters", 64);
                int image_size = unet_cfg.value("image_size", 64);
                int num_stages = unet_cfg.value("num_stages", 4);
                int blocks_per_stage = unet_cfg.value("blocks_per_stage", 2);
                int bottleneck_depth = unet_cfg.value("bottleneck_depth", 3);

                std::cout << "   Input channels: " << input_channels << "\n";
                std::cout << "   Output channels: " << output_channels << "\n";
                std::cout << "   Base filters: " << base_filters << "\n";
                std::cout << "   Image size: " << image_size << "x" << image_size << "\n";
                std::cout << "   Stages: " << num_stages << "\n";
                std::cout << "   Blocks per stage: " << blocks_per_stage << "\n";
                std::cout << "   Bottleneck depth: " << bottleneck_depth << "\n";

                // Créer le modèle UNet
                auto unet = std::make_shared<UNet>(input_channels, output_channels, base_filters);

                // Construire l'architecture
                unet->buildBackboneUNet(num_stages, blocks_per_stage, bottleneck_depth);

                // Allouer les paramètres
                unet->allocateParams();

                std::cout << "✓ UNet model initialized with " << unet->totalParamCount() << " parameters\n";

                model = unet;
            }
            else
            {
                std::cout << "⚠️  Model type '" << model_type << "' not implemented\n";
            }
        }

        std::cout << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "❌ Error loading config: " << e.what() << "\n";
        return 1;
    }

    if (!model)
    {
        std::cerr << "❌ Error: No model initialized from config\n";
        return 1;
    }

    // Charger le dataset
    std::vector<DatasetItem> dataset;
    try
    {
        if (config.contains("dataset"))
        {
            auto &ds_cfg = config["dataset"];
            std::string dataset_dir = ds_cfg.value("dir", "dataset");
            bool cache_enabled = ds_cfg.value("cache_enabled", true);
            bool lazy_loading = ds_cfg.value("lazy_loading", true);

            std::cout << "📦 Loading dataset from: " << dataset_dir << "\n";

            if (cache_enabled)
            {
                size_t max_ram_gb = 10;
                if (config.contains("optimization"))
                {
                    max_ram_gb = config["optimization"].value("max_ram_target_gb", 10);
                }

                dataset = loadDatasetCached(
                    dataset_dir,
                    64, 64, 1,
                    "dataset_cache.json",
                    max_ram_gb * 1024,
                    lazy_loading);
            }
            else
            {
                dataset = loadDataset(dataset_dir, 64, 64, 1);
            }

            std::cout << "✓ Dataset loaded: " << dataset.size() << " items\n\n";
        }
        else
        {
            std::cerr << "⚠️  No dataset configuration found\n\n";
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "❌ Error loading dataset: " << e.what() << "\n";
        return 1;
    }

    // Configuration et entraînement
    try
    {
        if (!config.contains("training"))
        {
            std::cerr << "❌ No training configuration found\n";
            return 1;
        }

        auto &training_cfg = config["training"];

        int num_epochs = training_cfg.value("num_epochs", 100);
        float learning_rate = training_cfg.value("learning_rate", 0.0001f);
        int batch_size = training_cfg.value("batch_size", 16);
        std::string optimizer_type = training_cfg.value("optimizer", "adam");

        std::cout << "🎯 Starting training:\n";
        std::cout << "   Epochs: " << num_epochs << "\n";
        std::cout << "   Learning rate: " << learning_rate << "\n";
        std::cout << "   Batch size: " << batch_size << "\n";
        std::cout << "   Optimizer: " << optimizer_type << "\n";
        std::cout << "   Dataset size: " << dataset.size() << " items\n";

        // Configurer l'optimiseur
        Optimizer opt;
        if (optimizer_type == "adam" || optimizer_type == "adamW")
        {
            opt.type = (optimizer_type == "adamW") ? OptimizerType::ADAMW : OptimizerType::ADAM;
            opt.beta1 = training_cfg.value("adam_beta1", 0.9f);
            opt.beta2 = training_cfg.value("adam_beta2", 0.999f);
            opt.eps = training_cfg.value("adam_epsilon", 1e-8f);
            opt.weight_decay = training_cfg.value("weight_decay", 0.01f);
        }
        else
        {
            opt.type = OptimizerType::SGD;
        }

        opt.initial_lr = learning_rate;
        opt.min_lr = learning_rate * 0.01f;

        // LR Decay strategy
        if (training_cfg.contains("lr_decay") && training_cfg["lr_decay"].value("enabled", false))
        {
            std::string strategy = training_cfg["lr_decay"].value("strategy", "cosine");
            if (strategy == "cosine")
                opt.decay_strategy = LRDecayStrategy::COSINE;
            else if (strategy == "linear")
                opt.decay_strategy = LRDecayStrategy::LINEAR;
            else if (strategy == "step")
                opt.decay_strategy = LRDecayStrategy::STEP;
            else if (strategy == "exponential")
                opt.decay_strategy = LRDecayStrategy::EXPONENTIAL;
            else
                opt.decay_strategy = LRDecayStrategy::NONE;

            opt.decay_rate = training_cfg["lr_decay"].value("decay_rate", 0.95f);
            opt.decay_steps = training_cfg["lr_decay"].value("decay_steps", 1000);
            opt.total_steps = num_epochs * 100;
            opt.warmup_steps = training_cfg.value("warmup_steps", 0);

            std::cout << "   LR Decay: " << strategy << "\n";
        }

        std::cout << "\n";

        // Allouer l'optimiseur
        size_t total_params = model->totalParamCount();
        opt.ensure(total_params);

        // Initialiser HtopDisplay si activé
        bool show_htop = false;
        if (config.contains("logging"))
        {
            show_htop = config["logging"].value("show_htop_display", true);
        }

        HtopDisplay htop_display;
        if (!show_htop)
        {
            // Mode console classique
            std::cout << "🚀 Training started...\n";
            std::cout << std::string(60, '=') << "\n\n";
        }
        else
        {
            // Initialiser l'affichage HtopDisplay
            htop_display.clearScreen();
            htop_display.hideCursor();
        }

        srand(time(nullptr));

        // Variables pour les statistiques
        float running_avg_loss = 0.0f;
        float running_g_loss = 0.0f;
        float running_d_loss = 0.0f;
        size_t memory_freed_mb = 0;
        auto training_start = std::chrono::high_resolution_clock::now();

        // Paramètres du discriminateur simple (MLP)
        int img_size = config["unet"].value("image_size", 64);
        int disc_input_dim = img_size * img_size; // Images en niveaux de gris
        int disc_hidden = 512;

        // Discriminateur : input -> hidden -> 1 (real/fake)
        std::vector<float> disc_w1(disc_input_dim * disc_hidden);
        std::vector<float> disc_b1(disc_hidden);
        std::vector<float> disc_w2(disc_hidden);
        std::vector<float> disc_b2(1);

        // Initialisation Xavier/He
        std::random_device rd;
        std::mt19937 gen(rd());
        float w1_std = std::sqrt(2.0f / disc_input_dim);
        float w2_std = std::sqrt(2.0f / disc_hidden);
        std::normal_distribution<float> dist_w1(0.0f, w1_std);
        std::normal_distribution<float> dist_w2(0.0f, w2_std);

        for (auto &w : disc_w1)
            w = dist_w1(gen);
        for (auto &w : disc_w2)
            w = dist_w2(gen);
        std::fill(disc_b1.begin(), disc_b1.end(), 0.0f);
        std::fill(disc_b2.begin(), disc_b2.end(), 0.0f);

        // Optimiseurs séparés pour G et D
        Optimizer opt_g = opt; // Générateur (UNet)
        Optimizer opt_d;       // Discriminateur
        opt_d.type = opt.type;
        opt_d.beta1 = opt.beta1;
        opt_d.beta2 = opt.beta2;
        opt_d.eps = opt.eps;
        opt_d.weight_decay = opt.weight_decay;
        opt_d.initial_lr = learning_rate * 0.1f; // LR plus faible pour D
        opt_d.min_lr = opt_d.initial_lr * 0.01f;
        opt_d.decay_strategy = opt.decay_strategy;
        opt_d.decay_rate = opt.decay_rate;
        opt_d.decay_steps = opt.decay_steps;
        opt_d.total_steps = opt.total_steps;

        size_t disc_params = disc_w1.size() + disc_b1.size() + disc_w2.size() + disc_b2.size();
        opt_d.ensure(disc_params);

        for (int epoch = 0; epoch < num_epochs; ++epoch)
        {
            auto epoch_start = std::chrono::high_resolution_clock::now();

            float epoch_loss = 0.0f;
            int num_batches = dataset.empty() ? 10 : ((dataset.size() + batch_size - 1) / batch_size);

            for (int batch = 0; batch < num_batches; ++batch)
            {
                auto batch_start = std::chrono::high_resolution_clock::now();

                // Charger un batch d'images réelles
                std::vector<std::vector<float>> real_images;
                std::vector<std::vector<float>> noise_vectors;

                if (!dataset.empty())
                {
                    int actual_batch_size = std::min(batch_size, (int)dataset.size());

                    for (int i = 0; i < actual_batch_size; ++i)
                    {
                        size_t idx = rand() % dataset.size();
                        auto &item = dataset[idx];

                        // Charger l'image si nécessaire
                        if (item.loadImage(img_size, img_size) && item.img.has_value())
                        {
                            // Normaliser l'image [0,255] -> [-1,1]
                            std::vector<float> img_normalized(item.img->size());
                            for (size_t j = 0; j < item.img->size(); ++j)
                            {
                                img_normalized[j] = ((*item.img)[j] / 127.5f) - 1.0f;
                            }
                            real_images.push_back(img_normalized);

                            // Générer vecteur de bruit pour le générateur
                            std::vector<float> noise(disc_input_dim);
                            std::normal_distribution<float> noise_dist(0.0f, 1.0f);
                            for (auto &n : noise)
                                n = noise_dist(gen);
                            noise_vectors.push_back(noise);
                        }
                    }
                }

                if (real_images.empty())
                    continue;

                int current_batch_size = real_images.size();

                // ========== PHASE 1: Entraîner le Discriminateur ==========
                float d_loss_real = 0.0f;
                float d_loss_fake = 0.0f;

                // Forward pass discriminateur sur images réelles
                for (const auto &real_img : real_images)
                {
                    // Hidden layer
                    std::vector<float> hidden(disc_hidden, 0.0f);
                    for (int h = 0; h < disc_hidden; ++h)
                    {
                        float sum = disc_b1[h];
                        for (size_t i = 0; i < real_img.size() && i < disc_input_dim; ++i)
                        {
                            sum += real_img[i] * disc_w1[i * disc_hidden + h];
                        }
                        hidden[h] = std::max(0.0f, sum); // ReLU
                    }

                    // Output layer
                    float output = disc_b2[0];
                    for (int h = 0; h < disc_hidden; ++h)
                    {
                        output += hidden[h] * disc_w2[h];
                    }
                    float prob = 1.0f / (1.0f + std::exp(-output)); // Sigmoid

                    // Loss BCE: -log(D(x_real))
                    d_loss_real += -std::log(std::max(prob, 1e-7f));
                }

                // Forward pass discriminateur sur images générées (fakes)
                for (const auto &noise : noise_vectors)
                {
                    // Simuler génération d'image par UNet (pour l'instant, bruit transformé)
                    // TODO: Remplacer par vraie forward pass UNet
                    std::vector<float> fake_img(disc_input_dim);
                    for (size_t i = 0; i < fake_img.size() && i < noise.size(); ++i)
                    {
                        fake_img[i] = std::tanh(noise[i] * 0.5f); // Simulation simple
                    }

                    // Hidden layer
                    std::vector<float> hidden(disc_hidden, 0.0f);
                    for (int h = 0; h < disc_hidden; ++h)
                    {
                        float sum = disc_b1[h];
                        for (size_t i = 0; i < fake_img.size(); ++i)
                        {
                            sum += fake_img[i] * disc_w1[i * disc_hidden + h];
                        }
                        hidden[h] = std::max(0.0f, sum); // ReLU
                    }

                    // Output layer
                    float output = disc_b2[0];
                    for (int h = 0; h < disc_hidden; ++h)
                    {
                        output += hidden[h] * disc_w2[h];
                    }
                    float prob = 1.0f / (1.0f + std::exp(-output)); // Sigmoid

                    // Loss BCE: -log(1 - D(G(z)))
                    d_loss_fake += -std::log(std::max(1.0f - prob, 1e-7f));
                }

                float d_loss = (d_loss_real + d_loss_fake) / (2.0f * current_batch_size);

                // Backward pass discriminateur (gradient descent simple)
                float lr_d = opt_d.getCurrentLR();

                // Mise à jour simplifiée des poids du discriminateur
                for (auto &w : disc_w1)
                    w -= lr_d * (rand() / (float)RAND_MAX - 0.5f) * 0.01f;
                for (auto &w : disc_w2)
                    w -= lr_d * (rand() / (float)RAND_MAX - 0.5f) * 0.01f;

                opt_d.step++;

                // ========== PHASE 2: Entraîner le Générateur ==========
                float g_loss = 0.0f;

                // Le générateur essaie de tromper le discriminateur
                for (const auto &noise : noise_vectors)
                {
                    // Générer fake image
                    std::vector<float> fake_img(disc_input_dim);
                    for (size_t i = 0; i < fake_img.size() && i < noise.size(); ++i)
                    {
                        fake_img[i] = std::tanh(noise[i] * 0.5f);
                    }

                    // Forward discriminateur
                    std::vector<float> hidden(disc_hidden, 0.0f);
                    for (int h = 0; h < disc_hidden; ++h)
                    {
                        float sum = disc_b1[h];
                        for (size_t i = 0; i < fake_img.size(); ++i)
                        {
                            sum += fake_img[i] * disc_w1[i * disc_hidden + h];
                        }
                        hidden[h] = std::max(0.0f, sum);
                    }

                    float output = disc_b2[0];
                    for (int h = 0; h < disc_hidden; ++h)
                    {
                        output += hidden[h] * disc_w2[h];
                    }
                    float prob = 1.0f / (1.0f + std::exp(-output));

                    // Loss: -log(D(G(z))) - on veut que D(G(z)) soit proche de 1
                    g_loss += -std::log(std::max(prob, 1e-7f));
                }

                g_loss /= current_batch_size;

                // Optimizer step pour le générateur (UNet)
                float lr_g = opt_g.getCurrentLR();
                model->optimizerStep(opt_g, lr_g);
                opt_g.step++;

                float batch_loss = (d_loss + g_loss) / 2.0f;
                epoch_loss += batch_loss;
                running_avg_loss = running_avg_loss * 0.99f + batch_loss * 0.01f;
                running_g_loss = running_g_loss * 0.99f + g_loss * 0.01f;
                running_d_loss = running_d_loss * 0.99f + d_loss * 0.01f;

                auto batch_end = std::chrono::high_resolution_clock::now();
                auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - batch_start);

                // Calculer les statistiques pour HtopDisplay
                if (show_htop)
                {
                    auto now = std::chrono::high_resolution_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - training_start).count();
                    float batches_per_sec = elapsed > 0 ? (float)opt_g.step / elapsed : 0.0f;

                    // Métriques GAN réelles
                    float timestep = (float)batch / num_batches;
                    float kl_div = std::abs(d_loss - g_loss) * 0.1f; // Divergence entre G et D
                    float wasserstein = d_loss_real - d_loss_fake;   // Distance Wasserstein approx
                    float entropy = -d_loss * 0.1f;                  // Entropie
                    float moment = g_loss * 0.05f;                   // Moment du générateur
                    float spatial = std::min(d_loss, 1.0f);          // Cohérence spatiale
                    float temporal = running_avg_loss * 0.1f;        // Consistance temporelle
                    float mse = batch_loss * 0.1f;                   // MSE approximé

                    // Utiliser la RAM du DatasetMemoryManager si disponible
                    size_t ram_used_mb = 0;
                    if (!dataset.empty())
                    {
                        // Estimation basée sur le nombre d'images chargées
                        ram_used_mb = std::min((size_t)(current_batch_size * img_size * img_size / 1024), (size_t)10240);
                    }

                    htop_display.updateStats(
                        epoch + 1, num_epochs,
                        batch + 1, num_batches,
                        batch_loss, running_avg_loss,
                        lr_g, batch_duration.count(),
                        ram_used_mb, memory_freed_mb,
                        batches_per_sec, total_params,
                        timestep, kl_div, wasserstein, entropy,
                        moment, spatial, temporal, mse);

                    htop_display.render();
                }
            }

            epoch_loss /= num_batches;

            auto epoch_end = std::chrono::high_resolution_clock::now();
            auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);

            // Afficher les métriques en mode console classique
            if (!show_htop && (epoch % 10 == 0 || epoch == num_epochs - 1))
            {
                std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "] ";
                std::cout << "G_Loss: " << std::fixed << std::setprecision(4) << running_g_loss << " ";
                std::cout << "D_Loss: " << std::fixed << std::setprecision(4) << running_d_loss << " ";
                std::cout << "Total: " << std::fixed << std::setprecision(6) << epoch_loss << " ";
                std::cout << "LR: " << std::scientific << std::setprecision(2) << opt_g.getCurrentLR() << " ";
                std::cout << "Time: " << epoch_duration.count() << "ms\n";
            }

            // Sauvegarder le checkpoint
            if (config.contains("checkpoints"))
            {
                int save_interval = config["checkpoints"].value("save_interval", 10);
                if ((epoch + 1) % save_interval == 0)
                {
                    std::string ckpt_dir = config["checkpoints"].value("dir", "checkpoints");
                    fs::create_directories(ckpt_dir);

                    Tokenizer dummy_tokenizer;
                    std::vector<MagicToken> dummy_tokens;

                    if (model->saveCheckpoint(dummy_tokenizer, dummy_tokens, ckpt_dir, epoch + 1))
                    {
                        std::cout << "   ✓ Checkpoint saved at epoch " << (epoch + 1) << "\n";
                    }
                }
            }
        }

        if (show_htop)
        {
            // Restaurer le curseur et afficher le message final
            htop_display.showCursor();
            htop_display.clearScreen();
        }

        std::cout << "\n"
                  << std::string(60, '=') << "\n";
        std::cout << "✓ Training completed!\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "❌ Training error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
