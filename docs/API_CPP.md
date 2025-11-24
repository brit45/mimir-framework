# API C++ - Mímir

Référence complète de l'API C++ pour développement avancé avec Mímir.

---

## 📋 Table des Matières

- [Classe Model](#classe-model)
- [Tensors](#tensors)
- [Optimizers](#optimizers)
- [Layers](#layers)
- [Tokenizer](#tokenizer)
- [Encoder](#encoder)
- [Visualizer](#visualizer)

---

## Classe Model

### Vue d'Ensemble

```cpp
class Model {
public:
    // Construction
    Model();
    virtual ~Model();
    
    void build();
    void allocateParams();
    virtual void buildArchitecture() = 0;
    
    // Forward/Backward
    void forward(std::vector<uint8_t>&) const;
    DecoderOutput eval(const std::vector<uint8_t>& target) const;
    
    // Optimisation
    void optimizerStep(Optimizer& opt, float lr, const Gradients* grads = nullptr);
    void applyParamUpdate(float learning_rate);
    
    // Sauvegarde
    bool saveCheckpoint(const Tokenizer& tok, 
                       const std::vector<MagicToken>& magic_tokens,
                       const fs::path& dir, int epoch);
    bool packToSafetensor(const fs::path& outpath,
                         const std::unordered_map<std::string, std::vector<float>>& tensors) const;
    
    // Accesseurs
    size_t totalParamCount() const;
    std::vector<tensor>& getMutableParams();
    const Tokenizer& getTokenizer() const;
    Tokenizer& getMutableTokenizer();
    
protected:
    std::vector<tensor> params;
    std::vector<LayerDesc> layers;
    Tokenizer tokenizer;
    Encoder encoder;
};
```

### Méthodes Principales

#### `void build()`

Construit l'architecture du modèle en appelant `buildArchitecture()`.

**Exemple :**

```cpp
class MyModel : public Model {
    void buildArchitecture() override {
        // Ajouter des layers
        push("embedding", "Embedding", 768 * 30000);
        push("transformer_1", "Transformer", 768 * 768 * 4);
        // ...
    }
};

MyModel model;
model.build();
```

---

#### `void allocateParams()`

Alloue la mémoire pour tous les paramètres du modèle.

**Exemple :**

```cpp
model.build();
model.allocateParams();
std::cout << "Paramètres: " << model.totalParamCount() << std::endl;
```

---

#### `void optimizerStep(Optimizer& opt, float lr, const Gradients* grads)`

Effectue une étape d'optimisation avec MAJ des paramètres.

**Paramètres :**
- `opt` : Optimiseur (contient m, v pour Adam)
- `lr` : Learning rate (peut être ignoré si opt.getCurrentLR() utilisé)
- `grads` : Gradients calculés (optionnel)

**Exemple :**

```cpp
Optimizer opt;
opt.type = OptimizerType::ADAM;
opt.initial_lr = 0.001f;
opt.decay_strategy = LRDecayStrategy::COSINE;
opt.total_steps = 10000;

for (int step = 0; step < 10000; ++step) {
    // Forward + backward...
    Gradients grads;
    // ... calcul des gradients
    
    model.optimizerStep(opt, 0.0f, &grads);
    opt.step++;
}
```

---

#### `size_t totalParamCount()`

Retourne le nombre total de paramètres du modèle.

**Retour :** Nombre de paramètres

**Exemple :**

```cpp
size_t params = model.totalParamCount();
std::cout << "Modèle avec " << params << " paramètres" << std::endl;
std::cout << "Taille mémoire: " << (params * sizeof(float) / 1024 / 1024) << " MB" << std::endl;
```

---

### Helpers Statiques

#### `static float weightToFloat(uint16_t w)`

Convertit un poids quantifié (uint16) en float [-1, 1].

```cpp
uint16_t w = 32768;  // 0.5 en uint16
float f = Model::weightToFloat(w);  // 0.0
```

---

#### `static void conv2d_same(...)`

Convolution 2D avec padding "same".

```cpp
std::vector<float> input(64 * 64);
std::vector<float> output;
std::vector<float> kernel = {1, 0, -1, 2, 0, -2, 1, 0, -1};  // Sobel
int ksize = 3;

Model::conv2d_same(input, output, 64, 64, kernel, ksize);
```

---

#### Activations In-Place

```cpp
// ReLU
std::vector<float> x = {-1, 0, 1, 2, -3};
Model::relu_inplace(x);  // {0, 0, 1, 2, 0}

// Leaky ReLU
Model::leaky_relu_inplace(x, 0.01f);

// Tanh
Model::tanh_inplace(x);

// Softmax
std::vector<float> logits = {1.0, 2.0, 3.0};
Model::softmax_inplace(logits);  // [0.09, 0.24, 0.67]
```

---


Namespace contenant les 7 architectures pré-implémentées.

### EncoderModel

```cpp
#include "Model.hpp"

EncoderModel::Config config;
config.vocab_size = 30000;
config.embed_dim = 768;
config.num_layers = 12;
config.num_heads = 12;
config.ffn_dim = 3072;
config.max_seq_len = 512;
config.dropout = 0.1f;
config.bidirectional = true;

EncoderModel encoder(config);
encoder.buildArchitecture();
encoder.allocateParams();

// Forward
std::vector<int> tokens = {101, 2023, 2003, 1037, 3231, 102};
auto embeddings = encoder.encode(tokens);

// Pooling
auto pooled = encoder.pool(embeddings, 
    EncoderModel::PoolingStrategy::CLS_TOKEN);
```

**Stratégies de Pooling :**
- `CLS_TOKEN` : Utiliser le token [CLS]
- `MEAN` : Mean pooling sur toute la séquence
- `MAX` : Max pooling
- `MEAN_SQRT` : Mean avec division par sqrt(seq_len)

---

### DecoderModel

```cpp
DecoderModel::Config config;
config.vocab_size = 50000;
config.embed_dim = 768;
config.num_layers = 12;
config.num_heads = 12;
config.use_causal_mask = true;  // Auto-régressif

DecoderModel decoder(config);
decoder.buildArchitecture();
decoder.allocateParams();

// Génération
std::vector<int> prompt = {1, 2, 3};  // "Once upon a time"
auto generated = decoder.generate(prompt, 
    /* max_new_tokens */ 50,
    /* temperature */ 1.0f,
    /* top_p */ 0.9f,
    /* seed */ 42);

for (int token : generated) {
    std::cout << token << " ";
}
```

---

### AutoencoderModel (VAE)

```cpp
AutoencoderModel::Config config;
config.input_dim = 784;  // 28×28
config.latent_dim = 128;
config.encoder_dims = {512, 256};
config.decoder_dims = {256, 512};
config.variational = true;  // VAE
config.kl_weight = 1.0f;

AutoencoderModel vae(config);
vae.buildArchitecture();
vae.allocateParams();

// Encode
std::vector<float> input(784, 0.5f);
auto latent = vae.encode(input);

// Decode
auto reconstructed = vae.decode(latent);

// Loss
float recon_loss = vae.reconstructionLoss(input, reconstructed);
float kl_loss = vae.klDivergence(mean, logvar);
float total_loss = recon_loss + config.kl_weight * kl_loss;
```

---

### UNetModel

```cpp
UNetModel::Config config;
config.input_channels = 3;   // RGB
config.output_channels = 21; // 21 classes
config.base_channels = 64;
config.num_levels = 4;
config.use_batch_norm = true;

UNetModel unet(config);
unet.buildArchitecture();

// Segmentation
std::vector<uint8_t> image(224 * 224 * 3);  // Image RGB
auto segmentation = unet.segment(image, 224, 224);

// Forward (float)
std::vector<float> image_f(224 * 224 * 3);
auto output = unet.forward(image_f, 224, 224);
```

---

### VisionTransformerModel (ViT)

```cpp
VisionTransformerModel::Config config;
config.image_size = 224;
config.patch_size = 16;
config.num_channels = 3;
config.embed_dim = 768;
config.num_layers = 12;
config.num_heads = 12;
config.num_classes = 1000;

VisionTransformerModel vit(config);
vit.buildArchitecture();

// Classification
std::vector<uint8_t> image(224 * 224 * 3);
int predicted_class = vit.classify(image);
std::cout << "Classe prédite: " << predicted_class << std::endl;

// Features
auto features = vit.extractFeatures(image);
```

---

### MultiModalModel

```cpp
MultiModalModel::Config config;
config.text_vocab_size = 30000;
config.text_embed_dim = 512;
config.image_size = 224;
config.vision_embed_dim = 512;
config.fusion_dim = 512;
config.use_audio = false;

MultiModalModel mm(config);
mm.buildArchitecture();

// Encode modalités
std::vector<int> text_tokens = {1, 2, 3, 4};
std::vector<uint8_t> image(224 * 224 * 3);

auto text_emb = mm.encodeText(text_tokens);
auto vision_emb = mm.encodeImage(image);

// Fusion
auto fused = mm.fuse(text_emb, vision_emb);

// Forward complet
auto output = mm.forward(text_tokens, image);
```

---

## Tensors

### Structure `tensor`

```cpp
struct tensor {
    std::uint16_t Weight;
    Vector4F Pos;
    std::uint8_t Value;
    std::uint16_t Length;
    std::vector<float> data;
    
    tensor();
    explicit tensor(size_t size);
    explicit tensor(const std::vector<float>& values);
};
```

**Exemple :**

```cpp
// Créer tensor
tensor t(1024);  // 1024 floats
t.data[0] = 0.5f;

// Depuis vecteur
std::vector<float> weights = {0.1, 0.2, 0.3};
tensor t2(weights);
```

---

### TensorSystem (OpenCL)

```cpp
class TensorSystem {
public:
    TensorSystem();
    ~TensorSystem();
    
    bool initialize();
    bool computeWeights(std::vector<tensor>& tensors);
};
```

**Utilisation :**

```cpp
TensorSystem ts;
if (!ts.initialize()) {
    std::cerr << "Échec initialisation OpenCL" << std::endl;
    return 1;
}

std::vector<tensor> tensors(1000, tensor(256));
ts.computeWeights(tensors);
```

---

## Optimizers

### Structure `Optimizer`

```cpp
struct Optimizer {
    OptimizerType type;  // SGD, ADAM, ADAMW
    
    // Adam parameters
    std::vector<float> m, v;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.01f;
    size_t step = 0;
    
    // LR Decay
    LRDecayStrategy decay_strategy;
    float initial_lr;
    float min_lr;
    int warmup_steps;
    int total_steps;
    
    void ensure(size_t n);
    float getCurrentLR() const;
};
```

**Exemple complet :**

```cpp
Optimizer opt;
opt.type = OptimizerType::ADAMW;
opt.beta1 = 0.9f;
opt.beta2 = 0.999f;
opt.weight_decay = 0.01f;

// LR Decay
opt.decay_strategy = LRDecayStrategy::COSINE;
opt.initial_lr = 0.001f;
opt.min_lr = 1e-6f;
opt.warmup_steps = 1000;
opt.total_steps = 100000;

// Entraînement
for (int step = 0; step < 100000; ++step) {
    // Forward + backward
    Gradients grads;
    // ...
    
    float lr = opt.getCurrentLR();
    model.optimizerStep(opt, lr, &grads);
    opt.step++;
    
    if (step % 100 == 0) {
        std::cout << "Step " << step << " | LR: " << lr << std::endl;
    }
}
```

---

## Layers

Voir `Layers.hpp` pour la bibliothèque complète.

### Activations

```cpp
#include "Layers.hpp"

std::vector<float> x = {-1.0, 0.0, 1.0, 2.0};

// ReLU
Activation::relu_inplace(x);

// GELU
for (auto& val : x) {
    val = Activation::gelu(val);
}

// Softmax
Activation::softmax(x);

// Application générique
Activation::apply_inplace(x, ActivationType::SWISH);
```

---

### Convolutions

```cpp
#include "Layers.hpp"

// Conv2D
std::vector<float> input(64 * 64 * 3);  // 64×64 RGB
std::vector<float> output;
std::vector<float> kernel(3 * 3 * 3 * 64);  // 3×3, 3 in, 64 out
std::vector<float> bias(64);

Conv::conv2d(input, output, kernel, bias,
    /* in_height */ 64,
    /* in_width */ 64,
    /* in_channels */ 3,
    /* out_channels */ 64,
    /* kernel_size */ 3,
    /* stride */ 1,
    /* padding */ 1);
```

---

### Pooling

```cpp
#include "Layers.hpp"

std::vector<float> input(128 * 128 * 64);
std::vector<float> output;

// Max Pooling 2×2
Pooling::maxpool2d(input, output,
    /* in_height */ 128,
    /* in_width */ 128,
    /* channels */ 64,
    /* kernel_size */ 2,
    /* stride */ 2);
// Output: 64×64×64

// Adaptive Average Pooling
Pooling::adaptive_avgpool2d(input, output,
    /* in_height */ 128,
    /* in_width */ 128,
    /* channels */ 64,
    /* out_height */ 7,
    /* out_width */ 7);
// Output: 7×7×64
```

---

### Normalization

```cpp
#include "Layers.hpp"

std::vector<float> data(batch_size * channels * spatial_size);
std::vector<float> gamma(channels, 1.0f);
std::vector<float> beta(channels, 0.0f);
std::vector<float> running_mean(channels, 0.0f);
std::vector<float> running_var(channels, 1.0f);

// Batch Normalization
Normalization::batch_norm(data, gamma, beta, 
    running_mean, running_var,
    batch_size, channels, spatial_size,
    /* eps */ 1e-5f,
    /* training */ true);

// Layer Normalization
Normalization::layer_norm(data, gamma, beta,
    /* normalized_size */ hidden_dim,
    /* eps */ 1e-5f);

// RMS Normalization
Normalization::rms_norm(data, gamma,
    /* normalized_size */ hidden_dim,
    /* eps */ 1e-6f);
```

---

## Tokenizer

```cpp
#include "Tokenizer.hpp"

Tokenizer tokenizer;
tokenizer.setMaxVocab(32000);

// Tokenize
std::vector<int> tokens = tokenizer.encode("Hello world!");
for (int id : tokens) {
    std::cout << id << " ";
}

// Detokenize
std::string text = tokenizer.decode(tokens);
std::cout << text << std::endl;

// Save/Load
tokenizer.save("tokenizer.json");
Tokenizer::load("tokenizer.json", tokenizer);
```

---

## Encoder

```cpp
#include "Encoder.hpp"

Encoder encoder;
encoder.setEmbeddingDim(256);
encoder.setUsePositionalEncoding(true);
encoder.setUseSpecialEmbeddings(true);

// Encode
std::vector<int> tokens = {1, 2, 3, 4, 5};
auto embeddings = encoder.encode(tokens);
// embeddings.size() == 5 * 256
```

---

## Visualizer

```cpp
#include "Visualizer.hpp"

Visualizer viz;
viz.configure(1280, 720, "Training Monitor", 60);

// Entraînement loop
for (int epoch = 0; epoch < 100; ++epoch) {
    for (int step = 0; step < 1000; ++step) {
        // Training...
        float loss = /* ... */;
        
        // Update visualization
        viz.recordLoss(loss);
        viz.render();
        
        if (viz.shouldClose()) {
            break;
        }
    }
}
```

---

## Exemple Complet

### Entraînement d'un Encoder

```cpp
#include "Model.hpp"
#include "Tokenizer.hpp"
#include "Encoder.hpp"

int main() {
    // 1. Configuration
    EncoderModel::Config config;
    config.vocab_size = 30000;
    config.embed_dim = 768;
    config.num_layers = 12;
    config.num_heads = 12;
    config.ffn_dim = 3072;
    config.max_seq_len = 512;
    
    // 2. Créer modèle
    EncoderModel model(config);
    model.buildArchitecture();
    model.allocateParams();
    
    std::cout << "Paramètres: " << model.totalParamCount() << std::endl;
    
    // 3. Optimizer
    Optimizer opt;
    opt.type = OptimizerType::ADAMW;
    opt.initial_lr = 0.0001f;
    opt.decay_strategy = LRDecayStrategy::COSINE;
    opt.total_steps = 100000;
    opt.warmup_steps = 4000;
    
    // 4. Entraînement
    for (int step = 0; step < 100000; ++step) {
        // Forward
        std::vector<int> tokens = {/* ... */};
        auto output = model.encode(tokens);
        
        // Compute loss
        // ...
        
        // Backward
        Gradients grads;
        // ...
        
        // Update
        model.optimizerStep(opt, 0.0f, &grads);
        opt.step++;
        
        if (step % 1000 == 0) {
            std::cout << "Step " << step 
                     << " | LR: " << opt.getCurrentLR() << std::endl;
        }
    }
    
    // 5. Save
    model.saveCheckpoint(tokenizer, magic_tokens, "checkpoints", 100);
    
    return 0;
}
```

---

## Compilation

```bash
g++ -std=c++17 -O3 -march=native -mavx2 -fopenmp \
    my_program.cpp \
    src/Model.cpp \
    src/tensors.cpp \
    src/Tokenizer.cpp \
    src/Encoder.cpp \
    src/Visualizer.cpp \
    src/Sha256.cpp \
    src/stb_image_impl.cpp \
    -I./src -I/usr/include/lua5.3 \
    -o my_program \
    -lOpenCL -lsfml-graphics -lsfml-window -lsfml-system -llua5.3 -fopenmp
```

---

## 📚 Ressources

- [Architecture](ARCHITECTURE.md)
- [Layers](LAYERS.md)
- [SIMD Operations](SIMD_OPS.md)
- [Exemples C++](../examples/)
