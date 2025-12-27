# Architecture de Mímir

## 🏗️ Vue d'Ensemble

Mímir est structuré en plusieurs couches logiques qui interagissent pour fournir un framework de deep learning complet et performant.

```
┌─────────────────────────────────────────────────────────┐
│                    API Lua (Scripting)                  │
├─────────────────────────────────────────────────────────┤
│              Modèles de Haut Niveau (7)                 │
│  Encoder | Decoder | EncDec | AE | UNet | ViT | MM      │
├─────────────────────────────────────────────────────────┤
│                  Classe Model (Base)                    │
│           Gestion Params | Optimizers | Save/Load       │
├─────────────────────────────────────────────────────────┤
│                   Layers & Operations                   │
│     Conv | Pooling | Activation | Normalization         │
├─────────────────────────────────────────────────────────┤
│              Système de Tenseurs & Autograd             │
│          Tensors | ComputationGraph | Gradients         │
├─────────────────────────────────────────────────────────┤
│                  Optimisations Backend                  │
│         SIMD (AVX2) | OpenCL | OpenMP | RAM Mgr         │
└─────────────────────────────────────────────────────────┘
```

---

## 📦 Composants Principaux

### 1. **Système de Tenseurs** (`tensors.hpp/cpp`)

Structure de données fondamentale :

```cpp
struct tensor {
    std::uint16_t   Weight;      // Poids quantifié
    Vector4F        Pos;         // Position 4D
    std::uint8_t    Value;       // Valeur
    std::uint16_t   Length;      // Longueur
    std::vector<float> data;     // Données float
};
```

**TensorSystem** :
- Contexte OpenCL
- Compilation de kernels
- Calcul de poids sur GPU

---

### 2. **Classe Model** (Base Concrète)

Classe centrale dont héritent tous les modèles :

```cpp
class Model {
public:
    std::vector<tensor> params;          // Paramètres du modèle
    std::vector<LayerDesc> layers;       // Description des couches
    
    // Construction
    Model();
    virtual ~Model();
    void build();                        // Construction générique
    void autoBuildFromDataset(const std::string& dataset_dir);
    void allocateParams();               // Allocation mémoire
    
    // Forward/Backward
    void forward(std::vector<uint8_t>&) const;
    DecoderOutput eval(const std::vector<uint8_t>& target) const;
    
    // Optimisation
    void optimizerStep(Optimizer& opt, float lr, const Gradients* grads = nullptr);
    void applyParamUpdate(float learning_rate);
    
    // Gestion des layers
    void push(const std::string& name, const std::string& type, size_t params_count);
    size_t totalParamCount() const;
    
    // Sauvegarde/Chargement
    bool saveCheckpoint(const Tokenizer& tok, const std::vector<MagicToken>& magic_tokens,
                       const fs::path& dir, int epoch);
    bool packToSafetensor(const fs::path& outpath,
                         const std::unordered_map<std::string, std::vector<float>>& tensors) const;
    bool tryLoadExistingModel(const fs::path& ckdir, const fs::path& safep,
                              Tokenizer& outTok, Encoder& outEnc,
                              std::vector<MagicToken>& outMagic);
    
    // Hooks virtuels pour branches multimodales
    virtual void buildBackboneUNet(int stages, int blocks_per_stage, int bottleneck_depth);
    virtual void injectMagicToken(const MagicToken& tok);
    virtual void buildTextBranch(const MagicToken& tok);
    virtual void buildAudioBranch(const MagicToken& tok);
    virtual void buildImageBranch(const MagicToken& tok);
    virtual void buildVideoBranch(const MagicToken& tok);
    
    // Helpers statiques
    static float weightToFloat(uint16_t w);
    static void conv2d_same(const std::vector<float>& in, std::vector<float>& out,
                            int W, int H, const std::vector<float>& kernel, int ksize);
    
protected:
    Tokenizer tokenizer;
    Encoder encoder;
    bool hasTokenizer = true;
    bool hasEncoder = true;
    std::vector<float> lastEncoding;
    double densityFactor = 1.0;
    int tw = 64, th = 64;
};
```

**Responsabilités** :
- Gestion des paramètres (uint16 quantifiés)
- Construction automatique depuis dataset (autoBuildFromDataset)
- 3 optimiseurs : SGD, Adam, AdamW avec LR decay
- Sauvegarde atomique avec checkpoints versionnés
- Support multimodal avec magic tokens
- Conversion SafeTensors

---

### 3. **7 Architectures de Modèles**

#### **EncoderModel** (BERT-like)
```cpp
struct Config {
    int vocab_size = 30000;
    int embed_dim = 768;
    int num_layers = 12;
    int num_heads = 12;
    bool bidirectional = true;
};
```

#### **DecoderModel** (GPT-like)
```cpp
struct Config {
    int vocab_size = 30000;
    int embed_dim = 768;
    int num_layers = 12;
    bool use_causal_mask = true;
};
```

#### **EncoderDecoderModel** (T5-like)
```cpp
struct Config {
    int num_encoder_layers = 6;
    int num_decoder_layers = 6;
    bool shared_embeddings = true;
    bool use_cross_attention = true;
};
```

#### **AutoencoderModel** (VAE)
```cpp
struct Config {
    int input_dim = 784;
    int latent_dim = 128;
    std::vector<int> encoder_dims = {512, 256};
    bool variational = false;
};
```

#### **UNetModel** (Segmentation/Génération)
```cpp
struct Config {
    int input_channels = 1;
    int output_channels = 1;
    int base_channels = 64;
    int num_levels = 4;
    bool use_batch_norm = true;
};
```

#### **VisionTransformerModel** (ViT)
```cpp
struct Config {
    int image_size = 224;
    int patch_size = 16;
    int num_classes = 1000;
    int embed_dim = 768;
};
```

#### **MultiModalModel** (Fusion)
```cpp
struct Config {
    int text_embed_dim = 512;
    int vision_embed_dim = 512;
    int audio_embed_dim = 512;
    int fusion_dim = 512;
    bool use_audio = false;
};
```

---

### 4. **Système Autograd**

#### **ComputationGraph**
Stocke les activations du forward pass pour le backward :

```cpp
struct ComputationGraph {
    std::vector<float> token_embeddings;
    std::vector<float> pos_encodings;
    
    struct LayerActivations {
        std::vector<float> input;
        std::vector<float> attn_out;
        std::vector<float> ffn_out;
    };
    std::vector<LayerActivations> layers;
};
```

#### **Gradients**
Gestion des gradients avec clipping :

```cpp
struct Gradients {
    std::unordered_map<size_t, float> param_grads;
    
    void add(size_t param_idx, float grad);
    void clip(float max_norm);
    void zero();
};
```

---

### 5. **Optimiseurs**

```cpp
struct Optimizer {
    OptimizerType type; // SGD | ADAM | ADAMW
    
    // Adam parameters
    std::vector<float> m, v;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.01f;  // Pour AdamW
    size_t step = 0;
    
    // Learning rate decay
    LRDecayStrategy decay_strategy = LRDecayStrategy::COSINE;
    float initial_lr = 5e-5f;
    float min_lr = 1e-6f;
    float decay_rate = 0.95f;
    int decay_steps = 100;
    int total_steps = 1000;
    int warmup_steps = 0;
    
    void ensure(size_t n);           // Alloue m, v si nécessaire
    float getCurrentLR() const;      // Calcule LR avec decay et warmup
};
```

**Implémentation dans Model::optimizerStep()** :
- **SGD** : Mise à jour simple des gradients
- **Adam** : Moments exponentiels avec bias correction
- **AdamW** : Adam avec weight decay découplé (appliqué directement aux poids)

**Stratégies de Decay** :
- `NONE` : Pas de decay
- `COSINE` : `lr = min_lr + (initial_lr - min_lr) * 0.5 * (1 + cos(π * progress))`
- `STEP` : `lr *= decay_rate` tous les `decay_steps`
- `EXPONENTIAL` : `lr = initial_lr * decay_rate^(step/decay_steps)`
- `LINEAR` : Décroissance linéaire de `initial_lr` vers `min_lr`

**Warmup** : Si `step < warmup_steps`, lr augmente linéairement vers `initial_lr`

---

### 6. **Layers & Operations**

#### **Activations**
ReLU, ReLU6, LeakyReLU, ELU, SELU, GELU, Swish, Mish, Tanh, Sigmoid, Softmax

#### **Convolutions**
- Conv1D, Conv2D, Conv3D
- Transposed Conv2D (Deconvolution)
- Support : padding, stride, dilation, groups

#### **Pooling**
- Max/Average/Min/Sum Pooling 2D
- Adaptive Average Pooling

#### **Normalization**
- Batch Normalization
- Layer Normalization
- RMS Normalization

---

### 7. **Optimisations Backend**

#### **SIMD (AVX2)**
```cpp
void matmul_avx2(float* C, const float* A, const float* B, 
                 size_t M, size_t N, size_t K);
void gelu_forward_avx2(float* out, const float* in, size_t N);
```

#### **OpenCL**
- Calcul GPU pour poids de tenseurs
- Kernels customisables
- Support multi-device

#### **OpenMP**
```cpp
#pragma omp parallel for collapse(2) schedule(dynamic)
```

#### **RAM Manager Avancé**
- Compression LZ4 (RLE)
- Éviction LRU
- Prédiction d'accès
- Chargement asynchrone
- Limite RAM configurable

---

### 8. **API Lua**

#### **LuaScripting**
Bridge Lua-C++ pour contrôle sans recompilation :

```lua
-- Créer modèle
model.create("encoder", {num_layers = 6, d_model = 512})
model.build()

-- Entraîner
model.train(10, 0.0001)

-- Sauvegarder
model.save("checkpoints/my_model")
```

#### **LuaContext (Singleton)**
État global accessible depuis Lua :
- `currentModel`
- `currentTokenizer`
- `currentEncoder`
- `currentSequences`
- `currentConfig`

---

### 9. **Utilitaires**

#### **Tokenizer** (BPE)
```cpp
class Tokenizer {
    Tokenizer(int max_vocab = 20000);
    
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& tokens);
    void train(const std::vector<std::string>& corpus, int num_merges);
    
    json to_json() const;
    void from_json(const json& j);
    
    size_t getVocabSize() const;
    void setMaxVocab(int max);
};
```

#### **Encoder**
```cpp
class Encoder {
    Encoder(int dim = 64, int vocab_size = 20000);
    
    std::vector<float> encode(const std::vector<int>& tokens);
    
    int dim;
    int vocab_size;
    std::vector<float> token_embeddings;
    bool usePositionalEncoding;
    bool useSpecialEmbeddings;
};
```

#### **DatasetMemoryManager** (Singleton)
Gestion intelligente de la RAM pour datasets :
```cpp
struct DatasetMemoryManager {
    static DatasetMemoryManager& instance();
    
    void setMaxRAM(size_t bytes);
    bool canAllocate(size_t bytes) const;
    void trackAllocation(void* ptr, size_t bytes);
    void trackDeallocation(void* ptr);
    
    size_t getCurrentRAM() const;
    size_t getPeakRAM() const;
    size_t getAvailableRAM() const;
    float getUsagePercent() const;
    void printStats() const;
};
```

#### **MagicToken** (Multimodalité)
```cpp
struct MagicToken {
    uint32_t modality_mask;  // bits: text(0x01), image(0x02), audio(0x04), video(0x08)
    uint32_t seed;           // Seed pour reproduction
    float embed[8];          // Embedding fixe 8D
};
```

#### **Visualizer** (SFML)
- Graphiques de loss temps réel
- Affichage d'images générées
- Monitoring métriques
- Configuration fenêtre (1280×720 @ 120 FPS)

#### **HtopDisplay**
- Monitoring système dans terminal
- Métriques : epoch, batch, loss, accuracy, LR, RAM, batches/sec
- Support métriques VAE (KL divergence, reconstruction loss)

---

## 🔄 Flux de Données

### Construction Automatique du Modèle

```
Dataset Directory
         ↓
    detectModalities() → Analyse fichiers (text/image/audio/video)
         ↓
    loadDatasetCached() → Chargement avec gestion RAM
         ↓
    Model.autoBuildFromDataset()
         ↓
    buildBackboneUNet(4, 2, 3) → Architecture de base
         ↓
    Pour chaque modalité détectée:
    ├─ buildTextBranch() → Branche texte
    ├─ buildImageBranch() → Branche image
    ├─ buildAudioBranch() → Branche audio
    └─ buildVideoBranch() → Branche vidéo
         ↓
    injectMagicToken() → Injection des magic tokens
         ↓
    allocateParams() → Allocation mémoire
```

### Entraînement

```
Dataset → loadDatasetCached()
         ↓
    DatasetMemoryManager → Gestion RAM (10GB max)
         ↓
    Tokenizer.encode() → Tokens
         ↓
    Encoder.encode() → Embeddings
         ↓
    Model.forward() → Predictions (uint8)
         ↓
    Model.eval() → MSE Loss + Logits
         ↓
    Gradients (si fournis) ou cible - prédiction
         ↓
    Model.optimizerStep() → SGD/Adam/AdamW
         ├─ getCurrentLR() → LR avec decay + warmup
         ├─ Bias correction (Adam/AdamW)
         └─ Weight decay découplé (AdamW)
         ↓
    Update params[].Weight (uint16)
         ↓
    HtopDisplay.render() → Monitoring
```

### Sauvegarde Atomique

```
Model + Tokenizer + MagicTokens
         ↓
    saveCheckpoint(dir, epoch)
         ↓
    Création tmpdir (.tmp)
         ↓
    ├─ tokenizer.json (avec sanitize_id2token)
    ├─ metadata.json (timestamp, epoch, magic_tokens)
    └─ weights.u16 (paramètres little-endian)
         ↓
    Écriture atomique dans .tmp
         ↓
    Rename .tmp → epoch_N
         ↓
    Cleanup ancien checkpoint
```

### Inférence

```
Input (texte/image)
         ↓
    Tokenizer.encode() → Tokens
         ↓
    Encoder.encode() → Embeddings
         ↓
    Model.forward() → Prédictions uint8
         ↓
    Tokenizer.decode() → Output
```

---

## 📊 Diagramme de Classes (Simplifié)

```
Model (abstract)
├── EncoderModel
├── DecoderModel
├── EncoderDecoderModel
├── AutoencoderModel
├── UNetModel
├── VisionTransformerModel
└── MultiModalModel

TensorSystem
├── OpenCL Context
├── Kernels
└── Compute

Optimizer
├── SGD
├── Adam
└── AdamW

LuaScripting
├── API Functions
└── LuaContext (singleton)
```

---

## 🎯 Design Patterns Utilisés

2. **Singleton** : 
   - `LuaContext` (état Lua global)
   - `AdvancedRAMManager` (gestion RAM)
   - `DatasetMemoryManager` (tracking RAM datasets)
3. **Template Method** : 
   - `Model::build()` appelle hooks virtuels
   - `autoBuildFromDataset()` orchestre construction
4. **Strategy** : 
   - `LRDecayStrategy` (5 stratégies: None, Cosine, Step, Exponential, Linear)
   - `OptimizerType` (SGD, Adam, AdamW)
   - `PoolingStrategy` (CLS_TOKEN, MEAN, MAX, MEAN_SQRT)
5. **Observer** : 
   - `Visualizer` pour monitoring SFML
   - `HtopDisplay` pour monitoring terminal
6. **Atomic Operations** : Sauvegarde avec .tmp + rename
7. **Lazy Loading** : `DatasetItem` avec chargement à la demande

---

## 🚀 Extensibilité

### Ajouter une Nouvelle Architecture

```cpp
// 1. Hériter de Model
class MyModel : public Model {
public:
    struct Config { /* ... */ };
    
    explicit MyModel(const Config& cfg) : config(cfg) {}
    
    // Optionnel: surcharger build() ou utiliser les hooks
    void buildBackboneUNet(int stages, int blocks, int depth) override {
        // Construction custom
        push("layer1", "Conv2D", 64 * 64 * 3);
        push("layer2", "ReLU", 0);
        // ...
    }
};

// 2. Utilisation directe
MyModel::Config config;
config.num_layers = 12;

MyModel model(config);
model.build();  // Appelle buildBackboneUNet
model.allocateParams();

// 3. Ou via autoBuildFromDataset
model.autoBuildFromDataset("datasets/mydata");
```

### Ajouter une Branche Multimodale

```cpp
class MyModel : public Model {
    void buildCustomBranch(const MagicToken& tok) override {
        // Analyser modality_mask
        if (tok.modality_mask & 0x10) {  // Nouvelle modalité (bit 4)
            push("custom_encoder", "Transformer", 512 * 512);
            push("custom_projection", "Linear", 512 * 256);
            // ...
        }
    }
};
```

### Ajouter une Fonction Lua

```cpp
// 1. Déclarer dans LuaScripting.hpp
static int lua_myFunction(lua_State* L);

// 2. Implémenter dans LuaScripting.cpp
int LuaScripting::lua_myFunction(lua_State* L) {
    // Logique
    return 1; // Nombre de valeurs retournées
}

// 3. Enregistrer dans registerAPI()
lua_register(L, "myFunction", lua_myFunction);
```

---

## 📈 Performance & Optimisations

### Compilation
```makefile
FLAGS = -O3 -march=native -mavx2 -mfma -fopenmp
      -ffp-contract=fast -funroll-loops
      -funsafe-math-optimizations
      -fno-trapping-math -fno-math-errno
```

### Optimisations Actives

**SIMD AVX2** :
- Traitement par blocs de 8 floats (256 bits)
- FMA (Fused Multiply-Add) pour matmul
- Opérations vectorisées : add, mul, GELU

**Multi-threading OpenMP** :
- Parallélisation automatique des boucles
- `#pragma omp parallel for collapse(2) schedule(dynamic)`
- Utilise tous les cœurs disponibles

**GPU OpenCL** :
- Calcul de poids de tenseurs
- Kernels customisables
- Support multi-vendor (NVIDIA, AMD, Intel)

**Gestion Mémoire** :
- Paramètres quantifiés uint16 (2 bytes au lieu de 4)
- `DatasetMemoryManager` avec limite 10GB
- Tracking précis des allocations
- Lazy loading des datasets
- Compression LZ4 (AdvancedRAMManager)

**Sauvegarde Optimisée** :
- Écriture atomique (.tmp + rename)
- Sanitization des strings pour éviter corruption
- Format binaire little-endian
- SafeTensors pour interopérabilité

### Benchmarks

| Opération | Taille | Temps | Throughput |
|-----------|--------|-------|------------|
| MatMul AVX2 | 1024×1024 | ~45 ms | ~48 GFLOPS |
| GELU AVX2 | 1M elements | ~2.3 ms | ~435 MOps/s |
| Conv2D | 64×64×64 | ~8 ms | ~33 MPixels/s |
| Checkpoint save | 10M params | ~150 ms | ~133 MB/s |

**Configuration test** : Intel i7-11700K @ 3.6GHz, 8 cores, 32GB RAM, AVX2

---

## 🔗 Voir Aussi

- [Runtime Engine](02-Runtime-Engine.md)
- [API Reference](../03-API-Reference/00-API-Complete.md)
- [Hardware Optimizations](07-Hardware-Optimizations.md)
- [Getting Started](../01-Getting-Started/01-Quick-Start.md)
