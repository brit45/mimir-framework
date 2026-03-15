# ⚙️ Runtime Engine - Moteur d'Exécution Interne

**Niveau** : Expert  
**Durée de lecture** : 30-45 minutes  
**Prérequis** : Connaissance C++17, concepts de deep learning

---

## 📋 Table des Matières

1. [Vue d'Ensemble](#vue-densemble)
2. [Architecture du Runtime](#architecture-du-runtime)
3. [Cycle d'Exécution](#cycle-dexécution)
4. [Gestion de la Mémoire](#gestion-de-la-mémoire)
5. [Bindings Lua](#bindings-lua)
6. [Threading et Async](#threading-et-async)
7. [Optimisations Runtime](#optimisations-runtime)
8. [Diagrammes de Flux](#diagrammes-de-flux)

---

## Vue d'Ensemble

Le **Runtime Engine** de Mímir est le cœur du framework. C'est lui qui orchestre l'exécution des modèles, la gestion de la mémoire, les calculs, et l'interface avec Lua.

### Composants Principaux

```text
┌─────────────────────────────────────────────────────────┐
│                    RUNTIME ENGINE                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌────────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │  Lua Runtime   │  │ Tensor Engine│  │Memory Engine│  │
│  │  (LuaScripting)│  │  (tensors.*) │  │(RAM Manager)│  │
│  └────────────────┘  └──────────────┘  └─────────────┘  │
│                                                         │
│  ┌────────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Model Engine  │  │Hardware Accel│  │Thread Pool   │ │
│  │  (Model.*)     │  │(HardwareOpt) │  │(AsyncMonitor)│ │
│  └────────────────┘  └──────────────┘  └──────────────┘ │
│                                                         │
│  ┌────────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │  Autograd      │  │   Layers     │  │ Tokenizer   │  │
│  │  (gradients)   │  │(computations)│  │ (vocab)     │  │
│  └────────────────┘  └──────────────┘  └─────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Flux d'Exécution Haut Niveau

```text
Script Lua → Lua Runtime → Model Engine → Tensor Engine → Hardware
    ↓            ↓              ↓               ↓             ↓
 read()      parse()       forward()       compute()     SIMD ops
    ↓            ↓              ↓               ↓             ↓
 execute()   call C++       backward()      allocate()    AVX2/FMA
    ↓            ↓              ↓               ↓             ↓
 return      push result   update params   free memory   result
```

---

## Architecture du Runtime

### 1. Initialisation

Lorsque `./bin/mimir --lua script.lua` est exécuté, le runtime fait (résumé fidèle à `src/main.cpp`) :

1. Affiche la bannière + exécute les checks `MemorySafety::*`.
2. Configure OpenMP (si compilé avec OpenMP) et affiche les capacités CPU (AVX2/FMA/F16C/BMI2).
3. Parse les options CLI.
4. Pour `--lua <script.lua>` : crée `LuaScripting`, injecte les arguments Lua, puis exécute le script.

Pseudo-code minimal (simplifié pour la doc) :

```cpp
// main.cpp (simplifié)
int main(int argc, char** argv) {
    // ... banner + MemorySafety + OpenMP + hardware flags ...

    if (argc >= 3 && std::string(argv[1]) == "--lua") {
        const std::string lua_script = argv[2];

        LuaScripting lua;
        std::vector<std::string> script_args;
        for (int i = 3; i < argc; ++i) script_args.emplace_back(argv[i]);

        // Injecte `arg[...]` et aussi `Mimir.Args[...]` côté Lua
        lua.setArgs(lua_script, script_args);

        // Exécute le script Lua (luaL_dofile)
        lua.loadScript(lua_script);
        return 0;
    }

    // ... autres modes: --config, --help ...
}
```

#### Détail de `registerAPI()`

Le runtime expose d'abord un namespace racine `Mimir`, puis ajoute des aliases globaux de confort.

```cpp
// LuaScripting::registerAPI() (simplifié)
void LuaScripting::registerAPI() {
    // Mimir.Model / Mimir.Architectures / Mimir.Dataset / ...
    lua_newtable(L);        // table Mimir
    lua_newtable(L);        // table Mimir.Model
    lua_pushcfunction(L, lua_createModel);
    lua_setfield(L, -2, "create");
    lua_pushcfunction(L, lua_trainModel);
    lua_setfield(L, -2, "train");
    // ... (build/allocate_params/init_weights/forward/backward/...) ...
    lua_setfield(L, -2, "Model");
    // ... autres sous-tables ...
    lua_setglobal(L, "Mimir");

    // Helpers globaux
    lua_pushcfunction(L, lua_print);
    lua_setglobal(L, "log");

    // Aliases rétrocompatibilité: model= Mimir.Model, dataset= Mimir.Dataset, ...
    // (voir `src/LuaScripting.cpp`)
}
```

### 2. Contexte Global (Singleton)

```cpp
class LuaContext {
public:
    static LuaContext& getInstance() {
        static LuaContext instance;  // Singleton Meyer's
        return instance;
    }
    
    // Objets partagés entre Lua et C++
    std::shared_ptr<Model> currentModel;
    std::shared_ptr<Tokenizer> currentTokenizer;
    std::shared_ptr<Encoder> currentEncoder;

    // Monitoring asynchrone (Htop/Viz)
    std::shared_ptr<AsyncMonitor> asyncMonitor;

    // Données (séquences tokenisées + dataset brut)
    std::vector<std::vector<int>> currentSequences;
    std::vector<DatasetItem> currentDataset;
    json currentConfig;

    // Configuration du modèle (registre)
    std::string modelType;
    json modelConfig;

    // Logs
    std::vector<std::string> logs;
    
private:
    LuaContext() = default;  // Privé pour singleton
};
```

**Pourquoi un singleton ?**

- Les callbacks Lua (fonctions `static int lua_*(lua_State*)`) ne peuvent pas accéder à `this`
- Le contexte global permet de passer des données entre Lua et C++
- Thread-safe car initialisé avant tout threading

---

## Cycle d'Exécution

### Exemple : `Mimir.Model.train(10, 3e-4)`

#### 1. Appel Lua

```lua
-- Recommandé (namespace explicite)
Mimir.Model.train(10, 3e-4)

-- Alias global disponible (rétrocompatibilité)
-- model.train(10, 3e-4)
```

#### 2. Callback C++

```cpp
int LuaScripting::lua_trainModel(lua_State* L) {
    // 1. Extraire paramètres depuis la pile Lua
    int epochs = lua_tointeger(L, 1);      // 10
    double lr = lua_tonumber(L, 2);        // 3e-4
    
    // 2. Récupérer le modèle depuis le contexte
    auto& ctx = LuaContext::getInstance();
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "No model created");
        return 2;
    }
    
    // 3. Appeler la méthode C++ du modèle
    try {
        ctx.currentModel->train(ctx.currentSequences, epochs, lr);
        lua_pushboolean(L, true);  // Succès
        return 1;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}
```

#### 3. Exécution Model::train()

```cpp
void Model::train(const std::vector<std::vector<int>>& sequences,
                  int epochs, double learning_rate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        
        for (size_t b = 0; b < sequences.size(); ++b) {
            // Forward pass
            auto outputs = forward(sequences[b]);
            
            // Compute loss
            double loss = computeLoss(outputs, sequences[b]);
            total_loss += loss;
            
            // Backward pass
            backward();
            
            // Optimizer step
            updateWeights(learning_rate);
            
            // Update monitoring (si activé)
            if (asyncMonitor) {
                asyncMonitor->updateMetrics({
                    .epoch = epoch,
                    .batch = b,
                    .loss = loss,
                    // ...
                });
            }
        }
        
        // Epoch terminée
        std::cout << "Epoch " << epoch << " loss: " 
                  << (total_loss / sequences.size()) << std::endl;
    }
}
```

#### 4. Forward Pass Détaillé

```cpp
Tensor Model::forward(const std::vector<int>& input_ids) {
    // 1. Embedding lookup
    Tensor embedded = embedding_layer->forward(input_ids);
    // embedded: [seq_len, embed_dim]
    
    // 2. Positional encoding
    embedded = positional_encoding(embedded);
    
    // 3. Transformer layers
    Tensor hidden = embedded;
    for (auto& layer : transformer_layers) {
        // Self-attention
        hidden = layer->attention(hidden);
        
        // Feed-forward
        hidden = layer->ffn(hidden);
        
        // Residual + LayerNorm
        hidden = layer->normalize(hidden);
    }
    
    // 4. Output projection
    Tensor logits = output_layer->forward(hidden);
    // logits: [seq_len, vocab_size]
    
    return logits;
}
```

#### 5. Backward Pass (Autograd)

```cpp
void Model::backward() {
    // Remonter la chaîne de calcul (computational graph)
    
    // 1. Gradient de la loss
    Tensor grad_loss = computeLossGradient();
    
    // 2. Backprop à travers output_layer
    Tensor grad_hidden = output_layer->backward(grad_loss);
    
    // 3. Backprop à travers transformer layers (ordre inverse)
    for (auto it = transformer_layers.rbegin(); 
         it != transformer_layers.rend(); ++it) {
        grad_hidden = (*it)->backward(grad_hidden);
    }
    
    // 4. Backprop à travers embedding
    embedding_layer->backward(grad_hidden);
    
    // Les gradients sont stockés dans chaque layer
}
```

#### 6. Optimizer Step

```cpp
void Model::updateWeights(double lr) {
    // Pour chaque layer avec des paramètres
    for (auto& layer : all_layers) {
        for (auto& [param, grad] : layer->parameters()) {
            // Simple SGD (peut être Adam, etc.)
            param = param - lr * grad;
            
            // Clear gradients pour prochaine itération
            grad.zero_();
        }
    }
}
```

---

## Gestion de la Mémoire

### Architecture Mémoire (réelle)

```text
┌──────────────────────────────────────────────────────┐
│              MEMORY MANAGEMENT STACK                 │
├──────────────────────────────────────────────────────┤
│                                                       │
│  Niveau 1: MemoryGuard (limite stricte, blocage)     │
│  ┌─────────────────────────────────────────────────┐ │
│  │ Compteur global │ refuse si dépassement         │ │
│  │ modes block/freeze │ stats current/peak         │ │
│  └─────────────────────────────────────────────────┘ │
│                      ↓                                │
│  Niveau 2: AdvancedRAMManager (cache compressé LZ4)  │
│  ┌─────────────────────────────────────────────────┐ │
│  │ stockage en RAM compressée │ eviction LRU       │ │
│  └─────────────────────────────────────────────────┘ │
│                      ↓                                │
│  Niveau 3: DynamicTensorAllocator (handles + lazy)   │
│  ┌─────────────────────────────────────────────────┐ │
│  │ réserve via MemoryGuard │ malloc/free interne   │ │
│  │ compression via AdvancedRAMManager              │ │
│  └─────────────────────────────────────────────────┘ │
│                      ↓                                │
│  Niveau 4: RuntimeAllocator (RAII pour intermédiaires)│
│  ┌─────────────────────────────────────────────────┐ │
│  │ allocate_tensor/allocate_buffer                 │ │
│  │ auto-release + tags + cap local optionnel       │ │
│  └─────────────────────────────────────────────────┘ │
│                      ↓                                │
│  Niveau 5: Allocations C++ (new/malloc)              │
│  ┌─────────────────────────────────────────────────┐ │
│  │ std::vector / new[] / malloc                     │ │
│  └─────────────────────────────────────────────────┘ │
│                                                       │
└──────────────────────────────────────────────────────┘
```

### Flux d'Allocation (deux chemins)

Le runtime a **deux usages mémoire** principaux :

1. **Tenseurs/poids long-vivants** (poids, caches) : stockés dans le modèle, souvent via `tensor` + `DynamicTensorAllocator`.
2. **Intermédiaires court-vivants** (activations temporaires, buffers) : alloués via `RuntimeAllocator` dans certains chemins du forward.

Exemple réaliste : le `forwardPass()` peut créer un `RuntimeAllocator` par layer et allouer un output temporaire taggé, automatiquement libéré en fin de scope.

```cpp
MemoryGuard& guard = MemoryGuard::instance();
RuntimeAllocator allocator(guard, /*cap_mb*/ 4096);

auto out = allocator.allocate_tensor({static_cast<int>(outN)}, "float32", "embedding_output");
std::vector<float>& data = out.data();
// ... compute ...
// Auto-release: le destructeur décrémente MemoryGuard.
```

### Lazy Loading (vrai comportement)

La classe `tensor` (voir `src/tensors.cpp`) supporte un mode `dynamic=true`.
Dans ce mode, l'objet crée un **handle** via `DynamicTensorAllocator` mais ne réserve/alloue la RAM réelle qu'au premier accès à `getData()`.

```cpp
// tensor::tensor(size, dynamic=true)
dynamic_handle = DynamicTensorAllocator::instance().allocateTensor(size, "tensor_data");

// Au premier getData(): réservation MemoryGuard + malloc interne (lazy)
float* p = DynamicTensorAllocator::instance().getTensorData(handle);
```

Quand la compression est activée, `DynamicTensorAllocator` peut compresser les données via `AdvancedRAMManager` (LZ4), puis libérer la RAM active.
Il n'y a **pas** d'offload disque dans l'implémentation actuelle (le cache est en RAM compressée).

---

## Bindings Lua

### Mécanisme de Binding

#### Côté C++ : Enregistrement

```cpp
// Enregistrer une fonction Lua (principe)
lua_pushcfunction(L, lua_myFunction);
lua_setglobal(L, "my_function");

// Fonction callback
static int lua_myFunction(lua_State* L) {
    // 1. Extraire arguments
    int arg1 = lua_tointeger(L, 1);
    const char* arg2 = lua_tostring(L, 2);
    
    // 2. Traitement C++
    int result = do_something(arg1, arg2);
    
    // 3. Pousser résultat
    lua_pushinteger(L, result);
    return 1;  // Nombre de valeurs de retour
}
```

Dans Mímir, l'API est structurée sous `Mimir.*` (ex: `Mimir.Model.train`).
Des aliases globaux existent pour la compatibilité (`model`, `dataset`, `tokenizer`, `Allocator`, `MemoryGuard`, ...).

#### Côté Lua : Appel

```lua
local result = my_function(42, "hello")
print(result)
```

### Conversion de Types

#### Lua → C++

```cpp
// Table Lua → json C++
json LuaScripting::luaTableToJson(lua_State* L, int index) {
    json j;
    
    lua_pushnil(L);  // Premier élément
    while (lua_next(L, index) != 0) {
        // Key à index -2, value à index -1
        
        std::string key;
        if (lua_type(L, -2) == LUA_TSTRING) {
            key = lua_tostring(L, -2);
        } else {
            key = std::to_string(lua_tointeger(L, -2));
        }
        
        // Convertir value selon son type
        if (lua_type(L, -1) == LUA_TNUMBER) {
            j[key] = lua_tonumber(L, -1);
        } else if (lua_type(L, -1) == LUA_TSTRING) {
            j[key] = lua_tostring(L, -1);
        } else if (lua_type(L, -1) == LUA_TTABLE) {
            j[key] = luaTableToJson(L, -1);  // Récursif
        }
        // ...
        
        lua_pop(L, 1);  // Enlever value, garder key
    }
    
    return j;
}
```

#### C++ → Lua

```cpp
// json C++ → Table Lua
void LuaScripting::jsonToLuaTable(lua_State* L, const json& j) {
    lua_newtable(L);  // Créer nouvelle table
    
    for (auto it = j.begin(); it != j.end(); ++it) {
        // Pousser clé
        lua_pushstring(L, it.key().c_str());
        
        // Pousser valeur selon type
        if (it.value().is_number()) {
            lua_pushnumber(L, it.value().get<double>());
        } else if (it.value().is_string()) {
            lua_pushstring(L, it.value().get<std::string>().c_str());
        } else if (it.value().is_object()) {
            jsonToLuaTable(L, it.value());  // Récursif
        }
        // ...
        
        lua_settable(L, -3);  // table[key] = value
    }
}
```

---

## Threading et Async

### Architecture Threading

```text
┌────────────────────────────────────────────┐
│              MAIN THREAD                   │
│  ┌──────────────────────────────────────┐  │
│  │  Lua Runtime + Model Training        │  │
│  │  (CPU intensive, blocks)             │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
              │
              │ Updates via queues
              ↓
┌────────────────────────────────────────────┐
│           MONITOR THREAD                   │
│  ┌──────────────────────────────────────┐  │
│  │  AsyncMonitor (coordonne)            │  │
│  │  - Reçoit métriques via queue        │  │
│  │  - Dispatch vers htop et viz         │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
       │                    │
       │                    │
       ↓                    ↓
┌─────────────┐      ┌─────────────┐
│ HTOP THREAD │      │  VIZ THREAD │
│  Terminal   │      │    SFML     │
│  Rendering  │      │   Window    │
└─────────────┘      └─────────────┘
```

### AsyncMonitor Implementation

```cpp
class AsyncMonitor {
    std::thread monitor_thread;
    std::thread htop_thread;
    std::thread viz_thread;
    
    std::queue<Metrics> metrics_queue;
    std::mutex queue_mutex;
    std::condition_variable cv;
    
    std::atomic<bool> running{true};
    
public:
    void start() {
        // Lancer thread monitor
        monitor_thread = std::thread([this]() {
            while (running) {
                // Attendre nouvelles métriques
                std::unique_lock<std::mutex> lock(queue_mutex);
                cv.wait(lock, [this] { 
                    return !metrics_queue.empty() || !running; 
                });
                
                if (!metrics_queue.empty()) {
                    Metrics m = metrics_queue.front();
                    metrics_queue.pop();
                    lock.unlock();
                    
                    // Dispatcher vers htop et viz
                    if (htop_enabled) htop->update(m);
                    if (viz_enabled) viz->update(m);
                }
            }
        });
        
        // Lancer threads htop et viz
        startHtopThread();
        startVizThread();
    }
    
    void updateMetrics(const Metrics& m) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        metrics_queue.push(m);
        cv.notify_one();
    }
};
```

### Thread Safety

**Zones critiques protégées :**

- ✅ `metrics_queue` : mutex + condition variable
- ✅ Allocateur de tenseurs : mutex global
- ✅ Stats mémoire : atomic counters
- ✅ Logs : mutex pour std::cout

**Zones sans protection (safe):**

- ✅ Calculs de tenseurs (pas de partage entre threads)
- ✅ Forward/backward (single-threaded dans main)
- ✅ Rendering htop/viz (threads dédiés, pas d'écriture partagée)

---

## Optimisations Runtime

### 1. Hardware Detection au Démarrage

```cpp
// HardwareOpt.hpp
struct HardwareCaps {
    bool avx2 = false;
    bool fma = false;
    bool f16c = false;
    bool bmi2 = false;
    
    static HardwareCaps detect() {
        HardwareCaps caps;
        
        #ifdef __x86_64__
        __builtin_cpu_init();
        caps.avx2 = __builtin_cpu_supports("avx2");
        caps.fma = __builtin_cpu_supports("fma");
        caps.f16c = __builtin_cpu_supports("f16c");
        caps.bmi2 = __builtin_cpu_supports("bmi2");
        #endif
        
        return caps;
    }
};

// Détection globale au démarrage
static const HardwareCaps g_hw_caps = HardwareCaps::detect();
```

### 2. Dispatch Dynamique

```cpp
Tensor computeLinear(const Tensor& input, const Tensor& weight) {
    if (g_hw_caps.avx2 && g_hw_caps.fma) {
        return computeLinear_AVX2_FMA(input, weight);  // Fastest
    } else if (g_hw_caps.avx2) {
        return computeLinear_AVX2(input, weight);      // Fast
    } else {
        return computeLinear_scalar(input, weight);    // Fallback
    }
}
```

### 3. SIMD Intrinsics

```cpp
// Matmul optimisé AVX2 + FMA
void matmul_avx2_fma(const float* A, const float* B, float* C,
                     int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; j += 8) {  // Process 8 floats
            __m256 sum = _mm256_setzero_ps();
            
            for (int k = 0; k < K; ++k) {
                __m256 a = _mm256_broadcast_ss(&A[i * K + k]);
                __m256 b = _mm256_loadu_ps(&B[k * N + j]);
                sum = _mm256_fmadd_ps(a, b, sum);  // FMA!
            }
            
            _mm256_storeu_ps(&C[i * N + j], sum);
        }
    }
}
```

### 4. Memory Pooling

```cpp
class TensorPool {
    std::vector<float*> free_blocks;
    std::unordered_map<size_t, std::vector<float*>> size_pools;
    
public:
    float* allocate(size_t size) {
        auto& pool = size_pools[size];
        if (!pool.empty()) {
            float* ptr = pool.back();
            pool.pop_back();
            return ptr;  // Réutiliser un block existant
        }
        return new float[size];  // Allouer nouveau
    }
    
    void deallocate(float* ptr, size_t size) {
        size_pools[size].push_back(ptr);  // Garder pour réutilisation
    }
};
```

---

## Diagrammes de Flux

### Flux Complet: Script Lua → Résultat

```text
┌───────────────┐
│  Script Lua   │ model.create("transformer", {...})
└───────┬───────┘
        │
        ↓
┌───────────────┐
│  Lua Runtime  │ Parse + Execute
└───────┬───────┘
        │
        ↓
┌───────────────┐
│lua_createModel│ Callback C++
└───────┬───────┘
        │
        ↓
┌───────────────┐
│  LuaContext   │ Store config
└───────┬───────┘
        │
        ↓
┌───────────────┐
│ model.build() │ Lua appelle
└───────┬───────┘
        │
        ↓
┌───────────────┐
│lua_buildModel │ Callback C++
└───────┬───────┘
        │
        ↓
┌───────────────┐
│ Model::build()│ Construire architecture
└───────┬───────┘
        │
        ↓
┌───────────────┐
│Allocate Params│ Tenseurs + Layers
└───────┬───────┘
        │
        ↓
┌───────────────┐
│Initialize Weights│ Xavier/He/...
└───────┬───────┘
        │
        ↓
┌───────────────┐
│ Return to Lua │ Push true + params count
└───────────────┘
```

### Flux Forward Pass

```text
Input IDs [seq_len]
    │
    ↓
Embedding Lookup
    │ [seq_len, embed_dim]
    ↓
Positional Encoding
    │ [seq_len, embed_dim]
    ↓
┌─────────────────┐
│ Transformer #1  │
│  - Attention    │ ← Hardware accelerated (AVX2)
│  - FFN          │ ← FMA optimized
│  - LayerNorm    │
└─────────────────┘
    │ [seq_len, embed_dim]
    ↓
┌─────────────────┐
│ Transformer #2  │
│  - Attention    │
│  - FFN          │
│  - LayerNorm    │
└─────────────────┘
    │ [seq_len, embed_dim]
    ↓
   ...
    │
    ↓
Output Projection
    │ [seq_len, vocab_size]
    ↓
Softmax
    │ [seq_len, vocab_size]
    ↓
Logits (output)
```

### Flux Backward Pass

```text
Loss Gradient
    │
    ↓
Backprop Output Projection
    │
    ↓
┌─────────────────┐
│ Transformer #N  │ (reverse order)
│  - LayerNorm    │
│  - FFN          │
│  - Attention    │
└─────────────────┘
    │
    ↓
   ...
    │
    ↓
┌─────────────────┐
│ Transformer #1  │
│  - LayerNorm    │
│  - FFN          │
│  - Attention    │
└─────────────────┘
    │
    ↓
Backprop Positional Encoding
    │
    ↓
Backprop Embedding
    │
    ↓
Gradients stored in layers
```

---

## 🎯 Points Clés à Retenir

### Performance

1. **Hardware detection** une seule fois au démarrage
2. **Dispatch dynamique** pour chaque opération
3. **SIMD intrinsics** (AVX2/FMA) pour les hot paths
4. **Memory pooling** pour réduire les allocations
5. **Lazy loading** pour économiser la RAM

### Mémoire

1. **4 composants** : MemoryGuard (limite) → AdvancedRAMManager (cache) → DynamicTensorAllocator (lazy/compression) → RuntimeAllocator (RAII)
2. **LRU eviction** automatique
3. **LZ4 compression** ~50% d'économie
4. **Cap local optionnel** côté `RuntimeAllocator` (sans modifier la limite globale)

### Threading

1. **Main thread** : training (CPU intensive)
2. **Monitor thread** : coordination
3. **Htop/Viz threads** : rendering non-bloquant
4. **Thread-safe** : queues + mutex sur zones critiques

### Lua Integration

1. **Singleton context** pour partager données
2. **Static callbacks** pour fonctions Lua
3. **Conversion automatique** Lua tables ↔ C++ json
4. **Error handling** robuste avec try/catch

---

## 📚 Voir Aussi

- [System Architecture](01-System-Architecture.md)
- [Technical Specifications](03-Technical-Specifications.md)
- [Threading And Compute](05-Threading-And-Compute.md)
- [Hardware Optimizations](07-Hardware-Optimizations.md)
- [API Reference](../03-API-Reference/00-API-Complete.md)

---

© 2025 Mímir Framework - Documentation Internals
