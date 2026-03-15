# Guide Technique : Ajouter un Nouveau Layer

Ce guide explique comment ajouter un nouveau type de layer au système unifié.

---

## 📋 Étapes Rapides

### 1. Ajouter l'Enum (LayerTypes.hpp)

```cpp
// Dans enum class LayerType (ligne ~50)
enum class LayerType {
    // ... existing types ...
    
    MyNewLayer,  // ← Ajouter ici
    
    // ...
};
```

### 2. Ajouter le Mapping String (LayerTypes.hpp)

```cpp
// Dans string_to_type() (ligne ~150)
inline LayerType string_to_type(const std::string& type_str) {
    // ... existing mappings ...
    
    if (norm == "mynewlayer") return LayerType::MyNewLayer;
    
    // ...
}

// Dans type_to_string() (ligne ~250)
inline std::string type_to_string(LayerType type) {
    switch (type) {
        // ... existing cases ...
        
        case LayerType::MyNewLayer: return "MyNewLayer";
        
        // ...
    }
}
```

### 3. Étendre Layer Struct (Layers.hpp)

```cpp
struct Layer {
    // ... existing fields ...
    
    // Paramètres spécifiques à MyNewLayer
    int my_param1 = 0;
    float my_param2 = 1.0f;
    bool my_flag = false;
    
    // ...
};
```

### 4. Implémenter Forward Pass (LayerOps.hpp)

```cpp
namespace LayerOps {

// ... existing functions ...

/**
 * @brief MyNewLayer forward pass
 * @param input Input tensor (flat vector)
 * @param params Layer parameters (weights, biases)
 * @param param1 Custom parameter 1
 * @param param2 Custom parameter 2
 * @return Output tensor
 */
inline std::vector<float> mynewlayer_forward(
    const std::vector<float>& input,
    const std::vector<float>& params,
    int param1,
    float param2
) {
    std::vector<float> output(input.size());
    
    // Votre implémentation ici
    #pragma omp parallel for if (input.size() > 10000)
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = compute_mynewlayer(input[i], params, param1, param2);
    }
    
    return output;
}

} // namespace LayerOps
```

### 5. Ajouter le Case dans Model.cpp

```cpp
// Dans forwardPass() (ligne ~1850)
switch (layer.type_enum) {
    // ... existing cases ...
    
    case LayerType::MyNewLayer: {
        try {
            output = LayerOps::mynewlayer_forward(
                input,
                params,
                layer.my_param1,
                layer.my_param2
            );
        } catch (const std::exception& e) {
            throw std::runtime_error(
                std::string("MyNewLayer: ") + e.what()
            );
        }
        break;
    }
    
    // ...
}
```

---

## 🔧 Template Complet

### Exemple: ConvDepthwise Layer

#### 1. Enum (LayerTypes.hpp)
```cpp
enum class LayerType {
    // ...
    ConvDepthwise,
    // ...
};
```

#### 2. String Mapping (LayerTypes.hpp)
```cpp
inline LayerType string_to_type(const std::string& type_str) {
    std::string norm = type_str;
    std::transform(norm.begin(), norm.end(), norm.begin(), ::tolower);
    
    // Aliases
    if (norm == "dwconv" || norm == "depthwiseconv") {
        return LayerType::ConvDepthwise;
    }
    if (norm == "convdepthwise") return LayerType::ConvDepthwise;
    
    // ...
}

inline std::string type_to_string(LayerType type) {
    switch (type) {
        case LayerType::ConvDepthwise: return "ConvDepthwise";
        // ...
    }
}
```

#### 3. Layer Parameters (Layers.hpp)
```cpp
struct Layer {
    // Paramètres ConvDepthwise (peuvent réutiliser existants)
    // kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w
    // déjà définis pour Conv2d
    
    int depth_multiplier = 1;  // Nouveau paramètre spécifique
};
```

#### 4. Forward Implementation (LayerOps.hpp)
```cpp
namespace LayerOps {

inline std::vector<float> convdepthwise_forward(
    const std::vector<float>& input,
    const std::vector<float>& params,
    int in_channels,
    int height, int width,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int depth_multiplier = 1
) {
    // Compute output dimensions
    int out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int out_channels = in_channels * depth_multiplier;
    
    std::vector<float> output(out_channels * out_h * out_w, 0.0f);
    
    // Depthwise convolution: each input channel has its own kernel
    #pragma omp parallel for collapse(3)
    for (int c = 0; c < in_channels; ++c) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                for (int m = 0; m < depth_multiplier; ++m) {
                    float sum = 0.0f;
                    
                    // Convolve kernel with input
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int ih = oh * stride_h + kh - pad_h;
                            int iw = ow * stride_w + kw - pad_w;
                            
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                int input_idx = c * height * width + ih * width + iw;
                                int kernel_idx = (c * depth_multiplier + m) * kernel_h * kernel_w + kh * kernel_w + kw;
                                
                                sum += input[input_idx] * params[kernel_idx];
                            }
                        }
                    }
                    
                    int out_c = c * depth_multiplier + m;
                    int output_idx = out_c * out_h * out_w + oh * out_w + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
    
    return output;
}

} // namespace LayerOps
```

#### 5. Dispatch (Model.cpp)
```cpp
case LayerType::ConvDepthwise: {
    try {
        // Récupérer paramètres
        int kh = layer.get_kernel_h();
        int kw = layer.get_kernel_w();
        int sh = layer.get_stride_h();
        int sw = layer.get_stride_w();
        int ph = layer.get_pad_h();
        int pw = layer.get_pad_w();
        int dm = layer.depth_multiplier;
        
        // Valider
        if (kh <= 0 || kw <= 0) {
            throw std::runtime_error("ConvDepthwise: invalid kernel size");
        }
        
        // Forward pass
        output = LayerOps::convdepthwise_forward(
            input, params,
            layer.in_channels, layer.height, layer.width,
            kh, kw, sh, sw, ph, pw, dm
        );
        
        // Update dimensions
        layer.out_channels = layer.in_channels * dm;
        layer.out_h = (layer.height + 2 * ph - kh) / sh + 1;
        layer.out_w = (layer.width + 2 * pw - kw) / sw + 1;
        
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Layer " + std::to_string(layer_idx) + 
            " (" + layer.name + ", type: ConvDepthwise): " + 
            e.what()
        );
    }
    break;
}
```

---

## 🚀 Optimisations Recommandées

### 1. OpenMP Parallelization
```cpp
// Pour les boucles intensives
#pragma omp parallel for if (size > threshold)
for (size_t i = 0; i < size; ++i) {
    // compute
}

// Pour les boucles imbriquées
#pragma omp parallel for collapse(2)
for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
        // compute
    }
}
```

### 2. AVX2 Vectorization
```cpp
#include <immintrin.h>

// Traiter 8 floats à la fois
for (size_t i = 0; i + 8 <= size; i += 8) {
    __m256 a = _mm256_loadu_ps(&input[i]);
    __m256 b = _mm256_loadu_ps(&other[i]);
    __m256 result = _mm256_mul_ps(a, b);  // multiply
    _mm256_storeu_ps(&output[i], result);
}

// Traiter les éléments restants
for (size_t i = (size / 8) * 8; i < size; ++i) {
    output[i] = input[i] * other[i];
}
```

### 3. Cache Locality
```cpp
// Mauvais: accès mémoire non continu
for (int c = 0; c < C; ++c) {
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            // data[c][h][w] - sauts de mémoire
        }
    }
}

// Bon: accès mémoire continu
for (int c = 0; c < C; ++c) {
    for (int h = 0; h < H; ++h) {
        // Traiter toute la ligne d'un coup
        for (int w = 0; w < W; ++w) {
            // data[c * H * W + h * W + w] - continu
        }
    }
}
```

---

## ✅ Checklist Avant Commit

- [ ] Enum ajouté dans LayerTypes.hpp
- [ ] String mapping (avec aliases) dans string_to_type()
- [ ] Reverse mapping dans type_to_string()
- [ ] Paramètres ajoutés dans Layer struct
- [ ] Forward pass implémenté dans LayerOps.hpp
- [ ] Case ajouté dans Model.cpp forwardPass()
- [ ] Exception handling avec contexte détaillé
- [ ] Optimisations (OpenMP/AVX2) si applicable
- [ ] Documentation (docstrings, commentaires)
- [ ] Compilation sans warnings
- [ ] Test Lua créé pour validation
- [ ] Test exécuté et passé

---

## 🧪 Template de Test

```lua
-- test_mynewlayer.lua

log("\n╔════════════════════════════════════════╗")
log("║   Test MyNewLayer                      ║")
log("╚════════════════════════════════════════╝\n")

-- Configuration
local config = {
    my_param1 = 10,
    my_param2 = 0.5
}

-- Créer modèle
local success, err = model.create("test_mynewlayer", config)
if not success then
    log("❌ Échec création: " .. (err or ""))
    os.exit(1)
end

-- Ajouter layer
model.push_layer("layer1", "MyNewLayer", params_count)

-- Allouer et initialiser
success, num_params = model.allocate_params()
if not success then
    log("❌ Échec allocation")
    os.exit(1)
end

success = model.init_weights("xavier", 42)
if not success then
    log("❌ Échec init")
    os.exit(1)
end

-- Forward pass
local input = {1, 2, 3, 4, 5}
local output = model.forward(input)

if output and #output > 0 then
    log(string.format("✓ Output: %d valeurs", #output))
    log(string.format("  First 5: [%.2f, %.2f, %.2f, %.2f, %.2f]",
        output[1], output[2], output[3], output[4], output[5]))
else
    log("❌ Forward pass échoué")
    os.exit(1)
end

log("\n✅ Test MyNewLayer réussi!\n")
```

---

## 📚 Ressources

### Documentation Interne
- `src/LayerTypes.hpp` - Tous les types supportés
- `src/LayerOps.hpp` - Implémentations de référence
- `src/Model.cpp` - Dispatch et intégration

### Exemples de Référence
- `LayerOps::linear_forward()` - GEMM avec AVX2
- `LayerOps::layernorm_forward()` - Normalisation complète
- `LayerOps::gelu_forward()` - Activation non-linéaire
- `Model.cpp` case Conv2d - Convolution 2D complexe

### Tests
- `scripts/tests/test_clean_system.lua` - Test complet du système
- `scripts/tests/test_unified_layers.lua` - Tests individuels

---

**Signature:** Mímir Framework Dev Team  
**Version:** v2.1.0  
**Last Update:** 2025-01
