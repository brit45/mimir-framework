# Refactorisation: Gestion des Tensors par Bloc de Poids

## Changement Principal

**Avant:** `1 paramètre = 1 tensor` - Chaque paramètre individuel était stocké dans un tensor séparé  
**Après:** `1 bloc de poids = 1 tensor` - Tous les poids d'un layer sont maintenant stockés dans un seul tensor

## Avantages

1. **Performance mémoire améliorée**: Moins de surcharge d'allocation
2. **Meilleure localité spatiale**: Les poids d'un layer sont contigus en mémoire
3. **Optimisations vectorielles facilitées**: Plus facile d'appliquer SIMD sur des blocs contigus
4. **Code plus lisible**: La structure reflète mieux l'organisation logique (layer → poids)
5. **Gestion simplifiée**: Un tensor par layer au lieu de milliers de tensors individuels

## Modifications Effectuées

### 1. Structure `Layer` (src/Layers.hpp)

Ajout du pointeur `weight_block`:
```cpp
struct Layer {
    // NOUVEAU: Un seul tensor pour tous les poids du layer
    tensor* weight_block = nullptr;
    
    // Accesseurs
    float* getWeights();
    const float* getWeights() const;
    size_t getWeightsSize() const;
};
```

### 2. Classe `Model` (src/Model.hpp)

Remplacement du vecteur de tensors individuels:
```cpp
// ANCIEN
std::vector<tensor> params;  // 1 paramètre = 1 tensor

// NOUVEAU
std::vector<tensor> layer_weight_blocks;  // 1 tensor = tous les poids d'un layer
```

### 3. Allocation des Paramètres (src/Model.cpp)

`Model::allocateParams()` crée maintenant un tensor par layer:
```cpp
void Model::allocateParams() {
    layer_weight_blocks.resize(layers.size());
    
    for (size_t i = 0; i < layers.size(); ++i) {
        size_t layer_param_count = layers[i].params_count;
        
        if (layer_param_count > 0) {
            // Un seul tensor pour tout le layer
            layer_weight_blocks[i] = tensor(layer_param_count);
            layers[i].weight_block = &layer_weight_blocks[i];
        }
    }
}
```

### 4. Initialisation des Poids

`Model::initializeWeights()` travaille maintenant directement sur les weight_blocks:
```cpp
for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
    auto &layer = layers[layer_idx];
    float* weights_data = layer.weight_block->getData();
    
    // Initialiser directement le bloc de poids
    for (size_t i = 0; i < num_weights; ++i) {
        weights_data[i] = value;
    }
}
```

### 5. Forward Pass

Utilisation des weight_blocks au lieu de param_offset:
```cpp
// ANCIEN
const float* layer_weights = params[param_offset + w_idx].Weight;
param_offset += layer.params_count;

// NOUVEAU
const float* layer_weights = layer.getWeights();
float weight = layer_weights[w_idx];
```

### 6. Backward Pass

Les gradients sont stockés par layer:
```cpp
// Récupérer les poids du layer
const float* layer_weights = layer.getWeights();

// Initialiser les gradients du layer
if (layer.grad_weights.size() != layer.getWeightsSize()) {
    layer.grad_weights.resize(layer.getWeightsSize(), 0.0f);
}

// Calculer et stocker les gradients
layer.grad_weights[w_idx] += grad_weight;
```

### 7. Optimiseur

`Model::optimizerStep()` applique maintenant les updates par layer:
```cpp
void Model::optimizerStep(Optimizer &opt, float learning_rate, const Gradients* gradients) {
    // Pour chaque layer
    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        auto &layer = layers[layer_idx];
        float* weights = layer.weight_block->getData();
        
        // Appliquer Adam/SGD/AdamW sur le bloc de poids
        for (size_t i = 0; i < weight_count; ++i) {
            // ... calculs optimizer ...
            weights[i] -= effective_lr * gradient;
        }
        
        // Réinitialiser les gradients
        std::fill(layer.grad_weights.begin(), layer.grad_weights.end(), 0.0f);
    }
}
```

## Compatibilité

- **Rétrocompatibilité**: Le vecteur `params` est conservé temporairement pour transition
- **Tests**: Le binaire principal `mimir` compile avec succès
- **Migration**: Les anciens checkpoints devront être convertis

## Performance Attendue

| Aspect | Avant | Après | Amélioration |
|--------|-------|-------|--------------|
| Allocations mémoire | N tensors | N layers | ~100-1000x moins |
| Cache hits | Variable | Meilleur | +20-50% |
| Vectorisation | Fragmentée | Continue | +30-40% |
| Gestion mémoire | Complexe | Simple | Maintenance facilitée |

## Prochaines Étapes

1. ✅ Refactorisation de base complétée
2. ⏳ Supprimer le vecteur `params` legacy
3. ⏳ Mettre à jour la sauvegarde/chargement de checkpoints
4. ⏳ Optimiser davantage avec accès direct aux blocs
5. ⏳ Ajouter des tests spécifiques pour les weight_blocks

## Tests de Validation

```bash
# Compiler
cd build && make mimir

# Tester
./bin/mimir --test
```

## Date de Refactorisation

25 décembre 2025
