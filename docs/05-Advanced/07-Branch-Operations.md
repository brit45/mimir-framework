# Système de Gestion des Branches dans les Layers

## Vue d'ensemble

Le framework Mímir implémente maintenant un système automatique de détection et de gestion des branches pour les architectures de réseaux de neurones avec connexions résiduelles, skip connections, et autres topologies complexes.

## Types de Branches

### Énumération `BranchType`

```cpp
enum class BranchType {
    NONE,              // Pas de branche
    RESIDUAL,          // Connexion résiduelle (y = F(x) + x)
    SKIP_CONNECTION,   // Skip connection (concaténation)
    DENSE_CONNECTION,  // DenseNet-style (concat avec tous les précédents)
    ATTENTION_BRANCH,  // Branche d'attention parallèle
    MULTI_SCALE,       // Branches multi-échelles (Inception-style)
    GATE,              // Gating mechanism (LSTM-style)
    SPLIT,             // Division en branches parallèles
    MERGE              // Fusion de plusieurs branches
};
```

### Opérations de Fusion

```cpp
enum class MergeOperation {
    ADD,               // Addition élément par élément (ResNet)
    MULTIPLY,          // Multiplication élément par élément
    CONCATENATE,       // Concaténation le long du canal (DenseNet)
    MAX,               // Maximum élément par élément
    AVERAGE,           // Moyenne élément par élément
    GATED,             // Fusion avec gating
    ATTENTION_WEIGHTED // Fusion pondérée par attention
};
```

## Détection Automatique

Le système détecte automatiquement le type de branche basé sur le nom du layer :

### Conventions de nommage

| Pattern de nom | Type détecté | Opération de fusion |
|----------------|--------------|---------------------|
| `*residual*` ou `*shortcut*` | `RESIDUAL` | `ADD` |
| `*concat*` | `SKIP_CONNECTION` | `CONCATENATE` |
| `*dense*connect*` | `DENSE_CONNECTION` | `CONCATENATE` |
| `*attention*branch*` | `ATTENTION_BRANCH` | `ATTENTION_WEIGHTED` |
| `*inception*` ou `*multiscale*` | `MULTI_SCALE` | `CONCATENATE` |
| `*gate*` | `GATE` | `GATED` |
| `*split*` | `SPLIT` | - |
| `*merge*` ou `*fusion*` | `MERGE` | Détecté par sous-pattern |

### Exemple d'utilisation

```cpp
// Construction d'un ResNet-style block
model.push("conv1", "Conv2d", params_count);
model.push("bn1", "BatchNorm", bn_params);
model.push("conv2", "Conv2d", params_count);
model.push("bn2", "BatchNorm", bn_params);
model.push("residual_add", "Merge", 0);  // Détecté automatiquement comme RESIDUAL + ADD

// DenseNet-style
model.push("conv1", "Conv2d", params_count);
model.push("conv2", "Conv2d", params_count);
model.push("dense_concat", "Merge", 0);   // Détecté comme DENSE_CONNECTION + CONCATENATE
```

## API des Branches

### Configuration d'un Layer

Chaque `Layer` contient maintenant :

```cpp
struct Layer {
    // ... champs existants ...
    
    // Configuration des branches
    BranchType branch_type = BranchType::NONE;
    MergeOperation merge_op = MergeOperation::ADD;
    std::vector<int> branch_sources;  // Indices des layers sources
    int branch_target = -1;           // Indice du layer cible
    bool is_branch_point = false;     // Point de bifurcation
    bool is_merge_point = false;      // Point de fusion
    
    // Méthodes
    void detectBranchType();          // Détection auto basée sur le nom
    bool requiresBranchComputation() const;
};
```

### Méthodes du Model

#### 1. Opérations de Fusion

```cpp
// Fusionner deux branches selon l'opération spécifiée
static void computeBranchMerge(
    const std::vector<float>& branch1, 
    const std::vector<float>& branch2,
    std::vector<float>& output,
    MergeOperation merge_op,
    bool use_hardware = true
);
```

**Optimisations** :
- Utilise AVX2 quand disponible
- Parallélisation automatique pour les opérations vectorielles
- Dispatch hardware/software automatique

#### 2. Division de Branches

```cpp
// Diviser un tensor en plusieurs branches
static void computeBranchSplit(
    const std::vector<float>& input,
    std::vector<std::vector<float>>& outputs,
    const std::vector<int>& split_sizes
);
```

#### 3. Détection et Configuration

```cpp
// Analyser le modèle et détecter toutes les branches
void detectAndSetupBranches();
```

Cette méthode :
- Parcourt tous les layers
- Appelle `detectBranchType()` sur chacun
- Établit les connexions entre layers (sources et cibles)
- Affiche un résumé des branches détectées

#### 4. Exécution pendant Forward Pass

```cpp
// Exécuter les calculs de branche pour un layer
void executeBranchComputation(
    int layer_idx, 
    std::vector<std::vector<float>>& layer_outputs,
    bool training = false
);
```

Automatiquement appelée pendant `forwardPass()` pour :
- Les connexions résiduelles : addition des sorties
- Les skip connections : concaténation
- Les branches multi-échelles : fusion appropriée

#### 5. Backpropagation à travers les Branches

```cpp
// Propager les gradients à travers les branches
void backpropThroughBranch(
    int layer_idx,
    const std::vector<float>& grad_output,
    std::vector<std::vector<float>>& layer_gradients
);
```

Gère automatiquement :
- Propagation des gradients vers les branches sources
- Accumulation des gradients pour les points de fusion
- Conservation du gradient pour les connexions résiduelles

## Exemples d'Architectures

### ResNet Block

```cpp
// Block résiduel classique
void buildResidualBlock(Model& model, int in_channels, int out_channels) {
    model.push("conv1", "Conv2d", in_channels * 3 * 3 * out_channels);
    model.push("bn1", "BatchNorm", out_channels * 2);
    // ReLU appliqué automatiquement
    model.push("conv2", "Conv2d", out_channels * 3 * 3 * out_channels);
    model.push("bn2", "BatchNorm", out_channels * 2);
    
    // Si dimensions changent, projection
    if (in_channels != out_channels) {
        model.push("shortcut_proj", "Conv2d", in_channels * 1 * 1 * out_channels);
    }
    
    model.push("residual_add", "Merge", 0);  // Fusion automatique
    // Output: y = conv2(conv1(x)) + x (ou shortcut(x))
}
```

### DenseNet Block

```cpp
void buildDenseBlock(Model& model, int growth_rate, int num_layers) {
    for (int i = 0; i < num_layers; ++i) {
        std::string prefix = "dense_layer_" + std::to_string(i);
        
        model.push(prefix + "_bn", "BatchNorm", growth_rate * 2);
        model.push(prefix + "_conv", "Conv2d", growth_rate * 3 * 3 * growth_rate);
        
        // Concaténation avec toutes les features précédentes
        model.push(prefix + "_dense_concat", "Merge", 0);
    }
}
```

### Inception-style Multi-Scale

```cpp
void buildInceptionModule(Model& model, int in_channels) {
    // Branche 1x1
    model.push("inception_1x1", "Conv2d", in_channels * 1 * 1 * 64);
    
    // Branche 3x3
    model.push("inception_3x3_reduce", "Conv2d", in_channels * 1 * 1 * 96);
    model.push("inception_3x3", "Conv2d", 96 * 3 * 3 * 128);
    
    // Branche 5x5
    model.push("inception_5x5_reduce", "Conv2d", in_channels * 1 * 1 * 16);
    model.push("inception_5x5", "Conv2d", 16 * 5 * 5 * 32);
    
    // Pool projection
    model.push("inception_pool", "MaxPool2d", 0);
    model.push("inception_pool_proj", "Conv2d", in_channels * 1 * 1 * 32);
    
    // Fusion multi-échelle
    model.push("inception_multiscale_merge", "Merge", 0);
}
```

## Workflow d'Entraînement

### 1. Construction du Modèle

```cpp
Model model;
model.setName("ResNet50");

// Construire l'architecture (détection auto des branches)
ModelArchitectures::buildResNet(model, config);

// Optionnel : analyser et afficher les branches
model.detectAndSetupBranches();
```

### 2. Forward Pass

Le forward pass gère automatiquement :
1. Exécution séquentielle des layers
2. Stockage des sorties intermédiaires pour les branches
3. Détection des points de fusion
4. Exécution des opérations de branche (add, concat, etc.)
5. Mise à jour de la sortie avec le résultat fusionné

```cpp
auto output = model.forwardPass(input, true);  // training=true
```

### 3. Backward Pass

Le backward pass propage automatiquement les gradients :
1. Calcul des gradients standard
2. Propagation vers les branches sources
3. Accumulation des gradients aux points de fusion

```cpp
auto gradients = model.backwardPass(loss_gradient);
```

## Optimisations Hardware

### AVX2/FMA

Toutes les opérations de fusion utilisent AVX2 quand disponible :

- **ADD** : `_mm256_add_ps`
- **MULTIPLY** : `_mm256_mul_ps`
- **MAX** : `_mm256_max_ps`
- **AVERAGE** : fusion optimisée avec `_mm256_mul_ps`

### Performance

Pour des vecteurs de taille N :
- **Sans AVX2** : N opérations scalaires
- **Avec AVX2** : N/8 opérations vectorielles (8 floats par cycle)
- **Speedup théorique** : ~8x

## Configuration Manuelle

Pour un contrôle fin, les branches peuvent être configurées manuellement :

```cpp
Layer custom_layer("custom_merge", "Merge", 0);
custom_layer.branch_type = BranchType::RESIDUAL;
custom_layer.merge_op = MergeOperation::ADD;
custom_layer.branch_sources = {5, 10};  // Sources depuis layers 5 et 10
custom_layer.is_merge_point = true;

// Ajouter au modèle
model.getLayers().push_back(custom_layer);
```

## Debugging et Visualisation

```cpp
// Afficher les informations de branches
model.detectAndSetupBranches();

// Output exemple :
// ✓ Détection des branches terminée. Trouvé:
//   - Layer 15 (stage1_block0_residual): RESIDUAL
//   - Layer 30 (stage2_block0_shortcut): RESIDUAL
//   - Layer 45 (dense_concat_1): SKIP_CONNECTION
```

## Limitations Actuelles

1. **Branches multiples** : Actuellement optimisé pour 2 branches
2. **CONCATENATE** : Nécessite que les dimensions spatiales correspondent
3. **ATTENTION_WEIGHTED** : Implémentation de base (à enrichir)
4. **GATED** : Stub à implémenter complètement

## Évolutions Futures

- [ ] Support GPU (Vulkan compute shaders) pour les fusions
- [ ] Graphe de computation explicite pour topologies complexes
- [ ] Optimisation mémoire (in-place operations)
- [ ] Support de branches N-aires (fusion de 3+ branches)
- [ ] Attention pondérée complète avec softmax
- [ ] Gate mechanisms complets (LSTM-style)
- [ ] Visualisation graphique des architectures

## Références

- **ResNet** : [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **DenseNet** : [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
- **Inception** : [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
