# Concepts Fondamentaux

**Version:** 2.3.0  
**API:** Conforme à mimir-api.lua (EmmyLua annotations)  

Guide complet des concepts de base du Mímir Framework.

> 💡 **Syntaxe Recommandée:** Utiliser `Mimir.Module.*` pour bénéficier de l'autocompletion IDE et du type checking.

---

## 📋 Table des Matières

- [Vue d'Ensemble](#vue-densemble)
- [Architecture Générale](#architecture-générale)
- [Modèle et Couches](#modèle-et-couches)
- [Tenseurs](#tenseurs)
- [Autograd](#autograd)
- [Forward et Backward](#forward-et-backward)
- [Optimiseurs](#optimiseurs)
- [Tokenization](#tokenization)
- [Datasets](#datasets)
- [Mémoire](#mémoire)
- [Philosophie CPU-Only](#philosophie-cpu-only)

---

## 🎯 Vue d'Ensemble

### Qu'est-ce que Mímir ?

**Mímir** est un framework de deep learning CPU-only écrit en C++17 avec une API Lua scriptable. Contrairement aux frameworks GPU populaires, Mímir est conçu pour :

- **Apprentissage sur CPU** avec optimisations SIMD (AVX2, FMA)
- **Prototypage rapide** via scripting Lua
- **Déploiement léger** sans dépendances GPU
- **Recherche et éducation** avec transparence du code

### Cas d'Usage Typiques

```lua
-- 1. Prototypage rapide de modèles
local model = Mimir.Model.create("transformer")

-- 2. Expérimentation avec architectures
local custom = Mimir.Architectures.transformer({vocab_size = 10000})

-- 3. Entraînement sur datasets modestes
Mimir.Model.train(model, dataset, 10)

-- 4. Forward pass
local output = Mimir.Model.forward(model, input)
```

---

## 🏗️ Architecture Générale

### Stack Technologique

```
┌─────────────────────────────────────┐
│      Scripts Lua (Utilisateur)      │
│  example_gpt.lua, train_llm.lua...  │
└─────────────────────────────────────┘
              ▼
┌─────────────────────────────────────┐
│          API Lua (94 fonctions)     │
│  model, layers, tokenizer, dataset  │
└─────────────────────────────────────┘
              ▼
┌─────────────────────────────────────┐
│      Runtime C++ (LuaContext)       │
│  Bindings, conversions JSON↔Lua     │
└─────────────────────────────────────┘
              ▼
┌─────────────────────────────────────┐
│     Core Engine C++ (~15K LOC)      │
│  Model, Layers, Tensors, Autograd   │
└─────────────────────────────────────┘
              ▼
┌─────────────────────────────────────┐
│    Optimisations Hardware (SIMD)    │
│      AVX2, FMA, F16C, BMI2          │
└─────────────────────────────────────┘
```

### Flux d'Exécution

1. **Chargement** : Lua charge le script utilisateur
2. **Création** : Construction du modèle (architecture + paramètres)
3. **Forward** : Propagation avant (input → output)
4. **Backward** : Calcul des gradients (autograd)
5. **Update** : Mise à jour des poids (optimizer)
6. **Repeat** : Boucle d'entraînement

---

## 🧩 Modèle et Couches

### Concept de Modèle

Un **modèle** est un conteneur de couches (layers) avec :
- **Paramètres** : Poids, biais (stockés en JSON)
- **Architecture** : Séquence de couches
- **État** : Mode train/eval, gradients

```lua
-- Création d'un modèle avec configuration JSON
local model = Mimir.Model.create("custom")

-- Le modèle est configuré via JSON, pas avec API layers
-- Voir scripts/example_*.lua pour exemples complets

-- Forward pass
local output = Mimir.Model.forward(model, input_data)
```

### Types de Couches

#### 1. **Linear (Fully Connected)**
```lua
layers.addLinear(model, "fc", input_dim, output_dim)
```
- **Opération** : `y = Wx + b`
- **Paramètres** : Matrice `W` (output_dim × input_dim), vecteur `b` (output_dim)
- **Usage** : Transformation linéaire, classification finale

#### 2. **Activation**
```lua
layers.addActivation(model, "relu", "ReLU")
```
- **Types** : ReLU, Sigmoid, Tanh, Softmax, GELU
- **Opération** : Fonction non-linéaire élément par élément
- **Usage** : Introduction de non-linéarité

#### 3. **Dropout**
```lua
layers.addDropout(model, "drop", 0.5)
```
- **Opération** : Mise à zéro aléatoire de neurones (training only)
- **Usage** : Régularisation, prévention overfitting

#### 4. **Normalization**
```lua
layers.addLayerNorm(model, "ln", 512)
layers.addBatchNorm(model, "bn", 128)
```
- **LayerNorm** : Normalisation par couche (Transformers)
- **BatchNorm** : Normalisation par batch (CNNs)
- **Usage** : Stabilisation de l'entraînement

#### 5. **Attention**
```lua
layers.addMultiHeadAttention(model, "mha", 512, 8)
```
- **Opération** : Self-attention multi-têtes
- **Paramètres** : Q, K, V projections
- **Usage** : Transformers, séquences

#### 6. **Convolutional**
```lua
layers.addConv2D(model, "conv", in_channels, out_channels, kernel_size)
```
- **Opération** : Convolution 2D
- **Usage** : Vision, CNNs, détection de features

#### 7. **Embedding**
```lua
layers.addEmbedding(model, "emb", vocab_size, embed_dim)
```
- **Opération** : Lookup table (token_id → vecteur)
- **Usage** : NLP, conversion tokens → vecteurs

---

## 🔢 Tenseurs

### Concept

Un **tenseur** est un tableau multidimensionnel de nombres (floats). Dans Mímir, les tenseurs sont :

```lua
-- Représentation Lua : table imbriquée
local tensor_2d = {
    {1.0, 2.0, 3.0},  -- ligne 1
    {4.0, 5.0, 6.0}   -- ligne 2
}

-- Forme (shape) : [2, 3] = 2 lignes × 3 colonnes
```

### Dimensions Courantes

```lua
-- 1D : Vecteur
local vector = {1.0, 2.0, 3.0}  -- [3]

-- 2D : Matrice
local matrix = {
    {1.0, 2.0},
    {3.0, 4.0}
}  -- [2, 2]

-- 3D : Batch de séquences
local batch = {
    {{1, 2}, {3, 4}},  -- séquence 1
    {{5, 6}, {7, 8}}   -- séquence 2
}  -- [2, 2, 2] = (batch, seq_len, dim)

-- 4D : Images (NCHW)
-- [batch, channels, height, width]
```

### Opérations sur Tenseurs

Dans Mímir, les opérations tensorielles sont gérées **automatiquement** par le moteur C++. L'utilisateur Lua manipule uniquement des structures JSON/tables.

```lua
-- Input : table Lua
local input = {
    {1.0, 2.0, 3.0},
    {4.0, 5.0, 6.0}
}

-- Forward : C++ fait matmul, activations, etc.
local output = Mimir.Model.forward(model, input)

-- Output : table Lua
-- [[0.12, 0.88], [0.34, 0.66]]
```

---

## 🎓 Autograd

### Principe

**Autograd** (Automatic Differentiation) calcule automatiquement les gradients pour la backpropagation.

```
Forward:  x → layer1 → a → layer2 → y
Backward: ∂L/∂x ← layer1 ← ∂L/∂a ← layer2 ← ∂L/∂y
```

### Graphe de Calcul

```lua
-- Forward construit le graphe
local y = model(x)  -- y = f(x, θ)

-- Backward traverse le graphe
loss = criterion(y, target)
backward(loss)  -- calcule ∂L/∂θ

-- Update utilise les gradients
optimizer.step()  -- θ ← θ - lr × ∂L/∂θ
```

### Dans Mímir

L'autograd est **intégré** et **transparent** :

```lua
-- Vous écrivez simplement :
Mimir.Model.train(model, dataset, epochs)

-- Mímir fait automatiquement :
-- 1. Forward → calcul outputs
-- 2. Loss → erreur
-- 3. Backward → calcul gradients (AUTOGRAD)
-- 4. Update → mise à jour poids
```

**Fichiers source** : `src/Autograd.hpp` (graphe computationnel, backprop)

---

## ⚡ Forward et Backward

### Forward Pass

**Propagation avant** : Input → Couches → Output

```lua
local input = {{1, 2, 3}, {4, 5, 6}}  -- [2, 3]
local output = Mimir.Model.forward(model, input)
-- output = [[0.2, 0.8], [0.3, 0.7]]  -- [2, 2]
```

**Que se passe-t-il ?**

1. **Layer 1** (Linear) : `a = W1 × input + b1`
2. **Layer 2** (ReLU) : `b = max(0, a)`
3. **Layer 3** (Linear) : `c = W2 × b + b2`
4. **Layer 4** (Softmax) : `output = softmax(c)`

### Backward Pass

**Propagation arrière** : Calcul des gradients ∂L/∂θ

```lua
-- Automatique dans train()
Mimir.Model.train(model, dataset, epochs)

-- Ou manuel pour contrôle fin
local output = Mimir.Model.forward(model, input)
local loss = compute_loss(output, target)
local grads = Mimir.Model.backward(model, loss)
model.updateWeights(model, grads, learning_rate)
```

**Que se passe-t-il ?**

1. **Loss** : `L = criterion(output, target)`
2. **∂L/∂output** : Gradient de la loss
3. **Softmax backward** : `∂L/∂c`
4. **Linear backward** : `∂L/∂W2, ∂L/∂b2, ∂L/∂b`
5. **ReLU backward** : `∂L/∂a`
6. **Linear backward** : `∂L/∂W1, ∂L/∂b1`

---

## 🔧 Optimiseurs

### Principe

Un **optimiseur** met à jour les poids du modèle en utilisant les gradients :

```
θ_new = θ_old - f(∇L)
```

### Types d'Optimiseurs

#### 1. **SGD (Stochastic Gradient Descent)**
```lua
model.configure(model, {optimizer = "sgd"})
```
- **Formule** : `θ ← θ - lr × ∇L`
- **Simple** : Mise à jour directe
- **Usage** : Baseline, petits modèles

#### 2. **Adam (Adaptive Moment Estimation)**
```lua
model.configure(model, {optimizer = "adam"})
```
- **Formule** : Utilise moments du 1er et 2e ordre
- **Avantage** : Adaptive learning rate par paramètre
- **Usage** : Défaut moderne, Transformers

#### 3. **RMSprop**
```lua
model.configure(model, {optimizer = "rmsprop"})
```
- **Formule** : Normalisation par écart-type cumulatif
- **Avantage** : Gère les gradients bruyants
- **Usage** : RNNs, séquences

### Hyperparamètres

```lua
model.configure(model, {
    optimizer = "adam",
    learning_rate = 0.001,    -- Pas de descente
    beta1 = 0.9,              -- Adam : moment 1er ordre
    beta2 = 0.999,            -- Adam : moment 2e ordre
    epsilon = 1e-8,           -- Adam : stabilité numérique
    weight_decay = 0.0001     -- L2 regularization
})
```

---

## 🔤 Tokenization

### Concept

**Tokenization** = conversion texte ↔ nombres pour le réseau.

```
Texte : "Hello world"
   ↓ encode
Tokens : [1523, 2134]
   ↓ embed
Vecteurs : [[0.1, 0.3, ...], [0.5, 0.2, ...]]
   ↓ modèle
Output : [[0.8, 0.1, 0.1], ...]
   ↓ decode
Texte : "Bonjour monde"
```

### Vocabulaire

```lua
-- Charger un tokenizer
local tokenizer = Mimir.Tokenizer.create()
Mimir.Tokenizer.loadVocab(tokenizer, "vocab.json")

-- Infos
local vocab_size = Mimir.Tokenizer.getVocabSize(tokenizer)
-- vocab_size = 50000

-- Token ↔ ID
local token_id = Mimir.Tokenizer.tokenToId(tokenizer, "hello")
local token = Mimir.Tokenizer.idToToken(tokenizer, token_id)
```

### Encodage/Décodage

```lua
-- Texte → IDs
local text = "The quick brown fox"
local ids = Mimir.Tokenizer.encode(tokenizer, text)
-- ids = {464, 2068, 7586, 21831}

-- IDs → Texte
local decoded = Mimir.Tokenizer.decode(tokenizer, ids)
-- decoded = "The quick brown fox"
```

### BPE (Byte Pair Encoding)

Mímir supporte BPE pour sous-mots :

```lua
-- Entraîner BPE
Mimir.Tokenizer.trainBPE(tokenizer, corpus_path, vocab_size, output_path)

-- Utiliser
local ids = Mimir.Tokenizer.encodeBPE(tokenizer, "running")
-- ids = {run, ##ning} → [2341, 4523]
```

**Voir** : [Tokenization Guide](04-Tokenization.md) pour détails

---

## 💾 Datasets

### Concept

Un **dataset** est une collection de paires (input, target) pour l'entraînement.

```lua
-- Structure dataset
local dataset = {
    {
        input = {{1, 2, 3}, {4, 5, 6}},
        target = {{0, 1}, {1, 0}}
    },
    {
        input = {{7, 8, 9}, {10, 11, 12}},
        target = {{1, 0}, {0, 1}}
    }
}
```

### Chargement

```lua
-- Depuis JSON
local dataset = Mimir.Dataset.loadFromJson("data.json")

-- Depuis texte (NLP)
local text_dataset = Mimir.Dataset.loadText("corpus.txt", tokenizer)
```

### Format Attendu

```json
[
    {
        "input": [[1.0, 2.0, 3.0]],
        "target": [[0.0, 1.0]]
    },
    {
        "input": [[4.0, 5.0, 6.0]],
        "target": [[1.0, 0.0]]
    }
]
```

**Voir** : [Data Management](05-Data-Management.md) pour détails

---

## 🧠 Mémoire

### Hiérarchie de Gestion

Mímir utilise 3 niveaux de gestion mémoire :

```
┌────────────────────────────────────┐
│   1. MemoryGuard (Limites)         │
│   Impose des limites strictes      │
└────────────────────────────────────┘
            ▼
┌────────────────────────────────────┐
│   2. AdvancedRAMManager (Suivi)    │
│   Tracking, statistiques, alertes  │
└────────────────────────────────────┘
            ▼
┌────────────────────────────────────┐
│   3. DynamicTensorAllocator (LRU)  │
│   Cache, compression, offload      │
└────────────────────────────────────┘
```

### Contrôle Utilisateur

```lua
-- Définir limite mémoire
Mimir.Mimir.MemoryGuard.setLimit(4 * 1024 * 1024 * 1024)  -- 4 GB

-- Activer mode strict
Mimir.Mimir.MemoryGuard.enableStrictMode()

-- Vérifier usage
local usage = memory.getUsage()
-- usage = {used = 1024000000, limit = 4294967296, ...}

-- Nettoyer cache
Mimir.Allocator.evictLRU(512 * 1024 * 1024)  -- Libérer 512 MB
```

**Voir** : [Runtime Engine](../04-Architecture-Internals/02-Runtime-Engine.md) section Mémoire

---

## 💻 Philosophie CPU-Only

### Pourquoi CPU-Only ?

1. **Accessibilité** : Pas besoin de GPU coûteux
2. **Portabilité** : Run anywhere (serveurs, laptops, edge)
3. **Déterminisme** : Comportement reproductible
4. **Debugging** : Plus facile à tracer
5. **Éducation** : Comprendre les algorithmes sans abstraction GPU

### Optimisations SIMD

Mímir compense l'absence de GPU par des optimisations CPU avancées :

```cpp
// AVX2 : 8 floats en parallèle
__m256 a = _mm256_load_ps(data);
__m256 b = _mm256_mul_ps(a, factor);
_mm256_store_ps(result, b);
```

**Résultat** : 3-8× plus rapide que code CPU naïf

### Performances Réalistes

- **Petits modèles** (< 100M params) : Entraînement viable
- **Prototypage** : Itération rapide
- **Inférence** : Temps réel pour beaucoup d'applications
- **Production** : Edge devices, embedded systems

**Voir** : [Why CPU Only](../01-Getting-Started/04-Why-CPU-Only.md)

---

## 🎯 Prochaines Étapes

Maintenant que vous comprenez les concepts, explorez :

- [Model Creation](02-Model-Creation.md) - Créer des modèles custom
- [Predefined Architectures](03-Predefined-Architectures.md) - Utiliser UNet, Transformer, etc.
- [Training](06-Training.md) - Boucles d'entraînement
- [API Reference](../03-API-Reference/00-API-Complete.md) - Documentation complète

---

**Questions ?** Consultez [INDEX](../00-INDEX.md) ou ouvrez une issue sur GitHub.
