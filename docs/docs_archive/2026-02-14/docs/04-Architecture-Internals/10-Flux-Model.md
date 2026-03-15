# FluxModel - Implémentation Complète

> ⚠️ **Obsolète (v2.3)** : `FluxModel` n'est pas présent/exposé dans l'implémentation actuelle (voir `src/LuaScripting.cpp` et le registre d'architectures).
> Ce document est conservé à titre historique.

## Vue d'ensemble

Le `FluxModel` est un modèle de diffusion complet avec VAE, text encoder et transformer, entièrement implémenté sans placeholders ni TODOs.

## Modes d'Exécution

### Mode Training
```cpp
model.train();
```
- Active le mode entraînement
- Applique la reparametrization trick dans le VAE (ajout de bruit)
- Active le dropout dans les transformers
- Implémente les comportements stochastiques appropriés

### Mode Evaluation/Inference
```cpp
model.eval();
```
- Active le mode inférence
- Désactive le bruit dans le VAE (mu uniquement)
- Désactive le dropout
- Comportement déterministe pour la génération

### Vérification du Mode
```cpp
bool training = model.isTraining();
```

## Architecture Complète

### 1. VAE Encoder (`encodeImage`)

**Pipeline complet :**
```
Image RGB (256×256×3)
  ↓
Input Convolution
  ↓
ResNet Blocks + Downsampling (4 étapes)
  ├─ Group Normalization
  ├─ Conv2D avec stride=2
  └─ Skip connections
  ↓
Middle Blocks avec Attention
  ↓
Projection vers latent space
  ├─ mu (moyenne)
  └─ logvar (variance logarithmique)
  ↓
Reparametrization Trick: z = μ + σ·ε
  ↓
Latent (32×32×4)
```

**Caractéristiques :**
- Downsampling progressif : 256→128→64→32
- Channel multipliers : [1, 2, 4, 4] × base_channels
- Reparametrization trick en mode training
- Sampling déterministe (mu uniquement) en mode eval

### 2. VAE Decoder (`decodeLatent`)

**Pipeline complet :**
```
Latent (32×32×4)
  ↓
Projection vers features
  ↓
Middle Blocks
  ↓
ResNet Blocks + Upsampling (4 étapes)
  ├─ ConvTranspose2D (stride=2)
  ├─ Group Normalization
  └─ Skip connections
  ↓
Output Convolution
  ↓
Tanh activation
  ↓
Image RGB (256×256×3)
```

**Caractéristiques :**
- Upsampling progressif : 32→64→128→256
- Interpolation bilinéaire pour reconstruction
- Activation tanh pour borner [-1, 1]
- Reconstruction symétrique à l'encoder

### 3. Text Encoder (`encodeText`)

**Pipeline complet :**
```
Tokens (77 tokens max)
  ↓
Token Embeddings
  ↓
Position Embeddings (sinusoidal)
  ↓
12 Transformer Layers
  ├─ Self-Attention
  │   ├─ Query, Key, Value projections
  │   ├─ Scaled dot-product attention
  │   └─ Multi-head (12 heads)
  ├─ Residual Connection
  ├─ Layer Normalization
  ├─ Feed-Forward Network
  │   ├─ Expansion (dim × 4)
  │   ├─ ReLU activation
  │   └─ Contraction (→ dim)
  └─ Residual Connection
  ↓
Projection vers transformer_dim
  ↓
Text Embeddings (77 × 768)
```

**Caractéristiques :**
- Style CLIP avec 12 couches transformer
- Position encoding sinusoïdal
- Multi-head attention (12 têtes)
- FFN avec ratio 4:1
- Projection finale vers dimension transformer

### 4. Diffusion Transformer (`predictNoise`)

**Pipeline complet :**
```
Noisy Latent (32×32×4) + Text Embedding + Timestep
  ↓
Timestep Embedding
  ├─ Sinusoidal encoding
  └─ MLP transformation
  ↓
Latent → Tokens (1024 tokens)
  ↓
12 Transformer Blocks
  ├─ Self-Attention sur latents
  ├─ AdaLN (Adaptive Layer Norm)
  │   └─ Modulation par timestep
  ├─ Cross-Attention avec texte
  │   ├─ Query depuis latents
  │   ├─ Key, Value depuis text
  │   └─ Conditioning textuel
  ├─ Feed-Forward Network
  │   └─ MLP ratio configurable
  └─ Residual Connections
  ↓
Projection vers espace latent
  ↓
Predicted Noise (32×32×4)
```

**Caractéristiques :**
- Self-attention spatiale sur latents
- Cross-attention pour conditioning textuel
- AdaLN (Adaptive Layer Normalization) modulé par timestep
- Architecture type DiT (Diffusion Transformer)
- Prédiction du bruit dans l'espace latent

## Tokenization

### `tokenizePrompt`

**Implémentation complète :**
1. **Avec Tokenizer externe :**
   - Utilise le tokenizer fourni (BPE, WordPiece, etc.)
   
2. **Fallback interne :**
   - Tokenization par mots
   - Hash des mots vers token IDs
   - Gestion BOS/EOS tokens
   - Padding/truncation vers max_length (77)

**Format :**
```
[BOS] token1 token2 ... tokenN [EOS] [PAD] [PAD] ...
```

## Embeddings

### Timestep Embedding (`getTimeEmbedding`)

**Formule sinusoïdale :**
```cpp
freq = exp(-2d/D × log(10000))
embedding[d] = sin(t × freq)  si d pair
             = cos(t × freq)  si d impair
```

**Caractéristiques :**
- Inspiré de "Attention is All You Need"
- Position embedding adapté au temps
- Fréquences exponentiellement décroissantes
- Permet d'encoder n'importe quel timestep

## Pipeline de Génération

### `generate(prompt, num_steps)`

**Processus complet :**
```
1. Tokenize prompt
2. Encode text → text_embedding
3. Random noise latent ~ N(0, I)
4. For t = T...1:
   a. Predict noise avec transformer
   b. Denoise step (DDPM/DDIM)
   c. Update latent
5. Decode latent → image RGB
6. Return image
```

## Différences Train vs Eval

| Aspect | Training | Inference |
|--------|----------|-----------|
| VAE Encoder | z = μ + σ·ε | z = μ |
| Dropout | Actif | Désactivé |
| Batch Norm | Stats par batch | Stats moyennées |
| Stochasticité | Aléatoire | Déterministe |

## Configuration

```cpp
FluxConfig {
    image_resolution = 256,
    latent_channels = 4,
    latent_resolution = 32,
    vae_base_channels = 128,
    vae_channel_mult = {1, 2, 4, 4},
    num_res_blocks = 2,
    vocab_size = 50000,
    text_max_length = 77,
    text_embed_dim = 768,
    transformer_dim = 768,
    num_transformer_blocks = 12,
    num_attention_heads = 12,
    mlp_ratio = 4.0,
    timestep_embed_dim = 256,
    num_diffusion_steps = 1000
}
```

## Optimisations

### Implémentées
- ✅ Interpolation bilinéaire pour upsampling
- ✅ Attention scalée (softmax avec sqrt(dim))
- ✅ Residual connections partout
- ✅ Layer normalization
- ✅ AdaLN pour modulation temporelle

### Possibles (futures)
- [ ] Flash Attention pour O(N) au lieu de O(N²)
- [ ] Grouped Query Attention (GQA)
- [ ] Mixed precision (FP16/BF16)
- [ ] Gradient checkpointing
- [ ] Multi-GPU avec model parallelism

## Exemple d'Utilisation

### C++
```cpp
#include "Models/FluxModel.hpp"

FluxConfig config;
config.image_resolution = 256;
// ... autres params

FluxModel model(config);

// Génération
model.eval();
auto image = model.generate("beautiful sunset", 50);

// Entraînement
model.train();
float loss = model.computeDiffusionLoss(image, tokens);
```

### Lua
```lua
local config = {
    image_resolution = 256,
    latent_channels = 4,
    -- ...
}

local model = FluxModel.new(config)

-- Génération
model:eval()
local image = model:generate("beautiful sunset", 50)

-- Entraînement
model:train()
local loss = model:computeDiffusionLoss(image, tokens)
```

## Tests

Script de test complet : `scripts/test_flux_complete.lua`

```bash
./bin/mimir scripts/test_flux_complete.lua
```

Vérifie :
- ✅ Modes train/eval
- ✅ VAE encode/decode
- ✅ Text encoding
- ✅ Noise prediction
- ✅ Génération complète

## Performance

### Complexité
- VAE Encoder : O(H×W×C×D)
- VAE Decoder : O(H×W×C×D)
- Text Encoder : O(L×D² × N_layers)
- Diffusion Transformer : O(L_latent×D² × N_blocks)

### Mémoire
- Image (256×256×3) : 192 KB
- Latent (32×32×4) : 4 KB (48× compression)
- Text Embedding (77×768) : 236 KB
- Activations : ~50 MB (dépend de batch size)

## Références

- VAE : "Auto-Encoding Variational Bayes" (Kingma & Welling, 2014)
- Diffusion : "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- Text Encoder : "Learning Transferable Visual Models From Natural Language Supervision" (CLIP, 2021)
- Transformer : "Attention is All You Need" (Vaswani et al., 2017)
- AdaLN : "Scalable Diffusion Models with Transformers" (DiT, 2023)

## Conclusion

Le FluxModel est maintenant **entièrement implémenté** avec :
- ✅ Aucun placeholder
- ✅ Aucun TODO
- ✅ Modes train/eval complets
- ✅ Forward passes fonctionnels
- ✅ Pipeline de génération end-to-end

Prêt pour l'entraînement et l'inférence ! 🚀
