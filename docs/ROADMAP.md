# Roadmap - Mímir

Plan de développement futur du framework Mímir.

---

## 🎯 Version Actuelle : 1.0.0

### ✅ Fonctionnalités Actuelles

- [x] 7 architectures de modèles (Encoder, Decoder, EncDec, AE, UNet, ViT, MultiModal)
- [x] Système de tenseurs avec OpenCL
- [x] Optimisations SIMD AVX2
- [x] Autograd (différentiation automatique basique)
- [x] 3 optimiseurs (SGD, Adam, AdamW)
- [x] 5 stratégies de LR decay
- [x] Tokenizer BPE
- [x] Visualisation SFML temps réel
- [x] RAM Manager avancé (compression, LRU, prédiction)
- [x] API Lua pour scripting
- [x] Format SafeTensors
- [x] Layers complets (Conv, Pooling, Norm, Activation)

---

## ✅ Complété (v1.0.0) - Décembre 2025

### Scripting Lua - Implémentation Complète
**Priorité : HAUTE** ✅

- [x] Implémentation complète de `model.train()` depuis Lua
  - Boucle d'entraînement configurable
  - Intégration avec Optimizer (Adam, AdamW, SGD)
  - Support des séquences batchées
  
- [x] Implémentation complète de `model.infer()` depuis Lua
  - Inférence avec tokenization automatique
  - Forward pass complet
  - Décodage des résultats
  
- [x] Implémentation de `model.save()/load()` depuis Lua
  - Sauvegarde SafeTensors complète
  - Métadonnées du modèle
  - Checkpoints atomiques

- [x] Chargement de datasets réels
  - `dataset.load()` fonctionnel
  - Support multi-format (images, texte)
  - Chargement depuis dossiers
  
- [x] `dataset.prepare_sequences()`
  - Batching intelligent
  - Tokenization automatique
  - Padding/truncation configurable

**Priorité : HAUTE** ✅

- [x] Fichiers `Model.hpp` et `Model.cpp` créés
- [x] Implémentation de toutes les 7 architectures:
  - EncoderModel::buildArchitecture() - BERT-like
  - DecoderModel::buildArchitecture() - GPT-like
  - EncoderDecoderModel::buildArchitecture() - T5-like
  - AutoencoderModel::buildArchitecture() - VAE
  - UNetModel::buildArchitecture() - Segmentation/Génération
  - VisionTransformerModel::buildArchitecture() - ViT
  - MultiModalModel::buildArchitecture() - Vision + Language

- [x] ModelFactory::create() pour création dynamique
- [x] Configs structurées pour chaque architecture
- [x] Méthodes spécialisées (encode, decode, pool, generate, etc.)

### Point d'Entrée Principal
**Priorité : MOYENNE** ✅

- [x] `main.cpp` implémenté avec CLI complet
  - Support `--script` pour exécution Lua
  - Support `--config` pour chargement JSON
  - Mode interactif REPL Lua
  - Gestion des arguments et aide

### Scripts Exemples
**Priorité : MOYENNE** ✅

- [x] `scripts/example_training.lua` - Workflow complet
- [x] `scripts/example_gpt.lua` - Génération de texte
- [x] Scripts existants mis à jour

---

## 🚧 En Cours (v1.1.0) - Q1 2026
  - Data augmentation

**Priorité : HAUTE**

- [ ] `Model.cpp` : Implémenter les 7 architectures
  - EncoderModel::buildArchitecture()
  - DecoderModel::buildArchitecture()
  - EncoderDecoderModel::buildArchitecture()
  - AutoencoderModel::buildArchitecture()
  - UNetModel::buildArchitecture()
  - VisionTransformerModel::buildArchitecture()
  - MultiModalModel::buildArchitecture()

---

## 🔮 Planifié (v1.2.0) - Q2 2026

### Export & Interopérabilité
**Priorité : MOYENNE**

- [ ] Export ONNX
  - Conversion modèles Mímir → ONNX
  - Support opérateurs custom
  - Optimisation du graphe
  
- [ ] Import ONNX
  - Chargement modèles ONNX
  - Conversion vers architecture Mímir
  
- [ ] Export TorchScript
  - Compatibilité PyTorch
  
- [ ] Import SafeTensors externes
  - Chargement checkpoints HuggingFace
  - Mapping automatique d'architecture

### Multi-GPU
**Priorité : MOYENNE**

- [ ] Data Parallelism
  - Distribution automatique sur GPU
  - Synchronisation gradients
  - Scaling efficace
  
- [ ] Model Parallelism
  - Découpage de modèles larges
  - Pipeline parallelism
  
- [ ] Mixed Precision Training
  - FP16/BF16 support
  - Loss scaling automatique
  - Gradient accumulation

### Distributed Training
**Priorité : BASSE**

- [ ] Multi-node training
  - MPI/NCCL backend
  - Elastic training
  - Fault tolerance

---

## 🌟 Fonctionnalités Avancées (v1.3.0) - Q3 2026

### Quantization
**Priorité : MOYENNE**

- [ ] Post-Training Quantization (PTQ)
  - INT8/INT4 quantization
  - Per-channel/per-tensor
  
- [ ] Quantization-Aware Training (QAT)
  - Fake quantization
  - Calibration automatique

### Pruning & Compression
**Priorité : BASSE**

- [ ] Magnitude Pruning
  - Structured/unstructured
  - Gradual pruning
  
- [ ] Knowledge Distillation
  - Teacher-student framework
  - Loss distillation

### AutoML
**Priorité : BASSE**

- [ ] Neural Architecture Search (NAS)
  - Search space définition
  - Stratégies de recherche (random, RL, evolutionary)
  
- [ ] Hyperparameter Optimization
  - Bayesian optimization
  - Grid/Random search

---

## 🔬 Recherche & Innovation (v2.0.0) - Q4 2026

### Architectures Nouvelles
**Priorité : RECHERCHE**

- [ ] Mamba/SSM (State Space Models)
  - Alternative aux Transformers
  - Linéaire en complexité
  
- [ ] Mixture of Experts (MoE)
  - Sparse activation
  - Expert routing
  
- [ ] Hyena Hierarchy
  - Long context
  - Sub-quadratic attention

### Optimisations Avancées
**Priorité : RECHERCHE**

- [ ] Flash Attention
  - Memory-efficient attention
  - IO-aware implementation
  
- [ ] Triton Kernels
  - Kernels GPU custom optimisés
  - Auto-tuning

### Reinforcement Learning
**Priorité : RECHERCHE**

- [ ] PPO (Proximal Policy Optimization)
- [ ] RLHF (RL from Human Feedback)
- [ ] Environment abstraction

---

## 🛠️ Infrastructure & Outils (Continu)

### Tests
**Priorité : HAUTE**

- [ ] Tests unitaires
  - Layers (Conv, Pooling, etc.)
  - Activations
  - Optimizers
  - Autograd
  
- [ ] Tests d'intégration
  - Entraînement end-to-end
  - Sauvegarde/chargement
  - Export ONNX
  
- [ ] Benchmarks
  - Performance CPU/GPU
  - Comparaison avec PyTorch/TensorFlow

### Documentation
**Priorité : MOYENNE**

- [x] Documentation complète (INDEX, ARCHITECTURE, APIs)
- [ ] Tutoriels vidéo
- [ ] Documentation API auto-générée (Doxygen)
- [ ] Notebooks Jupyter (via Python bindings)

### CI/CD
**Priorité : MOYENNE**

- [ ] GitHub Actions
  - Build automatique
  - Tests automatiques
  - Packaging
  
- [ ] Docker images
  - CPU-only
  - CUDA
  - ROCm (AMD)

---

## 🌍 Écosystème (v2.1.0+)

### Python Bindings
**Priorité : HAUTE**

- [ ] pybind11 wrapper
  - API Python complète
  - NumPy integration
  - PyTorch interop
  
- [ ] pip package
  - Installation simple : `pip install mimir-dl`

### Language Bindings
**Priorité : BASSE**

- [ ] Rust bindings
- [ ] Julia bindings
- [ ] JavaScript (WASM)

### Frameworks Integration
**Priorité : MOYENNE**

- [ ] PyTorch Lightning
- [ ] Hugging Face Transformers
- [ ] ONNX Runtime

---

## 📊 Optimisations Performance

### CPU
**Priorité : HAUTE**

- [ ] Support ARM Neon (Apple Silicon, ARM servers)
- [ ] AVX-512 support
- [ ] Auto-tuning BLAS
- [ ] Custom allocator (jemalloc, tcmalloc)

### GPU
**Priorité : HAUTE**

- [ ] CUDA backend natif (alternative à OpenCL)
  - cuBLAS, cuDNN integration
  - Tensor Cores (Ampere+)
  
- [ ] ROCm support optimisé (AMD)
- [ ] Metal support (Apple Silicon)
- [ ] Vulkan compute backend

---

## 🎓 Éducation & Communauté

### Ressources Pédagogiques
**Priorité : MOYENNE**

- [ ] Tutoriels interactifs
- [ ] Cours vidéo
- [ ] Notebooks exemples
- [ ] Blog posts techniques

### Communauté
**Priorité : BASSE**

- [ ] Forum/Discord
- [ ] Contributions guidelines
- [ ] Code of conduct
- [ ] Monthly releases

---

## 🐛 Bugs Connus & Limitations

### À Corriger
**Priorité : HAUTE**

- [ ] Memory leaks potentiels dans TensorSystem
- [ ] Thread safety pour LuaContext
- [ ] Validation des configs JSON incomplète

### Limitations Actuelles
**Priorité : MOYENNE**

- [ ] Pas de support Windows natif (WSL uniquement)
- [ ] OpenCL seulement (pas CUDA natif)
- [ ] Backward pass incomplet pour certaines opérations
- [ ] Pas de dynamic batching

---

## 📅 Timeline

```
2025 Q4 : v1.0.0 (ACTUEL)
├── 7 architectures
├── API Lua basique
└── Optimisations SIMD

2026 Q1 : v1.1.0
├── Lua API complète
└── Tests unitaires

2026 Q2 : v1.2.0
├── Export ONNX
├── Multi-GPU
└── Python bindings

2026 Q3 : v1.3.0
├── Quantization
├── AutoML
└── CI/CD

2026 Q4 : v2.0.0
├── Architectures nouvelles (Mamba, MoE)
├── Flash Attention
└── RLHF
```

---

## 🎯 Objectifs Stratégiques

### Court Terme (6 mois)
1. **Stabilité** : Compléter les TODOs critiques
2. **Tests** : Couverture 80%+
3. **Documentation** : Tutoriels complets

### Moyen Terme (1 an)
1. **Performance** : Compétitif avec PyTorch/TensorFlow
2. **Écosystème** : Python bindings production-ready
3. **Adoption** : 1000+ stars GitHub, 100+ utilisateurs actifs

### Long Terme (2 ans)
1. **Innovation** : Architectures state-of-the-art
2. **Industrie** : Adoption en production
3. **Recherche** : Publications académiques

---

## 🤝 Contributions

Les contributions sont bienvenues pour tous les items de la roadmap !

**Priorités pour contributeurs** :
1. 🔴 Critique : Lua API, Model.cpp, Tests
2. 🟡 Important : ONNX export, Multi-GPU, Python bindings
3. 🟢 Nice-to-have : AutoML, Nouvelles architectures

**Comment contribuer** :
- Voir [CONTRIBUTING.md](CONTRIBUTING.md)
- Choisir un item de la roadmap
- Ouvrir une issue/PR
- Rejoindre les discussions

---

## 📝 Notes de Version

### v1.0.0 (24 novembre 2025)
- 🎉 Première release publique
- ✨ 7 architectures de modèles
- ⚡ Optimisations SIMD AVX2
- 🔧 API Lua basique
- 📊 Visualisation SFML
- 💾 Format SafeTensors

---

**Mímir** évolue constamment. Cette roadmap est mise à jour trimestriellement.

Dernière mise à jour : 24 novembre 2025
