# Roadmap - Mímir Framework

## 🎯 Vision

**Mímir est un framework CPU-only** dont la mission est de démocratiser le deep learning en le rendant accessible à tous, sans nécessiter d'investissement dans du matériel GPU coûteux.

### Principes Fondamentaux

1. **CPU-only forever** - Pas de support GPU, jamais
2. **Accessibilité maximale** - Fonctionner sur tout CPU moderne
3. **Performance optimale** - Exploiter chaque cycle CPU disponible
4. **Simplicité** - Minimal dependencies, maximum d'impact
5. **Coût zéro** - Aucun investissement hardware supplémentaire

---

## 🚀 Roadmap v2.x (CPU-Only Evolution)

### v2.1 - Optimisations CPU Avancées (Q1 2026)

**Focus**: Exploiter encore plus les capacités CPU

- [ ] **Mixed precision training** (FP16/FP32 automatique)
  - Réduction mémoire 50%
  - Speedup 1.5-2× avec F16C
  
- [ ] **Gradient clipping et accumulation**
  - Entraînement stable avec petits batchs
  - Simulation de grands batchs sur CPU limités
  
- [ ] **Learning rate schedulers**
  - Cosine annealing
  - Linear warmup
  - Exponential decay
  
- [ ] **Cache-aware algorithms**
  - Tiling optimisé pour L1/L2/L3
  - Minimisation des cache misses

### v2.2 - Architectures Étendues (Q2 2026)

- [ ] **CLIP** - Vision-Language (CPU-friendly)
- [ ] **Whisper** (small) - Speech-to-text sur CPU
- [ ] **Stable Diffusion** (tiny) - Génération d'images optimisée CPU
- [ ] **BERT** - NLP encoder
- [ ] **T5** (small) - Text-to-text transfer

### v2.3 - Multi-Platform CPU (Q3 2026)

**Objectif**: Support de tous les CPUs modernes

- [ ] **ARM Neon optimizations**
  - Raspberry Pi 4/5
  - Apple M1/M2/M3
  - Support mobile ARM
  
- [ ] **AVX-512 support** (optionnel)
  - Pour les serveurs Intel Xeon
  - Détection automatique
  
- [ ] **AMD-specific optimizations**
  - Tuning pour Zen 3/4
  
- [ ] **WebAssembly SIMD**
  - IA dans le navigateur (CPU-only)

### v2.4 - Distributed CPU Training (Q4 2026)

**Objectif**: Utiliser plusieurs machines CPU

- [ ] **MPI support** - Distributed training multi-CPU
- [ ] **Parameter server** - Architecture master/worker
- [ ] **Ring AllReduce** - Communication optimisée
- [ ] **CPU cluster** - Entraînement sur cluster de CPUs pas chers

---

## 🎓 v3.0 - Bindings et Outils (2027)


### Outils de Production

- [ ] **Model quantization** (INT8/INT4)
  - Réduction mémoire 4-8×
  - Speedup 2-4× sur CPU
  - Pas de perte significative
  
- [ ] **ONNX export**
  - Interopérabilité
  - Déploiement CPU partout
  
- [ ] **Model pruning**
  - Réduction taille 50-90%
  - Speedup inférence
  
- [ ] **Knowledge distillation**
  - Créer des petits modèles depuis grands modèles

### Dataset & Training Tools

- [ ] **Data augmentation** intégrée
- [ ] **Automatic batching** optimisé CPU
- [x] **Checkpointing** automatique
- [ ] **TensorBoard** logging
- [ ] **Profiling tools** CPU

---

## ❌ Non-Goals (Ce qui ne sera JAMAIS fait)

### GPU Support

- ❌ **CUDA** - Jamais
- ❌ **ROCm** - Jamais  
- ❌ **Metal** - Jamais (sauf si CPU ARM via Neon)

**Raison**: Mímir est CPU-only par philosophie. L'objectif est l'accessibilité, pas la performance maximale absolue.

### Cloud Payant

- ❌ **Cloud inference services** - Pas de lock-in cloud
- ❌ **SaaS offerings** - Tout reste local et gratuit

### Complexité Inutile

- ❌ **Dependencies lourdes** - On garde minimal
- ❌ **Framework bloat** - Simplicité avant tout
- ❌ **Enterprise features** - On reste accessible

---

## 🎯 Use Cases Optimaux pour Mímir

### ✅ Excellents pour Mímir (CPU-only)

1. **Prototypage rapide** - Test d'architectures
2. **Petits modèles** (<100M params) - Entraînement rapide sur CPU
3. **Modèles moyens** (100-500M params) - Entraînement overnight
4. **Inférence locale** - Déploiement sur edge/desktop
5. **Apprentissage** - Enseignement du deep learning
6. **Recherche reproductible** - Pas de dépendance GPU
7. **Fine-tuning** - Adapter des modèles pré-entraînés

### ⚠️ Possibles mais lents

1. **Grands modèles** (500M-2B params) - Plusieurs jours d'entraînement
2. **Vision haute résolution** - Images >512×512
3. **Très grands batchs** - Limité par RAM CPU

### ❌ Non recommandés

1. **Modèles géants** (>2B params) - Utilisez un framework GPU
2. **Production à large échelle** - Considérez du hardware dédié
3. **Real-time vidéo** - Préférez GPU ou hardware spécialisé

---

## 💡 Philosophie de Développement

### Priorités

1. **Accessibilité** > Performance absolue
2. **Simplicité** > Features complexes
3. **CPU optimization** > Support multi-backend
4. **Local-first** > Cloud-first
5. **Open source** > Propriétaire

### Décisions de Design

- **Pas de GPU**: Choix assumé pour rester simple et accessible
- **C++17**: Moderne mais largement supporté
- **Minimal dependencies**: Juste Lua, OpenMP, standard library
- **Header-only quand possible**: Faciliter l'intégration
- **Documentation extensive**: Rendre accessible aux débutants

---

## 🤝 Contribution

Vous voulez contribuer? Consultez [CONTRIBUTING.md](../CONTRIBUTING.md).

**Important**: Les contributions ajoutant du support GPU seront poliment refusées. Mímir reste CPU-only par design.

---

## 📅 Timeline Résumé

| Version | Date | Focus |
|---------|------|-------|
| v2.0 | ✅ Dec 2025 | 8 architectures + API Lua + Optimisations CPU |
| v2.1 | Q1 2026 | Mixed precision + Schedulers |
| v2.2 | Q2 2026 | Plus d'architectures (CLIP, Whisper, etc.) |
| v2.3 | Q3 2026 | ARM Neon + Multi-platform CPU |
| v2.4 | Q4 2026 | Distributed CPU training (MPI) |

---

**Mission**: Garder l'IA accessible à tous, sans barrière financière ou technique.

**Vision**: Un framework CPU-only de référence pour le prototypage, l'apprentissage et le déploiement local d'IA.
