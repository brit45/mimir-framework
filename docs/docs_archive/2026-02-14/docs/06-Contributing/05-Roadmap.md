
---

# 📄 2) `05-Advanced/05-Model-Architectures.md`

👉 **Problème identifié**  
La doc laisse penser que toutes les architectures sont **entraînables end-to-end**, ce qui n’est pas encore vrai.

### ✅ Version corrigée

```md
# Model Architectures

Mímir fournit un ensemble d’architectures prédéfinies permettant de :
- construire rapidement des modèles cohérents,
- tester des pipelines CPU-first,
- valider la gestion mémoire et les performances.

⚠️ **Note importante**  
Certaines architectures sont actuellement fournies comme **structures complètes**,  
mais **toutes les opérations nécessaires à un entraînement end-to-end ne sont pas encore implémentées**.

---

## Architectures disponibles

- Transformer (Encoder / Decoder)
- UNet
- Variational Autoencoder (VAE)
- Vision Transformer (ViT)
- GAN (Generator / Discriminator)
- ResNet
- MobileNet
- Architectures de diffusion (structurelles)

Ces architectures permettent :
- l’allocation correcte des paramètres,
- l’initialisation,
- l’inférence partielle ou complète selon les couches,
- des benchmarks réalistes CPU.

---

## Statut des architectures

| Architecture | Structure | Inférence | Entraînement |
| ------------ | --------- | --------- | ------------ |
| Transformer  | ✅         | ✅         | ⚠️ partiel    |
| UNet         | ✅         | ⚠️         | ⚠️            |
| VAE          | ✅         | ⚠️         | ⚠️            |
| ViT          | ✅         | ⚠️         | ⚠️            |
| GAN          | ✅         | ⚠️         | ⚠️            |
| Diffusion    | ✅         | ❌         | ❌            |

⚠️ “Partiel” signifie que certaines opérations ne disposent pas encore d’un backward complet.

---

## Objectif de ces architectures

Ces modèles servent principalement à :
- valider le runtime,
- tester la scalabilité CPU,
- expérimenter des architectures,
- préparer l’implémentation complète des kernels.

Ils **ne sont pas encore destinés à la production ML intensive**.

---

## Voir aussi

- `02-User-Guide/06-Training.md`
- `06-Contributing/05-Roadmap.md`
