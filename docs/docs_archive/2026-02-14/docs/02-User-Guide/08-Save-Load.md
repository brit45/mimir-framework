# Save & Load (Checkpoints)

Mímir permet de sauvegarder et charger des modèles via un système de checkpoints.

---

## Ce qui est sauvegardé

Selon la configuration :
- structure du modèle
- paramètres (blocs de poids)
- métadonnées nécessaires à la reconstruction

⚠️ Les gradients et buffers internes peuvent également être présents selon la version.

---

## État du format

⚠️ **Le format de checkpoint est fonctionnel mais évolutif.**

Il est actuellement :
- stable pour les tests et benchmarks,
- compatible entre versions mineures proches,
- susceptible d’évoluer à mesure que :
  - la gestion des poids est optimisée,
  - la sérialisation est allégée,
  - la rétrocompatibilité est renforcée.

---

## Bonnes pratiques

- Utiliser les checkpoints pour :
  - tests
  - benchmarks
  - itérations locales
- Éviter pour l’instant une dépendance long-terme sans validation de version

---

## Évolutions prévues

- Séparation claire : poids / gradients / buffers
- Format plus compact
- Versioning explicite des checkpoints

Voir : `06-Contributing/03-Migration-Guide.md`
