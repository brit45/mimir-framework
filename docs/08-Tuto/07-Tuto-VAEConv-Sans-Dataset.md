# Valider VAEConv sans dataset

Construire un petit VAE, vérifier son contrat latent et tester le prior appris sans dépendre d’un corpus d’images.

**Public concerné :** Développeur qui modifie l’architecture VAEConv, l’autograd ou le runtime.

> **Prérequis**
>
> Le projet doit être configuré dans `build/`.

## Étape 1 — Compiler le test ciblé

```bash
cmake -S . -B build -DMIMIR_ENABLE_TESTS=ON
cmake --build build --target ModelTest.VAEConvContract ModelTest.VAEConvLogvar -j2
```

Le test utilise une image synthétique `4×4×1` créée en mémoire. Il ne lit aucun dataset.

## Étape 2 — Exécuter le contrat

```bash
ctest --test-dir build --output-on-failure -R '^ModelTest\.(VAEConvContract|VAEConvLogvar)$'
```

Résultat attendu :

```text
ModelTest.VAEConvContract ... Passed
ModelTest.VAEConvLogvar ... Passed
```

## Étape 3 — Comprendre les quatre assertions importantes

Le fichier `Tests/test_vae_conv_contract.cpp` vérifie successivement :

1. une couche `Constant` marquée `trainable_parameter` reçoit exactement le gradient amont ;
2. SGD modifie effectivement ses poids ;
3. la tranche centrale de `x` est identique au tenseur `vae_conv/mu`, même si `z` est stochastique et biaisé ;
4. un gradient placé uniquement sur la reconstruction remonte à travers les convolutions du décodeur jusqu’à `vae_conv/z_prior_bias`.

La quatrième assertion est essentielle : elle montre que le prior apprend à partir des features reconstruites, et pas seulement lorsqu’un gradient artificiel lui est appliqué directement.

`Tests/test_vae_conv_logvar.cpp` contrôle aussi le chemin stochastique et les
gradients de la tête `logvar`. `logvar=0` signifie un écart-type de 1, et non
l'absence de bruit. L'option `logvar_head_bias` ajoute un biais apprenable à
cette tête : elle est activée pour les nouveaux runs du script d'entraînement,
mais sa valeur native par défaut reste `false` pour les anciens graphes. Une
reprise conserve la topologie du checkpoint. Les métriques `logvar`,
`posterior_std` et les normes de gradients permettent de vérifier son activité ;
une valeur moyenne seule ne démontre pas que la tête est inutilisée.

## Étape 4 — Vérifier les gradients numériques voisins

```bash
cmake --build build --target \
  AutogradTest.Numerical \
  RuntimeTest.MathConv2d \
  RuntimeTest.MathNorms \
  RuntimeTest.MathAttention

ctest --test-dir build --output-on-failure \
  -R 'AutogradTest.Numerical|RuntimeTest.MathConv2d|RuntimeTest.MathNorms|RuntimeTest.MathAttention'
```

Ces tests isolent les primitives utilisées par VAEConv. Un succès du seul test de modèle ne remplace pas ces contrôles mathématiques.

## Étape 5 — Modifier le modèle sans casser le contrat

Après une modification de `VAEConvModel.cpp`, vérifie :

- que `out_concat` reçoit toujours `recon`, `mu`, puis `logvar` ;
- que le décodeur reçoit toujours `z` ou `z_biased` ;
- que les convolutions stride 2 et les upsamplings produisent des résolutions symétriques ;
- qu’un nouveau paramètre sans entrée est explicitement marqué apprenable ;
- que les constantes structurelles restent fixes ;
- que tout clamp du forward possède une dérivée cohérente dans le backward.

## Étape 6 — Contrôle final

```bash
git diff --check
ctest --test-dir build --output-on-failure \
  -R 'ModelTest.VAEConvContract|ModelTest.VAEConvLogvar|AutogradTest.Numerical|RuntimeTest.Math'
```

Si le contrat VAE change volontairement, mets à jour ensemble :

- `src/Models/Vision/VAEConvModel.hpp`;
- `src/Models/Vision/VAEConvModel.cpp`;
- `src/Models/Registry/ModelArchitectures.cpp`;
- les helpers de loss et métriques VAE dans `src/Models/VAETraining.cpp`;
- ce tutoriel et la page `docs/02-User-Guide/14-VAEConv.md`.

## Étapes suivantes

- [Page précédente : Tuto - Parcours complet du framework](06-Parcours-Complet-Framework.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Tutoriel : du registre au checkpoint](08-Tuto-Registre-Pipeline-Checkpoint.md)
