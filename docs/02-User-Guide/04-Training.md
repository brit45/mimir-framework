# Entraînement

## API

Deux styles existent :

1) Helpers “trainStep” fournis par le runtime (ex: VAE, diffusion, etc.).
2) Boucle manuelle : `forward` + `backward` + optimizer.

Dans la pratique, la majorité des scripts actuels s’appuie sur `Mimir.Model.train(epochs, lr)`, dont le comportement est **architecture-dépendant**.

Voir aussi:

- API `Mimir.Model`: `../03-API-Reference/10-Model.md`
- Datasets: `03-Data.md`
- Checkpoints: `08-Checkpoints.md`

## Workflow recommandé (scripts)

Un training “standard” ressemble à:

1) Configurer la mémoire (fortement recommandé)
2) Charger/préparer tokenizer si nécessaire
3) Charger le dataset
4) Créer/build/allouer le modèle
5) `init_weights()` **ou** `Serialization.load()` (resume)
6) Lancer `Model.train(epochs, lr)`
7) Sauvegarder

### 1) Mémoire / sécurité OOM

Les scripts d’exemple configurent généralement:

```lua
pcall(Mimir.MemoryGuard.setLimit, 10)
pcall(Mimir.Allocator.configure, {

  max_ram_gb = 10.0,
  enable_compression = true,
  swap_strategy = "lru",
})
pcall(Mimir.Model.set_hardware, true)
```

Exemples: `scripts/examples/example_simple.lua`, `scripts/examples/example_gpt.lua`.

### 2) Tokenizer (selon le modèle)

- Certains scripts créent un tokenizer “simple” (`Mimir.Tokenizer.create(vocab)`), utile pour les tests.
- Les trainings plus sérieux chargent un “base tokenizer” (voir `scripts/modules/base_tokenizer.lua`) et figent `vocab_size`.
- Pour l’inférence, il faut souvent charger le tokenizer qui a servi à l’entraînement (ex: `scripts/examples/ponyxl_generate.lua`).

### 3) Dataset

Le loader indexe récursivement un dossier et linke les modalités par basename.

- `Mimir.Dataset.load(dir)`
- pour les trainings texte legacy: `Mimir.Dataset.prepare_sequences(seq_len)`

Voir `03-Data.md` pour les détails et limitations.

## `Mimir.Model.train(epochs, lr)` (haut niveau)

Ce chemin est utilisé par plusieurs scripts.

Exemples:

- VAEText: `scripts/training/train_vae_texte.lua`
- PonyXL DDPM: `scripts/training/train_ponyxl_ddpm.lua`
- GPT démo: `scripts/examples/example_gpt.lua`

À retenir:

- le dataset doit être chargé si le trainer de l’archi l’utilise
- certains modèles gèrent des composants “internes” (ex: tokenizer interne), ce qui explique pourquoi tous les scripts ne font pas `Dataset.prepare_sequences()`

## Boucle manuelle (avancé)

Une boucle “générique” `forward + backward + optimizer_step` n’est pas toujours ergonomique en Lua car:

- `forward()` renvoie une table de floats, et `backward()` attend un `grad_out` correspondant à cette sortie
- la loss et ses gradients doivent être calculés d’une manière compatible avec les sorties du modèle

Si ton objectif est d’écrire un entraînement custom, pars d’un script existant du dossier `scripts/training/` et adapte-le (c’est la source la plus fiable sur le comportement réel).

## Calibration par feedback de validation

Lorsque la validation est activée (`validate_every_steps` ou `validate_every_epochs`), le framework peut automatiquement ajuster le learning rate effectif en fonction de l'évolution de la métrique de validation. C'est un mécanisme de **récompense/punition** : si la validation s'améliore, le LR est légèrement augmenté ; si elle se dégrade, il est réduit.

Ce comportement est actif par défaut (`val_feedback_enabled = true`).

### Fonctionnement

Un facteur `val_lr_scale` (initialisé à `1.0`) est multiplié sur le LR à chaque step :

```
lr_effectif = step_learning_rate() × val_lr_scale
```

Après chaque validation, la métrique courante (inférieure = meilleure) est comparée au meilleur résultat vu :

| Cas | Effet sur `val_lr_scale` |
|---|---|
| 1ère validation | initialise la référence, pas de changement |
| amélioration relative > seuil | `val_lr_scale × val_reward_factor` (plafonné) |
| dégradation relative > seuil | `val_lr_scale × val_penalty_factor` (planché) |
| plateau (variation < seuil) | aucun changement |

La métrique utilisée dépend du type de modèle :

- **VAE / VAEConv** → `final_recon` (perte de reconstruction MSE)
- **Tags / VGG16** → `val_loss` (BCE logits)
- **DDPM / PonyXL** → `final_eps` (MSE sur l'erreur de prédiction)

Les événements sont loggés avec le préfixe `[val_feedback]`.

### Paramètres

Tous les paramètres sont à passer dans le dictionnaire `cfg` transmis à `Mimir.Model.train` :

| Clé | Défaut | Description |
|---|---|---|
| `val_feedback_enabled` | `true` | Activer/désactiver le mécanisme |
| `val_reward_factor` | `1.05` | Multiplicateur LR en cas d'amélioration |
| `val_penalty_factor` | `0.70` | Multiplicateur LR en cas de dégradation |
| `val_lr_scale_min` | `0.10` | Plancher absolu du facteur LR |
| `val_lr_scale_max` | `1.50` | Plafond absolu du facteur LR |
| `val_improve_thresh` | `0.001` | Amélioration relative minimale pour déclencher reward/penalty |
| `val_feedback_min_steps` | `0` | N° de step à partir duquel le feedback est actif |

### Exemple (Lua)

```lua
cfg.validate_every_steps  = 100
cfg.validate_items        = 8
cfg.validate_holdout      = true
cfg.validate_holdout_frac = 0.1

-- Calibration (valeurs par défaut, on peut omettre ces lignes)
cfg.val_feedback_enabled   = true
cfg.val_reward_factor      = 1.05   -- +5 % LR si amélioration
cfg.val_penalty_factor     = 0.70   -- -30 % LR si dégradation
cfg.val_lr_scale_min       = 0.10   -- plancher à 10 % du LR initial
cfg.val_lr_scale_max       = 1.50   -- plafond à 150 % du LR initial
cfg.val_improve_thresh     = 0.001  -- seuil relatif : 0.1 %
cfg.val_feedback_min_steps = 200    -- attendre 200 steps avant d'activer
```

### Notes

- `val_lr_scale` est indépendant du schedule LR (warmup, cosine decay, etc.) : les deux se combinent.
- La récompense ne met à jour `val_best_metric` que quand il y a amélioration (le plancher n'est pas remonté en cas de plateau). La punition ne met pas à jour la référence, ce qui permet de punir à nouveau si la dégradation continue.
- Pour désactiver proprement : `cfg.val_feedback_enabled = false`.
- Pour une politique purement punitive (pas de reward) : `cfg.val_reward_factor = 1.0`.
- Pour un comportement ReduceLROnPlateau strict : `cfg.val_reward_factor = 1.0` + `cfg.val_improve_thresh = 0.0`.

## Recommandations

- Démarrer petit : dims réduites, peu de layers.
- Valider que `loss` baisse sur un smoke test.
- Sauvegarder souvent au début.

## Mémoire

- Utiliser les garde-fous (limite RAM).
- Sur modèles lourds, préférer des dimensions adaptées au CPU.

## Reproductibilité

- Fixer les seeds si possible.
- Tracer les configs exactes dans le checkpoint.
