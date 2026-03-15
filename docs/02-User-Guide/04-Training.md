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
