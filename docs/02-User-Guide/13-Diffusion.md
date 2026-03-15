# Tutoriel : diffusion (PonyXL / SD3.5 / autoencoder)

Ce dépôt contient des architectures diffusion/autoencoder exposées via le registre.

Voir aussi:

- Exemples: `10-Examples.md`
- Datasets (linking par basename): `03-Data.md`
- Checkpoints: `08-Checkpoints.md`

## PonyXL (SDXL-like)

- Architecture canonique : `ponyxl_sdxl` (anciens alias: `ponyxl_ddpm`, `t2i_autoencoder`, etc.)
- Scripts : `scripts/examples/ponyxl_generate.lua`, `scripts/training/train_ponyxl_ddpm.lua`

### Entraîner (PonyXL DDPM)

Script: `scripts/training/train_ponyxl_ddpm.lua`

Pré-requis:

- dataset chargé via `MIMIR_DATASET_ROOT` (sinon fallback `../dataset`)
- un dataset multi-modal “image+texte” est généralement attendu

Important (format dataset):

- le loader indexe récursivement et associe les modalités par **basename**
- en pratique, pour un exemple image+caption, il faut des fichiers comme:

```text
dataset/
  0001.png
  0001.txt
  0002.jpg
  0002.txt
```

L’organisation en sous-dossiers est possible, mais évite les collisions de basename.

Lancement:

```bash
export MIMIR_DATASET_ROOT="/chemin/vers/dataset"
./bin/mimir --lua scripts/training/train_ponyxl_ddpm.lua
```

Resume:

- le script supporte un mode auto-resume (variable `MIMIR_RESUME`)
- il charge un checkpoint existant dans `checkpoint_dir` si trouvé

Le script écrit généralement:

- un checkpoint `raw_folder` dans `cfg.checkpoint_dir`
- un `debug.json` (format debug, pour inspection humaine)

## SD3.5 (skeleton / démos)

- Architecture : `sd3_5` (alias: `SD3.5`)
- Démos : `scripts/examples/sd3_5_skeleton_demo.lua`, `sd3_5_subset_load_demo.lua`

Ces scripts sont des démos/squelettes: ils servent surtout à valider les chemins de load/config.

## Autoencoder image

- VAE conv : `vae_conv` (vision)
- Script training : `scripts/training/train_vae_conv.lua`

Astuce: commence par un VAE conv petit et valide le save/load avant d’attaquer une pipeline diffusion complète.

## Générer (PonyXL)

Script: `scripts/examples/ponyxl_generate.lua`

Ce script:

- crée/build/alloue un modèle `ponyxl` avec une config qui doit matcher le checkpoint
- charge un checkpoint (par défaut `checkpoint/PonyXL` en `raw_folder`)
- charge un tokenizer (priorité au “base tokenizer”, sinon tokenizer du checkpoint)
- calcule 2 forwards (cond/uncond) et applique une guidance simple
- écrit une image PGM (grayscale) sur disque

Lancement:

```bash
export MIMIR_BASE_TOKENIZER="checkpoint/PonyXL/tokenizer/tokenizer.json"
./bin/mimir --lua scripts/examples/ponyxl_generate.lua
```

## Statut

Les scripts de diffusion sont des **démos / squelettes** tant qu’aucun checkpoint entraîné n’est fourni.
Pour générer des images cohérentes, il faut :

- un VAE image entraîné
- un U-Net/diffusion entraîné
- une config de scheduler cohérente
