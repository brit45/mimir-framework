# Diffusion : PonyXL, SD3.5 et autoencodeurs

Comprendre et exécuter les chemins diffusion disponibles.

**Public concerné :** Intermédiaire guidé.

> **Prérequis**
>
> Checkpoints/config adaptés au modèle choisi.

## Diagrammes d'explication

![Architecture VAE Conv](../graphs/17_vaeconv_architecture.svg)

![Architecture UNet](../graphs/18_unet_architecture.svg)

![Architecture diffusion](../graphs/19_diffusion_architecture.svg)

![Transformer SD3.5](../graphs/20_sd35_transformer.svg)

Ce dépôt contient des architectures diffusion/autoencoder exposées via le registre.

Voir aussi:

- Exemples: `10-Examples.md`
- Datasets (linking par basename): `03-Data.md`
- Checkpoints: `08-Checkpoints.md`

## PonyXL (SDXL-like)

- Architecture canonique : `ponyxl_ddpm`
- Scripts : `scripts/training/ponyxl_ddpm_train.lua`, `scripts/inferences/ponyxl_ddpm_text2img.lua`

### Entraîner (PonyXL DDPM)

Script: `scripts/training/ponyxl_ddpm_train.lua`

Pré-requis:

- dataset passé via option `--dataset` (ex: `--dataset dataset_2`)
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
./bin/mimir --lua scripts/training/ponyxl_ddpm_train.lua --dataset "/chemin/vers/dataset"
```

Resume:

- le script supporte un mode resume via option CLI (`--resume`)
- il charge un checkpoint existant dans `checkpoint_dir` si trouvé

Le script écrit généralement:

- un checkpoint `raw_folder` dans `cfg.checkpoint_dir`
- un `debug.json` (format debug, pour inspection humaine)

## SD3.5 (skeleton / démos)

- Architecture : `sd3_5` (alias: `SD3.5`)
- Statut : squelette/placeholder enregistré, sans script de démonstration dédié dans
  le dépôt courant.

Inspection :

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- \
  --list sd3_5 --params --layers
```

## Autoencoder image

- VAE conv : `vae_conv` (vision)
- Script training : `scripts/training/train_vae_conv.lua`

Astuce: commence par un VAE conv petit et valide le save/load avant d’attaquer une pipeline diffusion complète.

## Générer (PonyXL)

Script: `scripts/inferences/ponyxl_ddpm_text2img.lua`

Ce script:

- crée/build/alloue un modèle PonyXL/DDPM avec une config alignée au checkpoint
- charge les composants nécessaires (checkpoint, tokenizer/encodeur, VAE selon script)
- exécute une chaîne texte -> latent -> image
- écrit le résultat image sur disque

Lancement:

```bash
export MIMIR_BASE_TOKENIZER="checkpoint/PonyXL/tokenizer/tokenizer.json"
./bin/mimir --lua scripts/inferences/ponyxl_ddpm_text2img.lua
```

## Statut

Les scripts de diffusion sont des **démos / squelettes** tant qu’aucun checkpoint entraîné n’est fourni.
Pour générer des images cohérentes, il faut :

- un VAE image entraîné
- un U-Net/diffusion entraîné
- une config de scheduler cohérente

## Étapes suivantes

- [Page précédente : Tutoriel : Transformer causal (GPT-style)](12-Transformer-GPT.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : VAEConv : architecture, configuration et entraînement](14-VAEConv.md)
