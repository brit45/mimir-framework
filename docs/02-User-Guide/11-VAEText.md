# Tutoriel : VAEText

VAEText est un VAE “texte” qui reconstruit des tokens via logits, entraîné avec reconstruction (Cross-Entropy) + KL.

## Entraîner

Script : `scripts/training/train_vae_texte.lua`

Exemple :

```bash
./bin/mimir --lua scripts/training/train_vae_texte.lua -- \
  --dataset-root ../dataset_2 \
  --out-dir checkpoint/vae_text_trained \
  --epochs 5 --lr 1e-4 \
  --seq-len 256 --d-model 256 --latent-tokens 32 \
  --kl-beta 0.01
```

Points importants :

- Utilise un “base tokenizer” commun (`scripts/modules/base_tokenizer.lua`) et `tokenizer_frozen=true`.
- Met `cfg.checkpoint_dir = out_dir` pour permettre des checkpoints d’interruption Ctrl+C.

## Sampler

Script : `scripts/examples/vae_text_sample.lua`

- `--mode posterior` : encode prompt → z → decode logits → sample tokens.
- `--mode recon` : decode déterministe (argmax).
- `--mode prior` : **true prior** via l’architecture `vae_text_decode` (z ~ N(0, I)).

Exemple :

```bash
./bin/mimir --lua scripts/examples/vae_text_sample.lua -- \
  --ckpt checkpoint/vae_text_trained \
  --prompt "bonjour le monde" \
  --mode posterior --temperature 1.0 --top-k 50
```

## Compat checkpoint

Le sampler lit `model/architecture.json` dans le checkpoint et refuse les overrides `seq_len` incompatibles.
C’est volontaire : changer `seq_len` après entraînement casse souvent les shapes.
