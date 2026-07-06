# Tutoriel : VAEText

## Pour qui

Intermédiaire guidé.

## Objectif

Entraîner et évaluer VAEText avec les bonnes options.

## Avant de commencer

Dataset texte disponible.

## Résultat attendu

Tu peux produire un checkpoint VAEText exploitable.


VAEText est un VAE “texte” qui reconstruit des tokens via logits, entraîné avec reconstruction (Cross-Entropy) + KL.

Version actuelle: VAEText peut aussi exposer un encodeur conditionnel de corrélation textuelle,
et des têtes de contexte internes (sémantique, thématique, dialogue) pour améliorer la régénération de dialogue.

## Entraîner

Script : `scripts/training/train_vae_texte.lua`

Exemple :

```bash
./bin/mimir --lua scripts/training/train_vae_texte.lua -- \
  --dataset-root ../dataset_2 \
  --out-dir checkpoint/vae_text_trained \
  --epochs 5 --lr 1e-4 \
  --seq-len 256 --d-model 256 --latent-tokens 32 \
  --kl-beta 0.01 \
  --decoder-causal true \
  --enable-conditional-encoder true \
  --enable-context-heads true \
  --context-semantic-dim 64 \
  --context-thematic-dim 32 \
  --context-dialog-dim 64 \
  --context-semantic-weight 0.08 \
  --context-thematic-weight 0.05 \
  --context-dialog-weight 0.10
```

Points importants :

- Utilise un “base tokenizer” commun (`scripts/modules/base_tokenizer.lua`) et `tokenizer_frozen=true`.
- Met `cfg.checkpoint_dir = out_dir` pour permettre des checkpoints d’interruption Ctrl+C.
- `decoder_causal=true` est recommandé pour la génération de dialogue auto-régressive.
- Les contextes internes sont appris en auto-supervision à partir des tokens d’entrée.

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
