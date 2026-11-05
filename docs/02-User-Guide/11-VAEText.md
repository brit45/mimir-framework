# VAEText

Entraîner et évaluer VAEText avec les bonnes options.

**Public concerné :** Intermédiaire guidé.

> **Prérequis**
>
> Dataset texte disponible.


VAEText est un VAE “texte” qui reconstruit des tokens via logits, entraîné avec reconstruction (Cross-Entropy) + KL.

Version actuelle: VAEText peut aussi exposer un encodeur conditionnel de corrélation textuelle,
et des têtes de contexte internes (sémantique, thématique, dialogue) pour améliorer la régénération de dialogue.

## Contexte et contenu du texte

Les nouveaux entraînements activent `contextual_encoding=true` (version 2).
L'encodeur combine des positions apprises et une attention bidirectionnelle,
indépendante du choix causal du décodeur. L'ordre des mots peut ainsi modifier
le latent, même lorsque le vocabulaire de deux descriptions est identique.

Pendant l'entraînement, `--text-mask-ratio 0.15` remplace une partie des tokens
par le token inconnu, sans modifier les cibles de reconstruction ni les PAD.
Au moins un token non PAD reste visible. La corruption est déterminée par la
seed et l'étape d'optimisation ; elle n'est pas appliquée en inférence.
`--content-weight 0.1` ajoute une cross-entropy sur la distribution des tokens
présents dans le texte propre, prédite directement depuis `mu`. Cet objectif
supervise le latent que Lumen consomme, sans dépendre des têtes auxiliaires.

Les anciennes têtes « semantic », « thematic » et « dialog » restent des
prédicteurs d'empreintes lexicales/bigrammes ; leurs noms ne prouvent pas une
compréhension sémantique. Dans la version 2, elles passent par la reconstruction
latente et ne lisent plus directement le texte cible. Les nouveaux objectifs
améliorent l'apprentissage contextuel ; ils n'ajoutent pas un moteur de
raisonnement ni une connaissance du monde. La qualité sémantique doit être
évaluée après entraînement sur les données réelles.

Les tips Viz identifient les embeddings, positions, blocs de contexte, `mu`,
log-variance, reconstruction et prédiction du vocabulaire. La vignette
« Texte / entrée et reconstruction » affiche le texte propre, l'entrée masquée,
la reconstruction et l'accord token par token (vert/rouge). L'aperçu est limité
à 128 tokens. Les logs exposent `contenu_ce` et `tokens_masqués`. Les captures sont
transmises à Viz dès le premier pas, puis à la cadence `viz_taps_every_steps`
(ou `viz_every_steps`, sinon `log_every`). Le script réserve 64 vignettes par
défaut et privilégie les tips enregistrés ; `--viz-registered-tips-only false`
permet de capturer aussi les couches intermédiaires non enregistrées.

Pour un nouvel entraînement destiné à Lumen, ajouter ou conserver :

```bash
--contextual-encoding true --encoder-causal false \
--text-mask-ratio 0.15 --content-weight 0.1
```

La reprise garde la topologie du checkpoint. Pour bénéficier des positions et
de la tête de contenu avec un ancien modèle, démarrer dans un nouveau dossier
avec `--resume false`. Les anciens checkpoints restent sur leur graphe historique.

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

## Conditionnement textuel de Lumen

Préparer un dossier de fichiers `.txt` contenant des phrases, descriptions et paragraphes
naturels correspondant au domaine visuel ciblé. Les annotations techniques ne remplacent
pas ce corpus. Chaque passage reconstruit ses propres tokens ; pour les longs fichiers,
une fenêtre de `seq_len` tokens est tirée à chaque visite, plutôt que de toujours
réutiliser le début du document. Aucun corpus n'est téléchargé automatiquement.

```bash
./run_mimir.sh --lua scripts/training/train_vae_texte.lua -- \
  --context-dir /chemin/contextes_naturels \
  --out-dir checkpoint/vae_text_trained \
  --base-tokenizer checkpoint/base_tokenizer/tokenizer.json \
  --seq-len 256 --d-model 256 --latent-tokens 32 --epochs 5

./run_mimir.sh --lua scripts/training/train_lumen_diffusion.lua -- \
  --dataset /chemin/images_et_captions \
  --vae-checkpoint checkpoint/vae_conv_trained \
  --text-vae-checkpoint checkpoint/vae_text_trained \
  --out checkpoints/lumen_diffusion --epochs 10
```

Le tokenizer du VAEText est conservé sans ajout de vocabulaire par Lumen.
L'encodeur et sa projection `mu` sont gelés ; les tokens latents déterministes
alimentent une projection entraînable puis les attentions croisées du DiT.
Les poids importés sont inclus dans les checkpoints Lumen.
La longueur de texte est celle du VAEText : les prompts plus longs sont tronqués.
Conserver les checkpoints sources pour reconstruire le modèle, comme pour le VAE image.
La reprise d'un ancien Lumen conserve son encodeur historique ; pour un nouvel
entraînement avec cet encodeur, utiliser explicitement `--text-conditioning transformer`.
`--natural-context-windows false` rétablit la sélection du début des textes du VAEText.

Lumen peut également être conditionné par l'encodeur T5 natif. Les deux
encodeurs externes sont mutuellement exclusifs ; voir le
[guide T5 pour Lumen](16-T5.md#conditionnement-de-lumen) pour le workflow
T5 et ses limites.

## Journal CSV des métriques

Utiliser un chemin de fichier explicite, par exemple :

```bash
--csv-path checkpoint/vae_text/loss_history.csv
```

Si le moniteur reçoit un dossier existant (ou un chemin terminé par `/`),
il utilise automatiquement `loss_history.csv` dans ce dossier. L'historique
existant est relu et les nouvelles mesures y sont ajoutées.

## Décodage et sampling

Le dépôt courant ne fournit pas de script `vae_text_sample.lua` prêt à l’emploi.
Le registre fournit toutefois `vae_text_decode`, qui transforme un latent en logits
de taille `seq_len × vocab_size`. Un sampler doit ensuite appliquer argmax,
température ou top-k sur ces logits, puis convertir les ids avec le même tokenizer
que celui du checkpoint.

Avant d’écrire ce workflow, inspecte les deux graphes :

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- \
  --list vae_text --params --layers

./bin/mimir --lua scripts/tools/inspect_architectures.lua -- \
  --list vae_text_decode --params --layers
```

## Compat checkpoint

Tout sampler externe doit lire la configuration du checkpoint et refuser les
overrides incompatibles de `seq_len`, `vocab_size`, `d_model`, `latent_tokens` ou
tokenizer.
C’est volontaire : changer `seq_len` après entraînement casse souvent les shapes.

## Étapes suivantes

- [Page précédente : Scripts et outils Lua](10-Examples.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Tutoriel : Transformer causal (GPT-style)](12-Transformer-GPT.md)
