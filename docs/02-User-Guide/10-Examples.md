# Scripts d’exemples

Les scripts sont principalement dans `scripts/templates/`, `scripts/tests/` et `scripts/training/`.

Cette page sert de “table des matières pratique”: quoi lancer, dans quel ordre, et avec quels pré-requis.

## Exemples (démos courtes)

- `scripts/templates/template_new_model.lua` : modèle minimal “de référence” (create/build/alloc/init + bonnes pratiques mémoire).
- `scripts/tests/test_list_archi_conf.lua` : lister les archis + configs par défaut.
- `scripts/tests/test_serialization_smoke.lua` : save/load SafeTensors (smoke).
- `scripts/tests/test_vae_conv_generate.lua` : génération VAE Conv (smoke).

### Ordre conseillé (si tu ne sais pas par où commencer)

1) `scripts/templates/template_new_model.lua` (valide MemoryGuard/Allocator + create/build/alloc/init)
2) `scripts/tests/test_serialization_smoke.lua` (valide save/load SafeTensors)
3) `scripts/tests/test_vae_conv_generate.lua` (valide un chemin “vision” simple)

## Entraînement (scripts/training)

- `scripts/training/train_vae_conv.lua` : VAE conv (image).
- `scripts/training/train_vae_texte.lua` : VAEText (token-level, CE/KL).
- `scripts/training/ponyxl_ddpm_train.lua` : pipeline d’entraînement PonyXL (DDPM-like).

## Comment les lancer

```bash
./bin/mimir --lua scripts/tests/test_serialization_smoke.lua
./bin/mimir --lua scripts/training/ponyxl_ddpm_train.lua -- --help
```

Notes:

- Tout ce qui suit `--` est passé au script Lua (pattern utilisé par certains scripts de training).
- Beaucoup de scripts configurent la mémoire au début (MemoryGuard + Allocator). Si tu enlèves ces lignes, tu risques des OOM.

## Variables d’environnement utiles (selon les scripts)

Certains scripts lisent des variables d’environnement (optionnel). La référence stable reste les arguments `--...` passés après `--`.

Recommandation : commencer par `template_new_model.lua` puis `test_serialization_smoke.lua`.
