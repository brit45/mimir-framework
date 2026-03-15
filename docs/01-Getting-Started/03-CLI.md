# CLI (binaire `mimir`)

Le binaire principal est `bin/mimir`.

## Usage

```bash
./bin/mimir --help
```

Options :

- `--lua <script.lua>` : exécute un script Lua.
- `--config <config.json>` : charge une config JSON et crée un modèle via le registre (chemin “starter”).

## Exécuter un script Lua avec arguments

Le programme injecte la table globale `arg` (comme l’interpréteur Lua) et aussi `Mimir.Args`.

Exemple :

```bash
./bin/mimir --lua scripts/examples/vae_text_sample.lua -- --ckpt checkpoint/_smoke_vae_text_ce --mode prior
```

Notes :

- Le séparateur `--` est un usage pratique pour “séparer” les args du binaire de ceux du script. Le binaire ne le consomme pas explicitement ; certains scripts l’acceptent/ignorent.
- Dans un script, parsage recommandé : `local Args = dofile("scripts/modules/args.lua")` puis `Args.parse(arg)`.

## Sortie au démarrage

Au lancement, Mímir affiche :

- tests d’intégrité mémoire
- capacités CPU (AVX2/FMA/F16C/BMI2)
- configuration OpenMP

Ces logs sont utiles pour diagnostiquer un build “lent” ou une machine non compatible AVX.
