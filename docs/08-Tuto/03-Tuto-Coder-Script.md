# Tuto - Coder un script

## Pour qui

Debutant/intermediaire qui veut piloter Mimir avec Lua.

## Objectif

Ecrire un script clair qui charge une config, construit un modele, puis lance train/inference.

## Avant de commencer

1. Connaissances Lua de base (variables, fonctions, tables).
2. Lecture conseillee: [docs/02-User-Guide/06-Lua-Scripting.md](../02-User-Guide/06-Lua-Scripting.md).
3. Avoir un template script de depart.

## Résultat attendu

Tu sais ecrire un script reutilisable avec arguments, logs, et gestion d'erreurs simple.

## Etape 1 - Partir d'un template

Templates recommandes:
- `scripts/templates/template_new_model.lua`
- `scripts/templates/template_pipeline_only.lua`
- `scripts/templates/template_pipeline_args.lua`

Regle simple: ne pars pas de zero si un template existe.

## Etape 2 - Structurer le script en blocs

Plan conseille:
1. imports/dofile modules,
2. lecture des arguments,
3. chargement config,
4. creation/build du modele,
5. train ou inference,
6. sauvegarde checkpoint,
7. logs de fin.

## Etape 3 - Ajouter des arguments utiles

Exemples d'arguments:
- `--epochs`
- `--lr`
- `--dataset`
- `--save`
- `--no-train`

Test rapide:

```bash
./bin/mimir --lua scripts/templates/template_pipeline_args.lua -- --help
```

## Etape 4 - Ajouter un chemin d'erreur propre

Toujours verifier les retours critiques:
1. creation du modele,
2. chargement config,
3. ouverture dataset,
4. save checkpoint.

Regle: un message d'erreur doit dire quoi corriger.

## Etape 5 - Mini checklist de qualite

1. Le script tourne sans modifier le code C++.
2. Le script accepte au moins 2-3 arguments utiles.
3. Les chemins de fichiers sont clairs.
4. Le run ecrit un resultat exploitable (checkpoint, metrique, log).

## Commandes utiles

Run template minimal:

```bash
./bin/mimir --lua scripts/templates/template_pipeline_only.lua
```

Run template avec arguments:

```bash
./bin/mimir --lua scripts/templates/template_pipeline_args.lua -- --no-train
```

## Exemple pratique

### Contexte

Tu veux un script court et robuste pour verifier rapidement qu'une architecture charge, se construit, puis infer correctement.

### Code commente

```lua
-- Charge le module pipeline moderne (registry-first).
local P = dofile("scripts/modules/pipeline.lua")

-- Cree un pipeline depuis le registre.
local pipe, err = P.FromRegistry("transformer")
if not pipe then
	error("Creation pipeline impossible: " .. tostring(err))
end

-- Charge la config par defaut de l'architecture.
local ok, cfg = pipe:loadDefaultConfig("transformer")
if not ok then
	error("Chargement config impossible: " .. tostring(cfg))
end

-- Applique un petit patch local pour un run rapide.
ok, err = pipe:patchConfig({ d_model = 128, num_layers = 2, seq_len = 64 })
if not ok then
	error("Patch config invalide: " .. tostring(err))
end

-- Construit le modele puis execute un mini run.
pipe:build()

-- Exemple inference-only: utile pour verifier que le pipeline tourne.
-- Remplace par pipe:train(...) quand le dataset est pret.
local out = pipe:infer({ prompt = "hello world" })
print("Inference terminee:", out and "ok" or "nil")
```

### Explication

1. verifier chaque retour `(ok, err)`,
2. garder un patch de config minimal pour debug,
3. separer clairement train et inference.

### Test rapide

```bash
./bin/mimir --lua scripts/templates/template_pipeline_only.lua
```

Verification attendue: la construction du pipeline passe et le run se termine sans erreur critique.

## Suite

- Scripting Lua: [docs/02-User-Guide/06-Lua-Scripting.md](../02-User-Guide/06-Lua-Scripting.md)
- Exemples: [docs/02-User-Guide/10-Examples.md](../02-User-Guide/10-Examples.md)
