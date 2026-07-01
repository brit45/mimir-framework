# Contrat API Scripting Inter-Langages

## Pour qui

Développeur framework (C/C++/runtime/scripting).

## Objectif

Implémenter ou modifier des briques techniques sans casser le contrat global.

## Avant de commencer

Comprendre le registre d'architectures et les conventions I/O.

## Résultat attendu

Tu peux livrer des évolutions compatibles avec la base existante.


Ce chapitre formalise le contrat API systeme que tous les bridges de scripting doivent respecter.

## 1. Pourquoi un contrat commun

Sans contrat central, chaque langage derive avec ses propres noms globaux et comportements.

Resultat :

- scripts non portables,
- bugs subtils en migration Lua -> Python/Ruby/JS,
- maintenance plus couteuse.

Le contrat commun est defini dans `src/scriptings/ScriptingContext.hpp`.

## 2. Noms systeme canoniques

Exemples normalises :

- namespace: `Mimir`
- conf: `CONF`, `CONF_PATH`, `CONF_DIR`
- args script: `arg`
- aliases: `model`, `architectures`, `tokenizer`, `dataset`, `Memory`, `MemoryGuard`, `Allocator`, `htop`, `viz`

Ces noms ne doivent pas varier entre bridges.

## 3. Interface runtime bridge

La base polymorphe est `src/scriptings/ScriptingRuntime.hpp`.

Contrat minimal :

- `setArgs(...)`
- `loadScript(...)`
- `executeScript(...)`
- `registerAPI()`

Tout nouveau bridge (Python/Ruby/JS/...) doit implementer ce contrat.

## 4. Contexte partage

Le contexte d'execution doit reutiliser `ScriptingContext` pour :

- stocker l'etat runtime (modele courant, tokenizer, encoder),
- garantir les memes conventions de reset/nettoyage,
- fournir un logging coherent.

## 5. Integration mode conf

Quel que soit le langage, l'execution config-driven doit injecter :

- `CONF` (objet config complet),
- `CONF_PATH`,
- `CONF_DIR`,
- `arg`.

Cela assure la compatibilite des workflows.

## 6. Checklist pour un nouveau bridge

1. Heriter de `ScriptingRuntime`.
2. Creer un contexte derive de `ScriptingContext`.
3. Exposer les memes noms systeme (`kGlobal*`, `kAlias*`).
4. Garantir le meme contrat d'erreur.
5. Ajouter un smoke test equivalent au bridge Lua.

## 7. Demo metier - script portable entre bridges

Script type attendu (conceptuellement identique entre Lua/Python/etc.) :

```lua
local conf = CONF or {}
local arch = conf.arch or "vae_conv"
local cfg, err = Mimir.Architectures.default_config(arch)
if not cfg then error(err) end

print("CONF_PATH=" .. tostring(CONF_PATH))
print("CONF_DIR=" .. tostring(CONF_DIR))
```

Ce qui compte metier :

1. Le script lit la conf via les memes noms globaux.
2. Le meme script logique peut etre porte dans un autre bridge.
3. Les erreurs ont une forme exploitable en CI.

## 8. Demo metier - test de non-regression du contrat

Pour chaque bridge, verifier ces points :

1. `Mimir` est visible.
2. `arg` est injecte.
3. `CONF`, `CONF_PATH`, `CONF_DIR` sont injectes en mode `--conf`.
4. aliases (`model`, `tokenizer`, `dataset`, `MemoryGuard`) sont presents.

Exemple de commande (bridge Lua actuel) :

```bash
./bin/mimir --conf config.json
```

Si un bridge ne passe pas cette matrice, il n'est pas pret pour un workflow metier stable.

## 9. Regle d'evolution API

Pour toute evolution du contrat :

1. Ajouter avant de supprimer (deprecation progressive).
2. Documenter la transition dans la doc dev.
3. Fournir un test de compat backward.
4. Eviter tout renommage silencieux des globals/aliases.
