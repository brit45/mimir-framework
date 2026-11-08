# Première exécution

Valider rapidement le chemin minimal: build, script Lua, création de modèle.

**Public concerné :** Débutant à intermédiaire.

> **Prérequis**
>
> Le dépôt est cloné localement.

Objectif : exécuter un script Lua, créer un modèle via le registre, initialiser les poids, faire un forward.

Si vous ne savez pas par où commencer, suis cette page de haut en bas sans improviser. Elle couvre le plus petit chemin utile pour vérifier que le runtime, l’API Lua et le registre d’architectures fonctionnent ensemble.

## Plan de progression

1. Vérifier que le binaire démarre.
2. Vérifier qu'un script Lua s'exécute.
3. Vérifier qu'un modèle peut être créé et exécuté.
4. Vérifier que la sérialisation fonctionne.

Temps moyen : 10 à 15 minutes.

## Pré-requis

- CMake, compilateur C++17 et Lua 5.3+ avec ses en-têtes.
- Un shell POSIX pour les commandes ci-dessous ; voir l'[installation](02-Installation.md) pour Windows.

## Démarrage rapide avec Docker (Alpine)

Si Docker Engine et le plugin Compose sont installés, le conteneur évite
d'installer localement les dépendances de compilation. Depuis la racine du dépôt,
construire l'image une fois :

```bash
docker compose build
```

Le service lance toujours Mímir via `run_mimir.sh`, qui applique la
configuration runtime. Le dépôt est monté dans `/workspace` pour fournir les
scripts et conserver les fichiers produits.
Choisir l'une des deux façons de désigner le script :

```bash
# Script Lua donné en argument
docker compose run --rm mimir \
  --lua scripts/templates/template_new_model.lua

# Ou script choisi avec SCRIPT, avec les arguments Lua après --
SCRIPT=scripts/benchmarks/benchmark_nms.lua \
  docker compose run --rm mimir -- --quick
```

Les scripts, données et sorties restent sur la machine hôte. Le conteneur
s'exécute avec UID/GID `1000` par défaut ; les adapter à l'utilisateur courant
pour éviter des fichiers appartenant à un autre utilisateur :

```bash
MIMIR_UID="$(id -u)" MIMIR_GID="$(id -g)" \
  docker compose run --rm mimir --version
```

La variable `SCRIPT` et l'option CLI sont deux modes alternatifs : ne pas
combiner `SCRIPT` avec un autre `--lua`.

## 1) Build

```bash
cmake -S . -B build -DMIMIR_VIZ_BACKEND=NONE \
  -DENABLE_VULKAN=OFF -DENABLE_OPENCL=OFF -DMIMIR_ENABLE_TESTS=ON
cmake --build build --parallel 2
```

Si le build passe mais que vous voulez un contrôle rapide avant de lire le reste de la doc :

```bash
./bin/mimir --help
```

Vous devez obtenir l’aide CLI et la détection matérielle au démarrage, sans crash immédiat.

Si la commande échoue :

1. Vérifier que vous êtes bien à la racine du dépôt.
2. Lire la première erreur et le résumé CMake pour identifier la dépendance manquante.
3. Si un autre générateur ou toolchain est nécessaire, utiliser un nouveau dossier de build.

## 2) Exécuter un script Lua

```bash
./bin/mimir --lua scripts/examples/inspect_vae_conv.lua
```

Cet exemple vérifie le câblage CLI → Lua, le registre et l’allocation d’un modèle sans charger de dataset. Pour une validation plus large :

```bash
./bin/mimir --lua scripts/tests/test_serialization_smoke.lua
```

L'inspection vérifie la création d'un petit VAE natif ; le smoke test vérifie
la sauvegarde et le rechargement. Ces contrôles dépassent le simple démarrage
du binaire, sans valider tous les modèles ni tous les backends.

Attendu :

1. Pas de crash.
2. Messages de run cohérents.
3. Retour shell à 0.

## 3) Créer un modèle (via le registre)

Enregistrez cet exemple dans `/tmp/mimir_quickstart.lua`, puis lancez-le avec
`./run_mimir.sh --lua /tmp/mimir_quickstart.lua`. Il configure un Transformer
natif du registre, sans définir de graphe dans Lua.

```lua
local cfg, err = Mimir.Architectures.default_config("transformer")
if not cfg then error(err) end

cfg.vocab_size = 128
cfg.seq_len = 8
cfg.d_model = 16
cfg.num_layers = 1
cfg.num_heads = 2
cfg.mlp_hidden = 32
cfg.output_dim = 4

assert(Mimir.Model.create("transformer", cfg))
assert(Mimir.Model.set_hardware(false)) -- exemple CPU
-- Model.build() n'est plus nécessaire (v3.0+: network construit automatiquement)
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier", 42))

-- TransformerModel attend des IDs entiers sur le port __input__.
local ids = {}
for i = 1, cfg.seq_len do ids[i] = 1 end
-- Les ports de conditionnement mag/mod attendent chacun d_model valeurs.
local conditioning = {}
for i = 1, cfg.d_model do conditioning[i] = 0 end
local out = assert(Mimir.Model.forward({
  __input__ = ids, mag = conditioning, mod = conditioning,
}, false))
assert(#out == cfg.output_dim)
print("out_len=", #out)
```

Pourquoi cet exemple est structuré comme ça :

- `default_config()` évite d’oublier des champs attendus par le builder.
- `create()` fixe le type et la config dans le contexte runtime, puis construit le réseau automatiquement (v3.0+).
- `allocate_params()` réserve les poids.
- `init_weights()` met le modèle dans un état exploitable avant le premier `forward()`.

Ce Transformer encode la séquence puis applique une tête dense de `output_dim`
valeurs. Ce n'est pas une prédiction du vocabulaire à chaque position ; pour ce
cas, utilisez le modèle `causal_lm` et son workflow dédié.

Si vous remplacez l’entrée par une table map nommée, vous gardez un script compatible avec les architectures multi-input. C’est la convention recommandée même pour un seul tenseur d’entrée.

Gardez ces petites dimensions pour le premier essai. Les configurations par
défaut d'autres architectures peuvent allouer beaucoup plus de mémoire.

### Variante courte avec sauvegarde

```lua
local ok, save_err = Mimir.Serialization.save("checkpoint/quickstart_model", "safetensors")
assert(ok ~= false, save_err)
print("checkpoint écrit")
```

Cette étape est utile si vous voulez valider tout de suite que la config, l’allocation et la sérialisation sont cohérentes dans votre environnement.

## Erreurs fréquentes

1. Oublier `allocate_params()` avant `init_weights()`.
2. Passer une entrée incompatible (ids int attendus vs tenseur float).
3. Croire que `Mimir.Model.build()` est obligatoire (ce n'est plus le cas en mode moderne).

## 4) Où regarder ensuite

- [Cycle de vie du modèle](../02-User-Guide/02-Model-Lifecycle.md).
- [Sérialisation](../03-API-Reference/16-Serialization.md) pour save/load.
- [Gestion mémoire](../02-User-Guide/09-Memory.md) si votre modèle dépasse la RAM disponible.

## Étapes suivantes

- [Page précédente : Démarrage rapide](00-GET-STARTED.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : 🔧 Installation & Compilation détaillée](02-Installation.md)
