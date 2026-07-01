# Construire Un Modele Et Ses Layers

Ce chapitre explique concretement comment assembler un modele, dont `model.push(...)`, le parametrage des layers, et les routes I/O.

## 1. API de base cote C++

La methode cle est :

```cpp
void Model::push(const std::string &name, const std::string &type, size_t params_count);
```

Effet : ajoute un `Layer` dans `Model::layers`.

Ce que fait `push` en pratique :

- normalise le type de layer,
- valide que le type est supporte,
- applique des champs de `modelConfig` sur le layer (in/out channels, kernel, stride, padding, etc.),
- calcule certaines dimensions de sortie (ex: Conv2d / ConvTranspose2d),
- ajoute le layer a la topologie.

## 2. Exemple minimal C++

```cpp
model.push("enc/conv_in", "Conv2d", 64 * 3 * 3 * 3);
model.push("enc/act0", "SiLU", 0);
model.push("enc/res/add", "Add", 0);
```

Regle generale :

- `params_count > 0` pour les layers parametres,
- `params_count = 0` pour les layers purement operationnels (activation, add, reshape, etc.).

## 3. Parametrer les layers

Le parametrage passe principalement par la config modele, puis est lu dans `push`.

Exemples de champs utilises par les convs :

- `in_channels`
- `out_channels`
- `height` / `width`
- `kernel`
- `stride`
- `padding`

Conseil :

- garder les noms de champs stables,
- documenter les defaults dans la config d'architecture,
- eviter les dependances implicites entre layers distants.

## 4. Renseigner les I/O nommees

Cote Lua, le bridge expose :

- `Mimir.Model.push_layer(name, type, params_count)`
- `Mimir.Model.set_layer_io(layer_name, inputs_table, output_name)`

Exemple :

```lua
Mimir.Model.push_layer("block/add", "Add", 0)
Mimir.Model.set_layer_io("block/add", {"skip", "main"}, "block/out")
```

Utilisation :

- `inputs_table` definit les dependances entrantes d'un layer,
- `output_name` definit la sortie nommee produite,
- pour les graphes simples, garder une convention de noms lisible (`enc/...`, `dec/...`, `x`, `text_ids`).

## 5. Erreurs frequentes

1. Type de layer invalide dans `push`.
2. `params_count` incoherent avec la structure reelle du layer.
3. Inputs non renseignes ou mal orthographies.
4. Sortie ecrasee accidentellement (meme nom `output` sur plusieurs branches).
5. Dimensions incompatibles entre layers relies.

## 6. Checklist avant commit

- topologie lisible (`name` explicites),
- I/O explicites pour les layers multi-entrees,
- shape checks passes,
- allocate/init/forward passent,
- test smoke ajoute (script court).

## 7. Demo complete - mini bloc avec branche skip

Objectif : montrer un cas reel de `model.push(...)` + `set_layer_io` avec une somme de branche.

```lua
-- 1) Layers
Mimir.Model.push_layer("blk/conv_main", "Conv2d", 64 * 64 * 3 * 3)
Mimir.Model.push_layer("blk/act_main", "SiLU", 0)
Mimir.Model.push_layer("blk/conv_skip", "Conv2d", 64 * 64 * 1 * 1)
Mimir.Model.push_layer("blk/add", "Add", 0)

-- 2) Routage explicite
Mimir.Model.set_layer_io("blk/conv_main", {"x"}, "blk/main0")
Mimir.Model.set_layer_io("blk/act_main", {"blk/main0"}, "blk/main1")
Mimir.Model.set_layer_io("blk/conv_skip", {"x"}, "blk/skip")
Mimir.Model.set_layer_io("blk/add", {"blk/main1", "blk/skip"}, "blk/out")
```

Ce que cette demo clarifie :

1. Les layers sont decrits separement du routage.
2. Le graphe est lisible uniquement par les noms d'IO.
3. `Add` consomme 2 flux explicites, sans ambiguite.

## 8. Demo complete - meme intention cote C++

```cpp
model.push("blk/conv_main", "Conv2d", 64 * 64 * 3 * 3);
model.push("blk/act_main", "SiLU", 0);
model.push("blk/conv_skip", "Conv2d", 64 * 64 * 1 * 1);
model.push("blk/add", "Add", 0);
```

Puis, via API d'IO du modele, associer les entrees/sorties nommees de la meme facon qu'en Lua.

## 9. Criteres metier de validation

Definition de done pour ce type d'assemblage :

1. Les noms d'IO restent stables entre entrainement et inference.
2. Le bloc est serialisable et rechargeable sans renommer les noeuds.
3. Les outils d'inspection affichent le graphe attendu.
4. Le bloc accepte une batch de smoke test sans NaN/shape mismatch.

## 10. Demos existantes a relire

- `scripts/templates/template_new_model.lua`
- `scripts/tests/test_vae_conv_resnet_smoke.lua`
- `scripts/tools/inspect_architectures.lua`
