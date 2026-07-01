# Construire Un Modele Et Ses Layers

## Pour qui

Développeur framework (C/C++/runtime/scripting).

## Objectif

Implémenter ou modifier des briques techniques sans casser le contrat global.

## Avant de commencer

Comprendre le registre d'architectures et les conventions I/O.

## Résultat attendu

Tu peux livrer des évolutions compatibles avec la base existante.


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

## 4. Regle source de verite: modele declare en C/C++

Dans ce framework, la declaration de topologie et le parametrage des layers se font cote C/C++ (classes de modeles + registre), pas cote Lua.

Flux normal :

1. Le registre cree le modele (`ModelArchitectures::create(...)`).
2. La classe C++ (ex: `VAEConvModel::buildInto`) pousse les layers (`model.push(...)`).
3. La classe C++ renseigne les I/O (`Layer.inputs`, `Layer.output`) et metadonnees (channels, kernel, stride, etc.).

Exemple reel a lire: `src/Models/Vision/VAEConvModel.cpp`.

## 5. Cote Lua: creation/chargement, pas definition principale du graphe

Cote script, le chemin standard est :

```lua
local cfg = Mimir.Architectures.default_config("vae_conv")
cfg.image_w = 512
cfg.image_h = 512
cfg.base_channels = 16

local ok, err = Mimir.Model.create("vae_conv", cfg)
if not ok then error(err) end
```

Note : `Mimir.Model.build()` est conserve pour compatibilite, mais en mode moderne la construction est deja faite par `Model.create(...)` via le registre.

## 6. API legacy a ne plus utiliser: set_layer_io

`Mimir.Model.set_layer_io(...)` est deprecie/obsolete et ne doit plus etre utilise dans les nouveaux scripts ni dans la documentation d'usage.

Regle projet :

- declaration des layers: C/C++,
- wiring I/O (`Layer.inputs`, `Layer.output`): C/C++,
- scripts Lua: creation via registre + execution/inspection uniquement.

Pour une architecture produit (ex: `vae_conv`, `ponyxl_ddpm`, `hf_vae_decoder`), modifier la topologie dans les fichiers C/C++ du modele et son enregistrement registre.

## 7. Erreurs frequentes

1. Type de layer invalide dans `push`.
2. `params_count` incoherent avec la structure reelle du layer.
3. Inputs non renseignes ou mal orthographies.
4. Sortie ecrasee accidentellement (meme nom `output` sur plusieurs branches).
5. Dimensions incompatibles entre layers relies.

## 8. Checklist avant commit

- topologie lisible (`name` explicites),
- I/O explicites pour les layers multi-entrees,
- shape checks passes,
- allocate/init/forward passent,
- test smoke ajoute (script court).

## 9. Demo correcte - meme intention cote C++ (source de verite)

```cpp
model.push("blk/conv_main", "Conv2d", 64 * 64 * 3 * 3);
model.push("blk/act_main", "SiLU", 0);
model.push("blk/conv_skip", "Conv2d", 64 * 64 * 1 * 1);
model.push("blk/add", "Add", 0);
```

Puis associer les I/O directement en C++ (exemple style `VAEConvModel.cpp`) :

```cpp
if (auto* l = model.getLayerByName("blk/conv_main")) {
	l->inputs = {"__input__"};
	l->output = "blk/main0";
	l->in_channels = 64;
	l->out_channels = 64;
	l->kernel_size = 3;
	l->stride = 1;
	l->padding = 1;
}
if (auto* l = model.getLayerByName("blk/act_main")) {
	l->inputs = {"blk/main0"};
	l->output = "blk/main1";
}
if (auto* l = model.getLayerByName("blk/conv_skip")) {
	l->inputs = {"__input__"};
	l->output = "blk/skip";
	l->in_channels = 64;
	l->out_channels = 64;
	l->kernel_size = 1;
	l->stride = 1;
	l->padding = 0;
}
if (auto* l = model.getLayerByName("blk/add")) {
	l->inputs = {"blk/main1", "blk/skip"};
	l->output = "x";
}
```

## 10. Demo script correcte (registre -> create)

```lua
local cfg = Mimir.Architectures.default_config("vae_conv")
cfg.image_w = 512
cfg.image_h = 512
cfg.base_channels = 16

assert(Mimir.Model.create("vae_conv", cfg))
```

Ce qui est important : la topologie vient de C/C++ (classe modele), le script fournit la config et declenche la creation.

## 11. Criteres metier de validation

Definition de done pour ce type d'assemblage :

1. Les noms d'IO restent stables entre entrainement et inference.
2. Le bloc est serialisable et rechargeable sans renommer les noeuds.
3. Les outils d'inspection affichent le graphe attendu.
4. Le bloc accepte une batch de smoke test sans NaN/shape mismatch.

## 12. Demos existantes a relire

- `scripts/templates/template_new_model.lua`
- `scripts/tests/test_vae_conv_resnet_smoke.lua`
- `scripts/tools/inspect_architectures.lua`
