# Internals : Autograd + gradients + backward (C++)

Cette page documente le système de gradients et le backward pass dans Mímir.

L’objectif est pragmatique : expliquer ce qui est *vraiment* supporté, comment les gradients sont stockés, et quelles informations doivent être snapshotées pendant le forward.

Source de vérité :

- Déclarations : `src/Model.hpp` (API training/gradients/optimizer)
- Backward runtime : `src/Model.cpp` (`Model::backwardPass`, `zeroGradients`, `getGradients`)
- Helpers autograd : `src/Autograd.hpp`
- Layout poids & champs layer : `src/Layers.hpp` + `src/LayerOps.hpp`

## 1) Vue rapide (Lua → C++)

| Appel Lua | Binding C++ | Cible | Effet |
|---|---|---|---|
| `Mimir.Model.zero_gradients()` | `LuaScripting::lua_zeroGradients` | `Model::zeroGradients` | Met tous les gradients à zéro + invalide l’état forward. |
| `Mimir.Model.backward(loss_grad)` | `LuaScripting::lua_backwardPass` | `Model::backwardPass` | Backprop “best-effort” à travers les layers supportés. |
| `Mimir.Model.get_gradients()` | `LuaScripting::lua_getGradients` | `Model::getGradients` | Exporte un dictionnaire d’index→valeur (format plat). |

Notes :

- Le backward n’est pas un moteur symbolique général : c’est un backward **implémenté à la main** par type de layer, avec des branches “skip si non supporté”.
- Une partie des fonctions utilitaires (ex: MSE backward, GELU backward) vit dans `src/Autograd.hpp`.

## 2) Deux notions à distinguer

### A) Gradients “par layer” (principal)

Chaque `Layer` stocke :

- `grad_weights` : gradient du bloc de poids
- `grad_bias` : gradient du biais (si applicable)

Ces buffers sont remplis par `Model::backwardPass`.

### B) Gradients “plats” (`Gradients`)

`struct Gradients` (dans `src/Autograd.hpp`) stocke :

- `std::unordered_map<size_t, float> param_grads`

Dans le code actuel, `Model::getGradients()` reconstruit un index plat en itérant les layers et en appendant `grad_weights` puis `grad_bias`.

Conséquence :

- ce format est pratique pour Lua/serialization/debug,
- mais l’optimizer moderne du runtime travaille surtout avec les buffers des layers (ou des blocs par pointeur de paramètre).

## 3) “Forward state” : pourquoi il faut snapshot

Pour faire le backward correctement, il faut les *inputs* du forward (par layer) et parfois des masques (dropout/relu).

Dans `src/Model.cpp`, quand `training==true`, le forward capture dans une structure interne (`forward_state`) :

- `layer_input_names` : quels noms de tenseurs ont été lus (`Layer.inputs` ou `{ "x" }`).
- `layer_input_sizes_multi` : tailles des inputs (utile même sans copier les valeurs).
- `layer_inputs_multi` : copie des valeurs **uniquement** pour certains types (`needs_input_value_snapshot`).
- `layer_output_masks` : masques de sortie (Dropout) pour ne pas recopier toute la sortie.
- parfois `layer_outputs` (best-effort selon path).

**Piège** : le store de tenseurs est “par nom” (`TensorStore`). Si un nom est réutilisé, relire le store en backward peut donner des données différentes de celles du forward. D’où le snapshot.

## 4) Routing des gradients : `grad_store` (par nom)

Le backward ne propage pas uniquement un vecteur unique “dx”. Il route des gradients par nom, miroir du forward.

Pattern :

- initialisation : `grad_store["x"] = loss_gradient`
- pour chaque layer (ordre inverse) :
  - on lit `grad_out = grad_store[layer.output]`
  - on calcule les gradients vers les inputs
  - on accumule dans `grad_store[input_name]`

Accumulation = somme, avec vérification de taille.

## 5) Ce qui est supporté (exemples concrets)

### Ops sans paramètres

- `Identity`, `Reshape` : gradient recopié (si tailles compatibles).
- `Add` : support du broadcast (même logique que `LayerOps::add_forward`), avec réduction du gradient pour l’entrée “petite”.
- `Concat` : découpe du gradient en tranches.
- `Split` : recolle les gradients des sorties `base_i`.
- `Multiply`, `Subtract`, `Divide` : élément-wise (avec règles de base).

### Ops avec paramètres

- `Embedding` :
  - input snapshoté = ids stockés en float (dans le path tokens).
  - `grad_weights` accumule par index de vocab.
  - pas de gradient d’entrée float utile (ids).

- `LayerNorm` :
  - backward implémenté en re-calculant mean/var (best-effort) et en accumulant `dgamma/dbeta` si `affine`.

- `Conv2d` (au moins une partie) :
  - chemin de backward tuilé (im2col + GEMM) quand AVX2+FMA et conditions OK.
  - sinon fallback/skip selon cas.

### Attention

Une partie du backward attention est implémentée dans `Model.cpp` (fonction(s) locales). Les poids sont layoutés en blocs (`Wqkv`, `Wout`) et les gradients sont écrits dans `grad_weights`.

## 6) Invariants et garde-fous

- Si `Model::freezeParameters(true)` :
  - `backwardPass`, `zeroGradients`, `initializeWeights`, `optimizerStep` doivent refuser (exception) ou no-op.

- Si `forward_state.is_valid == false` :
  - le backward refuse (message “call forwardPass() in training mode first”).

- `grad_weights` doit avoir la même taille que `getWeightsSize()` pour éviter overflow.

## 7) Autograd.hpp : ce que c’est (et ce que ce n’est pas)

`src/Autograd.hpp` contient des briques math (ex: `mse_backward`, `gelu_backward`, `layernorm_backward`) et des structures (`ComputationGraph`, `Gradients`).

Dans le runtime actuel :

- la logique principale du backward est dans `Model::backwardPass`.
- `Autograd.hpp` sert de bibliothèque d’outils et de types (notamment `Gradients`).

## 8) Debug : comment vérifier que le backward “fait quelque chose”

- Exécuter un script de test : `scripts/tests/test_gradients.lua` ou `scripts/tests/test_params.lua`.
- Vérifier :
  - `zeroGradients()` est appelé avant le step,
  - après backward, certains `grad_weights` ne sont pas tous zéro,
  - l’optimizer step change effectivement les poids.