# LLM readiness (état réel)

Cette page décrit ce qui est **effectivement** en place dans le codebase pour un LLM, et ce qui manque pour une génération utile au quotidien.

Point d’entrée conseillé : [04-Source-Code-Map.md](04-Source-Code-Map.md).

## 1) Ce qui existe déjà

- Une architecture Transformer côté registry (builder) : `src/Models/NLP/TransformerModel.cpp`.
- Un chemin “tokens int” (Embedding lit dans un store d’ints) : ex. conventions `__input__` côté NLP (voir aussi la carte du code source).
- Des kernels CPU pour attention et matmul (OpenMP/SIMD selon build) : `src/LayerOps.hpp`, `src/SIMD_Ops.hpp`.
- Une API tokenizer/encoder en Lua : `Mimir.Tokenizer.*` et `Mimir.Model.encode_prompt(...)` (bindings dans `src/LuaScripting.cpp`).

## 2) Ce que `Mimir.Model.infer()` fait vraiment (important)

`Mimir.Model.infer(prompt)` est présent, mais aujourd’hui ce n’est pas une “generation” autoregressive complète :

- côté C++ : `src/LuaScripting.cpp` tokenize une string si un tokenizer existe, encode si un encoder existe, puis appelle `Model::forward(output)`.
- la “décode” ensuite via `Model::eval(...)`.

Or `Model::eval(...)` (voir `src/Model.cpp`) produit actuellement :

- un MSE trivial sur un `target` binaire,
- puis des logits uniformes + des tokens top-k “factices” (0..7) si un tokenizer est présent.

Conclusion : **ne pas** considérer `infer()` comme une génération LLM valide pour l’instant.

## 3) Limites actuelles pour un LLM “pratique”

- Pas de KV-cache dédié : une génération token-par-token coûterait un forward complet à chaque token.
- Pas d’API runtime de sampling (top-k/top-p/temperature) côté C++ ; si vous voulez sampler, ça doit être implémenté côté scripts Lua pour l’instant.
- Offload GPU : l’offload Vulkan/OpenCL est ciblé sur `Linear` en inférence, pas une stack LLM complète (voir `src/Model.cpp`).

## 4) Roadmap minimale (concrète)

Pour passer de “Transformer entraînable” à “LLM utilisable” :

1) Exposer des logits utilisables (dernier layer / projection vocab) via une API stable.
2) Ajouter une API `prefill` + `decode` avec KV-cache (structures + sérialisation).
3) Ajouter un module de sampling (au moins greedy + top-k) et une boucle de génération.
4) Optimiser attention/matmul et/ou étendre l’offload.

## 5) Où regarder dans le code

- Registry + config merge : `src/Models/Registry/ModelArchitectures.*`.
- Builder Transformer : `src/Models/NLP/TransformerModel.cpp`.
- Ops d’attention : `src/LayerOps.hpp`.
- Bindings Lua (forward, infer, encode prompt) : `src/LuaScripting.cpp`.

Et côté scripts (utile pour comprendre l’intention et l’usage actuel) :

- `scripts/examples/example_gpt.lua` : montre la création d’un Transformer causal via registry, l’allocation des params et un appel à `Mimir.Model.infer(...)`.
- `scripts/training/train_llm.lua` : exemple d’entraînement “LLM-like” (mais sans API de génération autoregressive côté runtime).
