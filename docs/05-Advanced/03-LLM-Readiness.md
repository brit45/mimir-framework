# État de préparation aux LLM

Cette page décrit les fonctionnalités du checkout actuel pour entraîner et
utiliser un modèle causal. L'existence de ces mécanismes ne garantit pas la
qualité linguistique d'un checkpoint : celle-ci dépend des données, de la
capacité du modèle, de l'entraînement et des résultats sur un holdout.

**Public concerné :** Utilisateur intermédiaire à avancé.

## Architecture et entraînement natifs

`CausalLMModel`, sous `src/Models/NLP/`, est une classe dérivée de `Model`,
enregistrée comme `causal_lm`. Son graphe decoder-only est construit en C++ :
RMSNorm, RoPE, attention causale groupée, SwiGLU et poids embedding/LM head
partagés. La version actuelle ajoute un mélange local causal
`CausalDepthwiseConv1d` optionnel et un goulot latent contextuel optionnel.
Ces branches sont des transformations apprises, pas une preuve de compréhension.

Le modèle possède la préparation du corpus et du tokenizer, l'échantillonnage,
la loss et sa boucle native d'entraînement. Le script
`scripts/training/train_causal_lm.lua` configure le modèle, la reprise,
l'optimiseur, AMP/DDP, le monitoring et les sauvegardes. Il ne définit pas
l'architecture dans Lua.

La loss est une cross-entropy sur les positions cibles actives. Les métriques
incluent perplexité, exactitude top-1 du prochain token, nombre de tokens actifs
et marge des deux meilleurs logits. La validation périodique expose les mêmes
mesures ; `validation_source` et `validation_holdout` permettent de vérifier
si elle utilise réellement des données réservées.

## Tokenizer et frontières de séquence

`prepareTrainingCorpus(...)` charge un tokenizer disponible, prépare les
symboles du corpus puis vérifie sa couverture BPE. L'étude est lancée quand
aucun tokenizer n'est chargé ou que le vocabulaire laisse des `<UNK>` dans le
corpus ; un tokenizer déjà couvrant n'est pas réétudié à chaque lancement.
En reprise stricte, le vocabulaire du checkpoint reste gelé et un corpus non
couvert est refusé.

Les séquences utilisent `<BOS>` et `<EOS>`. Les balises de rôles et de contexte
sont traitées par les mécanismes natifs de préparation et de génération lorsque
les tokens de contrôle sont activés. Leur présence structure les exemples ;
elle ne fournit pas à elle seule un apprentissage de la logique.

`seq_len` est la fenêtre d'entraînement, également configurable par
`--train-window`. Le générateur dimensionne temporairement son graphe à partir
du prompt complet et du budget de sortie, puis restaure les dimensions.
Il ne tronque donc pas le prompt à cette fenêtre. Les limites de mémoire et
de dimensions entières demeurent ; la qualité au-delà des longueurs apprises
doit être mesurée.

## Génération et conversation

Le chemin dédié est natif :

| API Lua | Méthode C++ |
| --- | --- |
| `Mimir.Model.causal_lm_generate(prompt, options)` | `CausalLMModel::generateText` |
| `Mimir.Model.causal_lm_chat(message, memory_path, options)` | `CausalLMModel::chat` |
| `Mimir.Model.causal_lm_reset_memory(path)` | `resetConversationMemory` |
| `Mimir.Model.causal_lm_memory_size(path)` | `conversationMemorySize` |

`GenerationOptions` expose `max_new_tokens`, `temperature`, `top_k`, `top_p`,
`seed`, `stop_id`, `system_prompt` et `use_memory`. Le sampling est implémenté
en C++. L'historique de conversation peut être persisté en JSON ; il est
réinjecté comme contexte et ne modifie pas les poids.

Le script fourni charge la topologie et le tokenizer compatibles avec le
checkpoint, puis appelle ces méthodes. Exemple à exécuter après avoir produit
un checkpoint causal dans le dossier indiqué :

```bash
./run_mimir.sh --lua scripts/inferences/causal_lm_generate.lua -- \
  --checkpoint checkpoints/causal_lm \
  --prompt "Explique la différence entre une hypothèse et une preuve." \
  --max-new-tokens 128 --temperature 0.8 --top-k 40 --top-p 0.95
```

Pour la conversation, remplacer `--prompt ...` par `--interactive` ;
`--memory-file` choisit le fichier d'historique, `--no-memory` le désactive.
Ces commandes nécessitent des poids entraînés et n'ont pas été relancées dans
le cadre de la relecture documentaire.

## Limites à garder visibles

- Le binding générique `Mimir.Model.infer(prompt)` conserve un chemin historique
  `Model::forward`/`eval` avec des sorties de démonstration. Utiliser l'API
  `causal_lm_generate` pour une génération autoregressive réelle.
- La boucle `generateFromTokenContext` refait un forward du contexte complété
  à chaque token ; elle n'est pas une API incrémentale optimisée
  `prefill`/`decode` à KV-cache réutilisé.
- Les runtimes CPU, Vulkan et OpenCL possèdent des chemins pour les opérations
  causales. Les capacités CUDA/ROCm sont propres à chaque type ;
  `CausalDepthwiseConv1d`, `SwiGLU` et les primitives Primer restent en
  `HostFallback` sur ces deux backends. Une déclaration de capacité ne prouve
  pas l'exécution effective d'une forme sur un périphérique.
- Une loss en baisse ou des prédictions top-1 répétitives ne suffisent pas à
  établir la cohérence d'un texte généré. Contrôler le holdout et essayer des
  prompts absents de l'entraînement avec le checkpoint effectivement sauvegardé.

## Sources et vérifications ciblées

- `src/Models/NLP/CausalLMModel.hpp` et `.cpp` : graphe, corpus, validation,
  sampling et mémoire.
- `src/scriptings/Lua/luaScripting/LuaScriptingModelAndRegistry.cpp` : bindings.
- `scripts/training/train_causal_lm.lua` et
  `scripts/inferences/causal_lm_generate.lua` : entrées CLI.
- `scripts/tests/test_causal_lm_smoke.lua` et
  `scripts/tests/test_causal_lm_tokenizer_bootstrap.lua` : contrôles ciblés.

```bash
ctest --test-dir build --output-on-failure -R 'LuaTest\.CausalLM'
```

## Étapes suivantes

- [Carte du code source](04-Source-Code-Map.md)
- [Runtimes GPU](../04-Architecture-Internals/21-GPU-Runtimes.md)
- [Index de la documentation](../00-INDEX.md)
