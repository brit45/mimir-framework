# Internals du tokenizer et de l’encodeur

Comprendre le fonctionnement interne exact des composants runtime.

**Public concerné :** Développeur avancé qui modifie le moteur C/C++.

> **Prérequis**
>
> Connaître les bases C++ et la structure du dépôt.


Cette page documente les composants NLP côté C++ : le `Tokenizer`, le `ConditioningEncoder`, et comment ils s’intègrent au `Model` (conventions `mag/mod`, ids int vs floats).

Source de vérité :

- Tokenizer : `src/Tokenizer.hpp`, `src/Tokenizer.cpp`
- ConditioningEncoder : `src/Encoder.hpp`, `src/Encoder.cpp`
- Runtime forward tokens : `src/Model.cpp` (`forwardPassNamedView`)
- Corpus causal : `src/Models/NLP/CausalLMModel.hpp` et `.cpp`
- Bindings Lua : `src/scriptings/Lua/luaScripting/LuaScripting.cpp` et `src/scriptings/Lua/luaScripting/LuaScripting.hpp`

## 1) Tokenizer : responsabilités

`Tokenizer` (dans `src/Tokenizer.hpp`) est responsable de :

- construire/maintenir un vocab (`vocab`, `reverse_vocab`),
- tokeniser/dé-tokeniser (`tokenize`, `decode`),
- sérialiser (`to_json`, `from_json`),
- gérer les IDs spéciaux (pad/unk/bos/eos/seq/mag/mod) et les tokens de
  contrôle enregistrés avec `addSpecialToken`,
- fournir des utilitaires (split, normalisation, accents, ponctuation),
- optionnel : apprentissage BPE (`learnBPEFromCorpus`) et tokenisation BPE (`tokenizeBPE`).

Invariants typiques :

- le vocab doit contenir les tokens spéciaux (`ensureSpecialTokens`).
- `maxSequenceLength` borne certaines fonctions de padding/batching.

## 2) ConditioningEncoder : conventions `mag` / `mod`

Le runtime mentionne une convention :

- l’encoder peut fournir deux embeddings float `mag` et `mod`.
- le forward (path ids int) injecte ces tenseurs dans le store float si le graphe les utilise (détection par `Layer.inputs`).

Points importants :

- si la config contient `d_model` ou `text_d_model`, le runtime vérifie que `mag/mod` respectent la dimension attendue.

## 3) Entrées entières nommées et compatibilité

Le chemin courant `forwardPassNamedView(float_inputs, int_inputs, training)`
injecte les deux dictionnaires dans leurs stores, puis exécute le graphe
complet par le forward principal. `Embedding` peut consommer le store entier ;
les opérations suivantes restent des calculs float routés normalement.

La surcharge historique `forwardPassView(const std::vector<int>&)` convertit
les valeurs en floats avant délégation. Elle ne constitue pas à elle seule
un store entier distinct. Pour conserver explicitement les IDs entiers,
utiliser le dictionnaire `int_inputs` avec le nom attendu par le modèle :
`__input__` pour `causal_lm`, `text_ids` pour les architectures qui le déclarent.

## 4) Préparation BPE du modèle causal

`CausalLMModel::prepareTrainingCorpus` possède la préparation du corpus et du
tokenizer. Le script d'entraînement configure l'appel ; il ne doit pas relancer
une seconde étude BPE indépendante. Si un tokenizer chargé couvre le corpus
sans `<UNK>`, l'étude est réutilisée. En l'absence de tokenizer ou de couverture,
le modèle appelle `learnBPEFromCorpus`. En reprise stricte, une modification du
vocabulaire du checkpoint est refusée.

La capacité demandée et le vocabulaire effectivement appris sont distincts :
les merges ne remplissent pas nécessairement la capacité maximale. Le modèle
dimensionne ses embeddings et sa sortie d'après le vocabulaire effectif.
`seq_len` borne les fenêtres d'entraînement ; la génération dimensionne son
graphe au prompt complet et au budget de sortie. Voir
[les capacités et limites du modèle causal](../05-Advanced/03-LLM-Readiness.md).

## 5) Sérialisation

Le Tokenizer expose une sérialisation JSON (voir `to_json/from_json`). Lors d’un checkpoint, le writer peut décider d’inclure le tokenizer selon `SaveOptions.save_tokenizer`.

## 6) Tests utiles

- Tokenizer : `FrameworkTest.TokenizerBasics`,
  `FrameworkTest.TokenizerBPE` et `FrameworkTest.TokenizerJsonControlChars`.
- API Lua : `LuaTest.ApiBasics`, `LuaTest.CreateFromConfig` et
  `LuaTest.CreateFromConfigErrors`.
- Corpus/tokenizer causal : `LuaTest.CausalLMTokenizer` et
  `LuaTest.CausalLMTrainingSmoke`.

```bash
ctest --test-dir build --output-on-failure \
  -R 'FrameworkTest.Tokenizer|LuaTest'
```

## Étapes suivantes

- [Page précédente : Internals : sérialisation (SafeTensors / RawFolder / DebugJson)](15-Serialization-Internals.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Internals : bindings Lua (`LuaScripting` / `LuaContext`)](17-Lua-Bindings-Internals.md)
