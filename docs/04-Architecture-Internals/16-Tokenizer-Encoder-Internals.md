# Internals : Tokenizer / Encoder (C++)

Cette page documente les composants NLP côté C++ : le `Tokenizer`, l’`Encoder`, et comment ils s’intègrent au `Model` (conventions `mag/mod`, ids int vs floats).

Source de vérité :

- Tokenizer : `src/Tokenizer.hpp`, `src/Tokenizer.cpp`
- Encoder : `src/Encoder.hpp`, `src/Encoder.cpp`
- Runtime forward tokens : `src/Model.cpp` (`forwardPassView(const std::vector<int>&)`)
- Bindings Lua : `src/LuaScripting.cpp/.hpp`

## 1) Tokenizer : responsabilités

`Tokenizer` (dans `src/Tokenizer.hpp`) est responsable de :

- construire/maintenir un vocab (`vocab`, `reverse_vocab`),
- tokeniser/dé-tokeniser (`tokenize`, `decode`),
- sérialiser (`to_json`, `from_json`),
- gérer les IDs spéciaux (pad/unk/bos/eos/seq/mag/mod),
- fournir des utilitaires (split, normalisation, accents, ponctuation),
- optionnel : apprentissage BPE (`learnBPEFromCorpus`) et tokenisation BPE (`tokenizeBPE`).

Invariants typiques :

- le vocab doit contenir les tokens spéciaux (`ensureSpecialTokens`).
- `maxSequenceLength` borne certaines fonctions de padding/batching.

## 2) Encoder : conventions `mag` / `mod`

Le runtime mentionne une convention :

- l’encoder peut fournir deux embeddings float `mag` et `mod`.
- le forward (path ids int) injecte ces tenseurs dans le store float si le graphe les utilise (détection par `Layer.inputs`).

Points importants :

- si la config contient `d_model` ou `text_d_model`, le runtime vérifie que `mag/mod` respectent la dimension attendue.

## 3) Int path vs float path

Le modèle supporte deux entrées “principales” :

- float path : `forwardPassView(const std::vector<float>&)`
- int path : `forwardPassView(const std::vector<int>&)`

Dans l’int path :

- les ids sont stockés dans un store séparé (`tensor_store_int`), sous les clés `"x"` et `"__input__"`.
- seuls certains layers sont dispatchés (notamment `Embedding` et un sous-ensemble de layers float qui suivent).

Conséquence :

- une architecture NLP doit respecter cette contrainte (Embedding en tête, puis layers compatibles).

## 4) Sérialisation

Le Tokenizer expose une sérialisation JSON (voir `to_json/from_json`). Lors d’un checkpoint, le writer peut décider d’inclure le tokenizer selon `SaveOptions.save_tokenizer`.

## 5) Tests utiles

- Tokenizer : `scripts/tests/test_tokenizer.lua`
- Pipelines plus larges : `scripts/tests/test_lua_api.lua`