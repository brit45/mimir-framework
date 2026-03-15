# Tutoriel : Transformer causal (GPT-style)

Mímir expose une architecture `transformer` configurable, incluant un mode `causal=true`.

Voir aussi:

- API `Mimir.Model`: `../03-API-Reference/10-Model.md`
- Inférence: `05-Inference.md`
- Entraînement: `04-Training.md`

## Exemple

Voir : `scripts/examples/example_gpt.lua`

Ce script :

- configure l’allocateur et MemoryGuard
- crée un tokenizer
- crée un `transformer` causal via `Mimir.Model.create("transformer", cfg)`

À noter: dans le script, la config est explicitement “GPT-2 style” (vocab/seq_len/d_model/num_layers/num_heads/MLP).

## Pipeline typique (ce que fait la démo)

1) Configure `MemoryGuard` + `Allocator` (sinon risque d’OOM)
2) Active l’accélération CPU si dispo (`Model.set_hardware(true)`)
3) Crée un tokenizer (dans la démo: `Tokenizer.create(50000)`)
4) Crée le modèle `transformer` avec `causal=true`
5) `allocate_params()` puis `init_weights()`
6) (Optionnel) charge un dataset texte et appelle `Dataset.prepare_sequences(seq_len)`
7) Lance `Model.train(epochs, lr)`
8) Sauvegarde modèle + tokenizer

## Attention (réalité runtime)

- Le script montre la **construction** d’un modèle GPT-style.
- Une génération performante “token-by-token” nécessite généralement un **KV-cache** (prefill + decode). Sans KV-cache, la génération peut être très lente (recalcul complet à chaque token).

Autre point:

- La fonction `Mimir.Model.infer(prompt)` utilisée par certaines démos est un chemin **legacy**. Pour des workflows modernes, privilégie `Model.forward()` et une boucle de génération explicitement contrôlée.

## Dataset texte

La démo GPT charge un dataset via:

```lua
local dataset_path = os.getenv("MIMIR_TEXT_DATASET") or "datasets.old/text"
Mimir.Dataset.load(dataset_path)
Mimir.Dataset.prepare_sequences(cfg.seq_len)
```

Voir `03-Data.md` pour comprendre ce que fait réellement `prepare_sequences()` (tokenizer requis, padding/troncature, séquences stockées en interne).

## Conseils de sizing (éviter l’OOM)

Les paramètres `seq_len`, `d_model`, `num_layers` font exploser:

- le nombre de paramètres
- la RAM runtime (activations, buffers, etc.)

Pour un premier smoke test, commence petit:

- `seq_len`: 64–256
- `d_model`: 128–512
- `num_layers`: 2–6
- `num_heads`: 2–8

La démo `scripts/examples/example_gpt.lua` contient déjà des messages d’erreur indiquant quoi réduire si `allocate_params()` échoue.

## Recommandation

Pour un premier LLM fonctionnel :

- réduire `d_model`, `num_layers`, `seq_len`
- valider sur un dataset minuscule (smoke test)
- instrumenter RAM + temps par forward
