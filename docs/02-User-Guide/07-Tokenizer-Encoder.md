# Tokenizer et encodeur de conditionnement

Comprendre le rôle du tokenizer et de l'encoder dans les workflows texte.

**Public concerné :** Débutant NLP à intermédiaire.

> **Prérequis**
>
> Savoir ce qu'est un vocabulaire de tokens.

## Rôle des composants

- `Tokenizer` : texte ↔ ids.
- `ConditioningEncoder` : embeddings (vecteurs) associés au tokenizer/vocab.

## Synchronisation modèle

Le contexte Lua peut transmettre son tokenizer et son `ConditioningEncoder`
au modèle lors de `Mimir.Model.create(...)`. La classe `Model` possède des
assets par défaut ; chaque architecture peut toutefois déclarer qu'elle
n'utilise pas cet encodeur. Sa dimension et son rôle dépendent du builder
natif : `d_model` ne constitue pas une règle universelle de synchronisation.

Un layer `Embedding` interne au graphe et le `ConditioningEncoder` du contexte
sont deux composants distincts. `causal_lm`, par exemple, apprend son embedding
de tokens dans son graphe. Charger un encodeur externe n'ajoute pas
automatiquement ce conditionnement à toutes les architectures.

## Tokenizer gelé vs composé

Des scripts utilisent :

- `tokenizer_frozen = true` : on n’agrandit pas le vocab en lisant le dataset.
- `tokenizer_frozen = false` : le vocab peut être “composé” depuis le dataset (`tokenizeEnsure`).

Recommandation :

- Entraîner des briques réutilisables (VAEText, etc.) avec un **base tokenizer stable** et `tokenizer_frozen=true`.

Ces champs sont interprétés par les workflows concernés ; ils ne gèlent pas
globalement tous les appels de l'API tokenizer. En reprise, utilisez le
tokenizer du checkpoint et conservez les IDs appris. Pour le BPE, la couverture
par sous-mots suffit : un mot nouveau ne nécessite pas toujours un nouvel ID.
Le [guide causal](12-Transformer-GPT.md#tokenizer) détaille sa composition
native et sa réutilisation sans seconde étude.

## Fonctions pratiques

Voir la référence API :

- `Mimir.Tokenizer.tokenize`, `tokenize_ensure`
- `pad_sequence`, `vocab_size`, `save/load`
- BPE: `learn_bpe`, `tokenize_bpe`

## Étapes suivantes

- [Page précédente : Scripting Lua](06-Lua-Scripting.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Checkpoints / reprise d’entraînement](08-Checkpoints.md)
