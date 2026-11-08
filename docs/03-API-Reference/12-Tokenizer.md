# `Mimir.Tokenizer`

Trouver rapidement le contrat API réel et les paramètres utilisables.

**Public concerné :** Développeur et utilisateur intermédiaire/avancé.

> **Prérequis**
>
> Connaître les commandes de base de Mímir.


Sources : enregistrement dans `src/scriptings/Lua/luaScripting/LuaScripting.cpp`,
bindings dans `LuaScriptingTokenizerDataset.cpp` et `LuaScriptingRuntimeAndViz.cpp`,
algorithmes dans `src/Tokenizer.cpp`.

## Création

- `create(max_vocab: int) -> true` : remplace le tokenizer courant.
- `load(path: string) -> true | (false, err)` : charge son vocabulaire et ses merges
  depuis JSON, sans nécessiter un appel préalable à `create()`.

## Tokenisation

- `tokenize(text: string) -> table<int> | (nil, err)`
- `detokenize(ids: table<int>) -> string | (nil, err)`

`tokenize()` et `tokenize_bpe()` utilisent tous deux `tokenizeBPE` : ils appliquent
les merges appris sans agrandir le vocabulaire. Les symboles non représentables
retombent sur `<UNK>`. Les séquences sont entourées de `<BOS>` et `<EOS>` quand ces
deux tokens existent. Le tokenizer normalise le texte ; ce n'est pas un encodeur
d'octets réversible à l'identique.

## Vocab

- `vocab_size() -> int` : taille réellement occupée.
- `get_max_vocab() -> int`, `set_max_vocab(n: int) -> bool` : capacité du vocabulaire.
- `add_token(token: string) -> int` : identifiant existant ou ajouté ; retourne
  l'identifiant `<UNK>` lorsque la capacité est atteinte.
- `ensure_vocab_from_text(text: string) -> bool` : ajoute les mots normalisés.
- `tokenize_ensure(text: string) -> table<int> | (nil, err)` : tokenisation par mots
  avec ajout au vocabulaire, distincte de la segmentation BPE.
- `get_token_by_id(id: int) -> string` : `<UNK>` si l'identifiant est inconnu.

Ces ajouts ne redimensionnent pas les poids d'embedding ou la tête de sortie du
modèle. Préparez le vocabulaire avant d'allouer un nouveau modèle ; en inférence
et en reprise, gardez les identifiants et la topologie du checkpoint.

## Tokens spéciaux

- `pad_id()`, `unk_id()`, `seq_id()`, `mod_id()`, `mag_id()` retournent un identifiant.
- `<BOS>` et `<EOS>` sont des tokens natifs, mais `bos_id()` et `eos_id()` ne sont
  pas enregistrés dans l'API Lua actuelle. Le modèle causal gère leur usage dans
  ses méthodes de préparation et de génération.

## BPE

- `learn_bpe(texts: table<string>, num_merges?: int = 1000) -> bool`
- `bpe_merge_count() -> int`
- `tokenize_bpe(text: string) -> table<int> | (nil, err)`

`learn_bpe` exige une table de textes, pas un chemin de fichier. Il apprend au
plus le nombre de nouvelles fusions demandé, sous réserve de la capacité et
des paires disponibles ; ce nombre ne représente pas la taille finale du
vocabulaire. La préparation native du modèle causal sait réutiliser un tokenizer
déjà étudié. Appeler directement `learn_bpe()` demande explicitement une étude.

## Séquences

- `set_max_length(n: int) -> bool`
- `pad_sequence(ids: table<int>, seq_len?: int) -> table<int> | (nil, err)`
- `batch_tokenize(texts: table<string>, max_len?: int = 512) -> table<table<int>> | (nil, err)`

`pad_sequence` utilise la longueur configurée si l'argument est absent. Il
tronque le contenu au besoin et place les `<PAD>` entre `<BOS>` et le `<EOS>`
final lorsque la longueur permet ces deux balises. `batch_tokenize` appelle
`tokenizeEnsure` puis ce padding : il peut donc ajouter des mots et ne constitue
pas un batch BPE figé.

## Analyse auxiliaire

- `get_frequencies(text: string) -> table<string, int> | (nil, err)`
- `extract_keywords(text: string, top_n?: int = 5) -> table<string> | (nil, err)`
- `analyze_text(text: string) -> table | (nil, err)` : champs `entities`,
  `modifiers`, `actions`, `main_subject`, `context`, `complexity`.
- `print_stats() -> bool`

Ces analyses utilisent des règles et dictionnaires statiques, principalement
anglais. Ce sont des aides lexicales, pas une mesure de la compréhension apprise
par le réseau. Le binding `analyze_text` n'expose pas les champs C++ `sentiment`
et `relations`.

## I/O

- `save(path: string) -> true | (false, err)` : crée aussi les dossiers parents.
- `load(path: string) -> true | (false, err)`.

## Étapes suivantes

- [Page précédente : API : `Mimir.Architectures`](11-Architectures.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : API : `Mimir.Dataset`](13-Dataset.md)
