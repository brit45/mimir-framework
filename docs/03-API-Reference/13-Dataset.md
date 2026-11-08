# `Mimir.Dataset`

Trouver rapidement le contrat API réel et les paramètres utilisables.

**Public concerné :** Développeur et utilisateur intermédiaire/avancé.

> **Prérequis**
>
> Connaître les commandes de base de Mímir.


Source : `src/scriptings/Lua/luaScripting/LuaScriptingTokenizerDataset.cpp`.

## Chargement

```text
Mimir.Dataset.load(dir, target_w=64, target_h=64, min_modalities=1,
                   use_cache=false, cache_path="dataset_cache.json",
                   max_ram_mb=10240, lazy_loading=true) -> (true, n) | (false, err)
```

Charge un dataset multimodal depuis un dossier. `target_w` et `target_h`
définissent les dimensions du décodage d'image. Les options de cache s'appliquent
lorsque `use_cache=true` ; `max_ram_mb` est exprimé en Mio.

Le builder `Mimir.Database` prépare le même chargement avec cache :

```lua
-- Le dossier doit déjà contenir vos données.
local loader = Mimir.Database.load("dataset/mes_images", 64, 64, 1)
local ok, count = loader.cache("dataset_cache.json", 1024, true)
assert(ok, count)
```

`Database.load(...)` seul ne charge rien : c'est l'appel `cache(...)` qui remplit
le dataset courant.

Les items peuvent contenir :

- `text_file`, `image_file`, `mask_file`, `audio_file`, `video_file` ;
- `text`, `label_candidates`, `annotation_file`, `annotation` selon les données ;
- `width`, `height`, `channels` pour les images.

## `get(index: int, load_image?: bool = true) -> table | (nil, err)`

Les indices commencent à 1. Retourne les métadonnées et le texte disponible.
Avec `load_image=true`, ajoute `image`, un tableau aplati RGB d'octets dans
`[0,255]`, et `mask` si un masque est associé. Les images animées exposent aussi
`is_animated`, `frame_count`, `loop_count`, `frame_delays_ms` et `animation_frames`
lorsqu'elles ont été décodées. `get(index, false)` évite le décodage de l'image
et du masque, tout en conservant l'accès aux chemins et au texte.

`release(index: int) -> true | (false, err)` décharge les contenus de l'item tout
en gardant son entrée dans le dataset.

## `prepare_sequences(seq_len: int) -> (bool, n|err)`

Recharge le dossier courant et prépare des séquences par mots avec padding ou
troncature à longueur fixe. Ce chemin utilise `Tokenizer::tokenize`, pas BPE.

Notes :

- Sans tokenizer, il peut retourner un succès avec zéro séquence.
- Ce chemin est conservé pour les scripts legacy. Pour le causal, utiliser
  `Model.causal_lm_prepare_corpus`, qui possède sa préparation, son BPE et ses
  masques de cibles.

## Étapes suivantes

- [Page précédente : API : `Mimir.Tokenizer`](12-Tokenizer.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : API : mémoire](14-Memory.md)
