# API : `Mimir.Dataset`

Source : `src/LuaScripting.cpp`.

## `load(dir: string) -> (bool, n|err)`

Charge un dataset depuis un dossier.

Les items peuvent contenir :

- `text_file`, `image_file`, `audio_file`, `video_file`
- `width`, `height`
- `text` (si chargé)

## `get(index: int) -> table | (nil, err)`

Retourne les métadonnées et éventuellement le texte.

## `prepare_sequences(seq_len: int) -> (bool, n|err)`

Prépare des séquences tokenisées/paddées à longueur fixe.

Notes :

- Nécessite un tokenizer courant.
- Ce chemin est utile pour certains scripts legacy.
