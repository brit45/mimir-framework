# Données / datasets

Cette page décrit le comportement **réel** du loader de dataset actuellement exposé à Lua via `Mimir.Dataset`, et ses limitations.

Sources principales :

- API Lua: `src/LuaScripting.cpp` (`lua_loadDataset`, `lua_getDataset`, `lua_prepareSequences`)
- Indexation + lazy-loading: `src/Helpers.hpp` (`loadDataset`, `DatasetItem`, `DatasetMemoryManager`, `DatasetManager`)

## Vue d’ensemble

- `Mimir.Dataset.load(dir)` indexe récursivement un dossier et construit une liste d’items (métadonnées uniquement).
- `Mimir.Dataset.get(i)` retourne **des chemins** (et des dimensions cibles), pas les données binaires.
- `Mimir.Dataset.prepare_sequences(seq_len)` construit des séquences de tokens à partir des fichiers texte du dataset (tokenizer requis pour que des séquences soient effectivement produites).

## Format disque et règle de “linking”

L’indexation fonctionne par **nom de base** (basename) :

- Tous les fichiers sont parcourus récursivement.
- Ils sont groupés par `stem()` (nom de fichier **sans extension**).
- Un item peut donc regrouper plusieurs modalités si elles partagent le même basename.

Exemple (flat ou en sous-dossiers, peu importe pour l’indexation):

```text
dataset/
  0001.txt
  0001.png
  0002.txt
  0003.jpg
```

Ici, `0001` est “linké” (texte + image). `0002` est texte seul. `0003` est image seule.

Extensions reconnues (actuellement):

- Images: `.png .jpg .jpeg .bmp .tiff .webp`
- Texte: `.txt .md .json .csv`
- Audio: `.wav .flac .mp3 .ogg .m4a .aac`
- Vidéo: `.mp4 .mkv .avi .mov .webm .flv .ts`

### Points d’attention importants

- Collisions de basename: comme l’indexation ignore les dossiers et ne garde que `stem()`, deux fichiers `foo.txt` dans des sous-dossiers différents vont se retrouver dans **le même item**.
- Un seul fichier par modalité: si plusieurs fichiers “image” partagent le même basename (ex: `0001.jpg` et `0001.png`), le dernier rencontré remplace le précédent.
- Seuil de modalités: le loader C++ supporte `min_modalities`, mais l’API Lua actuelle utilise la valeur par défaut (voir plus bas).

## API Lua: `Mimir.Dataset`

### `Mimir.Dataset.load(dataset_dir)`

Charge (indexe) le dataset.

- Entrée: `dataset_dir` (string)
- Sortie: `(ok, n_or_err)`
  - si `ok == true`, `n_or_err` est le nombre d’items indexés
  - si `ok == false`, `n_or_err` est un message d’erreur

Remarques:

- La fonction vérifie que le dossier existe.
- Elle appelle `loadDataset(dataset_dir)` côté C++.
- **Important**: l’API Lua actuelle ne lit que le premier argument; des paramètres supplémentaires passés depuis Lua seront ignorés. En pratique, cela signifie que l’indexation utilise les valeurs par défaut C++: `target_w=64`, `target_h=64`, `min_modalities=1`.
- L’indexation est en mode “lazy”: les données (texte/image/audio/vidéo) ne sont pas chargées en RAM à ce stade.

### `Mimir.Dataset.get(index)`

Retourne un item du dataset (Lua est 1-indexed).

- Entrée: `index` (integer), doit être dans `[1, num_items]`
- Sortie: `(item)` en cas de succès, ou `(nil, err)` en cas d’erreur

Champs possibles dans `item`:

- `text_file`, `image_file`, `audio_file`, `video_file`: chemins vers les fichiers (si présents)
- `width`, `height`: dimensions **cibles** stockées dans l’item (par défaut 64x64)
- `text`: contenu texte **uniquement si déjà chargé en mémoire côté C++**

Limitation importante:

- Les données binaires (image/audio/vidéo) ne sont **pas** retournées par `get()`.
- Dans l’état actuel, `Mimir.Dataset.load()` n’effectue pas de chargement de texte, donc `item.text` est généralement absent avec un flux standard `load()` → `get()`.

### `Mimir.Dataset.prepare_sequences(seq_len)`

Prépare des séquences de tokens à partir des fichiers texte.

- Entrée: `seq_len` (integer)
- Sortie: `(ok, n_or_err)`
  - si `ok == true`, `n_or_err` est le nombre de séquences construites
  - si `ok == false`, `n_or_err` est un message d’erreur

Comportement:

- Nécessite qu’un dataset ait été chargé au préalable (via la config interne `dataset.dir`).
- Ré-indexe le dataset depuis `dataset.dir`.
- Pour chaque item: charge le texte à la demande (`DatasetItem::loadText()`), puis tokenise via le tokenizer courant (`ctx.currentTokenizer`).
- Padding/troncature:
  - si la séquence est plus courte que `seq_len`, elle est paddée avec `pad_id`
  - si elle est plus longue que `seq_len`, elle est tronquée

À savoir:

- Si aucun tokenizer n’est présent, la fonction ne génère pas de tokens (et peut retourner `ok=true` avec `0` séquences). Vérifie la valeur retournée.
- Les séquences sont stockées en interne (dans le contexte Lua/C++), pas retournées directement à Lua.

## Détails utiles (mémoire et lazy-loading)

Même si `Mimir.Dataset.load()` n’effectue que l’indexation, le C++ contient des loaders lazy (`DatasetItem::loadText/loadImage/loadAudio/loadVideo`) protégés par un petit gestionnaire RAM interne (`DatasetMemoryManager`).

Conséquences côté Lua:

- `prepare_sequences()` appelle `loadText()` en interne. Si un fichier texte est trop gros (ou si le gestionnaire RAM interne considère qu’il ne peut pas allouer), le texte ne sera pas chargé et l’item ne produira pas de séquence.
- Le gestionnaire RAM dataset a une limite par défaut de 10 GB dans `Helpers.hpp`, mais **elle n’est pas configurable via l’API Lua dataset actuelle**.

À retenir:

- Évite de mettre des fichiers texte énormes “bruts” dans le dataset; préfère des items plus petits (ou un pré-traitement hors runtime).
- Si tu observes `0` séquences préparées, vérifie d’abord: tokenizer présent, puis taille/qualité des fichiers texte.

## Exemples d’usage

### Dataset texte (préparation de séquences)

```lua
local ok_ds, n_or_err = Mimir.Dataset.load("datasets.old/text")
assert(ok_ds, n_or_err)

-- tokenizer doit exister (voir scripts/modules/base_tokenizer.lua, ou Mimir.Tokenizer)
local ok_seq, n_seq_or_err = Mimir.Dataset.prepare_sequences(512)
assert(ok_seq, n_seq_or_err)
print("sequences:", n_seq_or_err)
```

### Inspecter les items (chemins)

```lua
local ok_ds, n = Mimir.Dataset.load("checkpoints/llm_simple")
if ok_ds then

  local item, err = Mimir.Dataset.get(1)
  if item then
    print(item.text_file, item.image_file)
  else
    print("get failed:", err)
  end
end
```

## Bonnes pratiques

- Datasets multi-modaux: préfère des basenames uniques sur tout le dataset (évite les collisions entre sous-dossiers).
- Reproductibilité: loggue dans ton checkpoint les infos “pipeline” (chemin dataset, `seq_len`, vocab/tokenizer utilisé).
- Validation rapide: ajoute un script de smoke test qui fait `load()` puis `get(1)` et vérifie la présence des champs attendus.

## Limites actuelles (et implications)

- `Mimir.Dataset.load()` ne permet pas (encore) de choisir `target_w/target_h/min_modalities` via l’API Lua, même si le C++ le supporte.
- `Mimir.Dataset.get()` ne charge pas les fichiers à la demande: il expose principalement des chemins.
- Si tu as besoin des buffers image/audio/vidéo en Lua, il faut ajouter des fonctions dédiées côté C++ (ex: `Dataset.load_image(i)`), ou traiter ces données côté C++ dans la pipeline.
