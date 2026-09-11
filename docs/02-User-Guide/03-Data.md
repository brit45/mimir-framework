# Données et datasets

Préparer un dataset compatible et comprendre comment il est lu.

**Public concerné :** Débutant à intermédiaire.

> **Prérequis**
>
> Avoir un dossier de données local.


Cette page décrit le comportement **réel** du loader de dataset actuellement exposé à Lua via `Mimir.Dataset`, et ses limitations.

Sources principales :

- API Lua: `src/scriptings/Lua/luaScripting/LuaScripting.cpp` (`lua_loadDataset`, `lua_getDataset`, `lua_prepareSequences`)
- Indexation + lazy-loading: `src/Helpers.hpp` (`loadDataset`, `DatasetItem`, `DatasetMemoryManager`, `DatasetManager`)

## Sur cette page

- [Vue d’ensemble](#vue-densemble)
- [Format disque et règle de “linking”](#format-disque-et-règle-de-linking)
- [API Lua: Mimir.Dataset](#api-lua-mimirdataset)
- [Détails utiles (mémoire et lazy-loading)](#détails-utiles-mémoire-et-lazy-loading)
- [Exemples d’usage](#exemples-dusage)
- [Bonnes pratiques](#bonnes-pratiques)
- [Limites actuelles (et implications)](#limites-actuelles-et-implications)
- [Étapes suivantes](#étapes-suivantes)

## Vue d’ensemble

- `Mimir.Dataset.load(dir)` indexe récursivement un dossier et construit une liste d’items (métadonnées uniquement).
- `Mimir.Dataset.get(i)` charge le texte et les pixels RGB redimensionnés à la demande.
- `Mimir.Dataset.get(i, false)` retourne texte, chemins et dimensions sans décoder l'image.
- `Mimir.Dataset.release(i)` libère les buffers lazy de l'item après utilisation.
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
- Les paramètres optionnels permettent de choisir les dimensions cibles, le seuil de modalités, le cache, la limite RAM et le chargement lazy.
- L’indexation est en mode “lazy”: les données (texte/image/audio/vidéo) ne sont pas chargées en RAM à ce stade.

### `Mimir.Dataset.get(index, load_image?)`

Retourne un item du dataset (Lua est 1-indexed).

- Entrées: `index` dans `[1, num_items]`; `load_image` vaut `true` par défaut
- Sortie: `(item)` en cas de succès, ou `(nil, err)` en cas d’erreur

Champs possibles dans `item`:

- `text_file`, `image_file`, `audio_file`, `video_file`: chemins vers les fichiers (si présents)
- `width`, `height`, `channels`: dimensions cibles et canaux
- `text`: contenu texte chargé à la demande
- `image`: pixels RGB u8 si `load_image` est vrai et le décodage réussit

Pour les grands datasets, appeler `Mimir.Dataset.release(index)` après avoir consommé
l'image. Cela évite que les buffers lazy s'accumulent jusqu'à la limite RAM configurée.

Si seul le texte est requis, utiliser `Mimir.Dataset.get(index, false)` pour éviter le
décodage et la construction de la table de pixels Lua.

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
- Si vous observez `0` séquences préparées, vérifie d’abord: tokenizer présent, puis taille/qualité des fichiers texte.

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
- Reproductibilité: loggue dans votre checkpoint les infos “pipeline” (chemin dataset, `seq_len`, vocab/tokenizer utilisé).
- Validation rapide: ajoute un script de smoke test qui fait `load()` puis `get(1)` et vérifie la présence des champs attendus.

## Limites actuelles (et implications)

- `Mimir.Dataset.get()` expose actuellement les pixels image RGB, mais pas les buffers audio ou vidéo.
- Le chargement image lazy reste explicite côté appelant: utiliser `release(index)` après consommation sur les grands datasets.

## Étapes suivantes

- [Page précédente : Workflow modèle (lifecycle)](02-Model-Lifecycle.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Entraînement](04-Training.md)
