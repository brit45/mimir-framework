# Internals : `Helpers.hpp` (helpers C++)

Cette page documente le contenu de `src/Helpers.hpp`.

Objectif: donner une carte précise des helpers “transverses” (IO, hashing, dataset, image utils) et de leurs propriétés (déterminisme, perfs, limites).

## Rôle du header

`Helpers.hpp` sert de “boîte à outils” incluse dans plusieurs unités de compilation. La plupart des fonctions sont `static inline` (donc instanciées dans chaque TU qui inclut le header).

Domaines couverts:

- IO bas niveau (bytes <-> string)
- utilitaires image (resize nearest, decode via stb_image)
- utilitaires dataset (scan, linking, lazy-loading)
- hashing + tokens “magiques” (démos/branches)

## IO: fichiers et buffers

### `readFileToVector(path, out, err?) -> bool`

- Lit un fichier binaire dans `std::vector<uint8_t>`.
- Utilise un `ifstream` positionné à la fin (`ios::ate`) pour obtenir la taille puis relit depuis le début.
- En cas d’erreur: retourne `false` et remplit éventuellement `*err`.

### `bytesToString(buf, off, len) -> std::string`

- Construit une `std::string` depuis un buffer de bytes (slice).
- Si `off` dépasse la taille, retourne une string vide.

## Image utils

### `resizeNearest(src, iw, ih, channels, dst, ow, oh)`

Resize nearest-neighbor interne (remplace stb_image_resize ici).

- Simple et déterministe.
- Pas d’anti-aliasing: sert de base robuste pour des pipelines “dataset loader / debug”.

### STB image

Le header inclut `stb_image.h` et suppose que `STB_IMAGE_IMPLEMENTATION` est défini ailleurs (`src/stb_image_impl.cpp`).

Dans `DatasetItem::loadImage*`, la lecture se fait via `stbi_load(..., 3)` (force 3 canaux), puis resize nearest.

### `imageToEmbedding(img, iw, ih, dim) -> vector<float>`

Helper de démo/fallback:

- calcule la moyenne des pixels (sur le buffer fourni)
- mappe vers $[-1,1]$
- répète la même valeur sur `dim` dimensions

## Modalités / hashing / MagicToken

### `enum Modality` + `detectModalities(dataset_dir) -> unsigned mask`

Scanne récursivement un dossier et retourne un masque de modalités détectées par extensions.

Extensions dans `detectModalities`:

- texte: `txt json csv md xml`
- audio: `wav flac mp3 ogg m4a aac`
- image: `png jpg jpeg bmp tiff webp`
- vidéo: `mp4 mkv avi mov webm flv ts`

Note: `detectModalities()` est un détecteur “grossier” par extension. Le loader `loadDataset()` a sa propre liste d’extensions (voir plus bas).

### `datasetSHA256Hash(dir) -> string`

Calcule un SHA256 déterministe basé sur une liste triée d’entrées `filename|file_size`.

Propriétés importantes:

- Dépend du **nom de fichier** (pas du chemin relatif complet) et de la taille.
- Ne lit pas le contenu.

Implication: deux fichiers dans des sous-dossiers différents mais avec le même `filename` et la même taille peuvent produire des collisions dans la signature.

### `simpleDatasetHash(dir) -> uint32_t`

Hash rapide type FNV sur les `filename` uniquement (pas les tailles).

### `MagicToken` + `makeMagicToken(modality_mask, dir)`

Construit un token déterministe:

- `seed = simpleDatasetHash(dir) ^ (modality_mask * constant)`
- `embed[8]` tiré d’une RNG `mt19937(seed)` dans $[-1,1]$

## Nettoyage texte

### `sanitize_utf8(s) -> string`

Parcourt la string et remplace les séquences invalides par le caractère de remplacement U+FFFD.

Utilisé par le loader texte (`DatasetItem::loadText()`) pour éviter que des fichiers corrompus cassent la pipeline.

## Dataset: structures et lazy-loading

La partie dataset de `Helpers.hpp` fait deux choses distinctes:

1) **Indexation** (rapide): construire des `DatasetItem` avec des chemins + métadonnées
2) **Chargement lazy** (lourd): charger texte/image/audio/vidéo à la demande, sous contrainte RAM

### `DatasetMemoryManager` (singleton)

Gestionnaire global best-effort pour la RAM dataset:

- Limite par défaut: 10 GB (`max_ram_bytes`)
- `canAllocate(bytes)` pour refuser un chargement
- tracking “approx” des allocations via `(void*)data_ptr -> bytes`
- stats: current / peak / max + nombre d’allocations

Important: ce gestionnaire est indépendant de `Mimir.Allocator`/`MemoryGuard` (qui gèrent plutôt la mémoire runtime/tensors).

### `DatasetItem`

Représente un item dataset:

- chemins: `text_file`, `image_file`, `audio_file`, `video_file`
- data lazy: `text`, `img`, `audio_bytes`, `video_bytes` (tous `std::optional`)
- méta: `w/h` (taille cible), `img_c` (0 inconnu, 1 grayscale, 3 RGB)
- LRU: `last_access_time` + `touch()`
- RAM: `estimated_ram_usage`

Loaders:

- `loadText()`: lit le fichier, `sanitize_utf8`, tracke la RAM.
- `loadImage(target_w, target_h)`: decode via `stbi_load` (RGB), resize nearest, convertit en **grayscale** (1 canal), tracke la RAM.
- `loadImageRGB(target_w, target_h)`: decode + resize nearest, stocke **RGB** (3 canaux) et met `img_c=3`.
- `loadAudio() / loadVideo()`: charge les bytes bruts via `ifstream` binaire.

Libération:

- `unload()` libère toutes les données et décrémente le tracking.
- Le destructeur appelle `unload()` (donc destruction = libération dataset RAM).

### `DatasetManager` (LRU)

Outil pour charger un sous-ensemble et évincer le reste:

- `ensureLoaded(items, indices, target_w, target_h)`: estime la RAM nécessaire; évince LRU si besoin (`evictLRU`); charge image puis texte pour les indices demandés
- `preloadBatch(items, batch_start, batch_size, ...)`
- `printStats()` (stats du RAM manager)

## Dataset: indexation

### `loadDataset(root_dir, target_w=64, target_h=64, min_modalities=1) -> vector<DatasetItem>`

Indexation en deux phases:

1) Parcours récursif et groupement par `stem()` (basename sans extension)
2) Construction des items:
     - `name = basename`
     - `is_linked = (paths.size() > 1)`
     - classification par regex d’extension
     - filtrage par `min_modalities`

Extensions dans `loadDataset`:

- image: `png jpg jpeg bmp tiff webp`
- texte: `txt md json csv`
- audio: `wav flac mp3 ogg m4a aac`
- vidéo: `mp4 mkv avi mov webm flv ts`

Le loader ne charge pas les données à l’indexation (RAM ~0): uniquement les chemins + méta.

## Dataset: cache d’index

### `DatasetCache`

Sauvegarde/restaure un cache JSON:

- `dataset_hash` (SHA256)
- `items[]` (name, is_linked, fichiers, w/h)

### `loadDatasetCached(root_dir, ..., cache_path="dataset_cache.json", max_ram_mb=10240, lazy_loading=true)`

- Initialise la limite RAM du dataset (`DatasetMemoryManager::setMaxRAM`).
- Calcule `datasetSHA256Hash(root_dir)`.
- Charge le cache si le hash correspond.
- Sinon ré-indexe via `loadDataset()` puis écrit le cache.

Modes:

- `lazy_loading=true`: ne charge pas les images; retourne uniquement méta/paths.
- `lazy_loading=false`: tente de charger les images au démarrage (peut s’arrêter si RAM insuffisante).

## Correspondance Lua

Le binding Lua `Mimir.Dataset.load(dir)` utilise le loader `loadDataset(dir)` (pas `loadDatasetCached`).

Pour l’API user, voir `docs/02-User-Guide/03-Data.md`.
