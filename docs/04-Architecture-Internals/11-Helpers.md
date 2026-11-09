# Helpers C++

Comprendre le fonctionnement interne exact des composants runtime.

**Public concerné :** Développeur avancé qui modifie le moteur C/C++.

> **Prérequis**
>
> Connaître les bases C++ et la structure du dépôt.


Cette page documente le contenu de `src/Helpers.hpp`.

Objectif: donner une carte précise des helpers “transverses” (IO, hashing, dataset, image utils) et de leurs propriétés (déterminisme, perfs, limites).

## Sur cette page

- [Rôle du header](#rôle-du-header)
- [IO: fichiers et buffers](#io-fichiers-et-buffers)
- [Image utils](#image-utils)
- [Modalités / hashing / MagicToken](#modalités-hashing-magictoken)
- [Nettoyage texte](#nettoyage-texte)
- [Dataset: structures et lazy-loading](#dataset-structures-et-lazy-loading)
- [Dataset: indexation](#dataset-indexation)
- [Dataset: cache d’index](#dataset-cache-dindex)
- [Correspondance Lua](#correspondance-lua)
- [Étapes suivantes](#étapes-suivantes)

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

Les images statiques prises en charge par STB sont décodées en RGB avec
`stbi_load(..., 3)`. `loadImageRGB` applique ensuite
`resizeBicubicRGB_SRGBLinear`, également pour les frames animées. Le helper
nearest reste utilisé pour les masques et certains chemins de repli ; ce
n'est pas le resize principal des images RGB. WebP utilise FFmpeg quand ce
backend est compilé ; les variantes BMP refusées par STB peuvent aussi y revenir.

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
- image: `png jpg jpeg gif bmp tiff webp ppm pgm`
- vidéo: `mp4 mkv avi mov webm flv ts`

Note: `detectModalities()` est un détecteur “grossier” par extension. Le loader `loadDataset()` a sa propre liste d’extensions (voir plus bas).

### `datasetSHA256Hash(dir) -> string`

Calcule une signature SHA256 d'index, versionnée
`mimir_dataset_index_v5_directory_symlinks`, à partir de la racine lexicale
absolue et d'une liste triée de métadonnées par fichier.

Propriétés importantes:

- Chaque entrée contient le chemin relatif lexical, la cible résolue, la taille
  et la date de modification. Les JSON ajoutent un SHA256 de leur contenu.
- Les dossiers symboliques sont parcourus comme les dossiers ordinaires.
  Changer la cible d'un lien invalide le cache, même si taille et date des cibles
  sont identiques.
- Les fichiers internes et le fichier de cache explicitement exclu ne participent
  pas à la signature, y compris si un alias symbolique mène à ce cache.
  Les contenus image/audio/vidéo ne sont pas entièrement hachés.

Deux homonymes de sous-dossiers différents sont donc distingués. Le déplacement
de la racine invalide aussi le cache ; cette signature n'est pas un hash de
contenu exhaustif de chaque média.

### `simpleDatasetHash(dir) -> uint32_t`

Hash rapide type FNV sur les `filename` uniquement (pas les tailles), dans
l'ordre du parcours filesystem. Contrairement à la signature d'index, cette
fonction ne trie pas ses entrées et n'assure pas l'indépendance à cet ordre.

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

- chemins: `text_file`, `image_file`, `mask_file`, `audio_file`, `video_file`
- data lazy: `text`, `img`, `animation_frames`, `animation_frame_delays_ms`, `mask`, `audio_bytes`, `video_bytes`
- méta: `w/h` (taille cible), `img_c` (0 inconnu, 1 grayscale, 3 RGB)
- LRU: `last_access_time` + `touch()`
- RAM: `estimated_ram_usage`

Loaders:

- `loadText()`: lit le fichier, `sanitize_utf8`, tracke la RAM.
- `loadImage(target_w, target_h)`: alias du chargement RGB, conservé pour compatibilité.
- `loadImageRGB(target_w, target_h)`: décode et redimensionne en RGB bicubique. BMP passe par STB; WebP statique/animé passe par FFmpeg quand ce backend est compilé.
- GIF animé: toutes les frames compositées sont redimensionnées et stockées dans `animation_frames`; les délais sont dans `animation_frame_delays_ms` et la répétition dans `animation_loop_count` (`0` signifie infini). `img` reste la première frame pour compatibilité.
- `loadMask(target_w, target_h)`: charge le masque lié en mono avec un resize nearest-neighbor qui préserve les classes.
- `makeMaskLossWeights(channels)`: produit des poids HWC normalisés utilisés par la loss VAE pour renforcer la région annotée tout en conservant le contexte.
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

1) Parcours récursif et groupement par chemin relatif lexical sans extension
2) Construction des items:
     - nom dérivé de la clé relative
     - `is_linked = (paths.size() > 1)`
     - classification par regex d’extension
     - filtrage par `min_modalities`

Les masques ne deviennent pas des items autonomes. Ils sont liés à leur image standard via les suffixes
`_mask`, `-mask`, `.mask`, `_segmentation`, `-segmentation`, ou via des dossiers frères
`masks/` / `segmentations/` correspondant à `images/`. Un masque est une annotation et ne compte donc
pas comme une modalité supplémentaire.

Les clés lexicales préservent le nom des liens symboliques : une image liée
vers un autre disque reste associée à son sidecar local. Les annotations
structurées et les captions COCO ont leurs chemins d'indexation dédiés ;
elles ne doivent pas être confondues avec de simples fichiers de texte libres.

Le parcours `Mimir::DatasetTraversal::forEachRegularFile`, défini dans
`src/DatasetTraversal.hpp`, est commun à l'indexation, aux hash et à la détection
des modalités. Il suit
les liens de fichiers et de dossiers, y compris les chaînes et les cibles hors
racine. Il résout les dossiers pour détecter les retours vers un ancêtre, sans
remplacer leur chemin lexical d'indexation. Deux alias non cycliques restent
distincts ; il n'effectue pas de dédoublonnage global par cible. Les liens cassés
et les branches cycliques sont ignorés avec un diagnostic. La racine peut être
un lien vers un dossier, mais l'API attend toujours un dossier.

Extensions dans `loadDataset`:

- image: `png jpg jpeg gif bmp tiff webp ppm pgm`

`Mimir.Dataset.get(index, true)` expose en plus `is_animated`, `frame_count`, `loop_count`,
`frame_delays_ms` et `animation_frames` pour les images animées.

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
- Combine `datasetSHA256Hash(root_dir, cache_path)` avec les dimensions cibles
  et `min_modalities` pour la signature du cache.
- Charge le cache si le hash correspond.
- Sinon ré-indexe via `loadDataset()` puis écrit le cache.

Modes:

- `lazy_loading=true`: ne charge pas les images; retourne uniquement méta/paths.
- `lazy_loading=false`: tente de charger les images au démarrage (peut s’arrêter si RAM insuffisante).

## Correspondance Lua

Le binding Lua `Mimir.Dataset.load(dir, w, h, min_modalities, use_cache,
cache_path, max_ram_mb, lazy_loading)` utilise `loadDataset` par défaut.
Avec `use_cache=true`, il utilise `loadDatasetCached` et les options suivantes.

Pour l’API user, voir `docs/02-User-Guide/03-Data.md`.

## Étapes suivantes

- [Page précédente : Internals : classe `Model` (C++)](10-Model-Class.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Internals : stockage `tensor` + allocation dynamique (C++)](12-Tensor-Storage.md)
