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

Le dossier racine peut être un lien symbolique. Le parcours suit aussi les
liens de fichiers et de sous-dossiers, relatifs ou absolus, y compris vers
une cible extérieure. Il conserve les chemins locaux pour associer les
modalités et sidecars. Les liens cassés et les branches cycliques sont ignorés
avec un diagnostic ; deux alias distincts restent deux groupes d'items.
Le hash du cache inclut les cibles résolues et les fichiers traversés, afin de
réindexer après changement de cible. Voir l'[exemple de liens symboliques](../02-User-Guide/03-Data.md#liens-symboliques-de-fichiers-et-de-dossiers).

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

## Préchargement en arrière-plan

En mode lazy, le lecteur prépare un nombre fixe d'échantillons d'avance, dans
une file bornée, avec un emplacement supplémentaire pour l'échantillon utilisé.
`./run_mimir.sh` fixe ce nombre à **10** si la variable n'est pas définie ;
le défaut natif, pour un lancement direct sans variable, reste **8**.
Un worker natif lit les fichiers, décode/redimensionne les images RGB et les
masques, et prépare le texte et les annotations. Les boucles natives de VAEConv,
VGG et du détecteur médical demandent aussi la normalisation et la disposition
HWC/CHW attendues. VAEText prépare la tokenisation et le padding avec une copie
stable du tokenizer ; les fenêtres aléatoires restent sur le thread d'entraînement.
Les tampons de conversion sont réutilisés et les contenus natifs sont déchargés
sur le worker après leur dernière utilisation.

`get` anticipe les indices suivants en lecture séquentielle. Pour un ordre
mélangé ou une reprise à une position donnée, fournir l'ordre réel avant la boucle :

```lua
local order = { 7, 2, 2, 11, 1 } -- indices 1-based, doublons permis
assert(Mimir.Dataset.prefetch(order, true))
for _, index in ipairs(order) do
    local item = assert(Mimir.Dataset.get(index))
    -- Consommer item.image et item.text.
    assert(Mimir.Dataset.release(index))
end
local stats = Mimir.Dataset.prefetch_stats()
print(stats.hits, stats.waits, stats.wait_ms)
```

`prefetch(indices, load_image=true) -> (true, count) | (false, err)` ne change
ni l'ordre ni le nombre d'éléments consommés. Un tableau vide rétablit
l'anticipation séquentielle au prochain `get`. `load_image=false` prépare
uniquement le texte, les annotations et les métadonnées. Les boucles natives
fournissent automatiquement leur permutation ; le script Lumen fournit aussi
son ordre sauvegardé/mélangé. Les mêmes fonctions sont accessibles via les
bridges JavaScript et Rust.

`prefetch_stats() -> table` expose `enabled`, `capacity`, `queued`, `ready`,
`leased`, `loading`, `resident_bytes`, `peak_bytes`, `requests`, `hits`, `waits`,
`prepared`, `recycled`, `errors`, `buffer_growths`, `wait_ms` et `prepare_ms`.
`buffer_growths` compte les agrandissements des buffers de conversion réutilisés,
pas toutes les allocations des décodeurs. Un hit indique un échantillon déjà
prêt lors de l'appel. Les temps de préparation et d'attente sont des cumuls.

Les variables `MIMIR_DATASET_PREFETCH` (nombre d'éléments, défaut natif 8,
défaut du lanceur 10, maximum 256)
et `MIMIR_DATASET_PREFETCH_BYTES` (budget de préchargement en octets, défaut
64 Mio) sont lues au chargement. Mettre l'une à `0` conserve le chemin synchrone.
Le chargement explicitement eager (`lazy_loading=false` avec cache) conserve
son comportement. Les contrôles RAM du dataset et du décodeur restent actifs.
Si le prochain élément est plus grand que le budget d'anticipation, il peut
être préparé seul à la demande ; il doit encore tenir dans la RAM autorisée.

Le premier élément, un accès imprévu ou un producteur plus lent que le calcul
peuvent encore nécessiter une attente. La libération native demandée par
`release` peut être différée jusqu'au passage du worker. Les tables Lua/JSON
restent construites sur le thread du langage. Les vidéos gardent leur lecteur
de frames en streaming ; cette file ne préconvertit pas toutes leurs frames.
Un rechargement ou la fermeture du runtime arrête le worker et attend la
lecture déjà engagée, sans parcourir le reste du dataset.

Le [rapport de mesures et de validation](../04-Architecture-Internals/Runtime-Dataset-Prefetch-20261112.md)
détaille les gains CPU, la consommation mémoire et les comparaisons numériques.

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
