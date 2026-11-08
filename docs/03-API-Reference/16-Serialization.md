# `Mimir.Serialization`

Trouver rapidement le contrat API réel et les paramètres utilisables.

**Public concerné :** Développeur et utilisateur intermédiaire/avancé.

> **Prérequis**
>
> Connaître les commandes de base de Mímir.


Source : `src/scriptings/Lua/luaScripting/LuaScripting.cpp` et `src/Serialization/*`.

## Formats

- `safetensors` (alias: `st`)
- `raw_folder` (alias: `raw`, `folder`)
- `debug_json` (alias: `debug`, `json`)

Les packages `.mpk` sont gérés séparément par `scripts/modules/mpk.lua` et le
chargeur de modèles. Cette alternative expérimentale, encore partielle pour la
production, décrit une architecture/configuration sans transporter les poids. Voir
[MPK : packages d’architecture](../02-User-Guide/15-MPK.md).

## `save(path: string, format?: string="safetensors", opts?: table) -> bool | (false, err)`

Options (principales) :

- `save_tokenizer=true`, `save_encoder=true` (bool)
- `save_optimizer=false` (bool) : nécessaire pour restaurer les moments et le
  compteur optimiseur lors d'une reprise complète.
- `include_git_info=true`, `include_checksums=false` (bool)
- `debug_max_values=100` (int) : limite historique du format DebugJSON.
- `training_state` (table) : état de boucle ajouté à `model_config.training_state`.

La table d'options de `save` occupe le **troisième** argument : utilisez
`save(path, "safetensors", options)` ; la forme `save(path, options)` ne la lit pas.

Options DebugJson “enhanced” (v1.4) :

- `include_gradients` (bool)
- `include_optimizer_state` (bool)
- `include_activations` (bool)
- `include_weight_deltas` (bool)
- `max_values_per_tensor` (int)

`custom_metadata` et `save_dataset_info` existent en C++ dans `SaveOptions`, mais
ne sont pas lus par le binding Lua `save`.

## DType (stockage sur disque)

Le `dtype` du modèle participe au choix du dtype des tenseurs lors de
`save()`. Le dtype effectivement utilisé en calcul dépend ensuite du runtime et
de l'opération ; consultez `Architectures.dtypes()` pour les types reconnus.

- Source : `Model.default_dtype` (exposé via `Mimir.Model.dtype()`).
- Valeurs usuelles : `"float16"`, `"bfloat16"`, `"float32"`, `"float64"`.

Règles :

- `save()` (SafeTensors/RawFolder) convertit les tenseurs float du modèle en fonction de `Model.default_dtype`.
- `load()` convertit les tenseurs vers la représentation attendue par le
  modèle et **réapplique** `model_config.dtype` s'il est présent dans les
  métadonnées.

### Conformité SafeTensors

Les fichiers produits suivent le format SafeTensors officiel :

- longueur du header encodée sur 8 octets little-endian ;
- header UTF-8 commençant par `{`, complété uniquement avec des espaces ;
- `__metadata__` est strictement une map chaîne-vers-chaîne. L’option
  `custom_metadata` est validée puis stockée comme chaîne JSON canonique ;
- dtypes canoniques, formes et offsets vérifiés sans overflow ;
- buffer entièrement indexé, contigu, sans trou, chevauchement ou octet final
  non référencé ;
- tenseurs scalaires (`shape=[]`) et vides acceptés.

Le lecteur rejette les anciens fichiers Mímir dont `__metadata__` contient des
nombres ou objets JSON non conformes. Ils doivent être réexportés avec une
version corrigée du writer.

Bon pattern :

- mettre `cfg.dtype` dans la config passée à `Mimir.Model.create(...)`.
- ou appeler `Mimir.Model.dtype("float16")` après `create()` si vous voulez forcer un override.

## `load(path: string, format?: string|"auto", opts?: table) -> bool | (false, err)`

- Si `format` absent : auto-détection.

Options :

- `load_tokenizer=true`, `load_encoder=true`, `load_optimizer=false`
- `strict_mode=true`
- `validate_checksums=true`
- `apply_model_config=true` : applique la configuration enregistrée.
- `mapping_json` (chemin), alias `tensor_mapping_json` : mapping des tenseurs externes.
- `load_layer_metadata` : applique les métadonnées des couches aux graphes préconstruits (défaut : `true`). Pour importer les poids d'un VAE complet dans son décodeur seul, utiliser `false` avec `apply_model_config=false` et construire auparavant le décodeur depuis la configuration du checkpoint. Un graphe vide conserve sa reconstruction automatique depuis les métadonnées disponibles.

`load(path, options)` est accepté si le format est omis ; avec un format explicite,
utilisez le troisième argument. `apply_model_name` et `metadata_only` sont des
options C++ non exposées par ce binding Lua.

Les assets tokenizer/encodeur chargés sont resynchronisés dans le contexte Lua.
La topologie doit rester compatible : le chargement n'est pas une conversion
automatique entre deux architectures. Les scripts de reprise reconstruisent
d'abord la configuration et conservent l'état de boucle approprié au modèle.

## `detect_format(path: string) -> string | (nil, err)`

Retourne : `SAFETENSORS`, `RAWFOLDER`, `DEBUGJSON`.

## `save_enhanced_debug(path: string, opts?: table) -> bool | (false, err)`

Écrit un JSON d’inspection (stats + options avancées).

Depuis v1.3, le dump contient aussi `framework_state` avec un snapshot au moment du dump.
La v1.4 ajoute des sections aux périmètres explicites :

- `metadata` : version du format/framework, date de création, commit Git disponible ;
- `model` : identité, dtype, dimensions et nombre logique de paramètres ;
- `export_metrics` : éléments référencés et uniques, poids partagés, octets runtime
  et octets selon le dtype sérialisé. Ces compteurs excluent explicitement
  tokenizer, encodeur et optimiseur ;
- les statistiques de tenseurs comptent séparément les valeurs finies, zéros,
  `NaN`, `+Inf` et `-Inf`. Les échantillons non finis sont représentés par
  `null` afin que le fichier reste un JSON valide.

Le snapshot `framework_state` expose :

- runtime/backends (`cpu`, `cuda`, `rocm`, `opencl`, `vulkan`) + flags de config,
- capacités CPU (`avx2`, `fma`, `f16c`, `bmi2`),
- état mémoire (`MemoryGuard`, `DynamicTensorAllocator`, `AdvancedRAMManager`),
- registre de layers supportés,
- état modèle (dtype par défaut, params, layers, model_config, etc.).

## Étapes suivantes

- [Page précédente : API : monitoring / visualisation](15-Viz-Htop.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : API : `Mimir.Layers`](18-Layers-Module.md)
