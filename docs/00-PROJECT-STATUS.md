# État réel du projet

Cette page est la carte de maturité du checkout Mímir 3.6.0, relue contre les
sources le 8 novembre 2026. Les mentions de tests matériels concernent les
contrôles ciblés cités, pas une revalidation de tous les backends à cette date.
Elle distingue ce qui est compilé et exploitable de ce qui est partiel, en test,
prototype ou simple placeholder. Une entrée dans le registre, un fichier source
ou un bouton d'interface ne constitue pas à lui seul une garantie de production.

## Sur cette page

- [Comment lire les statuts](#comment-lire-les-statuts)
- [Chemin recommandé](#chemin-recommandé)
- [Moteur et sous-systèmes](#moteur-et-sous-systèmes)
- [Architectures](#architectures)
- [Runtimes matériels](#runtimes-matériels)
- [Scripting et configuration](#scripting-et-configuration)
- [Données, entraînement et checkpoints](#données-entraînement-et-checkpoints)
- [Fonctionnalités en construction](#fonctionnalités-en-construction)
- [Niveaux de preuve](#niveaux-de-preuve)
- [Reproduire l'inventaire](#reproduire-linventaire)

## Comment lire les statuts

| Statut | Signification vérifiable |
| --- | --- |
| Actif | Compilé dans la configuration concernée et relié à un chemin d'exécution. |
| Exploitable | Possède une API ou un workflow utilisable, avec au moins un test ou un exemple ciblé. |
| Partiel | Fonctionne dans un sous-ensemble documenté; fallback ou opérations manquantes possibles. |
| En test | Couvert par un test récent, mais sans validation représentative sur toutes les plateformes ou toutes les charges. |
| Expérimental | API et comportement susceptibles de changer; usage de recherche conseillé. |
| Placeholder | Nom/configuration présents, mais pas de pipeline complet exécutable. |
| En construction | Sources présentes dans le workspace, mais pas encore intégrées au build ou au registre principal. |

Ces statuts sont cumulatifs. Par exemple, CUDA peut être « actif » dans un build
CUDA et néanmoins « partiel » parce que certaines opérations utilisent le CPU.

## Chemin recommandé

Le chemin le plus reproductible aujourd'hui est :

1. construire en C++17 avec Lua 5.3+ ; OpenMP et LZ4 sont optionnels ;
2. garder le runtime CPU activé ;
3. créer un modèle natif via `Mimir.Model.create` ;
4. piloter le run en Lua ou avec `--conf` ;
5. sauvegarder en RawFolder ou SafeTensors ;
6. valider avec les tests CTest ciblés.

```bash
./config.sh --gui NONE --non-interactive --generator Ninja \
  --enable MIMIR_ENABLE_TESTS \
  --disable ENABLE_VULKAN --disable ENABLE_OPENCL
cmake --build build -j"$(nproc)"
ctest --test-dir build -R 'RegistryTest|LuaTest|SerializationTest' \
  --output-on-failure
```

Ce profil CPU minimise les dépendances optionnelles. Il ne prouve pas la
disponibilité d'un GPU, de SFML, de FFmpeg ou d'un bridge externe.

## Moteur et sous-systèmes

| Système | Statut | Ce qui est réellement utilisable | Limites principales |
| --- | --- | --- | --- |
| Graphe `Model` et layers | Actif, exploitable | Construction C++, forward/backward, paramètres nommés, routage nommé, entraînement ciblé. | La couverture backward dépend du type de layer et du runtime. |
| [Primitives Primer](04-Architecture-Internals/14-Layers-And-Ops.md) | Actives, testées sur CPU et Vulkan AMD | 18 types supplémentaires, forward/backward, architectures natives C++ et checkpoints RawFolder/SafeTensors ; import MPK expérimental. | OpenCL : kernels générés et syntaxe vérifiée, exécution matérielle non validée pour ces 18 types. CUDA/ROCm : repli CPU. Les domaines et budgets peuvent aussi provoquer un refus ou un repli. |
| Registry | Actif, exploitable | Création par nom, config par défaut, alias `sd3.5`, auto-enregistrement MPK. | « enregistré » ne veut pas dire « modèle complet ». |
| Planner | Expérimental, en test | Plans legacy/static/cost, calibration des opérations natives, politiques de travail, réutilisation de buffers, fusion et inspection JSON. | Parité et coût sont vérifiés sur des cas ciblés, pas sur tous les graphes ni toutes les charges. |
| RuntimeRouter | Actif, en test | Classe les capacités en `Unsupported`, `HostFallback`, `Native`, `NativeOptimized`; planifie forward et backward. | Le fallback hôte peut masquer une absence de kernel GPU natif. |
| CPU | Actif, exploitable | Référence générale, OpenMP, dispatch SIMD selon les capacités CPU/OS, parallélisme borné et kernels pointwise fusionnés en forward/backward. | Certains chemins restent float32 internes ou coûtent beaucoup de mémoire. |
| Mémoire | Actif, exploitable | `MemoryGuard`, allocator dynamique, scratchpads, métriques et limites explicites. | Les politiques strictes doivent être dimensionnées pour la charge réelle. |
| Sérialisation | Actif, exploitable | RawFolder, SafeTensors, DebugJson, vérification de formes/types/checksums selon format. | Un checkpoint externe doit correspondre à la topologie attendue. |
| Tokenizer/Encoder | Actif, exploitable | Tokenizer natif, BPE, séquences, encodeur et assets de checkpoint. | La compatibilité d'un tokenizer externe doit être vérifiée, pas supposée. |
| Dataset multimodal | Actif, partiel par format | Texte, image, audio et vidéo sont représentés; FFmpeg décode les médias si compilé. | Les octets compressés ne sont pas des tenseurs décodés; FFmpeg est optionnel. |
| Viz SFML/Qt/GTK/Web et Htop | Optionnel, exploitable | Monitoring, panneaux, contrôles et snapshots ; backend graphique exclusif choisi par CMake. Qt/GTK utilisent un hôte desktop séparé et des caches de textures. | Les hôtes Qt/GTK/Web ciblent Linux ; `NONE` permet le headless. |
| [Projection 3D générique](02-User-Guide/18-Projection-3D.md) | Optionnelle, testée sur des cas ciblés | Capture commune des poids et activations, ACP asynchrone, corrélations mesurées, vues neurones/poids actifs, temps réel optionnel et exports PNG/JSON/HTML. | Échantillonnage borné et poids déjà résidents ; les sous-modèles privés hors du graphe ne sont pas parcourus. Le rendu se valide par backend graphique. |
| WebSocket API | Expérimental | Serveur Lua dans `scripts/modules/api_ws_server.lua`. | Ce n'est pas le bridge REST natif CMake. |

## Architectures

### Architectures natives enregistrées

Le graphe d’une architecture native est défini en C/C++ dans une classe
dérivée de `Model`. Les scripts et JSON le configurent et l’exécutent, selon
le [contrat des architectures](07-Devs/08-Architecture-Contract.md).

Les entrées natives du registre se répartissent par famille ; l’inventaire
effectif se consulte avec les outils d’inspection :

| Famille | Entrées | Statut utile |
| --- | --- | --- |
| Général | `basic_mlp` | Exploitable et adapté aux smoke tests. |
| NLP | `transformer`, `causal_lm`, `t5`, `vae_text`, `vae_text_decode` | `t5` fournit un encodeur bidirectionnel et une tête de débruitage par tokens; ce n'est pas le pipeline T5 original avec décodeur text-to-text. |
| Vision | `vit`, `vae`, `vae_conv`, `vae_conv_decode`, `resnet`, `unet`, `mobilenet`, `vgg16`, `vgg19`, `vgg16_feat`, `patch_discriminator`, `medical_detector` | Graphes exécutables; entraînement et géométrie varient selon l'architecture. |
| Interopérabilité HF/SDXL | `hf_clip_text_encoder_1`, `hf_clip_text_encoder_2`, `hf_vae_decoder`, `hf_sdxl_transformer_block`, `external_safetensors_base` | Les quatre premiers construisent des composants exécutables; `external_safetensors_base` reflète des clés et n'est pas un modèle autonome. |
| Génératif | `diffusion`, `cond_diffusion`, `gan_latent`, `Gan_Style`, `lumen_diffusion` | Lumen fournit un pipeline latent texte-vers-image avec VAE gelé et conditionnement Transformer, VAEText ou T5 selon la configuration. |
| Placeholder | `sd3_5` | Registre et squelette seulement; ne pas annoncer un pipeline SD3.5 complet. |

### Architectures MPK livrées

MPK est l’autre voie de définition, encore expérimentale et pas entièrement
exploitable en production. Une entrée chargée dans le registre ne démontre pas
une compatibilité complète de ses opérations ou de son entraînement.

`_archi/` fournit `deeplab`, `r_cnn`, `ssd` et `yolo`. Elles démontrent le
format MPK et la composition avec des backbones natifs. Elles restent des
prototypes : les étapes spécialisées comme propositions ROI, priors, décodage
de détection ou convolution dilatée complète ne sont pas toutes exécutées par
le graphe MPK. La NMS native peut compléter certains pipelines, mais ne remplace
pas leur pré/post-traitement.

Pour afficher l'inventaire réellement chargé :

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- --show-archs
```

## Runtimes matériels

| Backend | Activation | Statut réel |
| --- | --- | --- |
| CPU | `ENABLE_OPENMP`, `ENABLE_SIMD` | Référence et fallback recommandé. Les capacités sont déclarées nativement par type de layer. |
| CUDA | `-DENABLE_CUDA=ON` | Plusieurs familles natives, dont NMS forward/backward et Conv3d/Conv4d ; `CausalDepthwiseConv1d`, `SwiGLU` et les 18 types Primer restent en `HostFallback`. Backward Lambda indisponible. Nécessite CUDA/cuBLAS et du matériel. |
| ROCm | `-DENABLE_ROCM=ON` | Partiel, symétrique à plusieurs chemins CUDA; nécessite HIP/rocBLAS et du matériel compatible. |
| Vulkan | `-DENABLE_VULKAN=ON` | Kernels portables, dont convolutions ND, opérations de séquence et les 18 types Primer. Ces derniers sont testés sur GPU AMD. La capacité par opération dépend des formes, valeurs et budgets. |
| OpenCL | `-DENABLE_OPENCL=ON` | Même famille portable générée, dont les 18 types Primer. Pour ces ajouts, la syntaxe des kernels est vérifiée ; l’exécution matérielle OpenCL reste à valider. |
| TPU | `MIMIR_TPU=1`, worker Python/JAX et device compatibles | Runtime optionnel entre OpenCL et CPU dans le routeur. Une session doit confirmer un device TPU ; le transport et les opérations disponibles restent propres à ce backend. Les tests avec worker de référence CPU ne prouvent pas une exécution TPU physique. |
| FPGA iCESugar Pro | session et I/O expérimentales intégrées au build | `FpgaRuntime`, `FpgaSession` et `FpgaIo` sont dans `MIMIR_SOURCES` ; `Model` initialise la session. Les transferts dépendent du protocole et du matériel détecté. `linearForward`, `forwardLayer` et `supports*LayerType` renvoient encore `false` : aucune accélération générale du graphe n’est déclarée. |

NMS possède un forward de sélection d'indices et un backward stop-gradient
sur CPU, CUDA, Vulkan et OpenCL. Les gradients de boîtes, scores et classes
sont nuls ; les losses d'entraînement doivent utiliser les prédictions avant
NMS. Les tests CPU et les comparaisons sur GPU AMD OpenCL/Vulkan passent ;
l'exécution NMS CUDA sur GPU NVIDIA reste à valider.

La page [Accélération GPU](05-Advanced/05-GPU-Acceleration.md) détaille les
familles d'opérations. Pour toute mesure, distinguez toujours : compilé,
initialisé, capacité native, fallback hôte, et backend effectivement exécuté.

## Scripting et configuration

| Surface | Statut | Contrat |
| --- | --- | --- |
| Lua | Actif, principal | Binding natif, table `Mimir`, global standard `arg`, modules sous `scripts/modules/`. |
| JSON `--conf` | Actif | Injecte `CONF`, `CONF_PATH`, `CONF_DIR` et les arguments; chaque script obtient une nouvelle VM Lua. |
| Tâches `--run` | Actif | Sélectionne `tasks.<nom>.lua` ou `tasks.<nom>.run.lua`. |
| `env` dans JSON | Actif | Appliqué au processus après les overrides et avant le run; remplace le shell. `OMP_NUM_THREADS` met aussi à jour OpenMP. |
| JavaScript | Optionnel, expérimental | Activé par défaut sous POSIX, indisponible sous Windows natif ; nécessite Node.js au runtime. |
| Rust | Optionnel, expérimental | Nécessite `rust-script`; passe par le contrat de bridge. |
| C# | Optionnel, expérimental | Nécessite `dotnet-script` ou un outil compatible. |
| REST natif | Non supporté | `ENABLE_SCRIPTING_REST=ON` est explicitement forcé à `OFF` par CMake. |
| [`Mimir.Layers`](03-API-Reference/18-Layers-Module.md) | Actif, inspection | `available()`, `by_type()` et filtres par type du registre : snapshots en lecture seule du modèle courant. |

Il n'existe pas de `Mimir.Args`. Utilisez :

```lua
local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local opts = Args.parse(arg)
```

## Données, entraînement et checkpoints

- Les scripts d'entraînement sont des programmes spécialisés, pas une boucle
  universelle magique. Lisez leurs options avec `--help` avant un long run.
- `train_vae_conv.lua` gère reconstruction, latent stochastique, KL/warmup,
  validation holdout, visualisation et reprise. Les nouveaux runs activent
  `logvar_head_bias` ; une reprise conserve la topologie de son checkpoint.
- `train_causal_lm.lua` pilote la préparation native du corpus/tokenizer et
  l’entraînement `CausalLMModel`, avec délimiteurs BOS/EOS, validation holdout
  et reprise. Un tokenizer compatible déjà étudié est réutilisé sans réapprendre
  systématiquement BPE ; voir le [guide causal](02-User-Guide/12-Transformer-GPT.md).
- `train_t5.lua` entraîne l'encodeur T5 depuis un corpus local; Lumen peut
  charger ce checkpoint ou un checkpoint VAEText pour le conditionnement texte.
- `lumen_text2img.lua` propose une génération interactive dans Viz quand un
  backend graphique est disponible; `--headless`/`--no-viz` conserve une
  génération ponctuelle en ligne de commande.
- `train_lumen_diffusion.lua` consomme les contextes texte et les annotations
  structurées documentées dans le guide des données.
- Les scripts d'entraînement partagés utilisent un résolveur de reprise qui
  compare les checkpoints `epoch_*`, `step_*`, `final` et racine quand leur
  progression est disponible dans les métadonnées.
- L'atelier Annoter sait importer les sidecars du composeur, associer les
  fichiers de masques usuels à leurs images, préremplir les boîtes et classes,
  et choisir un dossier à créer/ouvrir dans une boîte de sélection.
- Un dataset n'est jamais téléchargé automatiquement. Les configs publiques
  utilisent des chemins relatifs sous `datasets/`, dossier ignoré par Git.
- Les checkpoints, caches, CSV, snapshots et rapports sont des sorties locales
  ignorées. Ils ne doivent pas être ajoutés à une pull request.
- `bf16`/`fp16` sont des formats supportés par le système de dtype et la
  sérialisation; cela ne garantit pas que chaque kernel calcule nativement dans
  ce dtype. Vérifiez le runtime et le chemin d'opération.

## Fonctionnalités en construction

Les éléments suivants sont utiles pour contribuer mais ne doivent pas être
présentés comme stables :

- runtime FPGA iCESugar Pro, protocole et validation, non encore routés comme
  backend de graphe général ;
- planner cost/static, calibration, fusion et résidence device restent
  expérimentaux ; les tests ciblés ne démontrent pas un gain universel ;
- prototypes MPK de détection/segmentation ;
- bridge WebSocket Lua et bridges de langages externes ;
- `sd3_5`, dont le registre et le squelette ne constituent pas un pipeline complet.

`Mimir.Layers` inspecte les couches du modèle courant ; pour exécuter une
opération, construisez son graphe puis utilisez `Mimir.Model.forward()`.

## Niveaux de preuve

Une affirmation de support devrait citer au moins l'un des niveaux suivants :

1. **Source** : symbole présent et relié dans CMake/registre.
2. **Construction** : cible compilée dans une configuration précise.
3. **Test** : test CTest ou smoke réellement exécuté.
4. **Matériel** : exécution observée sur le backend concerné.
5. **Workflow** : entraînement/inférence complet avec données et checkpoint.

Un test CPU ne prouve pas CUDA, ROCm, Vulkan, OpenCL ou FPGA. Une compilation
SFML ne prouve pas que toutes les cartes soient visibles dans une fenêtre
réelle. Un chargement partiel de checkpoint ne prouve pas une inférence fidèle.

## Reproduire l'inventaire

Depuis la racine du dépôt :

```bash
# Registre et dtypes chargés
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- --show-archs

# Tests réellement déclarés dans le build courant
ctest --test-dir build -N

# API Lua C++ contre le stub EmmyLua
./scripts/tools/verify_api_sync.sh
./scripts/tools/verify_docs.py

# Références locales accidentelles avant publication
rg -n '/home/|/run/media/|Mimir\.Args|src/LuaScripting\.cpp' \
  README.md docs configs scripts

# Cohérence Git et JSON
git diff --check
jq empty configs/*.json
```

Les résultats sont propres au checkout et aux options CMake courantes. Mettez
cette page à jour dès qu'un système change de niveau de maturité.
