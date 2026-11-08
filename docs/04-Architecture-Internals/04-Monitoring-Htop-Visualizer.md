# Monitoring (HtopDisplay / Visualizer / AsyncMonitor)

Comprendre le fonctionnement interne exact des composants runtime.

**Public concerné :** Développeur avancé qui modifie le moteur C/C++.

> **Prérequis**
>
> Connaître les bases C++ et la structure du dépôt.

Cette page documente la stack “monitoring” côté C++.

## Sur cette page

- [Source de vérité (C++)](#source-de-vérité-c)
- [Intention](#intention)
- [Architecture globale](#architecture-globale)
- [HtopDisplay (UI terminal)](#htopdisplay-ui-terminal)
- [Visualizer (interface graphique optionnelle)](#visualizer-interface-graphique-optionnelle)
- [AsyncMonitor](#asyncmonitor)
- [Projection 3D du modèle courant](#projection-3d-du-modèle-courant)
- [Pilotage depuis Lua](#pilotage-depuis-lua)
- [Étapes suivantes](#étapes-suivantes)

## Source de vérité (C++)

- `src/HtopDisplay.hpp`
- `src/Visualizer.hpp` / `src/Visualizer.cpp` (backend graphique optionnel)
- `src/AsyncMonitor.hpp`
- [Capture du modèle](../../src/Model.cpp) : `Model::captureLatentSpaceSnapshot()` et ses points sûrs.
- [Snapshots et contrôle 3D](../../src/viz/LatentSpaceProjection.hpp) : copies, ACP, génération et révision.
- [Liaison des scripts au modèle courant](../../src/scriptings/ModelVizProjection.hpp).

## Intention

Le monitoring a 2 objectifs distincts, souvent activés ensemble pendant l’entraînement:

- **HtopDisplay**: UI terminal (style htop), très légère, export CSV.
- **Visualizer**: interface graphique, affichage d’images/frames, courbes loss, contexte dataset et projection 3D.

`AsyncMonitor` orchestre le tout de façon **asynchrone**, pour éviter de bloquer le thread principal.

## Architecture globale

### Qui tourne dans quel thread ?

- Le training loop (Lua/C++) tourne dans le **thread principal**.
- `AsyncMonitor` peut lancer:
  - un thread `htop_thread_` qui redessine l’UI terminal périodiquement.
  - un thread `viz_thread_` qui initialise l'interface graphique puis fait la boucle de rendu.
  - un worker `latent_space_thread_` qui calcule l'ACP des captures 3D déjà copiées.

Pourquoi c’est important:

- Le backend SFML/OpenGL impose que la fenêtre soit créée et manipulée dans **le même thread**. `AsyncMonitor::start()` initialise donc `Visualizer` dans le thread viz, et y conserve les opérations de rendu.
- La capture des poids s'effectue dans le thread qui exécute le modèle, séparément du calcul de l'ACP et du rendu.

### Synchronisation (best-effort)

- Les métriques sont copiées via `AsyncMonitor::updateMetrics()` (mutex + flag `has_update_`).
- Les frames/images destinées au visualizer sont stockées en “pending” (mutex dédié) puis consommées par la boucle viz.

Les échanges emploient des mutex et des indicateurs atomiques. La projection 3D
publie des `shared_ptr` vers des snapshots constants; aucun pointeur vers les
poids actifs n'est transmis à l'interface. Les interfaces affichent le dernier
état disponible, sans imposer un rythme de rendu au modèle.

## `HtopDisplay` (UI terminal)

### Rôle

- Afficher:
  - progression (epoch/batch)
  - loss, métriques diverses (KL, Wasserstein, etc.)
  - mémoire et perf (MB, batches/s, ETA)
  - état optimizer (type/step/betas/eps/weight_decay)
- Exporter un CSV (par défaut `checkpoints/htop_metrics.csv`).

### I/O

- Écrit des codes ANSI via `FdStreamBuf`, de préférence vers `/dev/tty` ou
  une copie du descripteur de sortie ; il peut ainsi conserver l'affichage
  pendant la capture des logs stdout/stderr.
- Utilise `ioctl(TIOCGWINSZ)` sous POSIX ; la version Windows emploie
  actuellement une taille fixe de 120×40 caractères.

### Export CSV (HtopDisplay)

- La structure `CsvRecord` reflète un superset des métriques.
- `saveLossHistoryCsv()`:
  - crée le dossier parent en best-effort
  - écrit l’en-tête + toutes les lignes

### Pièges

- Le rendu nécessite un terminal compatible ANSI ; un descripteur terminal
  indisponible ou une sortie redirigée ne constitue pas une interface graphique.
- Les écritures fréquentes peuvent coûter si le terminal est lent.

## `Visualizer` (interface graphique optionnelle)

### Gating compile-time

`src/Visualizer.hpp` expose 2 implémentations:

- si `ENABLE_VIZ` : classe graphique, avec le backend choisi par `MIMIR_VIZ_BACKEND` (`SFML`, `QT`, `GTK` ou `WEB`).
- sinon: version **headless** API-compatible (toutes les méthodes sont no-op / false).

Le code de scène utilise `vizgfx`, défini par [Graphics.hpp](../../src/viz/Graphics.hpp).
Les en-têtes SFML restent propres au backend SFML.

Conséquence:

- Le code peut appeler `Visualizer` sans `#ifdef` partout.

### Données gérées

Le visualizer stocke typiquement:

- images générées (surcharge historique grayscale 64×64 et surcharge explicite
  avec largeur, hauteur, canaux et statut de validation)
- image dataset courante (+ label)
- texte dataset (raw / tokenized / encoded)
- images de projection / “understanding”
- frames de blocs/layers (`setLayerBlockImages()`)
- métriques d’entraînement + historique de loss
- nuage 3D interactif de vecteurs échantillonnés du modèle courant

### “Architecture awareness”

Le visualizer peut (optionnel) charger un fichier d’architecture (json) pour:

- masquer certains blocs (`hide_activation_blocks`)
- filtrer par noms de layers / outputs / sinks

La logique est best-effort (`maybeLoadArchitecture()`), pensée pour améliorer la lisibilité.

### Export CSV

`Visualizer::saveLossHistory()` exporte une histoire structurée (similaire aux colonnes htop).

Note importante:

- `AsyncMonitor::configureMetricsCsv()` maintient un export unique : Viz est
  prioritaire lorsqu'elle est active, et htop sert de repli lorsqu'il tourne
  seul. Le chemin et l'état d'activation sont communs aux deux interfaces.

### Visibilité des sections

Le clic droit sur la vue principale ouvre la modale **Gérer les sections**,
même lorsque tous les panneaux ont été fermés. `PanelId` décrit les sept
sections : contexte, blocs, générations, entraînement, métriques, graphe et
sortie. `setPanelVisible()` synchronise les anciens flags d'affichage et
enregistre le layout `last` ; rouvrir un panneau conserve sa géométrie en le
replaçant dans la fenêtre si nécessaire.

`openPanelManager()` annule les interactions de déplacement et de saisie en
cours. Les événements de la modale sont consommés avant ceux de la scène du
script ; elle se rend dans `renderPanelManagerOverlay()` via `vizgfx`.
Le clic droit conserve son usage de sélection dans le zoom d'un tenseur :
sortir du zoom avec Échap pour gérer les sections. `VizTest.PanelManager`
vérifie ouverture, fermeture, réouverture, isolation et persistance.

## `AsyncMonitor`

### Rôle (Visualizer)

- Unifier htop + viz derrière une API thread-safe.
- Gérer le lifecycle:
  - `start(enable_htop, enable_viz, viz_config)`
  - `stop()` (join threads, restore cursor)

### Invariants

- `start()` est **idempotent** (peut être rappelé pour activer viz plus tard).
- La destruction (`~AsyncMonitor`) appelle `stop()`.
- Le thread viz:
  - initialise SFML
  - notifie `vizInitOk()` / `vizInitError()`
  - appelle `viz_->shutdown()` dans le même thread à la fin.

### API d’update

- `updateMetrics(Metrics)`
- `addImage(pixels, prompt)`
- `setDatasetImage(...)`, `setDatasetText(...)`, `setDatasetSample(...)`
- `setProjectionImage(...)`, `setUnderstandingImage(...)`

### Pièges (AsyncMonitor)

- Les images peuvent être volumineuses: garder des copies “pending” a un coût.
- Les métriques sont “dernier état connu”: pas d’historique côté `AsyncMonitor` (hormis ce que visualizer/htop accumulent).

## Projection 3D du modèle courant

Le bouton **Générer la projection 3D** de **Pilotage** utilise l'API commune de
`Model`. Il fonctionne pour les architectures du registre et les graphes natifs
personnalisés, sans exiger un `d_model` ou un espace latent explicite. Le titre
reprend le nom et l'architecture effectifs; la légende affiche les familles
présentes. Les commandes utilisateur sont décrites dans
[l'API Viz/Htop](../03-API-Reference/15-Viz-Htop.md#projection-3d-du-modèle-courant).

### Capture et threads

| Étape | Propriétaire | Contrat |
| --- | --- | --- |
| Demande | Interface | `LatentSpaceControl::requestSnapshot()` enregistre une demande, même hors entraînement. |
| Capture | Thread du modèle | `captureRequestedLatentSpace()` copie les vecteurs à un point sûr. |
| Projection | Worker d'`AsyncMonitor` | L'ACP travaille sur un `LatentSpaceRawSnapshot` immuable. |
| Affichage | Thread Viz | Le rendu lit uniquement le `LatentSpaceProjection` publié. |

Les demandes sont consommées après `Model::optimizerStep()` ou à la fin d'un
forward d'inférence. Les bindings peuvent aussi servir la demande depuis
`Mimir.Viz.update()`, sur le thread du script. Le helper
`Mimir::Scripting::bindModelProjection()` lie le modèle courant au contrôle;
`ProjectionTrainingScope` couvre aussi les sorties des branches d'entraînement
spécialisées. Une capture de l'état courant est forcée à la création de la Viz
et à l'initialisation des poids lorsqu'une Viz existe, puis à la sortie de
l'entraînement commun. Dans les chemins Lua de `main.cpp`, le modèle est aussi
synchronisé avant l'attente de fermeture de la fenêtre.

La case **Temps réel** publie deux états dans le contrôle partagé : le choix
de l'utilisateur et l'ouverture de la vue. `beginCapture()` vérifie ces états
sur le thread du modèle et espace les captures automatiques d'au moins 250 ms
avec `steady_clock`. Une seule capture/projection automatique peut être en
cours ; les appels intermédiaires laissent le worker terminer. Les demandes
manuelles et captures forcées conservent leur priorité. Fermer la vue suspend
les nouvelles captures automatiques sans perdre le choix de la case.

Pour une intégration C++ native, cette liaison s'effectue une fois au changement
de modèle, sur son thread, avec le moniteur déjà démarré et une configuration
effective non vide. Exemple de fonction d'intégration :

```cpp
#include "Model.hpp"
#include "AsyncMonitor.hpp"

void bindProjection(Model& model, AsyncMonitor& monitor,
                    const nlohmann::json& effective_config) {
    monitor.configureLatentSpace(effective_config);
    model.bindLatentSpaceControl(monitor.latentSpaceControl());
    model.captureRequestedLatentSpace(model.latentSpaceOptimizerStep(), true);
}
```

Cette fonction ne doit pas être appelée depuis le thread UI pendant une mise à
jour des poids. La capture directe `captureLatentSpaceSnapshot()` est en lecture
seule; le modèle doit néanmoins être stable pendant cette lecture.

### Contenu et limites de la capture

La capture lit les poids hôte déjà résidents du graphe principal et les
activations encore conservées par le modèle. Elle inclut les paramètres fixes,
déduplique les poids partagés et omet les vecteurs contenant des valeurs non
finies. Elle ne recharge pas les blocs dynamiques froids ou compressés et
n'alloue pas de poids manquants. Un graphe sans paramètres peut fournir ses
activations après un forward.

Elle ne parcourt pas récursivement les sous-modèles privés : les encodeur et
décodeur VAE de Lumen, lorsqu'ils sont détenus hors du graphe principal, ne sont
pas ajoutés au nuage. La présence d'un type de modèle dans le registre n'implique
donc pas que tous ses états internes privés soient capturés.

Les budgets communs sont **512 vecteurs**, **262 144 valeurs** au total et
**2 048 composantes par vecteur**. Les vecteurs plus larges sont échantillonnés
par une fenêtre contiguë réelle, identifiée par `original_dimension` et
`feature_offset`. Les filtres de convolution, projections, embeddings et
activations gardent leurs dimensions propres; le modèle n'invente pas une
largeur `d_model` pour les architectures qui n'en ont pas.

### ACP et publication

Le worker normalise chaque vecteur en L2, complète les composantes absentes par
zéro, puis centre les échantillons et calcule les trois premiers axes de l'ACP.
Un facteur d'affichage commun aux trois axes borne le rayon du nuage à 2,5 et
préserve les rapports de variance. Les petits modèles restent valides : une
capture d'un seul vecteur, ou constituée uniquement de vecteurs identiques/nuls,
apparaît à l'origine ; les axes de
rang manquant restent nuls. Seule l'absence de vecteur fini empêche l'affichage.

Il s'agit d'une **vue structurelle des échantillons**. La complétion par zéro ne
crée pas un espace sémantique partagé entre couches; les distances ne mesurent
ni la compréhension ni une similarité sémantique.

Le worker consulte le mailbox au plus tard à son prochain réveil périodique
(100 ms hors temps de calcul). `resetSession()` change la génération et efface
les résultats du modèle précédent. Une capture réserve sa révision dès
`beginCapture()`, avant la copie des vecteurs ; les soumissions directes de
compatibilité reçoivent aussi une révision.
`beginProjection()`, `publishProjection()` et les erreurs du worker vérifient
ces identifiants sous mutex : un calcul obsolète ne peut pas remplacer l'état du
nouveau modèle ou d'une capture plus récente. Une capture finale peut remplacer
une capture en attente pendant que l'ACP précédente termine.

La rotation et le zoom utilisent ces copies projetées. Le dessin, les marqueurs
et leurs zones de survol sont limités au canvas; la vue normale est restaurée
avant de dessiner la légende et les infobulles.

### Couverture de vérification

- [VizTest.GenericLatentSpace](../../Tests/test_latent_space_generic.cpp) : registre, graphes personnalisés, convolutions, poids partagés/fixes, budgets et absence de mutation.
- [VizTest.GenericLatentControl](../../Tests/test_latent_space_control.cpp) : dimensions variables, petits modèles, publication, points sûrs et rejet des générations/révisions obsolètes.
- [VizTest.GenericProjection](../../Tests/test_viz_events.cpp) : bouton, rotation, zoom, canvas, survol et fermeture de la scène.

Ces cibles sont enregistrées dans [Tests/CMakeLists.txt](../../Tests/CMakeLists.txt).
La couverture headless du contrôle ne remplace pas le test graphique, qui peut
être ignoré lorsqu'aucun backend d'affichage utilisable n'est disponible.

## Pilotage depuis Lua

La mise en place/activation du monitoring est exposée côté Lua via les fichiers
`src/scriptings/Lua/luaScripting/LuaScripting*.cpp`, notamment
`LuaScriptingRuntimeAndViz.cpp` pour Viz et `LuaScriptingModelAndRegistry.cpp`
pour le modèle courant. Voir [l'API Viz/Htop](../03-API-Reference/15-Viz-Htop.md).

## Étapes suivantes

- [Page précédente : Backends hardware : CPU / CUDA / ROCm / Vulkan / OpenCL](03-Hardware-Backends.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Internals: AdvancedRAMManager (cache RAM / compression / spill disque)](05-AdvancedRAMManager.md)
