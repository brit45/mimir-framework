# Monitoring et visualisation

Trouver rapidement le contrat API réel et les paramètres utilisables.

## Vue d'ensemble

Le monitoring Mímir repose sur 2 interfaces complémentaires :

- `HtopDisplay` : rendu terminal temps réel (métriques, progression, gradients, mémoire, logs).
- `Visualizer` (SFML, Qt, GTK ou Web au choix à la compilation) : rendu graphique des images/tensors intermédiaires, panels interactifs et réglages live.

En pratique, les deux passent par `AsyncMonitor`, qui met à jour l'UI sans bloquer l'entraînement.

La [projection 3D des poids et activations](../02-User-Guide/18-Projection-3D.md)
utilise le graphe courant de toute architecture native ou personnalisée.

Les options CLI communes sont indépendantes :

```bash
./run_mimir.sh --lua script.lua -- --viz          # Viz uniquement
./run_mimir.sh --lua script.lua -- --htop         # Htop uniquement
./run_mimir.sh --lua script.lua -- --viz --htop   # Viz et Htop
```

Avec le flag `--htop`, l'interface terminal est préparée immédiatement mais
n'entre en écran alternatif qu'à la première métrique du run. Les messages de
configuration, le chargement du modèle et les instructions de lancement restent
ainsi visibles jusqu'à la fin de l'initialisation. Viz démarre immédiatement.

**Public concerné :** Développeur et utilisateur intermédiaire/avancé.

> **Prérequis**
>
> Connaître les commandes de base de Mímir.

## `Mimir.Htop`

- `create()`
- `update()`
- `render()`
- `clear()`
- `enable(bool)`

`create` accepte aussi une table : `enable_htop=true`, `enable_viz=false`,
`defer_htop=false`, `viz_config={...}`, ainsi que les options CSV décrites
ci-dessous. `update` accepte une table de métriques ; les appels positionnels
historiques restent compatibles. Ce sont les chemins natifs d'entraînement qui
publient les métriques spécifiques au modèle, pas une introspection automatique
de n'importe quelle table passée à `update`.

État réel des méthodes de cycle de vie : `create()` démarre `AsyncMonitor`,
`render()` est conservée pour compatibilité mais le rendu est automatique, et
`enable(false)` arrête le monitor. `enable(true)` ne redémarre pas un monitor
arrêté : il faut rappeler `create()`.

L'export CSV est unique et appartient au thread d’écriture d’`AsyncMonitor`,
indépendamment de htop et de la Viz. Fermer la fenêtre Viz laisse l’entraînement
et l’écriture CSV actifs ; htop peut la rouvrir. Le CSV existant est relu au
démarrage pour restaurer le graphe, puis les nouvelles métriques sont ajoutées
sans effacer l’historique. Les options `csv`, `csv_enabled`, `csv_path` et
`csv_file` de `Htop.create()` configurent cet export commun.

Le CSV possède un schéma commun fixe. Les métriques supplémentaires affichées
dans **Métriques du modèle** ne deviennent pas automatiquement des colonnes CSV.
Par exemple, les diagnostics posterior de VAEConv sont transmis à HTOP et à la
Viz par son contrat natif, mais ne sont pas ajoutés au CSV commun.

### Graphe du modèle dans HTOP

Pendant un run, `G` ouvre ou ferme l'arbre nodal du modèle et du planner actif.
La vue utilise le snapshot immuable publié par la configuration runtime ; elle
n'inspecte pas directement les layers en cours d'exécution. Les nœuds indiquent
leurs dépendances, runtime, capacité, fusion et tailles disponibles. Le panneau
signale explicitement quand le planner n'a fourni que la topologie, et les
placements ne représentent pas une trace GPU.

Dans cette vue, `↑`/`↓` ou `Tab` sélectionnent un layer, `Page Up`/`Page Down`
déplacent la sélection par page, `←`/`→` déplacent la vue horizontalement et
`R` réinitialise sélection et position. `G` ou `Esc` ferme le graphe et reprend
l'affichage HTOP.

### Usage recommandé

1. Créer/activer le monitor en début de run.
2. Pousser des `Metrics` régulièrement (`updateMetrics`).
3. Laisser le rendu asynchrone faire l'affichage.

### Ce que vous voyez dans le terminal

- progression epoch/batch,
- loss courante + moyenne,
- type de `recon_loss` réellement utilisé (ex: `mse`, `l1`, `bce_logits`),
- composantes (KL, wasserstein, etc.),
- gradients, mémoire, ETA, logs.

Notes :

- Le label de la métrique de reconstruction est dynamique et suit `recon_loss_type` quand fourni.
- Si `recon_loss_type` est absent, l'affichage retombe sur un label générique `RECON`.
- `C` ouvre ou ferme la section de configuration du modèle. Les flèches sélectionnent
  un paramètre ; `Entrée` commence l'édition puis valide la nouvelle valeur.
- Les lignes `DIRECT` sont appliquées au prochain point de synchronisation du calcul.
  Les lignes `FIXE` restent consultables mais nécessitent une reconstruction ou un redémarrage.
- `Tab` ou `↑`/`↓` sélectionne `RECON`, puis `←`/`→` ou `-`/`+` change la loss pendant l'entraînement.
- `R` restaure tous les paramètres live, dont la reconstruction loss, à leur valeur native.

## `Mimir.Viz`

Le choix de l’hôte graphique conserve cette API. Voir [Configurer le backend Viz](../02-User-Guide/17-Viz-Backends.md) pour les dépendances, commandes et limites.

- `create()`
- `initialize()`
- `is_open()`
- `process_events()`
- `update()`
- `add_image(...)`
- `set_detections(detections, target)`
- `show_detections(bool)`
- `update_metrics(...)`
- `add_loss_point(...)`
- `clear()`
- `set_enabled(bool)`
- `save_loss_history(path)`
- `backend()`
- `configure(scene)`
- `poll_events(timeout_ms)`
- `set_validation(state)`
- `validation_enabled()`

`initialize()` vérifie que la fenêtre existe et est ouverte.
`process_events()` est conservée pour compatibilité : les événements sont déjà
traités par `AsyncMonitor`. `update()` relie le modèle courant au contrôle 3D et
sert sur le thread Lua une demande de capture ou une actualisation du mode
**Temps réel** arrivée à échéance ; le rendu reste asynchrone. Sans demande et
avec ce mode désactivé, cet appel ne force pas de capture.

`add_loss_point()` est un no-op ; `update_metrics()` alimente l'historique.
`set_enabled(false)` arrête le monitor, mais `set_enabled(true)` ne le redémarre
pas : utiliser `create()`.

### Démarrage / activation

- En script/JSON : active la visualisation via la config du run (`visualization.enabled=true`).
- En C++ : instancie `Visualizer`, puis `initialize()`, ensuite boucle `process_events()` + `update()`.

### Masquer et rouvrir les sections

Dans la vue principale, un **clic droit** ouvre la boîte **Gérer les sections**,
même si tous les panneaux ont été fermés. Les sept cases correspondent aux
panneaux de la Viz et reprennent leurs noms actuels, y compris ceux définis par
un script. Cocher une case rouvre le panneau ; la décocher le masque.
**Tout afficher** rouvre les sept panneaux.

Les choix s'appliquent immédiatement et sont mémorisés dans la dernière
disposition de `viz_ui_settings.json`. Les positions et tailles sont conservées ;
un panneau rouvert est ramené dans les limites de la fenêtre si nécessaire.
Hors de la boîte, les slots restent accessibles avec `S` + `0–9` pour sauvegarder
et `0–9` pour charger. Ces actions concernent l'interface et ne changent ni les
paramètres du modèle ni l'état de pause de l'entraînement.

Dans la boîte, `↑`/`↓` ou `Tab` sélectionnent une ligne et `Espace`/`Entrée`
l'active. La molette parcourt la liste lorsque la fenêtre est trop petite.
**Fermer**, la croix, `Échap` ou un clic extérieur ferment la boîte en conservant
les choix. Les événements de la boîte ne déclenchent pas les contrôles situés
dessous.

Le zoom et les autres boîtes gardent leurs commandes propres. Dans le picker
d'un tenseur zoomé, le clic droit continue d'effacer la sélection ; quitter
d'abord le zoom avec `Échap` pour accéder à la gestion des sections.

### Projection 3D du modèle courant

Le bouton **Générer la projection 3D** de **Pilotage** ouvre le nuage du modèle
actif. Nom, architecture, familles de vecteurs et dimensions viennent de la
capture effective. Après une première projection, le bouton permet de
l'afficher ou de l'actualiser.

Dans le mode structurel des poids, la case **Temps réel** de l'en-tête de la vue
3D est désactivée par défaut.
Cochée avec la vue ouverte, elle déclenche des captures aux points sûrs du modèle,
avec un intervalle d'au moins 250 ms en fonctionnement continu. La capture et
l'ACP du modèle courant terminent avant la prochaine capture automatique. Fermer la vue
ou décocher la case suspend ces captures ; la préférence est conservée à la
réouverture et au changement de modèle. Le libellé **Selon les étapes du modèle**
indique que la cadence dépend aussi du calcul du modèle.

La capture est commune à `Model` : filtres Conv1d/2d/3d/4d, matrices linéaires,
embeddings, normalisations, récurrence, paramètres génériques et activations
disponibles. Les poids partagés sont dédupliqués. Elle utilise au maximum 512
vecteurs, 262 144 valeurs FP32 au total et 2 048 valeurs par fenêtre de vecteur.
Ces bornes portent sur les valeurs copiées, pas sur toute la mémoire du worker.

Une demande est servie après une mise à jour d'optimiseur, après un forward avec
`training=false`, ou par `Viz.update()`. En pause, elle attend le prochain appel
sur le thread du modèle. La fin de `Model.train()` publie une capture finale
quand la Viz existe. Pour actualiser après l'entraînement, conserver une boucle
Lua qui appelle `Viz.update()` et `Viz.poll_events(50)`.

Le worker calcule l'ACP sur des copies immuables. Les échantillons sont
normalisés, complétés par des zéros lorsque leurs dimensions individuelles
diffèrent, puis centrés ; une échelle commune conserve les proportions des trois
axes. Cette projection structurelle ne suppose pas un espace sémantique commun
à toutes les couches. Un changement de modèle invalide les anciennes captures,
y compris les résultats d'un calcul encore en cours.

La case **Répartition des neurones**, désactivée par défaut, affiche les unités
et canaux du graphe effectif, avec ses vraies dépendances. L’axe X suit la
profondeur du routage ; Y et Z disposent les unités par couche. Cette vue
topologique n’utilise pas l’ACP. Les marqueurs précisent leur nature et les
opérations de largeur inconnue sont identifiées sans faux compte de neurones.
La capture est bornée à 4 096 points et 8 192 liens structurels agrégés, avec
comptes totaux connus et échantillonnés distincts. Ces liens indiquent le routage
entre couches ; ils n’expriment pas une intensité de poids apprise.
La capture manuelle et **Temps réel** suivent les points sûrs du mode des poids.
Ce mode, **Poids actifs (DL)** et **Corrélations du corpus** sont exclusifs ;
décocher ces modes revient au nuage des vecteurs.

La case **Poids actifs (DL)**, désactivée par défaut, ouvre une vue topologique
stable des coefficients réels et active **Temps réel**. Cyan/orange/gris
distinguent les signes positif/négatif/nul ; épaisseur et luminosité indiquent
l’amplitude. La variation entre captures de même identité souligne les
coefficients modifiés, avec les étapes d’optimiseur comparées. La première
capture n’a pas encore de référence de variation. Les correspondances de
paramètres fiables sont rendues comme liens pondérés ; les autres coefficients
restent des marqueurs de paramètres identifiables dans leur couche.
La capture garde au plus 2 048 coefficients, avec un rendu limité à 4 096 points,
2 048 liens pondérés et 1 024 flux structurels de contexte. Un changement de
mode, modèle ou schéma invalide les références de variation.
La capture suit les points sûrs, après mise à jour des poids pendant
l’entraînement, sans forward supplémentaire ni chargement de bloc froid.
La cadence automatique reste bornée à 250 ms ; **Temps réel** décoché fige la
vue après la capture éventuellement en cours. Le signe d’un poids et sa
variation décrivent les paramètres ; le mode des corrélations observe les
activations.

La case facultative **Corrélations du corpus** remplace ce nuage par des
composantes d’activations internes observées après les forwards réels. Après au
moins huit observations et deux composantes variables, les liens montrent les
coefficients de Pearson calculés avant l’ACP : cyan positif, orange négatif.
Le seuil `|r|` et **Effacer** pilotent le diagnostic. **Tous les candidats valides**
affiche toutes les paires calculées avec `|r| ≥ 0,30`, jusqu’à 8 128 liens,
au lieu des 512 plus forts filtrés par le seuil visuel. Décocher cette case
rétablit le seuil précédent ; le choix est conservé dans les exports 3D.
La fenêtre garde au plus
128 observations et 128 composantes ; elle échantillonne les entrées parcourues
pendant l’activation, sans relire le corpus ni lancer de forward supplémentaire.
Dans ce mode, **Temps réel** actualise la projection ; la collecte continue
tant que la vue et le mode sont ouverts. Le premier graphe calculable apparaît
automatiquement. Les compteurs distinguent la collecte du graphe affiché et
indiquent sa provenance entraînement/inférence.

Dans la vue : glisser pour tourner, molette pour zoomer, `R` pour rétablir le
cadrage, survol pour inspecter et `Échap` pour fermer. Les marqueurs sont limités
au canevas ; la légende reste hors de cette zone.

Le bouton **Exporter 3D**, ou `X` avec la vue 3D ouverte, sauvegarde la dernière
projection affichée dans un nouveau dossier
`exports/graph_3d_<mode>_step_<step>_<horodatage>/`. Le dossier contient `vue.png`
(cadrage courant), `graph.json` (points, liens, valeurs et métadonnées) et
`graph.html` (rotation, zoom et survol dans un navigateur, hors ligne et sans
CDN). L'interface affiche le chemin du HTML. Les quatre modes sont pris en charge.
L'export garde l'échantillon complet
de la projection, sa caméra, son seuil et ses identifiants de capture ; il ne
relit pas le modèle actif. Les valeurs des poids actifs restent celles de la
capture exportée, même si l'entraînement continue. Aucun checkpoint complet
ni historique futur n'est inclus.

Le bouton **Organique [O]** active un style arrondi et des liaisons courbes,
également disponible dans le HTML. `AsyncMonitor` écrit les projections publiées
et les dernières métriques connues dans un fichier annexe JSONL à côté du CSV,
avec un lecteur HTML chronologique hors ligne. Voir les
[détails de relecture et de synchronisation des mesures](../02-User-Guide/18-Projection-3D.md#rendu-organique-et-relecture-métrologique).

En C++, un `Visualizer` isolé ne branche pas automatiquement les poids : le
contrôle d'`AsyncMonitor` doit être associé à `Model`. Les wrappers Lua font ce
branchement à la création, au rechargement et aux points d'exécution communs.
Voir le [guide complet et son exemple sans dataset](../02-User-Guide/18-Projection-3D.md)
et le [pipeline interne](../04-Architecture-Internals/04-Monitoring-Htop-Visualizer.md#projection-3d-du-modèle-courant).

### Rapport HTML du run

Dans **Pilotage**, **Générer le rapport HTML** lance
`scripts/tools/show-graph.lua` en arrière-plan. Le modèle, l'optimiseur, les
réglages de validation et le dossier de sortie viennent de la configuration
active. Le générateur reçoit `--strict-context` et un snapshot des métriques du
moniteur ; il ne complète pas les champs absents avec la configuration d'un
autre modèle. Un checkpoint RawFolder complet du dossier courant est transmis
s'il est disponible.

Le rapport produit `<modèle>_graph_report.html`, `<modèle>_graph_report.md`,
`<modèle>_graph_metrics.csv` et un journal dans le dossier de checkpoints du run.
L'interface affiche l'état puis le chemin du HTML ; ouvrez ce fichier dans un
navigateur. Avant la première métrique, la génération ne dispose pas encore de
données et signale cette absence. Le bouton permet ensuite de régénérer le
rapport avec un nouveau snapshot, sans arrêter l'entraînement.

### Statistiques lisibles et métriques du modèle

Dans le **TIP INSPECTOR**, chaque partie des **STATISTIQUES DU TENSEUR** possède
son titre et son encadré : mode, contexte fourni, propositions top-1, accord et
marges selon le contenu publié par le modèle. Les textes sont renvoyés à la
ligne et accessibles à la molette, sans raccourcissement dans l'inspecteur. Les
pixels colorisés restent décrits séparément des valeurs brutes du tenseur.
Les panneaux réservent une marge aux barres de défilement afin de préserver
l'accès aux contrôles.

Le causal publie notamment `cross_entropy`, `perplexity`,
`next_token_accuracy_pct`, les compteurs de tokens, la marge top-1/top-2 et les
métriques de validation disponibles. La ligne **SIGNAL** suit son objectif de
cross-entropy. VAEConv publie les statistiques de `mu`, `logvar`, de l'écart-type
posterior et des gradients de sa tête `logvar` ; la mesure d'échantillonnage
latent est absente si son tenseur n'est plus disponible. Ces mesures décrivent
les valeurs et gradients observés, sans attester une compréhension sémantique.

### Contrôles clavier (UI)

- `H` : aide overlay.
- `C` : ouvre ou ferme la section de configuration du modèle.
- `M` : bascule mode de rendu Blocks/Layers (`HEATMAP` / `REEL`).
- `K` : change la palette heatmap (`CLASSIC` -> `TURBO` -> `INFERNO` -> `VIRIDIS`).
- `A` : active/désactive le lissage des previews de blocks.
- `B` : affiche/masque les encadrés de détection (`BOX ON/OFF`).
- `R` : resynchronise/rebuild textures + reload hints architecture.
- `N` : ouvre le graphe nodal du modèle ; `N` ou `Esc` le ferme.
- Dans le graphe : flèches ou molette pour naviguer, clic sur un nœud pour
  afficher ses détails, `R` pour revenir à l’origine et `X` pour exporter le
  graphe complet en PNG.
- Dans la vue 3D : `R` rétablit le cadrage ; `X` exporte la projection affichée
  en PNG, JSON et HTML interactif autonome.
- `E` : ouvre la boîte de sélection du snapshot PNG.
- `Tab`, `F1..F5`, `←/→` : navigation focus/éléments.
- `Z`/`Entrée` : zoom, `Esc` : quitter zoom.
- `I` dans le zoom : afficher ou masquer l’info-bulle du picker.
- Survol d’un tip zoomé : bulle des valeurs brutes et des indices du tenseur,
  jusqu’à 16 canaux échantillonnés. Les coordonnées suivent le redimensionnement
  de la vignette ; la valeur affichée précède la colorisation. Si le tip ne
  fournit pas de valeurs brutes, la bulle indique explicitement les pixels uint8.
- Dans le zoom, glisser sur l’image fige une sélection inspectable dans le panneau
  latéral ; la molette parcourt les valeurs et `Esc` / clic droit efface la sélection.
  Le panneau affiche les métadonnées du tensor, les statistiques signées et les
  activations les plus fortes avec leurs indices source. Une grille `/vec` est une
  projection de preview d’un vecteur : l’indice identifie une dimension apprise,
  pas à lui seul un concept ou un label sémantique.
- Bouton `PAUSE training` / `PLAY / Reprendre` dans Metrics : suspend/reprend
  le calcul entre étapes, tout en laissant les interfaces actives. Fermer la
  Viz pendant la pause libère l’entraînement ; `STOP` reste disponible.

Pour Lumen, les tokens et l’encodage affichés viennent de `text_ids` et de
`lumen/text/context` du dernier forward. Les vignettes de projection et de
compréhension suivent les couches texte capturées avec ce contexte.

### Encadrés de détection

`Mimir.Viz.set_detections()` reçoit des boîtes normalisées dans `[0,1]`, avec
un label, un score et un identifiant de classe optionnels. La cible vaut
`dataset` par défaut, ou `output`/`both`. Un tableau vide efface l'overlay.

```lua
Mimir.Viz.set_detections({
  {x1=0.12, y1=0.18, x2=0.64, y2=0.81, label="person", score=0.94, class_id=0}
}, "dataset")
```

Le badge `BOX ON/OFF` du panneau Context, le raccourci `B` et
`Mimir.Viz.show_detections(bool)` permettent de masquer ou réafficher les
encadrés sans perdre les prédictions publiées. Les architectures YOLO, SSD et
R-CNN décrivent encore un décodage externe : leur décodeur doit transmettre les
boîtes finales après NMS avec cette API.

Dans la section de configuration, les flèches ou la molette sélectionnent une
ligne. `Entrée` ou un clic commence l'édition, puis `Entrée` valide la valeur ;
`Esc` annule. Les paramètres `DIRECT` sont appliqués à la prochaine frontière de
calcul sûre, tandis que les paramètres `FIXE` indiquent qu'une reconstruction ou
un redémarrage est nécessaire.

Le panneau est alimenté automatiquement quand un modèle courant est utilisé par
`Model.train()`, `Model.forward()` ou `Model.optimizer_step()`. Il n'existe pas de
fonction Lua séparée pour écrire une ligne du panneau. Les paramètres `DIRECT`
réellement publiés dépendent du modèle et de l'optimiseur :

- entraînement : `learning_rate`, `lr_warmup_steps`, `validation_enabled` ;
- modèle : `grad_clip_norm`, skip connections, dropout, cache KV, latent
  stochastique et seuils NMS quand les couches correspondantes existent ;
- VAE : `kl_beta`, `kl_warmup_steps`, `recon_loss`, poids de loss et constantes
  Huber/Charbonnier/Gaussian NLL ;
- optimiseur courant : `beta1`, `beta2`, `epsilon`, `weight_decay` et
  `rmsprop_alpha`.

Toutes les autres clés de configuration sont affichées en `FIXE`. Modifier une
dimension, un nombre de couches ou une forme de tenseur exige de reconstruire le
modèle ; le panneau ne réalloue ni les poids ni l'état de l'optimiseur.

### Badges et état live

Le panneau `Blocks / Layers` affiche des badges en en-tête :

- mode de rendu courant (`HEATMAP` ou `REEL`),
- état du lissage (`LISSAGE ON/OFF`),
- palette active (`PAL CLASSIC/TURBO/INFERNO/VIRIDIS`).

Comportement de rendu important :

- Les sorties de packing (`out_concat`, `out_pack`, etc.) sont exclues des previews image.
- Dans la section `Outputs`, `recon` est priorisé juste avant `diff/resdiff` pour comparaison directe.
- En mode `HEATMAP`, la palette choisie recolorise les tips de `Blocks / Layers`,
  y compris les activations autres que `resdiff_abs`. Les layers `Identity`
  restent non recolorisés ; le mode `REEL` montre les couleurs naturelles.
- Les sections de tips sont regroupées par catégorie (par exemple `Encoder`,
  `Latent`, `Decoder`, `Outputs`). La section `Outputs` ne reçoit pas de tip
  artificiel `outputs` en tête : seuls les tips issus du modèle y apparaissent.

### Graphe nodal et export PNG

`N` affiche une vue arborescente du snapshot de graphe/planner associé au
modèle actif. Les nœuds sont placés par profondeur et par voie ; les arêtes
orthogonales entrent et sortent des cartes par des ports distincts. Les
embranchements et les connexions longues utilisent des rails séparés afin que
les trajets passent entre les cartes plutôt que sous elles. La sélection d'un
nœud affiche ses entrées/sortie, le runtime, les paramètres, les formes et les
informations de fusion. Les placements et octets sont des estimations du
planner, pas une trace garantie des kernels effectivement exécutés.

Dans cette vue, `X` exporte le graphe entier dans
`snapshots/model_graph_<horodatage>.png`. Le rendu est assemblé par tuiles pour
éviter la limite de dimension d'une texture de rendu unique ; l'image finale
conserve les dimensions complètes de la topologie.

`E` ouvre une boîte de dialogue permettant de choisir le contenu du snapshot :
fenêtre actuelle, tips des couches, projection et sortie. Les éléments absents
sont signalés comme indisponibles. Les tips sont ajoutés sous la capture de la
fenêtre, groupés par catégorie et accompagnés de leurs labels complets, renvoyés
à la ligne. L'export est écrit dans `snapshots/viz_snapshot_<horodatage>.png`.

### Annotations sur les sorties et correspondances de features

Quand l’image consommée possède les annotations JSON de l’éditeur, l’input et les taps spatiaux de sortie montrent les bbox, les relations entre leurs centres, les points et segments du squelette et ses attaches aux entités. Une même couleur identifie une entité d’un tap à l’autre ; les relations sont jaunes et les attaches aux entités roses. Les identifiants, noms et prédicats figurent dans les informations du tap. Les zones renforcées par la pondération spatiale reçoivent une teinte orange. Les vecteurs de packing et les matrices textuelles restent exclus.

Après `resdiff_abs`, dans la même section, deux taps sont ajoutés :

- `annotation_features` : chaque région bbox fournit un prototype moyen des features spatiales capturées. La couleur indique le prototype le plus similaire au vecteur local, et sa luminosité la similarité cosinus. Les informations indiquent la couche source, le nombre de canaux échantillonnés, les similarités entre régions liées et entre points de squelette et entités. Les valeurs inspectables sont ces similarités ; les valeurs brutes des taps source restent intactes.
- `annotation_impact` : poids régionaux de la loss rasterisés sur la grille du tap. `1` est neutre, orange signifie un renforcement, bleu une réduction. La géométrie utilise les mêmes coordonnées normalisées que l’input et les sorties. Le poids courant et les poids de rôles sont capturés avec l’annotation.

Ces vues montrent des correspondances de features et la pondération spatiale ; elles ne mesurent pas l’effet causal d’une annotation sur la prédiction. Sans features brutes alignées, `annotation_features` affiche uniquement la géométrie et indique explicitement que les features sont indisponibles. Les annotations de l’image et les taps sont publiés ensemble comme un snapshot ; aucun calcul de pooling ou de rasterisation ne conserve le verrou de publication. Les limites manuelles du nombre de taps continuent à s’appliquer.

### Progression Training

Le panneau `Training` affiche 2 barres distinctes :

- barre globale : avancement total du run (epochs + batch courant),
- barre batch : avancement du batch courant (sous la barre globale).

Détails :

- les deux barres utilisent un lissage visuel pour limiter les sauts,
- la barre batch est colorée selon `batch_time_ms` (rapide=vert, plus lent=orange/rouge),
- en absence de total epochs, la barre globale passe en mode fallback `running`.

### Sliders live (panneau Metrics)

Les sliders live supportent :

- drag souris sur le track/thumb,
- graduation visuelle (repères min/max),
- saisie directe via cellule de valeur (clic).

La ligne `Recon` est un sélecteur cyclique cliquable. Elle applique immédiatement l'une des losses supportées : `mse`, `mae`, `huber`, `charbonnier`, `gaussian_nll` ou `bce`.

Formats de saisie acceptés :

- décimal : `0.00025`, `0.5`, `1.0`
- scientifique : `1e-4`, `2.5e-3`, `5e4`

Validation édition :

- `Entrée` : applique,
- `Backspace` : efface,
- `Esc` : annule édition.

### Bonnes pratiques

- Garder `visualization.update_interval_ms` autour de `16..33` ms pour un bon compromis fluidité/coût.
- Utiliser la resync (`R`) seulement quand nécessaire (debug), pas en continu.
- Si vous envoyez des taps volumineux, réduisez la fréquence (`viz_taps_every_steps`) pour limiter la charge.

Notes :

- Le backend est choisi à la compilation : `SFML`, `QT`, `GTK`, `WEB` ou `NONE`.
- `Viz.backend()` retourne ce choix même avant `Viz.create()`.
- Le runtime peut publier des “viz taps” pendant `Model.forward()` si un monitor async est actif.

## Étapes suivantes

- [Page précédente : API : mémoire](14-Memory.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : API : `Mimir.Serialization`](16-Serialization.md)

## Scènes et événements pilotés en Lua

Les backends **SFML, QT, GTK et WEB** partagent `Viz.configure`,
`Viz.poll_events` et `Viz.backend`. `backend()` retourne le backend compilé,
y compris avant la création ; `NONE` ne permet pas de créer une interface.
L'aide **H** affiche le backend, le moteur de rendu, les raccourcis groupés
et le texte d'aide fourni par le script.

```lua
assert(Mimir.Viz.create({visualization={enabled=true}}))
local UI = dofile(ROOTWORK.."/scripts/modules/viz_ui.lua")
local ui = UI.new({
  panels={{id=2, title="Aperçu", x=20,y=100,w=800,h=500,visible=true}},
  controls={{id="apply",label="Appliquer",x=20,y=20,w=160,h=40}},
  help="Appliquer : recalculer la prévisualisation.",
  events=true,
})
ui:on("apply", function(event) print("Appliquer",event.x,event.y) end)
while Mimir.Viz.is_open() do ui:dispatch(30) end
Mimir.Viz.set_enabled(false)
```

| ID panneau | Contenu |
| --- | --- |
| 0 | Context |
| 1 | Blocks / Layers |
| 2 | Generated |
| 3 | Training |
| 4 | Metrics |
| 5 | Graph |
| 6 | Output |

`configure(scene)` valide les données et retourne `true` ou `false, erreur`.
Les champs omis restent inchangés. `panels` modifie les panneaux indiqués ;
`controls` remplace la liste de boutons (table vide pour les retirer).
`image_size` règle la taille des aperçus Generated (32 à 1024 pixels, défaut 200).
Maximum : 64 entrées par liste. Les coordonnées sont en pixels dans la fenêtre ;
le script peut réagir à `resize` pour adapter son placement. Les contrôles sont
rendus au-dessus des panneaux, sous les overlays d'aide et de zoom. Un clic sur
un bouton est consommé, sans déclencher le panneau situé dessous.

`poll_events(timeout_ms)` retire les événements disponibles, avec une attente
optionnelle de 0 à 1000 ms. Types : `configured`, `click` (`id`, `x`, `y`),
`close`, `resize` (`width`, `height`). Avec `events=true`, s'ajoutent `key`
(`code` enum clavier Viz, `control`, `shift`, `alt`), `text` (`unicode`),
`pointer_move`, `pointer_down`, `pointer_up` (`x`, `y`, et `button` pour les deux
derniers). Avec `events=true`, les événements bruts permettent d’observer les entrées,
mais les raccourcis intégrés gardent leur priorité. Pendant une saisie,
`capture_keys=true` réserve le clavier au script : notamment `N` et `C` ne
doivent pas ouvrir le graphe ou la configuration au milieu d’un prompt. La file est bornée à 256 entrées : les plus anciennes sont
supprimées en cas de saturation. Les clics ne rejouent pas automatiquement un
callback ; `viz_ui.lua` les distribue pendant `dispatch`, sur le thread Lua.
Les erreurs des callbacks remontent au script.

Les configurations sont transmises par une boîte aux lettres protégée ;
plusieurs mises à jour non encore appliquées fusionnent leurs champs de premier
niveau. Les modifications de panneaux fusionnent par ID ; la dernière liste
de boutons remplace la précédente. L'événement
`configured` confirme leur application par le thread de rendu. Les contrôles
restent des boutons rectangulaires ; cette API n'est pas un constructeur de
widgets Qt/GTK natifs.

### Éditeur d'image VAEConv

```bash
./run_mimir.sh --lua scripts/inferences/edit_vae_conv.lua -- \
  --checkpoint checkpoint/vae/epoch_0010 \
  --input photo.ppm --output outputs/edition.ppm
```

Le script accepte un checkpoint `vae_conv` entraîné en **raw_folder** ou en
**SafeTensors natif**, et une image PNG/JPEG/BMP ou PPM P3/P6 RGB. Il partage
`scripts/modules/vae_conv_inference.lua` avec `infer_vae_conv.lua` : la
configuration et les couches sauvegardées déterminent la géométrie et le
graphe C++ construit par le registre. Pour un ancien SafeTensors sans
architecture intégrée, `--config chemin/config_du_checkpoint.json` doit
fournir la configuration exacte. `--format raw_folder|safetensors` et
`--dtype float32|float16|bfloat16` sont également disponibles.

Les graphes Conv2d, Conv3d et Conv4d conservent le même contrat image RGB HWC
et latent CHW aplati ; les axes supplémentaires des modes ND sont singleton.
`--conv-mode 2d|3d|4d` vérifie le mode du checkpoint et refuse une valeur
incompatible. L’upsampling, les normalisations, le contexte spatial et le
chemin de détails latents sont repris du graphe sauvegardé, sans activer de
nouvelles branches sur un ancien checkpoint. La résolution native de
`conv_transpose` en `nearest_conv` pour les modes ND reste appliquée.

Les formats PNG/JPEG/BMP utilisent le chargement RGB natif Mímir avec
redimensionnement bicubique ; le PPM utilise le voisin le plus proche si nécessaire. Le script sélectionne directement
`vae_conv/mu` en inférence déterministe, charge
`vae_conv_decode`, puis permet de décaler un canal latent, mélanger le résultat
avec la source, réinitialiser et enregistrer en PPM. La source et le dernier
résultat sont présentés côte à côte ; `Z` agrandit l'image sélectionnée.
`L` bascule vers une galerie scrollable de tous les canaux du latent courant,
normalisés indépendamment pour rendre leur structure spatiale inspectable.
`Ctrl+U` permet de choisir une autre image puis recalcule son latent sans fermer
l'éditeur. Le script utilise un sélecteur système (`zenity`, `kdialog` ou `yad`)
s'il est disponible ; sinon, le chemin peut être saisi directement dans la Viz
et validé avec `Entrée`.

La galerie et les décalages de canaux portent sur la moyenne `mu`, avant
ajout du prior, avec le même ordre CHW que le RAW de `infer_vae_conv.lua`
(float32 little-endian sans en-tête). Si `use_encoder_prior=true`, son biais
appris est ajouté à une copie du latent avant le forward du décodeur ; il ne
modifie pas les valeurs affichées dans la galerie.

Ces modifications latentes ne sont pas des commandes sémantiques et leur effet
dépend de l'entraînement. Le décodeur autonome utilise des skips encodeur nuls
si le modèle comporte des connexions de ce type ; sa sortie à force zéro peut
donc différer d'une reconstruction par le VAE complet. `--mix 0` conserve
l'image source redimensionnée. L'interface ne sauvegarde pas à la fermeture :
utiliser **Enregistrer**. Les fichiers existants au chemin de sortie sont
remplacés à chaque enregistrement.

`--headless --strength 0.2 --channel 1 --mix 0.8` calcule et enregistre une fois,
sans interface. `--help` détaille les options sans charger de modèle. Le script
vérifie la présence et la taille de tous les poids requis avant le forward,
y compris pour le décodeur partiel. Une topologie incompatible ou un poids
manquant produit une erreur explicite ; le script ne modifie ni les poids ni
le checkpoint.


### Préparer une inférence Lumen dans Viz

Le script `scripts/inferences/lumen_text2img.lua` ouvre automatiquement un panneau
de génération si un backend Viz utilisable est disponible. Le checkpoint Lumen,
son tokenizer et son VAE sont chargés avant l’ouverture de ce panneau. Les
options `--prompt`, `--seed`, `--steps`, `--guidance` et `--out` préremplissent
les champs ; aucune inférence n’est lancée avant un clic sur **Générer**.

Cliquer sur un champ pour saisir sa valeur. Le texte est initialement sélectionné :
la saisie le remplace. `Ctrl+A` sélectionne tout, les flèches déplacent le curseur,
`Backspace` efface un caractère UTF-8, `Entrée` valide, `Tab` passe au champ suivant
et `Échap` annule l’édition du champ. Hors édition, `H` rappelle l’aide, `Tab`
ouvre le premier champ et `Échap` ferme le panneau. La seed et les étapes doivent être des
entiers compatibles avec le runtime ; les étapes sont positives et la guidance
est finie et positive ou nulle. Le runtime plafonne les étapes effectives à la
taille du scheduler enregistré. Un prompt vide permet le conditionnement vide.

Chaque clic sur **Générer** prend un snapshot des réglages validés, génère
l’image et écrit le PPM au chemin choisi. L’aperçu et les réglages restent
ouverts après le calcul pour permettre un nouvel essai. Le même chemin de sortie
est remplacé lors des générations suivantes. **Réinitialiser** restaure les
réglages de lancement ; **Fermer**, avant la première génération, quitte sans
inférence ni écriture d’image. Une erreur de génération est affichée et laisse
le panneau utilisable.

Le checkpoint, le tokenizer et la géométrie restent ceux chargés au lancement.
Les réglages modifiés s’appliquent à la prochaine génération, pas à un calcul en
cours. Sans Viz utilisable, ou avec `--headless` / `--no-viz`, le script conserve
sa génération unique en ligne de commande. Le panneau utilise
la scène et les événements communs de l’API Viz, sans dépendance à des widgets
propres à Qt ou GTK.
