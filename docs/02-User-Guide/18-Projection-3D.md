# Explorer les vecteurs d'un modèle en 3D

Le panneau **Pilotage** de Viz propose une vue interactive des poids, des
activations et de la répartition des unités du modèle courant. La capture commune de `Model` prend en
charge les architectures natives et les graphes personnalisés : MLP,
convolutions, VAE, GAN, Transformers et diffusion.

Le titre indique le modèle et son architecture. La légende présente uniquement
les familles de vecteurs effectivement capturées : embeddings, projections,
filtres de convolution, normalisations, récurrence, paramètres et activations.
Un modèle sans dimension latente explicite peut donc utiliser cette vue.

## Ouvrir et manipuler la vue

1. Activer Viz dans le script avec `Mimir.Viz.create(...)`, ou ajouter `--viz`
   à la commande habituelle d'un script qui utilise les options communes.
2. Construire et initialiser le modèle, ou charger son checkpoint.
3. Dans **Pilotage**, cliquer sur **Générer la projection 3D**. Après une première
   capture, le libellé permet d'afficher ou d'actualiser le nuage.
4. Pour suivre l'évolution du modèle, cocher **Temps réel** dans l'en-tête de la vue 3D.
5. Attendre la capture puis le calcul de la vue en arrière-plan.

| Action | Commande dans la vue 3D |
| --- | --- |
| Tourner le nuage | Glisser avec le bouton gauche dans le canevas |
| Zoomer / dézoomer | Molette |
| Revenir au cadrage initial | `R` |
| Identifier un vecteur | Survoler son marqueur |
| Actualiser automatiquement | Cocher **Temps réel** dans l'en-tête |
| Voir les unités et le routage du réseau | Cocher **Répartition des neurones** |
| Suivre l'évolution des coefficients appris | Cocher **Poids actifs (DL)** |
| Observer les coactivations internes | Cocher **Corrélations du corpus** |
| Filtrer les liens de corrélation | Boutons `−` / `+` du seuil `|r|` |
| Recommencer la collecte | Bouton **Effacer** du mode corrélations |
| Afficher tous les liens candidats | Case **Tous les candidats valides** du mode corrélations |
| Exporter le graphe affiché | Bouton **Exporter 3D** ou `X` |
| Fermer la vue | `Échap` ou bouton de fermeture |

Les marqueurs et leurs zones de survol restent dans le canevas. Le zoom peut
masquer les points qui sortent de cette zone ; `R` rétablit le cadrage initial.
Les axes utilisent une échelle commune, qui conserve les proportions de la
projection. Dans les modes de projection des vecteurs, PC1, PC2 et PC3 affichent
leur part de variance expliquée. Le mode de répartition utilise les axes
topologiques décrits ci-dessous.

## Exporter un graphe 3D

Dans l'en-tête de la vue 3D, cliquer sur **Exporter 3D**, ou appuyer sur `X`
pendant que cette vue est ouverte. Une projection doit déjà être affichée.
L'export fonctionne pour le nuage des poids, les corrélations du corpus,
la répartition des neurones et les poids actifs.

Chaque export crée un dossier distinct
`exports/graph_3d_<mode>_step_<step>_<horodatage>/`, relatif au répertoire de
lancement. Le mode est `structural`, `correlations`, `neurons` ou `live_weights` ;
l'horodatage distingue les exports d'une même étape.

| Fichier | Contenu |
| --- | --- |
| `vue.png` | Image de la vue courante, avec son cadrage |
| `graph.json` | Points, liens, valeurs et métadonnées de la projection affichée |
| `graph.html` | Vue 3D interactive autonome, à ouvrir directement dans un navigateur |

L'interface affiche le chemin du HTML généré.

Le HTML permet de tourner et zoomer, de revenir au cadrage exporté et
d'inspecter les points et liens au survol. En mode corrélations, son seuil
`|r|` reste réglable localement. La case **Tous les candidats valides** reprend
le choix de la Viz et permet de retrouver tous les liens candidats sauvegardés.
Il embarque ses données et son rendu : aucun
serveur, CDN ou accès réseau n'est nécessaire. Le JSON conserve notamment l'identité du
modèle, le mode, la génération et la révision de la projection, les étapes
d'optimiseur, les informations d'échantillonnage et les réglages de caméra et
de seuil de corrélation.

L'export porte sur la **dernière projection affichée**, même si une capture plus
récente est en préparation. Il conserve tous les points et liens de cet
échantillon, y compris ceux hors du cadrage ou masqués par le seuil visuel.
Ce n'est pas un checkpoint des poids complets ni un historique d'entraînement.
En mode **Poids actifs (DL)**, les valeurs et variations exportées restent
figées à cette capture. L'entraînement peut continuer ; l'export ne relit pas
le modèle et ne lance pas de forward supplémentaire.

## Répartition des neurones et des canaux

La case **Répartition des neurones** est désactivée par défaut. Elle ouvre une
disposition 3D des unités du graphe effectif du modèle courant. Les modes
**Répartition des neurones**, **Poids actifs (DL)** et **Corrélations du corpus**
sont exclusifs : activer l’un désactive les autres. Décocher ces modes revient
au nuage des poids et des activations.

L’axe X suit la profondeur des dépendances du graphe. Les axes Y et Z organisent
les unités de chaque couche sur une grille. Les branches et connexions
résiduelles viennent du routage réel des entrées et sorties des couches ; la
vue ne relie pas automatiquement deux couches parce qu’elles se suivent dans
une liste. Cette disposition sert à lire le réseau : ce ne sont ni des
coordonnées physiques de neurones, ni les axes d’une ACP.

Les unités sont identifiées selon leur couche : neurones d’une projection
linéaire, canaux convolutionnels ou composantes d’une représentation. Une
opération dont la largeur ne peut pas être établie est signalée comme telle,
sans inventer un nombre de neurones. Survolez les marqueurs et les liens pour
lire les noms, types et indices réellement capturés.
Les sources d’entrée externes sont aussi représentées et incluses dans le
nombre de couches affiché ; leur largeur reste inconnue si les consommateurs
ne permettent pas de la déterminer sans ambiguïté.
Le total additionne les unités représentées à chaque étape du graphe, y compris
les activations et normalisations. Il ne décompte pas des neurones indépendants
ou des paramètres ; pour les séquences, les mêmes composantes sont partagées
entre les positions.

La capture garde au plus **4 096 points** et **8 192 liens**. Le budget de points
est réparti entre les couches, puis les unités sont échantillonnées sur leur
largeur entière. Les compteurs distinguent les totaux connus des éléments
affichés et signalent les largeurs inconnues. Un grand réseau n’est donc pas
affiché neurone par neurone dans son intégralité.

Les liens sont des flux structurels agrégés entre couches. Leur présence
représente une dépendance déclarée, sans attribuer une connexion synaptique
individuelle ou une intensité apprise à chaque trait. Ce mode ne colore
pas ces liens selon les poids du modèle ; les liens de Pearson restent propres
au mode **Corrélations du corpus**.

Cette vue suit les mêmes points sûrs et la même case **Temps réel** que le
nuage structurel des poids. Elle ne lance aucun forward supplémentaire et ne
modifie pas l’architecture. Une demande pendant une pause attend un point sûr
sur le thread du modèle. Le changement de mode ou de modèle invalide le graphe
précédent.

## Poids actifs pendant l'apprentissage

La case **Poids actifs (DL)** est désactivée par défaut. Son activation ouvre une
vue topologique des coefficients réels du modèle et coche **Temps réel**.
L’axe X suit la profondeur du graphe ; les unités gardent une disposition
stable dans leur couche. Le signe et l’amplitude des poids, puis leur variation
entre captures, montrent l’évolution des paramètres appris pendant
l’entraînement.

1. Lancez l’entraînement avec Viz, puis ouvrez la vue 3D depuis **Pilotage**.
2. Cochez **Poids actifs (DL)** et laissez l’optimiseur effectuer ses mises à jour.
3. Repérez les coefficients positifs en cyan, négatifs en orange et nuls en gris.
   L’épaisseur et la luminosité rendent leur amplitude visible ; les variations
   non nulles connues sont soulignées.
4. Survolez les coefficients ou liens pour lire leur couche, l’indice du
   paramètre, sa valeur et son écart depuis la capture précédente, avec les
   étapes d’optimiseur comparées.

La première capture constitue la référence : sa variation est indiquée comme
indisponible. La comparaison utilise l’identité stable de chaque coefficient
du même modèle. Un changement de mode, de modèle ou de schéma de capture
invalide cette référence.
Un coefficient nul reste représentable. En inférence avec des poids fixes,
des entrées différentes peuvent modifier les activations tandis que les
coefficients et leur variation restent stables.

Les liens pondérés représentent uniquement les correspondances de paramètres
que la capture sait établir à partir du stockage et du routage réels. Les
autres coefficients sont affichés comme des marqueurs de paramètres près de
leur couche, avec leurs indices. Les flux structurels gris conservent leur
nature de dépendances entre couches. L’inspection distingue ces catégories.

La capture garde au plus **2 048 coefficients**. La vue associe au plus
2 048 unités du réseau et 2 048 marqueurs de coefficients, soit **4 096 points**,
avec au plus 2 048 liens pondérés et 1 024 flux structurels de contexte. Les
compteurs précisent les paramètres déclarés, les coefficients échantillonnés et
leur disponibilité. L’échantillonnage garde des identités stables pour comparer
les mêmes paramètres d’une capture à l’autre ; les totaux ne représentent pas
un affichage exhaustif des grands réseaux.

La capture lit des valeurs hôte déjà disponibles, sans charger de bloc froid,
modifier les poids ou lancer de forward supplémentaire. Elle utilise les
points sûrs communs, notamment après la mise à jour d’optimiseur. La cadence
automatique est bornée à une capture toutes les 250 ms ; les étapes
intermédiaires peuvent être sautées. Les valeurs et comparaisons portent donc
sur les captures affichées, avec leurs indices d’étape.
Une actualisation au même pas d’optimiseur ne constitue pas une mise à jour
d’apprentissage supplémentaire ; sa comparaison porte sur ces deux captures.

Décocher **Temps réel** fige la vue après la fin d’une éventuelle capture déjà
lancée. Le bouton d’actualisation reste disponible. Fermer la vue suspend les
captures automatiques. La disposition représente la topologie du réseau ;
l’apprentissage ordinaire fait évoluer ses paramètres sur cette structure.

## Corrélations internes sur les entrées observées

La case **Corrélations du corpus** est désactivée par défaut. Elle remplace le
nuage structurel des poids par des points représentant des composantes
d’activations internes : features d’un latent ou d’une couche cachée, canaux
convolutionnels ou, si la géométrie est inconnue, composantes plates explicitement
identifiées. La capture est commune à `Model` ; une architecture n’a pas besoin
d’un champ `latent_dim` pour fournir ces observations.

1. Ouvrez la vue 3D et activez **Corrélations du corpus**.
2. Laissez le modèle traiter ses entrées réelles, en entraînement ou en inférence.
3. Après au moins huit observations et deux composantes variables, le premier
   graphe apparaît. Les activations constantes n’ont pas de corrélation définie
   et sont exclues du nuage.
4. Survolez les points et les liens pour retrouver couche, composante,
   méthode de regroupement, moyenne, écart-type et corrélations mesurées.
5. Réglez le seuil absolu `|r|` avec `−` et `+`. Le seuil initial est `0,65` ;
   il est ajustable entre `0,30` et `0,95`.

Les liens cyan indiquent une corrélation positive ; les liens orange une
corrélation négative. Ils représentent le coefficient de Pearson calculé en
double précision dans les profils d’activations, **avant** la projection 3D.
La proximité à l’écran n’est donc pas utilisée pour fabriquer ces liens.
L’ACP projette les profils centrés et normalisés de ces composantes sur leurs
observations.

### Collecte et actualisation

La collecte s’effectue après chaque forward réussi pendant que le mode est
activé et la vue ouverte. Elle ne lance aucun passage neuronal supplémentaire
et ne modifie ni les poids, ni les gradients, ni le tokenizer. Un numéro de
forward évite de compter plusieurs fois le même passage lors des hooks de
capture. Une observation n’est pas nécessairement un document distinct : une
génération autorégressive peut fournir plusieurs passages sur un contexte qui
s’allonge.

La fenêtre conserve au plus 128 observations et 128 composantes issues d’au
plus huit sources du graphe. Les features sont sélectionnées sur toute leur
largeur ; leurs valeurs regroupent au plus 256 positions. Pour les séquences,
les positions PAD et le masque d’attention compatible sont exclus. Pour les
convolutions, les valeurs sont regroupées par canal et positions spatiales.
Le libellé précise si une valeur provient d’une entrée conservée de la couche
suivante, éventuellement arrondie par l’autocast.

La file d’attente du worker est bornée à 32 observations ; s’il prend du retard,
les observations les plus anciennes encore en attente sont remplacées par les
plus récentes. Une entrée sans position valide ou non finie est refusée. Ces
bornes maintiennent un échantillon de diagnostic, pas un inventaire exhaustif.

Le premier graphe calculable s’affiche même si **Temps réel** est décoché.
Ensuite, cette case pilote l’actualisation du graphe ; la collecte peut continuer
avec le graphe figé. Les compteurs distinguent les observations de la projection
affichée de celles de la fenêtre de collecte. Une demande manuelle actualise
aussi le graphe à partir de la fenêtre déjà disponible, y compris si le modèle
est en pause.

Par défaut, la vue montre les 512 liens les plus forts parmi ceux dont
`|r| ≥ 0,30`, filtrés par le seuil visuel. Cocher **Tous les candidats valides**
affiche tous les candidats calculés, sans la limite de 512 ni le seuil visuel
supplémentaire : jusqu’à 8 128 paires pour les 128 composantes observées.
Les corrélations restent finies, entre composantes variables, avec `|r| ≥ 0,30`.
Le seuil précédent est conservé et redevient actif en décochant la case ; ses
boutons sont désactivés tant que tous les candidats sont affichés. Le compteur
distingue les liens visibles des candidats calculés ; un lien hors cadrage peut
rester invisible. Le survol fonctionne aussi pour les candidats supplémentaires.

Ce choix ne relance pas le modèle, ne change pas les activations ni les
paramètres appris. Les exports PNG, JSON et HTML conservent l’option affichée,
et le HTML embarque les candidats disponibles même si la case était décochée
au moment de l’export. L’historique est effacé par **Effacer**, un changement
de mode ou de modèle, ou un changement du schéma de features qui rendrait les
observations incompatibles. Il n’est pas enregistré dans les checkpoints.

### Interpréter le corpus latent observé

Cette vue montre un échantillon des entrées effectivement traversées depuis
l’activation du mode, pas un recalcul exhaustif d’un fichier de corpus. Les
compteurs indiquent aussi la provenance entraînement/inférence et la plage
des étapes d’optimiseur. Pendant l’apprentissage, les poids évoluent : les
relations observées reflètent à la fois les variations d’entrées et cette
évolution des paramètres.

Pour isoler les relations entre entrées à poids fixes, parcourez un ensemble
représentatif en inférence, sans pas optimiseur. Un coefficient décrit une
co-variation de composantes internes ; il ne suffit pas à attribuer un concept
humain à une composante ou à démontrer une relation causale. Les sous-modèles
privés hors du graphe principal ne sont pas parcourus automatiquement.

Fermer la vue ou désactiver le mode arrête les nouvelles observations. Le retour
au nuage des poids attend une capture sûre du modèle, comme le mode structurel
habituel.

## Actualisation en temps réel

La case **Temps réel** est désactivée par défaut ; sélectionner **Poids actifs (DL)**
l’active. Lorsqu'elle est cochée dans les modes structurels et que
la vue 3D est ouverte, le modèle capture automatiquement ses données aux points
sûrs décrits ci-dessous, sans nouveau clic sur le bouton de génération.
L'indication **Selon les étapes du modèle** rappelle que les données suivent le
calcul du modèle, avec une cadence limitée à une capture toutes les 250 ms en
fonctionnement continu, soit au plus quatre actualisations par seconde.

Une capture ou un calcul de disposition déjà en cours pour le modèle courant termine avant
l'admission d'une nouvelle capture automatique. Les étapes intermédiaires peuvent donc être sautées : la
cadence réelle dépend aussi du modèle et du temps de projection. Le worker ne
lit aucun poids actif et n'empile pas les captures pour rattraper un retard.

Décocher la case ou fermer la vue 3D suspend les nouvelles captures automatiques
des modes structurels. Dans le mode corrélations, décocher **Temps réel** fige la
projection mais laisse la collecte active tant que sa case et la vue restent ouvertes.
Un calcul déjà lancé peut encore publier son résultat. Le choix de la case est
conservé à la réouverture de la vue et au changement de modèle, pendant la
session du moniteur. Le bouton de génération conserve son fonctionnement manuel
lorsque **Temps réel** est désactivé.

Si le thread du modèle est bloqué par une pause ou n'exécute plus d'opérations,
les poids restent stables et le dernier nuage reste affiché. Le mode ne crée
pas de boucle de calcul dans le thread graphique. Pour continuer à servir les
actualisations après l'entraînement, utiliser la boucle Lua de l'exemple.

## Quand les vecteurs structurels sont-ils capturés ?

Le clic dépose une demande manuelle. Le mode **Temps réel** rend aussi une
capture automatique éligible à son échéance, tant que la vue est ouverte. Le
thread du modèle copie les données à un point sûr ; le thread graphique ne lit
jamais les poids en cours de modification.

| Situation | Traitement de la demande |
| --- | --- |
| Entraînement actif | Au terme d'un `Model::optimizerStep`, après la mise à jour des poids |
| Inférence | Au terme d'un forward exécuté avec `training=false` |
| Boucle Lua personnalisée | `Mimir.Viz.update()` sert une demande manuelle ou une actualisation automatique arrivée à échéance, avec les données déjà disponibles |
| Entraînement en pause | La capture attend la reprise, sauf si le thread Lua reste disponible pour appeler `Viz.update()` |
| Fin de `Mimir.Model.train()` | Une capture finale est demandée quand la Viz existe |
| Fin du script Lua lancé par `mimir` | Une capture est forcée avant l'attente de fermeture d'une Viz encore ouverte |

Un forward avec `training=true` seul ne sert pas la demande : elle attend
l'optimiseur ou un appel explicite à `Viz.update()`. Le nuage déjà publié reste
consultable pendant une pause. Après le retour complet du script, la fenêtre
peut rester ouverte, mais son attente de fermeture n'exécute plus de code Lua.
Pour continuer à actualiser la vue, garder une boucle Lua comme dans l'exemple
ci-dessous.

Avec les wrappers Lua, créer un autre modèle ou recharger un checkpoint ouvre
une nouvelle session de projection. Les requêtes et résultats précédents sont invalidés ; un calcul
encore en cours ne peut pas remplacer la projection du nouveau modèle.

## Exemple minimal sans dataset

Ce petit MLP visualise des **poids initialisés**, avant tout entraînement.
Enregistrer ce bloc dans `/tmp/mimir_projection_3d.lua` :

```lua
assert(Mimir.Model.create("basic_mlp", {
  input_dim = 4, hidden_dim = 8, output_dim = 3,
  hidden_layers = 1, dropout = 0,
}))
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier", 51))

assert(Mimir.Viz.create({
  visualization = {
    enabled = true,
    window_title = "Mimir | Projection 3D - exemple MLP",
    window_width = 1280, window_height = 720,
  },
}))

local output, err = Mimir.Model.forward({0.2, -0.1, 0.7, 0}, false)
assert(output, err)

while Mimir.Viz.is_open() do
  Mimir.Viz.update()
  Mimir.Viz.poll_events(50)
end
```

Depuis la racine du dépôt, avec un binaire compilé avec un backend Viz :

```bash
./run_mimir.sh --lua /tmp/mimir_projection_3d.lua
```

Cliquer sur le bouton 3D pour inspecter les poids et les activations du forward.
La boucle laisse le thread Lua disponible pour les nouvelles captures, y compris
avec **Temps réel** activé. Fermer la fenêtre Viz termine l'exemple.

## Échantillonnage et interprétation

La capture utilise les poids déjà résidents du graphe actif et les activations
encore disponibles. Elle déduplique les poids partagés, y compris les alias
déclarés avant leur premier forward. Les blocs de poids déchargés restent
déchargés ; la vue ne retient pas de nouveaux tenseurs d'entraînement.

| Limite de la capture commune | Valeur maximale |
| --- | --- |
| Nombre de vecteurs | 512 |
| Nombre total de valeurs copiées | 262 144 |
| Largeur d'un vecteur capturé | 2 048 valeurs |

Une fenêtre d'un vecteur plus large est identifiée par sa plage et sa dimension
d'origine. Les quotas sont des bornes d'échantillonnage, pas une promesse
d'affichage de chaque poids. Les valeurs non finies sont exclues ; les vecteurs
nuls et les captures de rang faible restent affichables.

L'ACP normalise les échantillons, complète les dimensions plus courtes par des
zéros et centre les données avant de calculer les composantes principales.
C'est une représentation structurelle de ces échantillons : les distances ne
mesurent pas une proximité sémantique entre couches ou une capacité de
raisonnement. Les axes sont recalculés pour chaque capture.

## Vérifier la fonctionnalité

Après compilation avec les tests activés :

```bash
ctest --test-dir build --output-on-failure \
  -R '^(VizTest\.(GenericLatentSpace|GenericLatentControl|GenericProjection|Export3D|LatentSpaceExport|LatentCorrelationMath|LatentCorrelationControl|NeuronNetworkControl|LiveWeightControl)|ModelTest\.(LatentCorrelations|NeuronNetwork|LiveWeightNetwork))$'
```

Les deux premiers tests vérifient la capture multiarchitecture, les quotas,
l'absence de mutation, la publication, le rejet des résultats obsolètes et la
cadence du mode temps réel. Le test du contrôle vérifie aussi deux mises à jour
d'optimiseur publiées automatiquement, sans demande manuelle.
Les tests de corrélations vérifient les oracles Pearson `+1`, `−1` et `0`, les
faibles variances, les activations constantes, les fenêtres/schémas/sessions,
les masques, la capture native et l’absence de mutation du modèle. Le contrôle
teste aussi le worker asynchrone et la séparation collecte/actualisation.
Les tests neuronaux vérifient le routage et la profondeur, les unités connues
et inconnues, l’échantillonnage et la publication depuis le modèle natif.
Les tests des poids actifs vérifient les coefficients capturés, leur identité,
la comparaison entre captures, les limites et la publication asynchrone.
`LatentSpaceExport` vérifie les données JSON et le HTML autonome des quatre
modes. `Export3D` vérifie le bouton et le raccourci d'export, l'image PNG et
la projection affichée sauvegardée. `GenericProjection` vérifie le bouton
d'ouverture, les interactions et le cadrage. Avec SFML, ces tests de rendu
nécessitent un affichage accessible ; un test ignoré
(`Skipped`) ne valide pas le rendu. Les autres hôtes graphiques doivent être
validés avec leur propre compilation.

## Références

- [API Viz et Htop](../03-API-Reference/15-Viz-Htop.md#projection-3d-du-modèle-courant)
- [Choisir un backend Viz](17-Viz-Backends.md)
- [Architecture du monitoring](../04-Architecture-Internals/04-Monitoring-Htop-Visualizer.md)
