# Observer un modèle avec Viz, Htop et les rapports

Ce tutoriel apprend à distinguer la sortie du modèle, les métriques de calcul,
le graphe d'exécution et la projection 3D de ses vecteurs. Le premier atelier
utilise un MLP natif et aucune donnée externe. Les parties entraînement
réutilisent le [tutoriel validation et reprise](12-Tuto-Entrainement-Validation-Reprise-Inference.md).

| Niveau | Résultat à obtenir |
| --- | --- |
| Débutant | Ouvrir le modèle courant et reconnaître ses couches. |
| Intermédiaire | Relier une métrique affichée à un calcul et à son CSV. |
| Avancé | Comparer les captures, diagnostiquer un refus de publication et produire un rapport. |

## 1. Vérifier le profil d'affichage

Depuis la racine du dépôt, un binaire compilé est nécessaire :

```bash
./bin/mimir --help
```

Le backend Viz est choisi à la compilation : `SFML`, `QT`, `GTK`, `WEB` ou
`NONE`. Qt/GTK/Web ciblent actuellement Linux. Un build `NONE` peut entraîner
et exporter des métriques, mais n'ouvre pas de fenêtre Viz. Htop est indépendant
de ce choix. Les dépendances et recettes de compilation figurent dans
[les backends Viz](../02-User-Guide/17-Viz-Backends.md).

Selon le [contrat d'architecture](../07-Devs/08-Architecture-Contract.md), une
architecture native se définit en C/C++ dans une classe dérivée de `Model`.
Le script ci-dessous instancie `BasicMLPModel` via le registre ; il ne décrit
aucune couche. MPK reste une alternative expérimentale, pas un chemin pleinement
exploitable en production.

## 2. Inspecter un petit modèle sans dataset

Enregistrer dans `/tmp/mimir_observer.lua` sur un système POSIX :

```lua
local gui = os.getenv("MIMIR_TUTORIAL_GUI") == "1"
assert(Mimir.Model.create("basic_mlp", {
  input_dim = 4, hidden_dim = 8, output_dim = 3,
  hidden_layers = 1, dropout = 0, dtype = "float32",
}))
assert(Mimir.Model.set_hardware(false))
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier", 51))

for _, layer in ipairs(Mimir.Model.get_layers()) do
  print(layer.index, layer.name, layer.type, layer.param_count)
end
if gui then
  assert(Mimir.Viz.create({visualization = {
    enabled = true, window_title = "Mimir | Atelier observation",
    window_width = 1280, window_height = 720,
  }}))
end
local output = assert(Mimir.Model.forward({0.2, -0.1, 0.7, 0.0}, false))
assert(#output == 3)
print("OBSERVER_FORWARD_OK", #output)

if gui then
  while Mimir.Viz.is_open() do
    Mimir.Viz.update()
    Mimir.Viz.poll_events(50)
  end
end
```

Exécuter d'abord le chemin sans interface :

```bash
./run_mimir.sh --lua /tmp/mimir_observer.lua
```

Résultat attendu : une liste de couches et `OBSERVER_FORWARD_OK 3`. Les valeurs
de sortie proviennent de poids initialisés ; ce résultat ne démontre aucun
apprentissage. Un build et un environnement d'affichage compatibles permettent
ensuite la variante interactive :

```bash
MIMIR_TUTORIAL_GUI=1 ./run_mimir.sh --lua /tmp/mimir_observer.lua
```

Cette variante demande une interaction manuelle ; la fermeture de Viz termine
la boucle. Sur Windows, choisir un fichier local et définir la variable avec
la syntaxe du shell utilisé, comme expliqué dans le [tutoriel de portabilité](16-Tuto-Compilation-Portabilite-AMP-DDP.md).

## 3. Lire la vue plutôt que seulement la courbe

Dans Viz, parcourir ces zones dans l'ordre :

1. **Blocks / Layers** : vérifier les noms et familles réellement présents.
2. L'inspecteur de tenseur : lire les sous-sections, formes et statistiques ;
   la molette fait défiler le contenu complet.
3. Le graphe nodal avec `N` : retrouver les dépendances du modèle et du planner.
4. La projection 3D dans **Pilotage** : générer la capture du modèle courant.

Dans le graphe, sélectionner un nœud pour lire ses détails ; `R` recentre et
`N` ou `Échap` ferme la vue. `X` exporte le graphe complet en PNG. La touche
`E` ouvre la sélection des éléments à inclure dans une capture Viz.

Dans la vue 3D, glisser tourne le nuage, la molette zoome et `R` rétablit le
cadrage. Cocher **Temps réel** demande des captures répétées aux points sûrs du
modèle, avec une cadence bornée. La boucle `Viz.update()` de l'exemple permet
de traiter ces demandes après le forward.

La capture est échantillonnée : elle ne montre pas chaque poids. Une ACP des
vecteurs n'est pas une mesure de compréhension du modèle. Les axes peuvent
changer entre deux captures. Voir les [limites de la projection 3D](../02-User-Guide/18-Projection-3D.md).

Pour observer les coactivations internes, cochez **Corrélations du corpus** puis
laissez le modèle traiter des entrées variées. Après huit forwards au minimum,
les composantes variables deviennent des points ; les liens cyan et orange
indiquent respectivement leurs corrélations positives et négatives. Réglez le
seuil `|r|` et survolez un lien pour lire sa valeur mesurée avant la projection.
Le mode garde au plus 128 observations ; **Effacer** recommence cette fenêtre.
Il faut de nouveaux forwards pour la remplir : la boucle d’affichage seule de
l’exemple ne produit pas de nouvelles observations.

Dans ce mode, **Temps réel** actualise le graphe pendant que la collecte
continue. Comparez les deux compteurs si la projection est figée. Pour étudier
les variations du corpus à poids fixes, utilisez plusieurs entrées en inférence
sans mise à jour d’optimiseur. Pendant l’entraînement, les corrélations reflètent
aussi l’évolution des poids.

L’option **Tous les candidats valides** affiche tous les liens retenus par le seuil
minimal de collecte, au-delà du plafond d’affichage par défaut. Elle n’invente
pas de nouveaux liens : le calcul reste borné et fondé sur les observations
collectées. Les modes **Répartition des neurones** et **Poids actifs (DL)** montrent
respectivement la structure échantillonnée du réseau et les coefficients appris,
avec leurs variations entre captures. Ils ne sont pas
des mesures de compréhension. Les détails et limites figurent dans le
[guide 3D](../02-User-Guide/18-Projection-3D.md).

Le bouton **Exporter 3D** enregistre une capture PNG, les données JSON et une
page HTML interactive autonome. Le choix du mode, des candidats et la caméra
appartiennent à cette capture ; l’export n’est pas une connexion au run vivant.

Un clic droit dans la vue principale ouvre le **gestionnaire des sections**.
Les cases permettent de rouvrir les panneaux fermés ; **Tout afficher** les
réactive ensemble. La visibilité est conservée dans le layout. La vue zoomée
garde son geste de clic droit propre à la sélection de pixels ; voir les
[contrôles complets](../03-API-Reference/15-Viz-Htop.md).

## 4. Observer un véritable entraînement

Reprendre une commande fonctionnelle du tutoriel d'entraînement et lui ajouter
un des groupes d'options communs suivants :

| Options à ajouter après `--` | Effet |
| --- | --- |
| `--htop --no-viz` | Interface terminal seule. |
| `--viz --no-htop` | Interface graphique seule. |
| `--viz --htop` | Les deux interfaces. |
| `--no-viz --no-htop` | Exécution sans interface interactive. |
| `--htop --no-viz --csv --csv-path /tmp/mimir_tutoriels/metrics.csv` | Active le moniteur terminal et exporte dans un fichier CSV explicite pour l’atelier POSIX. |

Ces options sont traitées par les scripts qui utilisent les arguments communs ;
elles ne rendent pas un script quelconque compatible automatiquement. Dans
`train_causal_lm.lua`, DDP est initialisé avant les threads de monitoring et
seul le rang principal ouvre les interfaces.

Dans le chemin actuel des arguments communs, `--no-htop --no-viz` empêche la
création du moniteur et son export CSV. Pour cet atelier, remplacez ces options
par le groupe terminal/CSV du tableau ; ne comptez pas sur `--csv` seul pour
produire le fichier. Vous pouvez également utiliser un moniteur Viz disponible.

Pour un modèle causal, regarder notamment la cross-entropy, la perplexité,
l'exactitude du prochain token, les gradients et les métriques de validation.
Une précision Top-1 faible en début d'apprentissage peut coexister avec une loss
qui baisse. Le Top-1 de chaque position conditionnée n'est pas une génération
autonome de conversation.

Pour VAEConv, les métriques `logvar`, `posterior_std`, `sampling_rms` et les
normes de gradients des têtes aident à vérifier la contribution stochastique.
`logvar=0` correspond à une variance de 1. Les métriques spécifiques publiées
dans Viz/Htop ne deviennent pas automatiquement des colonnes du CSV commun,
dont le schéma est fixe ; vérifiez le fichier réellement exporté avant le rapport.

Htop expose aussi le graphe par `G` et la configuration par `C`. Certaines
valeurs sont modifiables en direct ; les champs fixes exigent une reconstruction
du modèle. Vérifier le libellé avant toute modification. Les [contrôles du monitor](../03-API-Reference/15-Viz-Htop.md)
décrivent les raccourcis et le cycle de vie exact.

## 5. Générer un rapport lié au bon run

Une fois le CSV `/tmp/mimir_tutoriels/metrics.csv` réellement produit par le run
causal du tutoriel 12 auquel ces options ont été ajoutées :

```bash
./run_mimir.sh --lua scripts/tools/show-graph.lua \
  --csv /tmp/mimir_tutoriels/metrics.csv \
  --checkpoint /tmp/mimir_tutoriels/causal/final --strict-context \
  --out /tmp/mimir_tutoriels/rapport.html \
  --out-text /tmp/mimir_tutoriels/rapport.md \
  --no-interactive
```

Adapter le répertoire de checkpoint au run observé. Ne pas utiliser un CSV
global d'une autre expérience. La variante depuis **Pilotage**, via le bouton
de rapport, prépare les arguments avec la configuration active et conserve un
snapshot des métriques pour le rapport.

Le chemin désigne ici le checkpoint `final` qui contient les métadonnées du
modèle, pas seulement le dossier parent de l’expérience. `--strict-context`
évite de compléter le rapport depuis un checkpoint global ou des préférences
d’interface étrangères à ce run. Exception au séparateur habituel : le parser
actuel de `show-graph.lua` reçoit ses options directement après son nom ;
un argument `--` isolé est refusé par cet outil.

Vérifier dans le rapport le nom du modèle, les chemins du CSV et du checkpoint,
le nombre d'étapes et les métriques enregistrées. Le script analyse le CSV ; il
ne réexécute pas le modèle pour recalculer ses prédictions. Les définitions sont
dans [Show-Graph et ses métriques](../02-User-Guide/16-Show-Graph-Metrics.md).

## 6. Exercices et diagnostic

| Niveau | Exercice | Vérification |
| --- | --- | --- |
| Débutant | Retrouver la première projection du MLP. | Son type et ses dimensions correspondent à la configuration. |
| Intermédiaire | Comparer une ligne CSV à l'interface au même pas. | La métrique et son index désignent le même calcul. |
| Avancé | Comparer une capture avant/après entraînement. | Les checkpoints et les étapes sont identifiés ; l'ACP seule ne sert pas de verdict. |

Une vue vide peut venir d'un modèle non alloué, d'une absence de forward ou
d'activation encore disponible. Une capture attendue pendant une pause dépend
d'un prochain point sûr. Une option graphique ne change pas le runtime de
calcul. Un test graphique ignoré faute d'écran ne valide pas le rendu.

Pour les tests, après construction des cibles correspondantes :

```bash
ctest --test-dir build --output-on-failure \
  -R '^(VizTest\.(GenericLatentSpace|GenericLatentControl|GenericProjection|PanelManager|Export3D|LatentCorrelationMath|LatentCorrelationControl)|ModelTest\.LatentCorrelations)$'
```

## Étapes suivantes

- [Mémoire, performances et runtimes](15-Tuto-Memoire-Performances-Runtimes.md)
- [Index des tutoriels](00-INDEX.md)
