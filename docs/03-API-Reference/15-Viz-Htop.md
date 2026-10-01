# Monitoring et visualisation

Trouver rapidement le contrat API réel et les paramètres utilisables.

## Vue d'ensemble

Le monitoring Mímir repose sur 2 interfaces complémentaires :

- `HtopDisplay` : rendu terminal temps réel (métriques, progression, gradients, mémoire, logs).
- `Visualizer` (SFML, Qt, GTK ou Web au choix à la compilation) : rendu graphique des images/tensors intermédiaires, panels interactifs et réglages live.

En pratique, les deux passent par `AsyncMonitor`, qui met à jour l'UI sans bloquer l'entraînement.

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

État réel des méthodes de cycle de vie : `create()` démarre `AsyncMonitor`,
`render()` est conservée pour compatibilité mais le rendu est automatique, et
`enable(false)` arrête le monitor. `enable(true)` ne redémarre pas un monitor
arrêté : il faut rappeler `create()`.

L'export CSV est unique. Viz le porte lorsqu'elle est active ; htop sert de
repli lorsqu'il tourne seul. Les options `csv`, `csv_enabled`, `csv_path` et
`csv_file` de `Htop.create()` configurent cet export commun.

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

`initialize()` vérifie que la fenêtre existe et est ouverte. `process_events()`
et `update()` sont des fonctions de compatibilité : `AsyncMonitor` exécute déjà
ces opérations sur le thread Viz. De même, `add_loss_point()` est un no-op ;
`update_metrics()` alimente l'historique. `set_enabled(false)` arrête le monitor,
mais `set_enabled(true)` ne le redémarre pas : utiliser `create()`.

### Démarrage / activation

- En script/JSON : active la visualisation via la config du run (`visualization.enabled=true`).
- En C++ : instancie `Visualizer`, puis `initialize()`, ensuite boucle `process_events()` + `update()`.

### Contrôles clavier (UI)

- `H` : aide overlay.
- `C` : ouvre ou ferme la section de configuration du modèle.
- `M` : bascule mode de rendu Blocks/Layers (`HEATMAP` / `REEL`).
- `K` : change la palette heatmap (`CLASSIC` -> `TURBO` -> `INFERNO` -> `VIRIDIS`).
- `A` : active/désactive le lissage des previews de blocks.
- `R` : resynchronise/rebuild textures + reload hints architecture.
- `Tab`, `F1..F5`, `←/→` : navigation focus/éléments.
- `Z`/`Entrée` : zoom, `Esc` : quitter zoom.

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
derniers). Les événements bruts observent les entrées sans désactiver les
raccourcis intégrés. La file est bornée à 256 entrées : les plus anciennes sont
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
./bin/mimir --lua scripts/inferences/edit_vae_conv.lua -- \
  --checkpoint checkpoint/vae/epoch_0010 \
  --input photo.ppm --output outputs/edition.ppm
```

Le script accepte un checkpoint **raw_folder** `vae_conv` entraîné et une image
PNG/JPEG/BMP ou PPM P3/P6 RGB. La géométrie provient du checkpoint.
Les formats PNG/JPEG/BMP utilisent le chargement RGB natif Mímir avec
redimensionnement bicubique ; le PPM utilise le voisin le plus proche si nécessaire. Il encode la moyenne latente, charge
`vae_conv_decode`, puis permet de décaler un canal latent, mélanger le résultat
avec la source, réinitialiser et enregistrer en PPM. La source et le dernier
résultat sont présentés côte à côte ; `Z` agrandit l'image sélectionnée.

Ces modifications latentes ne sont pas des commandes sémantiques et leur effet
dépend de l'entraînement. Le décodeur autonome utilise des skips encodeur nuls
si le modèle comporte des connexions de ce type ; sa sortie à force zéro peut
donc différer d'une reconstruction par le VAE complet. `--mix 0` conserve
l'image source redimensionnée. L'interface ne sauvegarde pas à la fermeture :
utiliser **Enregistrer**. Les fichiers existants au chemin de sortie sont
remplacés à chaque enregistrement.

`--headless --strength 0.2 --channel 1 --mix 0.8` calcule et enregistre une fois,
sans interface. `--help` détaille les options sans charger de modèle. Le script
vérifie la présence et la taille des poids requis avant de charger le décodeur
partiel ; il ne modifie ni les poids ni le checkpoint.
