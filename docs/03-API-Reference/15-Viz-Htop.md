# Monitoring et visualisation

Trouver rapidement le contrat API réel et les paramètres utilisables.

## Vue d'ensemble

Le monitoring Mímir repose sur 2 interfaces complémentaires :

- `HtopDisplay` : rendu terminal temps réel (métriques, progression, gradients, mémoire, logs).
- `Visualizer` (SFML) : rendu graphique des images/tensors intermédiaires, panels interactifs et réglages live.

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

## `Mimir.Viz`

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

### Démarrage / activation

- En script/JSON : active la visualisation via la config du run (`visualization.enabled=true`).
- En C++ : instancie `Visualizer`, puis `initialize()`, ensuite boucle `process_events()` + `update()`.

### Contrôles clavier (UI)

- `H` : aide overlay.
- `M` : bascule mode de rendu Blocks/Layers (`HEATMAP` / `REEL`).
- `K` : change la palette heatmap (`CLASSIC` -> `TURBO` -> `INFERNO` -> `VIRIDIS`).
- `A` : active/désactive le lissage des previews de blocks.
- `R` : resynchronise/rebuild textures + reload hints architecture.
- `Tab`, `F1..F5`, `←/→` : navigation focus/éléments.
- `Z`/`Entrée` : zoom, `Esc` : quitter zoom.

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

- `Viz` peut dépendre de SFML selon le build.
- Le runtime peut publier des “viz taps” pendant `Model.forward()` si un monitor async est actif.

## Étapes suivantes

- [Page précédente : API : mémoire](14-Memory.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : API : `Mimir.Serialization`](16-Serialization.md)
