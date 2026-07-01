# API : monitoring / visualisation

## Pour qui

Développeur et utilisateur intermédiaire/avancé.

## Objectif

Trouver rapidement le contrat API réel et les paramètres utilisables.

## Avant de commencer

Connaître les commandes de base de Mímir.

## Résultat attendu

Tu peux appeler l'API sans ambiguïté de signature ou de comportement.


## `Mimir.Htop`

- `create()`
- `update()`
- `render()`
- `clear()`
- `enable(bool)`

## `Mimir.Viz` / `Mimir.visualiser`

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

Notes :

- `Viz` peut dépendre de SFML selon le build.
- Le runtime peut publier des “viz taps” pendant `Model.forward()` si un monitor async est actif.
