# API : monitoring / visualisation

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
